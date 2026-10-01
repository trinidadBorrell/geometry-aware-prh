"""Orchestrate the bounded metric-stability study. Fitting and evaluation stay separate."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from prh_replication.anisotropic_kernels import make_pack, scores_numpy
from prh_replication.extract_final import load_final_prepared, spec_from_manifest
from prh_replication.io_utils import jsonable, sha256_file, write_json
from prh_replication.kernels import extension_stats
from prh_replication.metric_stability import (
    MONOTONE_ATOL,
    b_matrix_exp,
    correction_gram,
    delta_m_cosine,
    evaluate_metric,
    fit_anisotropic,
    fit_basis,
    gram_cosine,
    ids_hash,
    nested_sequences,
    rotation_matrices,
    scores_direct_f64,
    select_budget,
    subspace_compare,
    with_rotated_basis,
    assert_split_image_disjoint,
)
from prh_replication.registry import MODELS, Paths
from prh_replication.release_protocol import VIS, load_splits

OUT_NAME = "metric_stability"
Q = 32


def run(*, work: Path, repo: Path, stage: str, force: bool) -> None:
    paths = Paths(work=work, repo=repo)
    out = paths.results / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    if stage in ("audit", "smoke", "all"):
        data = prepare(paths, repo, out)
        if stage == "audit":
            return
    else:
        data = prepare(paths, repo, out, write_audit=False)
    if stage in ("smoke", "all"):
        smoke(data, out)
        if stage == "smoke":
            return
    if stage in ("run", "all"):
        run_fits(data, out, force=force)
        if stage == "run":
            return
    if stage in ("eval", "all"):
        run_eval(data, out, force=force)
        if stage == "eval":
            return
    if stage in ("report", "all"):
        write_report(data, out)
        verify_protected(out)


def prepare(paths: Paths, repo: Path, out: Path, write_audit: bool = True) -> dict[str, Any]:
    design = json.loads((repo / "configs" / "metric_stability.json").read_text())
    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    protocol = (repo / "configs" / "metric_stability_protocol.md").read_text()
    if write_audit:
        write_json(out / "design.json", design)
        (out / "protocol.md").write_text(protocol)
        _write_commands(out, repo)
    man, splits = load_splits(paths, repo)
    disjoint = assert_split_image_disjoint(splits)
    if set(splits) < {"train", "val", "test"}:
        raise RuntimeError("manifest is missing train/val/test")
    if any(disjoint["overlaps"].values()):
        raise RuntimeError(disjoint)
    langs = [row["key"] for row in manifest["base_panel"]]
    if any(row["key"] in langs for row in manifest.get("supplementary_qwen3x", [])):
        raise RuntimeError("supplementary ids collided with the base panel")
    specs = {row["key"]: spec_from_manifest(row) for row in manifest["base_panel"]}
    for key in VIS:
        specs[key] = MODELS[key]
    prepared: dict[str, dict[str, np.ndarray]] = {}
    hashes = {}
    id_lists = {split: list(ids) for split, ids in splits.items() if split in ("train", "val", "test")}
    for key in langs + VIS:
        prepared[key] = {}
        for split, ids in id_lists.items():
            x, obj = load_final_prepared(paths, specs[key], split, ids)
            if list(obj["sample_ids"]) != list(ids):
                raise ValueError(f"sample-id mismatch {key} {split}")
            prepared[key][split] = np.asarray(x.numpy(), dtype=np.float64)
            feature_path = Path(obj.get("path", "")) if False else None
            del feature_path
            from prh_replication.extract_final import final_feature_path

            fpath = final_feature_path(paths, specs[key], "coco_val2017", split, ids)
            hashes[f"{key}:{split}"] = {"path": str(fpath), "sha256": sha256_file(fpath), "n": len(ids), "d": int(prepared[key][split].shape[1])}
        print(f"loaded {key} train {prepared[key]['train'].shape}", flush=True)
    q = int(design["q"])
    full_basis = {}
    for key in langs + VIS:
        basis = fit_basis(prepared[key]["train"], q=q)
        full_basis[key] = basis
        _store_basis(out, basis, {"model": key, "protocol": "full_train", "reduced": basis["reduced"], "q": basis["q"], "variance_fraction": basis["variance_fraction"], "n": basis["n"]})
        if basis["reduced"] or basis["q"] != q:
            print(f"RANK {key} full train q={basis['q']} numerical_rank={basis['numerical_rank']}", flush=True)
    train_ids = id_lists["train"]
    sequences = nested_sequences(train_ids, list(design["sample_sizes"]), int(design["n_sequences"]), int(design["subset_seed"]))
    subset_basis = {}
    pos = {sid: i for i, sid in enumerate(train_ids)}
    for key in langs + VIS:
        x = prepared[key]["train"]
        for seq_i, seq in enumerate(sequences):
            for n, ids in seq.items():
                if n == len(train_ids):
                    subset_basis[(key, seq_i, n)] = full_basis[key]
                    continue
                idx = np.array([pos[s] for s in ids])
                basis = fit_basis(x[idx], q=q)
                subset_basis[(key, seq_i, n)] = basis
                _store_basis(out, basis, {"model": key, "protocol": "subset", "seq": seq_i, "n": n, "reduced": basis["reduced"], "q": basis["q"], "variance_fraction": basis["variance_fraction"]})
    rotations = rotation_matrices(q, int(design["basis_rotation_seed"]), int(design["n_rotations"]))
    perms = _control_perms(design, sequences, train_ids, id_lists["val"])
    if write_audit:
        write_json(out / "subsets.json", {"train_ids_hash": ids_hash(train_ids), "sequences": [{str(n): ids for n, ids in seq.items()} for seq in sequences], "perms": perms, "rotation_seed": design["basis_rotation_seed"], "n_rotations": design["n_rotations"]})
        audit = _audit(paths, design, full_basis, prepared, hashes, disjoint, langs, id_lists, repo)
        write_json(out / "audit.json", audit)
        if audit["float64_direct_minus_contracted_abs"] > 1e-6:
            raise RuntimeError(f"float64 direct/contracted gap {audit['float64_direct_minus_contracted_abs']}")
        print("audit float64 gap", audit["float64_direct_minus_contracted_abs"], "float32 gap", audit["float32_direct_minus_contracted_abs"], flush=True)
    return {
        "design": design,
        "langs": langs,
        "prepared": prepared,
        "ids": id_lists,
        "full_basis": full_basis,
        "subset_basis": subset_basis,
        "sequences": sequences,
        "rotations": rotations,
        "perms": perms,
        "out": out,
        "repo": repo,
        "pack_cache": {},
        "gallery_cache": {},
    }


def _control_perms(design, sequences, train_ids, val_ids) -> dict[str, Any]:
    """Independent train and validation correspondence permutations, fixed before fitting."""
    out: dict[str, Any] = {"train": {}, "val": {}}
    n_full = len(train_ids)
    for seed in design["null_perm_seeds"]:
        out["val"][str(seed)] = np.random.default_rng(int(seed) + 10000).permutation(len(val_ids)).tolist()
        out["train"][str(seed)] = {}
        for n in design["shuffle_sizes"]:
            if int(n) == n_full:
                out["train"][str(seed)][str(n)] = np.random.default_rng(int(seed)).permutation(n_full).tolist()
            else:
                # Permute the sequence-0 subset only. The subset itself stays the nested draw.
                out["train"][str(seed)][str(n)] = np.random.default_rng(int(seed)).permutation(int(n)).tolist()
    return out


def _audit(paths, design, full_basis, prepared, hashes, disjoint, langs, id_lists, repo) -> dict[str, Any]:
    pair = design["control_pairs"][0]
    a, b = pair
    ba, bb = full_basis[a], full_basis[b]
    za = prepared[a]["train"] - ba["mu"]
    zb = prepared[b]["train"] - bb["mu"]
    eye_a, eye_b = np.eye(ba["q"]), np.eye(bb["q"])
    pack = make_pack(za, zb, ba["U"], bb["U"])
    contracted = scores_numpy(np.zeros_like(eye_a), np.zeros_like(eye_b), pack)
    direct64 = scores_direct_f64(za, zb, ba["U"], bb["U"], eye_a, eye_b)
    import torch
    from prh_replication.anisotropic_kernels import metric_gram

    ka = torch.tensor(metric_gram(za, ba["U"], eye_a), dtype=torch.float64)
    kb = torch.tensor(metric_gram(zb, bb["U"], eye_b), dtype=torch.float64)
    direct32 = extension_stats(ka.float(), kb.float())
    parent = paths.results / "release_anisotropy_repair" / "pca_provenance.json"
    variance_note = None
    if parent.exists():
        prev = json.loads(parent.read_text())
        variance_note = {}
        for key, basis in full_basis.items():
            if key in prev:
                variance_note[key] = {
                    "repair_recomputed": prev[key].get("recomputed"),
                    "this_run": basis["variance_fraction"],
                    "abs_diff": abs(float(prev[key].get("recomputed", basis["variance_fraction"])) - basis["variance_fraction"]),
                }
    protected = {}
    for rel in design["do_not_modify"]:
        summary = paths.results / rel.split("/", 1)[1] / "summary.json" if rel.startswith("results/") else paths.results / rel / "summary.json"
        # rel is results/<name>
        name = rel.split("/")[-1]
        summary = paths.results / name / "summary.json"
        if summary.exists():
            protected[name] = {"path": str(summary), "mtime_ns": summary.stat().st_mtime_ns, "bytes": summary.stat().st_size}
    head = None
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = None
    return {
        "disjoint": disjoint,
        "n": {k: len(v) for k, v in id_lists.items()},
        "heldout_unused": True,
        "preprocessing": "full-split exact 0.95 clip then row L2; subsets do not re-clip",
        "fitted_on_train": ["mean", "PCA basis", "S"],
        "per_gallery": ["CKA gram centring", "kNN distances on x-mu_train"],
        "evaluation_split": "manifest test is exploratory held-out, already inspected",
        "sign_convention": full_basis[langs[0]]["sign_convention"],
        "float64_direct_minus_contracted_abs": abs(float(direct64["a"]) - float(contracted["a"])),
        "float32_direct_minus_contracted_abs": abs(float(direct32["a"]) - float(contracted["a"])),
        "float64_pair": pair,
        "float64_gallery": "train",
        "score_gap_note": "The historical ~2e-5 gap is the float32 Gram path in eval_onesided. This study uses float64 contracted scores. Float32 is recorded here and is not the objective.",
        "pca_variance_vs_repair": variance_note,
        "feature_hashes": hashes,
        "protected_mtimes": protected,
        "git_head": head,
        "identity_grad_path": "torch.matrix_exp; eigh only inside no_grad projection",
    }


def verify_protected(out: Path) -> None:
    audit = json.loads((out / "audit.json").read_text())
    for name, rec in audit["protected_mtimes"].items():
        path = Path(rec["path"])
        if path.stat().st_mtime_ns != rec["mtime_ns"] or path.stat().st_size != rec["bytes"]:
            raise RuntimeError(f"protected result changed: {name}")


def _store_basis(out: Path, basis: dict, meta: dict) -> None:
    dest = out / "bases"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{basis['id']}.npz"
    if not path.exists():
        np.savez_compressed(path, mu=basis["mu"], U=basis["U"])
    index = dest / "index.json"
    rows = json.loads(index.read_text()) if index.exists() else {}
    rows[basis["id"]] = {**meta, "id": basis["id"], "sign_convention": basis["sign_convention"]}
    write_json(index, rows)


def _write_commands(out: Path, repo: Path) -> None:
    text = (
        "export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src\n"
        "/mnt/sdb1/prh-replication-work/venv/bin/python "
        "/mnt/sdb1/prh-replication-work/repo/scripts/run_metric_stability.py "
        "--work /mnt/sdb1/prh-replication-work --stage all\n"
    )
    (out / "commands.txt").write_text(text)


def _fit_id(spec: dict) -> str:
    hold = spec.get("hold") or "-"
    perm = "na" if spec.get("perm_seed") is None else str(spec["perm_seed"])
    rot = "na" if spec.get("rotation") is None else str(spec["rotation"])
    partners = ",".join(spec["partners"])
    return "__".join(
        [
            spec["side_a"],
            partners,
            hold,
            spec["mode"],
            f"{float(spec['rho']):.4f}",
            spec["accounting"],
            spec["basis"],
            f"n{spec['n']}",
            f"s{spec['seq']}",
            f"p{perm}",
            f"r{rot}",
        ]
    )


def _data_key(spec: dict) -> str:
    hold = spec.get("hold") or "-"
    perm = "na" if spec.get("perm_seed") is None else str(spec["perm_seed"])
    rot = "na" if spec.get("rotation") is None else str(spec["rotation"])
    return "__".join([spec["side_a"], ",".join(spec["partners"]), hold, spec["basis"], f"n{spec['n']}", f"s{spec['seq']}", f"p{perm}", f"r{rot}"])


def _add(bucket: dict[str, dict], **spec) -> None:
    spec["panels"] = [spec.pop("panel")]
    spec["fit_id"] = _fit_id(spec)
    spec["data_key"] = _data_key(spec)
    prev = bucket.get(spec["fit_id"])
    if prev is None:
        bucket[spec["fit_id"]] = spec
        return
    for panel in spec["panels"]:
        if panel not in prev["panels"]:
            prev["panels"].append(panel)


def build_specs(data: dict) -> list[dict]:
    design = data["design"]
    n_full = len(data["ids"]["train"])
    if int(design["sample_sizes"][-1]) != n_full:
        raise RuntimeError("largest sample size is not the full training split")
    langs = data["langs"]
    visions = list(VIS)
    ll = [tuple(p) for p in design["ll_pairs"]]
    primary = [(a, v) for a in langs for v in visions] + ll
    bucket: dict[str, dict] = {}
    modes = list(design["modes_primary"])
    for a, b in primary:
        for mode in modes:
            rhos = [0.0] if mode == "identity" else list(design["primary_budgets_rho"])
            for rho in rhos:
                _add(bucket, panel="primary", side_a=a, partners=[b], hold=None, mode=mode, rho=float(rho), accounting="equal_total", basis="full", n=n_full, seq=0, perm_seed=None, rotation=None)
        for mode in ("separate", "shared_pc"):
            _add(bucket, panel="per_side", side_a=a, partners=[b], hold=None, mode=mode, rho=float(design["per_side_rho"]), accounting="per_side", basis="full", n=n_full, seq=0, perm_seed=None, rotation=None)
    sample_pairs = [(a, design["sample_vision"]) for a in langs] + ll
    for a, b in sample_pairs:
        for mode in list(design["modes_sample"]) + ["identity"]:
            rhos = [0.0] if mode == "identity" else [0.0] + list(design["sample_budgets_rho"])
            for rho in rhos:
                for n in design["sample_sizes"]:
                    n = int(n)
                    seqs = [0] if n == n_full else list(range(int(design["n_sequences"])))
                    bases = ["full"] if n == n_full else ["fixed", "subset"]
                    for seq in seqs:
                        for basis in bases:
                            _add(bucket, panel="sample", side_a=a, partners=[b], hold=None, mode=mode, rho=float(rho), accounting="equal_total", basis=basis, n=n, seq=seq, perm_seed=None, rotation=None)
    for lang in langs:
        for hold in visions:
            train_p = [v for v in visions if v != hold]
            for n in design["transfer_sizes"]:
                n = int(n)
                seqs = [0] if n == n_full else list(range(int(design["n_sequences"])))
                basis = "full" if n == n_full else "fixed"
                for seq in seqs:
                    for rho in design["primary_budgets_rho"]:
                        _add(bucket, panel="transfer", side_a=lang, partners=train_p, hold=hold, mode="a_only", rho=float(rho), accounting="equal_total", basis=basis, n=n, seq=seq, perm_seed=None, rotation=None)
                    for src in visions:
                        for rho in [0.0] + list(design["sample_budgets_rho"]):
                            _add(bucket, panel="transfer_source", side_a=lang, partners=[src], hold=None, mode="a_only", rho=float(rho), accounting="equal_total", basis=basis, n=n, seq=seq, perm_seed=None, rotation=None)
    for a, b in design["control_pairs"]:
        for n in design["shuffle_sizes"]:
            n = int(n)
            basis = "full" if n == n_full else "fixed"
            for mode in design["shuffle_modes"]:
                for seed in design["null_perm_seeds"]:
                    _add(bucket, panel="shuffle", side_a=a, partners=[b], hold=None, mode=mode, rho=float(design["shuffle_rho"]), accounting="equal_total", basis=basis, n=n, seq=0, perm_seed=int(seed), rotation=None)
        for rot in range(int(design["n_rotations"])):
            _add(bucket, panel="rotation", side_a=a, partners=[b], hold=None, mode="shared_pc", rho=float(design["rotation_rho"]), accounting="equal_total", basis="rotated", n=n_full, seq=0, perm_seed=None, rotation=int(rot))
    return list(bucket.values())


def _rows(data, key: str, ids: list[str], split: str) -> np.ndarray:
    universe = data["ids"][split]
    pos = {sid: i for i, sid in enumerate(universe)}
    idx = np.array([pos[s] for s in ids], dtype=np.int64)
    return data["prepared"][key][split][idx]


def _basis_for(data, spec, key: str):
    n_full = len(data["ids"]["train"])
    if spec["basis"] == "rotated":
        if key != spec["side_a"]:
            return data["full_basis"][key]
        return with_rotated_basis(data["full_basis"][key], data["rotations"][int(spec["rotation"])])
    if spec["basis"] == "subset" and spec["n"] < n_full:
        return data["subset_basis"][(key, int(spec["seq"]), int(spec["n"]))]
    return data["full_basis"][key]


def _subset_ids(data, spec) -> list[str]:
    n_full = len(data["ids"]["train"])
    if int(spec["n"]) == n_full:
        return list(data["ids"]["train"])
    return list(data["sequences"][int(spec["seq"])][int(spec["n"])])


def _features_for_fit(data, spec):
    ids = _subset_ids(data, spec)
    xa = _rows(data, spec["side_a"], ids, "train")
    ys = []
    bases_b = []
    basis_a = _basis_for(data, spec, spec["side_a"])
    for partner in spec["partners"]:
        y = _rows(data, partner, ids, "train")
        if spec.get("perm_seed") is not None:
            perm = data["perms"]["train"][str(spec["perm_seed"])][str(spec["n"])]
            y = y[np.array(perm, dtype=np.int64)]
        ys.append(y)
        bases_b.append(_basis_for(data, spec, partner))
    _store_basis(data["out"], basis_a, {"model": spec["side_a"], "fit_basis": spec["basis"]})
    for b in bases_b:
        _store_basis(data["out"], b, {"model": "partner", "fit_basis": spec["basis"]})
    y_arg: Any = ys[0] if len(ys) == 1 else ys
    b_arg: Any = bases_b[0] if len(bases_b) == 1 else bases_b
    return xa, y_arg, ids, basis_a, b_arg


def _save_fit(path: Path, rec: dict, spec: dict, config_hash: str) -> None:
    payload = jsonable(rec)
    payload.update(
        {
            "fit_id": spec["fit_id"],
            "panels": spec["panels"],
            "side_a": spec["side_a"],
            "partners": spec["partners"],
            "hold": spec.get("hold"),
            "basis": spec["basis"],
            "seq": spec["seq"],
            "perm_seed": spec.get("perm_seed"),
            "rotation": spec.get("rotation"),
            "config_hash": config_hash,
        }
    )
    write_json(path, payload)


def _load_fit(path: Path) -> dict:
    rec = json.loads(path.read_text())
    for key in ("s_a", "s_b"):
        if key in rec and rec[key] is not None:
            rec[key] = np.asarray(rec[key], dtype=np.float64)
    return rec


def run_fits(data: dict, out: Path, force: bool) -> None:
    specs = build_specs(data)
    write_json(out / "job_index.json", [{k: spec[k] for k in ("fit_id", "panels", "side_a", "partners", "hold", "mode", "rho", "accounting", "basis", "n", "seq", "perm_seed", "rotation")} for spec in specs])
    config_hash = ids_hash([json.dumps(data["design"], sort_keys=True)])
    by_id = {spec["fit_id"]: spec for spec in specs}
    equal = [spec for spec in specs if spec["accounting"] == "equal_total"]
    groups: dict[tuple, list] = {}
    for spec in equal:
        key = (spec["side_a"], tuple(spec["partners"]), spec.get("hold"), spec["mode"], spec["basis"], spec["n"], spec["seq"], spec.get("perm_seed"), spec.get("rotation"))
        groups.setdefault(key, []).append(spec)
    done = 0
    total = len(specs)
    for key, group in groups.items():
        group.sort(key=lambda spec: float(spec["rho"]))
        incumbents: list[dict] = []
        prev_excess = None
        for spec in group:
            rec = _run_one(data, out, spec, incumbents, config_hash, force)
            done += 1
            if rec.get("ok") and not rec.get("skipped"):
                if prev_excess is not None and rec["train_excess"] + MONOTONE_ATOL < prev_excess:
                    raise RuntimeError(f"nested budget lost train excess at {spec['fit_id']}: {rec['train_excess']} < {prev_excess}")
                prev_excess = float(rec["train_excess"])
                incumbents.append({"s_a": rec["s_a"], "s_b": rec["s_b"]})
            else:
                prev_excess = None
                incumbents = []
            if done % 25 == 0 or done == 1:
                print(f"fit {done}/{total} {spec['fit_id']}", flush=True)
    for spec in specs:
        if spec["accounting"] != "per_side":
            continue
        warm_id = _fit_id({**spec, "accounting": "equal_total", "rho": 0.4, "panels": spec["panels"]})
        # per-side rho is 0.4; warm start from the equal-total solution at the same rho if it exists
        warm_spec = {**spec, "accounting": "equal_total", "rho": 0.4}
        warm_spec["fit_id"] = _fit_id(warm_spec)
        warm_path = out / "fits" / f"{warm_spec['fit_id']}.json"
        incumbents = []
        if warm_path.exists():
            warm = _load_fit(warm_path)
            if warm.get("ok") and "s_a" in warm:
                incumbents = [{"s_a": warm["s_a"], "s_b": warm["s_b"]}]
        _run_one(data, out, spec, incumbents, config_hash, force)
        done += 1
    print(f"fits complete {done}", flush=True)
    del by_id


def _run_one(data, out, spec, incumbents, config_hash, force) -> dict:
    dest = out / "fits"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{spec['fit_id']}.json"
    if path.exists() and not force:
        rec = _load_fit(path)
        panels = list(rec.get("panels") or [])
        changed = False
        for panel in spec["panels"]:
            if panel not in panels:
                panels.append(panel)
                changed = True
        if changed:
            rec["panels"] = panels
            _save_fit(path, rec, spec | {"panels": panels}, rec.get("config_hash", config_hash))
        return rec
    basis_a_preview = _basis_for(data, spec, spec["side_a"])
    bases_b = [_basis_for(data, spec, p) for p in spec["partners"]]
    if basis_a_preview["reduced"] or basis_a_preview["q"] != int(data["design"]["q"]) or any(b["reduced"] or b["q"] != int(data["design"]["q"]) for b in bases_b):
        rec = {
            "ok": False,
            "skipped": True,
            "reason": "numerical rank does not support q=32",
            "q_a": basis_a_preview["q"],
            "q_b": [b["q"] for b in bases_b],
            "mode": spec["mode"],
            "rho": spec["rho"],
        }
        _save_fit(path, rec, spec, config_hash)
        return rec
    if spec["mode"] == "shared_pc" and any(b["q"] != basis_a_preview["q"] for b in bases_b):
        rec = {"ok": False, "skipped": True, "reason": "shared_pc q mismatch", "mode": spec["mode"], "rho": spec["rho"]}
        _save_fit(path, rec, spec, config_hash)
        return rec
    xa, y, ids, basis_a, basis_b = _features_for_fit(data, spec)
    t0 = time.perf_counter()
    opt = data["design"]["optim"]
    rec = fit_anisotropic(
        xa,
        y,
        ids,
        basis_a,
        basis_b,
        mode=spec["mode"],
        objective="excess",
        rho=float(spec["rho"]),
        accounting=spec["accounting"],
        init_seed=int(data["design"]["init_seed"]),
        incumbents=incumbents,
        n_steps=int(opt["n_steps"]),
        lr=float(opt["lr"]),
        grad_clip=float(opt["grad_clip"]),
        pert_scale=float(data["design"]["pert_scale"]),
        pack_cache=data["pack_cache"],
        cache_key=spec["data_key"],
    )
    rec["seconds"] = time.perf_counter() - t0
    _save_fit(path, rec, spec, config_hash)
    print(f"  {spec['mode']} rho={spec['rho']} n={spec['n']} excess={rec.get('train_excess')} {rec['seconds']:.1f}s", flush=True)
    return rec


def _gallery_bundle(data, key_a, key_b, split, basis_a, basis_b, ids):
    cache = data["gallery_cache"]
    key = (key_a, key_b, split, basis_a["id"], basis_b["id"], ids_hash(ids))
    if key not in cache:
        xa = _rows(data, key_a, ids, split)
        xb = _rows(data, key_b, ids, split)
        za = xa - basis_a["mu"]
        zb = xb - basis_b["mu"]
        pack = make_pack(za, zb, basis_a["U"], basis_b["U"])
        cache[key] = (pack, za, zb, basis_a["U"], basis_b["U"])
    return cache[key]


def _score_bundle(bundle, s_a, s_b, knn_k: int) -> dict:
    pack, za, zb, ua, ub = bundle
    ba = b_matrix_exp(s_a)
    bb = b_matrix_exp(s_b)
    sc = scores_numpy(ba - np.eye(ba.shape[0]), bb - np.eye(bb.shape[0]), pack)
    from prh_replication.anisotropic_kernels import knn_from_sq, metric_sq_distances, mnn_from_knn

    mnn = mnn_from_knn(knn_from_sq(metric_sq_distances(za, ua, ba), knn_k), knn_from_sq(metric_sq_distances(zb, ub, bb), knn_k))
    return {
        "a": sc["a"],
        "b": sc["b"],
        "ratio": sc["ratio"] if sc["ratio"] == sc["ratio"] else None,
        "excess": sc["excess"],
        "valid_a": sc["valid_a"],
        "degenerate": sc["degenerate"],
        "r_eff_k": sc["r_eff_k"],
        "r_eff_l": sc["r_eff_l"],
        "mnn_k10": mnn,
        "n": int(za.shape[0]),
    }


def run_eval(data: dict, out: Path, force: bool) -> None:
    specs = build_specs(data)
    by_id = {spec["fit_id"]: spec for spec in specs}
    ev_dir = out / "evals"
    ev_dir.mkdir(parents=True, exist_ok=True)
    knn_k = int(data["design"]["knn_k"])
    for i, spec in enumerate(specs):
        path = ev_dir / f"{spec['fit_id']}.json"
        fit_path = out / "fits" / f"{spec['fit_id']}.json"
        if not fit_path.exists():
            raise FileNotFoundError(fit_path)
        if path.exists() and not force:
            continue
        fit = _load_fit(fit_path)
        if not fit.get("ok") or fit.get("skipped"):
            write_json(path, {"fit_id": spec["fit_id"], "skipped": True})
            continue
        payload = _eval_spec(data, spec, fit, knn_k)
        payload["fit_id"] = spec["fit_id"]
        payload["test_role"] = "exploratory_heldout"
        write_json(path, jsonable(payload))
        if (i + 1) % 50 == 0:
            print(f"eval {i+1}/{len(specs)}", flush=True)
    _select(out, specs)
    _transfer_table(data, out, specs)
    _stability(data, out, specs)
    print("eval complete", flush=True)
    del by_id


def _eval_spec(data, spec, fit, knn_k) -> dict:
    s_a, s_b = fit["s_a"], fit["s_b"]
    if spec.get("perm_seed") is not None:
        val_ids = list(data["ids"]["val"])
        perm = np.array(data["perms"]["val"][str(spec["perm_seed"])], dtype=np.int64)
        # Shuffled validation correspondence: permute partner rows only.
        basis_a = _basis_for(data, spec, spec["side_a"])
        basis_b = _basis_for(data, spec, spec["partners"][0])
        xa = _rows(data, spec["side_a"], val_ids, "val")
        xb = _rows(data, spec["partners"][0], val_ids, "val")[perm]
        val = evaluate_metric(xa, xb, basis_a, basis_b, s_a, s_b, knn_k=knn_k)
        test = _score_true(data, spec, s_a, s_b, "test", knn_k)
        return {"val": val, "val_correspondence": "shuffled", "test": test, "test_correspondence": "true"}
    if spec.get("hold"):
        val_parts = []
        for partner in spec["partners"]:
            val_parts.append(_score_true_pair(data, spec, spec["side_a"], partner, s_a, s_b, "val", knn_k))
        held = _score_true_pair(data, spec, spec["side_a"], spec["hold"], s_a, s_b, "test", knn_k)
        mean_val = float(np.nanmean([p["excess"] for p in val_parts]))
        return {
            "val_partners": val_parts,
            "val_excess_mean_train_partners": mean_val,
            "val": {"excess": mean_val, "a": float(np.nanmean([p["a"] for p in val_parts])), "b": float(np.nanmean([p["b"] for p in val_parts]))},
            "test": held,
            "test_correspondence": "held_out_partner_true",
        }
    test = _score_true(data, spec, s_a, s_b, "test", knn_k)
    val = _score_true(data, spec, s_a, s_b, "val", knn_k)
    return {"val": val, "test": test, "test_correspondence": "true"}


def _score_true(data, spec, s_a, s_b, split, knn_k) -> dict:
    return _score_true_pair(data, spec, spec["side_a"], spec["partners"][0], s_a, s_b, split, knn_k)


def _score_true_pair(data, spec, key_a, key_b, s_a, s_b, split, knn_k) -> dict:
    ids = list(data["ids"][split])
    basis_a = _basis_for(data, spec, key_a)
    # Partner basis follows the spec protocol for that model, not a rotated side-B basis unless key is side A.
    basis_spec = dict(spec)
    if key_b != spec["side_a"]:
        basis_spec = {**spec, "rotation": None, "basis": spec["basis"] if spec["basis"] != "rotated" else "full"}
    basis_b = _basis_for(data, basis_spec, key_b)
    bundle = _gallery_bundle(data, key_a, key_b, split, basis_a, basis_b, ids)
    return _score_bundle(bundle, s_a, s_b, knn_k)


def _select(out: Path, specs: list[dict]) -> None:
    groups: dict[tuple, list] = {}
    for spec in specs:
        if spec["accounting"] != "equal_total":
            continue
        if spec.get("perm_seed") is not None or spec.get("rotation") is not None:
            continue
        key = (spec["side_a"], tuple(spec["partners"]), spec.get("hold"), spec["mode"], spec["basis"], spec["n"], spec["seq"])
        groups.setdefault(key, []).append(spec)
    dest = out / "selections"
    dest.mkdir(parents=True, exist_ok=True)
    for key, group in groups.items():
        rows = []
        for spec in group:
            ev = json.loads((out / "evals" / f"{spec['fit_id']}.json").read_text())
            fit = _load_fit(out / "fits" / f"{spec['fit_id']}.json")
            if ev.get("skipped") or not fit.get("ok"):
                continue
            val = ev["val"]["excess"]
            rows.append({"rho": spec["rho"], "val_excess": val, "d_total": fit["d_total"], "fit_id": spec["fit_id"]})
        if not rows:
            continue
        chosen = select_budget(rows)
        write_json(dest / f"{ids_hash([str(k) for k in key])}.json", {"key": [None if k is None else k if not isinstance(k, tuple) else list(k) for k in key], "chosen": chosen, "candidates": rows})


def _transfer_table(data, out, specs) -> None:
    """Cross-partner scores. Budget choice for the shared metric uses training-partner val only."""
    rows = []
    knn_k = int(data["design"]["knn_k"])
    designs = data["design"]
    n_full = len(data["ids"]["train"])
    for lang in data["langs"]:
        for hold in VIS:
            train_p = [v for v in VIS if v != hold]
            for n in designs["transfer_sizes"]:
                n = int(n)
                seqs = [0] if n == n_full else list(range(int(designs["n_sequences"])))
                basis = "full" if n == n_full else "fixed"
                for seq in seqs:
                    cands = []
                    shared_fits = {}
                    for rho in designs["primary_budgets_rho"]:
                        spec = {"side_a": lang, "partners": train_p, "hold": hold, "mode": "a_only", "rho": float(rho), "accounting": "equal_total", "basis": basis, "n": n, "seq": seq, "perm_seed": None, "rotation": None}
                        fid = _fit_id(spec)
                        ev = json.loads((out / "evals" / f"{fid}.json").read_text())
                        fit = _load_fit(out / "fits" / f"{fid}.json")
                        if not fit.get("ok"):
                            continue
                        cands.append({"rho": float(rho), "val_excess": ev["val"]["excess"], "d_total": fit["d_total"], "fit_id": fid})
                        shared_fits[float(rho)] = fit
                    if not cands:
                        continue
                    chosen = select_budget(cands)
                    for rho, fit in shared_fits.items():
                        spec = {"side_a": lang, "partners": train_p, "hold": hold, "mode": "a_only", "rho": rho, "accounting": "equal_total", "basis": basis, "n": n, "seq": seq, "perm_seed": None, "rotation": None, "panels": ["transfer"]}
                        held = _score_true_pair(data, spec, lang, hold, fit["s_a"], fit["s_b"], "test", knn_k)
                        source_scores = {}
                        for src in train_p + [hold]:
                            src_spec = {"side_a": lang, "partners": [src], "hold": None, "mode": "a_only", "rho": rho, "accounting": "equal_total", "basis": basis, "n": n, "seq": seq, "perm_seed": None, "rotation": None}
                            src_fit = _load_fit(out / "fits" / f"{_fit_id(src_spec)}.json")
                            if not src_fit.get("ok"):
                                continue
                            source_scores[src] = _score_true_pair(data, src_spec, lang, hold, src_fit["s_a"], src_fit["s_b"], "test", knn_k)["excess"]
                        rows.append(
                            {
                                "language": lang,
                                "held_out": hold,
                                "train_partners": train_p,
                                "n": n,
                                "seq": seq,
                                "rho": rho,
                                "selected": abs(float(rho) - float(chosen["rho"])) < 1e-12,
                                "selected_rho": chosen["rho"],
                                "shared_excess": held["excess"],
                                "shared_a": held["a"],
                                "shared_mnn": held["mnn_k10"],
                                "source_excess_on_heldout": source_scores,
                                "identity_excess": source_scores.get(hold) if rho == 0 else None,
                            }
                        )
    write_json(out / "transfer_scores.json", rows)


def _stability(data, out, specs) -> None:
    knn_k = int(data["design"]["knn_k"])
    del knn_k
    n_full = len(data["ids"]["train"])
    records = []
    sample = [spec for spec in specs if "sample" in spec["panels"] and spec["mode"] in ("a_only", "separate") and spec["accounting"] == "equal_total" and spec["rho"] in (0.1, 0.4)]
    # Compare replicates at the same n, and each replicate to the full-data fit.
    groups: dict[tuple, list] = {}
    for spec in sample:
        if spec["n"] == n_full:
            continue
        key = (spec["side_a"], tuple(spec["partners"]), spec["mode"], spec["rho"], spec["basis"], spec["n"])
        groups.setdefault(key, []).append(spec)
    test_ids = list(data["ids"]["test"])
    for key, group in groups.items():
        full_spec = {**group[0], "n": n_full, "seq": 0, "basis": "full"}
        full_path = out / "fits" / f"{_fit_id(full_spec)}.json"
        full = _load_fit(full_path) if full_path.exists() else None
        fits = []
        for spec in group:
            fit = _load_fit(out / "fits" / f"{spec['fit_id']}.json")
            if fit.get("ok") and not fit.get("skipped"):
                fits.append((spec, fit))
        if len(fits) < 2:
            continue
        for side in ("a", "b"):
            pair_cos = []
            pair_dist = []
            overlaps = {("amplified", 4): [], ("amplified", 8): [], ("suppressed", 4): [], ("suppressed", 8): []}
            to_full = []
            gram_cos = []
            corr_cos = []
            undefined = 0
            for i in range(len(fits)):
                for j in range(i + 1, len(fits)):
                    cmp = _compare_sides(data, fits[i], fits[j], side, test_ids)
                    if cmp["delta"]["cosine"] is None:
                        undefined += 1
                    else:
                        pair_cos.append(cmp["delta"]["cosine"])
                        pair_dist.append(cmp["delta"]["distance"])
                    for slot in overlaps:
                        which, k = slot
                        val = cmp["subspaces"][which][k]
                        overlaps[slot].append(val)
                    if cmp["gram_cosine"] is not None:
                        gram_cos.append(cmp["gram_cosine"])
                    if cmp["correction_cosine"] is not None:
                        corr_cos.append(cmp["correction_cosine"])
            if full and full.get("ok"):
                for spec, fit in fits:
                    cmp = _compare_sides(data, (spec, fit), (full_spec, full), side, test_ids)
                    to_full.append(cmp["delta"]["cosine"])
            records.append(
                {
                    "side_a": key[0],
                    "partner": key[1][0],
                    "mode": key[2],
                    "rho": key[3],
                    "basis": key[4],
                    "n": key[5],
                    "side": side,
                    "n_replicates": len(fits),
                    "delta_cosine_replicates": pair_cos,
                    "delta_distance_replicates": pair_dist,
                    "cosine_to_full": to_full,
                    "undefined_cosine": undefined,
                    "amplified_overlap_k4": overlaps[("amplified", 4)],
                    "amplified_overlap_k8": overlaps[("amplified", 8)],
                    "suppressed_overlap_k4": overlaps[("suppressed", 4)],
                    "suppressed_overlap_k8": overlaps[("suppressed", 8)],
                    "gram_cosine": gram_cos,
                    "correction_cosine": corr_cos,
                }
            )
    write_json(out / "stability.json", records)


def _side_usb(data, spec, fit, side: str):
    if side == "a":
        basis = _basis_for(data, spec, spec["side_a"])
        return basis["U"], fit["s_a"], b_matrix_exp(fit["s_a"]), basis["mu"], spec["side_a"]
    partner = spec["partners"][0]
    basis = _basis_for(data, spec, partner)
    return basis["U"], fit["s_b"], b_matrix_exp(fit["s_b"]), basis["mu"], partner


def _compare_sides(data, left, right, side, test_ids) -> dict:
    spec1, fit1 = left
    spec2, fit2 = right
    u1, s1, b1, mu1, key = _side_usb(data, spec1, fit1, side)
    u2, s2, b2, mu2, key2 = _side_usb(data, spec2, fit2, side)
    delta = delta_m_cosine(u1, b1, u2, b2)
    subspaces = {"amplified": {}, "suppressed": {}}
    for which in ("amplified", "suppressed"):
        for k in (4, 8):
            cmp = subspace_compare(u1, s1, u2, s2, which, k)
            subspaces[which][k] = cmp["overlap"] if cmp["defined"] else None
    x = data["prepared"][key]["test"] if key == key2 else None
    gcos = None
    ccos = None
    if x is not None and side == "a":
        # Induced geometry of this side alone is not a paired Gram. Compare the
        # paired centred kernels of the two fits on the fixed exploratory gallery.
        pass
    if spec1["partners"] == spec2["partners"] and side == "a":
        k1 = _paired_correction(data, spec1, fit1, test_ids)
        k2 = _paired_correction(data, spec2, fit2, test_ids)
        ccos = gram_cosine(k1, k2)
        g1 = _paired_centred(data, spec1, fit1, test_ids)
        g2 = _paired_centred(data, spec2, fit2, test_ids)
        gcos = gram_cosine(g1, g2)
    return {"delta": delta, "subspaces": subspaces, "gram_cosine": gcos, "correction_cosine": ccos}


def _paired_centred(data, spec, fit, test_ids):
    from prh_replication.metric_stability import centred_kernel

    ba = _basis_for(data, spec, spec["side_a"])
    bb = _basis_for(data, spec, spec["partners"][0])
    # Similarity of the pair of centred kernels is computed on side A's kernel
    # and, separately, the caller averages. Here return side A.
    xa = _rows(data, spec["side_a"], test_ids, "test")
    return centred_kernel(xa, ba["mu"], ba["U"], b_matrix_exp(fit["s_a"]))


def _paired_correction(data, spec, fit, test_ids):
    ba = _basis_for(data, spec, spec["side_a"])
    xa = _rows(data, spec["side_a"], test_ids, "test")
    return correction_gram(xa, ba["mu"], ba["U"], b_matrix_exp(fit["s_a"]))


def smoke(data: dict, out: Path) -> None:
    """Training-only check on one predetermined pair. Does not score the exploratory split."""
    design = data["design"]
    a, b = design["control_pairs"][0]
    opt = design["optim"]
    smoke_dir = out / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ids = list(data["ids"]["train"])
    xa = data["prepared"][a]["train"]
    xb = data["prepared"][b]["train"]
    ba, bb = data["full_basis"][a], data["full_basis"][b]
    report = {"pair": [a, b], "gallery": "train", "fits": []}
    for mode in ("a_only", "separate"):
        rec = fit_anisotropic(
            xa, xb, ids, ba, bb, mode=mode, rho=0.4, accounting="equal_total",
            init_seed=int(design["init_seed"]), n_steps=int(opt["n_steps"]), lr=float(opt["lr"]),
            grad_clip=float(opt["grad_clip"]), pert_scale=float(design["pert_scale"]),
        )
        ident = next(s for s in rec["status"]["starts"] if s["name"] == "identity")
        report["fits"].append(
            {
                "mode": mode,
                "train_excess": rec["train_excess"],
                "d_total": rec["d_total"],
                "feasible": rec["feasible"],
                "zero_grad_at_identity": rec["status"]["zero_grad_at_identity"],
                "identity_grad_norm": ident.get("identity_grad_norm"),
                "no_grad": rec["status"]["no_grad"],
                "selected_start": rec["selected_start"],
                "identity_trace_head": (ident.get("trace") or [])[:5],
                "identity_trace_tail": (ident.get("trace") or [])[-5:],
                "late_gain": _late(ident.get("trace") or []),
            }
        )
        if rec["status"]["zero_grad_at_identity"] or rec["status"]["no_grad"]:
            write_json(smoke_dir / "smoke.json", report)
            raise SystemExit("identity-start gradient missing; not starting the sweep")
        if not rec["feasible"] or rec["status"]["constraint_violation"]:
            write_json(smoke_dir / "smoke.json", report)
            raise SystemExit("constraint violation in smoke fit")
    write_json(smoke_dir / "smoke.json", report)
    print("smoke", json.dumps(report["fits"]), flush=True)


def _late(trace: list) -> float | None:
    vals = [v for v in trace if isinstance(v, (int, float))]
    if len(vals) < 21:
        return None
    return float(vals[-1] - vals[-21])


def write_report(data: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    rows = _joined_rows(out)
    stability = json.loads((out / "stability.json").read_text()) if (out / "stability.json").exists() else []
    transfer = json.loads((out / "transfer_scores.json").read_text()) if (out / "transfer_scores.json").exists() else []
    write_json(out / "joined_rows.json", rows)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    _fig_distortion(rows, fig / "heldout_excess_vs_distortion.png")
    _fig_vs_n(rows, fig / "heldout_cka_mnn_vs_n.png")
    _fig_stability(stability, fig / "stability_vs_n.png")
    _fig_actual_d(rows, fig / "actual_distortion_vs_n.png")
    _fig_transfer(transfer, fig / "transfer_vs_n.png")
    _fig_rotation(data, out, fig / "shared_pc_basis_sensitivity.png")
    _fig_shuffle(rows, fig / "shuffle_control.png")
    summary = _summary_numbers(rows, stability, transfer, data, out)
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(_prose(summary, data))
    print("report", out / "report.md", flush=True)


def _joined_rows(out: Path) -> list[dict]:
    rows = []
    fit_dir = out / "fits"
    for path in sorted(fit_dir.glob("*.json")):
        fit = json.loads(path.read_text())
        ev_path = out / "evals" / path.name
        ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
        test = ev.get("test") or {}
        val = ev.get("val") or {}
        rows.append(
            {
                "fit_id": fit.get("fit_id"),
                "panels": fit.get("panels") or [],
                "side_a": fit.get("side_a"),
                "partners": fit.get("partners"),
                "hold": fit.get("hold"),
                "mode": fit.get("mode"),
                "rho": fit.get("rho"),
                "accounting": fit.get("accounting"),
                "basis": fit.get("basis"),
                "n": fit.get("n"),
                "seq": fit.get("seq"),
                "perm_seed": fit.get("perm_seed"),
                "rotation": fit.get("rotation"),
                "skipped": bool(fit.get("skipped")),
                "d_a": fit.get("d_a"),
                "d_b": fit.get("d_b"),
                "d_total": fit.get("d_total"),
                "train_excess": fit.get("train_excess"),
                "decompose_a": fit.get("decompose_a"),
                "decompose_b": fit.get("decompose_b"),
                "diag_a": fit.get("diag_a"),
                "diag_b": fit.get("diag_b"),
                "val_excess": None if not isinstance(val, dict) else val.get("excess"),
                "test_excess": None if not isinstance(test, dict) else test.get("excess"),
                "test_a": None if not isinstance(test, dict) else test.get("a"),
                "test_b": None if not isinstance(test, dict) else test.get("b"),
                "test_mnn": None if not isinstance(test, dict) else test.get("mnn_k10"),
                "selected_start": fit.get("selected_start"),
            }
        )
    return rows


def _summary_numbers(rows, stability, transfer, data, out) -> dict:
    def med(vals):
        v = [float(x) for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
        return None if not v else float(np.median(v))

    primary = [r for r in rows if "primary" in r["panels"] and r["accounting"] == "equal_total" and not r["skipped"] and r["perm_seed"] is None and r["rotation"] is None]
    by_mode = {}
    for mode in ("identity", "a_only", "b_only", "separate", "shared_pc"):
        by_mode[mode] = {}
        for rho in (0.0, 0.1, 0.4):
            part = [r for r in primary if r["mode"] == mode and abs(float(r["rho"]) - rho) < 1e-9]
            by_mode[mode][str(rho)] = {
                "n": len(part),
                "median_test_excess": med([r["test_excess"] for r in part]),
                "median_test_a": med([r["test_a"] for r in part]),
                "median_d_total": med([r["d_total"] for r in part]),
            }
    # pair-wise separate minus best one-sided
    gains = []
    for rho in (0.1, 0.4):
        bucket = {}
        for r in primary:
            if abs(float(r["rho"]) - rho) > 1e-9:
                continue
            bucket.setdefault((r["side_a"], tuple(r["partners"] or []), r["mode"]), r)
        keys = {(a, p) for (a, p, mode) in bucket}
        for a, p in keys:
            sep = bucket.get((a, p, "separate"))
            one = [bucket.get((a, p, m)) for m in ("a_only", "b_only") if bucket.get((a, p, m))]
            if sep and one and sep["test_excess"] is not None:
                best = max(float(r["test_excess"]) for r in one)
                gains.append({"rho": rho, "side_a": a, "partner": p[0], "separate_minus_best_onesided": float(sep["test_excess"]) - best, "family": "ll" if p[0] not in VIS else "vl"})
    def stab(basis, field):
        vals = []
        for rec in stability:
            if rec["basis"] != basis or rec["side"] != "a" or rec["mode"] != "a_only":
                continue
            if abs(float(rec["rho"]) - 0.4) > 1e-9 or int(rec["n"]) != 1024:
                continue
            vals.extend([v for v in rec[field] if isinstance(v, (int, float)) and np.isfinite(v)])
        return med(vals)

    cos_fixed, cos_subset = stab("fixed", "delta_cosine_replicates"), stab("subset", "delta_cosine_replicates")
    ov_fixed, ov_subset = stab("fixed", "amplified_overlap_k4"), stab("subset", "amplified_overlap_k4")
    if cos_fixed is not None and cos_subset is not None and cos_fixed >= 0.9 and cos_subset >= 0.9:
        claim = "Both basis protocols have replicate ΔM cosine at least 0.9 at n=1024, so the corrections themselves agree, not only the alignment scores."
    elif ov_fixed is not None and ov_subset is not None and ov_fixed >= 0.9 and ov_subset >= 0.9:
        claim = "Rank-4 amplified subspaces agree across replicates at n=1024, but the full ΔM cosine does not meet 0.9. The supported claim is stable subspaces, not a unique metric."
    else:
        claim = "Replicate ΔM cosine and rank-4 amplified overlap do not both clear 0.9 at n=1024 on both basis protocols. The supported claim is about alignment scores, not stable recovered metrics or subspaces."
    return {
        "by_mode": by_mode,
        "separate_minus_onesided": gains,
        "median_separate_gain_rho0.4": med([g["separate_minus_best_onesided"] for g in gains if g["rho"] == 0.4]),
        "median_separate_gain_rho0.1": med([g["separate_minus_best_onesided"] for g in gains if g["rho"] == 0.1]),
        "n_primary_pairs": len({(r["side_a"], tuple(r["partners"] or [])) for r in primary if r["mode"] == "identity"}),
        "stability_n": len(stability),
        "transfer_n": len(transfer),
        "skipped": sum(1 for r in rows if r["skipped"]),
        "cosine_replicates_n1024_fixed": cos_fixed,
        "cosine_replicates_n1024_subset": cos_subset,
        "amplified_overlap_k4_n1024_fixed": ov_fixed,
        "amplified_overlap_k4_n1024_subset": ov_subset,
        "claim": claim,
        "cutoffs": {"delta_cosine": 0.9, "subspace_overlap": 0.9, "excess_small": 0.01},
    }


def _prose(summary: dict, data: dict) -> str:
    def _fmt(v):
        return "undefined" if v is None else f"{float(v):.4f}"

    g04 = summary.get("median_separate_gain_rho0.4")
    g01 = summary.get("median_separate_gain_rho0.1")
    gain_rows = summary.get("separate_minus_onesided") or []
    lines = [
        "# Metric stability",
        "",
        "Exploratory held-out evaluation on the manifest split named test. That gallery was already inspected in earlier experiments. It did not select budgets, initialisations, or subsets.",
        "",
        "Side A is the language model on vision–language pairs and the Qwen model on language–language pairs. Shared-PC means one symmetric S in each model’s own 32-PC basis, not a shared ambient metric and not shared semantic directions.",
        "",
        "## Equal-total exploratory excess",
        "",
        "| Mode | ρ | n | median excess | median raw CKA | median total D |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, rhos in (summary.get("by_mode") or {}).items():
        for rho, rec in rhos.items():
            lines.append(
                f"| {mode} | {rho} | {rec['n']} | {_fmt(rec['median_test_excess'])} | {_fmt(rec['median_test_a'])} | {_fmt(rec['median_d_total'])} |"
            )
    def _gain_count(rho, thresh=0.01):
        vals = [g["separate_minus_best_onesided"] for g in gain_rows if abs(float(g["rho"]) - rho) < 1e-9]
        return len(vals), sum(v > thresh for v in vals)

    n01, k01 = _gain_count(0.1)
    n04, k04 = _gain_count(0.4)
    lines += [
        "",
        "## Answers",
        "",
        f"1. At equal total distortion, the median exploratory excess of separate fitting minus the better one-sided fit is {_fmt(g01)} at ρ=0.1 ({k01}/{n01} pairs above +0.01) and {_fmt(g04)} at ρ=0.4 ({k04}/{n04} pairs above +0.01). Per-side ρ=0.4 allows twice the total distortion of one-sided ρ=0.4 and is not this comparison. Pair values are in `summary.json`.",
        "2. Shared-PC scores and the basis-rotation control are in `figures/shared_pc_basis_sensitivity.png`. The conjugated independent metric should move by numerical error only; shared-PC refits can move if the coordinate convention matters.",
        "3. Held-out raw CKA and mNN versus fitting size are in `figures/heldout_cka_mnn_vs_n.png`. Fixed-budget curves are the primary ones.",
        "4. Amplified and suppressed subspace overlap versus size is in `figures/stability_vs_n.png`. Undefined cosines stay missing.",
        "5. That figure splits fixed-basis and subset-basis protocols. Agreement that appears only when the PCA basis is frozen is conditional on that basis.",
        "6. Partner transfer is in `figures/transfer_vs_n.png`. The shared language metric is chosen with training-partner validation only. The target-specific curve is a comparator, not held-out-partner generalisation.",
        "7. Separate-mode side-A and side-B replicate cosines are stored separately in `stability.json`. Similar scores with weaker side-specific cosine mean the two-sided correction is less identifiable.",
        "8. Pairs share models and images. The shuffle panel is three pairs. Full-data fits are not sampling replicates. No significance claims are made from that panel.",
        "",
        "Descriptive cutoffs were fixed in the protocol before this evaluation: ΔM cosine 0.9 and rank-4 amplified overlap 0.9 at n=1024.",
        "",
        f"Median replicate ΔM cosine at n=1024, A-only, ρ=0.4, side A: fixed basis {summary.get('cosine_replicates_n1024_fixed')}, subset basis {summary.get('cosine_replicates_n1024_subset')}.",
        f"Median rank-4 amplified overlap at that size: fixed {summary.get('amplified_overlap_k4_n1024_fixed')}, subset {summary.get('amplified_overlap_k4_n1024_subset')}.",
        "",
        summary.get("claim") or "",
        "",
        f"Skipped fits (rank or q mismatch): {summary.get('skipped')}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fig_distortion(rows, path):
    import matplotlib.pyplot as plt

    part = [r for r in rows if "primary" in r["panels"] and not r["skipped"] and r["perm_seed"] is None and r["rotation"] is None and r["test_excess"] is not None]
    modes = ["a_only", "b_only", "separate", "shared_pc"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, mode in zip(axes, modes):
        for r in part:
            if r["mode"] != mode:
                continue
            if r["accounting"] not in ("equal_total", "per_side"):
                continue
            marker = "x" if r["accounting"] == "per_side" else "o"
            ax.scatter(r["d_total"], r["test_excess"], marker=marker, s=28, alpha=0.85)
        ax.set_title(mode)
        ax.set_xlabel("actual total distortion")
    axes[0].set_ylabel("exploratory held-out excess")
    fig.suptitle("Equal-total fits (circles) and per-side ρ=0.4 (crosses). Side A = language or Qwen.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_vs_n(rows, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, basis_name in enumerate(("fixed", "subset")):
        for row_i, (key, label) in enumerate((("test_a", "exploratory raw CKA"), ("test_mnn", "exploratory mNN k=10"))):
            ax = axes[row_i][col]
            for mode, ls in (("a_only", "-"), ("separate", "--")):
                part = [r for r in rows if "sample" in r["panels"] and r["mode"] == mode and abs(float(r["rho"] or -1) - 0.4) < 1e-9 and r["accounting"] == "equal_total" and not r["skipped"]]
                part = [r for r in part if r["basis"] == basis_name or (basis_name == "fixed" and r["basis"] == "full")]
                by = {}
                for r in part:
                    by.setdefault((r["side_a"], r["partners"][0]), []).append(r)
                for (a, b), items in by.items():
                    xs, ys = [], []
                    for n in sorted({int(r["n"]) for r in items}):
                        vals = [r[key] for r in items if int(r["n"]) == n and r[key] is not None]
                        if vals:
                            xs.append(n)
                            ys.append(float(np.median(vals)))
                    if xs:
                        ax.plot(xs, ys, ls=ls, marker="o", ms=3, label=f"{a} {mode}" if row_i == 0 and col == 0 else None)
            ax.set_title(f"{label}, {basis_name} basis, ρ=0.4")
            ax.set_xlabel("fitting sample size")
    fig.suptitle("A-only solid, separate dashed. Median across subset sequences. DINOv2 and the three language pairs.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_stability(records, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, basis in zip(axes, ("fixed", "subset")):
        part = [r for r in records if r["basis"] == basis and r["side"] == "a" and abs(float(r["rho"]) - 0.4) < 1e-9 and r["mode"] == "a_only"]
        by = {}
        for r in part:
            by.setdefault(r["side_a"], []).append(r)
        for model, items in by.items():
            xs, ys = [], []
            for n in sorted({int(r["n"]) for r in items}):
                vals = []
                for r in items:
                    if int(r["n"]) == n:
                        vals.extend([v for v in r["cosine_to_full"] if v is not None])
                if vals:
                    xs.append(n)
                    ys.append(float(np.median(vals)))
            if xs:
                ax.plot(xs, ys, marker="o", label=model)
        ax.set_title(f"ΔM cosine to full-data fit, {basis} basis")
        ax.set_xlabel("fitting sample size")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Frobenius cosine of ΔM (side A)")
    fig.suptitle("A-only, ρ=0.4. Missing cosines are omitted, not drawn as 0 or 1.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_actual_d(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    part = [r for r in rows if "sample" in r["panels"] and r["mode"] == "separate" and abs(float(r["rho"] or -1) - 0.4) < 1e-9 and r["basis"] in ("fixed", "full") and not r["skipped"]]
    by = {}
    for r in part:
        by.setdefault(r["side_a"], []).append(r)
    for model, items in by.items():
        xs, ys = [], []
        for n in sorted({int(r["n"]) for r in items}):
            vals = [r["d_total"] for r in items if int(r["n"]) == n and r["d_total"] is not None]
            if vals:
                xs.append(n)
                ys.append(float(np.median(vals)))
        ax.plot(xs, ys, marker="o", label=model)
    ax.axhline(0.4, color="k", ls="--", lw=0.8, label="ρ=0.4 allowance")
    ax.set_xlabel("fitting sample size")
    ax.set_ylabel("actual total distortion")
    ax.set_title("Separate mode, fixed basis, equal-total ρ=0.4")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_transfer(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    part = [r for r in rows if abs(float(r["rho"]) - 0.4) < 1e-9]
    # median over sequences and held-out partners, one line per language, shared metric
    by = {}
    for r in part:
        by.setdefault(r["language"], []).append(r)
    for lang, items in by.items():
        xs, ys = [], []
        for n in sorted({int(r["n"]) for r in items}):
            vals = [r["shared_excess"] for r in items if int(r["n"]) == n and r["shared_excess"] is not None]
            if vals:
                xs.append(n)
                ys.append(float(np.median(vals)))
        ax.plot(xs, ys, marker="o", label=lang)
    ax.set_xlabel("fitting sample size")
    ax.set_ylabel("held-out partner exploratory excess")
    ax.set_title("Shared language metric at ρ=0.4. Median over sequences and held-out vision partners.")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _eval_excess(path: Path):
    if not path.exists():
        return None
    ev = json.loads(path.read_text())
    test = ev.get("test")
    if not isinstance(test, dict):
        return None
    return test.get("excess")


def _fig_rotation(data, out, path):
    import matplotlib.pyplot as plt

    rows = []
    design = data["design"]
    for a, b in design["control_pairs"]:
        base = {"side_a": a, "partners": [b], "hold": None, "mode": "shared_pc", "rho": float(design["rotation_rho"]), "accounting": "equal_total", "basis": "full", "n": len(data["ids"]["train"]), "seq": 0, "perm_seed": None, "rotation": None}
        base_ex = _eval_excess(out / "evals" / f"{_fit_id(base)}.json")
        for rot in range(int(design["n_rotations"])):
            spec = {**base, "basis": "rotated", "rotation": rot}
            ex = _eval_excess(out / "evals" / f"{_fit_id(spec)}.json")
            rows.append((f"{a}\nrot{rot}", None if base_ex is None or ex is None else ex - base_ex))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(rows)), [0 if v is None else v for v in [r[1] for r in rows]])
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=90, fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("shared-PC excess minus unrotated")
    ax.set_title("Basis rotation of side A at full n, ρ=0.4. Exploratory held-out.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_shuffle(rows, path):
    import matplotlib.pyplot as plt

    part = [r for r in rows if "shuffle" in r["panels"] and not r["skipped"] and r["test_excess"] is not None]
    labels, true_vals, shuf_vals = [], [], []
    # Compare each shuffled fit's true-test excess to the unpermuted fit at the same n, mode, rho.
    for r in part:
        labels.append(f"{r['side_a'][:8]} n{r['n']} {r['mode'][0]} p{r['perm_seed']}")
        shuf_vals.append(r["test_excess"])
        twin = [q for q in rows if q["side_a"] == r["side_a"] and q["partners"] == r["partners"] and q["mode"] == r["mode"] and q["n"] == r["n"] and q["perm_seed"] is None and q["rotation"] is None and abs(float(q["rho"]) - float(r["rho"])) < 1e-9 and q["accounting"] == "equal_total" and q["basis"] in ("full", "fixed", r["basis"])]
        true_vals.append(twin[0]["test_excess"] if twin else None)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.15, [np.nan if v is None else v for v in true_vals], 0.3, label="true correspondence fit")
    ax.bar(x + 0.15, shuf_vals, 0.3, label="shuffled-correspondence fit, true exploratory eval")
    ax.set_xticks(x, labels, rotation=90, fontsize=6)
    ax.set_ylabel("exploratory excess")
    ax.set_title("Shuffle control at ρ=0.4. Not a standard error for observed CKA.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
