"""Bounded comparison of exp(S) and regularised nonorthogonal factor metrics.

The exploratory-test gallery is scored only in the eval stage. Checkpoint
selection uses the validation gallery. Historical result directories are read
only to record their modification times.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from prh_replication.anisotropic_kernels import BOUND_ATOL, LOG4
from prh_replication.extract_final import final_feature_path, load_final_prepared, spec_from_manifest
from prh_replication.io_utils import jsonable, sha256_file, write_json
from prh_replication.metric_stability import assert_split_image_disjoint, fit_basis, ids_hash
from prh_replication.plots import _save
from prh_replication.registry import MODELS, Paths
from prh_replication.release_protocol import VIS, load_splits
from prh_replication.structured_factors import (
    build_jobs,
    complexity_total,
    condition_number,
    distortion_of_b,
    induced_b,
    job_name,
    learns_basis,
    off_norm,
    penalised_selection,
    r_basis,
    r_off,
    run_factor_stream,
    score_metrics,
    singular_values,
    sorted_log_spectrum,
    spectrum_distance,
    total_distortion,
)

CTX: dict[str, Any] = {}
MODES_PLOT = ("identity", "a_only", "shared_pc", "separate")
MODE_LABEL = {"identity": "Identity", "a_only": "One-sided", "shared_pc": "Shared-PC", "separate": "Separate"}


def run(work: Path, repo: Path, stage: str, force: bool = False, workers: int = 8) -> None:
    paths = Paths(work=work, repo=repo)
    design = json.loads((repo / "configs" / "structured_metric_factors.json").read_text())
    out = paths.results / "structured_metric_factors"
    out.mkdir(parents=True, exist_ok=True)
    if stage in ("prepare", "smoke", "fit", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val"), force=force)
        _prepare(ctx)
    if stage in ("smoke", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val"), force=False)
        _smoke(ctx)
    if stage in ("fit", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val"), force=False)
        _fit(ctx, force=force, workers=workers)
        print("FIT_DONE", flush=True)
    if stage in ("eval", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val", "test"), force=False)
        _eval(ctx, force=force, workers=workers)
        print("EVAL_DONE", flush=True)
    if stage in ("report", "all"):
        _figures(out)
        _write_report(out, design)
        _verify_protected(out)
        print("REPORT_DONE", flush=True)


def _context(paths: Paths, repo: Path, design: dict, out: Path, *, splits: tuple[str, ...], force: bool) -> dict[str, Any]:
    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    langs = [row["key"] for row in manifest["base_panel"]]
    if [tuple(p) for p in design["ll_pairs"]] != [
        ("qwen2-7b", "olmo-7b-0724"),
        ("qwen2.5-7b", "olmo2-1124-7b"),
        ("qwen3-8b-base", "olmo-3-1025-7b"),
    ]:
        raise RuntimeError("L-L pairs do not match the release-matched list")
    specs = {row["key"]: spec_from_manifest(row) for row in manifest["base_panel"]}
    for key in VIS:
        specs[key] = MODELS[key]
    _, split_ids = load_splits(paths, repo)
    use = {name: list(split_ids[name]) for name in ("train", "val", "test")}
    assert_split_image_disjoint(use)
    if len(use["train"]) != 2048 or len(use["val"]) != 1024 or len(use["test"]) != 1024:
        raise RuntimeError("unexpected split sizes")
    vl = [(lang, vis) for lang in langs for vis in VIS]
    ll = [tuple(p) for p in design["ll_pairs"]]
    jobs = build_jobs(design, vl, ll)
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for key in langs + list(VIS):
        prepared[key] = {}
        for split in splits:
            x, obj = load_final_prepared(paths, specs[key], split, use[split])
            if list(obj["sample_ids"]) != use[split]:
                raise ValueError(f"sample-id mismatch {key} {split}")
            prepared[key][split] = np.asarray(x.numpy(), dtype=np.float64)
    bases = _bases(prepared, langs + list(VIS), int(design["q"]), out, force=force)
    protocol = (repo / "configs" / "structured_metric_factors_protocol.md").read_text()
    config_hash = hashlib.sha256((json.dumps(design, sort_keys=True) + protocol).encode()).hexdigest()
    return {
        "paths": paths,
        "repo": repo,
        "design": design,
        "out": out,
        "prepared": prepared,
        "ids": use,
        "bases": bases,
        "langs": langs,
        "vl": vl,
        "ll": ll,
        "jobs": jobs,
        "specs": specs,
        "splits": splits,
        "config_hash": config_hash,
        "protocol": protocol,
        "protected": {
            "metric_stability_report": _mtime(paths.results / "metric_stability" / "report.md"),
            "resampled_report": _mtime(paths.results / "resampled_metric_training" / "report.md"),
        },
    }


def _bases(prepared, keys, q, out: Path, force: bool) -> dict[str, dict]:
    dest = out / "bases"
    dest.mkdir(parents=True, exist_ok=True)
    bases = {}
    for key in keys:
        path = dest / f"{key}.npz"
        meta_path = dest / f"{key}.json"
        if path.exists() and meta_path.exists() and not force:
            meta = json.loads(meta_path.read_text())
            blob = np.load(path)
            basis = {"mu": blob["mu"], "U": blob["U"], "q": int(meta["q"]), "reduced": bool(meta["reduced"]), "id": meta["id"], "sign_convention": meta["sign_convention"]}
        else:
            basis = fit_basis(prepared[key]["train"], q=q)
            np.savez(path, mu=basis["mu"], U=basis["U"])
            write_json(meta_path, {k: basis[k] for k in ("q", "numerical_rank", "variance_fraction", "reduced", "n", "d", "sign_convention", "id")})
        if int(basis["q"]) != q or basis["reduced"]:
            raise RuntimeError(f"{key} does not support q={q}")
        bases[key] = basis
    return bases


def _prepare(ctx: dict[str, Any]) -> None:
    out = ctx["out"]
    design = ctx["design"]
    inventory = {
        "config_hash": ctx["config_hash"],
        "n_jobs": len(ctx["jobs"]),
        "n_vl": len(ctx["vl"]),
        "n_ll": len(ctx["ll"]),
        "jobs": ctx["jobs"],
    }
    existing = out / "inventory.json"
    if existing.exists():
        prev = json.loads(existing.read_text())
        if prev.get("config_hash") != ctx["config_hash"] or int(prev.get("n_jobs", -1)) != len(ctx["jobs"]):
            raise RuntimeError("inventory hash does not match this configuration; refusing to mix runs")
    else:
        (out / "protocol.md").write_text(ctx["protocol"])
        write_json(out / "design.json", design)
        write_json(out / "inventory.json", inventory)
        write_json(
            out / "audit.json",
            {
                "config_hash": ctx["config_hash"],
                "train_ids_hash": ids_hash(ctx["ids"]["train"]),
                "val_ids_hash": ids_hash(ctx["ids"]["val"]),
                "test_ids_hash": ids_hash(ctx["ids"]["test"]),
                "basis_ids": {key: ctx["bases"][key]["id"] for key in ctx["bases"]},
                "protected_mtime": ctx["protected"],
                "feature_sha256": {
                    f"{key}:{split}": sha256_file(final_feature_path(ctx["paths"], ctx["specs"][key], "coco_val2017", split, ctx["ids"][split]))
                    for key in ctx["bases"]
                    for split in ctx["splits"]
                },
                "n_jobs": len(ctx["jobs"]),
            },
        )
    print(f"INVENTORY {len(ctx['jobs'])} jobs hash={ctx['config_hash'][:12]}", flush=True)


def _smoke(ctx: dict[str, Any]) -> None:
    if (ctx["out"] / "smoke.json").exists():
        print("smoke already recorded", flush=True)
        return
    lang, vis = "qwen2-7b", "dinov2-small"
    ids = ctx["ids"]["train"]
    checks = []
    for family, mode, lam_off in (("exp_s", "a_only", 0.0), ("oblique_diag", "a_only", 0.0), ("block4", "separate", 1.0)):
        raw = run_factor_stream(
            ctx["prepared"][lang]["train"],
            ctx["prepared"][vis]["train"],
            ids,
            ctx["bases"][lang],
            ctx["bases"][vis],
            family=family,
            mode=mode,
            rho=0.1,
            lambda_basis=0.1 if family != "exp_s" else 0.0,
            lambda_off=lam_off,
            init_seed=0,
            sampling_seed=0,
            n_updates=2,
            val_features_a=ctx["prepared"][lang]["val"],
            val_features_b=ctx["prepared"][vis]["val"],
            record_indices=True,
        )
        if not raw["ok"]:
            raise RuntimeError(f"smoke {family} {mode} stopped: {raw['status']}")
        if raw["status"]["max_dist_residual"] > 1e-5 or raw["status"]["max_eig_residual"] > 1e-5:
            raise RuntimeError(f"smoke constraint residual {family}")
        if len(raw["subset_indices"][0]) != 512:
            raise RuntimeError("smoke quarter was not 512")
        if family != "exp_s":
            part = raw["final_parts"][0]
            if not np.allclose(induced_b(part["A"], part["D"]), part["B"], atol=1e-7):
                raise RuntimeError("smoke factors do not reproduce B")
        checks.append({"family": family, "mode": mode, "val0": raw["curves"][0]["full_val_excess"], "updates": raw["updates_done"]})
    write_json(ctx["out"] / "smoke.json", {"ok": True, "checks": checks})
    print("SMOKE_OK", flush=True)


def _init_fit(work: str, repo: str) -> None:
    """Spawned workers must rebuild arrays. Forking after torch starts deadlocks."""
    global CTX
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    repo_path = Path(repo)
    paths = Paths(work=Path(work), repo=repo_path)
    design = json.loads((repo_path / "configs" / "structured_metric_factors.json").read_text())
    out = paths.results / "structured_metric_factors"
    CTX = _context(paths, repo_path, design, out, splits=("train", "val"), force=False)
    CTX["force"] = False
    print(f"worker ready {os.getpid()}", flush=True)


def _init_eval(work: str, repo: str, force: bool) -> None:
    global CTX
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    repo_path = Path(repo)
    paths = Paths(work=Path(work), repo=repo_path)
    design = json.loads((repo_path / "configs" / "structured_metric_factors.json").read_text())
    out = paths.results / "structured_metric_factors"
    CTX = _context(paths, repo_path, design, out, splits=("train", "val", "test"), force=False)
    CTX["force_eval"] = bool(force)


def _fit(ctx: dict[str, Any], force: bool, workers: int) -> None:
    global CTX
    ctx = dict(ctx)
    ctx["force"] = force
    CTX = ctx
    jobs = list(ctx["jobs"])
    pending = []
    for job in jobs:
        dest = ctx["out"] / "runs" / f"{job_name(job)}.json"
        if dest.exists() and not force:
            prev = json.loads(dest.read_text())
            if prev.get("ok") and prev.get("config_hash") == ctx["config_hash"]:
                continue
        pending.append(job)
    print(f"fit pending {len(pending)} / {len(jobs)} workers={workers}", flush=True)
    if not pending:
        return
    t0 = time.perf_counter()
    done = 0
    errors = []
    if workers == 1:
        iterator = (_fit_worker(job) for job in pending)
        pool = None
    else:
        pool = get_context("spawn").Pool(workers, initializer=_init_fit, initargs=(str(ctx["paths"].work), str(ctx["repo"])))
        iterator = pool.imap_unordered(_fit_worker, pending, chunksize=1)
    try:
        for name, status, seconds in iterator:
            done += 1
            elapsed = time.perf_counter() - t0
            rate = elapsed / done
            eta = rate * (len(pending) - done)
            print(f"{status} {done}/{len(pending)} {name} step_s={seconds:.1f} eta_h={eta/3600:.2f}", flush=True)
            if str(status).startswith("error"):
                errors.append(f"{name}: {status}")
    finally:
        if workers != 1:
            pool.close()
            pool.join()
    if errors:
        (ctx["out"] / "fit_errors.txt").write_text("\n".join(errors) + "\n")
        raise RuntimeError(f"{len(errors)} fits failed; see fit_errors.txt")


def _fit_worker(job: dict[str, Any]) -> tuple[str, str, float]:
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        return _fit_one(CTX, job)
    except Exception:
        return job_name(job), "error: " + traceback.format_exc().splitlines()[-1], 0.0


def _correspondence_perm(n: int, perm_seed: int, split: str) -> np.ndarray:
    """Separate streams for train and validation. The same seed does not share one permutation of the wrong length."""
    salt = 0 if split == "train" else 17
    return np.random.Generator(np.random.PCG64(int(perm_seed) + salt)).permutation(int(n))


def _fit_one(ctx: dict[str, Any], job: dict[str, Any]) -> tuple[str, str, float]:
    name = job_name(job)
    runs = ctx["out"] / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    dest = runs / f"{name}.json"
    if dest.exists() and not ctx.get("force"):
        prev = json.loads(dest.read_text())
        if prev.get("ok") and prev.get("config_hash") == ctx["config_hash"]:
            return name, "skip", 0.0
    stem = runs / name
    resume = _load_state(stem)
    side_a, side_b = job["side_a"], job["side_b"]
    xa = ctx["prepared"][side_a]["train"]
    xb = np.array(ctx["prepared"][side_b]["train"], copy=True)
    va = ctx["prepared"][side_a]["val"]
    vb = np.array(ctx["prepared"][side_b]["val"], copy=True)
    if job["correspondence"] == "shuffled":
        xb = xb[_correspondence_perm(xb.shape[0], int(job["perm_seed"]), "train")]
        vb = vb[_correspondence_perm(vb.shape[0], int(job["perm_seed"]), "val")]

    def on_checkpoint(state: dict[str, Any]) -> None:
        _save_state(stem, state)

    t0 = time.perf_counter()
    raw = run_factor_stream(
        xa,
        xb,
        ctx["ids"]["train"],
        ctx["bases"][side_a],
        ctx["bases"][side_b],
        family=job["family"],
        mode=job["mode"],
        rho=float(job["rho"]),
        lambda_basis=float(job["lambda_basis"]),
        lambda_off=float(job["lambda_off"]),
        init_seed=int(job["seed"]),
        sampling_seed=int(job["seed"]),
        n_updates=int(ctx["design"]["n_updates"]),
        lr=float(ctx["design"]["optim"]["lr"]),
        grad_clip=float(ctx["design"]["optim"]["grad_clip"]),
        pert_scale=float(ctx["design"]["pert_scale"]),
        quarter_fraction=float(ctx["design"]["quarter_fraction"]),
        val_features_a=va,
        val_features_b=vb,
        resume=resume,
        on_checkpoint=on_checkpoint,
    )
    payload = {
        "name": name,
        "config_hash": ctx["config_hash"],
        "job": job,
        "ok": bool(raw["ok"]),
        "status": raw["status"],
        "updates_done": raw["updates_done"],
        "selected": raw["selected"],
        "optim_seconds": raw["optim_seconds"],
        "validation_seconds": raw["validation_seconds"],
        "train_eval_seconds": raw["train_eval_seconds"],
        "examples_processed": raw["examples_processed"],
        "n_val_checks": raw["n_val_checks"],
        "objective_note": raw["objective_note"],
        "curves": raw["curves"],
        "perm_seed": job.get("perm_seed"),
    }
    write_json(dest, jsonable(payload))
    _save_factors(runs / f"{name}.factors.npz", raw)
    if not raw["ok"]:
        raise RuntimeError(raw["status"])
    return name, "ok", time.perf_counter() - t0


def _save_factors(path: Path, raw: dict[str, Any]) -> None:
    blob = {}
    for label in ("selected_parts", "final_parts"):
        for i, part in enumerate(raw.get(label) or []):
            for key, val in part.items():
                if isinstance(val, np.ndarray):
                    blob[f"{label}_{i}_{key}"] = np.asarray(val)
    np.savez(path, **blob)


def _save_state(stem: Path, state: dict[str, Any]) -> None:
    meta = {
        "updates_done": state["updates_done"],
        "rng": state["rng"],
        "curves": state["curves"],
        "selected": state["selected"],
        "status": state["status"],
        "optim_seconds": state["optim_seconds"],
        "validation_seconds": state["validation_seconds"],
        "train_eval_seconds": state["train_eval_seconds"],
        "n_adam": len(state.get("adam") or []),
        "n_raw": len(state["raw"]) if "raw" in state else 0,
        "n_vech": len(state["vechs"]) if "vechs" in state else 0,
    }
    write_json(Path(str(stem) + ".state.json"), jsonable(meta))
    blob: dict[str, np.ndarray] = {}
    for i, blob_adam in enumerate(state.get("adam") or []):
        if not blob_adam:
            continue
        blob[f"adam_avg_{i}"] = np.asarray(blob_adam["exp_avg"])
        blob[f"adam_sq_{i}"] = np.asarray(blob_adam["exp_avg_sq"])
        blob[f"adam_step_{i}"] = np.asarray([blob_adam["step"]], dtype=np.float64)
    for i, vech in enumerate(state.get("vechs") or []):
        blob[f"vech_{i}"] = np.asarray(vech)
    for i, (a, h) in enumerate(state.get("raw") or []):
        blob[f"rawA_{i}"] = np.asarray(a)
        blob[f"rawH_{i}"] = np.asarray(h)
    for label in ("parts", "selected_parts"):
        for i, part in enumerate(state.get(label) or []):
            for key, val in part.items():
                if isinstance(val, np.ndarray):
                    blob[f"{label}_{i}_{key}"] = np.asarray(val, copy=True)
    np.savez(Path(str(stem) + ".state.npz"), **blob)


def _load_state(stem: Path) -> dict[str, Any] | None:
    meta_path = Path(str(stem) + ".state.json")
    npz_path = Path(str(stem) + ".state.npz")
    if not meta_path.exists() or not npz_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    blob = np.load(npz_path)
    state: dict[str, Any] = dict(meta)
    state["adam"] = []
    for i in range(int(meta["n_adam"])):
        key = f"adam_avg_{i}"
        if key not in blob:
            state["adam"].append(None)
            continue
        state["adam"].append({"exp_avg": blob[key], "exp_avg_sq": blob[f"adam_sq_{i}"], "step": float(blob[f"adam_step_{i}"][0])})
    if int(meta["n_vech"]):
        state["vechs"] = [blob[f"vech_{i}"] for i in range(int(meta["n_vech"]))]
    if int(meta["n_raw"]):
        state["raw"] = [(blob[f"rawA_{i}"], blob[f"rawH_{i}"]) for i in range(int(meta["n_raw"]))]
    state["parts"] = _parts_from_blob(blob, "parts")
    state["selected_parts"] = _parts_from_blob(blob, "selected_parts")
    return state


def _parts_from_blob(blob, label: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    i = 0
    while f"{label}_{i}_B" in blob or f"{label}_{i}_S" in blob:
        part = {}
        for key in ("A", "H", "D", "B", "S", "singular_values", "eig"):
            name = f"{label}_{i}_{key}"
            if name in blob:
                part[key] = blob[name]
        if "B" in part and "distortion" not in part:
            dist, ev = distortion_of_b(part["B"])
            part["distortion"] = dist
            part["eig"] = ev
        if "A" in part:
            part["r_basis"] = r_basis(part["A"])
            if "singular_values" not in part:
                part["singular_values"] = singular_values(part["A"])
            part["cond"] = condition_number(part["singular_values"])
        if "H" in part:
            part["r_off"] = r_off(part["H"])
        parts.append(part)
        i += 1
    return parts


def _eval(ctx: dict[str, Any], force: bool, workers: int) -> None:
    global CTX
    CTX = ctx
    CTX["force_eval"] = force
    jobs = list(ctx["jobs"])
    if workers == 1:
        iterator = (_eval_worker(job) for job in jobs)
        pool = None
    else:
        pool = get_context("spawn").Pool(workers, initializer=_init_eval, initargs=(str(ctx["paths"].work), str(ctx["repo"]), force))
        iterator = pool.imap_unordered(_eval_worker, jobs, chunksize=1)
    done = 0
    try:
        for name in iterator:
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"eval {done}/{len(jobs)} {name}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    _write_identity(ctx)
    _join(ctx)


def _eval_worker(job: dict[str, Any]) -> str:
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ctx = CTX
    name = job_name(job)
    dest = ctx["out"] / "evals" / f"{name}.json"
    if dest.exists() and not ctx.get("force_eval"):
        return name
    npz = np.load(ctx["out"] / "runs" / f"{name}.factors.npz")
    record = _score_job(ctx, job, npz)
    record["test_role"] = "exploratory_heldout"
    record["name"] = name
    write_json(dest, jsonable(record))
    return name


def _score_job(ctx, job, npz) -> dict[str, Any]:
    q = int(ctx["design"]["q"])
    out = {"job": job, "config_hash": ctx["config_hash"]}
    for label, with_mnn_splits in (("selected", ("train", "val", "test")), ("final", ("test",))):
        parts = _parts_from_blob(npz, f"{label}_parts")
        if not parts:
            out[label] = None
            continue
        b_a = parts[0]["B"]
        if job["mode"] == "a_only":
            b_b = np.eye(q)
        elif job["mode"] == "shared_pc":
            b_b = parts[0]["B"]
        else:
            b_b = parts[1]["B"]
        scores = {}
        for split in with_mnn_splits:
            xa = ctx["prepared"][job["side_a"]][split]
            xb = np.array(ctx["prepared"][job["side_b"]][split], copy=True)
            if job["correspondence"] == "shuffled" and split in ("train", "val"):
                xb = xb[_correspondence_perm(xb.shape[0], int(job["perm_seed"]), split)]
            key = "exploratory" if split == "test" else split
            scores[key] = score_metrics(
                xa,
                xb,
                ctx["bases"][job["side_a"]],
                ctx["bases"][job["side_b"]],
                b_a,
                b_b,
                knn_k=int(ctx["design"]["knn_k"]),
            )
            if split == "test":
                scores[key]["correspondence"] = "true"
        check = parts if job["mode"] == "separate" else [parts[0]]
        dist = total_distortion(check, job["mode"])
        sides = [_side_public(part, job["family"]) for part in parts]
        c_total = complexity_total(q, job["family"], job["mode"])
        val_excess = float(scores["val"]["excess"]) if "val" in scores else None
        out[label] = {
            "scores": scores,
            "d_total": dist,
            "sides": sides,
            "c_total": c_total,
            "j_select": None if val_excess is None else penalised_selection(val_excess, c_total, q, float(ctx["design"]["complexity_coef"])),
            "r_basis_total": _reg_total(sides, job["mode"], "r_basis"),
            "r_off_total": _reg_total(sides, job["mode"], "r_off"),
        }
    return out


def _side_public(part: dict[str, Any], family: str) -> dict[str, Any]:
    ev = np.asarray(part["eig"], dtype=np.float64)
    log_abs = np.abs(np.log(np.clip(ev, 1e-30, None)))
    rec = {
        "distortion": float(part["distortion"]),
        "eig_min": float(ev.min()),
        "eig_max": float(ev.max()),
        "frac_at_bounds": float(np.mean(log_abs >= LOG4 - BOUND_ATOL)),
        "log_b": sorted_log_spectrum(part["B"]).tolist(),
        "r_basis": float(part.get("r_basis", 0.0)),
        "r_off": float(part.get("r_off", 0.0)),
        "cond": float(part.get("cond", 1.0)),
        "sv_min": float(np.min(part["singular_values"])) if "singular_values" in part else 1.0,
        "sv_max": float(np.max(part["singular_values"])) if "singular_values" in part else 1.0,
        "off_norm": float(off_norm(part["H"])) if "H" in part else 0.0,
    }
    if "D" in part and learns_basis(family) or family == "fixed_diag" and "D" in part:
        rec["log_d"] = sorted_log_spectrum(part["D"]).tolist()
    return rec


def _reg_total(sides: list[dict[str, Any]], mode: str, key: str) -> float:
    if mode == "shared_pc":
        return float(2.0 * sides[0][key])
    if mode == "a_only":
        return float(sides[0][key])
    return float(sum(side[key] for side in sides))


def _write_identity(ctx) -> None:
    dest_dir = ctx["out"] / "evals"
    dest_dir.mkdir(parents=True, exist_ok=True)
    q = int(ctx["design"]["q"])
    eye = np.eye(q)
    for kind, pairs in (("vl", ctx["vl"]), ("ll", ctx["ll"])):
        for side_a, side_b in pairs:
            name = f"identity__{side_a}__{side_b}"
            path = dest_dir / f"{name}.json"
            if path.exists() and not ctx.get("force_eval"):
                continue
            scores = {}
            for split in ("train", "val", "test"):
                key = "exploratory" if split == "test" else split
                scores[key] = score_metrics(
                    ctx["prepared"][side_a][split],
                    ctx["prepared"][side_b][split],
                    ctx["bases"][side_a],
                    ctx["bases"][side_b],
                    eye,
                    eye,
                    knn_k=int(ctx["design"]["knn_k"]),
                )
            write_json(
                path,
                jsonable(
                    {
                        "name": name,
                        "test_role": "exploratory_heldout",
                        "job": {"kind": kind, "side_a": side_a, "side_b": side_b, "mode": "identity", "family": "identity", "profile": "na", "rho": 0.0, "seed": None, "correspondence": "true"},
                        "selected": {"scores": scores, "d_total": 0.0, "sides": [], "c_total": 0, "j_select": scores["val"]["excess"], "r_basis_total": 0.0, "r_off_total": 0.0},
                    }
                ),
            )


def _join(ctx) -> None:
    rows = []
    eval_dir = ctx["out"] / "evals"
    for path in sorted(eval_dir.glob("*.json")):
        ev = json.loads(path.read_text())
        if not ev.get("selected"):
            continue
        job = ev["job"]
        sel = ev["selected"]
        exploratory = sel["scores"]["exploratory"]
        row = {
            "name": ev["name"],
            "kind": job["kind"],
            "side_a": job["side_a"],
            "side_b": job["side_b"],
            "mode": job["mode"],
            "family": job["family"],
            "profile": job["profile"],
            "rho": job["rho"],
            "seed": job["seed"],
            "correspondence": job["correspondence"],
            "exploratory_a": exploratory["a"],
            "exploratory_excess": exploratory["excess"],
            "exploratory_mnn": exploratory["mnn_k10"],
            "val_excess": sel["scores"]["val"]["excess"] if "val" in sel["scores"] else None,
            "d_total": sel["d_total"],
            "r_basis_total": sel["r_basis_total"],
            "r_off_total": sel["r_off_total"],
            "c_total": sel["c_total"],
            "j_select": sel["j_select"],
            "sides": sel["sides"],
        }
        if ev.get("final") and ev["final"].get("scores", {}).get("exploratory"):
            row["final_exploratory_excess"] = ev["final"]["scores"]["exploratory"]["excess"]
        rows.append(row)
    write_json(ctx["out"] / "joined.json", rows)
    print(f"joined {len(rows)}", flush=True)


def _rows(out: Path) -> list[dict[str, Any]]:
    return json.loads((out / "joined.json").read_text())


def _spec_label(row: dict[str, Any]) -> str:
    if row["family"] in ("block2", "block4"):
        return f"{row['family']}_{row['profile']}"
    return row["family"]


def _figures(out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in _rows(out) if r["correspondence"] == "true"]
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    for kind in ("vl", "ll"):
        for spec in _specs(rows):
            part = [r for r in rows if r["kind"] == kind and _spec_label(r) == spec or (r["kind"] == kind and r["family"] == "identity")]
            # identity is drawn inside _condition_figure from the full kind slice
            path = fig_dir / f"conditions_{kind}_{spec}.png"
            _condition_figure(plt, [r for r in rows if r["kind"] == kind], spec, path)
            captions.append((path.name, "Connected lines are pairs. Points are seeds. Y is exploratory held-out raw CKA a, excess a-b, or mutual kNN at k=10. X is the fitting condition. Columns are distortion budgets. Identity is M=I."))
            del part
    _distortion_figure(plt, rows, fig_dir / "alignment_vs_distortion.png")
    captions.append(("alignment_vs_distortion.png", "Each point is one seed of one pair. Horizontal position is realised total distortion d(B_A)+d(B_B), with shared-PC counted twice and one-sided using only side A. Identity sits at distortion 0."))
    _spectra_figures(plt, rows, fig_dir, captions)
    _basis_figure(plt, rows, fig_dir / "basis_diagnostics.png")
    captions.append(("basis_diagnostics.png", "R_basis = ||A^T A - I||_F^2 / 32 on the A that enters B. Condition number is the ratio of singular values of that A, clipped to [0.5, 2]. Gain is exploratory excess minus identity on the same pair."))
    _block_figure(plt, rows, fig_dir / "block_diagnostics.png")
    captions.append(("block_diagnostics.png", "C_total counts untied off-diagonal weighting parameters. The vertical axis is exploratory excess minus identity. Off-diagonal norm is ||H - diag(H)||_F."))
    shuf = [r for r in _rows(out) if r["correspondence"] == "shuffled"]
    _shuffle_figure(plt, rows, shuf, fig_dir / "shuffle_control.png")
    captions.append(("shuffle_control.png", "Shuffled fits permute the partner rows on train and validation. Exploratory scores use true correspondence. Spectrum distance compares sorted log-eigenvalues of B."))
    (fig_dir / "captions.md").write_text("\n".join(f"- `{name}`: {text}" for name, text in captions) + "\n")


def _specs(rows) -> list[str]:
    labels = []
    for row in rows:
        if row["family"] == "identity":
            continue
        label = _spec_label(row)
        if label not in labels:
            labels.append(label)
    return labels


def _condition_figure(plt, rows, spec, path: Path) -> None:
    learned = [r for r in rows if _spec_label(r) == spec]
    if not learned:
        return
    identity = {(r["side_a"], r["side_b"]): r for r in rows if r["family"] == "identity"}
    metrics = (("exploratory_a", "Exploratory raw CKA a"), ("exploratory_excess", "Exploratory excess a-b"), ("exploratory_mnn", "Exploratory mutual kNN, k=10"))
    rhos = sorted({float(r["rho"]) for r in learned})
    fig, axes = plt.subplots(len(metrics), len(rhos), figsize=(4.2 * len(rhos), 9), sharex=True, squeeze=False)
    pairs = sorted({(r["side_a"], r["side_b"]) for r in learned})
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(pairs), 1)))
    for col, rho in enumerate(rhos):
        for row_i, (field, ylabel) in enumerate(metrics):
            ax = axes[row_i][col]
            for color, pair in zip(colors, pairs):
                ys = []
                for mode_i, mode in enumerate(("identity", "a_only", "shared_pc", "separate")):
                    if mode == "identity":
                        ys.append(float(identity[pair][field]))
                        ax.scatter([mode_i], [ys[-1]], color=color, s=12, zorder=3)
                        continue
                    vals = [float(r[field]) for r in learned if (r["side_a"], r["side_b"]) == pair and r["mode"] == mode and abs(float(r["rho"]) - rho) < 1e-9]
                    ax.scatter([mode_i] * len(vals), vals, color=color, s=10, alpha=0.35)
                    ys.append(float(np.median(vals)) if vals else np.nan)
                ax.plot(range(4), ys, color=color, lw=1.0, label=f"{pair[0]} / {pair[1]}")
            ax.set_xticks(range(4), [MODE_LABEL[m] for m in MODES_PLOT], rotation=15)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_i == 0:
                ax.set_title(f"rho={rho:g}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.suptitle(spec)
    _save(fig, path)


def _distortion_figure(plt, rows, path: Path) -> None:
    specs = _specs(rows)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    axes = axes.ravel()
    colors = {"a_only": "C0", "shared_pc": "C1", "separate": "C2", "identity": "k"}
    for ax, spec in zip(axes, specs):
        part = [r for r in rows if _spec_label(r) == spec]
        for row in part:
            ax.scatter(row["d_total"], row["exploratory_excess"], s=12, c=colors[row["mode"]], alpha=0.7)
        ident = [r for r in rows if r["family"] == "identity"]
        ax.scatter([0.0] * len(ident), [r["exploratory_excess"] for r in ident], s=14, c="k", marker="x")
        ax.set_title(spec, fontsize=9)
        ax.set_xlabel("total distortion")
        ax.set_ylabel("exploratory excess")
    for ax in axes[len(specs) :]:
        ax.axis("off")
    _save(fig, path)


def _spectra_figures(plt, rows, fig_dir: Path, captions: list) -> None:
    focus = [r for r in rows if r["family"] in ("exp_s", "oblique_diag", "fixed_diag", "block2", "block4") and r["mode"] in ("a_only", "separate") and r["kind"] == "vl"]
    seen = set()
    for row in focus:
        key = (_spec_label(row), row["mode"], float(row["rho"]))
        if key in seen:
            continue
        seen.add(key)
        part = [r for r in focus if _spec_label(r) == key[0] and r["mode"] == key[1] and abs(float(r["rho"]) - key[2]) < 1e-9]
        rho_tag = str(key[2]).replace(".", "p")
        name = f"spectra_{key[0]}_{key[1]}_rho{rho_tag}.png"
        _spectrum_panel(plt, part, fig_dir / name, side=0)
        captions.append((name, "Side A of each fit. Each curve is one seed. Colour is the vision partner. Top row: sorted log-eigenvalues of induced B. Bottom row: those curves after subtracting their mean. Columns are language models. These are the 32 eigenvalues of B, not the ambient identity eigenvalues."))
        if key[1] == "separate":
            side_b_name = f"spectra_{key[0]}_{key[1]}_sideB_rho{rho_tag}.png"
            _spectrum_panel(plt, part, fig_dir / side_b_name, side=1)
            captions.append((side_b_name, "Side B of separate fits, same layout. The curve is that partner's own induced B, not a copy of side A."))
        if any("log_d" in (r["sides"][0] if r["sides"] else {}) for r in part):
            heat = f"heatmap_B_{name[8:]}"
            _heatmap(plt, part, fig_dir / f"heatmap_B_{key[0]}_{key[1]}_rho{str(key[2]).replace('.', 'p')}.png", which="log_b")
            _heatmap(plt, part, fig_dir / f"heatmap_D_{key[0]}_{key[1]}_rho{str(key[2]).replace('.', 'p')}.png", which="log_d")
            captions.append((f"heatmap_B_{key[0]}_{key[1]}_rho{str(key[2]).replace('.', 'p')}.png", "Cell is the seed-median permutation-invariant log-spectrum distance of induced B between two vision partners of the same language model."))
            captions.append((f"heatmap_D_{key[0]}_{key[1]}_rho{str(key[2]).replace('.', 'p')}.png", "Same layout for sorted log-eigenvalues of D. These are factor spectra, not eigenvalues of B."))
            del heat


def _spectrum_panel(plt, rows, path: Path, side: int) -> None:
    models = []
    for row in rows:
        if row["side_a"] not in models:
            models.append(row["side_a"])
    partners = []
    for row in rows:
        if row["side_b"] not in partners:
            partners.append(row["side_b"])
    fig, axes = plt.subplots(2, len(models), figsize=(2.6 * len(models), 5.4), sharex=True, sharey="row", squeeze=False)
    colors = {partner: f"C{i}" for i, partner in enumerate(partners)}
    for col, model in enumerate(models):
        for row in rows:
            if row["side_a"] != model or len(row["sides"]) <= side:
                continue
            log_b = np.asarray(row["sides"][side]["log_b"], dtype=np.float64)
            axes[0][col].plot(log_b, color=colors[row["side_b"]], lw=0.8, alpha=0.8)
            axes[1][col].plot(log_b - log_b.mean(), color=colors[row["side_b"]], lw=0.8, alpha=0.8)
        axes[0][col].set_title(model, fontsize=8)
        axes[1][col].set_xlabel("sorted index")
    axes[0][0].set_ylabel("log eigenvalue of B")
    axes[1][0].set_ylabel("mean-subtracted")
    _save(fig, path)


def _heatmap(plt, rows, path: Path, which: str) -> None:
    models = []
    partners = []
    for row in rows:
        if row["side_a"] not in models:
            models.append(row["side_a"])
        if row["side_b"] not in partners:
            partners.append(row["side_b"])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        mat = np.full((len(partners), len(partners)), np.nan)
        for i, left in enumerate(partners):
            for j, right in enumerate(partners):
                dists = []
                seeds = sorted({r["seed"] for r in rows if r["side_a"] == model})
                for seed in seeds:
                    a = [r for r in rows if r["side_a"] == model and r["side_b"] == left and r["seed"] == seed]
                    b = [r for r in rows if r["side_a"] == model and r["side_b"] == right and r["seed"] == seed]
                    if not a or not b or which not in a[0]["sides"][0] or which not in b[0]["sides"][0]:
                        continue
                    dists.append(spectrum_distance(a[0]["sides"][0][which], b[0]["sides"][0][which]))
                if dists:
                    mat[i, j] = float(np.median(dists))
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(partners)), partners, rotation=90, fontsize=6)
        ax.set_yticks(range(len(partners)), partners, fontsize=6)
        ax.set_title(model, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    _save(fig, path)


def _basis_figure(plt, rows, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ident = {(r["side_a"], r["side_b"]): r["exploratory_excess"] for r in rows if r["family"] == "identity"}
    for row in rows:
        if not row["sides"] or row["family"] in ("identity", "exp_s", "fixed_diag"):
            continue
        gain = float(row["exploratory_excess"]) - float(ident[(row["side_a"], row["side_b"])])
        ax.scatter(row["r_basis_total"], gain, c=row["sides"][0]["cond"], cmap="magma", s=16, vmin=1.0, vmax=4.0)
    ax.set_xlabel("total R_basis")
    ax.set_ylabel("exploratory excess gain over identity")
    ax.set_title("Basis obliqueness against held-out gain")
    _save(fig, path)


def _block_figure(plt, rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ident = {(r["side_a"], r["side_b"]): r["exploratory_excess"] for r in rows if r["family"] == "identity"}
    for row in rows:
        if row["family"] not in ("fixed_diag", "oblique_diag", "block2", "block4"):
            continue
        gain = float(row["exploratory_excess"]) - float(ident[(row["side_a"], row["side_b"])])
        axes[0].scatter(row["c_total"], gain, s=12, alpha=0.7, label=row["family"])
        off = row["sides"][0]["off_norm"] if row["sides"] else 0.0
        axes[1].scatter(off, gain, s=12, alpha=0.7)
    axes[0].set_xlabel("C_total")
    axes[0].set_ylabel("exploratory excess gain")
    axes[1].set_xlabel("||H - diag(H)||_F")
    axes[1].set_ylabel("exploratory excess gain")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0].legend(uniq.values(), uniq.keys(), fontsize=7)
    _save(fig, path)


def _shuffle_figure(plt, rows, shuf, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for row in shuf:
        match = [
            r
            for r in rows
            if r["side_a"] == row["side_a"]
            and r["side_b"] == row["side_b"]
            and r["family"] == row["family"]
            and r["profile"] == row["profile"]
            and r["mode"] == row["mode"]
            and r["seed"] == row["seed"]
            and abs(float(r["rho"]) - float(row["rho"])) < 1e-9
        ]
        if not match or not row["sides"] or not match[0]["sides"]:
            continue
        axes[0].scatter(match[0]["exploratory_excess"], row["exploratory_excess"], s=20)
        dist = spectrum_distance(match[0]["sides"][0]["log_b"], row["sides"][0]["log_b"])
        axes[1].scatter(match[0]["d_total"], dist, s=20)
    axes[0].set_xlabel("true-correspondence exploratory excess")
    axes[0].set_ylabel("shuffled-fit exploratory excess")
    axes[1].set_xlabel("true-fit total distortion")
    axes[1].set_ylabel("spectrum distance, shuffled vs true B")
    _save(fig, path)


def _write_report(out: Path, design: dict) -> None:
    rows = _rows(out)
    true = [r for r in rows if r["correspondence"] == "true" and r["family"] != "identity"]
    ident = {(r["kind"], r["side_a"], r["side_b"]): r for r in rows if r["family"] == "identity"}
    lines = [
        "# Structured metric factors",
        "",
        "Exploratory held-out scores use the manifest test split. Checkpoints were chosen by full-validation excess, then lower total distortion, then the earlier step. The test split did not choose regularisation, block size, or checkpoints.",
        "",
        f"Configuration hash `{json.loads((out / 'inventory.json').read_text())['config_hash']}`. Planned fits: {json.loads((out / 'inventory.json').read_text())['n_jobs']}. Completed true-correspondence learned rows in this table: {len(true)}.",
        "",
        "Shared-PC ties the reduced coordinates of two independently estimated PCA bases. It is not a shared ambient metric. Column normalisation does not uniquely identify A and D. Eigenvalues of D are not eigenvalues of B when A is nonorthogonal.",
        "",
        "## Answers",
        "",
    ]
    lines.extend(_answer_condition(true, ident))
    lines.extend(_answer_factor_vs_exp(true, ident))
    lines.extend(_answer_obliqueness(true))
    lines.extend(_answer_blocks(true, ident, float(design["complexity_coef"])))
    lines.extend(_answer_spectra(true, [r for r in rows if r["correspondence"] == "shuffled"]))
    lines.extend(_answer_factors_vs_b(true))
    lines.extend(_answer_mnn(true, ident))
    lines.extend(_answer_sensitivity(true))
    lines.extend(_constraint_note(out))
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src",
            "/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_structured_metric_factors.py \\",
            "  --work /mnt/sdb1/prh-replication-work --stage all --workers 8",
            "```",
            "",
            "Quarter updates: 120. Adam learning rate 0.03, gradient clip 5. Singular values of column-normalised A are clipped to [0.5, 2]. Induced eigenvalues of B lie in [1/4, 4].",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines))
    write_json(out / "summary.json", {"n_true_learned": len(true), "n_rows": len(rows)})


def _constraint_note(out: Path) -> list[str]:
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"rej": [], "ret": [], "bt": [], "eig": [], "dist": []})
    runs = out / "runs"
    if not runs.exists():
        return ["## Constraint diagnostics", "", "No fit records.", ""]
    for path in sorted(runs.glob("*.json")):
        if ".state." in path.name:
            continue
        payload = json.loads(path.read_text())
        status = payload.get("status") or {}
        job = payload.get("job") or {}
        if "family" not in job:
            continue
        bucket = buckets[(str(job["family"]), str(job["profile"]))]
        bucket["rej"].append(float(status.get("full_rejects", 0)))
        bucket["ret"].append(float(status.get("retractions", 0)))
        bucket["bt"].append(float(status.get("backtracks", 0)))
        bucket["eig"].append(float(status.get("max_eig_residual", 0.0)))
        bucket["dist"].append(float(status.get("max_dist_residual", 0.0)))
    lines = [
        "## Constraint diagnostics",
        "",
        "A retraction spectrally projects the induced B with the same clip and equal-total scale as exp(S), then rebuilds A and D so they produce that B. A full reject leaves the previous factors in place. Residuals are measured on the accepted factors.",
        "",
    ]
    for key in sorted(buckets):
        bucket = buckets[key]
        lines.append(
            f"- {key[0]}/{key[1]}: median full rejects {float(np.median(bucket['rej'])):.0f}, "
            f"median retractions {float(np.median(bucket['ret'])):.0f}, "
            f"median backtracks {float(np.median(bucket['bt'])):.0f}, "
            f"max eigenvalue residual {float(np.max(bucket['eig'])):.2e}, "
            f"max distortion residual {float(np.max(bucket['dist'])):.2e}."
        )
    lines.append("")
    return lines


def _pair_medians(rows, kind, family, profile, mode, rho, field):
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] == kind and row["family"] == family and row["profile"] == profile and row["mode"] == mode and abs(float(row["rho"]) - rho) < 1e-9:
            groups[(row["side_a"], row["side_b"])].append(float(row[field]))
    return {pair: float(np.median(vals)) for pair, vals in groups.items()}


def _answer_condition(rows, ident) -> list[str]:
    lines = ["### 1. Fitting condition at equal total distortion", ""]
    for kind in ("vl", "ll"):
        lines.append(f"{kind.upper()} medians of per-pair seed-medians of exploratory excess:")
        for family, profile in _family_profiles(rows):
            for rho in (0.1, 0.4):
                scores = {}
                for mode in ("a_only", "shared_pc", "separate"):
                    meds = list(_pair_medians(rows, kind, family, profile, mode, rho, "exploratory_excess").values())
                    if meds:
                        scores[mode] = float(np.median(meds))
                if not scores:
                    continue
                winner = max(scores, key=scores.get)
                text = ", ".join(f"{MODE_LABEL[mode]} {scores[mode]:.4f}" for mode in scores)
                lines.append(f"- {family}/{profile}, rho={rho:g}: {text}. Highest median is {MODE_LABEL[winner]}.")
        id_vals = [float(row["exploratory_excess"]) for key, row in ident.items() if key[0] == kind]
        if id_vals:
            lines.append(f"- Identity median exploratory excess on {kind.upper()}: {float(np.median(id_vals)):.4f}.")
    lines.append("")
    return lines


def _family_profiles(rows) -> list[tuple[str, str]]:
    seen = []
    for row in rows:
        key = (row["family"], row["profile"])
        if key not in seen:
            seen.append(key)
    return seen


def _answer_factor_vs_exp(rows, ident) -> list[str]:
    lines = ["### 2. Factorizations relative to exp(S)", ""]
    for family, profile in _family_profiles(rows):
        if family == "exp_s":
            continue
        deltas = []
        for rho in (0.1, 0.4):
            for mode in ("a_only", "shared_pc", "separate"):
                exp_m = _pair_medians(rows, "vl", "exp_s", "na", mode, rho, "exploratory_excess")
                fac_m = _pair_medians(rows, "vl", family, profile, mode, rho, "exploratory_excess")
                for pair, val in fac_m.items():
                    if pair in exp_m:
                        deltas.append(val - exp_m[pair])
        if deltas:
            lines.append(f"- {family}/{profile} minus exp(S), V-L pair-level exploratory excess, median {float(np.median(deltas)):+.4f}, range {float(np.min(deltas)):+.4f} to {float(np.max(deltas)):+.4f}.")
    lines.append("A positive number means the factorization scored higher than exp(S) on that matched pair, mode, and budget. It is not a claim that the factors are simpler or more stable.")
    lines.append("")
    return lines


def _answer_obliqueness(rows) -> list[str]:
    lines = ["### 3. Was nonorthogonality used?", ""]
    part = [r for r in rows if r["family"] in ("oblique_diag", "block2", "block4") and r["sides"]]
    if not part:
        return lines + ["No oblique fits.", ""]
    rb = [float(r["r_basis_total"]) for r in part]
    cond = [float(r["sides"][0]["cond"]) for r in part]
    lines.append(f"Median total R_basis {float(np.median(rb)):.4g}, range {float(np.min(rb)):.4g} to {float(np.max(rb)):.4g}. Median condition number of side A {float(np.median(cond)):.3f}. Fraction with condition number above 1.05: {float(np.mean([c > 1.05 for c in cond])):.2f}.")
    if float(np.median(rb)) < 1e-3 and float(np.median(cond)) < 1.05:
        lines.append("The fitted bases stayed nearly orthogonal. This run does not show that obliqueness improved the metric.")
    else:
        lines.append("Some accepted bases leave the orthogonal matrices. R_basis and the condition number above are the quantities that measure that, not a claim that the extra obliqueness caused the alignment gain.")
    lines.append("")
    return lines


def _answer_blocks(rows, ident, coef) -> list[str]:
    lines = ["### 4. Block weights against diagonal weights", ""]
    for rho in (0.1, 0.4):
        for mode in ("a_only", "shared_pc", "separate"):
            base = _pair_medians(rows, "vl", "oblique_diag", "basis", mode, rho, "exploratory_excess")
            for family, profile in (("block2", "strong"), ("block2", "stronger"), ("block4", "strong"), ("block4", "stronger"), ("fixed_diag", "na")):
                other = _pair_medians(rows, "vl", family, profile, mode, rho, "exploratory_excess")
                deltas = [other[pair] - base[pair] for pair in other if pair in base]
                if deltas:
                    lines.append(f"- rho={rho:g} {MODE_LABEL[mode]}: {family}/{profile} minus oblique diagonal, median exploratory excess {float(np.median(deltas)):+.4f}.")
    lines.append(f"The secondary selection score subtracts {coef:g} times C_total / (q(q-1)/2) from validation excess. It penalises off-diagonal weighting parameters only. Fixed-family exploratory numbers above are the primary comparison.")
    lines.append("")
    return lines


def _answer_spectra(rows, shuf) -> list[str]:
    lines = ["### 5. Induced spectra across partners", ""]
    part = [r for r in rows if r["kind"] == "vl" and r["mode"] == "a_only" and r["family"] in ("oblique_diag", "exp_s", "block4") and r["sides"]]
    for family in ("exp_s", "oblique_diag", "block4"):
        for rho in (0.1, 0.4):
            group = [r for r in part if r["family"] == family and abs(float(r["rho"]) - rho) < 1e-9 and (family != "block4" or r["profile"] == "strong")]
            if not group:
                continue
            rep = _replicate_distances(group)
            across = _partner_distances(group)
            centred_across = _partner_distances(group, centred=True)
            lines.append(
                f"- {family} one-sided rho={rho:g}: median replicate log-spectrum distance {rep:.4f}; median partner-to-partner distance {across:.4f}; mean-subtracted partner distance {centred_across:.4f}."
            )
    if shuf:
        dists = []
        gains = []
        for row in shuf:
            match = [
                r
                for r in rows
                if r["side_a"] == row["side_a"] and r["side_b"] == row["side_b"] and r["family"] == row["family"] and r["seed"] == row["seed"] and r["mode"] == row["mode"] and abs(float(r["rho"]) - float(row["rho"])) < 1e-9 and r["sides"] and row["sides"]
            ]
            if not match:
                continue
            dists.append(spectrum_distance(match[0]["sides"][0]["log_b"], row["sides"][0]["log_b"]))
            key = (row["kind"], row["side_a"], row["side_b"])
            gains.append(float(row["exploratory_excess"]))
        if dists:
            lines.append(f"Shuffled one-sided fits versus the matched true fits: median spectrum distance {float(np.median(dists)):.4f}. Median shuffled exploratory excess {float(np.median(gains)):.4f}. This is a descriptive control on three predetermined pairs, not a significance test.")
    lines.append("Similar spectra can be produced by the shared eigenvalue bounds and the distortion budget. Agreement of sorted spectra is not agreement of semantic directions.")
    lines.append("")
    return lines


def _replicate_distances(group) -> float:
    dists = []
    buckets = defaultdict(list)
    for row in group:
        buckets[(row["side_a"], row["side_b"])].append(row)
    for items in buckets.values():
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                dists.append(spectrum_distance(left["sides"][0]["log_b"], right["sides"][0]["log_b"]))
    return float(np.median(dists)) if dists else float("nan")


def _partner_distances(group, centred: bool = False) -> float:
    dists = []
    buckets = defaultdict(list)
    for row in group:
        buckets[(row["side_a"], row["seed"])].append(row)
    for items in buckets.values():
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                a = np.asarray(left["sides"][0]["log_b"], dtype=np.float64)
                b = np.asarray(right["sides"][0]["log_b"], dtype=np.float64)
                if centred:
                    a = a - a.mean()
                    b = b - b.mean()
                dists.append(spectrum_distance(a, b))
    return float(np.median(dists)) if dists else float("nan")


def _answer_factors_vs_b(rows) -> list[str]:
    lines = ["### 6. Factor spectra versus induced B", ""]
    dists_b = []
    dists_d = []
    buckets = defaultdict(list)
    for row in rows:
        if row["family"] not in ("oblique_diag", "block2", "block4") or not row["sides"] or "log_d" not in row["sides"][0]:
            continue
        buckets[(row["side_a"], row["side_b"], row["family"], row["profile"], row["mode"], float(row["rho"]))].append(row)
    for items in buckets.values():
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                dists_b.append(spectrum_distance(left["sides"][0]["log_b"], right["sides"][0]["log_b"]))
                dists_d.append(spectrum_distance(left["sides"][0]["log_d"], right["sides"][0]["log_d"]))
    if dists_b:
        lines.append(f"Across replicate seeds, median distance of induced log-spectra of B is {float(np.median(dists_b)):.4f}. Median distance of log-spectra of D is {float(np.median(dists_d)):.4f}.")
        if float(np.median(dists_d)) > float(np.median(dists_b)) + 0.01:
            lines.append("D moves more across seeds than B. Different factorizations are producing similar induced metrics. The factorization is not uniquely identified.")
        else:
            lines.append("On this panel the seed-to-seed movement of D is not larger than that of B. That still does not identify the factors: column signs, block-wise rotations inside a constant D, and the scale gauge removed by normalisation remain free when they leave B unchanged.")
    lines.append("")
    return lines


def _answer_mnn(rows, ident) -> list[str]:
    lines = ["### 7. Mutual kNN", ""]
    for family, profile in (("exp_s", "na"), ("oblique_diag", "basis"), ("block4", "strong")):
        for rho in (0.1, 0.4):
            for mode in ("a_only", "separate"):
                meds = _pair_medians(rows, "vl", family, profile, mode, rho, "exploratory_mnn")
                gains = []
                for (side_a, side_b), val in meds.items():
                    base = ident[("vl", side_a, side_b)]["exploratory_mnn"]
                    gains.append(val - float(base))
                if gains:
                    lines.append(f"- {family}/{profile} {MODE_LABEL[mode]} rho={rho:g}: median V-L exploratory mNN gain over identity {float(np.median(gains)):+.4f}.")
    lines.append("")
    return lines


def _answer_sensitivity(rows) -> list[str]:
    lines = ["### 8. Seed, model, and PCA coordinates", ""]
    spreads = []
    buckets = defaultdict(list)
    for row in rows:
        if row["kind"] != "vl":
            continue
        buckets[(row["side_a"], row["side_b"], row["family"], row["profile"], row["mode"], float(row["rho"]))].append(float(row["exploratory_excess"]))
    for vals in buckets.values():
        if len(vals) >= 2:
            spreads.append(float(max(vals) - min(vals)))
    if spreads:
        lines.append(f"Median within-pair seed range of V-L exploratory excess is {float(np.median(spreads)):.4f}, maximum {float(np.max(spreads)):.4f}.")
    by_model = defaultdict(list)
    for row in rows:
        if row["kind"] == "vl" and row["family"] == "exp_s" and row["mode"] == "separate" and abs(float(row["rho"]) - 0.4) < 1e-9:
            by_model[row["side_a"]].append(float(row["exploratory_excess"]))
    if by_model:
        text = ", ".join(f"{model} {float(np.median(vals)):.3f}" for model, vals in by_model.items())
        lines.append(f"exp(S) separate rho=0.4 median exploratory excess by language model: {text}.")
    lines.append("Every comparison uses the frozen training-PCA sign convention, largest-absolute entry nonnegative. Shared-PC still ties unrelated coordinate systems. A result that moves when those coordinates are rotated is a property of this basis, not of an ambient semantic metric. This study did not refit PCA.")
    lines.append("")
    return lines


def _verify_protected(out: Path) -> None:
    audit = json.loads((out / "audit.json").read_text())
    for key, previous in audit["protected_mtime"].items():
        if key == "metric_stability_report":
            path = out.parent / "metric_stability" / "report.md"
        else:
            path = out.parent / "resampled_metric_training" / "report.md"
        current = _mtime(path)
        if previous != current:
            raise RuntimeError(f"protected report changed: {path}")


def _mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None
