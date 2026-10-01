"""Full-gallery versus quarter-gallery training for one language-side metric.

The exploratory-test gallery is scored only in the eval stage. Checkpoint
selection uses the validation gallery.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from prh_replication.anisotropic_kernels import metric_diagnostics, sample_center
from prh_replication.extract_final import final_feature_path, load_final_prepared, spec_from_manifest
from prh_replication.io_utils import jsonable, sha256_file, write_json
from prh_replication.metric_stability import (
    assert_split_image_disjoint,
    b_matrix_exp,
    delta_c,
    delta_m_cosine,
    distortions,
    evaluate_metric,
    fit_basis,
    gram_cosine,
    ids_hash,
    run_a_only_stream,
    subspace_compare,
)
from prh_replication.plots import _save
from prh_replication.registry import MODELS, Paths
from prh_replication.release_protocol import VIS, load_splits

LABELS = {
    "qwen2-7b": "Qwen2-7B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen3-8b-base": "Qwen3-8B-Base",
    "olmo-7b-0724": "OLMo-7B-0724",
    "olmo2-1124-7b": "OLMo-2-1124-7B",
    "olmo-3-1025-7b": "OLMo-3-1025-7B",
}
S_KEYS = ("s_at_primary", "s_selected_primary", "s_at_final", "s_selected_final")


def run(work: Path, repo: Path, stage: str, force: bool = False) -> None:
    paths = Paths(work=work, repo=repo)
    design = json.loads((repo / "configs" / "resampled_metric_training.json").read_text())
    out = paths.results / "resampled_metric_training"
    out.mkdir(parents=True, exist_ok=True)
    if stage in ("smoke", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val"), force=force)
        _smoke(ctx)
    if stage in ("fit", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val"), force=force)
        _fit(ctx, force=force)
        print("FIT_DONE", flush=True)
    if stage in ("eval", "all"):
        ctx = _context(paths, repo, design, out, splits=("train", "val", "test"), force=False)
        _eval(ctx, force=force)
        _stability(ctx, force=force)
        print("EVAL_DONE", flush=True)
    if stage in ("report", "all"):
        _figures(out)
        _write_report(out, design)
        _verify_protected(out)
        print("REPORT_DONE", flush=True)


def _context(paths: Paths, repo: Path, design: dict, out: Path, *, splits: tuple[str, ...], force: bool) -> dict[str, Any]:
    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    panel = list(design["language_models"])
    base = [row["key"] for row in manifest["base_panel"]]
    if panel != base:
        raise RuntimeError("language panel does not match release_models base_panel order")
    specs = {row["key"]: spec_from_manifest(row) for row in manifest["base_panel"]}
    partner = design["partner"]
    if partner not in VIS:
        raise RuntimeError(partner)
    specs[partner] = MODELS[partner]
    _, split_ids = load_splits(paths, repo)
    use = {name: list(split_ids[name]) for name in ("train", "val", "test")}
    assert_split_image_disjoint(use)
    if len(use["train"]) != 2048 or len(use["val"]) != 1024 or len(use["test"]) != 1024:
        raise RuntimeError("unexpected split sizes")
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for key in panel + [partner]:
        prepared[key] = {}
        for split in splits:
            x, obj = load_final_prepared(paths, specs[key], split, use[split])
            if list(obj["sample_ids"]) != use[split]:
                raise ValueError(f"sample-id mismatch {key} {split}")
            prepared[key][split] = np.asarray(x.numpy(), dtype=np.float64)
    bases = _bases(prepared, panel, partner, int(design["q"]), out, force=force)
    protected = {
        "metric_stability_summary": _mtime(paths.results / "metric_stability" / "summary.json"),
        "metric_stability_report": _mtime(paths.results / "metric_stability" / "report.md"),
    }
    if not (out / "protocol.md").exists() or force:
        (out / "protocol.md").write_text((repo / "configs" / "resampled_metric_training_protocol.md").read_text())
        write_json(out / "design.json", design)
        write_json(
            out / "audit.json",
            {
                "train_ids_hash": ids_hash(use["train"]),
                "val_ids_hash": ids_hash(use["val"]),
                "test_ids_hash": ids_hash(use["test"]),
                "basis_ids": {key: bases[key]["id"] for key in panel + [partner]},
                "protected_mtime": protected,
                "feature_sha256": {
                    f"{key}:{split}": sha256_file(final_feature_path(paths, specs[key], "coco_val2017", split, use[split]))
                    for key in panel + [partner]
                    for split in splits
                },
            },
        )
    return {
        "paths": paths,
        "design": design,
        "out": out,
        "prepared": prepared,
        "ids": use,
        "bases": bases,
        "panel": panel,
        "partner": partner,
        "protected": protected,
    }


def _bases(prepared, panel, partner, q, out: Path, force: bool) -> dict[str, dict]:
    dest = out / "bases"
    dest.mkdir(parents=True, exist_ok=True)
    bases = {}
    for key in panel + [partner]:
        path = dest / f"{key}.npz"
        meta_path = dest / f"{key}.json"
        if path.exists() and meta_path.exists() and not force:
            meta = json.loads(meta_path.read_text())
            blob = np.load(path)
            basis = {
                "mu": blob["mu"],
                "U": blob["U"],
                "q": int(meta["q"]),
                "q_requested": q,
                "numerical_rank": int(meta["numerical_rank"]),
                "variance_fraction": float(meta["variance_fraction"]),
                "reduced": bool(meta["reduced"]),
                "n": int(meta["n"]),
                "d": int(meta["d"]),
                "sign_convention": meta["sign_convention"],
                "id": meta["id"],
            }
        else:
            basis = fit_basis(prepared[key]["train"], q=q)
            np.savez(path, mu=basis["mu"], U=basis["U"])
            write_json(meta_path, {k: basis[k] for k in ("q", "numerical_rank", "variance_fraction", "reduced", "n", "d", "sign_convention", "id")})
        if int(basis["q"]) != q or basis["reduced"]:
            raise RuntimeError(f"{key} does not support q={q}")
        bases[key] = basis
    return bases


def _fit(ctx: dict[str, Any], force: bool) -> None:
    design = ctx["design"]
    runs = ctx["out"] / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(design)
    t0 = time.perf_counter()
    for i, job in enumerate(jobs, start=1):
        name = _run_name(job)
        dest = runs / f"{name}.json"
        if dest.exists() and not force:
            print(f"skip {name}", flush=True)
            continue
        print(f"fit {i}/{len(jobs)} {name}", flush=True)
        raw = _fit_job(ctx, job, runs / name)
        if not raw["ok"]:
            _write_run(dest, raw, job)
            raise RuntimeError(f"{name} stopped: {raw['status']}")
        _write_run(dest, raw, job)
        _save_s(runs / f"{name}.s.npz", raw)
        elapsed = time.perf_counter() - t0
        print(
            f"done {name} updates={raw['updates_done']} optim_s={raw['optim_seconds']:.1f} "
            f"val_s={raw['validation_seconds']:.1f} elapsed={elapsed:.0f}",
            flush=True,
        )


def _jobs(design: dict) -> list[dict[str, Any]]:
    jobs = []
    for lang in design["language_models"]:
        for rho in design["budgets_rho"]:
            for init in design["init_seeds"]:
                jobs.append({"language": lang, "procedure": "full", "rho": float(rho), "init_seed": int(init), "sampling_seed": None, "n_updates": int(design["n_updates"])})
                for samp in design["sampling_seeds"]:
                    jobs.append(
                        {
                            "language": lang,
                            "procedure": "quarter",
                            "rho": float(rho),
                            "init_seed": int(init),
                            "sampling_seed": int(samp),
                            "n_updates": int(design["extended_updates"]),
                        }
                    )
    return jobs


def _fit_job(ctx, job, stem: Path) -> dict[str, Any]:
    design = ctx["design"]
    lang, partner = job["language"], ctx["partner"]
    resume = _load_state(stem) if _state_json(stem).exists() else None
    primary = int(design["n_updates"])

    def on_checkpoint(state: dict[str, Any]) -> None:
        step = int(state["updates_done"])
        if step == 0 or step % 20 == 0 or step in (primary, int(job["n_updates"])):
            _save_state(stem, state)

    return run_a_only_stream(
        ctx["prepared"][lang]["train"],
        ctx["prepared"][partner]["train"],
        ctx["ids"]["train"],
        ctx["bases"][lang],
        ctx["bases"][partner],
        rho=float(job["rho"]),
        init_seed=int(job["init_seed"]),
        n_updates=int(job["n_updates"]),
        procedure=job["procedure"],
        sampling_seed=job["sampling_seed"],
        lr=float(design["optim"]["lr"]),
        grad_clip=float(design["optim"]["grad_clip"]),
        pert_scale=float(design["pert_scale"]),
        val_features_a=ctx["prepared"][lang]["val"],
        val_features_b=ctx["prepared"][partner]["val"],
        resume=resume,
        primary_updates=primary,
        quarter_fraction=float(design["quarter_fraction"]),
        on_checkpoint=on_checkpoint,
    )


def _write_run(path: Path, raw: dict[str, Any], job: dict[str, Any]) -> None:
    payload = {
        "language": job["language"],
        "partner": "dinov2-small",
        "procedure": raw["procedure"],
        "objective": raw["objective"],
        "objective_note": raw["objective_note"],
        "rho": raw["rho"],
        "init_seed": raw["init_seed"],
        "sampling_seed": raw["sampling_seed"],
        "n_updates_requested": raw["n_updates_requested"],
        "updates_done": raw["updates_done"],
        "primary_updates": raw["primary_updates"],
        "n_train": raw["n_train"],
        "examples_per_update": raw["examples_per_update"],
        "examples_processed": raw["examples_processed"],
        "examples_processed_primary": raw["examples_processed_primary"],
        "n_val_checks": raw["n_val_checks"],
        "n_val_checks_primary": raw["n_val_checks_primary"],
        "curves": raw["curves"],
        "selected_primary": raw["selected_primary"],
        "selected_final": raw["selected_final"],
        "feasible": raw["feasible"],
        "optim_seconds": raw["optim_seconds"],
        "validation_seconds": raw["validation_seconds"],
        "train_eval_seconds": raw["train_eval_seconds"],
        "basis_a_id": raw["basis_a_id"],
        "basis_b_id": raw["basis_b_id"],
        "lr": raw["lr"],
        "grad_clip": raw["grad_clip"],
        "schedule": "constant Adam lr through the last requested update",
        "status": raw["status"],
        "ok": raw["ok"],
    }
    write_json(path, jsonable(payload))


def _save_s(path: Path, raw: dict[str, Any]) -> None:
    arrays = {}
    for key in S_KEYS:
        value = raw.get(key)
        arrays[key] = np.zeros((0, 0)) if value is None else np.asarray(value, dtype=np.float64)
    np.savez(path, **arrays)


def _state_json(stem: Path) -> Path:
    return Path(str(stem) + ".state.json")


def _state_npz(stem: Path) -> Path:
    return Path(str(stem) + ".state.npz")


def _save_state(stem: Path, state: dict[str, Any]) -> None:
    arrays = {}
    meta: dict[str, Any] = {}
    adam = state.get("adam")
    if adam is not None:
        arrays["adam_exp_avg"] = np.asarray(adam["exp_avg"], dtype=np.float64)
        arrays["adam_exp_avg_sq"] = np.asarray(adam["exp_avg_sq"], dtype=np.float64)
        meta["adam_step"] = int(adam["step"])
    else:
        meta["adam_step"] = None
    for key in ("vech",) + S_KEYS:
        value = state.get(key)
        if isinstance(value, np.ndarray):
            arrays[key] = value
    for key, value in state.items():
        if key in arrays or key in ("adam",) + S_KEYS or key == "vech":
            continue
        meta[key] = value
    np.savez(_state_npz(stem), **arrays)
    write_json(_state_json(stem), meta)


def _load_state(stem: Path) -> dict[str, Any] | None:
    meta_path = _state_json(stem)
    npz_path = _state_npz(stem)
    if not meta_path.exists() or not npz_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    blob = np.load(npz_path)
    state = dict(meta)
    state["vech"] = np.array(blob["vech"], dtype=np.float64, copy=True)
    for key in S_KEYS:
        if key in blob.files and blob[key].size:
            state[key] = np.array(blob[key], dtype=np.float64, copy=True)
        else:
            state[key] = None
    if meta.get("adam_step") is None:
        state["adam"] = None
    else:
        state["adam"] = {
            "exp_avg": np.array(blob["adam_exp_avg"], dtype=np.float64, copy=True),
            "exp_avg_sq": np.array(blob["adam_exp_avg_sq"], dtype=np.float64, copy=True),
            "step": int(meta["adam_step"]),
        }
    return state


def _smoke(ctx: dict[str, Any]) -> None:
    lang = ctx["panel"][0]
    partner = ctx["partner"]
    ids = ctx["ids"]["train"]
    common = dict(
        rho=0.1,
        init_seed=0,
        lr=0.03,
        grad_clip=5.0,
        pert_scale=0.05,
        val_features_a=ctx["prepared"][lang]["val"],
        val_features_b=ctx["prepared"][partner]["val"],
        primary_updates=4,
    )
    full = run_a_only_stream(
        ctx["prepared"][lang]["train"],
        ctx["prepared"][partner]["train"],
        ids,
        ctx["bases"][lang],
        ctx["bases"][partner],
        n_updates=8,
        procedure="full",
        **common,
    )
    quarter = run_a_only_stream(
        ctx["prepared"][lang]["train"],
        ctx["prepared"][partner]["train"],
        ids,
        ctx["bases"][lang],
        ctx["bases"][partner],
        n_updates=8,
        procedure="quarter",
        sampling_seed=0,
        quarter_fraction=0.25,
        record_indices=True,
        **common,
    )
    if not full["feasible"] or not quarter["feasible"]:
        raise RuntimeError("smoke left the feasible set")
    if full["status"]["zero_grad_at_identity"] or quarter["status"]["zero_grad_at_identity"]:
        raise RuntimeError("identity initialisation has a zero gradient")
    idx = quarter["subset_indices"][0]
    if len(idx) != 512 or len(set(idx)) != 512:
        raise RuntimeError("smoke quarter was not 512 unique training rows")
    per = quarter["optim_seconds"] / max(quarter["updates_done"], 1)
    write_json(
        ctx["out"] / "smoke.json",
        {
            "language": lang,
            "full_train_excess_step0": full["curves"][0]["full_train_excess"],
            "full_train_excess_step8": full["curves"][-1]["full_train_excess"],
            "quarter_seconds_per_update": per,
            "quarter_n": len(idx),
            "feasible": True,
        },
    )
    print(f"smoke quarter seconds/update {per:.3f}", flush=True)


def _eval(ctx: dict[str, Any], force: bool) -> None:
    if "test" not in ctx["prepared"][ctx["panel"][0]]:
        raise RuntimeError("eval requires the exploratory-test gallery")
    ev = ctx["out"] / "evals"
    ev.mkdir(parents=True, exist_ok=True)
    design = ctx["design"]
    for lang in ctx["panel"]:
        ident_path = ev / f"identity__{lang}.json"
        if not ident_path.exists() or force:
            write_json(ident_path, _score_record(ctx, lang, np.zeros((design["q"], design["q"])), kind="identity"))
    for path in sorted((ctx["out"] / "runs").glob("*.json")):
        if path.name.endswith(".state.json"):
            continue
        dest = ev / path.name
        if dest.exists() and not force:
            continue
        raw = json.loads(path.read_text())
        blob = np.load(path.with_suffix(".s.npz"))
        record = {"language": raw["language"], "procedure": raw["procedure"], "rho": raw["rho"], "init_seed": raw["init_seed"], "sampling_seed": raw["sampling_seed"], "checkpoints": {}}
        for key in S_KEYS:
            if key not in blob.files or blob[key].size == 0:
                continue
            record["checkpoints"][key] = _score_record(ctx, raw["language"], blob[key], kind=key)
        write_json(dest, jsonable(record))
        print(f"eval {path.stem}", flush=True)


def _score_record(ctx, lang: str, s: np.ndarray, kind: str) -> dict[str, Any]:
    partner = ctx["partner"]
    s = np.asarray(s, dtype=np.float64)
    zeros = np.zeros((ctx["bases"][partner]["q"], ctx["bases"][partner]["q"]))
    galleries = {}
    for split in ("train", "val", "test"):
        galleries[split if split != "test" else "exploratory"] = evaluate_metric(
            ctx["prepared"][lang][split],
            ctx["prepared"][partner][split],
            ctx["bases"][lang],
            ctx["bases"][partner],
            s,
            zeros,
            knn_k=int(ctx["design"]["knn_k"]),
        )
    b = b_matrix_exp(s)
    eig = np.linalg.eigvalsh(b)
    da, db, total = distortions(s, zeros)
    return {
        "kind": kind,
        "galleries": galleries,
        "d_a": da,
        "d_b": db,
        "d_total": total,
        "diagnostics": metric_diagnostics(b, s, eig),
    }


def _stability(ctx: dict[str, Any], force: bool) -> None:
    dest = ctx["out"] / "stability.json"
    if dest.exists() and not force:
        return
    rows = []
    runs = _load_runs(ctx["out"])
    for rho in ctx["design"]["budgets_rho"]:
        for lang in ctx["panel"]:
            full = [r for r in runs if r["language"] == lang and r["procedure"] == "full" and r["rho"] == float(rho)]
            quarter = [r for r in runs if r["language"] == lang and r["procedure"] == "quarter" and r["rho"] == float(rho)]
            specs = [
                ("full_inits", "equal_update_selected", combinations(full, 2), "s_selected_primary", "s_selected_primary"),
                ("full_inits", "equal_update_final", combinations(full, 2), "s_at_primary", "s_at_primary"),
                ("streams_same_init", "equal_update_selected", _grouped_pairs(quarter, "init_seed"), "s_selected_primary", "s_selected_primary"),
                ("streams_same_init", "equal_update_final", _grouped_pairs(quarter, "init_seed"), "s_at_primary", "s_at_primary"),
                ("resampled_inits", "equal_update_selected", _grouped_pairs(quarter, "sampling_seed"), "s_selected_primary", "s_selected_primary"),
                ("resampled_inits", "equal_update_final", _grouped_pairs(quarter, "sampling_seed"), "s_at_primary", "s_at_primary"),
                ("full_vs_resampled", "equal_update_selected", _matched_init_pairs(full, quarter), "s_selected_primary", "s_selected_primary"),
                ("full_vs_resampled", "equal_update_final", _matched_init_pairs(full, quarter), "s_at_primary", "s_at_primary"),
                ("streams_same_init", "extended_selected", _grouped_pairs(quarter, "init_seed"), "s_selected_final", "s_selected_final"),
                ("streams_same_init", "extended_final", _grouped_pairs(quarter, "init_seed"), "s_at_final", "s_at_final"),
                ("resampled_inits", "extended_selected", _grouped_pairs(quarter, "sampling_seed"), "s_selected_final", "s_selected_final"),
                ("full_vs_resampled", "extended_selected", _matched_init_pairs(full, quarter), "s_selected_primary", "s_selected_final"),
                ("full_vs_resampled", "extended_final", _matched_init_pairs(full, quarter), "s_at_primary", "s_at_final"),
            ]
            g_cache = _gallery_cache(ctx, lang)
            for comparison, horizon, pairs, key_a, key_b in specs:
                for left, right in pairs:
                    rows.append(_compare_pair(ctx, lang, float(rho), comparison, horizon, left, right, key_a, key_b, g_cache))
    write_json(dest, jsonable({"rows": rows, "note": "Optimisation reproducibility on the frozen full-training PCA basis. Not a resampling of the corpus."}))


def _gallery_cache(ctx, lang: str) -> dict[str, np.ndarray]:
    zc = sample_center(ctx["prepared"][lang]["test"] - ctx["bases"][lang]["mu"])
    u = ctx["bases"][lang]["U"]
    return {"linear": zc @ zc.T, "projected": zc @ u, "U": u}


def _correction_from_projected(projected: np.ndarray, s: np.ndarray) -> np.ndarray:
    c = delta_c(b_matrix_exp(s))
    return projected @ c @ projected.T


def _compare_pair(ctx, lang, rho, comparison, horizon, left, right, key_a, key_b, cache) -> dict[str, Any]:
    sa, sb = left["S"][key_a], right["S"][key_b]
    u = cache["U"]
    cos = delta_m_cosine(u, b_matrix_exp(sa), u, b_matrix_exp(sb))
    corr_a = _correction_from_projected(cache["projected"], sa)
    corr_b = _correction_from_projected(cache["projected"], sb)
    gram_a = cache["linear"] + corr_a
    gram_b = cache["linear"] + corr_b
    return {
        "language": lang,
        "rho": rho,
        "comparison": comparison,
        "horizon": horizon,
        "init_left": left["init_seed"],
        "init_right": right["init_seed"],
        "sample_left": left["sampling_seed"],
        "sample_right": right["sampling_seed"],
        "cosine": cos["cosine"],
        "norm_left": cos["norm_a"],
        "norm_right": cos["norm_b"],
        "nearly_zero": cos["nearly_zero"],
        "d_left": float(distortions(sa, np.zeros_like(sa))[0]),
        "d_right": float(distortions(sb, np.zeros_like(sb))[0]),
        "amplified_rank4": subspace_compare(u, sa, u, sb, "amplified", 4)["overlap"],
        "amplified_rank8": subspace_compare(u, sa, u, sb, "amplified", 8)["overlap"],
        "suppressed_rank4": subspace_compare(u, sa, u, sb, "suppressed", 4)["overlap"],
        "suppressed_rank8": subspace_compare(u, sa, u, sb, "suppressed", 8)["overlap"],
        "centred_gram_cosine": gram_cosine(gram_a, gram_b),
        "correction_gram_cosine": gram_cosine(corr_a, corr_b),
    }


def _load_runs(out: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out / "runs").glob("*.json")):
        if path.name.endswith(".state.json"):
            continue
        raw = json.loads(path.read_text())
        blob = np.load(path.with_suffix(".s.npz"))
        raw["S"] = {key: blob[key] for key in S_KEYS if key in blob.files and blob[key].size}
        rows.append(raw)
    return rows


def _grouped_pairs(runs: list[dict], field: str):
    values = sorted({r[field] for r in runs})
    for value in values:
        group = [r for r in runs if r[field] == value]
        yield from combinations(group, 2)


def _matched_init_pairs(full, quarter):
    for left in full:
        for right in quarter:
            if int(left["init_seed"]) == int(right["init_seed"]):
                yield left, right


def _figures(out: Path) -> None:
    import matplotlib.pyplot as plt

    runs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json")) if not p.name.endswith(".state.json")]
    joined = _joined(out)
    fig_dir = out / "figures"
    for rho in (0.1, 0.4):
        fig, axes = plt.subplots(2, 6, figsize=(16, 5.5), sharex=True)
        for col, lang in enumerate(LABELS):
            _curve_panel(axes[0, col], runs, lang, rho, "full_train_excess", "Full-training excess")
            _curve_panel(axes[1, col], runs, lang, rho, "full_val_excess", "Full-validation excess")
            axes[0, col].set_title(LABELS[lang], fontsize=9)
            axes[1, col].set_xlabel("Update")
        fig.suptitle(f"Learning curves at ρ={rho}. Vertical line is the equal-update horizon.")
        _save(fig, fig_dir / f"learning_curves_rho{rho}.png")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        _gain_panel(axes[0], joined, rho, "excess", "Exploratory excess minus identity")
        _gain_panel(axes[1], joined, rho, "mnn_k10", "Exploratory mutual kNN minus identity")
        fig.suptitle(f"Held-out gains at ρ={rho}. Points are repeated optimisations, not data replicates.")
        _save(fig, fig_dir / f"heldout_gains_rho{rho}.png")
    stab = json.loads((out / "stability.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, rho in zip(axes, (0.1, 0.4)):
        _stability_panel(ax, stab, rho)
        ax.set_title(f"ρ={rho}")
    axes[0].set_ylabel("ΔM Frobenius cosine")
    fig.suptitle("Correction similarity. Missing cosines are omitted, not drawn as zero.")
    _save(fig, fig_dir / "correction_cosine.png")


def _curve_panel(ax, runs, lang, rho, field, ylabel):
    full, quarter = [], []
    for run in runs:
        if run["language"] != lang or float(run["rho"]) != float(rho):
            continue
        steps = [row["step"] for row in run["curves"]]
        vals = [row[field] for row in run["curves"]]
        if run["procedure"] == "full":
            ax.plot(steps, vals, color="C0", alpha=0.35, lw=1)
            full.append(vals)
        else:
            ax.plot(steps, vals, color="C1", alpha=0.25, lw=1)
            quarter.append(vals)
    if full:
        ax.plot(range(len(full[0])), np.median(full, axis=0), color="C0", lw=2, label="full")
    if quarter:
        ax.plot(range(len(quarter[0])), np.median(quarter, axis=0), color="C1", lw=2, label="quarter")
    ax.axvline(120, color="k", lw=0.6, ls="--")
    ax.set_ylabel(ylabel, fontsize=8)
    if ax is not None and lang == "qwen2-7b" and field == "full_train_excess":
        ax.legend(fontsize=7, frameon=False)


def _gain_panel(ax, joined, rho, field, ylabel):
    kinds = [
        ("full", "selected_primary", "Full T"),
        ("quarter", "selected_primary", "Quarter T"),
        ("quarter", "selected_final", "Quarter 4T"),
    ]
    langs = list(LABELS)
    for i, (procedure, checkpoint, label) in enumerate(kinds):
        xs, ys = [], []
        for j, lang in enumerate(langs):
            ident = _identity_value(joined, lang, field)
            chunk = []
            for row in joined:
                if row["language"] == lang and row["procedure"] == procedure and float(row["rho"]) == float(rho) and row["checkpoint"] == checkpoint:
                    chunk.append(row["exploratory"][field] - ident)
            if not chunk:
                continue
            x = j + (i - 1) * 0.18
            xs.extend([x] * len(chunk))
            ys.extend(chunk)
            ax.plot([x], [float(np.median(chunk))], "k_", ms=10, zorder=3)
        ax.scatter(xs, ys, s=18, label=label, zorder=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(langs)), [LABELS[k] for k in langs], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, frameon=False)


def _stability_panel(ax, rows, rho):
    order = ["full_inits", "streams_same_init", "resampled_inits", "full_vs_resampled"]
    labels = ["Full starts", "Streams", "Resampled starts", "Full vs quarter"]
    langs = list(LABELS)
    for i, comparison in enumerate(order):
        for j, lang in enumerate(langs):
            vals = [
                row["cosine"]
                for row in rows
                if row["comparison"] == comparison and row["horizon"] == "equal_update_selected" and float(row["rho"]) == float(rho) and row["language"] == lang and row["cosine"] is not None
            ]
            ax.scatter(np.full(len(vals), i) + (j - 2.5) * 0.04, vals, s=16, color=f"C{j}", label=LABELS[lang] if i == 0 else None)
    ax.set_xticks(range(len(order)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    if rho == 0.1:
        ax.legend(fontsize=6, frameon=False, loc="lower left")


def _joined(out: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out / "evals").glob("*.json")):
        raw = json.loads(path.read_text())
        if path.name.startswith("identity__"):
            rows.append({"language": path.name.replace("identity__", "").replace(".json", ""), "procedure": "identity", "rho": 0.0, "checkpoint": "identity", "init_seed": None, "sampling_seed": None, "exploratory": raw["galleries"]["exploratory"], "val": raw["galleries"]["val"], "train": raw["galleries"]["train"], "d_total": raw["d_total"], "diagnostics": raw["diagnostics"]})
            continue
        run = json.loads((out / "runs" / path.name).read_text())
        for key, scored in raw["checkpoints"].items():
            checkpoint = {
                "s_selected_primary": "selected_primary",
                "s_at_primary": "final_primary",
                "s_selected_final": "selected_final",
                "s_at_final": "final_final",
            }[key]
            rows.append(
                {
                    "language": raw["language"],
                    "procedure": raw["procedure"],
                    "rho": float(raw["rho"]),
                    "checkpoint": checkpoint,
                    "init_seed": raw["init_seed"],
                    "sampling_seed": raw["sampling_seed"],
                    "step": (run["selected_primary"] or {}).get("step") if "selected_primary" in checkpoint else (run["selected_final"] or {}).get("step") if "selected" in checkpoint else int(run["primary_updates"] if "primary" in checkpoint else run["updates_done"]),
                    "exploratory": scored["galleries"]["exploratory"],
                    "val": scored["galleries"]["val"],
                    "train": scored["galleries"]["train"],
                    "d_total": scored["d_total"],
                    "diagnostics": scored["diagnostics"],
                    "optim_seconds": run["optim_seconds"],
                    "validation_seconds": run["validation_seconds"],
                    "n_val_checks": run["n_val_checks"],
                    "n_val_checks_primary": run["n_val_checks_primary"],
                    "examples_processed": run["examples_processed"],
                    "examples_processed_primary": run["examples_processed_primary"],
                    "updates_done": run["updates_done"],
                }
            )
    return rows


def _identity_value(joined, lang, field) -> float:
    for row in joined:
        if row["procedure"] == "identity" and row["language"] == lang:
            return float(row["exploratory"][field])
    raise KeyError(lang)


def _write_report(out: Path, design: dict) -> None:
    joined = _joined(out)
    write_json(out / "joined_rows.json", jsonable(joined))
    stab = json.loads((out / "stability.json").read_text())["rows"]
    runs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json")) if not p.name.endswith(".state.json")]
    lines = [
        "# Resampled metric training",
        "",
        "Exploratory held-out evaluation on the manifest split named test. That gallery was not used to choose steps, budgets, or seeds. The language-side basis is the signed full-training PCA basis, frozen for every update. Quarter-gallery updates maximise expected subset excess. That is not full-gallery excess, and the subset gradient is not an unbiased estimator of the full-gallery gradient.",
        "",
        f"Equal-update horizon: {design['n_updates']} updates. Full gallery sees {design['n_updates']}×2048 pairs. Quarter gallery sees {design['n_updates']}×512 pairs at that horizon, and {design['extended_updates']}×512 pairs if the same trajectory is continued. The continuation uses the same Adam learning rate 0.03 and clip 5. The longer run has more validation checks, so it has more chances to pick a checkpoint. Neither comparison is compute-matched.",
        "",
        _budget_paragraph(runs),
        "",
        "## Exploratory scores",
        "",
        _score_table(joined),
        "",
        "## Answers",
        "",
    ]
    lines.extend(_answers(joined, stab))
    lines.extend(["", "## Claim", "", _claim(joined, stab), ""])
    (out / "report.md").write_text("\n".join(lines))
    write_json(out / "summary.json", {"n_runs": len(runs), "n_joined": len(joined), "n_stability": len(stab)})


def _budget_paragraph(runs: list[dict]) -> str:
    full = [r for r in runs if r["procedure"] == "full"]
    quarter = [r for r in runs if r["procedure"] == "quarter"]
    return (
        f"Runs kept: {len(full)} full-gallery and {len(quarter)} quarter-gallery. "
        f"Validation checks through the equal-update horizon: {full[0]['n_val_checks_primary'] if full else 'n/a'} "
        f"(full) and {quarter[0]['n_val_checks_primary'] if quarter else 'n/a'} (quarter). "
        f"Extended quarter checks: {quarter[0]['n_val_checks'] if quarter else 'n/a'}. "
        f"Optimisation wall time {sum(r['optim_seconds'] for r in runs) / 3600:.2f} h; "
        f"validation scoring {sum(r['validation_seconds'] for r in runs) / 3600:.2f} h."
    )


def _score_table(joined) -> str:
    header = "| Model | ρ | Procedure | Checkpoint | excess | raw CKA | mNN | D | eig range | frac at bound |"
    rule = "|---|---:|---|---|---:|---:|---:|---:|---|---:|"
    body = [header, rule]
    for lang in LABELS:
        ident = next(r for r in joined if r["language"] == lang and r["procedure"] == "identity")
        body.append(_score_line(LABELS[lang], ident))
        for rho in (0.1, 0.4):
            for procedure, checkpoint, label in (
                ("full", "selected_primary", "selected T"),
                ("full", "final_primary", "final T"),
                ("quarter", "selected_primary", "selected T"),
                ("quarter", "final_primary", "final T"),
                ("quarter", "selected_final", "selected 4T"),
                ("quarter", "final_final", "final 4T"),
            ):
                rows = [r for r in joined if r["language"] == lang and r["procedure"] == procedure and float(r["rho"]) == rho and r["checkpoint"] == checkpoint]
                if rows:
                    body.append(_score_line(LABELS[lang], _median_row(rows, rho, procedure, label)))
    return "\n".join(body)


def _median_row(rows, rho, procedure, label) -> dict:
    def med(field):
        return float(np.median([r["exploratory"][field] for r in rows]))

    return {
        "rho": rho,
        "procedure": procedure,
        "checkpoint": label,
        "exploratory": {"excess": med("excess"), "a": med("a"), "mnn_k10": med("mnn_k10")},
        "d_total": float(np.median([r["d_total"] for r in rows])),
        "diagnostics": {
            "eig_min": float(np.median([r["diagnostics"]["eig_min"] for r in rows])),
            "eig_max": float(np.median([r["diagnostics"]["eig_max"] for r in rows])),
            "frac_at_bounds": float(np.median([r["diagnostics"]["frac_at_bounds"] for r in rows])),
        },
    }


def _score_line(model, row) -> str:
    ex = row["exploratory"]
    diag = row["diagnostics"]
    return (
        f"| {model} | {float(row['rho']):.1f} | {row['procedure']} | {row['checkpoint']} | "
        f"{ex['excess']:.4f} | {ex['a']:.4f} | {ex['mnn_k10']:.4f} | {row['d_total']:.4f} | "
        f"{diag['eig_min']:.3f}–{diag['eig_max']:.3f} | {diag['frac_at_bounds']:.3f} |"
    )


def _answers(joined, stab) -> list[str]:
    lines = []
    for number, text in enumerate(_answer_sentences(joined, stab), start=1):
        lines.append(f"{number}. {text}")
        lines.append("")
    return lines


def _answer_sentences(joined, stab) -> list[str]:
    out = []
    excess_bits, mnn_bits = [], []
    for rho in (0.1, 0.4):
        excess_bits.append(_delta_sentence(joined, rho, "excess"))
        mnn_bits.append(_delta_sentence(joined, rho, "mnn_k10"))
    out.append("Equal-update validation-selected exploratory excess, quarter minus full: " + " ".join(excess_bits))
    out.append("Equal-update validation-selected exploratory mutual kNN, quarter minus full: " + " ".join(mnn_bits))
    out.append(_repro_sentence(stab, "equal_update_selected"))
    out.append(_extended_sentence(joined))
    out.append(_distortion_sentence(joined, stab))
    out.append(_consistency_sentence(joined))
    return out


def _delta_sentence(joined, rho, field) -> str:
    diffs = []
    signs = []
    for lang in LABELS:
        full = _median_metric(joined, lang, "full", rho, "selected_primary", field)
        quarter = _median_metric(joined, lang, "quarter", rho, "selected_primary", field)
        final_q = _median_metric(joined, lang, "quarter", rho, "final_primary", field)
        delta = quarter - full
        diffs.append(delta)
        signs.append(f"{LABELS[lang]} {delta:+.4f} (final-iterate {final_q - _median_metric(joined, lang, 'full', rho, 'final_primary', field):+.4f})")
    return f"at ρ={rho}, median of model medians {float(np.median(diffs)):+.4f} ({'; '.join(signs)})."


def _median_metric(joined, lang, procedure, rho, checkpoint, field) -> float:
    rows = [r for r in joined if r["language"] == lang and r["procedure"] == procedure and float(r["rho"]) == float(rho) and r["checkpoint"] == checkpoint]
    return float(np.median([r["exploratory"][field] for r in rows]))


def _repro_sentence(stab, horizon: str) -> str:
    parts = []
    for comparison, label in (
        ("full_inits", "full-gallery starts"),
        ("streams_same_init", "sampling streams at a fixed start"),
        ("resampled_inits", "resampled initialisations"),
        ("full_vs_resampled", "full versus quarter"),
    ):
        vals = [r["cosine"] for r in stab if r["comparison"] == comparison and r["horizon"] == horizon and r["cosine"] is not None]
        missing = sum(1 for r in stab if r["comparison"] == comparison and r["horizon"] == horizon and r["cosine"] is None)
        if not vals:
            parts.append(f"{label}: no defined cosine ({missing} undefined)")
            continue
        amp = [r["amplified_rank4"] for r in stab if r["comparison"] == comparison and r["horizon"] == horizon and r["amplified_rank4"] is not None]
        sup = [r["suppressed_rank4"] for r in stab if r["comparison"] == comparison and r["horizon"] == horizon and r["suppressed_rank4"] is not None]
        gram = [r["centred_gram_cosine"] for r in stab if r["comparison"] == comparison and r["horizon"] == horizon and r["centred_gram_cosine"] is not None]
        corr = [r["correction_gram_cosine"] for r in stab if r["comparison"] == comparison and r["horizon"] == horizon and r["correction_gram_cosine"] is not None]
        parts.append(
            f"{label}: ΔM cosine median {float(np.median(vals)):.3f} "
            f"(min {float(np.min(vals)):.3f}, max {float(np.max(vals)):.3f}, undefined {missing}); "
            f"rank-4 amplified overlap median {_fmt_med(amp)}; suppressed {_fmt_med(sup)}; "
            f"centred-Gram cosine {_fmt_med(gram)}; signed correction-Gram cosine {_fmt_med(corr)}"
        )
    return "Equal-update validation-selected corrections. " + " ".join(parts) + "."


def _fmt_med(vals) -> str:
    if not vals:
        return "undefined"
    return f"{float(np.median(vals)):.3f}"


def _extended_sentence(joined) -> str:
    bits = []
    for rho in (0.1, 0.4):
        for field, name in (("excess", "excess"), ("mnn_k10", "mNN")):
            diffs = []
            for lang in LABELS:
                full = _median_metric(joined, lang, "full", rho, "selected_primary", field)
                long = _median_metric(joined, lang, "quarter", rho, "selected_final", field)
                diffs.append(long - full)
            bits.append(f"ρ={rho} {name} median of model medians {float(np.median(diffs)):+.4f}")
    return (
        "Extended quarter trajectory, validation-selected on the longer check schedule, minus full-gallery selected at T: "
        + "; ".join(bits)
        + ". This horizon has extra optimisation and extra checkpoint-selection opportunities."
    )


def _distortion_sentence(joined, stab) -> str:
    bits = []
    for rho in (0.1, 0.4):
        full_d = [r["d_total"] for r in joined if r["procedure"] == "full" and float(r["rho"]) == rho and r["checkpoint"] == "selected_primary"]
        quarter_d = [r["d_total"] for r in joined if r["procedure"] == "quarter" and float(r["rho"]) == rho and r["checkpoint"] == "selected_primary"]
        norms = [0.5 * (r["norm_left"] + r["norm_right"]) for r in stab if r["horizon"] == "equal_update_selected" and float(r["rho"]) == rho and r["comparison"] == "full_vs_resampled" and not r["nearly_zero"]]
        bits.append(
            f"ρ={rho} median selected distortion full {float(np.median(full_d)):.3f}, quarter {float(np.median(quarter_d)):.3f}"
            + (f", mean correction norm on full-versus-quarter pairs {float(np.median(norms)):.3f}" if norms else ", corrections near zero")
        )
    return (
        "Actual distortion of the validation-selected equal-update metrics: "
        + "; ".join(bits)
        + ". A higher cosine with distortion well below the budget would be a smaller correction, not a more reproducible nontrivial metric."
    )


def _consistency_sentence(joined) -> str:
    bits = []
    for rho in (0.1, 0.4):
        signs = []
        for lang in LABELS:
            delta = _median_metric(joined, lang, "quarter", rho, "selected_primary", "excess") - _median_metric(joined, lang, "full", rho, "selected_primary", "excess")
            signs.append((LABELS[lang], delta))
        pos = [name for name, delta in signs if delta > 0]
        neg = [name for name, delta in signs if delta <= 0]
        bits.append(f"ρ={rho}: higher quarter excess on {len(pos)}/6 ({', '.join(pos) or 'none'}); not higher on {', '.join(neg) or 'none'}")
    return "Model-wise equal-update selected excess. " + ". ".join(bits) + "."


def _claim(joined, stab) -> str:
    deltas = []
    for rho in (0.1, 0.4):
        for lang in LABELS:
            deltas.append(
                _median_metric(joined, lang, "quarter", rho, "selected_primary", "excess")
                - _median_metric(joined, lang, "full", rho, "selected_primary", "excess")
            )
    med = float(np.median(deltas))
    cos = [r["cosine"] for r in stab if r["comparison"] == "streams_same_init" and r["horizon"] == "equal_update_selected" and r["cosine"] is not None]
    full_cos = [r["cosine"] for r in stab if r["comparison"] == "full_inits" and r["horizon"] == "equal_update_selected" and r["cosine"] is not None]
    cos_bit = ""
    if cos and full_cos:
        cos_bit = (
            f" Sampling-stream ΔM cosine has median {float(np.median(cos)):.3f}, "
            f"against {float(np.median(full_cos)):.3f} across full-gallery starts."
        )
    direction = "does not raise" if med <= 0 else "raises"
    return (
        f"On this panel, quarter-gallery resampling {direction} exploratory excess relative to full-gallery "
        f"training at the equal-update horizon (median of the twelve model-budget differences {med:+.4f})."
        f"{cos_bit} "
        "The comparison is optimisation reproducibility on one frozen training set and one frozen PCA basis. "
        "Improved sampled-subset scores would not by themselves be evidence of better held-out alignment."
    )


def _run_name(job: dict[str, Any]) -> str:
    name = f"{job['language']}__{job['procedure']}__rho{float(job['rho']):.1f}__init{int(job['init_seed'])}"
    if job.get("sampling_seed") is not None:
        name += f"__samp{int(job['sampling_seed'])}"
    return name


def _mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def _verify_protected(out: Path) -> None:
    audit = json.loads((out / "audit.json").read_text())
    work = out.parent
    checks = {
        "metric_stability_summary": work / "metric_stability" / "summary.json",
        "metric_stability_report": work / "metric_stability" / "report.md",
    }
    for key, path in checks.items():
        previous = audit.get("protected_mtime", {}).get(key)
        current = _mtime(path)
        if previous is not None and current != previous:
            raise RuntimeError(f"protected file changed: {path}")
