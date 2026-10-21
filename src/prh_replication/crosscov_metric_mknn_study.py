"""Truncated shared-direction metrics evaluated by held-out mutual kNN."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from prh_replication.crosscov_metric_mknn import (
    build_weights,
    factorized_distance_sq,
    family_directions,
    knn_indices_stable,
    metric_factor,
    mknn_from_features,
    mknn_metric,
    quadratic_form_distance_sq,
    select_rank,
)
from prh_replication.crosscov_weighted_cka import fit_raw_and_cca
from prh_replication.crosscov_weighted_cka_study import (
    _branch_matrices,
    _fit_matched_pca,
    _load_saes,
    _prepare_features,
)
from prh_replication.io_utils import write_json
from prh_replication.registry import Paths
from prh_replication.sae_crosscov import project
from prh_replication.sae_crosscov_study import _load_panel, _verify_manifest
CONFIG_NAME = "crosscov_metric_mknn_v1.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_archived_exponents(path: Path) -> dict:
    """Return {(language, vision, branch): {alpha, beta}} for both-sided d=64."""
    out: dict = {}
    alphas: dict = {}
    betas: dict = {}
    for row in csv.DictReader(path.open()):
        if str(row["d"]) != "64" or row["side"] != "both":
            continue
        key = (row["language"], row["vision"], row["branch"])
        if row["family"] == "raw_power" and row["selected_alpha"] != "":
            alphas[key] = float(row["selected_alpha"])
        if row["family"] == "cca_reliability" and row["selected_beta"] != "":
            betas[key] = float(row["selected_beta"])
    for key in set(alphas) | set(betas):
        out[key] = {"alpha": alphas.get(key), "beta": betas.get(key)}
    return out


def _load_archived_mknn(path: Path) -> dict:
    """Map (language, vision, branch, arm_key) -> mnn for r=64 reproduction."""
    out = {}
    for row in csv.DictReader(path.open()):
        if str(row["d"]) != "64":
            continue
        key = (row["language"], row["vision"], row["branch"], row["arm"])
        out[key] = float(row["mnn_k10"])
    return out


def run(work: Path, repo: Path, force: bool = False) -> dict:
    started = time.perf_counter()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    out = paths.results / "crosscov_metric_mknn_v1"
    out.mkdir(parents=True, exist_ok=True)
    spec = repo / "results" / "crosscov_metric_mknn_v1" / "SPECIFICATION.md"
    if spec.exists():
        (out / "SPECIFICATION.md").write_text(spec.read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if (out / "summary.json").exists() and not force:
        print("crosscov_metric_mknn already complete; pass --force to rerun", flush=True)
        return json.loads((out / "summary.json").read_text())

    parent = paths.results / "crosscov_weighted_cka_v1"
    split = json.loads((parent / "split_manifest.json").read_text())
    train_ids = list(split["train_ids"])
    test_ids = list(split["test_ids"])
    fit_ids = list(split["fit_ids"])
    val_ids = list(split["val_ids"])
    id_to_train = {sid: i for i, sid in enumerate(train_ids)}
    fit_idx = np.array([id_to_train[sid] for sid in fit_ids], dtype=np.int64)
    val_idx = np.array([id_to_train[sid] for sid in val_ids], dtype=np.int64)
    assert len(fit_idx) == 1536 and len(val_idx) == 512
    assert set(fit_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    write_json(
        out / "split_manifest.json",
        {
            **{k: split[k] for k in ("fit_ids_sha256", "val_ids_sha256", "inner_split_seed", "sae_validation_caveat") if k in split},
            "fit_n": len(fit_ids),
            "val_n": len(val_ids),
            "test_n": len(test_ids),
            "source": str(parent / "split_manifest.json"),
            "rank_selection": "validation mKNN on fit-fitted projectors; exponents archived from prior CKA selection",
        },
    )

    panel = {"splits": {"train": train_ids, "test": test_ids}}
    full_panel_cfg = json.loads((parent / "config.json").read_text())
    loaded = _load_panel(paths, repo, full_panel_cfg)
    _verify_manifest(paths, repo, full_panel_cfg, loaded)
    # Ensure we use the exact archived train/test order from the weighted-CKA split manifest
    if list(loaded["splits"]["train"]) != train_ids or list(loaded["splits"]["test"]) != test_ids:
        raise RuntimeError("parent split_manifest IDs disagree with caption-reliability panel splits")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    languages = list(config["languages"])
    visions = list(config["visions"])
    prepared = _prepare_features(paths, full_panel_cfg, panel, languages, visions, fit_idx)
    sae_bundle = _load_saes(paths.results / "sae_crosscov_v1" / "sae", languages, visions, int(config["sae_seed"]), device)
    exponents = _load_archived_exponents(parent / "selected_hyperparameters.csv")
    archived_mknn = _load_archived_mknn(parent / "test_metrics.csv")
    subsample_indices = np.load(parent / "subsample_indices.npy")

    checks = _run_checks()
    write_json(out / "verification_checks.json", checks)
    if not all(v for k, v in checks.items() if k != "notes"):
        bad = {k: v for k, v in checks.items() if v is False}
        raise RuntimeError(f"verification failed: {bad}")

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    val_rows = []
    selected_rows = []
    test_rows = []
    paired_rows = []
    caption_rows = []
    null_rows = []
    subsample_rows = []
    tie_rows = []
    repro_gaps = []

    k = int(config["knn_k"])
    ranks = [int(r) for r in config["rank_grid"]]

    for language in languages:
        for vision in visions:
            print(f"pair {language} × {vision}", flush=True)
            for branch in ("original", "sae"):
                feats = _branch_matrices(prepared, sae_bundle, language, vision, branch, device)
                exp = exponents[(language, vision, branch)]
                alpha_star = exp["alpha"]
                beta_star = exp["beta"]
                if alpha_star is None or beta_star is None:
                    raise RuntimeError(f"missing archived exponents for {language}/{vision}/{branch}")

                # --- validation maps from fit only ---
                sel_lang = feats["lang_train_select"]
                sel_vis = feats["vis_train_select"]
                pca_l_v, pca_v_v, width_v = _fit_matched_pca(sel_lang[fit_idx], sel_vis[fit_idx], int(config["pca_d"]), full_panel_cfg)
                if width_v != int(config["pca_d"]):
                    raise RuntimeError(f"unsupported width {width_v}")
                a_fit = project(sel_lang[fit_idx], pca_l_v["mean"], pca_l_v["components"])
                b_fit = project(sel_vis[fit_idx], pca_v_v["mean"], pca_v_v["components"])
                a_val = project(sel_lang[val_idx], pca_l_v["mean"], pca_l_v["components"])
                b_val = project(sel_vis[val_idx], pca_v_v["mean"], pca_v_v["components"])
                maps_val = fit_raw_and_cca(a_fit, b_fit, float(config["eta"]), float(config["support_tol"]))

                # --- test maps: prefer archived full-train projectors ---
                proj_path = parent / "artifacts" / f"{branch}_{language}_{vision}_d{config['pca_d']}_projectors.npz"
                if not proj_path.exists():
                    raise FileNotFoundError(proj_path)
                proj = np.load(proj_path)
                a_tr = project(feats["lang_train"], proj["pca_lang_mean"], proj["pca_lang_components"])
                b_tr = project(feats["vis_train"], proj["pca_vis_mean"], proj["pca_vis_components"])
                a_te = project(feats["lang_test"], proj["pca_lang_mean"], proj["pca_lang_components"])
                b_te = project(feats["vis_test"], proj["pca_vis_mean"], proj["pca_vis_components"])
                # rebuild maps from full train for weights/rho consistency with archived SVD order
                maps_te = fit_raw_and_cca(a_tr, b_tr, float(config["eta"]), float(config["support_tol"]))
                # Prefer archived U,V,L for exact continuity with parent CKA run
                maps_te_arch = {
                    "svd": {"U": proj["U"], "V": proj["V"], "sigma": proj["sigma"]},
                    "cca": {"L_A": proj["L_A"], "L_B": proj["L_B"], "s": proj["s"]},
                    "rho": proj["rho"],
                }
                # sanity: sigma close
                if float(np.max(np.abs(maps_te["svd"]["sigma"] - maps_te_arch["svd"]["sigma"]))) > 1e-6:
                    # keep archived projectors; recompute rho from archived L
                    from prh_replication.sae_crosscov import pearson_cols, sample_covariances

                    cov = sample_covariances(a_tr, b_tr)
                    rho = pearson_cols(cov["A"] @ maps_te_arch["cca"]["L_A"], cov["B"] @ maps_te_arch["cca"]["L_B"])
                    maps_te_arch["rho"] = rho
                else:
                    maps_te_arch["rho"] = maps_te["rho"]

                b0_val = mknn_from_features(a_val, b_val, k=k)
                b0_te = mknn_from_features(a_te, b_te, k=k)

                for family in config["families"]:
                    l_a_v, l_b_v = family_directions(family, maps_val["svd"], maps_val["cca"])
                    w_v = build_weights(
                        family,
                        maps_val["svd"]["sigma"],
                        maps_val["rho"],
                        alpha_star,
                        beta_star,
                        float(config["sigma_floor"]),
                        float(config["rho_floor"]),
                        bool(config["normalize_weights_by_mean"]),
                    )
                    # both-sided: same weight vector on each side from shared spectrum/rho
                    curve = []
                    for rank in ranks:
                        sc = mknn_metric(a_val, b_val, l_a_v, l_b_v, w_v, w_v, rank, k=k)
                        row = {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "family": family,
                            "rank": rank,
                            "val_mknn": sc["mknn"],
                            "alpha_archived": alpha_star,
                            "beta_archived": beta_star,
                            "n_rows_with_kth_tie_left": sc["n_rows_with_kth_tie_left"],
                            "n_rows_with_kth_tie_right": sc["n_rows_with_kth_tie_right"],
                        }
                        val_rows.append(row)
                        curve.append({"rank": rank, "mknn": sc["mknn"]})
                    r_star = select_rank(curve, float(config["tie_atol_rank"]))
                    selected_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "family": family,
                            "selected_rank": r_star,
                            "val_mknn_selected": next(c["mknn"] for c in curve if c["rank"] == r_star),
                            "alpha_archived": alpha_star,
                            "beta_archived": beta_star,
                        }
                    )

                    # test evaluation with archived full-train projectors
                    l_a, l_b = family_directions(family, maps_te_arch["svd"], maps_te_arch["cca"])
                    sigma = maps_te_arch["svd"]["sigma"]
                    rho = maps_te_arch["rho"]
                    w = build_weights(
                        family,
                        sigma,
                        rho,
                        alpha_star,
                        beta_star,
                        float(config["sigma_floor"]),
                        float(config["rho_floor"]),
                        bool(config["normalize_weights_by_mean"]),
                    )
                    np.savez_compressed(
                        art / f"{branch}_{language}_{vision}_{family}_metric.npz",
                        L_A=l_a,
                        L_B=l_b,
                        w=w,
                        sigma=sigma,
                        rho=rho,
                        selected_rank=r_star,
                        alpha=alpha_star,
                        beta=beta_star,
                    )

                    sc64 = mknn_metric(a_te, b_te, l_a, l_b, w, w, 64, k=k)
                    sc_sel = mknn_metric(a_te, b_te, l_a, l_b, w, w, r_star, k=k)
                    tie_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "family": family,
                            "rank": r_star,
                            "n_rows_with_kth_tie_left": sc_sel["n_rows_with_kth_tie_left"],
                            "n_rows_with_kth_tie_right": sc_sel["n_rows_with_kth_tie_right"],
                            "n_zero_distance_offdiag_counts": sc_sel["n_zero_distance_offdiag_counts"],
                            "partition": "test_selected",
                        }
                    )

                    # optional randomized tie-break sensitivity
                    for seed in config["tie_break_seeds"]:
                        rng = np.random.Generator(np.random.PCG64(int(seed)))
                        pr_l = rng.random(a_te.shape[0])
                        pr_r = rng.random(b_te.shape[0])
                        sens = mknn_metric(a_te, b_te, l_a, l_b, w, w, r_star, k=k, priorities_left=pr_l, priorities_right=pr_r)
                        tie_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "family": family,
                                "rank": r_star,
                                "tie_break_seed": int(seed),
                                "mknn": sens["mknn"],
                                "delta_vs_stable": sens["mknn"] - sc_sel["mknn"],
                                "partition": "test_selected_random_tiebreak",
                            }
                        )

                    row = {
                        "language": language,
                        "vision": vision,
                        "branch": branch,
                        "family": family,
                        "arm": "primary",
                        "selected_rank": r_star,
                        "val_mknn": next(c["mknn"] for c in curve if c["rank"] == r_star),
                        "test_b0_mknn": b0_te["mknn"],
                        "test_r64_mknn": sc64["mknn"],
                        "test_selected_mknn": sc_sel["mknn"],
                        "delta_vs_b0": sc_sel["mknn"] - b0_te["mknn"],
                        "delta_vs_r64": sc_sel["mknn"] - sc64["mknn"],
                        "alpha_archived": alpha_star,
                        "beta_archived": beta_star,
                    }
                    test_rows.append(row)
                    paired_rows.append(row)

                    # descriptive fixed-r test curve
                    for rank in ranks:
                        sc = mknn_metric(a_te, b_te, l_a, l_b, w, w, rank, k=k)
                        test_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "family": family,
                                "selected_rank": r_star,
                                "rank_fixed": rank,
                                "val_mknn": "",
                                "test_b0_mknn": b0_te["mknn"],
                                "test_r64_mknn": sc64["mknn"],
                                "test_selected_mknn": sc["mknn"] if rank == r_star else "",
                                "test_fixed_r_mknn": sc["mknn"],
                                "delta_vs_b0": sc["mknn"] - b0_te["mknn"],
                                "delta_vs_r64": sc["mknn"] - sc64["mknn"],
                                "alpha_archived": alpha_star,
                                "beta_archived": beta_star,
                                "arm": "fixed_r_curve",
                            }
                        )

                    # reproduce archived r=64 mknn where arms exist
                    arch_arm = {
                        "raw_equal": "raw_both_alpha1_alpha1",
                        "raw_power_selected": "raw_both_selected_alpha3",  # may vary; match by loading
                        "cca_equal": "cca_both_beta0_beta0",
                        "cca_beta1": "cca_both_beta1_beta1",
                        "cca_beta_selected": None,
                    }
                    # find selected arm name dynamically
                    candidates = [
                        arm
                        for (lang, vis, br, arm), _ in archived_mknn.items()
                        if lang == language and vis == vision and br == branch and "both" in arm and "selected" in arm
                    ]
                    if family == "raw_power_selected":
                        match = [a for a in candidates if a.startswith("raw_both_selected")]
                    elif family == "cca_beta_selected":
                        match = [a for a in candidates if a.startswith("cca_both_selected")]
                    else:
                        match = [arch_arm[family]] if arch_arm[family] else []
                    for arm in match:
                        key = (language, vision, branch, arm)
                        if key in archived_mknn:
                            gap = abs(sc64["mknn"] - archived_mknn[key])
                            repro_gaps.append(
                                {
                                    "language": language,
                                    "vision": vision,
                                    "branch": branch,
                                    "family": family,
                                    "archived_arm": arm,
                                    "archived_mknn": archived_mknn[key],
                                    "recomputed_r64_mknn": sc64["mknn"],
                                    "abs_gap": gap,
                                }
                            )

                    # single-caption sensitivity
                    if feats["lang_test_positions"].ndim == 3:
                        pos_sel, pos_b0, pos_r64 = [], [], []
                        for position in range(feats["lang_test_positions"].shape[1]):
                            pos_a = project(feats["lang_test_positions"][:, position, :], proj["pca_lang_mean"], proj["pca_lang_components"])
                            s0 = mknn_from_features(pos_a, b_te, k=k)["mknn"]
                            s64 = mknn_metric(pos_a, b_te, l_a, l_b, w, w, 64, k=k)["mknn"]
                            ss = mknn_metric(pos_a, b_te, l_a, l_b, w, w, r_star, k=k)["mknn"]
                            pos_b0.append(s0)
                            pos_r64.append(s64)
                            pos_sel.append(ss)
                            caption_rows.append(
                                {
                                    "language": language,
                                    "vision": vision,
                                    "branch": branch,
                                    "family": family,
                                    "position": position,
                                    "selected_rank": r_star,
                                    "mknn_b0": s0,
                                    "mknn_r64": s64,
                                    "mknn_selected": ss,
                                    "delta_vs_b0": ss - s0,
                                    "delta_vs_r64": ss - s64,
                                }
                            )
                        caption_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "family": family,
                                "position": "mean",
                                "selected_rank": r_star,
                                "mknn_b0": float(np.mean(pos_b0)),
                                "mknn_r64": float(np.mean(pos_r64)),
                                "mknn_selected": float(np.mean(pos_sel)),
                                "delta_vs_b0": float(np.mean(pos_sel) - np.mean(pos_b0)),
                                "delta_vs_r64": float(np.mean(pos_sel) - np.mean(pos_r64)),
                            }
                        )

                    # test pairing nulls: permute vision rows AND keep IDs consistent
                    # (reindex vision features; neighbour IDs remain positional after same permutation)
                    for seed in config["shuffle_seeds"]:
                        order = np.random.Generator(np.random.PCG64(int(seed))).permutation(b_te.shape[0])
                        # Apply same permutation to vision features; language stays; compare N_lang(i) to N_vis_perm(i)
                        # where vis_perm rows are b_te[order], so image i's vision neighbour search uses shuffled embedding
                        # but we need correspondence: query i language vs query i' = order^{-1}? Spec: reindex vision
                        # queries and gallery identities together — permute vision feature rows with order, score
                        # mKNN(lang, vis[order]) which compares N_A(i) with N_{B_shuffled}(i) for the same row index i
                        # (now mismatched identities). Correct for correspondence null.
                        sc_n = mknn_metric(a_te, b_te[order], l_a, l_b, w, w, r_star, k=k)
                        null_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "family": family,
                                "selected_rank": r_star,
                                "shuffle_seed": int(seed),
                                "mknn_shuffled": sc_n["mknn"],
                                "mknn_genuine": sc_sel["mknn"],
                                "genuine_minus_shuffled": sc_sel["mknn"] - sc_n["mknn"],
                            }
                        )

                    # subsamples
                    for s_i, idx in enumerate(subsample_indices):
                        idx = np.asarray(idx, dtype=np.int64)
                        s0 = mknn_from_features(a_te[idx], b_te[idx], k=k)["mknn"]
                        s64 = mknn_metric(a_te[idx], b_te[idx], l_a, l_b, w, w, 64, k=k)["mknn"]
                        ss = mknn_metric(a_te[idx], b_te[idx], l_a, l_b, w, w, r_star, k=k)["mknn"]
                        subsample_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "family": family,
                                "subsample": int(s_i),
                                "selected_rank": r_star,
                                "mknn_b0": s0,
                                "mknn_r64": s64,
                                "mknn_selected": ss,
                                "delta_vs_b0": ss - s0,
                                "delta_vs_r64": ss - s64,
                            }
                        )

    _write_csv(out / "validation_rank_curves.csv", val_rows)
    _write_csv(out / "selected_ranks.csv", selected_rows)
    _write_csv(out / "test_metrics.csv", test_rows)
    _write_csv(out / "paired_differences.csv", [r for r in paired_rows])
    _write_csv(out / "single_caption_metrics.csv", caption_rows)
    _write_csv(out / "test_pairing_nulls.csv", null_rows)
    _write_csv(out / "subsample_metrics.csv", subsample_rows)
    _write_csv(out / "tie_diagnostics.csv", tie_rows)
    _write_csv(out / "archived_mknn_reproduction.csv", repro_gaps)

    if repro_gaps:
        max_gap = max(float(r["abs_gap"]) for r in repro_gaps)
        if max_gap > 1e-4:
            raise RuntimeError(f"archived mKNN reproduction gap too large: {max_gap}")

    summary = _summarize(config, test_rows, selected_rows, null_rows, caption_rows, tie_rows, subsample_rows, checks, repro_gaps)
    summary["seconds"] = time.perf_counter() - started
    summary["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    summary["python"] = platform.python_version()
    summary["torch"] = str(getattr(torch, "__version__", "unknown"))
    summary["device"] = device
    write_json(out / "summary.json", summary)

    code_files = [
        repo / "src/prh_replication/crosscov_metric_mknn.py",
        repo / "src/prh_replication/crosscov_metric_mknn_study.py",
        repo / "configs" / CONFIG_NAME,
    ]
    write_json(
        out / "provenance.json",
        {
            "config": config["name"],
            "parent": str(parent),
            "code_sha256": {str(p.relative_to(repo)): _sha256_file(p) for p in code_files if p.exists()},
            "split_source": str(parent / "split_manifest.json"),
            "seconds": summary["seconds"],
            "peak_rss_kb": summary["peak_rss_kb"],
            "reproduce": "PYTHONPATH=src python scripts/run_crosscov_metric_mknn.py --work /mnt/sdb1/prh-replication-work",
            "max_archived_mknn_repro_gap": max((float(r["abs_gap"]) for r in repro_gaps), default=0.0),
        },
    )
    _write_results(out / "RESULTS.md", summary, test_rows, selected_rows)
    try:
        _figures(out, val_rows, test_rows, selected_rows)
    except Exception as exc:
        (out / "figure_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({"seconds": summary["seconds"], "answers": summary["answers"]}, indent=2)[:2500], flush=True)
    return summary


def _run_checks() -> dict:
    rng = np.random.default_rng(0)
    n, d = 120, 16
    a = rng.normal(size=(n, d))
    b = a + 0.1 * rng.normal(size=(n, d))
    u, _, vt = np.linalg.svd(a.T @ b / (n - 1), full_matrices=False)
    v = vt.T
    w = np.linspace(0.5, 2.0, d)
    # 1 factorized == quadratic
    q = quadratic_form_distance_sq(a, u, w, 8)
    f = factorized_distance_sq(a, u, w, 8)
    ok1 = float(np.max(np.abs(q - f))) < 1e-9
    # 2 raw full equal reproduces Euclidean
    ones = np.ones(d)
    d0 = sq_euclidean_rows_local(a)
    d1 = factorized_distance_sq(a, u, ones, d)
    # U orthogonal => distances equal
    ok2 = float(np.max(np.abs(d0 - d1))) < 1e-9
    knn0 = knn_indices_stable(d0, 10)
    knn1 = knn_indices_stable(d1, 10)
    ok2b = bool(np.array_equal(knn0, knn1))
    # 3 global rescaling preserves neighbours
    knn_s = knn_indices_stable(factorized_distance_sq(a, u, 3.0 * w, 8), 10)
    knn_w = knn_indices_stable(factorized_distance_sq(a, u, w, 8), 10)
    ok3 = bool(np.array_equal(knn_s, knn_w))
    # 4 sign flips
    flips = rng.choice([-1.0, 1.0], size=d)
    ok4 = float(np.max(np.abs(factorized_distance_sq(a, u * flips, w, 8) - factorized_distance_sq(a, u, w, 8)))) < 1e-9
    # 5 truncation columns
    f8 = metric_factor(u, w, 8)
    ok5 = f8.shape == (d, 8) and float(np.max(np.abs(f8 - u[:, :8] * np.sqrt(w[:8])))) < 1e-15
    # 6 self excluded, exactly k
    knn = knn_indices_stable(d0, 10)
    ok6 = bool(np.all(knn != np.arange(n)[:, None])) and knn.shape[1] == 10 and len(set(knn[0].tolist())) == 10
    return {
        "quadratic_equals_factorized": ok1,
        "raw_full_equal_matches_euclidean_distances": ok2,
        "raw_full_equal_matches_neighbour_sets": ok2b,
        "global_rescale_preserves_neighbours": ok3,
        "sign_flips_preserve_distances": ok4,
        "truncation_uses_ranked_prefix": ok5,
        "neighbours_exclude_self_and_have_k": ok6,
        "validation_projectors_fit_only_by_construction": True,
        "test_ranks_from_validation_by_construction": True,
        "subsample_rebuilds_neighbours_by_construction": True,
        "notes": " Archived mKNN reproduction checked during the run against parent CSVs.",
    }


def sq_euclidean_rows_local(features: np.ndarray) -> np.ndarray:
    from prh_replication.spectral_calibration import sq_euclidean_rows

    return sq_euclidean_rows(features)


def _summarize(config, test_rows, selected_rows, null_rows, caption_rows, tie_rows, subsample_rows, checks, repro_gaps) -> dict:
    primary = [r for r in test_rows if r.get("arm") == "primary"]

    def mean_field(rows, field):
        vals = [float(r[field]) for r in rows if r.get(field) != "" and r.get(field) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    by_family = {}
    for fam in config["families"]:
        rows = [r for r in primary if r["family"] == fam]
        by_family[fam] = {
            "mean_selected_rank": mean_field(rows, "selected_rank"),
            "mean_delta_vs_b0": mean_field(rows, "delta_vs_b0"),
            "mean_delta_vs_r64": mean_field(rows, "delta_vs_r64"),
            "mean_selected_mknn": mean_field(rows, "test_selected_mknn"),
            "mean_r64_mknn": mean_field(rows, "test_r64_mknn"),
            "mean_b0_mknn": mean_field(rows, "test_b0_mknn"),
            "n": len(rows),
        }

    # equal vs weighted at selected ranks: compare raw_equal selected vs raw_power selected, etc.
    answers = {
        "truncation_improves_vs_r64": bool(mean_field(primary, "delta_vs_r64") > 0),
        "mean_delta_vs_r64": mean_field(primary, "delta_vs_r64"),
        "truncation_improves_vs_b0": bool(mean_field(primary, "delta_vs_b0") > 0),
        "mean_delta_vs_b0": mean_field(primary, "delta_vs_b0"),
        "by_family": by_family,
        "mean_selected_rank_raw_equal": by_family["raw_equal"]["mean_selected_rank"],
        "mean_selected_rank_raw_power": by_family["raw_power_selected"]["mean_selected_rank"],
        "mean_selected_rank_cca_equal": by_family["cca_equal"]["mean_selected_rank"],
        "mean_selected_rank_cca_selected": by_family["cca_beta_selected"]["mean_selected_rank"],
        "sae_minus_original_selected_mknn": _sae_minus_orig(primary),
        "mean_caption_delta_vs_b0": mean_field([r for r in caption_rows if r["position"] == "mean"], "delta_vs_b0"),
        "mean_genuine_minus_shuffled": mean_field(null_rows, "genuine_minus_shuffled"),
        "mean_tiebreak_abs_delta": float(
            np.mean([abs(float(r["delta_vs_stable"])) for r in tie_rows if r.get("partition") == "test_selected_random_tiebreak"])
        )
        if any(r.get("partition") == "test_selected_random_tiebreak" for r in tie_rows)
        else float("nan"),
        "mean_subsample_delta_sd": _subsample_sd(subsample_rows),
        "max_archived_repro_gap": max((float(r["abs_gap"]) for r in repro_gaps), default=0.0),
        "checks": checks,
    }
    # equal better than weighted?
    answers["equal_beats_power_at_selected"] = bool(
        by_family["raw_equal"]["mean_selected_mknn"] > by_family["raw_power_selected"]["mean_selected_mknn"]
    )
    answers["equal_beats_cca_selected_at_selected"] = bool(
        by_family["cca_equal"]["mean_selected_mknn"] > by_family["cca_beta_selected"]["mean_selected_mknn"]
    )
    return {"config": config["name"], "answers": answers, "n_primary_rows": len(primary)}


def _sae_minus_orig(primary):
    deltas = []
    keys = {(r["language"], r["vision"], r["family"]) for r in primary}
    for language, vision, family in sorted(keys):
        o = next((r for r in primary if r["language"] == language and r["vision"] == vision and r["family"] == family and r["branch"] == "original"), None)
        s = next((r for r in primary if r["language"] == language and r["vision"] == vision and r["family"] == family and r["branch"] == "sae"), None)
        if o and s:
            deltas.append(float(s["test_selected_mknn"]) - float(o["test_selected_mknn"]))
    return float(np.mean(deltas)) if deltas else float("nan")


def _subsample_sd(rows):
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        key = (r["language"], r["vision"], r["branch"], r["family"])
        buckets[key].append(float(r["delta_vs_b0"]))
    sds = [float(np.std(v, ddof=1)) for v in buckets.values() if len(v) > 1]
    return float(np.mean(sds)) if sds else float("nan")


def _write_results(path: Path, summary: dict, test_rows, selected_rows) -> None:
    a = summary["answers"]
    lines = [
        "# Truncated metric mKNN — RESULTS",
        "",
        "Exploratory follow-up to crosscov_weighted_cka_v1. Metric used only for neighbour search. Not confirmatory.",
        "",
        f"1. Truncation improve mKNN under learned metric (vs r=64)? mean Δ={a['mean_delta_vs_r64']}; flag={a['truncation_improves_vs_r64']}.",
        f"2. Improve vs ordinary PCA B0? mean Δ={a['mean_delta_vs_b0']}; flag={a['truncation_improves_vs_b0']}.",
        f"3. Equal vs weighted: equal beats power={a['equal_beats_power_at_selected']}; equal beats CCA-selected={a['equal_beats_cca_selected_at_selected']}.",
        f"4. Preferred ranks: raw_equal={a['mean_selected_rank_raw_equal']}, raw_power={a['mean_selected_rank_raw_power']}, cca_equal={a['mean_selected_rank_cca_equal']}, cca_selected={a['mean_selected_rank_cca_selected']}.",
        f"5. SAE−original mean selected mKNN={a['sae_minus_original_selected_mknn']}.",
        f"6. Single-caption mean Δ vs B0={a['mean_caption_delta_vs_b0']}.",
        f"7. Tie-break sensitivity mean |Δ|={a['mean_tiebreak_abs_delta']}.",
        f"8. Uncertainties: sequential CKA-then-mKNN selection; reused test cohort; grid-edge exponents; SAE pretrained on full train; pairs share endpoints.",
        "",
        f"By family: {json.dumps(a['by_family'], indent=2)}",
        "",
        f"Archived mKNN max reproduction gap={a['max_archived_repro_gap']}.",
        f"Checks: {a['checks']}.",
        f"Runtime {summary.get('seconds')} s.",
        "",
        "Reproduce: `PYTHONPATH=src python scripts/run_crosscov_metric_mknn.py --work /mnt/sdb1/prh-replication-work`",
    ]
    # main table
    lines.extend(["", "## Primary test table (selected rank)", "", "| pair | branch | family | r★ | B0 | r=64 | selected | ΔB0 | Δr64 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in test_rows:
        if r.get("arm") != "primary":
            continue
        pair = f"{r['language'][:8]}×{r['vision'][:8]}"
        lines.append(
            f"| {pair} | {r['branch']} | {r['family']} | {r['selected_rank']} | {float(r['test_b0_mknn']):.4f} | {float(r['test_r64_mknn']):.4f} | {float(r['test_selected_mknn']):.4f} | {float(r['delta_vs_b0']):+.4f} | {float(r['delta_vs_r64']):+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _figures(out, val_rows, test_rows, selected_rows) -> None:
    import matplotlib.pyplot as plt

    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    # validation curves mean over pairs for original
    plt.figure(figsize=(7, 4))
    for family in sorted({r["family"] for r in val_rows}):
        xs, ys = [], []
        for rank in sorted({int(r["rank"]) for r in val_rows}):
            chunk = [float(r["val_mknn"]) for r in val_rows if r["family"] == family and int(r["rank"]) == rank and r["branch"] == "original"]
            if chunk:
                xs.append(rank)
                ys.append(float(np.mean(chunk)))
        if xs:
            plt.plot(xs, ys, marker="o", label=family)
    # mark selected
    for r in selected_rows:
        if r["branch"] != "original":
            continue
        # only mark mean-ish: skip clutter
    plt.xlabel("retained rank r")
    plt.ylabel("validation mKNN (mean over original pairs)")
    plt.legend(fontsize=7)
    plt.title("Validation mKNN vs rank")
    plt.tight_layout()
    plt.savefig(fig / "val_mknn_vs_rank.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4))
    labels, d0, d64 = [], [], []
    for r in test_rows:
        if r.get("arm") == "fixed_r_curve":
            continue
        if r.get("test_selected_mknn") in ("", None):
            continue
        if r["branch"] != "original":
            continue
        labels.append(f"{r['family'][:10]}:{r['language'][:4]}×{r['vision'][:3]}")
        d0.append(float(r["delta_vs_b0"]))
        d64.append(float(r["delta_vs_r64"]))
    x = np.arange(len(labels))
    if len(x):
        plt.bar(x - 0.2, d0, width=0.4, label="Δ vs B0")
        plt.bar(x + 0.2, d64, width=0.4, label="Δ vs r=64")
        plt.xticks(x, labels, rotation=60, ha="right", fontsize=6)
        plt.ylabel("mKNN difference")
        plt.legend()
        plt.title("Test selected-r gains (original)")
        plt.tight_layout()
        plt.savefig(fig / "test_selected_gains.png", dpi=120)
        plt.close()
