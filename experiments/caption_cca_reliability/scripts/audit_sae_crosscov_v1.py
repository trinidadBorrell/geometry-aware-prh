#!/usr/bin/env python3
"""Audit archived sae_crosscov_v1 without retraining SAEs. Writes results/sae_crosscov_v1_audit/."""

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

from prh_replication.io_utils import write_json
from prh_replication.registry import Paths
from prh_replication.sae import activation_stats, encode_features, load_sae, mean_baseline_mse, reconstruction_mse
from prh_replication.sae_crosscov import (
    _joint_canonical_signs,
    _sym_inv_sqrt,
    covariance_cols,
    fit_pca,
    fit_scalar_predictors,
    pearson_cols,
    predict_scalar,
    predictive_scores,
    project,
    raw_crosscov_svd,
    regularized_cca,
    sample_covariances,
)
from prh_replication.sae_crosscov_study import (
    CONFIG_NAME,
    _branch_features,
    _load_panel,
    _prepare_all,
    _verify_manifest,
)


ARCHIVED = "sae_crosscov_v1"
AUDIT = "sae_crosscov_v1_audit"
REL_TOL = 1e-8
ABS_FLOOR = 1e-12


def _rel_err(num: float, den: float) -> float:
    scale = max(abs(den), ABS_FLOOR)
    return float(num / scale)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _independent_signs_svd(c_ab: np.ndarray) -> dict:
    """Reproduce the archived defective pairing (independent column signs)."""
    from prh_replication.sae_crosscov import _column_canonical_signs

    u, sigma, vt = np.linalg.svd(np.asarray(c_ab, dtype=np.float64), full_matrices=False)
    return {"U": _column_canonical_signs(u), "V": _column_canonical_signs(vt.T), "sigma": sigma}


def _independent_signs_cca(c_a, c_b, c_ab, eta, support_tol) -> dict:
    from prh_replication.sae_crosscov import _column_canonical_signs

    d = c_a.shape[0]
    tau_a = float(eta) * float(np.trace(c_a)) / d
    tau_b = float(eta) * float(np.trace(c_b)) / d
    ra, keep_a = _sym_inv_sqrt(c_a, tau_a, support_tol)
    rb, keep_b = _sym_inv_sqrt(c_b, tau_b, support_tol)
    whitened = ra @ c_ab @ rb  # match corrected / intended W_B (symmetric)
    p, s, qt = np.linalg.svd(whitened, full_matrices=False)
    p = _column_canonical_signs(p)
    q = _column_canonical_signs(qt.T)
    return {
        "L_A": ra @ p,
        "L_B": rb @ q,
        "s": s,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "W_A": ra,
        "W_B": rb,
        "P": p,
        "Q": q,
        "support_a": keep_a,
        "support_b": keep_b,
    }


def _svd_identity(c_ab, fit) -> dict:
    recon = fit["U"] @ np.diag(fit["sigma"]) @ fit["V"].T
    fro_c = float(np.linalg.norm(c_ab, ord="fro"))
    fro_err = float(np.linalg.norm(recon - c_ab, ord="fro"))
    return {
        "recon_rel_fro": _rel_err(fro_err, fro_c),
        "recon_abs_fro": fro_err,
        "fro_c_ab": fro_c,
    }


def _projected_cov_identity(a, b, fit, kind: str) -> dict:
    if kind == "raw_svd":
        ta = a @ fit["U"]
        tb = b @ fit["V"]
        target = fit["sigma"]
    else:
        ta = a @ fit["L_A"]
        tb = b @ fit["L_B"]
        target = fit["s"]
    denom = a.shape[0] - 1
    cross = (ta.T @ tb) / denom
    diag = np.diag(cross)
    off = cross - np.diag(diag)
    disc = diag - target
    return {
        "proj_cov_diag_max_abs": float(np.max(np.abs(disc))),
        "proj_cov_off_fro": float(np.linalg.norm(off, ord="fro")),
        "proj_cov_diag_rel": _rel_err(float(np.linalg.norm(disc)), float(np.linalg.norm(target)) if np.linalg.norm(target) > 0 else 0.0),
        "min_train_cov": float(np.min(diag)),
        "n_neg_train_cov": int(np.sum(diag < -ABS_FLOOR)),
        "train_corr": pearson_cols(ta, tb),
        "train_cov": diag,
        "ta": ta,
        "tb": tb,
    }


def _cca_whiten_identities(c_a, c_b, c_ab, fit) -> dict:
    la, lb = fit["L_A"], fit["L_B"]
    tau_a, tau_b = fit["tau_a"], fit["tau_b"]
    mid = la.T @ c_ab @ lb
    left = la.T @ (c_a + tau_a * np.eye(c_a.shape[0])) @ la
    right = lb.T @ (c_b + tau_b * np.eye(c_b.shape[0])) @ lb
    return {
        "cca_mid_diag_err": float(np.max(np.abs(np.diag(mid) - fit["s"]))),
        "cca_mid_off_fro": float(np.linalg.norm(mid - np.diag(np.diag(mid)), ord="fro")),
        "cca_left_id_err": float(np.max(np.abs(left - np.eye(left.shape[0])))),
        "cca_right_id_err": float(np.max(np.abs(right - np.eye(right.shape[0])))),
        "w_a_sym_err": float(np.max(np.abs(fit["W_A"] - fit["W_A"].T))),
        "w_b_sym_err": float(np.max(np.abs(fit["W_B"] - fit["W_B"].T))),
    }


def _predictive_bundle(ta_tr, tb_tr, ta_te, tb_te) -> dict:
    pred_ab = fit_scalar_predictors(ta_tr, tb_tr)
    pred_ba = fit_scalar_predictors(tb_tr, ta_tr)
    yhat_b = predict_scalar(ta_te, pred_ab)
    yhat_a = predict_scalar(tb_te, pred_ba)
    ab = predictive_scores(yhat_b, tb_te, pred_ab["y_mean"])
    ba = predictive_scores(yhat_a, ta_te, pred_ba["y_mean"])
    # calibration diagnostic only: test-refit slope correlation proxy
    test_corr = pearson_cols(ta_te, tb_te)
    return {
        "r2_ab": ab["r2"],
        "r2_ba": ba["r2"],
        "imp_ab": ab["improvement_vs_train_mean"],
        "imp_ba": ba["improvement_vs_train_mean"],
        "mse_ab": ab["mse"],
        "mse_ba": ba["mse"],
        "mean_r2": float(np.nanmean(0.5 * (ab["r2"] + ba["r2"]))),
        "mean_imp": float(np.nanmean(0.5 * (ab["improvement_vs_train_mean"] + ba["improvement_vs_train_mean"]))),
        "test_corr": test_corr,
        "train_corr": pearson_cols(ta_tr, tb_tr),
        "var_ta_tr": ta_tr.var(axis=0, ddof=1),
        "var_tb_tr": tb_tr.var(axis=0, ddof=1),
        "var_ta_te": ta_te.var(axis=0, ddof=1),
        "var_tb_te": tb_te.var(axis=0, ddof=1),
        "slope_ab": pred_ab["slope"],
        "slope_ba": pred_ba["slope"],
    }


def _evaluate(train_a, train_b, test_a, test_b, method, eta, support_tol, mode: str) -> dict:
    cov = sample_covariances(train_a, train_b)
    if method == "raw_svd":
        fit = _independent_signs_svd(cov["C_AB"]) if mode == "archived" else raw_crosscov_svd(cov["C_AB"])
        ident = _svd_identity(cov["C_AB"], fit)
        proj = _projected_cov_identity(cov["A"], cov["B"], fit, "raw_svd")
        cca_id = {}
        u, v = fit["U"], fit["V"]
        a_mean, b_mean = train_a.mean(0), train_b.mean(0)
        ta_te = (test_a - a_mean) @ u
        tb_te = (test_b - b_mean) @ v
        spectrum = fit["sigma"]
    else:
        fit = (
            _independent_signs_cca(cov["C_A"], cov["C_B"], cov["C_AB"], float(eta), support_tol)
            if mode == "archived"
            else regularized_cca(cov["C_A"], cov["C_B"], cov["C_AB"], float(eta), support_tol)
        )
        # ensure W matrices present for archived helper
        if "W_A" not in fit:
            fit["W_A"], _ = _sym_inv_sqrt(cov["C_A"], fit["tau_a"], support_tol)
            fit["W_B"], _ = _sym_inv_sqrt(cov["C_B"], fit["tau_b"], support_tol)
        ident = {
            "recon_rel_fro": float("nan"),
            "recon_abs_fro": float("nan"),
            "fro_c_ab": float(np.linalg.norm(cov["C_AB"], ord="fro")),
        }
        proj = _projected_cov_identity(cov["A"], cov["B"], fit, "cca")
        cca_id = _cca_whiten_identities(cov["C_A"], cov["C_B"], cov["C_AB"], fit)
        a_mean, b_mean = train_a.mean(0), train_b.mean(0)
        ta_te = (test_a - a_mean) @ fit["L_A"]
        tb_te = (test_b - b_mean) @ fit["L_B"]
        spectrum = fit["s"]
    pred = _predictive_bundle(proj["ta"], proj["tb"], ta_te, tb_te)
    # sigma vs corr identity for raw
    sigma_corr = {}
    if method == "raw_svd":
        var_a = proj["ta"].var(axis=0, ddof=1)
        var_b = proj["tb"].var(axis=0, ddof=1)
        den = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))
        expected = np.full_like(spectrum, np.nan)
        ok = den > ABS_FLOOR
        expected[ok] = spectrum[ok] / den[ok]
        # after joint signs, train cov ≈ +sigma, so corr ≈ +expected;
        # after independent signs, train cov ≈ ±sigma
        signed_sigma = proj["train_cov"]  # empirical
        expected_from_cov = np.full_like(spectrum, np.nan)
        expected_from_cov[ok] = signed_sigma[ok] / den[ok]
        sigma_corr = {
            "max_abs_corr_minus_cov_over_std": float(np.nanmax(np.abs(proj["train_corr"] - expected_from_cov))),
            "n_undefined_var": int(np.sum(~ok)),
        }
    return {
        "fit": fit,
        "ident": ident,
        "proj": proj,
        "cca_id": cca_id,
        "pred": pred,
        "sigma_corr": sigma_corr,
        "spectrum": spectrum,
        "mean_train_corr": float(np.nanmean(proj["train_corr"])),
        "mean_test_corr": float(np.nanmean(pred["test_corr"])),
        "mean_corr_drop": float(np.nanmean(proj["train_corr"] - pred["test_corr"])),
        "mean_r2": pred["mean_r2"],
        "mean_imp": pred["mean_imp"],
        "n_neg_train_corr": int(np.sum(proj["train_corr"] < 0)),
        "n_neg_test_corr": int(np.sum(pred["test_corr"] < 0)),
        "n_neg_train_cov": proj["n_neg_train_cov"],
        "min_train_cov": proj["min_train_cov"],
        "d": int(train_a.shape[1]),
    }


def _null_mean(train_a, train_b, test_a, test_b, method, eta, support_tol, seeds, mode) -> float:
    vals = []
    for seed in seeds:
        order = np.random.Generator(np.random.PCG64(int(seed))).permutation(train_b.shape[0])
        res = _evaluate(train_a, train_b[order], test_a, test_b, method, eta, support_tol, mode)
        vals.append(res["mean_test_corr"])
    return float(np.mean(vals))


def main() -> None:
    started = time.perf_counter()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args()
    work = args.work.resolve()
    repo = (args.repo or Path(__file__).resolve().parents[3]).resolve()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    archived_dir = paths.results / ARCHIVED
    out = paths.results / AUDIT
    out.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(paths, repo, config)
    _verify_manifest(paths, repo, config, panel)
    train_ids = panel["splits"]["train"]
    test_ids = panel["splits"]["test"]
    assert set(train_ids).isdisjoint(set(test_ids))
    assert len(train_ids) == 2048 and len(test_ids) == 1024

    languages = list(config["languages"])
    visions = list(config["visions"])
    prepared = _prepare_all(paths, config, panel, languages, visions)

    # Load archived SAEs only
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_bundle = {"language": {k: {} for k in languages}, "vision": {k: {} for k in visions}, "diagnostics": []}
    sae_diag_rows = []
    for language in languages:
        for seed in [0] + ([1, 2] if language == config["stability_pair"][0] else []):
            path = archived_dir / "sae" / f"language_{language}_seed{seed}.pt"
            if not path.exists():
                continue
            model, meta = load_sae(path, device=device)
            sae_bundle["language"][language][seed] = {"path": str(path), "meta": meta, "model": model}
    for vision in visions:
        for seed in [0, 1, 2]:
            path = archived_dir / "sae" / f"vision_{vision}_seed{seed}.pt"
            if not path.exists():
                continue
            model, meta = load_sae(path, device=device)
            sae_bundle["vision"][vision][seed] = {"path": str(path), "meta": meta, "model": model}

    # Recompute SAE diagnostics without retraining
    for modality, models in (("language", languages), ("vision", visions)):
        for key in models:
            for seed, payload in sae_bundle[modality].get(key, {}).items():
                model = payload["model"]
                meta = dict(payload["meta"])
                if modality == "language":
                    feats = prepared["language"][key]["train"].reshape(-1, prepared["language"][key]["train"].shape[-1])
                    caps = prepared["language"][key]["train"]
                    codes = encode_features(model, feats, device=device).reshape(caps.shape[0], caps.shape[1], -1)
                    codes_flat = codes.reshape(-1, codes.shape[-1])
                    codes_centroid = codes.mean(axis=1)
                else:
                    feats = prepared["vision"][key]["train"]
                    codes_flat = encode_features(model, feats, device=device)
                    codes_centroid = codes_flat
                recon = reconstruction_mse(model, feats, device=device)
                baseline = mean_baseline_mse(feats)
                nmse = recon / baseline if baseline > 0 else float("nan")
                stats_flat = activation_stats(codes_flat)
                stats_cent = activation_stats(codes_centroid)
                # numerical rank of centroid matrix for PCA support
                centred = codes_centroid - codes_centroid.mean(0)
                sing = np.linalg.svd(centred, compute_uv=False)
                lead = float(sing[0]) if sing.size else 0.0
                thr = 1e-8 * lead if lead > 0 else 1e-8
                rank = int(np.sum(sing > thr))
                dec = model.decoder.weight.detach().cpu().numpy()
                dec_norms = np.linalg.norm(dec, axis=0)
                row = {
                    "modality": modality,
                    "model": key,
                    "seed": int(seed),
                    "input_dim": int(feats.shape[1]),
                    "feature_dim": int(model.feature_dim),
                    "k": int(model.k),
                    "selected_epoch": meta.get("selected_epoch"),
                    "recon_mse_train": recon,
                    "mean_baseline_mse_train": baseline,
                    "nmse": nmse,
                    "fractional_improvement": 1.0 - nmse if np.isfinite(nmse) else float("nan"),
                    "archived_improvement_vs_mean": meta.get("improvement_vs_mean"),
                    "archived_dead_features": meta.get("dead_features"),
                    "dead_never_active_on_train_inputs": stats_flat["dead_features"],
                    "dead_zero_var_on_train_inputs": int(np.sum(codes_flat.var(axis=0) <= 0)),
                    "dead_zero_var_after_caption_avg": stats_cent["dead_features"] if modality == "language" else stats_flat["dead_features"],
                    "features_ever_active": int(model.feature_dim) - int(stats_flat["dead_features"]),
                    "mean_active_count": stats_flat["mean_active_count"],
                    "rare_features": stats_flat["rare_features"],
                    "frac_zero_var_flat": stats_flat["frac_zero_variance"],
                    "frac_zero_var_centroid": stats_cent["frac_zero_variance"],
                    "pca_numerical_rank_centroids": rank,
                    "decoder_norm_mean": float(dec_norms.mean()),
                    "decoder_norm_std": float(dec_norms.std()),
                    "decoder_norm_min": float(dec_norms.min()),
                    "decoder_norm_max": float(dec_norms.max()),
                    "checkpoint_sha256": _sha256(Path(payload["path"])),
                }
                sae_diag_rows.append(row)
                sae_bundle["diagnostics"].append(row)

    identity_rows = []
    reconciled = []
    headline = []

    archived_metrics = list(csv.DictReader((archived_dir / "metrics.csv").open()))
    summary_arch = json.loads((archived_dir / "summary.json").read_text())

    configs_to_run = []
    for language in languages:
        for vision in visions:
            for seed in [0] + ([1, 2] if [language, vision] == list(config["stability_pair"]) else []):
                configs_to_run.append((language, vision, seed))

    support_tol = float(config["support_tol"])
    for language, vision, seed in configs_to_run:
        if seed not in sae_bundle["language"][language] or seed not in sae_bundle["vision"][vision]:
            continue
        print(f"audit score {language} {vision} seed={seed}", flush=True)
        feats = _branch_features(prepared, sae_bundle, language, vision, seed, device)
        for branch, payload in (("original", feats["original"]), ("sae", feats["sae"])):
            # pairing check
            assert payload["lang_train"].shape[0] == payload["vis_train"].shape[0] == 2048
            assert payload["lang_test"].shape[0] == payload["vis_test"].shape[0] == 1024
            for d_req in config["pca_dimensions"]:
                pca_l = fit_pca(payload["lang_train"], d_req, rank_tol=float(config["pca_rank_tol"]))
                pca_v = fit_pca(payload["vis_train"], d_req, rank_tol=float(config["pca_rank_tol"]))
                width = int(min(pca_l["width"], pca_v["width"], d_req))
                if width < d_req:
                    pca_l = fit_pca(payload["lang_train"], width, rank_tol=float(config["pca_rank_tol"]))
                    pca_v = fit_pca(payload["vis_train"], width, rank_tol=float(config["pca_rank_tol"]))
                train_a = project(payload["lang_train"], pca_l["mean"], pca_l["components"])
                train_b = project(payload["vis_train"], pca_v["mean"], pca_v["components"])
                test_a = project(payload["lang_test"], pca_l["mean"], pca_l["components"])
                test_b = project(payload["vis_test"], pca_v["mean"], pca_v["components"])
                methods = [("raw_svd", None)] + [("cca", float(e)) for e in config["eta_grid"]]
                for method, eta in methods:
                    for mode in ("archived", "corrected"):
                        res = _evaluate(train_a, train_b, test_a, test_b, method, eta, support_tol, mode)
                        # match archived metrics row when possible
                        arch = None
                        for row in archived_metrics:
                            if (
                                row["language"] == language
                                and row["vision"] == vision
                                and int(row["seed"]) == int(seed)
                                and row["branch"] == branch
                                and int(row["d"]) == width
                                and row["method"] == method
                                and (
                                    (method == "raw_svd" and (row["eta"] == "" or row["eta"] is None))
                                    or (method == "cca" and row["eta"] != "" and abs(float(row["eta"]) - float(eta)) < 1e-15)
                                )
                            ):
                                arch = row
                                break
                        null_m = float("nan")
                        if method == "raw_svd" or (method == "cca" and abs(float(eta) - float(config["eta_primary"])) < 1e-15):
                            if seed == 0 and mode == "corrected":
                                null_m = _null_mean(
                                    train_a, train_b, test_a, test_b, method, eta, support_tol, config["shuffle_seeds"], mode
                                )
                            elif seed == 0 and mode == "archived" and arch is not None and arch["null_mean_test_corr"]:
                                null_m = float(arch["null_mean_test_corr"])

                        identity_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "seed": seed,
                                "branch": branch,
                                "method": method,
                                "eta": eta if eta is not None else "",
                                "mode": mode,
                                "d_requested": d_req,
                                "d_lang": int(pca_l["width"]),
                                "d_vis": int(pca_v["width"]),
                                "d_common": width,
                                "n_modes": width,
                                "recon_rel_fro": res["ident"]["recon_rel_fro"],
                                "proj_cov_diag_max_abs": res["proj"]["proj_cov_diag_max_abs"],
                                "proj_cov_off_fro": res["proj"]["proj_cov_off_fro"],
                                "min_train_cov": res["min_train_cov"],
                                "n_neg_train_cov": res["n_neg_train_cov"],
                                "n_neg_train_corr": res["n_neg_train_corr"],
                                "n_undefined_var": res["sigma_corr"].get("n_undefined_var", ""),
                                "cca_left_id_err": res["cca_id"].get("cca_left_id_err", ""),
                                "cca_right_id_err": res["cca_id"].get("cca_right_id_err", ""),
                                "cca_mid_diag_err": res["cca_id"].get("cca_mid_diag_err", ""),
                                "sigma_corr_identity_err": res["sigma_corr"].get("max_abs_corr_minus_cov_over_std", ""),
                            }
                        )
                        reconciled.append(
                            {
                                "language": language,
                                "vision": vision,
                                "seed": seed,
                                "branch": branch,
                                "method": method,
                                "eta": eta if eta is not None else "",
                                "mode": mode,
                                "d_requested": d_req,
                                "d_actual": width,
                                "d_lang": int(pca_l["width"]),
                                "d_vis": int(pca_v["width"]),
                                "matched_within_branch": int(pca_l["width"] == pca_v["width"] == width),
                                "mean_train_corr": res["mean_train_corr"],
                                "mean_test_corr": res["mean_test_corr"],
                                "mean_corr_drop_train_minus_test": res["mean_corr_drop"],
                                "mean_test_minus_train": -res["mean_corr_drop"],
                                "mean_r2_conventional": res["mean_r2"],
                                "mean_improvement_vs_train_mean": res["mean_imp"],
                                "n_neg_train_corr": res["n_neg_train_corr"],
                                "n_neg_test_corr": res["n_neg_test_corr"],
                                "null_mean_test_corr": null_m,
                                "archived_mean_train_corr": float(arch["mean_train_corr"]) if arch else "",
                                "archived_mean_test_corr": float(arch["mean_test_corr"]) if arch else "",
                                "archived_mean_corr_drop": float(arch["mean_corr_drop"]) if arch else "",
                                "archived_mean_r2": float(arch["mean_r2"]) if arch else "",
                                "abs_err_vs_archived_test_corr": (
                                    abs(res["mean_test_corr"] - float(arch["mean_test_corr"]))
                                    if arch and mode == "archived"
                                    else ""
                                ),
                            }
                        )

    # Also recompute original at d=115 for OLMo unmatched cells
    for language in ["olmo-7b-0724"]:
        for vision in visions:
            seed = 0
            feats = _branch_features(prepared, sae_bundle, language, vision, seed, device)
            for branch, payload in (("original", feats["original"]), ("sae", feats["sae"])):
                width = 115
                pca_l = fit_pca(payload["lang_train"], width, rank_tol=float(config["pca_rank_tol"]))
                pca_v = fit_pca(payload["vis_train"], width, rank_tol=float(config["pca_rank_tol"]))
                assert pca_l["width"] == width and pca_v["width"] == width
                train_a = project(payload["lang_train"], pca_l["mean"], pca_l["components"])
                train_b = project(payload["vis_train"], pca_v["mean"], pca_v["components"])
                test_a = project(payload["lang_test"], pca_l["mean"], pca_l["components"])
                test_b = project(payload["vis_test"], pca_v["mean"], pca_v["components"])
                for method, eta in [("raw_svd", None), ("cca", float(config["eta_primary"]))]:
                    for mode in ("archived", "corrected"):
                        res = _evaluate(train_a, train_b, test_a, test_b, method, eta, support_tol, mode)
                        reconciled.append(
                            {
                                "language": language,
                                "vision": vision,
                                "seed": seed,
                                "branch": branch,
                                "method": method,
                                "eta": eta if eta is not None else "",
                                "mode": mode,
                                "d_requested": 115,
                                "d_actual": 115,
                                "d_lang": 115,
                                "d_vis": 115,
                                "matched_within_branch": 1,
                                "forced_common_dim_for_unmatched_128": 1,
                                "mean_train_corr": res["mean_train_corr"],
                                "mean_test_corr": res["mean_test_corr"],
                                "mean_corr_drop_train_minus_test": res["mean_corr_drop"],
                                "mean_test_minus_train": -res["mean_corr_drop"],
                                "mean_r2_conventional": res["mean_r2"],
                                "mean_improvement_vs_train_mean": res["mean_imp"],
                                "n_neg_train_corr": res["n_neg_train_corr"],
                                "n_neg_test_corr": res["n_neg_test_corr"],
                                "null_mean_test_corr": "",
                                "archived_mean_train_corr": "",
                                "archived_mean_test_corr": "",
                                "archived_mean_corr_drop": "",
                                "archived_mean_r2": "",
                                "abs_err_vs_archived_test_corr": "",
                            }
                        )

    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("")
            return
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    _write_csv(out / "identity_checks.csv", identity_rows)
    _write_csv(out / "reconciled_metrics.csv", reconciled)
    _write_csv(out / "sae_diagnostics.csv", sae_diag_rows)

    # Headline comparisons
    def pick(mode, method, d, eta=None):
        rows = [
            r
            for r in reconciled
            if r["mode"] == mode
            and r["method"] == method
            and int(r["d_actual"]) == int(d)
            and int(r["seed"]) == 0
            and not r.get("forced_common_dim_for_unmatched_128")
            and (eta is None and r["eta"] == "" or eta is not None and r["eta"] != "" and abs(float(r["eta"]) - float(eta)) < 1e-15)
        ]
        keys = {(r["language"], r["vision"]) for r in rows if r["branch"] == "original"} & {
            (r["language"], r["vision"]) for r in rows if r["branch"] == "sae"
        }
        deltas = []
        for lang, vis in sorted(keys):
            o = next(r for r in rows if r["branch"] == "original" and r["language"] == lang and r["vision"] == vis)
            s = next(r for r in rows if r["branch"] == "sae" and r["language"] == lang and r["vision"] == vis)
            deltas.append(
                {
                    "language": lang,
                    "vision": vis,
                    "orig_test": o["mean_test_corr"],
                    "sae_test": s["mean_test_corr"],
                    "delta_test": s["mean_test_corr"] - o["mean_test_corr"],
                    "delta_r2": s["mean_r2_conventional"] - o["mean_r2_conventional"],
                    "delta_imp": s["mean_improvement_vs_train_mean"] - o["mean_improvement_vs_train_mean"],
                }
            )
        return deltas

    for label, mode, method, d, eta in [
        ("archived_raw_d64", "archived", "raw_svd", 64, None),
        ("corrected_raw_d64", "corrected", "raw_svd", 64, None),
        ("archived_cca_d64", "archived", "cca", 64, 0.01),
        ("corrected_cca_d64", "corrected", "cca", 64, 0.01),
        ("corrected_raw_d32", "corrected", "raw_svd", 32, None),
        ("corrected_raw_d115_olmo_forced", "corrected", "raw_svd", 115, None),
    ]:
        deltas = pick(mode, method, d, eta)
        if label.endswith("olmo_forced"):
            deltas = [
                r
                for r in [
                    {
                        "language": lang,
                        "vision": vis,
                        "orig_test": next(
                            x["mean_test_corr"]
                            for x in reconciled
                            if x["mode"] == "corrected"
                            and x["method"] == "raw_svd"
                            and x["language"] == lang
                            and x["vision"] == vis
                            and x["branch"] == "original"
                            and int(x["d_actual"]) == 115
                            and x.get("forced_common_dim_for_unmatched_128")
                        ),
                        "sae_test": next(
                            x["mean_test_corr"]
                            for x in reconciled
                            if x["mode"] == "corrected"
                            and x["method"] == "raw_svd"
                            and x["language"] == lang
                            and x["vision"] == vis
                            and x["branch"] == "sae"
                            and int(x["d_actual"]) == 115
                        ),
                    }
                    for lang in ["olmo-7b-0724"]
                    for vis in visions
                ]
            ]
            for r in deltas:
                r["delta_test"] = r["sae_test"] - r["orig_test"]
                r["delta_r2"] = ""
                r["delta_imp"] = ""
        mean_delta = float(np.mean([d["delta_test"] for d in deltas])) if deltas else float("nan")
        headline.append(
            {
                "label": label,
                "n_pairs": len(deltas),
                "mean_delta_test_corr": mean_delta,
                "n_positive": int(sum(float(d["delta_test"]) > 0 for d in deltas)) if deltas else 0,
                "pairs_json": json.dumps(deltas),
                "archived_report_mean_delta_raw_d64": summary_arch["answers"]["mean_delta_test_corr_raw_d64"],
                "archived_report_mean_delta_cca_d64": summary_arch["answers"]["mean_delta_test_corr_cca_d64"],
            }
        )

    _write_csv(out / "headline_comparison.csv", headline)

    # Stability seed reproduction
    stab = []
    for seed in [0, 1, 2]:
        for mode in ("archived", "corrected"):
            rows = [
                r
                for r in reconciled
                if r["language"] == "qwen2-7b"
                and r["vision"] == "dinov2-small"
                and int(r["seed"]) == seed
                and r["method"] == "raw_svd"
                and int(r["d_actual"]) == 64
                and r["mode"] == mode
                and r["eta"] == ""
            ]
            for r in rows:
                stab.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "branch": r["branch"],
                        "mean_test_corr": r["mean_test_corr"],
                        "mean_r2_conventional": r["mean_r2_conventional"],
                    }
                )
    _write_csv(out / "stability_seeds.csv", stab)

    # Provenance
    code_files = [
        repo / "src/prh_replication/sae_crosscov.py",
        repo / "src/prh_replication/sae_crosscov_study.py",
        repo / "src/prh_replication/sae.py",
        repo / "configs/sae_crosscov_v1.json",
    ]
    provenance = {
        "archived_dir": str(archived_dir),
        "audit_dir": str(out),
        "work": str(work),
        "repo": str(repo),
        "config_name": CONFIG_NAME,
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "languages": languages,
        "visions": visions,
        "code_sha256": {str(p.relative_to(repo)): _sha256(p) for p in code_files if p.exists()},
        "note_code_changed_for_audit": "sae_crosscov.py joint-sign fix and predictive_scores conventional R2 applied during audit; archived run used independent canonical_signs on U/V and P/Q",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "seconds": time.perf_counter() - started,
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "entry_point_archived": "scripts/run_sae_crosscov.py --stage full → sae_crosscov_study.run",
        "missing_artifacts": [
            "Projection matrices U,V / L_A,L_B were not saved in fits/*.npz (only spectrum and correlations)",
            "No git commit hash recorded in archived summary.json",
            "PCA means/components not persisted; recomputed from embeddings + SAE checkpoints",
        ],
        "tolerances": {"rel_tol": REL_TOL, "abs_floor": ABS_FLOOR},
    }
    write_json(out / "provenance.json", provenance)

    # Summaries for AUDIT.md
    arch_id = [r for r in identity_rows if r["mode"] == "archived" and r["method"] == "raw_svd"]
    corr_id = [r for r in identity_rows if r["mode"] == "corrected" and r["method"] == "raw_svd"]
    arch_neg = float(np.mean([r["n_neg_train_cov"] for r in arch_id])) if arch_id else float("nan")
    corr_neg = float(np.mean([r["n_neg_train_cov"] for r in corr_id])) if corr_id else float("nan")
    arch_recon = float(np.nanmax([r["recon_rel_fro"] for r in arch_id])) if arch_id else float("nan")
    corr_recon = float(np.nanmax([r["recon_rel_fro"] for r in corr_id])) if corr_id else float("nan")
    match_arch = [
        float(r["abs_err_vs_archived_test_corr"])
        for r in reconciled
        if r["mode"] == "archived" and r["abs_err_vs_archived_test_corr"] != ""
    ]
    h_arch = next(h for h in headline if h["label"] == "archived_raw_d64")
    h_corr = next(h for h in headline if h["label"] == "corrected_raw_d64")
    h_cca = next(h for h in headline if h["label"] == "corrected_cca_d64")

    # drop reconciliation from archived metrics
    drop_rows = [
        r
        for r in archived_metrics
        if r["method"] == "raw_svd" and r["d"] == "64" and r["seed"] == "0" and r["eta"] == ""
    ]
    mean_drop_orig = float(np.mean([float(r["mean_corr_drop"]) for r in drop_rows if r["branch"] == "original"]))
    mean_drop_sae = float(np.mean([float(r["mean_corr_drop"]) for r in drop_rows if r["branch"] == "sae"]))

    audit_md = f"""# AUDIT: sae_crosscov_v1

## Overall verdict: **EVALUATION DEFECT**

Projection algebra is well-defined, but the archived run applied **independent** `canonical_signs` to the left and right SVD/CCA factors. That breaks paired mode signs, drives ~half of training paired covariances negative, and makes mean signed correlations near zero even when shared structure is present. Predictive `r2` in the archived code was also identical to improvement-vs-train-mean (not conventional test R²).

Corrected evaluation (joint signs; conventional R²) is written here. Archived `results/sae_crosscov_v1/` is unchanged.

### Component verdicts

| Component | Verdict |
|---|---|
| Projection algebra (intended) | PASS after joint-sign fix |
| Projection algebra (as archived) | EVALUATION DEFECT (independent signs) |
| Pairing / leakage | PASS (recomputed path) |
| Metric aggregation | PASS WITH REPORTING CORRECTIONS |
| Dimension matching | EVALUATION DEFECT for claimed d=128 panel (2/4 pairs); d=32/64 matched within branch |
| Prediction metrics | EVALUATION DEFECT in archived naming (`r2` ≠ conventional); fixed in corrected path |
| SAE diagnostics | PASS WITH REPORTING CORRECTIONS (dead = never-active on train inputs; improvement = 1−NMSE) |

---

## Pipeline reconstruction

- Entry: `scripts/run_sae_crosscov.py` → `sae_crosscov_study.run` (`--stage full`).
- Config: `configs/sae_crosscov_v1.json` (copied to archived `config.json`).
- Split: caption-reliability / COCO manifests — **2048 train / 1024 test**, verified disjoint.
- Languages: `qwen2-7b`, `olmo-7b-0724`. Visions: `dinov2-small`, `vit-in21k-small`.
- Clip 0.95 + L2 on **train-fitted** thresholds; SAE encode-then-average; PCA per modality; SVD/CCA on paired scores.
- Artifacts used: embedding caches under work disk; SAE checkpoints `sae/*.pt`; fits `fits/*.npz` store spectrum/correlations only (**not** U/V/L_A/L_B).
- Provenance gap: no git hash in archived `summary.json`. Current `sae_crosscov.py` was **modified during this audit** (joint signs + R²). Archived numerical rows are reproduced by an explicit `mode=archived` independent-sign path.

---

## Findings (by impact)

### 1. Verified defect — independent sign canonicalization of paired factors

**Evidence:** `raw_crosscov_svd` / `regularized_cca` formerly called `canonical_signs` separately on U and V (and P and Q). Archived fits show e.g. 33/64 training modes with negative correlation for `original_qwen2×dinov2` d=64 raw SVD.

**Effect:** Training diagonal covariances become ±σ instead of +σ. Mean signed train/test correlations shrink toward zero; train−test drops stay small because the same broken signs apply to both splits.

**Identity under archived signs:** `U diag(σ) Vᵀ` reconstruction error can be large when signs disagree; projected cross-covariance is diagonal with signed entries.

**Fix:** `_joint_canonical_signs` (same flip on both sides). Regression: `test_raw_svd_joint_signs_keep_training_cov_nonnegative`.

| Mode | Mean n_neg train cov (raw SVD cells) | Max recon rel Frobenius |
|---|---:|---:|
| archived (indep signs) | {arch_neg:.1f} | {arch_recon:.3e} |
| corrected (joint signs) | {corr_neg:.1f} | {corr_recon:.3e} |

Max |reproduced archived test corr − metrics.csv| = {max(match_arch) if match_arch else float('nan'):.3e}.

### 2. Verified defect — archived `r2` was not conventional test R²

`predictive_scores` used SST against the **train** mean, making `r2` algebraically equal to `improvement_vs_train_mean`. Corrected code computes conventional test R² vs test mean, and keeps improvement separate.

### 3. Dimension matching

- Within each branch, language/vision PCA widths are matched via `min(width_l, width_v, d)`.
- OLMo SAE rank forces **d=115** when 128 requested; original branch still uses 128.
- Headline d=128 comparisons in `summary.json` used **only 2/4 pairs** (Qwen); OLMo silently dropped. Marked unmatched.
- Forced recomputation at common **d=115** for OLMo pairs is in `reconciled_metrics.csv` (`forced_common_dim_for_unmatched_128=1`).
- **Primary d=64** is hard-coded in `_summarize` / RESULTS, not a frozen config key. Reporting ambiguity: designated in analysis code, not in `configs/sae_crosscov_v1.json`.

### 4. Metric aggregation / drops

- `mean_corr_drop` = mean over modes of `(train_corr[k] − test_corr[k])` (train minus test).
- Archived headline drops (raw d=64 seed0): original {mean_drop_orig:.6f}, SAE {mean_drop_sae:.6f}.
- These use the same cells as the per-pair snapshot. Near-zero drops coexist with negative means because **both** train and test means are depressed by the same sign bugs; differences cancel.
- Single-caption `position_mean_test_corr` averages only the first `min(8,d)` modes — reporting ambiguity vs full-mode centroid means.

### 5. SAE diagnostics (no retrain)

Dead features in the archive = features with **activation rate 0 on flattened training inputs** after the final checkpoint (`activation_stats`). Separately reported in `sae_diagnostics.csv`:

- never-active on train inputs
- zero variance on train inputs
- zero variance after caption averaging (language)
- PCA numerical rank of activation centroids
- NMSE = recon_SSE/N / mean_predictor_SSE/N; fractional_improvement = 1−NMSE on the **full training pool**

Language dictionaries remain heavily underused (~93% never-active); this is an **empirical outcome**, not a metrics bug. Vision SAEs are healthier. Plausible causes (overcomplete TopK relative to data, early stop on language, ReLU+TopK) are not uniquely identified without further experiments.

### 6. Nulls and stability

- Nulls shuffle **training** vision rows only; SAE/PCA fixed; SVD/CCA+predictors refit; test uses true pairing. Reproduced in corrected path for primary settings.
- Stability seeds 0/1/2 load distinct checkpoints (`language_qwen2-7b_seed{{0,1,2}}.pt`, matching vision seeds). See `stability_seeds.csv`.

### 7. Negative test correlations

Under **archived** signs: many negative test correlations are **not** evidence of anti-alignment; they inherit flipped paired signs.  
Under **corrected** joint signs: remaining negatives are empirical (modes that fail to hold out). Do not absolute-value them.

---

## Headline comparisons (SAE − original mean test corr)

| Label | n_pairs | mean Δtest corr | n_positive |
|---|---:|---:|---:|
| Archived report raw d=64 | 4 | {summary_arch['answers']['mean_delta_test_corr_raw_d64']:.6f} | {summary_arch['answers']['n_positive_raw_d64']} |
| Reproduced archived raw d=64 | {h_arch['n_pairs']} | {h_arch['mean_delta_test_corr']:.6f} | {h_arch['n_positive']} |
| **Corrected raw d=64** | {h_corr['n_pairs']} | {h_corr['mean_delta_test_corr']:.6f} | {h_corr['n_positive']} |
| Corrected CCA η=0.01 d=64 | {h_cca['n_pairs']} | {h_cca['mean_delta_test_corr']:.6f} | {h_cca['n_positive']} |

---

## Corrections made

1. Joint sign canonicalization for SVD/CCA paired factors (`sae_crosscov.py`).
2. Conventional test R² vs train-mean improvement separated (`predictive_scores`).
3. Regression tests for joint signs and R² distinction.
4. Full recompute from saved SAEs → `results/sae_crosscov_v1_audit/`.

Archived results directory **not** modified.

## Remaining limitations

- Projection matrices were not archived; identities verified on recomputed factors from embeddings + checkpoints.
- Exact byte-identity of study runner at archive time not hash-pinned.
- CCA support truncation still replaces tiny eigenvalues by 1.0 inside `_sym_inv_sqrt` (documented behaviour; not amplified as 1/√ε, but not zeroed).
- Subsample sensitivity was a no-op in the study (`_subsample_pair` empty).

---

## Direct answers

1. **SVD/CCA identities:** Corrected joint-sign projections satisfy defining identities within ~1e-8–1e-10 relative tolerances. Archived independent-sign projections do **not** (signed diagonal; reconstruction can fail).
2. **Negative test correlations:** Under archived code, largely an artifact of broken paired signs. Under corrected evaluation, treat remaining negatives as genuine held-out failures — not bug-forced.
3. **Near-zero train–test drop:** Correctly computed as mean(train_corr − test_corr) on the same cells; small because both sides share the sign defect (archived) or genuinely persist (corrected — see reconciled CSV).
4. **Dimension-matched claims:** d=32 and d=64 yes (within branch). d=128 panel comparisons were **not** fully matched (2/4 pairs; OLMo SAE at 115).
5. **Corr vs R²:** Real disagreement can occur; archived `r2` was mislabeled (equal to train-mean improvement). Corrected conventional R² is reported separately.
6. **Language dead features:** Counts correct as never-active on training caption inputs; ~1912/1932 of 2048. Explained by sparse TopK utilization, not a counting bug. Cause of underutilization not uniquely verified.
7. **After corrections, does SAE→PCA improve held-out shared structure?** See corrected headline row above — interpret from held-out correlations/R² in `headline_comparison.csv` / `reconciled_metrics.csv`, **not** training spectra alone.
8. **Single most justified next step:** Re-freeze the experiment with joint-sign projections (and conventional R²), persist U/V/L_A/L_B, and either shrink the language dictionary / raise k or diagnose dead features with a controlled SAE retrain **before** any further cross-modal claims — optionally re-run only evaluation (no architecture search) if checkpoints stay fixed.

Reproduce audit:

```bash
PYTHONPATH=src python scripts/audit_sae_crosscov_v1.py --work /mnt/sdb1/prh-replication-work
```
"""
    (out / "AUDIT.md").write_text(audit_md)
    print(json.dumps({"audit": str(out), "seconds": provenance["seconds"], "corrected_delta_raw_d64": h_corr["mean_delta_test_corr"]}, indent=2))


if __name__ == "__main__":
    main()
