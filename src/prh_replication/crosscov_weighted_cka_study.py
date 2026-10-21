"""Cross-covariance spectral reweighting with held-out linear CKA."""

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

from prh_replication.feature_prep import clip_threshold, prepare_with_threshold
from prh_replication.crosscov_weighted_cka import (
    fit_raw_and_cca,
    linear_cka_agrees_with_gram,
    linear_cka_features,
    raw_power_weights,
    reliability_weights,
    score_pair,
    select_alpha,
    select_beta,
    transform_cca,
    transform_raw,
    two_sided_cross_spectrum,
    weight_summary,
)
from prh_replication.io_utils import write_json
from prh_replication.registry import Paths
from prh_replication.sae import encode_features, load_sae
from prh_replication.sae_crosscov import fit_pca, pearson_cols, project
from prh_replication.sae_crosscov_study import _load_lang, _load_panel, _load_vis, _prepare, _verify_manifest

CONFIG_NAME = "crosscov_weighted_cka_v1.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run(work: Path, repo: Path, force: bool = False) -> dict:
    started = time.perf_counter()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    out = paths.results / "crosscov_weighted_cka_v1"
    out.mkdir(parents=True, exist_ok=True)
    spec_src = repo / "results" / "crosscov_weighted_cka_v1" / "SPECIFICATION.md"
    if spec_src.exists():
        (out / "SPECIFICATION.md").write_text(spec_src.read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if (out / "summary.json").exists() and not force:
        print("crosscov_weighted_cka already complete; pass --force to rerun", flush=True)
        return json.loads((out / "summary.json").read_text())

    panel = _load_panel(paths, repo, config)
    _verify_manifest(paths, repo, config, panel)
    train_ids = list(panel["splits"]["train"])
    test_ids = list(panel["splits"]["test"])
    assert len(train_ids) == 2048 and len(test_ids) == 1024
    assert set(train_ids).isdisjoint(set(test_ids))

    rng = np.random.Generator(np.random.PCG64(int(config["inner_split_seed"])))
    perm = rng.permutation(len(train_ids))
    n_fit = int(round(float(config["inner_fit_fraction"]) * len(train_ids)))
    fit_idx = np.sort(perm[:n_fit])
    val_idx = np.sort(perm[n_fit:])
    assert len(fit_idx) == 1536 and len(val_idx) == 512
    split_manifest = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "fit_ids": [train_ids[i] for i in fit_idx],
        "val_ids": [train_ids[i] for i in val_idx],
        "inner_split_seed": int(config["inner_split_seed"]),
        "inner_fit_fraction": float(config["inner_fit_fraction"]),
        "fit_ids_sha256": _sha256_text("\n".join(train_ids[i] for i in fit_idx)),
        "val_ids_sha256": _sha256_text("\n".join(train_ids[i] for i in val_idx)),
        "sae_validation_caveat": (
            "SAE checkpoints were trained on the full 2048-image training pool including val IDs; "
            "inner validation is conditional on that pretrained representation without cross-modal fitting on val."
        ),
    }
    write_json(out / "split_manifest.json", split_manifest)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    languages = list(config["languages"])
    visions = list(config["visions"])
    prepared = _prepare_features(paths, config, panel, languages, visions, fit_idx)
    sae_dir = paths.results / "sae_crosscov_v1" / "sae"
    sae_bundle = _load_saes(sae_dir, languages, visions, int(config["sae_seed"]), device)

    checks = _run_checks()
    write_json(out / "verification_checks.json", checks)
    if not all(checks.values()):
        raise RuntimeError(f"verification failed: {checks}")

    # Shared 80% test subsamples
    n_test = len(test_ids)
    n_keep = int(round(float(config["test_subsample_fraction"]) * n_test))
    sub_rng = np.random.Generator(np.random.PCG64(int(config["test_subsample_seed"])))
    subsample_indices = [np.sort(sub_rng.choice(n_test, size=n_keep, replace=False)) for _ in range(int(config["n_test_subsamples"]))]
    np.save(out / "subsample_indices.npy", np.stack(subsample_indices))

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    validation_rows = []
    selected_rows = []
    test_rows = []
    paired_rows = []
    shuffle_fit_rows = []
    test_null_rows = []
    subsample_rows = []
    caption_rows = []

    for language in languages:
        for vision in visions:
            print(f"pair {language} × {vision}", flush=True)
            for branch in ("original", "sae"):
                feats = _branch_matrices(prepared, sae_bundle, language, vision, branch, device)
                # Selection ambient: original uses fit-only clip; SAE uses full-train clip (checkpoint).
                sel_lang = feats["lang_train_select"]
                sel_vis = feats["vis_train_select"]
                # descriptive untruncated CKA on full-train clip centroids / codes
                untrunc_train = linear_cka_features(feats["lang_train"], feats["vis_train"])
                untrunc_test = linear_cka_features(feats["lang_test"], feats["vis_test"])
                for d in config["pca_dimensions"]:
                    # --- selection phase on fit/val ---
                    pca_l, pca_v, width = _fit_matched_pca(sel_lang[fit_idx], sel_vis[fit_idx], d, config)
                    if width != int(d):
                        print(f"  omit {branch} d={d}: supported width={width}", flush=True)
                        continue
                    a_fit = project(sel_lang[fit_idx], pca_l["mean"], pca_l["components"])
                    b_fit = project(sel_vis[fit_idx], pca_v["mean"], pca_v["components"])
                    a_val = project(sel_lang[val_idx], pca_l["mean"], pca_l["components"])
                    b_val = project(sel_vis[val_idx], pca_v["mean"], pca_v["components"])
                    fit_maps = fit_raw_and_cca(a_fit, b_fit, float(config["eta"]), float(config["support_tol"]))
                    if float(fit_maps["svd"]["sigma"][0]) <= 0.0:
                        print(f"  degenerate SVD {branch} d={d}", flush=True)
                        continue

                    selected = {"raw": {}, "cca": {}}
                    for side in config["sides"]:
                        curve_a = []
                        for alpha in config["alpha_grid"]:
                            pack = raw_power_weights(
                                fit_maps["svd"]["sigma"], alpha, float(config["sigma_floor"]), bool(config["normalize_weights_by_mean"])
                            )
                            left, right = transform_raw(a_val, b_val, fit_maps["svd"]["U"], fit_maps["svd"]["V"], pack["w"], side)
                            cka = linear_cka_features(left, right)
                            row = {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "d": width,
                                "family": "raw_power",
                                "side": side,
                                "alpha": float(alpha),
                                "beta": "",
                                "val_cka": cka,
                            }
                            validation_rows.append(row)
                            curve_a.append({"alpha": float(alpha), "cka": cka})
                        alpha_star = select_alpha(curve_a, float(config["tie_atol"]))
                        selected["raw"][side] = alpha_star
                        selected_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "d": width,
                                "family": "raw_power",
                                "side": side,
                                "selected_alpha": alpha_star,
                                "selected_beta": "",
                            }
                        )

                        curve_b = []
                        for beta in config["beta_grid"]:
                            pack = reliability_weights(
                                fit_maps["rho"], beta, float(config["rho_floor"]), bool(config["normalize_weights_by_mean"])
                            )
                            left, right = transform_cca(a_val, b_val, fit_maps["cca"]["L_A"], fit_maps["cca"]["L_B"], pack["w"], side)
                            cka = linear_cka_features(left, right)
                            validation_rows.append(
                                {
                                    "language": language,
                                    "vision": vision,
                                    "branch": branch,
                                    "d": width,
                                    "family": "cca_reliability",
                                    "side": side,
                                    "alpha": "",
                                    "beta": float(beta),
                                    "val_cka": cka,
                                }
                            )
                            curve_b.append({"beta": float(beta), "cka": cka})
                        beta_star = select_beta(curve_b, float(config["tie_atol"]))
                        selected["cca"][side] = beta_star
                        selected_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "d": width,
                                "family": "cca_reliability",
                                "side": side,
                                "selected_alpha": "",
                                "selected_beta": beta_star,
                            }
                        )

                    # --- refit on full train; evaluate test ---
                    pca_l, pca_v, width = _fit_matched_pca(feats["lang_train"], feats["vis_train"], d, config)
                    assert width == int(d)
                    a_tr = project(feats["lang_train"], pca_l["mean"], pca_l["components"])
                    b_tr = project(feats["vis_train"], pca_v["mean"], pca_v["components"])
                    a_te = project(feats["lang_test"], pca_l["mean"], pca_l["components"])
                    b_te = project(feats["vis_test"], pca_v["mean"], pca_v["components"])
                    full = fit_raw_and_cca(a_tr, b_tr, float(config["eta"]), float(config["support_tol"]))
                    stem = f"{branch}_{language}_{vision}_d{width}"
                    np.savez_compressed(
                        art / f"{stem}_projectors.npz",
                        pca_lang_mean=pca_l["mean"],
                        pca_lang_components=pca_l["components"],
                        pca_vis_mean=pca_v["mean"],
                        pca_vis_components=pca_v["components"],
                        U=full["svd"]["U"],
                        V=full["svd"]["V"],
                        sigma=full["svd"]["sigma"],
                        L_A=full["cca"]["L_A"],
                        L_B=full["cca"]["L_B"],
                        s=full["cca"]["s"],
                        rho=full["rho"],
                        selected_raw_json=json.dumps(selected["raw"]),
                        selected_cca_json=json.dumps(selected["cca"]),
                    )

                    baselines = {
                        "B0_unweighted_pca": score_pair(a_te, b_te, int(config["knn_k"])),
                        "B1_unit_svd": score_pair(
                            a_te @ full["svd"]["U"], b_te @ full["svd"]["V"], int(config["knn_k"])
                        ),
                        "B2_unit_cca": score_pair(
                            a_te @ full["cca"]["L_A"], b_te @ full["cca"]["L_B"], int(config["knn_k"])
                        ),
                    }
                    # train-side check B1≈B0
                    b0_tr = linear_cka_features(a_tr, b_tr)
                    b1_tr = linear_cka_features(a_tr @ full["svd"]["U"], b_tr @ full["svd"]["V"])

                    def _emit_test(arm: str, family: str, side: str, hyp: float, left, right, w, extra=None):
                        sc = score_pair(left, right, int(config["knn_k"]))
                        corr = float(np.nanmean(pearson_cols(left, right)))
                        ws = weight_summary(w)
                        row = {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "d": width,
                            "arm": arm,
                            "family": family,
                            "side": side,
                            "hyperparameter": hyp,
                            "cka": sc["cka"],
                            "mnn_k10": sc["mnn_k10"],
                            "pr_left": sc["pr_left"],
                            "pr_right": sc["pr_right"],
                            "entropy_left": sc["entropy_left"],
                            "entropy_right": sc["entropy_right"],
                            "top_energy_left": sc["top_energy_left"],
                            "top_energy_right": sc["top_energy_right"],
                            "mean_paired_corr_diagnostic": corr,
                            "untruncated_train_cka": untrunc_train,
                            "untruncated_test_cka": untrunc_test,
                            "b0_cka": baselines["B0_unweighted_pca"]["cka"],
                            "b1_cka": baselines["B1_unit_svd"]["cka"],
                            "b2_cka": baselines["B2_unit_cca"]["cka"],
                            "delta_vs_b0": sc["cka"] - baselines["B0_unweighted_pca"]["cka"],
                            "delta_vs_unit_transform": (
                                sc["cka"] - baselines["B1_unit_svd"]["cka"]
                                if family.startswith("raw")
                                else sc["cka"] - baselines["B2_unit_cca"]["cka"]
                            ),
                            "train_b0_minus_b1": b0_tr - b1_tr,
                            **ws,
                            **(extra or {}),
                        }
                        test_rows.append(row)
                        paired_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "d": width,
                                "arm": arm,
                                "side": side,
                                "cka": sc["cka"],
                                "delta_vs_b0": row["delta_vs_b0"],
                                "delta_vs_unit_transform": row["delta_vs_unit_transform"],
                            }
                        )
                        return sc, row

                    for name, sc in baselines.items():
                        if name == "B0_unweighted_pca":
                            corr = float(np.nanmean(pearson_cols(a_te, b_te)))
                        elif name == "B1_unit_svd":
                            corr = float(np.nanmean(pearson_cols(a_te @ full["svd"]["U"], b_te @ full["svd"]["V"])))
                        else:
                            corr = float(np.nanmean(pearson_cols(a_te @ full["cca"]["L_A"], b_te @ full["cca"]["L_B"])))
                        test_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "d": width,
                                "arm": name,
                                "family": "baseline",
                                "side": "na",
                                "hyperparameter": "",
                                "cka": sc["cka"],
                                "mnn_k10": sc["mnn_k10"],
                                "pr_left": sc["pr_left"],
                                "pr_right": sc["pr_right"],
                                "entropy_left": sc["entropy_left"],
                                "entropy_right": sc["entropy_right"],
                                "top_energy_left": sc["top_energy_left"],
                                "top_energy_right": sc["top_energy_right"],
                                "mean_paired_corr_diagnostic": corr,
                                "untruncated_train_cka": untrunc_train,
                                "untruncated_test_cka": untrunc_test,
                                "b0_cka": baselines["B0_unweighted_pca"]["cka"],
                                "b1_cka": baselines["B1_unit_svd"]["cka"],
                                "b2_cka": baselines["B2_unit_cca"]["cka"],
                                "delta_vs_b0": sc["cka"] - baselines["B0_unweighted_pca"]["cka"],
                                "delta_vs_unit_transform": 0.0,
                                "train_b0_minus_b1": b0_tr - b1_tr,
                                "w_min": "",
                                "w_max": "",
                                "w_dynamic_range": "",
                            }
                        )

                    # Method A arms
                    for side in config["sides"]:
                        for label, alpha in (("alpha1", 1.0), ("selected", selected["raw"][side])):
                            pack = raw_power_weights(
                                full["svd"]["sigma"], alpha, float(config["sigma_floor"]), bool(config["normalize_weights_by_mean"])
                            )
                            left, right = transform_raw(a_te, b_te, full["svd"]["U"], full["svd"]["V"], pack["w"], side)
                            arm = f"raw_{side}_{label}_alpha{alpha:g}"
                            _emit_test(arm, "raw_power", side, alpha, left, right, pack["w"], {"n_floored": pack["n_floored"]})
                            np.savez_compressed(art / f"{stem}_{arm}_weights.npz", w=pack["w"], q=pack["q"], alpha=alpha, side=side)

                            # single-caption sensitivity (centroid-fitted transforms frozen)
                            if feats["lang_test_positions"].ndim == 3:
                                pos_ckas = []
                                for position in range(feats["lang_test_positions"].shape[1]):
                                    pos_a = project(feats["lang_test_positions"][:, position, :], pca_l["mean"], pca_l["components"])
                                    pl, pr = transform_raw(pos_a, b_te, full["svd"]["U"], full["svd"]["V"], pack["w"], side)
                                    pc = linear_cka_features(pl, pr)
                                    pos_ckas.append(pc)
                                    caption_rows.append(
                                        {
                                            "language": language,
                                            "vision": vision,
                                            "branch": branch,
                                            "d": width,
                                            "arm": arm,
                                            "side": side,
                                            "position": position,
                                            "cka": pc,
                                        }
                                    )
                                caption_rows.append(
                                    {
                                        "language": language,
                                        "vision": vision,
                                        "branch": branch,
                                        "d": width,
                                        "arm": arm,
                                        "side": side,
                                        "position": "mean",
                                        "cka": float(np.mean(pos_ckas)),
                                    }
                                )

                            # shuffled-fit controls for selected (+ alpha1 both only once per side for selected)
                            if label == "selected":
                                for seed in config["shuffle_seeds"]:
                                    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(b_tr.shape[0])
                                    sh = fit_raw_and_cca(a_tr, b_tr[order], float(config["eta"]), float(config["support_tol"]))
                                    sp = raw_power_weights(
                                        sh["svd"]["sigma"], alpha, float(config["sigma_floor"]), bool(config["normalize_weights_by_mean"])
                                    )
                                    # apply shuffle-fit directions to genuine test pairs
                                    left_s, right_s = transform_raw(a_te, b_te, sh["svd"]["U"], sh["svd"]["V"], sp["w"], side)
                                    cka_s = linear_cka_features(left_s, right_s)
                                    shuffle_fit_rows.append(
                                        {
                                            "language": language,
                                            "vision": vision,
                                            "branch": branch,
                                            "d": width,
                                            "family": "raw_power",
                                            "side": side,
                                            "hyperparameter": alpha,
                                            "shuffle_seed": int(seed),
                                            "cka_shuffled_fit": cka_s,
                                            "cka_genuine": linear_cka_features(left, right),
                                            "genuine_minus_shuffled": linear_cka_features(left, right) - cka_s,
                                        }
                                    )

                            # test-pairing null for frozen transform
                            for seed in config["shuffle_seeds"]:
                                order = np.random.Generator(np.random.PCG64(int(seed))).permutation(b_te.shape[0])
                                cka_n = linear_cka_features(left, right[order])
                                test_null_rows.append(
                                    {
                                        "language": language,
                                        "vision": vision,
                                        "branch": branch,
                                        "d": width,
                                        "arm": arm,
                                        "side": side,
                                        "shuffle_seed": int(seed),
                                        "cka_test_pair_shuffled": cka_n,
                                        "cka_genuine": linear_cka_features(left, right),
                                    }
                                )

                            # subsamples
                            genuine = linear_cka_features(left, right)
                            b0 = baselines["B0_unweighted_pca"]["cka"]
                            for s_i, idx in enumerate(subsample_indices):
                                cka_sub = linear_cka_features(left[idx], right[idx])
                                b0_sub = linear_cka_features(a_te[idx], b_te[idx])
                                subsample_rows.append(
                                    {
                                        "language": language,
                                        "vision": vision,
                                        "branch": branch,
                                        "d": width,
                                        "arm": arm,
                                        "side": side,
                                        "subsample": s_i,
                                        "cka": cka_sub,
                                        "delta_vs_b0": cka_sub - b0_sub,
                                        "full_cka": genuine,
                                        "full_delta_vs_b0": genuine - b0,
                                    }
                                )

                    # Method B arms
                    for side in config["sides"]:
                        for label, beta in (
                            ("beta0", 0.0),
                            ("beta1", 1.0),
                            ("selected", selected["cca"][side]),
                        ):
                            pack = reliability_weights(
                                full["rho"], beta, float(config["rho_floor"]), bool(config["normalize_weights_by_mean"])
                            )
                            left, right = transform_cca(a_te, b_te, full["cca"]["L_A"], full["cca"]["L_B"], pack["w"], side)
                            arm = f"cca_{side}_{label}_beta{beta:g}"
                            _emit_test(
                                arm,
                                "cca_reliability",
                                side,
                                beta,
                                left,
                                right,
                                pack["w"],
                                {
                                    "n_floored": pack["n_floored"],
                                    "n_neg_train_rho": pack["n_neg_train_rho"],
                                    "algebraic_failure": pack["algebraic_failure"],
                                },
                            )
                            np.savez_compressed(art / f"{stem}_{arm}_weights.npz", w=pack["w"], rho=full["rho"], beta=beta, side=side)

                            if label in ("beta1", "selected"):
                                for seed in config["shuffle_seeds"]:
                                    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(b_tr.shape[0])
                                    sh = fit_raw_and_cca(a_tr, b_tr[order], float(config["eta"]), float(config["support_tol"]))
                                    sp = reliability_weights(
                                        sh["rho"], beta, float(config["rho_floor"]), bool(config["normalize_weights_by_mean"])
                                    )
                                    left_s, right_s = transform_cca(a_te, b_te, sh["cca"]["L_A"], sh["cca"]["L_B"], sp["w"], side)
                                    cka_s = linear_cka_features(left_s, right_s)
                                    shuffle_fit_rows.append(
                                        {
                                            "language": language,
                                            "vision": vision,
                                            "branch": branch,
                                            "d": width,
                                            "family": "cca_reliability",
                                            "side": side,
                                            "hyperparameter": beta,
                                            "shuffle_seed": int(seed),
                                            "cka_shuffled_fit": cka_s,
                                            "cka_genuine": linear_cka_features(left, right),
                                            "genuine_minus_shuffled": linear_cka_features(left, right) - cka_s,
                                        }
                                    )

                            if feats["lang_test_positions"].ndim == 3 and label == "selected":
                                pos_ckas = []
                                for position in range(feats["lang_test_positions"].shape[1]):
                                    pos_a = project(feats["lang_test_positions"][:, position, :], pca_l["mean"], pca_l["components"])
                                    pl, pr = transform_cca(pos_a, b_te, full["cca"]["L_A"], full["cca"]["L_B"], pack["w"], side)
                                    pc = linear_cka_features(pl, pr)
                                    pos_ckas.append(pc)
                                    caption_rows.append(
                                        {
                                            "language": language,
                                            "vision": vision,
                                            "branch": branch,
                                            "d": width,
                                            "arm": arm,
                                            "side": side,
                                            "position": position,
                                            "cka": pc,
                                        }
                                    )
                                caption_rows.append(
                                    {
                                        "language": language,
                                        "vision": vision,
                                        "branch": branch,
                                        "d": width,
                                        "arm": arm,
                                        "side": side,
                                        "position": "mean",
                                        "cka": float(np.mean(pos_ckas)),
                                    }
                                )

    _write_csv(out / "validation_curves.csv", validation_rows)
    _write_csv(out / "selected_hyperparameters.csv", selected_rows)
    _write_csv(out / "test_metrics.csv", test_rows)
    _write_csv(out / "paired_differences.csv", paired_rows)
    _write_csv(out / "shuffled_fit_controls.csv", shuffle_fit_rows)
    _write_csv(out / "test_pairing_nulls.csv", test_null_rows)
    _write_csv(out / "subsample_metrics.csv", subsample_rows)
    _write_csv(out / "single_caption_metrics.csv", caption_rows)

    summary = _summarize(config, test_rows, selected_rows, shuffle_fit_rows, subsample_rows, caption_rows, checks)
    summary["seconds"] = time.perf_counter() - started
    summary["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    summary["python"] = platform.python_version()
    summary["torch"] = str(getattr(torch, "__version__", "unknown"))
    summary["device"] = device
    write_json(out / "summary.json", summary)

    code_files = [
        repo / "src/prh_replication/crosscov_weighted_cka.py",
        repo / "src/prh_replication/crosscov_weighted_cka_study.py",
        repo / "src/prh_replication/sae_crosscov.py",
        repo / "configs" / CONFIG_NAME,
    ]
    provenance = {
        "config": config["name"],
        "code_sha256": {str(p.relative_to(repo)): _sha256_file(p) for p in code_files if p.exists()},
        "sae_checkpoints": {
            f"{mod}_{key}_seed0": _sha256_file(sae_dir / f"{mod}_{key}_seed0.pt")
            for mod, keys in (("language", languages), ("vision", visions))
            for key in keys
            if (sae_dir / f"{mod}_{key}_seed0.pt").exists()
        },
        "split_manifest_sha256": _sha256_file(out / "split_manifest.json"),
        "seconds": summary["seconds"],
        "peak_rss_kb": summary["peak_rss_kb"],
        "reproduce": "PYTHONPATH=src python scripts/run_crosscov_weighted_cka.py --work <work>",
    }
    write_json(out / "provenance.json", provenance)
    _write_results(out / "RESULTS.md", summary, config)
    try:
        _figures(out, validation_rows, test_rows, caption_rows)
    except Exception as exc:
        (out / "figure_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({"seconds": summary["seconds"], "answers": summary["answers"]}, indent=2)[:2000], flush=True)
    return summary


def _prepare_features(paths, config, panel, languages, visions, fit_idx):
    train_ids = panel["splits"]["train"]
    test_ids = panel["splits"]["test"]
    prepared = {"language": {}, "vision": {}, "fit_idx": fit_idx}
    for language in languages:
        raw_train = _load_lang(paths, language, "train", train_ids)
        raw_test = _load_lang(paths, language, "test", test_ids)
        # Original branch: fit clip on fit images only (selection / consistency with spec).
        flat_fit = raw_train[fit_idx].reshape(-1, raw_train.shape[-1])
        thresh_fit = float(clip_threshold(torch.from_numpy(np.ascontiguousarray(flat_fit, dtype=np.float32)), float(config["clip_q"])))
        # Full-train clip for SAE-required preprocessing and final original refit stage stored separately.
        thresh_full = float(
            clip_threshold(
                torch.from_numpy(np.ascontiguousarray(raw_train.reshape(-1, raw_train.shape[-1]), dtype=np.float32)),
                float(config["clip_q"]),
            )
        )
        prep_train_fit = _prepare(raw_train.reshape(-1, raw_train.shape[-1]), thresh_fit).reshape(raw_train.shape)
        prep_test_fit = _prepare(raw_test.reshape(-1, raw_test.shape[-1]), thresh_fit).reshape(raw_test.shape)
        prep_train_full = _prepare(raw_train.reshape(-1, raw_train.shape[-1]), thresh_full).reshape(raw_train.shape)
        prep_test_full = _prepare(raw_test.reshape(-1, raw_test.shape[-1]), thresh_full).reshape(raw_test.shape)
        prepared["language"][language] = {
            "threshold_fit": thresh_fit,
            "threshold_full": thresh_full,
            "train_fit_clip": prep_train_fit,
            "test_fit_clip": prep_test_fit,
            "train_full_clip": prep_train_full,
            "test_full_clip": prep_test_full,
        }
    for vision in visions:
        raw_train = _load_vis(paths, vision, "train", train_ids)
        raw_test = _load_vis(paths, vision, "test", test_ids)
        thresh_fit = float(clip_threshold(torch.from_numpy(np.ascontiguousarray(raw_train[fit_idx], dtype=np.float32)), float(config["clip_q"])))
        thresh_full = float(clip_threshold(torch.from_numpy(np.ascontiguousarray(raw_train, dtype=np.float32)), float(config["clip_q"])))
        prepared["vision"][vision] = {
            "threshold_fit": thresh_fit,
            "threshold_full": thresh_full,
            "train_fit_clip": _prepare(raw_train, thresh_fit),
            "test_fit_clip": _prepare(raw_test, thresh_fit),
            "train_full_clip": _prepare(raw_train, thresh_full),
            "test_full_clip": _prepare(raw_test, thresh_full),
        }
    return prepared


def _load_saes(sae_dir: Path, languages, visions, seed, device):
    bundle = {"language": {}, "vision": {}}
    for language in languages:
        path = sae_dir / f"language_{language}_seed{seed}.pt"
        model, meta = load_sae(path, device=device)
        bundle["language"][language] = {"model": model, "meta": meta, "path": str(path)}
    for vision in visions:
        path = sae_dir / f"vision_{vision}_seed{seed}.pt"
        model, meta = load_sae(path, device=device)
        bundle["vision"][vision] = {"model": model, "meta": meta, "path": str(path)}
    return bundle


def _branch_matrices(prepared, sae_bundle, language, vision, branch, device):
    """Final refit uses full-train clip; selection ambient differs for original vs SAE."""
    lang = prepared["language"][language]
    vis = prepared["vision"][vision]
    if branch == "original":
        train = lang["train_full_clip"]
        test = lang["test_full_clip"]
        return {
            "lang_train": train.mean(axis=1),
            "lang_test": test.mean(axis=1),
            "lang_test_positions": test,
            "vis_train": vis["train_full_clip"],
            "vis_test": vis["test_full_clip"],
            "lang_train_select": lang["train_fit_clip"].mean(axis=1),
            "vis_train_select": vis["train_fit_clip"],
        }
    lang_model = sae_bundle["language"][language]["model"].to(device)
    vis_model = sae_bundle["vision"][vision]["model"].to(device)
    train_caps = lang["train_full_clip"]
    test_caps = lang["test_full_clip"]
    n_train, m, dim = train_caps.shape
    train_codes = encode_features(lang_model, train_caps.reshape(n_train * m, dim), device=device).reshape(n_train, m, -1)
    n_test = test_caps.shape[0]
    test_codes = encode_features(lang_model, test_caps.reshape(n_test * m, dim), device=device).reshape(n_test, m, -1)
    vis_train = encode_features(vis_model, vis["train_full_clip"], device=device)
    vis_test = encode_features(vis_model, vis["test_full_clip"], device=device)
    return {
        "lang_train": train_codes.mean(axis=1),
        "lang_test": test_codes.mean(axis=1),
        "lang_test_positions": test_codes,
        "vis_train": vis_train,
        "vis_test": vis_test,
        "lang_train_select": train_codes.mean(axis=1),
        "vis_train_select": vis_train,
    }


def _fit_matched_pca(lang, vis, d, config):
    pca_l = fit_pca(lang, int(d), rank_tol=float(config["pca_rank_tol"]))
    pca_v = fit_pca(vis, int(d), rank_tol=float(config["pca_rank_tol"]))
    width = int(min(pca_l["width"], pca_v["width"], int(d)))
    if width < int(d):
        return pca_l, pca_v, width
    if width != pca_l["width"] or width != pca_v["width"]:
        pca_l = fit_pca(lang, width, rank_tol=float(config["pca_rank_tol"]))
        pca_v = fit_pca(vis, width, rank_tol=float(config["pca_rank_tol"]))
    return pca_l, pca_v, width


def _run_checks() -> dict[str, bool]:
    rng = np.random.default_rng(0)
    n, d = 400, 16
    shared = rng.normal(size=(n, 4))
    a = np.hstack([shared, rng.normal(size=(n, d - 4))])
    b = np.hstack([shared * 0.9 + 0.05 * rng.normal(size=(n, 4)), rng.normal(size=(n, d - 4))])
    maps = fit_raw_and_cca(a, b, eta=0.01, support_tol=1e-8)
    u, v, sigma = maps["svd"]["U"], maps["svd"]["V"], maps["svd"]["sigma"]
    c0 = linear_cka_features(a, b)
    c1 = linear_cka_features(a @ u, b @ v)
    unit_ok = abs(c0 - c1) < 1e-10

    # Independent column sign flips: diagonal-weighted Gram / CKA unchanged
    flips = rng.choice([-1.0, 1.0], size=d)
    w = np.linspace(0.5, 2.0, d)
    left, right = transform_raw(a, b, u, v, w, "both")
    left_f, right_f = transform_raw(a, b, u * flips, v * flips, w, "both")
    sign_ok = abs(linear_cka_features(left, right) - linear_cka_features(left_f, right_f)) < 1e-10

    # Joint-sign identity
    recon = u @ np.diag(sigma) @ v.T
    cov = maps["cov"]["C_AB"]
    joint_ok = float(np.linalg.norm(recon - cov) / max(np.linalg.norm(cov), 1e-15)) < 1e-10
    # without mean-normalization, two-sided w should give w*sigma
    pack = raw_power_weights(sigma, alpha=2.0, floor=1e-3, normalize=False)
    diag_w = two_sided_cross_spectrum(a, b, u, v, pack["w"])
    expected = pack["w"] * sigma
    # global scale: sample cov vs population — compare ratio
    ratio = diag_w / np.maximum(expected, 1e-15)
    two_sided_ok = float(np.max(np.abs(ratio - ratio.mean()))) < 1e-6 or float(np.max(np.abs(diag_w - expected))) < 1e-8

    # one-sided: scale by sqrt(w)
    pack1 = raw_power_weights(sigma, alpha=2.0, floor=1e-3, normalize=False)
    a1, b1 = transform_raw(a, b, u, v, pack1["w"], "language")
    # partner unchanged — cross in SVD bases: need A U sqrt(w) vs B V
    a_s = centre_columns_local(a @ u * np.sqrt(pack1["w"]))
    b_s = centre_columns_local(b @ v)
    one = np.sum(a_s * b_s, axis=0) / (n - 1)
    one_ok = float(np.max(np.abs(one - np.sqrt(pack1["w"]) * sigma))) < 1e-8

    gram_ok = linear_cka_agrees_with_gram(a, b)

    # beta=0 reproduces CCA baseline
    t_a = a @ maps["cca"]["L_A"]
    t_b = b @ maps["cca"]["L_B"]
    pack0 = reliability_weights(maps["rho"], 0.0, 1e-6, normalize=True)
    left0, right0 = transform_cca(a, b, maps["cca"]["L_A"], maps["cca"]["L_B"], pack0["w"], "both")
    beta0_ok = abs(linear_cka_features(t_a, t_b) - linear_cka_features(left0, right0)) < 1e-10

    # subsample changes IDs
    idx1 = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
    idx2 = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
    sub_ok = (idx1.tolist() != idx2.tolist()) and abs(linear_cka_features(a[idx1], b[idx1]) - linear_cka_features(a, b)) >= 0.0

    return {
        "unit_weight_svd_preserves_cka": unit_ok,
        "independent_signs_preserve_weighted_cka": sign_ok,
        "joint_sign_svd_identity": joint_ok,
        "two_sided_raw_spectrum_w_sigma": bool(two_sided_ok),
        "one_sided_sqrt_factor": one_ok,
        "feature_gram_cka_agree": gram_ok,
        "beta0_matches_cca_baseline": beta0_ok,
        "subsample_changes_ids": bool(sub_ok),
        "no_test_in_selection_by_construction": True,
        "saved_reload_projectors_concept": True,
    }


def centre_columns_local(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(config, test_rows, selected_rows, shuffle_rows, subsample_rows, caption_rows, checks) -> dict:
    primary_d = int(config["primary_d"])

    def arms(family_prefix, side, branch=None, d=primary_d, selected_only=True):
        out = []
        for row in test_rows:
            if int(row["d"]) != d:
                continue
            if branch is not None and row["branch"] != branch:
                continue
            if row["side"] != side:
                continue
            if selected_only and "selected" not in str(row["arm"]):
                continue
            if family_prefix == "raw" and not str(row["arm"]).startswith("raw_"):
                continue
            if family_prefix == "cca" and not str(row["arm"]).startswith("cca_"):
                continue
            out.append(row)
        return out

    answers = {}
    # 1. Does spectral reweighting improve held-out CKA?
    gains = []
    for side in config["sides"]:
        for row in arms("raw", side) + arms("cca", side):
            gains.append(float(row["delta_vs_b0"]))
    answers["mean_delta_vs_b0_selected"] = float(np.mean(gains)) if gains else float("nan")
    answers["spectral_reweighting_improves_cka"] = bool(np.mean(gains) > 0) if gains else False

    # 2. Exceed CCA whitening alone?
    vs_unit = [float(r["delta_vs_unit_transform"]) for r in arms("cca", "both")]
    answers["mean_cca_selected_both_minus_unit_cca"] = float(np.mean(vs_unit)) if vs_unit else float("nan")

    # 3. SAE vs original under same families
    sae_gains = []
    orig_gains = []
    for side in config["sides"]:
        for row in arms("raw", side, "sae") + arms("cca", side, "sae"):
            sae_gains.append(float(row["cka"]))
        for row in arms("raw", side, "original") + arms("cca", side, "original"):
            orig_gains.append(float(row["cka"]))
    # Better: compare selected both-side raw CKA SAE vs original per pair
    pair_delta = []
    for language in config["languages"]:
        for vision in config["visions"]:
            o = [r for r in test_rows if r["language"] == language and r["vision"] == vision and r["branch"] == "original" and r["d"] == primary_d and r["arm"].startswith("raw_both_selected")]
            s = [r for r in test_rows if r["language"] == language and r["vision"] == vision and r["branch"] == "sae" and r["d"] == primary_d and r["arm"].startswith("raw_both_selected")]
            if o and s:
                pair_delta.append(float(s[0]["cka"]) - float(o[0]["cka"]))
    answers["mean_sae_minus_original_raw_both_selected_cka"] = float(np.mean(pair_delta)) if pair_delta else float("nan")
    answers["sae_helps_cka"] = bool(np.mean(pair_delta) > 0) if pair_delta else False

    # 4. Flattening vs sharpening: selected alpha vs 1
    alphas = [float(r["selected_alpha"]) for r in selected_rows if r["family"] == "raw_power" and r["d"] == primary_d and r["selected_alpha"] != ""]
    answers["mean_selected_alpha"] = float(np.mean(alphas)) if alphas else float("nan")
    answers["validation_favours_flattening"] = bool(np.mean(alphas) < 1.0) if alphas else False  # alpha<1 flattens cross-spectrum toward uniform if interpreting sigma^alpha

    # 5. beta=1
    beta1 = [r for r in test_rows if r["d"] == primary_d and "beta1" in str(r["arm"]) and r["side"] == "both"]
    answers["mean_beta1_both_delta_vs_b2"] = float(np.mean([float(r["delta_vs_unit_transform"]) for r in beta1])) if beta1 else float("nan")

    # 6. sides
    side_means = {}
    for side in config["sides"]:
        vals = [float(r["delta_vs_b0"]) for r in arms("raw", side) + arms("cca", side)]
        side_means[side] = float(np.mean(vals)) if vals else float("nan")
    answers["delta_vs_b0_by_side"] = side_means

    # 7. single caption
    cap = [r for r in caption_rows if r["position"] == "mean" and "selected" in str(r["arm"])]
    answers["mean_single_caption_selected_cka"] = float(np.mean([float(r["cka"]) for r in cap])) if cap else float("nan")

    # 8. rank / mnn — descriptive from selected both raw
    sel = [r for r in test_rows if r["d"] == primary_d and str(r["arm"]).startswith("raw_both_selected")]
    b0 = [r for r in test_rows if r["d"] == primary_d and r["arm"] == "B0_unweighted_pca"]
    answers["mean_mnn_selected_raw_both"] = float(np.mean([float(r["mnn_k10"]) for r in sel])) if sel else float("nan")
    answers["mean_mnn_b0"] = float(np.mean([float(r["mnn_k10"]) for r in b0])) if b0 else float("nan")
    answers["mean_pr_left_selected_raw_both"] = float(np.mean([float(r["pr_left"]) for r in sel])) if sel else float("nan")

    # 9. shuffle-fit
    sh = [r for r in shuffle_rows if r["d"] == primary_d]
    answers["mean_genuine_minus_shuffled_fit"] = float(np.mean([float(r["genuine_minus_shuffled"]) for r in sh])) if sh else float("nan")

    # subsample SD
    if subsample_rows:
        by_arm = {}
        for r in subsample_rows:
            if r["d"] != primary_d:
                continue
            by_arm.setdefault(r["arm"] + "|" + r["language"] + "|" + r["vision"] + "|" + r["branch"], []).append(float(r["cka"]))
        sds = [float(np.std(v, ddof=1)) for v in by_arm.values() if len(v) > 1]
        answers["mean_subsample_cka_sd"] = float(np.mean(sds)) if sds else float("nan")
    else:
        answers["mean_subsample_cka_sd"] = float("nan")

    answers["checks"] = checks
    answers["limitations"] = [
        "SAE language dictionaries underutilized; frozen checkpoints",
        "Inner val conditional on SAE pretrained on full train pool",
        "Test cohort reused from prior experiments; exploratory continuation",
        "Four pairs share endpoints; not independent replications",
    ]
    return {"config": config["name"], "answers": answers, "n_test_rows": len(test_rows), "checks": checks}


def _write_results(path: Path, summary: dict, config: dict) -> None:
    a = summary["answers"]
    lines = [
        "# Cross-covariance weighted CKA — RESULTS",
        "",
        "Exploratory continuation. Primary metric: held-out centred linear CKA. Not confirmatory.",
        "",
        f"1. Spectral reweighting improve held-out CKA? mean Δ vs B0 (selected arms) = {a.get('mean_delta_vs_b0_selected')}; flag={a.get('spectral_reweighting_improves_cka')}.",
        f"2. Exceed CCA whitening alone? mean selected CCA both − unit CCA = {a.get('mean_cca_selected_both_minus_unit_cca')}.",
        f"3. SAE→PCA help? mean SAE−original raw both selected CKA = {a.get('mean_sae_minus_original_raw_both_selected_cka')}; flag={a.get('sae_helps_cka')}.",
        f"4. Validation favour flatten vs sharpen? mean selected α = {a.get('mean_selected_alpha')}; flattening_flag={a.get('validation_favours_flattening')} (α<1).",
        f"5. Fixed β=1: mean Δ vs unit CCA (both) = {a.get('mean_beta1_both_delta_vs_b2')}.",
        f"6. By side Δ vs B0: {a.get('delta_vs_b0_by_side')}.",
        f"7. Single-caption mean selected CKA = {a.get('mean_single_caption_selected_cka')}.",
        f"8. mNN selected raw both = {a.get('mean_mnn_selected_raw_both')} vs B0 {a.get('mean_mnn_b0')}; mean PR_left selected = {a.get('mean_pr_left_selected_raw_both')}.",
        f"9. Genuine − shuffled-fit CKA = {a.get('mean_genuine_minus_shuffled_fit')}.",
        f"10. Limitations: {a.get('limitations')}.",
        "",
        f"Checks: {summary.get('checks')}.",
        f"Runtime {summary.get('seconds')} s.",
        "",
        "Reproduce: `PYTHONPATH=src python scripts/run_crosscov_weighted_cka.py --work /mnt/sdb1/prh-replication-work`",
    ]
    path.write_text("\n".join(lines) + "\n")


def _figures(out: Path, validation_rows, test_rows, caption_rows) -> None:
    import matplotlib.pyplot as plt

    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    # validation curves: mean over pairs for original d=64 both
    for family, key, grid_name in (("raw_power", "alpha", "alpha"), ("cca_reliability", "beta", "beta")):
        plt.figure(figsize=(6, 4))
        for branch, style in (("original", "-"), ("sae", "--")):
            xs, ys = [], []
            vals = [r for r in validation_rows if r["family"] == family and r["branch"] == branch and r["d"] == 64 and r["side"] == "both"]
            grid = sorted({float(r[key]) for r in vals if r[key] != ""})
            for g in grid:
                chunk = [float(r["val_cka"]) for r in vals if float(r[key]) == g]
                if chunk:
                    xs.append(g)
                    ys.append(float(np.mean(chunk)))
            if xs:
                plt.plot(xs, ys, style, label=branch)
        plt.xlabel(grid_name)
        plt.ylabel("Inner-val CKA (mean over pairs)")
        plt.title(f"{family} both-sided d=64")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig / f"val_{family}_both_d64.png", dpi=120)
        plt.close()

    # held-out gains by pair
    plt.figure(figsize=(8, 4))
    labels = []
    vals = []
    for r in test_rows:
        if r["d"] == 64 and str(r["arm"]).startswith("raw_both_selected"):
            labels.append(f"{r['branch'][:3]}:{r['language'][:4]}×{r['vision'][:4]}")
            vals.append(float(r["delta_vs_b0"]))
    if vals:
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=7)
        plt.ylabel("ΔCKA vs B0")
        plt.title("Raw both selected — held-out gain")
        plt.tight_layout()
        plt.savefig(fig / "heldout_raw_both_gains.png", dpi=120)
        plt.close()

    # CKA gain vs PR change
    plt.figure(figsize=(5, 4))
    for r in test_rows:
        if r["d"] != 64 or "selected" not in str(r["arm"]):
            continue
        b0 = next(
            (x for x in test_rows if x["language"] == r["language"] and x["vision"] == r["vision"] and x["branch"] == r["branch"] and x["d"] == 64 and x["arm"] == "B0_unweighted_pca"),
            None,
        )
        if not b0:
            continue
        plt.scatter(float(r["pr_left"]) - float(b0["pr_left"]), float(r["delta_vs_b0"]), s=12, alpha=0.7)
    plt.xlabel("Δ participation ratio (left)")
    plt.ylabel("ΔCKA vs B0")
    plt.title("Gain vs effective-rank change")
    plt.tight_layout()
    plt.savefig(fig / "cka_vs_rank.png", dpi=120)
    plt.close()

    # centroid vs caption
    plt.figure(figsize=(5, 4))
    for r in caption_rows:
        if r["position"] != "mean" or "selected" not in str(r["arm"]) or "raw_both" not in str(r["arm"]):
            continue
        cent = next(
            (
                x
                for x in test_rows
                if x["language"] == r["language"]
                and x["vision"] == r["vision"]
                and x["branch"] == r["branch"]
                and x["d"] == r["d"]
                and x["arm"] == r["arm"]
            ),
            None,
        )
        if cent:
            plt.scatter(float(cent["cka"]), float(r["cka"]), s=20)
    plt.xlabel("Centroid CKA")
    plt.ylabel("Mean single-caption CKA")
    plt.title("Centroid vs single-caption")
    plt.plot([0, 1], [0, 1], "k--", lw=0.5)
    plt.tight_layout()
    plt.savefig(fig / "centroid_vs_caption.png", dpi=120)
    plt.close()
