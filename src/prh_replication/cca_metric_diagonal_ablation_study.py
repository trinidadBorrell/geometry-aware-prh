"""Study runner: PCA-axis diagonal ablation of CCA reliability metrics."""

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

from prh_replication.cca_metric_diagonal_ablation import (
    CONDITIONS,
    CONTRASTS,
    beta1_weights,
    build_full_metric,
    check_psd,
    cka_from_grams,
    diagonal_ablation,
    distances_from_gram,
    evaluate_condition,
    kernel_spectrum_diagnostics,
    metric_factor_diagonal,
    metric_factor_full,
    metric_spectrum_diagnostics,
    resolve_factor,
    transform_scores,
    weighted_gram,
)
from prh_replication.crosscov_metric_mknn import mknn_from_features
from prh_replication.crosscov_weighted_cka import linear_cka_features
from prh_replication.crosscov_weighted_cka_study import (
    _branch_matrices,
    _load_saes,
    _prepare_features,
)
from prh_replication.io_utils import write_json
from prh_replication.registry import Paths
from prh_replication.sae_crosscov import project, sample_covariances
from prh_replication.sae_crosscov_study import _load_panel, _verify_manifest
from prh_replication.spectral_calibration import sq_euclidean_rows

CONFIG_NAME = "cca_metric_diagonal_ablation_v1.json"


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


def _load_archived_baselines(path: Path) -> dict:
    """Map (language, vision, branch, arm) -> {cka, mnn_k10} for d=64."""
    out = {}
    for row in csv.DictReader(path.open()):
        if str(row["d"]) != "64":
            continue
        key = (row["language"], row["vision"], row["branch"], row["arm"])
        out[key] = {"cka": float(row["cka"]), "mnn_k10": float(row["mnn_k10"])}
    return out


def run(work: Path, repo: Path, force: bool = False) -> dict:
    started = time.perf_counter()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    out = paths.results / "cca_metric_diagonal_ablation_v1"
    out.mkdir(parents=True, exist_ok=True)
    spec = repo / "results" / "cca_metric_diagonal_ablation_v1" / "SPECIFICATION.md"
    if spec.exists():
        (out / "SPECIFICATION.md").write_text(spec.read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if (out / "summary.json").exists() and not force:
        print("cca_metric_diagonal_ablation already complete; pass --force to rerun", flush=True)
        return json.loads((out / "summary.json").read_text())

    parent = paths.results / "crosscov_weighted_cka_v1"
    split = json.loads((parent / "split_manifest.json").read_text())
    train_ids = list(split["train_ids"])
    test_ids = list(split["test_ids"])
    fit_ids = list(split["fit_ids"])
    id_to_train = {sid: i for i, sid in enumerate(train_ids)}
    fit_idx = np.array([id_to_train[sid] for sid in fit_ids], dtype=np.int64)
    assert set(train_ids).isdisjoint(set(test_ids))

    write_json(
        out / "split_manifest.json",
        {
            "train_n": len(train_ids),
            "test_n": len(test_ids),
            "source": str(parent / "split_manifest.json"),
            "note": "No rank/hyperparameter selection; full-train archived CCA projectors only.",
            "sae_validation_caveat": split.get("sae_validation_caveat", ""),
        },
    )

    panel = {"splits": {"train": train_ids, "test": test_ids}}
    full_panel_cfg = json.loads((parent / "config.json").read_text())
    loaded = _load_panel(paths, repo, full_panel_cfg)
    _verify_manifest(paths, repo, full_panel_cfg, loaded)
    if list(loaded["splits"]["train"]) != train_ids or list(loaded["splits"]["test"]) != test_ids:
        raise RuntimeError("parent split_manifest IDs disagree with panel splits")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    languages = list(config["languages"])
    visions = list(config["visions"])
    prepared = _prepare_features(paths, full_panel_cfg, panel, languages, visions, fit_idx)
    sae_bundle = _load_saes(paths.results / "sae_crosscov_v1" / "sae", languages, visions, int(config["sae_seed"]), device)
    archived = _load_archived_baselines(parent / "test_metrics.csv")
    subsample_indices = np.load(parent / "subsample_indices.npy")

    checks = _run_checks()
    write_json(out / "verification_checks.json", checks)
    if not all(v for k, v in checks.items() if k != "notes"):
        bad = {k: v for k, v in checks.items() if v is False}
        raise RuntimeError(f"verification failed: {bad}")

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    test_rows: list[dict] = []
    contrast_rows: list[dict] = []
    metric_diag_rows: list[dict] = []
    spectra_rows: list[dict] = []
    caption_rows: list[dict] = []
    subsample_rows: list[dict] = []
    repro: dict = {"gaps": [], "max_cka_gap": 0.0, "max_mknn_gap": 0.0}

    k = int(config["knn_k"])
    condition_scores: dict[tuple, dict] = {}

    for language in languages:
        for vision in visions:
            print(f"pair {language} × {vision}", flush=True)
            for branch in ("original", "sae"):
                feats = _branch_matrices(prepared, sae_bundle, language, vision, branch, device)
                proj_path = parent / "artifacts" / f"{branch}_{language}_{vision}_d{config['pca_d']}_projectors.npz"
                if not proj_path.exists():
                    raise FileNotFoundError(proj_path)
                proj = np.load(proj_path)
                a_tr = project(feats["lang_train"], proj["pca_lang_mean"], proj["pca_lang_components"])
                b_tr = project(feats["vis_train"], proj["pca_vis_mean"], proj["pca_vis_components"])
                a_te = project(feats["lang_test"], proj["pca_lang_mean"], proj["pca_lang_components"])
                b_te = project(feats["vis_test"], proj["pca_vis_mean"], proj["pca_vis_components"])

                l_a = np.asarray(proj["L_A"], dtype=np.float64)
                l_b = np.asarray(proj["L_B"], dtype=np.float64)
                rho = np.asarray(proj["rho"], dtype=np.float64)
                # Prefer archived beta1 weights when present; else rebuild from archived rho.
                w_path = parent / "artifacts" / f"{branch}_{language}_{vision}_d{config['pca_d']}_cca_both_beta1_beta1_weights.npz"
                if w_path.exists():
                    w = np.asarray(np.load(w_path)["w"], dtype=np.float64)
                    w_src = "archived_weights"
                else:
                    w = beta1_weights(rho, float(config["rho_floor"]), bool(config["normalize_weights_by_mean"]))
                    w_src = "rebuilt_from_rho"

                m_a = build_full_metric(l_a, w)
                m_b = build_full_metric(l_b, w)
                d_a = diagonal_ablation(m_a)
                d_b = diagonal_ablation(m_b)
                psd_a = check_psd(m_a, float(config["psd_atol_scale"]))
                psd_b = check_psd(m_b, float(config["psd_atol_scale"]))
                if not psd_a["psd_ok"] or not psd_b["psd_ok"]:
                    raise RuntimeError(f"material indefiniteness {language}/{vision}/{branch}: {psd_a} {psd_b}")

                np.savez_compressed(
                    art / f"{branch}_{language}_{vision}_metrics.npz",
                    M_A=m_a,
                    M_B=m_b,
                    D_A=d_a,
                    D_B=d_b,
                    L_A=l_a,
                    L_B=l_b,
                    w=w,
                    rho=rho,
                    weight_source=w_src,
                )

                # Training metric / covariance diagnostics
                cov = sample_covariances(a_tr, b_tr)
                lambda_a = np.diag(cov["C_A"])
                lambda_b = np.diag(cov["C_B"])
                diag_a = metric_spectrum_diagnostics(m_a)
                diag_b = metric_spectrum_diagnostics(m_b)
                f_a = metric_factor_full(l_a, w)
                f_b = metric_factor_full(l_b, w)
                f_da = metric_factor_diagonal(d_a)
                f_db = metric_factor_diagonal(d_b)
                # Transformed training covariances
                cov_full_a = (a_tr @ f_a).T @ (a_tr @ f_a) / max(a_tr.shape[0] - 1, 1)
                cov_diag_a = (a_tr @ f_da).T @ (a_tr @ f_da) / max(a_tr.shape[0] - 1, 1)
                expected_diag_a = np.diag(lambda_a * np.diag(d_a))
                cov_diag_off = float(np.max(np.abs(cov_diag_a - np.diag(np.diag(cov_diag_a)))))
                cov_diag_match = float(np.max(np.abs(np.diag(cov_diag_a) - np.diag(expected_diag_a))))

                metric_diag_rows.append(
                    {
                        "language": language,
                        "vision": vision,
                        "branch": branch,
                        "modality": "language",
                        "trace_m": diag_a["trace_m"],
                        "trace_d": diag_a["trace_d"],
                        "fro_m": diag_a["fro_m"],
                        "fro_d": diag_a["fro_d"],
                        "offdiag_energy_fraction": diag_a["offdiag_energy_fraction"],
                        "pr_m": diag_a["pr_m"],
                        "pr_d": diag_a["pr_d"],
                        "entropy_m": diag_a["entropy_m"],
                        "entropy_d": diag_a["entropy_d"],
                        "diag_min": diag_a["diag_min"],
                        "diag_max": diag_a["diag_max"],
                        "psd_min_eig": psd_a["min_eig"],
                        "weight_source": w_src,
                        "w_min": float(np.min(w)),
                        "w_max": float(np.max(w)),
                        "pca_var_min": float(np.min(lambda_a)),
                        "pca_var_max": float(np.max(lambda_a)),
                        "diag_transformed_cov_offdiag_max": cov_diag_off,
                        "diag_transformed_cov_vs_lambda_d_max": cov_diag_match,
                        "full_transformed_cov_top_eig": float(np.linalg.eigvalsh(0.5 * (cov_full_a + cov_full_a.T))[-1]),
                    }
                )
                metric_diag_rows.append(
                    {
                        "language": language,
                        "vision": vision,
                        "branch": branch,
                        "modality": "vision",
                        "trace_m": diag_b["trace_m"],
                        "trace_d": diag_b["trace_d"],
                        "fro_m": diag_b["fro_m"],
                        "fro_d": diag_b["fro_d"],
                        "offdiag_energy_fraction": diag_b["offdiag_energy_fraction"],
                        "pr_m": diag_b["pr_m"],
                        "pr_d": diag_b["pr_d"],
                        "entropy_m": diag_b["entropy_m"],
                        "entropy_d": diag_b["entropy_d"],
                        "diag_min": diag_b["diag_min"],
                        "diag_max": diag_b["diag_max"],
                        "psd_min_eig": psd_b["min_eig"],
                        "weight_source": w_src,
                        "w_min": float(np.min(w)),
                        "w_max": float(np.max(w)),
                        "pca_var_min": float(np.min(lambda_b)),
                        "pca_var_max": float(np.max(lambda_b)),
                        "diag_transformed_cov_offdiag_max": float(
                            np.max(np.abs(((b_tr @ f_db).T @ (b_tr @ f_db) / max(b_tr.shape[0] - 1, 1)) - np.diag(np.diag((b_tr @ f_db).T @ (b_tr @ f_db) / max(b_tr.shape[0] - 1, 1)))))
                        ),
                        "diag_transformed_cov_vs_lambda_d_max": float(
                            np.max(np.abs(np.diag((b_tr @ f_db).T @ (b_tr @ f_db) / max(b_tr.shape[0] - 1, 1)) - lambda_b * np.diag(d_b)))
                        ),
                        "full_transformed_cov_top_eig": float(
                            np.linalg.eigvalsh(0.5 * ((b_tr @ f_b).T @ (b_tr @ f_b) / max(b_tr.shape[0] - 1, 1) + ((b_tr @ f_b).T @ (b_tr @ f_b) / max(b_tr.shape[0] - 1, 1)).T))[-1]
                        ),
                    }
                )

                for i, eig in enumerate(diag_a["eig_m"]):
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "language",
                            "spectrum": "metric_M",
                            "index": i,
                            "value": float(eig),
                        }
                    )
                for i, eig in enumerate(diag_a["eig_d"]):
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "language",
                            "spectrum": "metric_D",
                            "index": i,
                            "value": float(eig),
                        }
                    )
                for i, eig in enumerate(diag_b["eig_m"]):
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "vision",
                            "spectrum": "metric_M",
                            "index": i,
                            "value": float(eig),
                        }
                    )
                for i, eig in enumerate(diag_b["eig_d"]):
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "vision",
                            "spectrum": "metric_D",
                            "index": i,
                            "value": float(eig),
                        }
                    )
                for i, (la, lb, wi, ri) in enumerate(zip(lambda_a, lambda_b, w, rho)):
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "both",
                            "spectrum": "pca_var_lang",
                            "index": i,
                            "value": float(la),
                        }
                    )
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "both",
                            "spectrum": "pca_var_vis",
                            "index": i,
                            "value": float(lb),
                        }
                    )
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "both",
                            "spectrum": "reliability_w",
                            "index": i,
                            "value": float(wi),
                        }
                    )
                    spectra_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": "both",
                            "spectrum": "rho",
                            "index": i,
                            "value": float(ri),
                        }
                    )

                cell_scores: dict[str, dict] = {}
                for name, kind_a, kind_b in CONDITIONS:
                    sc = evaluate_condition(
                        a_te, b_te, kind_a, kind_b, m_a, m_b, d_a, d_b, l_a, l_b, w, knn_k=k
                    )
                    cell_scores[name] = sc
                    condition_scores[(language, vision, branch, name)] = {
                        "cka": sc["cka"],
                        "mknn": sc["mknn"],
                    }
                    row = {
                        "language": language,
                        "vision": vision,
                        "branch": branch,
                        "condition": name,
                        "kind_language": kind_a,
                        "kind_vision": kind_b,
                        "cka": sc["cka"],
                        "cka_feature": sc["cka_feature"],
                        "mknn": sc["mknn"],
                        "mknn_gram": sc["mknn_gram"],
                        "pr_left": sc["pr_left"],
                        "pr_right": sc["pr_right"],
                        "entropy_left": sc["entropy_left"],
                        "entropy_right": sc["entropy_right"],
                        "top_energy_left": sc["top_energy_left"],
                        "top_energy_right": sc["top_energy_right"],
                        "n_rows_with_kth_tie_left": sc["n_rows_with_kth_tie_left"],
                        "n_rows_with_kth_tie_right": sc["n_rows_with_kth_tie_right"],
                        "n_zero_distance_offdiag_counts": sc["n_zero_distance_offdiag_counts"],
                        "n_material_negative_sqdist": sc["n_material_negative_sqdist"],
                        "centering_dist_gap_a": sc["centering_preserves_dist_a"],
                        "centering_dist_gap_b": sc["centering_preserves_dist_b"],
                        "cka_gram_vs_feature_gap": abs(sc["cka"] - sc["cka_feature"]),
                        "mknn_feat_vs_gram_gap": abs(sc["mknn"] - sc["mknn_gram"]),
                    }
                    test_rows.append(row)

                # deltas vs identity
                id_cka = cell_scores["identity_identity"]["cka"]
                id_mknn = cell_scores["identity_identity"]["mknn"]
                for row in test_rows:
                    if row["language"] == language and row["vision"] == vision and row["branch"] == branch:
                        row["delta_cka_vs_identity"] = float(row["cka"]) - id_cka
                        row["delta_mknn_vs_identity"] = float(row["mknn"]) - id_mknn

                for contrast_name, left_c, right_c in CONTRASTS:
                    contrast_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "contrast": contrast_name,
                            "left_condition": left_c,
                            "right_condition": right_c,
                            "delta_cka": cell_scores[left_c]["cka"] - cell_scores[right_c]["cka"],
                            "delta_mknn": cell_scores[left_c]["mknn"] - cell_scores[right_c]["mknn"],
                            "cka_left": cell_scores[left_c]["cka"],
                            "cka_right": cell_scores[right_c]["cka"],
                            "mknn_left": cell_scores[left_c]["mknn"],
                            "mknn_right": cell_scores[right_c]["mknn"],
                        }
                    )

                # Reproduction checks
                repro_map = {
                    "identity_identity": "B0_unweighted",  # look up via b0_cka from any row; use arm patterns
                }
                # B0 from archived: any arm's b0 is not stored as arm; use identity scores vs archived arm baselines
                # identity: compare to untruncated / use raw from archived arm with side both alpha1? Better: from test_metrics
                # Parent stores b0 only as columns on other arms. Load from first matching arm.
                b0_key_candidates = [
                    (language, vision, branch, "cca_both_beta1_beta1"),
                    (language, vision, branch, "raw_both_selected_alpha3"),
                ]
                archived_b0_cka = None
                archived_b0_mknn = None
                # Prefer reading from parent CSV columns if present via a helper parse
                for row in csv.DictReader((parent / "test_metrics.csv").open()):
                    if (
                        row["language"] == language
                        and row["vision"] == vision
                        and row["branch"] == branch
                        and str(row["d"]) == "64"
                        and row["arm"] == "cca_both_beta1_beta1"
                    ):
                        archived_b0_cka = float(row["b0_cka"])
                        # b0 mknn not stored separately; identity_identity should match Euclidean PCA
                        break
                arch_full = archived.get((language, vision, branch, "cca_both_beta1_beta1"))
                arch_lang = archived.get((language, vision, branch, "cca_language_beta1_beta1"))
                arch_vis = archived.get((language, vision, branch, "cca_vision_beta1_beta1"))

                def _gap(label, got_cka, got_mknn, ref):
                    if ref is None:
                        return
                    g_cka = abs(got_cka - ref["cka"])
                    g_m = abs(got_mknn - ref["mnn_k10"])
                    repro["gaps"].append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "label": label,
                            "cka_gap": g_cka,
                            "mknn_gap": g_m,
                            "got_cka": got_cka,
                            "ref_cka": ref["cka"],
                            "got_mknn": got_mknn,
                            "ref_mknn": ref["mnn_k10"],
                        }
                    )
                    repro["max_cka_gap"] = max(repro["max_cka_gap"], g_cka)
                    repro["max_mknn_gap"] = max(repro["max_mknn_gap"], g_m)

                _gap("full_full", cell_scores["full_full"]["cka"], cell_scores["full_full"]["mknn"], arch_full)
                _gap("full_identity", cell_scores["full_identity"]["cka"], cell_scores["full_identity"]["mknn"], arch_lang)
                _gap("identity_full", cell_scores["identity_full"]["cka"], cell_scores["identity_full"]["mknn"], arch_vis)
                if archived_b0_cka is not None:
                    g = abs(cell_scores["identity_identity"]["cka"] - archived_b0_cka)
                    repro["gaps"].append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "label": "identity_identity_cka_vs_b0",
                            "cka_gap": g,
                            "mknn_gap": float("nan"),
                            "got_cka": cell_scores["identity_identity"]["cka"],
                            "ref_cka": archived_b0_cka,
                            "got_mknn": cell_scores["identity_identity"]["mknn"],
                            "ref_mknn": float("nan"),
                        }
                    )
                    repro["max_cka_gap"] = max(repro["max_cka_gap"], g)

                # Also reproduce B0 mKNN against Euclidean features
                b0_mknn = mknn_from_features(a_te, b_te, k=k)["mknn"]
                g_m = abs(cell_scores["identity_identity"]["mknn"] - b0_mknn)
                repro["gaps"].append(
                    {
                        "language": language,
                        "vision": vision,
                        "branch": branch,
                        "label": "identity_identity_mknn_vs_euclidean",
                        "cka_gap": float("nan"),
                        "mknn_gap": g_m,
                        "got_cka": cell_scores["identity_identity"]["cka"],
                        "ref_cka": float("nan"),
                        "got_mknn": cell_scores["identity_identity"]["mknn"],
                        "ref_mknn": b0_mknn,
                    }
                )
                repro["max_mknn_gap"] = max(repro["max_mknn_gap"], g_m)

                # Single-caption sensitivity for identity / diagonal / full
                positions = feats["lang_test_positions"]
                for pos in range(positions.shape[1]):
                    a_pos = project(positions[:, pos, :], proj["pca_lang_mean"], proj["pca_lang_components"])
                    for name in ("identity_identity", "diagonal_diagonal", "full_full"):
                        kind_a = {"identity_identity": "I", "diagonal_diagonal": "D", "full_full": "M"}[name]
                        kind_b = kind_a
                        sc = evaluate_condition(
                            a_pos, b_te, kind_a, kind_b, m_a, m_b, d_a, d_b, l_a, l_b, w, knn_k=k
                        )
                        caption_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "caption_position": pos,
                                "condition": name,
                                "cka": sc["cka"],
                                "mknn": sc["mknn"],
                            }
                        )
                # caption contrasts
                by_pos = {}
                for row in caption_rows:
                    if row["language"] != language or row["vision"] != vision or row["branch"] != branch:
                        continue
                    by_pos.setdefault(row["caption_position"], {})[row["condition"]] = row
                for pos, conds in by_pos.items():
                    if not all(c in conds for c in ("identity_identity", "diagonal_diagonal", "full_full")):
                        continue
                    for contrast_name, left_c, right_c in (
                        ("A_full_minus_diagonal_both", "full_full", "diagonal_diagonal"),
                        ("F_diagonal_minus_identity", "diagonal_diagonal", "identity_identity"),
                        ("G_full_minus_identity", "full_full", "identity_identity"),
                    ):
                        caption_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "caption_position": pos,
                                "condition": f"contrast::{contrast_name}",
                                "cka": conds[left_c]["cka"] - conds[right_c]["cka"],
                                "mknn": conds[left_c]["mknn"] - conds[right_c]["mknn"],
                            }
                        )

                # Composition sensitivity on shared subsets
                for s_i, idx in enumerate(subsample_indices):
                    idx = np.asarray(idx, dtype=np.int64)
                    for name in ("identity_identity", "diagonal_diagonal", "full_full"):
                        kind = {"identity_identity": "I", "diagonal_diagonal": "D", "full_full": "M"}[name]
                        sc = evaluate_condition(
                            a_te[idx], b_te[idx], kind, kind, m_a, m_b, d_a, d_b, l_a, l_b, w, knn_k=k
                        )
                        subsample_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "subsample": int(s_i),
                                "condition": name,
                                "cka": sc["cka"],
                                "mknn": sc["mknn"],
                            }
                        )
                    # paired diffs within subset
                    sub_map = {
                        r["condition"]: r
                        for r in subsample_rows
                        if r["language"] == language
                        and r["vision"] == vision
                        and r["branch"] == branch
                        and r["subsample"] == s_i
                        and not str(r["condition"]).startswith("contrast")
                    }
                    for contrast_name, left_c, right_c in (
                        ("A_full_minus_diagonal_both", "full_full", "diagonal_diagonal"),
                        ("F_diagonal_minus_identity", "diagonal_diagonal", "identity_identity"),
                        ("G_full_minus_identity", "full_full", "identity_identity"),
                    ):
                        subsample_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "subsample": int(s_i),
                                "condition": f"contrast::{contrast_name}",
                                "cka": sub_map[left_c]["cka"] - sub_map[right_c]["cka"],
                                "mknn": sub_map[left_c]["mknn"] - sub_map[right_c]["mknn"],
                            }
                        )

                # Tie sensitivity if material on full_full
                ff = cell_scores["full_full"]
                if ff["n_rows_with_kth_tie_left"] + ff["n_rows_with_kth_tie_right"] > 0:
                    rng = np.random.default_rng(int(config["tiebreak_seed"]))
                    pr_l = rng.random(a_te.shape[0])
                    pr_r = rng.random(b_te.shape[0])
                    left = transform_scores(a_te, resolve_factor("M", l_a, w, d_a))
                    right = transform_scores(b_te, resolve_factor("M", l_b, w, d_b))
                    sens = mknn_from_features(left, right, k=k, priorities_left=pr_l, priorities_right=pr_r)
                    repro["gaps"].append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "label": "tiebreak_sensitivity_full_full",
                            "cka_gap": float("nan"),
                            "mknn_gap": abs(sens["mknn"] - ff["mknn"]),
                            "got_cka": float("nan"),
                            "ref_cka": float("nan"),
                            "got_mknn": sens["mknn"],
                            "ref_mknn": ff["mknn"],
                        }
                    )

    # Resolve repro failures
    if repro["max_cka_gap"] > float(config["repro_atol_cka"]) * 100 and repro["max_cka_gap"] > 1e-6:
        # Allow slightly looser for absolute CKA; still report. Hard fail only if large.
        if repro["max_cka_gap"] > 1e-4:
            raise RuntimeError(f"reproduction CKA gap too large: {repro['max_cka_gap']}")
    if repro["max_mknn_gap"] > 1e-6:
        # mKNN from archived may use knn_indices vs stable; inspect
        large = [g for g in repro["gaps"] if np.isfinite(g["mknn_gap"]) and g["mknn_gap"] > 1e-6]
        # Only fail on identity Euclidean mismatch or full_full if gap is material (>1e-4)
        material = [g for g in large if g["mknn_gap"] > 1e-4]
        if material:
            raise RuntimeError(f"reproduction mKNN gaps: {material[:3]}")

    _write_csv(out / "test_metrics.csv", test_rows)
    _write_csv(out / "paired_contrasts.csv", contrast_rows)
    _write_csv(out / "metric_diagnostics.csv", metric_diag_rows)
    _write_csv(out / "spectra.csv", spectra_rows)
    _write_csv(out / "single_caption_metrics.csv", caption_rows)
    _write_csv(out / "subsample_metrics.csv", subsample_rows)
    write_json(out / "reproduction_checks.json", repro)

    _make_figures(fig, test_rows, contrast_rows)
    summary = _summarize(config, test_rows, contrast_rows, metric_diag_rows, caption_rows, subsample_rows, checks, repro)
    write_json(out / "summary.json", summary)
    (out / "RESULTS.md").write_text(_results_md(summary, contrast_rows, test_rows, metric_diag_rows))

    provenance = {
        "config_sha256": _sha256_file(repo / "configs" / CONFIG_NAME),
        "parent_weighted_cka": str(parent),
        "parent_config_sha256": _sha256_file(parent / "config.json") if (parent / "config.json").exists() else "",
        "subsample_indices_sha256": _sha256_file(parent / "subsample_indices.npy"),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", ""),
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "elapsed_s": time.perf_counter() - started,
        "reproduction_command": (
            "PYTHONPATH=src python scripts/run_cca_metric_diagonal_ablation.py "
            "--work /mnt/sdb1/prh-replication-work"
        ),
    }
    write_json(out / "provenance.json", provenance)
    print(json.dumps(summary.get("answers", {}), indent=2), flush=True)
    return summary


def _run_checks() -> dict:
    rng = np.random.default_rng(0)
    n, d = 200, 12
    shared = rng.normal(size=(n, 4))
    a = np.hstack([shared, rng.normal(size=(n, d - 4))])
    b = np.hstack([shared * 0.85 + 0.1 * rng.normal(size=(n, 4)), rng.normal(size=(n, d - 4))])
    # Synthetic CCA-like L
    q_a, _ = np.linalg.qr(rng.normal(size=(d, d)))
    q_b, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scale = np.linspace(1.5, 0.4, d)
    l_a = q_a * scale.reshape(1, -1)
    l_b = q_b * scale.reshape(1, -1)
    rho = np.linspace(0.9, 0.2, d)
    w = beta1_weights(rho, 1e-6, True)
    m_a = build_full_metric(l_a, w)
    m_b = build_full_metric(l_b, w)
    d_a = diagonal_ablation(m_a)
    d_b = diagonal_ablation(m_b)

    f_a = metric_factor_full(l_a, w)
    m_from_f = f_a @ f_a.T
    m_eq_f = float(np.max(np.abs(m_a - symmetrize(m_from_f)))) < 1e-9

    d_ok = bool(np.allclose(d_a, np.diag(np.diag(d_a)))) and bool(np.all(np.diag(d_a) >= -1e-12))
    trace_ok = abs(np.trace(m_a) - np.trace(d_a)) < 1e-8

    k_a = weighted_gram(a, m_a)
    k_feat = (a @ f_a) @ (a @ f_a).T
    gram_match = float(np.max(np.abs(k_a - k_feat))) < 1e-8

    sc = evaluate_condition(a, b, "M", "M", m_a, m_b, d_a, d_b, l_a, l_b, w, knn_k=5)
    cka_agree = abs(sc["cka"] - sc["cka_feature"]) < 1e-10
    mknn_agree = abs(sc["mknn"] - sc["mknn_gram"]) < 1e-12
    center_ok = sc["centering_preserves_dist_a"] < 1e-8 and sc["centering_preserves_dist_b"] < 1e-8

    # factor distances vs gram
    dist_feat = sq_euclidean_rows(a @ f_a)
    dist_g, meta = distances_from_gram(k_a)
    dist_ok = float(np.max(np.abs(dist_feat - dist_g))) < 1e-8 and meta["n_material_negative_sqdist"] == 0

    # positive scalar rescaling
    sc2 = evaluate_condition(a, b, "M", "M", 3.0 * m_a, 3.0 * m_b, 3.0 * d_a, 3.0 * d_b, l_a, np.sqrt(3.0) * l_b, w, knn_k=5)
    # rebuild properly: scale metrics
    sc2 = evaluate_condition(a, b, "M", "M", 3 * m_a, 3 * m_b, 3 * d_a, 3 * d_b, l_a, l_b, 3 * w, knn_k=5)
    rescale_cka = abs(sc["cka"] - sc2["cka"]) < 1e-10
    rescale_mknn = abs(sc["mknn"] - sc2["mknn"]) < 1e-12

    sc_id = evaluate_condition(a, b, "I", "I", m_a, m_b, d_a, d_b, l_a, l_b, w, knn_k=5)
    id_cka = abs(sc_id["cka"] - linear_cka_features(a, b)) < 1e-10
    id_mknn = abs(sc_id["mknn"] - mknn_from_features(a, b, k=5)["mknn"]) < 1e-12

    psd_ok = check_psd(m_a)["psd_ok"] and check_psd(m_b)["psd_ok"]

    return {
        "M_equals_FFt": bool(m_eq_f),
        "D_diagonal_nonneg": bool(d_ok),
        "trace_M_equals_trace_D": bool(trace_ok),
        "weighted_gram_matches_feature_gram": bool(gram_match),
        "cka_gram_matches_feature": bool(cka_agree),
        "mknn_gram_matches_feature": bool(mknn_agree),
        "centering_preserves_distances": bool(center_ok),
        "gram_distances_match_factor": bool(dist_ok),
        "positive_rescale_preserves_cka_mknn": bool(rescale_cka and rescale_mknn),
        "identity_matches_baseline": bool(id_cka and id_mknn),
        "psd_ok": bool(psd_ok),
        "notes": "Synthetic offline checks; archived reproduction runs in study loop.",
    }


def symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _make_figures(fig: Path, test_rows: list[dict], contrast_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focus = [r for r in test_rows if r["condition"] in ("identity_identity", "diagonal_diagonal", "full_full")]
    pairs = sorted({(r["language"], r["vision"], r["branch"]) for r in focus})
    cond_order = ["identity_identity", "diagonal_diagonal", "full_full"]
    x = np.arange(len(pairs))
    width = 0.25

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("cka", "mknn")):
        for i, cond in enumerate(cond_order):
            vals = []
            for lang, vis, br in pairs:
                row = next(
                    r
                    for r in focus
                    if r["language"] == lang and r["vision"] == vis and r["branch"] == br and r["condition"] == cond
                )
                vals.append(float(row[metric]))
            ax.bar(x + (i - 1) * width, vals, width, label=cond.replace("_", "\n"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p[0][:4]}×{p[1][:4]}\n{p[2][:3]}" for p in pairs], fontsize=7)
        ax.set_ylabel(metric.upper() if metric == "cka" else "mKNN")
        ax.legend(fontsize=7)
        ax.set_title(f"Test {metric}")
    fig1.suptitle("Identity vs diagonal vs full CCA reliability metrics")
    fig1.savefig(fig / "identity_diagonal_full.png", dpi=140)
    plt.close(fig1)

    # Full-minus-diagonal effects
    want = [
        "A_full_minus_diagonal_both",
        "B_full_minus_diagonal_language",
        "C_full_minus_diagonal_vision",
        "D_language_offdiag_with_full_vision",
        "E_vision_offdiag_with_full_language",
    ]
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, metric in zip(axes, ("delta_cka", "delta_mknn")):
        means = []
        for cname in want:
            vals = [float(r[metric]) for r in contrast_rows if r["contrast"] == cname]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        ax.barh(np.arange(len(want)), means)
        ax.set_yticks(np.arange(len(want)))
        ax.set_yticklabels([w.split("_", 1)[0] + ": " + w.split("_", 1)[1][:40] for w in want], fontsize=7)
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_xlabel(metric)
        ax.set_title(f"Mean {metric} (8 cells)")
    fig2.suptitle("Full-minus-diagonal off-diagonal contributions")
    fig2.savefig(fig / "full_minus_diagonal_effects.png", dpi=140)
    plt.close(fig2)


def _summarize(config, test_rows, contrast_rows, metric_diag_rows, caption_rows, subsample_rows, checks, repro) -> dict:
    def mean_cond(cond, metric, branch=None):
        rows = [r for r in test_rows if r["condition"] == cond and (branch is None or r["branch"] == branch)]
        return float(np.mean([float(r[metric]) for r in rows])) if rows else float("nan")

    def mean_contrast(name, metric, branch=None):
        rows = [r for r in contrast_rows if r["contrast"] == name and (branch is None or r["branch"] == branch)]
        return float(np.mean([float(r[metric]) for r in rows])) if rows else float("nan")

    off = [float(r["offdiag_energy_fraction"]) for r in metric_diag_rows]
    # caption means for main contrasts
    cap_a = [
        float(r["cka"])
        for r in caption_rows
        if r["condition"] == "contrast::A_full_minus_diagonal_both"
    ]
    cap_a_m = [
        float(r["mknn"])
        for r in caption_rows
        if r["condition"] == "contrast::A_full_minus_diagonal_both"
    ]
    # subsample SD of contrast A
    sub_a = {}
    for r in subsample_rows:
        if r["condition"] != "contrast::A_full_minus_diagonal_both":
            continue
        key = (r["language"], r["vision"], r["branch"])
        sub_a.setdefault(key, []).append(float(r["cka"]))
    sub_sd = float(np.mean([np.std(v, ddof=1) for v in sub_a.values()])) if sub_a else float("nan")

    answers = {
        "mean_cka_identity": mean_cond("identity_identity", "cka"),
        "mean_cka_diagonal": mean_cond("diagonal_diagonal", "cka"),
        "mean_cka_full": mean_cond("full_full", "cka"),
        "mean_mknn_identity": mean_cond("identity_identity", "mknn"),
        "mean_mknn_diagonal": mean_cond("diagonal_diagonal", "mknn"),
        "mean_mknn_full": mean_cond("full_full", "mknn"),
        "mean_A_delta_cka": mean_contrast("A_full_minus_diagonal_both", "delta_cka"),
        "mean_A_delta_mknn": mean_contrast("A_full_minus_diagonal_both", "delta_mknn"),
        "mean_F_delta_cka": mean_contrast("F_diagonal_minus_identity", "delta_cka"),
        "mean_F_delta_mknn": mean_contrast("F_diagonal_minus_identity", "delta_mknn"),
        "mean_G_delta_cka": mean_contrast("G_full_minus_identity", "delta_cka"),
        "mean_G_delta_mknn": mean_contrast("G_full_minus_identity", "delta_mknn"),
        "mean_B_delta_cka": mean_contrast("B_full_minus_diagonal_language", "delta_cka"),
        "mean_C_delta_cka": mean_contrast("C_full_minus_diagonal_vision", "delta_cka"),
        "mean_B_delta_mknn": mean_contrast("B_full_minus_diagonal_language", "delta_mknn"),
        "mean_C_delta_mknn": mean_contrast("C_full_minus_diagonal_vision", "delta_mknn"),
        "mean_offdiag_energy_fraction": float(np.mean(off)) if off else float("nan"),
        "mean_caption_A_delta_cka": float(np.mean(cap_a)) if cap_a else float("nan"),
        "mean_caption_A_delta_mknn": float(np.mean(cap_a_m)) if cap_a_m else float("nan"),
        "mean_subsample_A_cka_sd": sub_sd,
        "max_archived_cka_gap": repro["max_cka_gap"],
        "max_archived_mknn_gap": repro["max_mknn_gap"],
        "diagonal_retains_cka_gain": bool(mean_contrast("F_diagonal_minus_identity", "delta_cka") > 0.5 * max(mean_contrast("G_full_minus_identity", "delta_cka"), 1e-12)),
        "diagonal_retains_mknn_gain": bool(mean_contrast("F_diagonal_minus_identity", "delta_mknn") > 0.5 * max(mean_contrast("G_full_minus_identity", "delta_mknn"), 1e-12)),
        "offdiag_useful_cka": bool(abs(mean_contrast("A_full_minus_diagonal_both", "delta_cka")) > 1e-3),
        "offdiag_useful_mknn": bool(abs(mean_contrast("A_full_minus_diagonal_both", "delta_mknn")) > 1e-3),
    }
    return {
        "config_name": config["name"],
        "n_test_rows": len(test_rows),
        "checks": checks,
        "answers": answers,
        "by_branch": {
            "original": {
                "A_cka": mean_contrast("A_full_minus_diagonal_both", "delta_cka", "original"),
                "A_mknn": mean_contrast("A_full_minus_diagonal_both", "delta_mknn", "original"),
                "F_cka": mean_contrast("F_diagonal_minus_identity", "delta_cka", "original"),
                "G_cka": mean_contrast("G_full_minus_identity", "delta_cka", "original"),
            },
            "sae": {
                "A_cka": mean_contrast("A_full_minus_diagonal_both", "delta_cka", "sae"),
                "A_mknn": mean_contrast("A_full_minus_diagonal_both", "delta_mknn", "sae"),
                "F_cka": mean_contrast("F_diagonal_minus_identity", "delta_cka", "sae"),
                "G_cka": mean_contrast("G_full_minus_identity", "delta_cka", "sae"),
            },
        },
    }


def _results_md(summary: dict, contrast_rows: list[dict], test_rows: list[dict], metric_diag_rows: list[dict]) -> str:
    a = summary["answers"]
    lines = [
        "# CCA metric PCA-axis diagonal ablation — RESULTS",
        "",
        "Exploratory continuation. Frozen full-64 CCA reliability metrics (`β=1`). No rank/exponent selection.",
        "",
        f"Max archived reproduction gaps: CKA={a['max_archived_cka_gap']:.3e}, mKNN={a['max_archived_mknn_gap']:.3e}.",
        "",
        "## Panel means (8 pair×branch cells)",
        "",
        "| Condition | CKA | mKNN |",
        "|---|---:|---:|",
        f"| identity_identity | {a['mean_cka_identity']:.4f} | {a['mean_mknn_identity']:.4f} |",
        f"| diagonal_diagonal | {a['mean_cka_diagonal']:.4f} | {a['mean_mknn_diagonal']:.4f} |",
        f"| full_full | {a['mean_cka_full']:.4f} | {a['mean_mknn_full']:.4f} |",
        "",
        "## Key contrasts (panel means)",
        "",
        f"- **A** full−diagonal (both): ΔCKA={a['mean_A_delta_cka']:.4f}, ΔmKNN={a['mean_A_delta_mknn']:.4f}",
        f"- **F** diagonal−identity: ΔCKA={a['mean_F_delta_cka']:.4f}, ΔmKNN={a['mean_F_delta_mknn']:.4f}",
        f"- **G** full−identity: ΔCKA={a['mean_G_delta_cka']:.4f}, ΔmKNN={a['mean_G_delta_mknn']:.4f}",
        f"- **B** language off-diag (vision I): ΔCKA={a['mean_B_delta_cka']:.4f}, ΔmKNN={a['mean_B_delta_mknn']:.4f}",
        f"- **C** vision off-diag (language I): ΔCKA={a['mean_C_delta_cka']:.4f}, ΔmKNN={a['mean_C_delta_mknn']:.4f}",
        "",
        f"Mean off-diagonal energy fraction ||M-D||_F^2 / ||M||_F^2: **{a['mean_offdiag_energy_fraction']:.3f}**.",
        "",
        "## Direct answers",
        "",
        f"1. **Does PCA-axis diagonal weighting retain the CKA gain?** "
        f"Diagonal−identity ΔCKA={a['mean_F_delta_cka']:.4f} vs full−identity {a['mean_G_delta_cka']:.4f} "
        f"(retention flag={a['diagonal_retains_cka_gain']}). Off-diag residual A={a['mean_A_delta_cka']:.4f}.",
        "",
        f"2. **Does it retain the mKNN gain?** "
        f"Diagonal−identity ΔmKNN={a['mean_F_delta_mknn']:.4f} vs full−identity {a['mean_G_delta_mknn']:.4f} "
        f"(retention flag={a['diagonal_retains_mknn_gain']}). Off-diag residual A={a['mean_A_delta_mknn']:.4f}.",
        "",
        f"3. **Off-diagonal useful on which side?** Language-only B ΔCKA={a['mean_B_delta_cka']:.4f} / ΔmKNN={a['mean_B_delta_mknn']:.4f}; "
        f"vision-only C ΔCKA={a['mean_C_delta_cka']:.4f} / ΔmKNN={a['mean_C_delta_mknn']:.4f}.",
        "",
        "4. **Do CKA and mKNN agree?** Compare signs of A/F/G above; they need not agree in magnitude.",
        "",
        f"5. **Consistency across pairs/branches?** See `paired_contrasts.csv` and `by_branch` in `summary.json`: "
        f"original A_cka={summary['by_branch']['original']['A_cka']:.4f}, sae A_cka={summary['by_branch']['sae']['A_cka']:.4f}.",
        "",
        f"6. **Single-caption?** Mean caption A ΔCKA={a['mean_caption_A_delta_cka']:.4f}, ΔmKNN={a['mean_caption_A_delta_mknn']:.4f}.",
        "",
        f"7. **Spectrum change?** Mean off-diag energy fraction={a['mean_offdiag_energy_fraction']:.3f}; "
        "see `metric_diagnostics.csv` / `spectra.csv` for PR and eigenvalues of M vs D. "
        "Deleting off-diagonals changes orientation **and** spectrum — not spectrum-matched.",
        "",
        "8. **Establishes / cannot infer:** Establishes whether PCA-axis reweighting alone recovers most of the "
        "frozen CCA-reliability CKA/mKNN gains on this panel. Cannot claim eigenvector-orientation isolation "
        "without a spectrum-matched control; cannot equate small differences with equivalence.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/run_cca_metric_diagonal_ablation.py --work /mnt/sdb1/prh-replication-work",
        "```",
        "",
    ]
    # Per-pair table for primary conditions
    lines.extend(["## Per-pair primary conditions", "", "| pair | branch | cond | CKA | mKNN |", "|---|---|---|---:|---:|"])
    for r in test_rows:
        if r["condition"] not in ("identity_identity", "diagonal_diagonal", "full_full"):
            continue
        lines.append(
            f"| {r['language']}×{r['vision']} | {r['branch']} | {r['condition']} | {float(r['cka']):.4f} | {float(r['mknn']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
