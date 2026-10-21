"""Study runner: feature vs PCA vs full CCA reliability metric comparison."""

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
    build_full_metric,
    check_psd,
    diagonal_ablation,
    metric_factor_full,
    symmetrize,
    weighted_gram,
)
from prh_replication.crosscov_weighted_cka_study import (
    _branch_matrices,
    _load_saes,
    _prepare_features,
)
from prh_replication.feature_vs_pca_metric import (
    KINDS,
    KIND_LABELS,
    PRIMARY_BOTH,
    PRIMARY_CONTRASTS,
    ambient_block_energies,
    ambient_diag_from_factor,
    condition_name,
    evaluate_ambient_gram,
    evaluate_pca_pair,
    feature_restricted_metric,
    isotropy_deviation,
    metric_rank,
    reconstruct_parallel,
    score_from_grams,
)
from prh_replication.io_utils import write_json
from prh_replication.registry import Paths
from prh_replication.sae_crosscov import project
from prh_replication.sae_crosscov_study import _load_panel, _verify_manifest

CONFIG_NAME = "feature_vs_pca_metric_v1.json"


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


def _load_ablation_baselines(path: Path) -> dict:
    out = {}
    for row in csv.DictReader(path.open()):
        key = (row["language"], row["vision"], row["branch"], row["condition"])
        out[key] = {"cka": float(row["cka"]), "mknn": float(row["mknn"])}
    return out


def run(work: Path, repo: Path, force: bool = False) -> dict:
    started = time.perf_counter()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    out = paths.results / "feature_vs_pca_metric_v1"
    out.mkdir(parents=True, exist_ok=True)
    spec = repo / "results" / "feature_vs_pca_metric_v1" / "SPECIFICATION.md"
    if not spec.exists():
        spec = out / "SPECIFICATION.md"
    if spec.exists():
        (out / "SPECIFICATION.md").write_text(spec.read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if (out / "summary.json").exists() and not force:
        print("feature_vs_pca_metric already complete; pass --force to rerun", flush=True)
        return json.loads((out / "summary.json").read_text())

    parent = paths.results / "crosscov_weighted_cka_v1"
    ablation = paths.results / "cca_metric_diagonal_ablation_v1"
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
            "note": "Frozen-metric ablation; matched-subspace primary + ambient diagnostic.",
        },
    )

    panel = {"splits": {"train": train_ids, "test": test_ids}}
    full_panel_cfg = json.loads((parent / "config.json").read_text())
    loaded = _load_panel(paths, repo, full_panel_cfg)
    _verify_manifest(paths, repo, full_panel_cfg, loaded)
    if list(loaded["splits"]["train"]) != train_ids or list(loaded["splits"]["test"]) != test_ids:
        raise RuntimeError("split mismatch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    languages = list(config["languages"])
    visions = list(config["visions"])
    prepared = _prepare_features(paths, full_panel_cfg, panel, languages, visions, fit_idx)
    sae_bundle = _load_saes(paths.results / "sae_crosscov_v1" / "sae", languages, visions, int(config["sae_seed"]), device)
    ablation_base = _load_ablation_baselines(ablation / "test_metrics.csv") if (ablation / "test_metrics.csv").exists() else {}
    subsample_indices = np.load(parent / "subsample_indices.npy")

    checks = _run_checks()
    write_json(out / "verification_checks.json", {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in checks.items()})
    if not all(bool(v) for k, v in checks.items() if k != "notes"):
        raise RuntimeError(f"verification failed: {checks}")

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    primary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    ambient_rows: list[dict] = []
    metric_diag_rows: list[dict] = []
    spectra_rows: list[dict] = []
    caption_rows: list[dict] = []
    subsample_rows: list[dict] = []
    repro: dict = {"gaps": [], "max_cka_gap": 0.0, "max_mknn_gap": 0.0}
    k = int(config["knn_k"])

    for language in languages:
        for vision in visions:
            print(f"pair {language} × {vision}", flush=True)
            for branch in ("original", "sae"):
                feats = _branch_matrices(prepared, sae_bundle, language, vision, branch, device)
                proj_path = parent / "artifacts" / f"{branch}_{language}_{vision}_d{config['pca_d']}_projectors.npz"
                proj = np.load(proj_path)
                p_a = np.asarray(proj["pca_lang_components"], dtype=np.float64)
                p_b = np.asarray(proj["pca_vis_components"], dtype=np.float64)
                mu_a = np.asarray(proj["pca_lang_mean"], dtype=np.float64)
                mu_b = np.asarray(proj["pca_vis_mean"], dtype=np.float64)

                # Pre-PCA features
                x_tr = np.asarray(feats["lang_train"], dtype=np.float64)
                y_tr = np.asarray(feats["vis_train"], dtype=np.float64)
                x_te = np.asarray(feats["lang_test"], dtype=np.float64)
                y_te = np.asarray(feats["vis_test"], dtype=np.float64)
                x_c_te = x_te - mu_a.reshape(1, -1)
                y_c_te = y_te - mu_b.reshape(1, -1)
                a_te = x_c_te @ p_a
                b_te = y_c_te @ p_b
                a_tr = (x_tr - mu_a.reshape(1, -1)) @ p_a
                b_tr = (y_tr - mu_b.reshape(1, -1)) @ p_b

                # Orthonormality
                if float(np.max(np.abs(p_a.T @ p_a - np.eye(p_a.shape[1])))) > 1e-8:
                    raise RuntimeError("language P not orthonormal")
                if float(np.max(np.abs(p_b.T @ p_b - np.eye(p_b.shape[1])))) > 1e-8:
                    raise RuntimeError("vision P not orthonormal")

                l_a = np.asarray(proj["L_A"], dtype=np.float64)
                l_b = np.asarray(proj["L_B"], dtype=np.float64)
                rho = np.asarray(proj["rho"], dtype=np.float64)
                w_path = parent / "artifacts" / f"{branch}_{language}_{vision}_d{config['pca_d']}_cca_both_beta1_beta1_weights.npz"
                if w_path.exists():
                    w = np.asarray(np.load(w_path)["w"], dtype=np.float64)
                else:
                    from prh_replication.crosscov_weighted_cka import reliability_weights

                    w = reliability_weights(rho, 1.0, float(config["rho_floor"]), bool(config["normalize_weights_by_mean"]))["w"]

                m_a = build_full_metric(l_a, w)
                m_b = build_full_metric(l_b, w)
                d_pc_a = diagonal_ablation(m_a)
                d_pc_b = diagonal_ablation(m_b)
                f_a = metric_factor_full(l_a, w)
                f_b = metric_factor_full(l_b, w)
                d_feat_a = ambient_diag_from_factor(p_a, f_a)
                d_feat_b = ambient_diag_from_factor(p_b, f_b)
                n_a = feature_restricted_metric(p_a, d_feat_a)
                n_b = feature_restricted_metric(p_b, d_feat_b)

                for mat, name in ((m_a, "M_A"), (m_b, "M_B"), (n_a, "N_A"), (n_b, "N_B"), (d_pc_a, "D_PC_A"), (d_pc_b, "D_PC_B")):
                    psd = check_psd(mat, float(config["psd_atol_scale"]))
                    if not psd["psd_ok"]:
                        raise RuntimeError(f"PSD failure {language}/{vision}/{branch}/{name}: {psd}")

                # Trace identities
                tr_m_a = float(np.trace(m_a))
                tr_df_a = float(np.sum(d_feat_a))
                tr_m_b = float(np.trace(m_b))
                tr_df_b = float(np.sum(d_feat_b))
                if abs(tr_df_a - tr_m_a) > 1e-6 * max(1.0, abs(tr_m_a)):
                    raise RuntimeError(f"trace(D_feature) != trace(M) language: {tr_df_a} vs {tr_m_a}")
                if abs(tr_df_b - tr_m_b) > 1e-6 * max(1.0, abs(tr_m_b)):
                    raise RuntimeError(f"trace(D_feature) != trace(M) vision: {tr_df_b} vs {tr_m_b}")

                # Lifted full metric kernel == PCA metric kernel
                k_full_pca = weighted_gram(a_te, m_a)
                # X_parallel M via scores: same
                parallel_a, _ = reconstruct_parallel(x_c_te, p_a)
                # Efficient: A M A.T already; verify against (X_c P) M (P.T X_c.T)
                if float(np.max(np.abs(k_full_pca - (a_te @ m_a @ a_te.T)))) > 1e-8:
                    raise RuntimeError("internal gram mismatch")

                np.savez_compressed(
                    art / f"{branch}_{language}_{vision}_metrics.npz",
                    M_A=m_a,
                    M_B=m_b,
                    D_PC_A=d_pc_a,
                    D_PC_B=d_pc_b,
                    N_A=n_a,
                    N_B=n_b,
                    d_feature_A=d_feat_a,
                    d_feature_B=d_feat_b,
                    w=w,
                    rho=rho,
                    P_A=p_a,
                    P_B=p_b,
                )

                # Metric diagnostics per modality
                for modality, m, d_pc, n, d_feat, p, a_train, dim_p in (
                    ("language", m_a, d_pc_a, n_a, d_feat_a, p_a, a_tr, p_a.shape[0]),
                    ("vision", m_b, d_pc_b, n_b, d_feat_b, p_b, b_tr, p_b.shape[0]),
                ):
                    eigs_m = np.linalg.eigvalsh(symmetrize(m))[::-1]
                    eigs_n = np.linalg.eigvalsh(symmetrize(n))[::-1]
                    eigs_d = np.sort(np.diag(d_pc))[::-1]
                    off_n = n - np.diag(np.diag(n))
                    fro_n = float(np.linalg.norm(n, ord="fro"))
                    off_frac = float(np.sum(off_n * off_n) / max(fro_n * fro_n, 1e-30))
                    # PCA-diagonal transformed cov check
                    ac = a_train - a_train.mean(0, keepdims=True)
                    c = (ac.T @ ac) / max(ac.shape[0] - 1, 1)
                    lambda_i = np.diag(c)
                    d_ii = np.diag(d_pc)
                    f_d = np.diag(np.sqrt(np.maximum(d_ii, 0.0)))
                    z = a_train @ f_d
                    zc = z - z.mean(0, keepdims=True)
                    cov_d = (zc.T @ zc) / max(zc.shape[0] - 1, 1)
                    pc_cov_gap = float(np.max(np.abs(np.diag(cov_d) - lambda_i * d_ii)))
                    # Feature-restricted transformed cov spectrum (direct)
                    fn = np.linalg.eigvalsh(symmetrize(_cov_after_metric(a_train, n)))[::-1]

                    metric_diag_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "modality": modality,
                            "p": int(dim_p),
                            "d_pca": int(p.shape[1]),
                            "trace_M": float(np.trace(m)),
                            "trace_D_PC": float(np.trace(d_pc)),
                            "trace_D_feature": float(np.sum(d_feat)),
                            "trace_N": float(np.trace(n)),
                            "fro_M": float(np.linalg.norm(m, ord="fro")),
                            "fro_D_PC": float(np.linalg.norm(d_pc, ord="fro")),
                            "fro_N": fro_n,
                            "offdiag_energy_N": off_frac,
                            "rank_M": metric_rank(m),
                            "rank_D_PC": metric_rank(d_pc),
                            "rank_N": metric_rank(n),
                            "d_feat_min": float(np.min(d_feat)),
                            "d_feat_max": float(np.max(d_feat)),
                            "d_feat_mean": float(np.mean(d_feat)),
                            "d_pc_min": float(np.min(d_ii)),
                            "d_pc_max": float(np.max(d_ii)),
                            "isotropy_dev_N": isotropy_deviation(n),
                            "pc_diag_cov_gap": pc_cov_gap,
                            "top_eig_M": float(eigs_m[0]),
                            "top_eig_N": float(eigs_n[0]),
                            "top_eig_D_PC": float(eigs_d[0]),
                            "top_eig_featR_cov": float(fn[0]) if fn.size else float("nan"),
                        }
                    )
                    for i, v in enumerate(eigs_m):
                        spectra_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "modality": modality,
                                "spectrum": "M",
                                "index": i,
                                "value": float(v),
                            }
                        )
                    for i, v in enumerate(eigs_n):
                        spectra_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "modality": modality,
                                "spectrum": "N_feature",
                                "index": i,
                                "value": float(v),
                            }
                        )
                    for i, v in enumerate(eigs_d):
                        spectra_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "modality": modality,
                                "spectrum": "D_PC",
                                "index": i,
                                "value": float(v),
                            }
                        )
                    # subsample of feature diagonal weights (store summary via percentiles in spectra as special)
                    for q in (0, 25, 50, 75, 100):
                        spectra_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "modality": modality,
                                "spectrum": "d_feature_percentile",
                                "index": q,
                                "value": float(np.percentile(d_feat, q)),
                            }
                        )

                # --- Primary 4×4 matched-subspace ---
                cell: dict[str, dict] = {}
                for ka in KINDS:
                    for kb in KINDS:
                        name = condition_name(ka, kb)
                        sc = evaluate_pca_pair(
                            a_te, b_te, ka, kb, m_a, m_b, d_pc_a, d_pc_b, n_a, n_b, knn_k=k
                        )
                        cell[name] = sc
                        primary_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "condition": name,
                                "kind_language": ka,
                                "kind_vision": kb,
                                "label_language": KIND_LABELS[ka],
                                "label_vision": KIND_LABELS[kb],
                                "panel": "matched_subspace",
                                "cka": sc["cka"],
                                "mknn": sc["mknn"],
                                "cka_feature": sc["cka_feature"],
                                "cka_gap": sc["cka_gap"],
                                "mknn_gap": sc["mknn_gap"],
                                "pr_left": sc["pr_left"],
                                "pr_right": sc["pr_right"],
                                "entropy_left": sc["entropy_left"],
                                "entropy_right": sc["entropy_right"],
                                "top_energy_left": sc["top_energy_left"],
                                "top_energy_right": sc["top_energy_right"],
                                "n_material_negative_sqdist": sc["n_material_negative_sqdist"],
                                "n_rows_with_kth_tie_left": sc["n_rows_with_kth_tie_left"],
                                "n_rows_with_kth_tie_right": sc["n_rows_with_kth_tie_right"],
                            }
                        )

                id_cka = cell["I_I"]["cka"]
                id_mknn = cell["I_I"]["mknn"]
                for row in primary_rows:
                    if row["language"] == language and row["vision"] == vision and row["branch"] == branch and row["panel"] == "matched_subspace":
                        row["delta_cka_vs_II"] = float(row["cka"]) - id_cka
                        row["delta_mknn_vs_II"] = float(row["mknn"]) - id_mknn

                for cname, left_c, right_c in PRIMARY_CONTRASTS:
                    contrast_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "contrast": cname,
                            "left": left_c,
                            "right": right_c,
                            "delta_cka": cell[left_c]["cka"] - cell[right_c]["cka"],
                            "delta_mknn": cell[left_c]["mknn"] - cell[right_c]["mknn"],
                            "cka_left": cell[left_c]["cka"],
                            "cka_right": cell[right_c]["cka"],
                            "mknn_left": cell[left_c]["mknn"],
                            "mknn_right": cell[right_c]["mknn"],
                        }
                    )
                # Mixed one-sided: change language holding vision I / FULL; change vision holding language I / FULL
                for hold, hold_kind in (("vision_I", "I"), ("vision_FULL", "FULL")):
                    for new_kind in ("PC", "FEAT_R", "FULL"):
                        if new_kind == hold_kind:
                            continue
                        left = condition_name(new_kind, hold_kind)
                        right = condition_name("I", hold_kind)
                        contrast_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "contrast": f"lang_{new_kind}_vs_I_hold_{hold}",
                                "left": left,
                                "right": right,
                                "delta_cka": cell[left]["cka"] - cell[right]["cka"],
                                "delta_mknn": cell[left]["mknn"] - cell[right]["mknn"],
                                "cka_left": cell[left]["cka"],
                                "cka_right": cell[right]["cka"],
                                "mknn_left": cell[left]["mknn"],
                                "mknn_right": cell[right]["mknn"],
                            }
                        )
                for hold, hold_kind in (("language_I", "I"), ("language_FULL", "FULL")):
                    for new_kind in ("PC", "FEAT_R", "FULL"):
                        if new_kind == hold_kind:
                            continue
                        left = condition_name(hold_kind, new_kind)
                        right = condition_name(hold_kind, "I")
                        contrast_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "contrast": f"vis_{new_kind}_vs_I_hold_{hold}",
                                "left": left,
                                "right": right,
                                "delta_cka": cell[left]["cka"] - cell[right]["cka"],
                                "delta_mknn": cell[left]["mknn"] - cell[right]["mknn"],
                                "cka_left": cell[left]["cka"],
                                "cka_right": cell[right]["cka"],
                                "mknn_left": cell[left]["mknn"],
                                "mknn_right": cell[right]["mknn"],
                            }
                        )

                # Reproduction vs prior ablation
                map_repro = {
                    "I_I": "identity_identity",
                    "PC_PC": "diagonal_diagonal",
                    "FULL_FULL": "full_full",
                }
                for here, there in map_repro.items():
                    ref = ablation_base.get((language, vision, branch, there))
                    if ref is None:
                        continue
                    g_cka = abs(cell[here]["cka"] - ref["cka"])
                    g_m = abs(cell[here]["mknn"] - ref["mknn"])
                    repro["gaps"].append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "label": f"{here}_vs_{there}",
                            "cka_gap": g_cka,
                            "mknn_gap": g_m,
                        }
                    )
                    repro["max_cka_gap"] = max(repro["max_cka_gap"], g_cka)
                    repro["max_mknn_gap"] = max(repro["max_mknn_gap"], g_m)
                # Mixed I/PC/FULL from ablation where available
                for here, there in (
                    ("FULL_I", "full_identity"),
                    ("I_FULL", "identity_full"),
                    ("PC_I", "diagonal_identity"),
                    ("I_PC", "identity_diagonal"),
                    ("FULL_PC", "full_diagonal"),
                    ("PC_FULL", "diagonal_full"),
                ):
                    ref = ablation_base.get((language, vision, branch, there))
                    if ref is None:
                        continue
                    g_cka = abs(cell[here]["cka"] - ref["cka"])
                    g_m = abs(cell[here]["mknn"] - ref["mknn"])
                    repro["gaps"].append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "label": f"{here}_vs_{there}",
                            "cka_gap": g_cka,
                            "mknn_gap": g_m,
                        }
                    )
                    repro["max_cka_gap"] = max(repro["max_cka_gap"], g_cka)
                    repro["max_mknn_gap"] = max(repro["max_mknn_gap"], g_m)

                # --- Ambient diagnostic ---
                k_a_I = evaluate_ambient_gram(x_c_te, None, "I")
                k_b_I = evaluate_ambient_gram(y_c_te, None, "I")
                k_a_D = evaluate_ambient_gram(x_c_te, d_feat_a, "D")
                k_b_D = evaluate_ambient_gram(y_c_te, d_feat_b, "D")
                # Restricted via reconstructed features
                x_par, _ = reconstruct_parallel(x_c_te, p_a)
                y_par, _ = reconstruct_parallel(y_c_te, p_b)
                k_a_R = evaluate_ambient_gram(x_par, d_feat_a, "D")
                k_b_R = evaluate_ambient_gram(y_par, d_feat_b, "D")
                # Verify restricted matches FEAT_R FEAT_R
                sc_rr = score_from_grams(k_a_R, k_b_R, knn_k=k)
                if abs(sc_rr["cka"] - cell["FEAT_R_FEAT_R"]["cka"]) > 1e-8:
                    raise RuntimeError("restricted ambient gram != FEAT_R PCA metric")
                if abs(sc_rr["mknn"] - cell["FEAT_R_FEAT_R"]["mknn"]) > 1e-10:
                    raise RuntimeError("restricted ambient mknn != FEAT_R")

                ambient_conds = {
                    "ambient_I_I": (k_a_I, k_b_I),
                    "ambient_D_D": (k_a_D, k_b_D),
                    "restricted_D_D": (k_a_R, k_b_R),
                    "ambient_D_restricted": (k_a_D, k_b_R),
                    "restricted_ambient_D": (k_a_R, k_b_D),
                }
                energies_a = ambient_block_energies(x_c_te, p_a, d_feat_a)
                energies_b = ambient_block_energies(y_c_te, p_b, d_feat_b)
                for aname, (ka, kb) in ambient_conds.items():
                    sc = score_from_grams(ka, kb, knn_k=k)
                    ambient_rows.append(
                        {
                            "language": language,
                            "vision": vision,
                            "branch": branch,
                            "condition": aname,
                            "panel": "ambient_diagnostic",
                            "cka": sc["cka"],
                            "mknn": sc["mknn"],
                            "pr_left": sc["pr_left"],
                            "pr_right": sc["pr_right"],
                            "entropy_left": sc["entropy_left"],
                            "entropy_right": sc["entropy_right"],
                            "top_energy_left": sc["top_energy_left"],
                            "top_energy_right": sc["top_energy_right"],
                            "fro_parallel_lang": energies_a["fro_parallel"],
                            "fro_perp_lang": energies_a["fro_perp"],
                            "fro_cross_lang": energies_a["fro_cross"],
                            "fro_parallel_vis": energies_b["fro_parallel"],
                            "fro_perp_vis": energies_b["fro_perp"],
                            "fro_cross_vis": energies_b["fro_cross"],
                        }
                    )

                # Single-caption: four primary both-sided
                positions = feats["lang_test_positions"]
                for pos in range(positions.shape[1]):
                    a_pos = project(positions[:, pos, :], mu_a, p_a)
                    for both_name, ka, kb in PRIMARY_BOTH:
                        sc = evaluate_pca_pair(a_pos, b_te, ka, kb, m_a, m_b, d_pc_a, d_pc_b, n_a, n_b, knn_k=k)
                        caption_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "caption_position": pos,
                                "condition": both_name,
                                "cka": sc["cka"],
                                "mknn": sc["mknn"],
                            }
                        )

                # Subsamples for primary both-sided + key contrasts
                for s_i, idx in enumerate(subsample_indices):
                    idx = np.asarray(idx, dtype=np.int64)
                    sub_cell = {}
                    for both_name, ka, kb in PRIMARY_BOTH:
                        sc = evaluate_pca_pair(
                            a_te[idx], b_te[idx], ka, kb, m_a, m_b, d_pc_a, d_pc_b, n_a, n_b, knn_k=k
                        )
                        sub_cell[both_name] = sc
                        subsample_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "subsample": int(s_i),
                                "condition": both_name,
                                "cka": sc["cka"],
                                "mknn": sc["mknn"],
                            }
                        )
                    for cname, left_c, right_c in PRIMARY_CONTRASTS:
                        subsample_rows.append(
                            {
                                "language": language,
                                "vision": vision,
                                "branch": branch,
                                "subsample": int(s_i),
                                "condition": f"contrast::{cname}",
                                "cka": sub_cell[left_c]["cka"] - sub_cell[right_c]["cka"],
                                "mknn": sub_cell[left_c]["mknn"] - sub_cell[right_c]["mknn"],
                            }
                        )

    if repro["max_cka_gap"] > 1e-4 or repro["max_mknn_gap"] > 1e-4:
        raise RuntimeError(f"reproduction gaps too large: cka={repro['max_cka_gap']} mknn={repro['max_mknn_gap']}")

    _write_csv(out / "primary_metrics.csv", primary_rows)
    _write_csv(out / "paired_contrasts.csv", contrast_rows)
    _write_csv(out / "ambient_diagnostics.csv", ambient_rows)
    _write_csv(out / "metric_diagnostics.csv", metric_diag_rows)
    _write_csv(out / "spectra.csv", spectra_rows)
    _write_csv(out / "single_caption_metrics.csv", caption_rows)
    _write_csv(out / "subsample_metrics.csv", subsample_rows)
    write_json(out / "reproduction_checks.json", repro)

    _make_figures(fig, primary_rows, ambient_rows)
    summary = _summarize(config, primary_rows, contrast_rows, ambient_rows, caption_rows, subsample_rows, checks, repro)
    write_json(out / "summary.json", summary)
    (out / "RESULTS.md").write_text(_results_md(summary, primary_rows, contrast_rows, ambient_rows))

    write_json(
        out / "provenance.json",
        {
            "config_sha256": _sha256_file(repo / "configs" / CONFIG_NAME),
            "parent_weighted_cka": str(parent),
            "parent_diagonal_ablation": str(ablation),
            "subsample_indices_sha256": _sha256_file(parent / "subsample_indices.npy"),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", ""),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "elapsed_s": time.perf_counter() - started,
            "reproduction_command": (
                "PYTHONPATH=src python scripts/run_feature_vs_pca_metric.py "
                "--work /mnt/sdb1/prh-replication-work --repo /mnt/sdb1/prh-replication-work/repo"
            ),
        },
    )
    print(json.dumps(summary.get("answers", {}), indent=2), flush=True)
    return summary


def _cov_after_metric(scores: np.ndarray, metric: np.ndarray) -> np.ndarray:
    from prh_replication.feature_vs_pca_metric import psd_factor

    fac = psd_factor(metric)
    z = scores @ fac
    zc = z - z.mean(0, keepdims=True)
    return (zc.T @ zc) / max(zc.shape[0] - 1, 1)


def _run_checks() -> dict:
    rng = np.random.default_rng(0)
    n, p, d = 120, 40, 8
    p_mat, _ = np.linalg.qr(rng.normal(size=(p, d)))
    # Ensure first d columns orthonormal spanning
    p_mat = p_mat[:, :d]
    f = rng.normal(size=(d, d))
    m = symmetrize(f @ f.T)
    from prh_replication.feature_vs_pca_metric import psd_factor as _pf

    fac = _pf(m)
    d_feat = ambient_diag_from_factor(p_mat, fac)
    # Explicit small construction
    m_amb = p_mat @ m @ p_mat.T
    d_exp = np.diag(m_amb)
    diag_ok = float(np.max(np.abs(d_feat - d_exp))) < 1e-10

    n_feat = feature_restricted_metric(p_mat, d_feat)
    x = rng.normal(size=(n, p))
    a = x @ p_mat
    k1 = weighted_gram(a, n_feat)
    x_par, x_perp = reconstruct_parallel(x, p_mat)
    k2 = evaluate_ambient_gram(x_par, d_feat, "D")
    recon_ok = float(np.max(np.abs(k1 - k2))) < 1e-8

    # Lifted full == PCA full on scores
    k_full = weighted_gram(a, m)
    k_lift_par = (x_par @ p_mat) @ m @ (x_par @ p_mat).T  # x_par @ p = a
    # Actually x_par = a @ P.T so x_par @ anything in range
    lift_ok = float(np.max(np.abs(k_full - a @ m @ a.T))) < 1e-10

    # Restricted invariant to perp perturbation; ambient can respond
    delta = rng.normal(size=(n, p))
    delta = delta - (delta @ p_mat) @ p_mat.T
    x3 = x + delta
    a3 = x3 @ p_mat
    inv_ok = float(np.max(np.abs(weighted_gram(a, n_feat) - weighted_gram(a3, n_feat)))) < 1e-8
    k_amb = evaluate_ambient_gram(x, d_feat, "D")
    k_amb3 = evaluate_ambient_gram(x3, d_feat, "D")
    amb_resp = float(np.max(np.abs(k_amb - k_amb3))) > 1e-8

    pt_p = float(np.max(np.abs(p_mat.T @ p_mat - np.eye(d)))) < 1e-10
    psd_ok = check_psd(m)["psd_ok"] and check_psd(n_feat)["psd_ok"]
    tr_ok = abs(float(np.sum(d_feat)) - float(np.trace(m))) < 1e-8

    b = a * 0.9 + 0.1 * rng.normal(size=a.shape)
    sc = evaluate_pca_pair(a, b, "FULL", "FULL", m, m, diagonal_ablation(m), diagonal_ablation(m), n_feat, n_feat, knn_k=5)
    sc1 = evaluate_pca_pair(a, b, "FULL", "PC", m, m, diagonal_ablation(m), diagonal_ablation(m), n_feat, n_feat, knn_k=5)
    sc2 = evaluate_pca_pair(a, b, "FULL", "PC", 5 * m, 5 * m, 5 * diagonal_ablation(m), 5 * diagonal_ablation(m), 5 * n_feat, 5 * n_feat, knn_k=5)
    scale_ok = abs(sc1["cka"] - sc2["cka"]) < 1e-10 and abs(sc1["mknn"] - sc2["mknn"]) < 1e-12

    return {
        "P_T_P_identity": bool(pt_p),
        "efficient_diag_matches_explicit": bool(diag_ok),
        "featR_gram_matches_reconstructed": bool(recon_ok),
        "full_lift_kernel_identity": bool(lift_ok),
        "restricted_invariant_to_perp": bool(inv_ok),
        "ambient_responds_to_perp": bool(amb_resp),
        "psd_ok": bool(psd_ok),
        "trace_D_feature_equals_trace_M": bool(tr_ok),
        "positive_rescale_preserves_cka_mknn": bool(scale_ok),
        "gram_feature_cka_agree": bool(sc["cka_gap"] < 1e-10),
        "notes": "Synthetic offline checks.",
    }


def _make_figures(fig: Path, primary_rows: list[dict], ambient_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = [r for r in primary_rows if r["condition"] in ("I_I", "PC_PC", "FEAT_R_FEAT_R", "FULL_FULL")]
    pairs = sorted({(r["language"], r["vision"], r["branch"]) for r in both})
    order = ["I_I", "PC_PC", "FEAT_R_FEAT_R", "FULL_FULL"]
    x = np.arange(len(pairs))
    width = 0.2
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("cka", "mknn")):
        for i, cond in enumerate(order):
            vals = [
                float(
                    next(
                        r[metric]
                        for r in both
                        if r["language"] == p[0] and r["vision"] == p[1] and r["branch"] == p[2] and r["condition"] == cond
                    )
                )
                for p in pairs
            ]
            ax.bar(x + (i - 1.5) * width, vals, width, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p[0][:4]}×{p[1][:3]}\n{p[2][:3]}" for p in pairs], fontsize=7)
        ax.legend(fontsize=7)
        ax.set_title(metric)
    fig1.suptitle("Primary both-sided conditions (matched subspace)")
    fig1.savefig(fig / "primary_both_sided.png", dpi=140)
    plt.close(fig1)

    # 4x4 heatmaps averaged over panel
    kinds = list(KINDS)
    for metric in ("cka", "mknn"):
        mat = np.zeros((4, 4))
        for i, ka in enumerate(kinds):
            for j, kb in enumerate(kinds):
                name = condition_name(ka, kb)
                vals = [float(r[metric]) for r in primary_rows if r["condition"] == name]
                mat[i, j] = float(np.mean(vals)) if vals else np.nan
        fig2, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(kinds, fontsize=8)
        ax.set_yticklabels(kinds, fontsize=8)
        ax.set_xlabel("vision")
        ax.set_ylabel("language")
        ax.set_title(f"Mean {metric} 4×4")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
        fig2.colorbar(im, ax=ax, fraction=0.046)
        fig2.savefig(fig / f"heatmap_{metric}.png", dpi=140)
        plt.close(fig2)

    # Ambient vs restricted
    focus = ["ambient_I_I", "ambient_D_D", "restricted_D_D"]
    cells = sorted({(r["language"], r["vision"], r["branch"]) for r in ambient_rows})
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    x = np.arange(len(cells))
    width = 0.25
    for ax, metric in zip(axes, ("cka", "mknn")):
        for i, cond in enumerate(focus):
            vals = [
                float(
                    next(
                        r[metric]
                        for r in ambient_rows
                        if r["language"] == c[0] and r["vision"] == c[1] and r["branch"] == c[2] and r["condition"] == cond
                    )
                )
                for c in cells
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0][:4]}×{c[1][:3]}\n{c[2][:3]}" for c in cells], fontsize=7)
        ax.legend(fontsize=6)
        ax.set_title(metric)
    fig3.suptitle("Ambient-space diagnostic vs restricted feature diagonal")
    fig3.savefig(fig / "ambient_vs_restricted.png", dpi=140)
    plt.close(fig3)


def _summarize(config, primary_rows, contrast_rows, ambient_rows, caption_rows, subsample_rows, checks, repro) -> dict:
    def mean_cond(cond, metric):
        vals = [float(r[metric]) for r in primary_rows if r["condition"] == cond]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_contrast(name, metric, branch=None):
        vals = [
            float(r[metric])
            for r in contrast_rows
            if r["contrast"] == name and (branch is None or r["branch"] == branch)
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_amb(cond, metric):
        vals = [float(r[metric]) for r in ambient_rows if r["condition"] == cond]
        return float(np.mean(vals)) if vals else float("nan")

    g = mean_contrast("3_FULL_minus_I", "delta_cka")
    answers = {
        "mean_cka_I": mean_cond("I_I", "cka"),
        "mean_cka_PC": mean_cond("PC_PC", "cka"),
        "mean_cka_FEAT_R": mean_cond("FEAT_R_FEAT_R", "cka"),
        "mean_cka_FULL": mean_cond("FULL_FULL", "cka"),
        "mean_mknn_I": mean_cond("I_I", "mknn"),
        "mean_mknn_PC": mean_cond("PC_PC", "mknn"),
        "mean_mknn_FEAT_R": mean_cond("FEAT_R_FEAT_R", "mknn"),
        "mean_mknn_FULL": mean_cond("FULL_FULL", "mknn"),
        "c1_featR_minus_I_cka": mean_contrast("1_FEAT_R_minus_I", "delta_cka"),
        "c1_featR_minus_I_mknn": mean_contrast("1_FEAT_R_minus_I", "delta_mknn"),
        "c2_pc_minus_I_cka": mean_contrast("2_PC_minus_I", "delta_cka"),
        "c2_pc_minus_I_mknn": mean_contrast("2_PC_minus_I", "delta_mknn"),
        "c3_full_minus_I_cka": mean_contrast("3_FULL_minus_I", "delta_cka"),
        "c3_full_minus_I_mknn": mean_contrast("3_FULL_minus_I", "delta_mknn"),
        "c4_featR_minus_pc_cka": mean_contrast("4_FEAT_R_minus_PC", "delta_cka"),
        "c4_featR_minus_pc_mknn": mean_contrast("4_FEAT_R_minus_PC", "delta_mknn"),
        "c5_full_minus_featR_cka": mean_contrast("5_FULL_minus_FEAT_R", "delta_cka"),
        "c5_full_minus_featR_mknn": mean_contrast("5_FULL_minus_FEAT_R", "delta_mknn"),
        "c6_full_minus_pc_cka": mean_contrast("6_FULL_minus_PC", "delta_cka"),
        "c6_full_minus_pc_mknn": mean_contrast("6_FULL_minus_PC", "delta_mknn"),
        "frac_featR_of_full_cka": (
            mean_contrast("1_FEAT_R_minus_I", "delta_cka") / g if abs(g) > 1e-12 else float("nan")
        ),
        "frac_pc_of_full_cka": (
            mean_contrast("2_PC_minus_I", "delta_cka") / g if abs(g) > 1e-12 else float("nan")
        ),
        "mean_ambient_I_cka": mean_amb("ambient_I_I", "cka"),
        "mean_ambient_D_cka": mean_amb("ambient_D_D", "cka"),
        "mean_restricted_D_cka": mean_amb("restricted_D_D", "cka"),
        "mean_ambient_I_mknn": mean_amb("ambient_I_I", "mknn"),
        "mean_ambient_D_mknn": mean_amb("ambient_D_D", "mknn"),
        "mean_restricted_D_mknn": mean_amb("restricted_D_D", "mknn"),
        "ambient_minus_restricted_cka": mean_amb("ambient_D_D", "cka") - mean_amb("restricted_D_D", "cka"),
        "ambient_minus_restricted_mknn": mean_amb("ambient_D_D", "mknn") - mean_amb("restricted_D_D", "mknn"),
        "max_archived_cka_gap": repro["max_cka_gap"],
        "max_archived_mknn_gap": repro["max_mknn_gap"],
        "mean_caption_featR_minus_I_cka": float(
            np.mean(
                [
                    float(r["cka"])
                    for r in caption_rows
                    if r["condition"] == "FEAT_R_FEAT_R"
                ]
            )
            - np.mean([float(r["cka"]) for r in caption_rows if r["condition"] == "I_I"])
        )
        if caption_rows
        else float("nan"),
    }
    # subsample SD of contrast 5
    sub = {}
    for r in subsample_rows:
        if r["condition"] != "contrast::5_FULL_minus_FEAT_R":
            continue
        key = (r["language"], r["vision"], r["branch"])
        sub.setdefault(key, []).append(float(r["cka"]))
    answers["mean_subsample_full_minus_featR_cka_sd"] = (
        float(np.mean([np.std(v, ddof=1) for v in sub.values()])) if sub else float("nan")
    )
    return {
        "config_name": config["name"],
        "n_primary_rows": len(primary_rows),
        "checks": {k: (bool(v) if not isinstance(v, str) else v) for k, v in checks.items()},
        "answers": answers,
        "by_branch": {
            br: {
                "featR_minus_I_cka": mean_contrast("1_FEAT_R_minus_I", "delta_cka", br),
                "pc_minus_I_cka": mean_contrast("2_PC_minus_I", "delta_cka", br),
                "full_minus_I_cka": mean_contrast("3_FULL_minus_I", "delta_cka", br),
                "featR_minus_pc_cka": mean_contrast("4_FEAT_R_minus_PC", "delta_cka", br),
            }
            for br in ("original", "sae")
        },
    }


def _results_md(summary, primary_rows, contrast_rows, ambient_rows) -> str:
    a = summary["answers"]
    lines = [
        "# Feature vs PCA vs full CCA metric — RESULTS",
        "",
        "Exploratory frozen-metric ablation. Matched-subspace primary panel; ambient diagnostic separate.",
        f"Reproduction gaps: CKA={a['max_archived_cka_gap']:.3e}, mKNN={a['max_archived_mknn_gap']:.3e}.",
        "",
        "## Primary both-sided (matched subspace, panel means)",
        "",
        "| Condition | CKA | mKNN |",
        "|---|---:|---:|",
        f"| I / I | {a['mean_cka_I']:.4f} | {a['mean_mknn_I']:.4f} |",
        f"| PC / PC | {a['mean_cka_PC']:.4f} | {a['mean_mknn_PC']:.4f} |",
        f"| FEATURE_RESTRICTED / FEATURE_RESTRICTED | {a['mean_cka_FEAT_R']:.4f} | {a['mean_mknn_FEAT_R']:.4f} |",
        f"| FULL / FULL | {a['mean_cka_FULL']:.4f} | {a['mean_mknn_FULL']:.4f} |",
        "",
        "## Primary contrasts",
        "",
        f"1. FEAT_R−I: ΔCKA={a['c1_featR_minus_I_cka']:.4f}, ΔmKNN={a['c1_featR_minus_I_mknn']:.4f}",
        f"2. PC−I: ΔCKA={a['c2_pc_minus_I_cka']:.4f}, ΔmKNN={a['c2_pc_minus_I_mknn']:.4f}",
        f"3. FULL−I: ΔCKA={a['c3_full_minus_I_cka']:.4f}, ΔmKNN={a['c3_full_minus_I_mknn']:.4f}",
        f"4. FEAT_R−PC: ΔCKA={a['c4_featR_minus_pc_cka']:.4f}, ΔmKNN={a['c4_featR_minus_pc_mknn']:.4f}",
        f"5. FULL−FEAT_R: ΔCKA={a['c5_full_minus_featR_cka']:.4f}, ΔmKNN={a['c5_full_minus_featR_mknn']:.4f}",
        f"6. FULL−PC: ΔCKA={a['c6_full_minus_pc_cka']:.4f}, ΔmKNN={a['c6_full_minus_pc_mknn']:.4f}",
        "",
        f"Fraction of full CKA gain retained: FEAT_R={a['frac_featR_of_full_cka']:.3f}, PC={a['frac_pc_of_full_cka']:.3f}.",
        "",
        "## Ambient diagnostic (separate)",
        "",
        f"| ambient I/I CKA={a['mean_ambient_I_cka']:.4f} mKNN={a['mean_ambient_I_mknn']:.4f}",
        f"| ambient D/D CKA={a['mean_ambient_D_cka']:.4f} mKNN={a['mean_ambient_D_mknn']:.4f}",
        f"| restricted D/D CKA={a['mean_restricted_D_cka']:.4f} mKNN={a['mean_restricted_D_mknn']:.4f}",
        f"| ambient−restricted ΔCKA={a['ambient_minus_restricted_cka']:.4f}, ΔmKNN={a['ambient_minus_restricted_mknn']:.4f}",
        "",
        "## Direct answers",
        "",
        f"1. **Feature weighting within retained subspace?** FEAT_R−I ΔCKA={a['c1_featR_minus_I_cka']:.4f}, ΔmKNN={a['c1_featR_minus_I_mknn']:.4f}.",
        f"2. **Better or worse than PCA-axis weighting?** FEAT_R−PC ΔCKA={a['c4_featR_minus_pc_cka']:.4f}, ΔmKNN={a['c4_featR_minus_pc_mknn']:.4f}.",
        f"3. **Retention of full gain?** FEAT_R retains {a['frac_featR_of_full_cka']:.1%} of FULL−I CKA; PC retains {a['frac_pc_of_full_cka']:.1%}.",
        "4. **Language vs vision?** See mixed contrasts in `paired_contrasts.csv` (lang_* / vis_*).",
        f"5. **Original vs SAE?** by_branch in summary.json: original featR−I CKA={summary['by_branch']['original']['featR_minus_I_cka']:.4f}, "
        f"sae={summary['by_branch']['sae']['featR_minus_I_cka']:.4f}.",
        "6. **CKA vs mKNN?** Compare contrast signs above; magnitudes differ.",
        f"7. **Ambient discarded directions?** ambient−restricted ΔCKA={a['ambient_minus_restricted_cka']:.4f}, ΔmKNN={a['ambient_minus_restricted_mknn']:.4f}.",
        f"8. **Single-caption?** Mean caption FEAT_R−I ΔCKA≈{a['mean_caption_featR_minus_I_cka']:.4f}.",
        "9. **Unresolved:** diagonals are extracted from a fixed full metric, not separately optimized diagonal metric classes; "
        "cannot claim general superiority of feature- vs PCA-diagonal families.",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/run_feature_vs_pca_metric.py --work /mnt/sdb1/prh-replication-work --repo /mnt/sdb1/prh-replication-work/repo",
        "```",
        "",
    ]
    lines.extend(["## Per-pair primary both-sided", "", "| pair | branch | cond | CKA | mKNN |", "|---|---|---|---:|---:|"])
    for r in primary_rows:
        if r["condition"] not in ("I_I", "PC_PC", "FEAT_R_FEAT_R", "FULL_FULL"):
            continue
        lines.append(
            f"| {r['language']}×{r['vision']} | {r['branch']} | {r['condition']} | {float(r['cka']):.4f} | {float(r['mknn']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
