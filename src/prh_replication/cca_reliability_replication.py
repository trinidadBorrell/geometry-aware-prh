"""Fixed CCA reliability metric conditions I / W / D / F for procedure replication."""

from __future__ import annotations

import numpy as np

from prh_replication.cca_metric_diagonal_ablation import (
    build_full_metric,
    check_psd,
    cka_from_grams,
    diagonal_ablation,
    distances_from_gram,
    kernel_spectrum_diagnostics,
    symmetrize,
    weighted_gram,
)
from prh_replication.crosscov_metric_mknn import knn_indices_stable, mknn_from_features
from prh_replication.crosscov_weighted_cka import linear_cka_features, reliability_weights
from prh_replication.feature_vs_pca_metric import psd_factor
from prh_replication.spectral_calibration import mutual_knn

CONDITIONS = ("I", "W", "D", "F")

CONTRASTS = (
    ("F_minus_I", "F", "I"),
    ("F_minus_W", "F", "W"),
    ("F_minus_D", "F", "D"),
    ("D_minus_I", "D", "I"),
)


def beta1_weights(rho: np.ndarray, rho_floor: float, normalize: bool) -> np.ndarray:
    return reliability_weights(rho, 1.0, float(rho_floor), bool(normalize))["w"]


def resolve_metric(kind: str, l: np.ndarray, w: np.ndarray, m_full: np.ndarray, d_pc: np.ndarray) -> np.ndarray:
    if kind == "I":
        return np.eye(m_full.shape[0], dtype=np.float64)
    if kind == "W":
        return symmetrize(np.asarray(l, dtype=np.float64) @ np.asarray(l, dtype=np.float64).T)
    if kind == "D":
        return d_pc
    if kind == "F":
        return m_full
    raise ValueError(kind)


def evaluate_pair(
    a: np.ndarray,
    b: np.ndarray,
    kind_a: str,
    kind_b: str,
    l_a: np.ndarray,
    l_b: np.ndarray,
    w: np.ndarray,
    m_a: np.ndarray,
    m_b: np.ndarray,
    d_a: np.ndarray,
    d_b: np.ndarray,
    knn_k: int = 10,
) -> dict:
    p_a = resolve_metric(kind_a, l_a, w, m_a, d_a)
    p_b = resolve_metric(kind_b, l_b, w, m_b, d_b)
    k_a = weighted_gram(a, p_a)
    k_b = weighted_gram(b, p_b)
    cka_g = cka_from_grams(k_a, k_b)
    f_a = psd_factor(p_a)
    f_b = psd_factor(p_b)
    left = a @ f_a
    right = b @ f_b
    cka_f = linear_cka_features(left, right)
    dist_a, meta_a = distances_from_gram(k_a)
    dist_b, meta_b = distances_from_gram(k_b)
    pack = mknn_from_features(left, right, k=knn_k)
    mknn_g = float(mutual_knn(knn_indices_stable(dist_a, knn_k), knn_indices_stable(dist_b, knn_k)))
    kern_a = kernel_spectrum_diagnostics(k_a)
    kern_b = kernel_spectrum_diagnostics(k_b)
    return {
        "cka": float(cka_g),
        "cka_feature": float(cka_f),
        "mknn": float(pack["mknn"]),
        "mknn_gram": float(mknn_g),
        "cka_gap": abs(float(cka_g) - float(cka_f)),
        "mknn_gap": abs(float(pack["mknn"]) - float(mknn_g)),
        "n_material_negative_sqdist": meta_a["n_material_negative_sqdist"] + meta_b["n_material_negative_sqdist"],
        "n_rows_with_kth_tie_left": pack["n_rows_with_kth_tie_left"],
        "n_rows_with_kth_tie_right": pack["n_rows_with_kth_tie_right"],
        "n_zero_distance_offdiag_counts": pack["n_zero_distance_offdiag_counts"],
        "pr_left": kern_a["participation_ratio"],
        "pr_right": kern_b["participation_ratio"],
        "entropy_left": kern_a["entropy_eff_rank"],
        "entropy_right": kern_b["entropy_eff_rank"],
        "top_energy_left": kern_a["top_mode_energy_fraction"],
        "top_energy_right": kern_b["top_mode_energy_fraction"],
        "rank_metric_a": int(np.sum(np.linalg.eigvalsh(0.5 * (p_a + p_a.T)) > 1e-10)),
        "rank_metric_b": int(np.sum(np.linalg.eigvalsh(0.5 * (p_b + p_b.T)) > 1e-10)),
    }


def build_metrics(l_a: np.ndarray, l_b: np.ndarray, rho: np.ndarray, rho_floor: float, normalize: bool) -> dict:
    w = beta1_weights(rho, rho_floor, normalize)
    m_a = build_full_metric(l_a, w)
    m_b = build_full_metric(l_b, w)
    d_a = diagonal_ablation(m_a)
    d_b = diagonal_ablation(m_b)
    for name, mat in (("M_A", m_a), ("M_B", m_b), ("D_A", d_a), ("D_B", d_b)):
        psd = check_psd(mat)
        if not psd["psd_ok"]:
            raise RuntimeError(f"PSD failure {name}: {psd}")
    return {"w": w, "M_A": m_a, "M_B": m_b, "D_A": d_a, "D_B": d_b}
