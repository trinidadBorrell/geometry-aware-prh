"""Matched-subspace and ambient diagnostics for feature vs PCA vs full CCA metrics."""

from __future__ import annotations

import math

import numpy as np

from prh_replication.cca_metric_diagonal_ablation import (
    build_full_metric,
    center_gram,
    check_psd,
    cka_from_grams,
    diagonal_ablation,
    distances_from_gram,
    kernel_spectrum_diagnostics,
    metric_factor_full,
    symmetrize,
    weighted_gram,
)
from prh_replication.crosscov_metric_mknn import mknn_from_features
from prh_replication.crosscov_weighted_cka import feature_spectrum_diagnostics, linear_cka_features
from prh_replication.sae_crosscov import entropy_effective_rank, participation_ratio
from prh_replication.spectral_calibration import mutual_knn, sq_euclidean_rows
from prh_replication.crosscov_metric_mknn import knn_indices_stable

# Matched-subspace metric kinds (act on PCA scores)
KINDS = ("I", "PC", "FEAT_R", "FULL")

KIND_LABELS = {
    "I": "identity",
    "PC": "pca_diagonal",
    "FEAT_R": "feature_restricted",
    "FULL": "full",
}

PRIMARY_BOTH = [
    ("I_I", "I", "I"),
    ("PC_PC", "PC", "PC"),
    ("FEAT_R_FEAT_R", "FEAT_R", "FEAT_R"),
    ("FULL_FULL", "FULL", "FULL"),
]

PRIMARY_CONTRASTS = [
    ("1_FEAT_R_minus_I", "FEAT_R_FEAT_R", "I_I"),
    ("2_PC_minus_I", "PC_PC", "I_I"),
    ("3_FULL_minus_I", "FULL_FULL", "I_I"),
    ("4_FEAT_R_minus_PC", "FEAT_R_FEAT_R", "PC_PC"),
    ("5_FULL_minus_FEAT_R", "FULL_FULL", "FEAT_R_FEAT_R"),
    ("6_FULL_minus_PC", "FULL_FULL", "PC_PC"),
]


def ambient_diag_from_factor(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """diag(P M P.T) via row-wise squared norms of P F, where M = F F.T."""
    pf = np.asarray(p, dtype=np.float64) @ np.asarray(f, dtype=np.float64)
    return np.sum(pf * pf, axis=1)


def feature_restricted_metric(p: np.ndarray, d_feature: np.ndarray) -> np.ndarray:
    """N = P.T D_feature P (generally dense in PCA coordinates)."""
    p = np.asarray(p, dtype=np.float64)
    d = np.asarray(d_feature, dtype=np.float64)
    # (P.T * sqrt(d)) @ (sqrt(d) * P) without forming p×p
    scaled = p * np.sqrt(np.maximum(d, 0.0)).reshape(-1, 1)
    return symmetrize(scaled.T @ scaled)


def psd_factor(metric: np.ndarray, eig_atol_scale: float = 1e-12) -> np.ndarray:
    """Return F with metric ≈ F F.T via eigendecomposition (drop tiny negatives)."""
    m = symmetrize(metric)
    vals, vecs = np.linalg.eigh(m)
    fro = float(np.linalg.norm(m, ord="fro"))
    atol = float(eig_atol_scale) * max(fro, 1.0)
    if np.any(vals < -atol):
        raise ValueError(f"material indefiniteness min_eig={vals[0]}")
    vals = np.maximum(vals, 0.0)
    return vecs * np.sqrt(vals).reshape(1, -1)


def resolve_pca_metric(kind: str, m: np.ndarray, d_pc: np.ndarray, n_feat: np.ndarray) -> np.ndarray:
    if kind == "I":
        return np.eye(m.shape[0], dtype=np.float64)
    if kind == "PC":
        return d_pc
    if kind == "FEAT_R":
        return n_feat
    if kind == "FULL":
        return m
    raise ValueError(kind)


def condition_name(kind_a: str, kind_b: str) -> str:
    return f"{kind_a}_{kind_b}"


def evaluate_pca_pair(
    a: np.ndarray,
    b: np.ndarray,
    kind_a: str,
    kind_b: str,
    m_a: np.ndarray,
    m_b: np.ndarray,
    d_pc_a: np.ndarray,
    d_pc_b: np.ndarray,
    n_a: np.ndarray,
    n_b: np.ndarray,
    knn_k: int = 10,
) -> dict:
    p_a = resolve_pca_metric(kind_a, m_a, d_pc_a, n_a)
    p_b = resolve_pca_metric(kind_b, m_b, d_pc_b, n_b)
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
    knn_l = knn_indices_stable(dist_a, knn_k)
    knn_r = knn_indices_stable(dist_b, knn_k)
    mknn_g = float(mutual_knn(knn_l, knn_r))
    kern_a = kernel_spectrum_diagnostics(k_a)
    kern_b = kernel_spectrum_diagnostics(k_b)
    return {
        "cka": float(cka_g),
        "cka_feature": float(cka_f),
        "mknn": float(pack["mknn"]),
        "mknn_gram": float(mknn_g),
        "cka_gap": abs(float(cka_g) - float(cka_f)) if math.isfinite(cka_g) and math.isfinite(cka_f) else float("nan"),
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
    }


def evaluate_ambient_gram(
    x_c: np.ndarray,
    d_feature: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    """Build sample Gram for ambient diagnostic modes."""
    x = np.asarray(x_c, dtype=np.float64)
    if mode == "I":
        return x @ x.T
    if mode == "D":
        d = np.asarray(d_feature, dtype=np.float64)
        scaled = x * np.sqrt(np.maximum(d, 0.0)).reshape(1, -1)
        return scaled @ scaled.T
    raise ValueError(mode)


def reconstruct_parallel(x_c: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_c, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    scores = x @ p
    parallel = scores @ p.T
    perp = x - parallel
    return parallel, perp


def ambient_block_energies(x_c: np.ndarray, p: np.ndarray, d_feature: np.ndarray) -> dict:
    parallel, perp = reconstruct_parallel(x_c, p)
    d = np.asarray(d_feature, dtype=np.float64)
    sqrt_d = np.sqrt(np.maximum(d, 0.0)).reshape(1, -1)

    def weighted_gram_rows(u, v):
        return (u * sqrt_d) @ (v * sqrt_d).T

    g_pp = weighted_gram_rows(parallel, parallel)
    g_qq = weighted_gram_rows(perp, perp)
    g_pq = weighted_gram_rows(parallel, perp)
    g_qp = weighted_gram_rows(perp, parallel)
    cross = g_pq + g_qp
    return {
        "fro_parallel": float(np.linalg.norm(g_pp, ord="fro")),
        "fro_perp": float(np.linalg.norm(g_qq, ord="fro")),
        "fro_cross": float(np.linalg.norm(cross, ord="fro")),
        "fro_total_ambient": float(np.linalg.norm(weighted_gram_rows(x_c, x_c), ord="fro")),
    }


def score_from_grams(k_a: np.ndarray, k_b: np.ndarray, knn_k: int = 10) -> dict:
    cka = cka_from_grams(k_a, k_b)
    dist_a, meta_a = distances_from_gram(k_a)
    dist_b, meta_b = distances_from_gram(k_b)
    knn_l = knn_indices_stable(dist_a, knn_k)
    knn_r = knn_indices_stable(dist_b, knn_k)
    mknn = float(mutual_knn(knn_l, knn_r))
    kern_a = kernel_spectrum_diagnostics(k_a)
    kern_b = kernel_spectrum_diagnostics(k_b)
    return {
        "cka": float(cka),
        "mknn": mknn,
        "n_material_negative_sqdist": meta_a["n_material_negative_sqdist"] + meta_b["n_material_negative_sqdist"],
        "pr_left": kern_a["participation_ratio"],
        "pr_right": kern_b["participation_ratio"],
        "entropy_left": kern_a["entropy_eff_rank"],
        "entropy_right": kern_b["entropy_eff_rank"],
        "top_energy_left": kern_a["top_mode_energy_fraction"],
        "top_energy_right": kern_b["top_mode_energy_fraction"],
    }


def isotropy_deviation(n_feat: np.ndarray) -> float:
    n = symmetrize(n_feat)
    tr = float(np.trace(n))
    d = n.shape[0]
    if abs(tr) <= 1e-15 * max(1.0, float(np.linalg.norm(n, ord="fro"))):
        return float("nan")
    target = np.eye(d, dtype=np.float64) / d
    return float(np.linalg.norm(n / tr - target, ord="fro"))


def metric_rank(m: np.ndarray, rel: float = 1e-10) -> int:
    vals = np.linalg.eigvalsh(symmetrize(m))
    thresh = rel * max(1.0, float(np.max(np.abs(vals))))
    return int(np.sum(vals > thresh))
