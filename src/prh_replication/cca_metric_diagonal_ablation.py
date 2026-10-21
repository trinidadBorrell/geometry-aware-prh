"""PCA-axis diagonal ablation of full-rank CCA reliability metrics."""

from __future__ import annotations

import math

import numpy as np

from prh_replication.crosscov_metric_mknn import knn_indices_stable, mknn_from_features
from prh_replication.crosscov_weighted_cka import (
    feature_spectrum_diagnostics,
    linear_cka_features,
    reliability_weights,
)
from prh_replication.sae_crosscov import entropy_effective_rank, participation_ratio


CONDITIONS: list[tuple[str, str, str]] = [
    ("identity_identity", "I", "I"),
    ("diagonal_identity", "D", "I"),
    ("full_identity", "M", "I"),
    ("identity_diagonal", "I", "D"),
    ("identity_full", "I", "M"),
    ("diagonal_diagonal", "D", "D"),
    ("full_diagonal", "M", "D"),
    ("diagonal_full", "D", "M"),
    ("full_full", "M", "M"),
]

CONTRASTS: list[tuple[str, str, str]] = [
    ("A_full_minus_diagonal_both", "full_full", "diagonal_diagonal"),
    ("B_full_minus_diagonal_language", "full_identity", "diagonal_identity"),
    ("C_full_minus_diagonal_vision", "identity_full", "identity_diagonal"),
    ("D_language_offdiag_with_full_vision", "full_full", "diagonal_full"),
    ("E_vision_offdiag_with_full_language", "full_full", "full_diagonal"),
    ("F_diagonal_minus_identity", "diagonal_diagonal", "identity_identity"),
    ("G_full_minus_identity", "full_full", "identity_identity"),
]


def symmetrize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return 0.5 * (m + m.T)


def build_full_metric(l: np.ndarray, w: np.ndarray) -> np.ndarray:
    l = np.asarray(l, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return symmetrize(l @ np.diag(w) @ l.T)


def diagonal_ablation(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.diag(np.diag(m))


def metric_factor_full(l: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(l, dtype=np.float64) * np.sqrt(np.asarray(w, dtype=np.float64)).reshape(1, -1)


def metric_factor_diagonal(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    diag = np.diag(d)
    if np.any(diag < -1e-10):
        raise ValueError("diagonal metric has material negative entries")
    return np.diag(np.sqrt(np.maximum(diag, 0.0)))


def resolve_metric(kind: str, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    if kind == "I":
        return np.eye(m.shape[0], dtype=np.float64)
    if kind == "D":
        return d
    if kind == "M":
        return m
    raise ValueError(kind)


def resolve_factor(kind: str, l: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
    if kind == "I":
        return np.eye(l.shape[0], dtype=np.float64)
    if kind == "D":
        return metric_factor_diagonal(d)
    if kind == "M":
        return metric_factor_full(l, w)
    raise ValueError(kind)


def check_psd(m: np.ndarray, atol_scale: float = 1e-8) -> dict:
    m = symmetrize(m)
    eigs = np.linalg.eigvalsh(m)
    fro = float(np.linalg.norm(m, ord="fro"))
    atol = float(atol_scale) * max(fro, 1.0)
    neg = eigs[eigs < -atol]
    return {
        "min_eig": float(eigs[0]),
        "max_eig": float(eigs[-1]),
        "n_material_negative": int(neg.size),
        "atol": atol,
        "psd_ok": bool(neg.size == 0),
    }


def weighted_gram(scores: np.ndarray, metric: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    p = np.asarray(metric, dtype=np.float64)
    return x @ p @ x.T


def center_gram(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    row = k.mean(axis=1, keepdims=True)
    col = k.mean(axis=0, keepdims=True)
    return k - row - col + k.mean()


def cka_from_grams(k_a: np.ndarray, k_b: np.ndarray) -> float:
    kc = center_gram(k_a)
    lc = center_gram(k_b)
    num = float(np.sum(kc * lc))
    den_a = float(np.linalg.norm(kc))
    den_b = float(np.linalg.norm(lc))
    if den_a <= 0.0 or den_b <= 0.0 or not math.isfinite(den_a * den_b):
        return float("nan")
    return num / (den_a * den_b)


def distances_from_gram(k: np.ndarray, clip_neg_atol: float = 1e-10) -> tuple[np.ndarray, dict]:
    k = np.asarray(k, dtype=np.float64)
    diag = np.diag(k)
    dist = diag[:, None] + diag[None, :] - 2.0 * k
    material = dist < -float(clip_neg_atol) * max(1.0, float(np.max(np.abs(k))))
    n_material = int(np.sum(material))
    tiny = (dist < 0.0) & (~material)
    dist = dist.copy()
    dist[tiny] = 0.0
    return dist, {"n_material_negative_sqdist": n_material, "n_clipped_tiny_negative": int(np.sum(tiny))}


def kernel_spectrum_diagnostics(k: np.ndarray) -> dict[str, float]:
    kc = center_gram(k)
    # eigenvalues of centred Gram (sample space)
    eigs = np.linalg.eigvalsh(0.5 * (kc + kc.T))
    eigs = np.sort(eigs[eigs > 1e-12 * max(1.0, float(np.max(np.abs(eigs))))])[::-1]
    total = float(np.sum(eigs))
    top = float(eigs[0] / total) if total > 0 and eigs.size else float("nan")
    return {
        "participation_ratio": participation_ratio(eigs) if eigs.size else float("nan"),
        "entropy_eff_rank": entropy_effective_rank(eigs) if eigs.size else float("nan"),
        "top_mode_energy_fraction": top,
        "n_positive_eigs": int(eigs.size),
        "trace": float(np.trace(kc)),
    }


def metric_spectrum_diagnostics(m: np.ndarray) -> dict:
    m = symmetrize(m)
    eigs = np.linalg.eigvalsh(m)
    eigs_desc = eigs[::-1]
    pos = eigs_desc[eigs_desc > 1e-12 * max(1.0, float(np.max(np.abs(eigs_desc))))]
    d = diagonal_ablation(m)
    fro_m = float(np.linalg.norm(m, ord="fro"))
    fro_d = float(np.linalg.norm(d, ord="fro"))
    off = m - d
    off_frac = float(np.sum(off * off) / max(fro_m * fro_m, 1e-30))
    diag = np.diag(m)
    return {
        "trace_m": float(np.trace(m)),
        "trace_d": float(np.trace(d)),
        "fro_m": fro_m,
        "fro_d": fro_d,
        "offdiag_energy_fraction": off_frac,
        "eig_m": eigs_desc,
        "eig_d": np.sort(diag)[::-1],
        "pr_m": participation_ratio(pos) if pos.size else float("nan"),
        "pr_d": participation_ratio(np.maximum(diag, 0.0)) if diag.size else float("nan"),
        "entropy_m": entropy_effective_rank(pos) if pos.size else float("nan"),
        "entropy_d": entropy_effective_rank(np.maximum(diag, 0.0)) if diag.size else float("nan"),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
    }


def transform_scores(scores: np.ndarray, factor: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) @ np.asarray(factor, dtype=np.float64)


def evaluate_condition(
    a: np.ndarray,
    b: np.ndarray,
    kind_a: str,
    kind_b: str,
    m_a: np.ndarray,
    m_b: np.ndarray,
    d_a: np.ndarray,
    d_b: np.ndarray,
    l_a: np.ndarray,
    l_b: np.ndarray,
    w: np.ndarray,
    knn_k: int = 10,
) -> dict:
    p_a = resolve_metric(kind_a, m_a, d_a)
    p_b = resolve_metric(kind_b, m_b, d_b)
    f_a = resolve_factor(kind_a, l_a, w, d_a)
    f_b = resolve_factor(kind_b, l_b, w, d_b)

    k_a = weighted_gram(a, p_a)
    k_b = weighted_gram(b, p_b)
    cka_gram = cka_from_grams(k_a, k_b)

    left = transform_scores(a, f_a)
    right = transform_scores(b, f_b)
    cka_feat = linear_cka_features(left, right)

    dist_a, dist_meta_a = distances_from_gram(k_a)
    dist_b, dist_meta_b = distances_from_gram(k_b)
    # centred Gram distances must match uncentred
    dist_ac, _ = distances_from_gram(center_gram(k_a))
    dist_bc, _ = distances_from_gram(center_gram(k_b))

    knn_pack = mknn_from_features(left, right, k=knn_k)
    # also from gram distances with same stable knn
    knn_l = knn_indices_stable(dist_a, knn_k)
    knn_r = knn_indices_stable(dist_b, knn_k)
    from prh_replication.spectral_calibration import mutual_knn

    mknn_gram = float(mutual_knn(knn_l, knn_r))

    kern_a = kernel_spectrum_diagnostics(k_a)
    kern_b = kernel_spectrum_diagnostics(k_b)
    feat_a = feature_spectrum_diagnostics(left)
    feat_b = feature_spectrum_diagnostics(right)

    return {
        "cka": float(cka_gram),
        "cka_feature": float(cka_feat),
        "mknn": float(knn_pack["mknn"]),
        "mknn_gram": float(mknn_gram),
        "n_rows_with_kth_tie_left": knn_pack["n_rows_with_kth_tie_left"],
        "n_rows_with_kth_tie_right": knn_pack["n_rows_with_kth_tie_right"],
        "n_zero_distance_offdiag_counts": knn_pack["n_zero_distance_offdiag_counts"],
        "n_material_negative_sqdist": dist_meta_a["n_material_negative_sqdist"] + dist_meta_b["n_material_negative_sqdist"],
        "centering_preserves_dist_a": float(np.max(np.abs(dist_a - dist_ac))),
        "centering_preserves_dist_b": float(np.max(np.abs(dist_b - dist_bc))),
        "pr_left": kern_a["participation_ratio"],
        "pr_right": kern_b["participation_ratio"],
        "entropy_left": kern_a["entropy_eff_rank"],
        "entropy_right": kern_b["entropy_eff_rank"],
        "top_energy_left": kern_a["top_mode_energy_fraction"],
        "top_energy_right": kern_b["top_mode_energy_fraction"],
        "feat_pr_left": feat_a["participation_ratio"],
        "feat_pr_right": feat_b["participation_ratio"],
        "left": left,
        "right": right,
        "k_a": k_a,
        "k_b": k_b,
        "dist_a": dist_a,
        "dist_b": dist_b,
    }


def beta1_weights(rho: np.ndarray, rho_floor: float, normalize: bool) -> np.ndarray:
    return reliability_weights(rho, 1.0, float(rho_floor), bool(normalize))["w"]
