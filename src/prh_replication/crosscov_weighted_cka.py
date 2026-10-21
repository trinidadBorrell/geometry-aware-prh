"""Spectral reweighting of cross-covariance / CCA directions for linear CKA."""

from __future__ import annotations

import math

import numpy as np

from prh_replication.sae_crosscov import (
    entropy_effective_rank,
    participation_ratio,
    pearson_cols,
    raw_crosscov_svd,
    regularized_cca,
    sample_covariances,
)
from prh_replication.spectral_calibration import feature_scores_rows, gram_scores, mnn_rows


def centre_columns(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    return values - values.mean(axis=0, keepdims=True)


def linear_cka_features(left: np.ndarray, right: np.ndarray) -> float:
    """Centred linear CKA: ||A_cᵀB_c||_F² / (||A_cᵀA_c||_F ||B_cᵀB_c||_F)."""
    a = centre_columns(left)
    b = centre_columns(right)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return float("nan")
    cross = a.T @ b
    gram_a = a.T @ a
    gram_b = b.T @ b
    num = float(np.sum(cross * cross))
    den_a = float(np.linalg.norm(gram_a, ord="fro"))
    den_b = float(np.linalg.norm(gram_b, ord="fro"))
    if den_a <= 0.0 or den_b <= 0.0 or not math.isfinite(den_a * den_b):
        return float("nan")
    return num / (den_a * den_b)


def linear_cka_agrees_with_gram(left: np.ndarray, right: np.ndarray, atol: float = 1e-10) -> bool:
    a = centre_columns(left)
    b = centre_columns(right)
    feature = linear_cka_features(a, b)
    gram = float(gram_scores(a @ a.T, b @ b.T)["a"])
    return bool(np.isfinite(feature) and np.isfinite(gram) and abs(feature - gram) <= atol)


def feature_spectrum_diagnostics(features: np.ndarray) -> dict[str, float]:
    """Eigenvalue diagnostics of centred feature Gram (nonzero spectrum = sample Gram)."""
    centred = centre_columns(features)
    # eigenvalues of (1/(n-1)) XᵀX — same PR/entropy as sample Gram up to scale
    cov = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    values = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    values = np.sort(values[values > 0.0])[::-1]
    total = float(np.sum(values))
    top = float(values[0] / total) if total > 0 and values.size else float("nan")
    return {
        "participation_ratio": participation_ratio(values),
        "entropy_eff_rank": entropy_effective_rank(values),
        "top_mode_energy_fraction": top,
        "n_positive_eigs": int(values.size),
    }


def score_pair(left: np.ndarray, right: np.ndarray, knn_k: int = 10) -> dict[str, float]:
    cka = linear_cka_features(left, right)
    # archived pipeline CKA with eps (descriptive only)
    try:
        archived = float(feature_scores_rows(left, right)["a"])
    except Exception:
        archived = float("nan")
    diag_l = feature_spectrum_diagnostics(left)
    diag_r = feature_spectrum_diagnostics(right)
    return {
        "cka": cka,
        "cka_gram_form": archived,
        "mnn_k10": float(mnn_rows(left, right, k=int(knn_k))),
        "pr_left": diag_l["participation_ratio"],
        "pr_right": diag_r["participation_ratio"],
        "entropy_left": diag_l["entropy_eff_rank"],
        "entropy_right": diag_r["entropy_eff_rank"],
        "top_energy_left": diag_l["top_mode_energy_fraction"],
        "top_energy_right": diag_r["top_mode_energy_fraction"],
    }


def raw_power_weights(sigma: np.ndarray, alpha: float, floor: float, normalize: bool) -> dict:
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0 or not np.isfinite(sigma[0]) or float(sigma[0]) <= 0.0:
        return {"w": np.ones(0), "q": np.ones(0), "degenerate": True, "n_floored": 0}
    q = np.maximum(sigma / float(sigma[0]), float(floor))
    w = np.power(q, float(alpha) - 1.0)
    n_floored = int(np.sum((sigma / float(sigma[0])) < float(floor)))
    if normalize and w.size and float(np.mean(w)) > 0.0:
        w = w / float(np.mean(w))
    return {"w": w, "q": q, "degenerate": False, "n_floored": n_floored}


def reliability_weights(rho: np.ndarray, beta: float, floor: float, normalize: bool, clamp_tol: float = 1e-8) -> dict:
    rho = np.asarray(rho, dtype=np.float64)
    n_neg = int(np.sum(rho < -float(clamp_tol)))
    # Clamp only tiny numerical excursions outside [0,1]; material negatives remain a failure flag.
    clipped = np.clip(rho, 0.0, 1.0)
    n_clamped = int(np.sum(np.abs(clipped - rho) > 0.0) - n_neg)  # ignore negatives already counted
    base = np.maximum(clipped * clipped, float(floor))
    w = np.power(base, float(beta))
    if normalize and w.size and float(np.mean(w)) > 0.0:
        w = w / float(np.mean(w))
    return {
        "w": w,
        "rho": rho,
        "n_neg_train_rho": n_neg,
        "n_clamped_to_unit_interval": max(n_clamped, 0),
        "n_floored": int(np.sum((clipped * clipped) < float(floor))),
        "algebraic_failure": n_neg > 0,
    }


def apply_diag_scale(features: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) * np.asarray(scale, dtype=np.float64).reshape(1, -1)


def transform_raw(
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_w = np.sqrt(np.asarray(w, dtype=np.float64))
    a_rot = np.asarray(a, dtype=np.float64) @ np.asarray(u, dtype=np.float64)
    b_rot = np.asarray(b, dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    if side == "language":
        return apply_diag_scale(a_rot, sqrt_w), np.asarray(b, dtype=np.float64)
    if side == "vision":
        return np.asarray(a, dtype=np.float64), apply_diag_scale(b_rot, sqrt_w)
    if side == "both":
        return apply_diag_scale(a_rot, sqrt_w), apply_diag_scale(b_rot, sqrt_w)
    raise ValueError(side)


def transform_cca(
    a: np.ndarray,
    b: np.ndarray,
    l_a: np.ndarray,
    l_b: np.ndarray,
    w: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_w = np.sqrt(np.asarray(w, dtype=np.float64))
    t_a = np.asarray(a, dtype=np.float64) @ np.asarray(l_a, dtype=np.float64)
    t_b = np.asarray(b, dtype=np.float64) @ np.asarray(l_b, dtype=np.float64)
    if side == "language":
        return apply_diag_scale(t_a, sqrt_w), np.asarray(b, dtype=np.float64)
    if side == "vision":
        return np.asarray(a, dtype=np.float64), apply_diag_scale(t_b, sqrt_w)
    if side == "both":
        return apply_diag_scale(t_a, sqrt_w), apply_diag_scale(t_b, sqrt_w)
    raise ValueError(side)


def fit_raw_and_cca(a_train: np.ndarray, b_train: np.ndarray, eta: float, support_tol: float) -> dict:
    cov = sample_covariances(a_train, b_train)
    svd = raw_crosscov_svd(cov["C_AB"])
    cca = regularized_cca(cov["C_A"], cov["C_B"], cov["C_AB"], eta=float(eta), support_tol=float(support_tol))
    t_a = cov["A"] @ cca["L_A"]
    t_b = cov["B"] @ cca["L_B"]
    rho = pearson_cols(t_a, t_b)
    return {"cov": cov, "svd": svd, "cca": cca, "rho": rho, "t_a": t_a, "t_b": t_b}


def select_alpha(curve: list[dict], tie_atol: float) -> float:
    """curve items: {alpha, cka}. Prefer max CKA; ties → closest to 1; then smaller alpha."""
    if not curve:
        raise ValueError("empty alpha curve")
    best = max(float(row["cka"]) for row in curve if np.isfinite(row["cka"]))
    tied = [row for row in curve if np.isfinite(row["cka"]) and abs(float(row["cka"]) - best) <= float(tie_atol)]
    tied.sort(key=lambda row: (abs(float(row["alpha"]) - 1.0), float(row["alpha"])))
    return float(tied[0]["alpha"])


def select_beta(curve: list[dict], tie_atol: float) -> float:
    if not curve:
        raise ValueError("empty beta curve")
    best = max(float(row["cka"]) for row in curve if np.isfinite(row["cka"]))
    tied = [row for row in curve if np.isfinite(row["cka"]) and abs(float(row["cka"]) - best) <= float(tie_atol)]
    tied.sort(key=lambda row: float(row["beta"]))
    return float(tied[0]["beta"])


def weight_summary(w: np.ndarray) -> dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return {"w_min": float("nan"), "w_max": float("nan"), "w_dynamic_range": float("nan")}
    wmin = float(np.min(w))
    wmax = float(np.max(w))
    return {
        "w_min": wmin,
        "w_max": wmax,
        "w_dynamic_range": (wmax / wmin) if wmin > 0 else float("inf"),
    }


def two_sided_cross_spectrum(a: np.ndarray, b: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Diagonal of sample cross-cov of both-sided weighted SVD coords (divisor G-1)."""
    left, right = transform_raw(a, b, u, v, w, "both")
    left = centre_columns(left)
    right = centre_columns(right)
    return np.sum(left * right, axis=0) / (left.shape[0] - 1)
