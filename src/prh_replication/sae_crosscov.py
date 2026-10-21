"""Cross-covariance SVD and regularized CCA on matched PCA scores."""

from __future__ import annotations

import math

import numpy as np


def _column_canonical_signs(matrix: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-magnitude entry is nonnegative (single matrix)."""
    out = np.array(matrix, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0.0:
            out[:, j] *= -1.0
    return out


def fit_pca(features: np.ndarray, q: int, rank_tol: float = 1e-8) -> dict:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA needs a matrix with at least two rows")
    mean = values.mean(axis=0)
    centred = values - mean
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    leading = float(singular[0]) if singular.size else 0.0
    threshold = float(rank_tol) * leading if leading > 0.0 else float(rank_tol)
    supported = int(np.sum(singular > threshold))
    width = int(min(int(q), supported, values.shape[0] - 1, values.shape[1]))
    if width < 1:
        raise ValueError("PCA retained no mode")
    components = _column_canonical_signs(vt[:width].T.copy())
    total = float(np.sum(singular * singular))
    covered = float(np.sum(singular[:width] ** 2) / total) if total > 0.0 else float("nan")
    return {
        "mean": mean,
        "components": components,
        "singular_values": np.asarray(singular, dtype=np.float64),
        "width": width,
        "numerical_rank": supported,
        "variance_fraction": covered,
        "explained_ratios": (singular[:width] ** 2 / total) if total > 0.0 else np.full(width, np.nan),
    }


def project(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64).reshape(1, -1)) @ np.asarray(
        components, dtype=np.float64
    )


def sample_covariances(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        raise ValueError("paired covariances need matching n>=2")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = a.shape[0] - 1
    return {
        "C_A": (a.T @ a) / denom,
        "C_B": (b.T @ b) / denom,
        "C_AB": (a.T @ b) / denom,
        "A": a,
        "B": b,
    }


def scale_normalized_energy(c_ab: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> float:
    num = float(np.sum(np.asarray(c_ab, dtype=np.float64) ** 2))
    den = float(np.trace(c_a) * np.trace(c_b))
    if den <= 0.0 or not math.isfinite(den):
        return float("nan")
    return num / den


def participation_ratio(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return float("nan")
    total = float(np.sum(values))
    if total <= 0.0:
        return float("nan")
    share = values / total
    return float(1.0 / np.sum(share * share))


def entropy_effective_rank(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return float("nan")
    total = float(np.sum(values))
    share = values / total
    return float(np.exp(-np.sum(share * np.log(share))))


def _joint_canonical_signs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip paired factor columns together so left's largest-magnitude entry is nonnegative.

    Independent per-matrix sign canonicalization would break the SVD/CCA pairing:
    ``C = U diag(σ) Vᵀ`` requires the same flip on both sides of each mode.
    """
    u = np.array(left, dtype=np.float64, copy=True)
    v = np.array(right, dtype=np.float64, copy=True)
    if u.shape[1] != v.shape[1]:
        raise ValueError("paired factors need matching column counts")
    for j in range(u.shape[1]):
        col = u[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0.0:
            u[:, j] *= -1.0
            v[:, j] *= -1.0
    return u, v


def raw_crosscov_svd(c_ab: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(c_ab, dtype=np.float64)
    u, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
    u, v = _joint_canonical_signs(u, vt.T)
    return {"U": u, "V": v, "sigma": np.asarray(sigma, dtype=np.float64)}


def _sym_inv_sqrt(matrix: np.ndarray, ridge: float, support_tol: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T) + ridge * np.eye(matrix.shape[0]))
    keep = values > float(support_tol) * max(float(values[-1]), float(support_tol))
    safe = np.where(keep, values, 1.0)
    inv = vectors * (1.0 / np.sqrt(safe))
    factor = inv @ vectors.T
    return factor, keep


def regularized_cca(c_a: np.ndarray, c_b: np.ndarray, c_ab: np.ndarray, eta: float, support_tol: float = 1e-8) -> dict:
    c_a = np.asarray(c_a, dtype=np.float64)
    c_b = np.asarray(c_b, dtype=np.float64)
    c_ab = np.asarray(c_ab, dtype=np.float64)
    d = c_a.shape[0]
    tau_a = float(eta) * float(np.trace(c_a)) / d
    tau_b = float(eta) * float(np.trace(c_b)) / d
    ra, keep_a = _sym_inv_sqrt(c_a, tau_a, support_tol)
    rb, keep_b = _sym_inv_sqrt(c_b, tau_b, support_tol)
    # ra, rb are symmetric; rb.T == rb. Spec: R = W_A C_AB W_B.
    whitened = ra @ c_ab @ rb
    p, s, qt = np.linalg.svd(whitened, full_matrices=False)
    p, q = _joint_canonical_signs(p, qt.T)
    return {
        "L_A": ra @ p,
        "L_B": rb @ q,
        "s": np.asarray(s, dtype=np.float64),
        "tau_a": tau_a,
        "tau_b": tau_b,
        "support_a": keep_a,
        "support_b": keep_b,
        "P": p,
        "Q": q,
        "W_A": ra,
        "W_B": rb,
    }


def pearson_cols(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = np.sum(a * b, axis=0)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    ok = den > 0.0
    out[ok] = num[ok] / den[ok]
    return out


def covariance_cols(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return np.sum(a * b, axis=0) / (a.shape[0] - 1)


def fit_scalar_predictors(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_c = x - x_mean
    y_c = y - y_mean
    var = np.sum(x_c * x_c, axis=0)
    cov = np.sum(x_c * y_c, axis=0)
    slope = np.zeros_like(var)
    ok = var > 0.0
    slope[ok] = cov[ok] / var[ok]
    intercept = y_mean - slope * x_mean
    return {"slope": slope, "intercept": intercept, "x_mean": x_mean, "y_mean": y_mean}


def predict_scalar(source: np.ndarray, predictors: dict[str, np.ndarray]) -> np.ndarray:
    return predictors["intercept"] + predictors["slope"] * np.asarray(source, dtype=np.float64)


def predictive_scores(pred: np.ndarray, target: np.ndarray, train_mean: np.ndarray) -> dict[str, np.ndarray]:
    """Score frozen train-fitted predictions on held-out targets.

    ``r2`` is conventional test R² vs the *test* target mean.
    ``improvement_vs_train_mean`` uses the train-fitted mean predictor denominator.
    These are not interchangeable when train and test means differ.
    """
    y = np.asarray(target, dtype=np.float64)
    yhat = np.asarray(pred, dtype=np.float64)
    residual = y - yhat
    mse = np.mean(residual * residual, axis=0)
    sse = np.sum(residual * residual, axis=0)
    y_c = y - y.mean(axis=0, keepdims=True)
    sst_test = np.sum(y_c * y_c, axis=0)
    r2 = np.full_like(mse, np.nan)
    ok = sst_test > 0.0
    r2[ok] = 1.0 - sse[ok] / sst_test[ok]
    baseline = y - np.asarray(train_mean, dtype=np.float64).reshape(1, -1)
    mean_mse = np.mean(baseline * baseline, axis=0)
    improvement = np.full_like(mse, np.nan)
    ok_m = mean_mse > 0.0
    improvement[ok_m] = 1.0 - mse[ok_m] / mean_mse[ok_m]
    return {"mse": mse, "r2": r2, "improvement_vs_train_mean": improvement}


def cross_correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = a.T @ b
    den = np.sqrt(np.sum(a * a, axis=0)[:, None] * np.sum(b * b, axis=0)[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = num / den
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def prefix_means(values: np.ndarray, prefixes: list[int]) -> dict[str, float]:
    out = {}
    for k in prefixes:
        if k > values.shape[0]:
            continue
        chunk = values[:k]
        finite = chunk[np.isfinite(chunk)]
        out[str(k)] = float(np.mean(finite)) if finite.size else float("nan")
    return out
