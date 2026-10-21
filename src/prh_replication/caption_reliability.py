"""Caption-group reliability weights in a fixed language PCA basis.

Image ids group repeated captions. No vision embedding enters these calculations.
"""

from __future__ import annotations

import math

import numpy as np

METHODS = (
    "ORIGINAL",
    "PCA_IDENTITY",
    "RELIABILITY_SQRT",
    "RELIABILITY_STRONG",
    "RELIABILITY_WHITENED",
    "WHITENED_CONTROL",
)
PRIMARY = "RELIABILITY_SQRT"


def sign_components(components: np.ndarray) -> np.ndarray:
    """Match ``metric_stability.canonical_signs``: largest-magnitude entry is nonnegative."""
    signed = np.array(np.asarray(components, dtype=np.float64), copy=True)
    for index in range(signed.shape[1]):
        column = signed[:, index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            signed[:, index] *= -1.0
    return signed


def variance_components(scores: np.ndarray, denominator_atol: float = 0.0) -> dict[str, np.ndarray]:
    """Unbiased within-image variance and nonnegative between-image signal.

    ``scores`` has shape ``(images, captions, modes)``.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("scores must have shape (images, captions, modes)")
    n_images, n_captions, n_modes = values.shape
    if n_images < 2 or n_captions < 2 or n_modes < 1:
        raise ValueError("variance components need at least two images, two captions and one mode")
    image_mean = values.mean(axis=1)
    within = values - image_mean[:, None, :]
    within_variance = np.sum(within * within, axis=(0, 1)) / (n_images * (n_captions - 1))
    grand = image_mean.mean(axis=0)
    mean_variance = np.sum((image_mean - grand) ** 2, axis=0) / (n_images - 1)
    unclipped = mean_variance - within_variance / n_captions
    signal = np.maximum(unclipped, 0.0)
    denominator = signal + within_variance
    reliability = np.zeros(n_modes, dtype=np.float64)
    usable = np.isfinite(denominator) & (denominator > float(denominator_atol))
    reliability[usable] = signal[usable] / denominator[usable]
    flat = values.reshape(n_images * n_captions, n_modes)
    caption_mean = flat.mean(axis=0)
    total = np.sum((flat - caption_mean) ** 2, axis=0)
    sample_variance = total / (n_images * n_captions - 1)
    left = (n_images * n_captions - 1) * sample_variance
    right = n_images * (n_captions - 1) * within_variance + n_captions * (n_images - 1) * mean_variance
    return {
        "W": within_variance,
        "Vmean": mean_variance,
        "B_unclipped": unclipped,
        "B": signal,
        "w": reliability,
        "lambda": sample_variance,
        "negative_signal": unclipped < 0.0,
        "excluded_mode": ~usable,
        "decomposition_abs": np.abs(left - right),
    }


def stabilizer(sample_variance: np.ndarray, coef: float = 1e-3) -> float:
    values = np.asarray(sample_variance, dtype=np.float64)
    if values.size == 0 or not np.any(np.isfinite(values)):
        raise ValueError("cannot form tau from an empty spectrum")
    return float(coef) * float(np.mean(values))


def feature_scales(method: str, reliability: np.ndarray, sample_variance: np.ndarray, tau: float) -> np.ndarray:
    weight = np.asarray(reliability, dtype=np.float64)
    variance = np.asarray(sample_variance, dtype=np.float64)
    if method == "PCA_IDENTITY":
        return np.ones_like(weight)
    if method == "RELIABILITY_SQRT":
        return np.sqrt(weight)
    if method == "RELIABILITY_STRONG":
        return weight.copy()
    if method == "RELIABILITY_WHITENED":
        return np.sqrt(weight / (variance + tau))
    if method == "WHITENED_CONTROL":
        return 1.0 / np.sqrt(variance + tau)
    raise ValueError(f"unknown method {method}")


def apply_scales(scores: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    factor = np.asarray(scale, dtype=np.float64).reshape(1, -1)
    if values.shape[-1] != factor.shape[1]:
        raise ValueError("scale width does not match scores")
    return values * factor


def fit_pca(prepared: np.ndarray, rank_tol: float = 1e-8) -> dict[str, np.ndarray | int | float]:
    """Economy SVD. Retained width is every numerically supported mode, at most ``n - 1``."""
    values = np.asarray(prepared, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA expects a matrix with at least two rows")
    mean = values.mean(axis=0)
    centred = values - mean
    _left, singular, vt = np.linalg.svd(centred, full_matrices=False)
    leading = float(singular[0]) if singular.size else 0.0
    threshold = float(rank_tol) * leading if leading > 0.0 else float(rank_tol)
    supported = int(np.sum(singular > threshold))
    width = int(min(supported, values.shape[0] - 1, values.shape[1]))
    if width < 1:
        raise ValueError("PCA retained no mode")
    components = sign_components(vt[:width].T.copy())
    return {
        "mean": mean,
        "components": components,
        "singular_values": np.asarray(singular, dtype=np.float64),
        "width": width,
        "numerical_rank": supported,
        "n": int(values.shape[0]),
        "d": int(values.shape[1]),
        "threshold": threshold,
    }


def project(prepared: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    values = np.asarray(prepared, dtype=np.float64)
    return (values - np.asarray(mean, dtype=np.float64).reshape(1, -1)) @ np.asarray(components, dtype=np.float64)


def regroup(scores: np.ndarray, seed: int) -> np.ndarray:
    """Permute captions and pack them back into groups of the original size."""
    values = np.asarray(scores, dtype=np.float64)
    n_images, n_captions, n_modes = values.shape
    flat = values.reshape(n_images * n_captions, n_modes)
    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(flat.shape[0])
    return flat[order].reshape(n_images, n_captions, n_modes)


def weighted_dot(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(np.asarray(weight, dtype=np.float64) * np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)))


def centred_dot_gap(prepared: np.ndarray, mean: np.ndarray, components: np.ndarray) -> float:
    centred = np.asarray(prepared, dtype=np.float64) - np.asarray(mean, dtype=np.float64).reshape(1, -1)
    scores = centred @ np.asarray(components, dtype=np.float64)
    return float(np.max(np.abs(centred @ centred.T - scores @ scores.T)))


def loglog_fit(signal: np.ndarray, sample_variance: np.ndarray) -> dict[str, float]:
    """Descriptive slope of log B on log lambda over modes with both positive. Not a power-law claim."""
    positive = (np.asarray(signal) > 0.0) & (np.asarray(sample_variance) > 0.0) & np.isfinite(signal) & np.isfinite(sample_variance)
    count = int(np.sum(positive))
    empty = {
        "n": count,
        "slope": float("nan"),
        "intercept": float("nan"),
        "r2": float("nan"),
        "lambda_min": float("nan"),
        "lambda_max": float("nan"),
    }
    if count < 3:
        return empty
    x = np.log(np.asarray(sample_variance, dtype=np.float64)[positive])
    y = np.log(np.asarray(signal, dtype=np.float64)[positive])
    x_centred = x - x.mean()
    y_centred = y - y.mean()
    denom = float(np.dot(x_centred, x_centred))
    if denom <= 0.0:
        return empty
    slope = float(np.dot(x_centred, y_centred) / denom)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    total = float(np.sum(y_centred ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / total) if total > 0.0 else float("nan")
    kept = np.asarray(sample_variance, dtype=np.float64)[positive]
    return {
        "n": count,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "lambda_min": float(kept.min()),
        "lambda_max": float(kept.max()),
    }


def effective_rank(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return float("nan")
    centred = values - values.mean(axis=0, keepdims=True)
    _left, singular, _right = np.linalg.svd(centred, full_matrices=False)
    energy = singular ** 2
    total = float(np.sum(energy))
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    share = energy / total
    share = share[share > 0.0]
    return float(np.exp(-np.sum(share * np.log(share))))


def gram_offdiag_fraction(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    gram = values @ values.T
    centred = gram - gram.mean(axis=1, keepdims=True) - gram.mean(axis=0, keepdims=True) + gram.mean()
    total = float(np.sum(centred * centred))
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    diagonal = np.sum(np.diag(centred) ** 2)
    return float((total - diagonal) / total)


def feature_norm_stats(features: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(features, dtype=np.float64), axis=1)
    return {"mean": float(np.mean(norms)), "std": float(np.std(norms)), "max": float(np.max(norms))}


def features_vanished(features: np.ndarray, atol: float = 1e-12) -> bool:
    values = np.asarray(features, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return True
    return bool(np.max(np.abs(values)) <= atol)
