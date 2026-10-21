"""Truncated cross-covariance / CCA metrics for mutual kNN neighbour search."""

from __future__ import annotations

import numpy as np

from prh_replication.crosscov_weighted_cka import raw_power_weights, reliability_weights
from prh_replication.spectral_calibration import knn_indices, mutual_knn, sq_euclidean_rows


def metric_factor(directions: np.ndarray, weights: np.ndarray, rank: int) -> np.ndarray:
    """F_r = L_r diag(sqrt(w_r)), shape (d, r)."""
    l = np.asarray(directions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    r = int(rank)
    if r < 1 or r > l.shape[1] or r > w.shape[0]:
        raise ValueError(f"invalid rank {r} for directions {l.shape} weights {w.shape}")
    return l[:, :r] * np.sqrt(w[:r]).reshape(1, -1)


def apply_metric_factor(scores: np.ndarray, factor: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) @ np.asarray(factor, dtype=np.float64)


def quadratic_form_distance_sq(scores: np.ndarray, directions: np.ndarray, weights: np.ndarray, rank: int) -> np.ndarray:
    """Explicit (x-y)^T M (x-y) for all pairs; for verification only."""
    f = metric_factor(directions, weights, rank)
    m = f @ f.T
    x = np.asarray(scores, dtype=np.float64)
    # d_ij^2 = (x_i-x_j)^T M (x_i-x_j) = g_ii + g_jj - 2 g_ij with g = x M x^T
    g = x @ m @ x.T
    diag = np.diag(g)
    return diag[:, None] + diag[None, :] - 2.0 * g


def factorized_distance_sq(scores: np.ndarray, directions: np.ndarray, weights: np.ndarray, rank: int) -> np.ndarray:
    return sq_euclidean_rows(apply_metric_factor(scores, metric_factor(directions, weights, rank)))


def knn_indices_stable(distances: np.ndarray, k: int, priorities: np.ndarray | None = None) -> np.ndarray:
    """kNN with deterministic tie-break: smaller distance, then lower priority, then lower column index."""
    work = np.array(distances, copy=True, dtype=np.float64)
    n = work.shape[0]
    np.fill_diagonal(work, np.inf)
    k = int(min(k, n - 1))
    if priorities is None:
        priorities = np.arange(n, dtype=np.float64)
    else:
        priorities = np.asarray(priorities, dtype=np.float64)
        if priorities.shape != (n,):
            raise ValueError("priorities must be length-n")
    # lexsort last key is primary: we want ascending distance, then priority, then index
    order = np.lexsort(
        (
            np.broadcast_to(np.arange(n, dtype=np.float64), (n, n)),
            np.broadcast_to(priorities, (n, n)),
            work,
        ),
        axis=1,
    )
    return order[:, :k]


def mknn_from_features(left: np.ndarray, right: np.ndarray, k: int = 10, priorities_left=None, priorities_right=None) -> dict:
    d_left = sq_euclidean_rows(left)
    d_right = sq_euclidean_rows(right)
    knn_l = knn_indices_stable(d_left, k, priorities_left)
    knn_r = knn_indices_stable(d_right, k, priorities_right)
    score = mutual_knn(knn_l, knn_r)
    # tie diagnostics at k-th neighbour boundary
    n = d_left.shape[0]
    tie_left = 0
    tie_right = 0
    zero_off = 0
    atol = 1e-12
    for i in range(n):
        row_l = d_left[i].copy()
        row_r = d_right[i].copy()
        row_l[i] = np.inf
        row_r[i] = np.inf
        kth_l = row_l[knn_l[i, -1]]
        kth_r = row_r[knn_r[i, -1]]
        tie_left += int(np.sum(np.abs(row_l - kth_l) <= atol) > 1)
        tie_right += int(np.sum(np.abs(row_r - kth_r) <= atol) > 1)
        zero_off += int(np.sum((row_l <= atol) & np.isfinite(row_l)))
        zero_off += int(np.sum((row_r <= atol) & np.isfinite(row_r)))
    return {
        "mknn": float(score),
        "knn_left": knn_l,
        "knn_right": knn_r,
        "n_rows_with_kth_tie_left": int(tie_left),
        "n_rows_with_kth_tie_right": int(tie_right),
        "n_zero_distance_offdiag_counts": int(zero_off),
    }


def mknn_metric(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    l_a: np.ndarray,
    l_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    rank: int,
    k: int = 10,
    priorities_left=None,
    priorities_right=None,
) -> dict:
    left = apply_metric_factor(scores_a, metric_factor(l_a, w_a, rank))
    right = apply_metric_factor(scores_b, metric_factor(l_b, w_b, rank))
    out = mknn_from_features(left, right, k=k, priorities_left=priorities_left, priorities_right=priorities_right)
    out["rank"] = int(rank)
    out["embedded_dim"] = int(rank)
    return out


def select_rank(curve: list[dict], tie_atol: float) -> int:
    """curve items have keys rank, mknn. Prefer max mknn; ties → larger rank."""
    best = max(float(row["mknn"]) for row in curve)
    tied = [row for row in curve if abs(float(row["mknn"]) - best) <= float(tie_atol)]
    tied.sort(key=lambda row: -int(row["rank"]))
    return int(tied[0]["rank"])


def build_weights(
    family: str,
    sigma: np.ndarray,
    rho: np.ndarray,
    alpha: float | None,
    beta: float | None,
    sigma_floor: float,
    rho_floor: float,
    normalize: bool,
) -> np.ndarray:
    d = int(sigma.shape[0])
    if family == "raw_equal":
        w = np.ones(d, dtype=np.float64)
    elif family == "raw_power_selected":
        if alpha is None:
            raise ValueError("alpha required")
        w = raw_power_weights(sigma, float(alpha), float(sigma_floor), bool(normalize))["w"]
    elif family == "cca_equal":
        w = np.ones(d, dtype=np.float64)
    elif family == "cca_beta1":
        w = reliability_weights(rho, 1.0, float(rho_floor), bool(normalize))["w"]
    elif family == "cca_beta_selected":
        if beta is None:
            raise ValueError("beta required")
        w = reliability_weights(rho, float(beta), float(rho_floor), bool(normalize))["w"]
    else:
        raise ValueError(family)
    if normalize and family.endswith("equal") and float(np.mean(w)) > 0:
        w = w / float(np.mean(w))
    return np.asarray(w, dtype=np.float64)


def family_directions(family: str, svd: dict, cca: dict) -> tuple[np.ndarray, np.ndarray]:
    if family.startswith("raw"):
        return svd["U"], svd["V"]
    return cca["L_A"], cca["L_B"]
