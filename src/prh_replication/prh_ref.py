"""PRH released-code alignment protocol (platonic-rep@dcd76ba).

Operational reference: extract_features.py + measure_alignment.py.
Does not include paper-only concatenated features unless explicitly requested.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from prh_replication.metrics import (
    linear_cka,
    nearest_neighbors,
    rbf_cka,
    remove_outliers,
)


PRH_TOPK = 10
PRH_CLIP_Q = 0.95
PRH_RBF_SIGMA = 1.0  # official AlignmentMetrics.cka default
PRH_CKA_EPS = 1e-6


def prh_prepare_layer(feats: torch.Tensor) -> torch.Tensor:
    """Clip then L2, matching measure_alignment.prepare_features + compute_score."""
    return F.normalize(remove_outliers(feats.float(), q=PRH_CLIP_Q, exact=True), p=2, dim=-1)


def stack_prepared(feats_nld: torch.Tensor) -> torch.Tensor:
    layers = [prh_prepare_layer(feats_nld[:, i]) for i in range(feats_nld.shape[1])]
    return torch.stack(layers, dim=1)


def concat_prepared(prepared_nld: torch.Tensor) -> torch.Tensor:
    """Paper Appendix C extra candidate: concatenate already-prepared layer vectors."""
    return prepared_nld.flatten(start_dim=1)


def mknn_overlap_fast(knn_a: torch.Tensor, knn_b: torch.Tensor) -> float:
    eq = knn_a.unsqueeze(2) == knn_b.unsqueeze(1)
    topk = knn_a.shape[1]
    return float((eq.any(dim=2).float().sum(dim=1) / topk).mean().item())


def max_mknn(xa: torch.Tensor, xb: torch.Tensor, topk: int = PRH_TOPK) -> tuple[float, tuple[int, int]]:
    knn_x = [nearest_neighbors(xa[:, i], topk) for i in range(xa.shape[1])]
    knn_y = [nearest_neighbors(xb[:, j], topk) for j in range(xb.shape[1])]
    best, best_ij = float("-inf"), (0, 0)
    for i, kx in enumerate(knn_x):
        for j, ky in enumerate(knn_y):
            s = mknn_overlap_fast(kx, ky)
            if s > best:
                best, best_ij = s, (i, j)
    return best, best_ij


def max_linear_cka(xa: torch.Tensor, xb: torch.Tensor) -> tuple[float, tuple[int, int]]:
    best, best_ij = float("-inf"), (0, 0)
    for i in range(xa.shape[1]):
        for j in range(xb.shape[1]):
            s = linear_cka(xa[:, i], xb[:, j])
            if s > best:
                best, best_ij = s, (i, j)
    return best, best_ij


def max_rbf_cka(xa: torch.Tensor, xb: torch.Tensor, sigma: float = PRH_RBF_SIGMA) -> tuple[float, tuple[int, int]]:
    best, best_ij = float("-inf"), (0, 0)
    for i in range(xa.shape[1]):
        for j in range(xb.shape[1]):
            s = rbf_cka(xa[:, i], xb[:, j], sigma, sigma)
            if s > best:
                best, best_ij = s, (i, j)
    return best, best_ij


def _apply_perm_knn(knn: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Map neighbour indices after feats_b -> feats_b[perm]."""
    n = perm.shape[0]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=perm.device)
    return inv[knn[perm]]


def selection_aware_mknn_null(
    xa: torch.Tensor,
    xb: torch.Tensor,
    n_perm: int = 100,
    seed: int = 0,
    topk: int = PRH_TOPK,
) -> dict:
    """Max-over-layers mNN under one-sided correspondence permutation.

    One permutation is reused across all layer candidates (required for a
    null of the selected maximum). Geometry of each side is preserved.
    """
    g = torch.Generator().manual_seed(seed)
    knn_x = [nearest_neighbors(xa[:, i], topk) for i in range(xa.shape[1])]
    knn_y = [nearest_neighbors(xb[:, j], topk) for j in range(xb.shape[1])]
    n = xa.shape[0]
    scores = []
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=g)
        best = float("-inf")
        for kx in knn_x:
            for ky in knn_y:
                s = mknn_overlap_fast(kx, _apply_perm_knn(ky, perm))
                if s > best:
                    best = s
        scores.append(best)
    t = torch.tensor(scores)
    return {
        "shuffle_mean": float(t.mean()),
        "shuffle_std": float(t.std(unbiased=True)) if n_perm > 1 else 0.0,
        "n_perm": n_perm,
        "null_label": "selection-aware one-sided permutation null for max-over-layers mNN (not a CI)",
    }


def selection_aware_cka_null(
    xa: torch.Tensor,
    xb: torch.Tensor,
    n_perm: int = 100,
    seed: int = 0,
    score_fn: Callable | None = None,
) -> dict:
    score_fn = score_fn or (lambda u, v: linear_cka(u, v))
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    scores = []
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=g)
        best = float("-inf")
        y = xb[perm]
        for i in range(xa.shape[1]):
            for j in range(xb.shape[1]):
                s = score_fn(xa[:, i], y[:, j])
                if s > best:
                    best = s
        scores.append(best)
    t = torch.tensor(scores)
    return {
        "shuffle_mean": float(t.mean()),
        "shuffle_std": float(t.std(unbiased=True)) if n_perm > 1 else 0.0,
        "n_perm": n_perm,
        "null_label": "selection-aware one-sided permutation null for max-over-layers CKA (not a CI)",
    }


def fixed_layer_mknn_null(x: torch.Tensor, y: torch.Tensor, n_perm: int = 100, seed: int = 0, topk: int = PRH_TOPK) -> dict:
    from prh_replication.controls import mknn_shuffle

    sh = mknn_shuffle(x, y, topk, n_perm, seed)
    return {
        "shuffle_mean": sh.shuffle_mean,
        "shuffle_std": sh.shuffle_std,
        "n_perm": n_perm,
        "null_label": "fixed-layer one-sided permutation null (sanity ~ k/(n-1); not a null for the max statistic)",
    }


def centred_gram_diag_share(feats: torch.Tensor) -> dict:
    """Diagnostic for RBF near-identity claims: off-diagonal kernel mass and centred Gram diagonal share."""
    dist = torch.cdist(feats, feats)
    k = torch.exp(-dist.pow(2) / (2.0 * PRH_RBF_SIGMA ** 2))
    n = k.shape[0]
    h = torch.eye(n, dtype=k.dtype, device=k.device) - 1.0 / n
    kc = h @ k @ h
    fro2 = torch.sum(kc * kc)
    diag2 = torch.sum(torch.diag(kc) ** 2)
    off = fro2 - diag2
    off_mask = ~torch.eye(n, dtype=torch.bool, device=k.device)
    return {
        "centred_diag_energy_frac": float((diag2 / fro2).item()) if float(fro2) > 0 else float("nan"),
        "centred_offdiag_energy_frac": float((off / fro2).item()) if float(fro2) > 0 else float("nan"),
        "mean_kernel": float(k.mean()),
        "mean_offdiag_kernel": float(k[off_mask].mean()),
    }
