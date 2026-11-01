"""Paired per-sample statistics between two models (exp-014).

For a per-sample quantity q (node Ollivier-Ricci curvature, local density) computed in every
layer of model A and of model B on the same samples, the agreement of layers i and j is the
Spearman correlation across samples of q_A[i] and q_B[j]. A rank correlation ignores any
monotone, model- or layer-specific bias of q (a shift set by the ID or k, a rescaling), so it
asks only whether the same samples are high or low in both models.

`residualize` removes a covariate (density) from q within each layer on the rank scale:
ranks of q are regressed on ranks of the covariate and the residuals kept, so the correlation of
residuals is the partial Spearman correlation given each model's own density.

All layer pairs at once: with Z the standardized ranks [L, n], Spearman = Z_A Z_B^T / n. The
permutation null shuffles the pairing of B's samples (the same permutation for all of B's
layers, as in exp-012) and each cell is calibrated with `layer_calibration.calibrate_cells`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata

from geoprh.layer_calibration import calibrate_cells


def standardized_ranks(q: np.ndarray) -> np.ndarray:
    """[L, n] -> per-row ranks, centred and scaled to unit variance."""
    r = rankdata(q, axis=1)
    r = r - r.mean(axis=1, keepdims=True)
    return r / r.std(axis=1, keepdims=True)


def residualize(q: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Standardized residual ranks of q [L, n] after removing the ranks of covariate [L, n]."""
    zq, zc = standardized_ranks(q), standardized_ranks(covariate)
    beta = (zq * zc).mean(axis=1, keepdims=True)
    res = zq - beta * zc
    return res / res.std(axis=1, keepdims=True)


def rank_corr_matrix(za: np.ndarray, zb: np.ndarray, perm: np.ndarray | None = None) -> np.ndarray:
    """[La, Lb] Pearson correlations of standardized rows (Spearman if rows are ranks)."""
    if perm is not None:
        zb = zb[:, perm]
    return za @ zb.T / za.shape[1]


def calibrated_rank_corr(
    za: np.ndarray, zb: np.ndarray, permutations: int = 200, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Observed [La, Lb] correlations, calibrated cell by cell against a pairing-permutation null."""
    rng = np.random.default_rng(seed)
    n = za.shape[1]
    obs = rank_corr_matrix(za, zb)
    null = np.stack([rank_corr_matrix(za, zb, rng.permutation(n)) for _ in range(permutations)])
    res = calibrate_cells(torch.from_numpy(obs), torch.from_numpy(null), alpha)
    return {k: v.numpy() for k, v in res.items()}
