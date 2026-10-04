"""Intrinsic dimension estimators for exp-010, via scikit-dimension: Levina-Bickel and TwoNN.

Levina-Bickel maximum-likelihood intrinsic dimension:

Levina & Bickel 2004, "Maximum likelihood estimation of intrinsic dimension", Eqs. 8-9:

    m_k(x) = [ 1/(k-1) sum_{j=1}^{k-1} log(T_k(x) / T_j(x)) ]^{-1}
    m_k    = 1/n sum_i m_k(x_i)
    m      = 1/(k2-k1+1) sum_{k=k1}^{k2} m_k                  (paper: k1 = 10, k2 = 20)

`skdim.id.MLE` (default noise-free settings) computes m_k(x) for one k. Its default aggregation
`comb="mle"` is the MacKay-Ghahramani harmonic mean; the paper's plain mean is `comb="mean"`.
The average over k is done here, on one precomputed kNN search (skdim's `get_nn`, sklearn,
float64, self excluded).

Exact duplicate rows (e.g. identical captions) are removed first: a zero distance makes
log T_k / T_j infinite. The number of unique points is returned.
"""

from __future__ import annotations

import numpy as np
import skdim
from skdim._commonfuncs import get_nn
from sklearn.decomposition import PCA


def levina_bickel(x, k1: int = 10, k2: int = 20, n_jobs: int = 1) -> dict:
    """x: [n, d] array. Returns {"id": m, "id_k": {k: m_k}, "n_unique": int}."""
    x = np.unique(np.asarray(x, dtype=np.float64), axis=0)
    dists, idx = get_nn(x, k=k2, n_jobs=n_jobs)
    id_k = {}
    for k in range(k1, k2 + 1):
        est = skdim.id.MLE().fit(x, precomputed_knn_arrays=(dists[:, :k], idx[:, :k]), comb="mean")
        id_k[k] = float(est.dimension_)
    return {"id": sum(id_k.values()) / len(id_k), "id_k": id_k, "n_unique": len(x)}


def two_nn(x, discard_fraction: float = 0.1, n_jobs: int = 1) -> dict:
    """TwoNN (Facco et al. 2017) as in DADApy's `compute_id_2NN` (algorithm="base"), via skdim.

    mu_i = r_i2 / r_i1; the largest `discard_fraction` of the mu are dropped; the ID is the slope
    of the least-squares line through the origin of -log(1 - i/N) against log mu_(i) (sorted,
    N = all points). This is what Valeriani et al. 2023 use (DADApy, mu_fraction = 0.9).

    x: [n, d] array; duplicate rows dropped first. Returns {"id": d, "n_unique": int}.
    """
    x = np.unique(np.asarray(x, dtype=np.float64), axis=0)
    dists, _ = get_nn(x, k=2, n_jobs=n_jobs)
    est = skdim.id.TwoNN(discard_fraction=discard_fraction, dist=True).fit(dists)
    return {"id": float(est.dimension_), "n_unique": len(x)}


def pca_dim(x, variance: float = 0.9) -> dict:
    """Linear dimension from the PCA spectrum (sklearn PCA, centred), via skdim `lPCA`.

    - "id" = "n90": number of principal components needed for `variance` of the total variance
      (`lPCA(ver="ratio")`); the "PC-ID" Ansuini et al. 2019 compare with the nonlinear ID
    - "pr": participation ratio (sum lambda)^2 / sum lambda^2 (`lPCA(ver="participation_ratio")`)

    Both are bounded by min(n - 1, d).
    """
    ev = PCA().fit(np.asarray(x, dtype=np.float64)).explained_variance_
    n90 = skdim.id.lPCA(ver="ratio", alphaRatio=variance, fit_explained_variance=True).fit(ev)
    pr = skdim.id.lPCA(ver="participation_ratio", fit_explained_variance=True).fit(ev)
    return {"id": int(n90.dimension_), "n90": int(n90.dimension_), "pr": float(pr.dimension_)}


ESTIMATORS = {"lb": levina_bickel, "twonn": two_nn, "pca": pca_dim}
