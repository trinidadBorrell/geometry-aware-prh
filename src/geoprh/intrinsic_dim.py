"""Levina-Bickel maximum-likelihood intrinsic dimension (exp-010), via scikit-dimension.

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


def levina_bickel(x, k1: int = 10, k2: int = 20, n_jobs: int = 1) -> dict:
    """x: [n, d] array. Returns {"id": m, "id_k": {k: m_k}, "n_unique": int}."""
    x = np.unique(np.asarray(x, dtype=np.float64), axis=0)
    dists, idx = get_nn(x, k=k2, n_jobs=n_jobs)
    id_k = {}
    for k in range(k1, k2 + 1):
        est = skdim.id.MLE().fit(x, precomputed_knn_arrays=(dists[:, :k], idx[:, :k]), comb="mean")
        id_k[k] = float(est.dimension_)
    return {"id": sum(id_k.values()) / len(id_k), "id_k": id_k, "n_unique": len(x)}
