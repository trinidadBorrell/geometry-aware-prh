"""geoprh.intrinsic_dim (skdim wrapper) against the paper's formula and data of known dimension."""

import math

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from geoprh import intrinsic_dim as idim


def _literal_lb(x: np.ndarray, k1: int, k2: int) -> float:
    """Levina & Bickel Eqs. 8-9 as loops, brute-force distances."""
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    d = np.sort(d, axis=1)
    n = len(x)
    m_ks = []
    for k in range(k1, k2 + 1):
        per_point = []
        for i in range(n):
            s = sum(math.log(d[i, k - 1] / d[i, j]) for j in range(k - 1))
            per_point.append((k - 1) / s)
        m_ks.append(sum(per_point) / n)
    return sum(m_ks) / len(m_ks)


def test_matches_literal_formula():
    x = np.random.default_rng(1).standard_normal((150, 8))
    assert idim.levina_bickel(x, 10, 20)["id"] == pytest.approx(_literal_lb(x, 10, 20), rel=1e-10)


def test_duplicates_are_dropped():
    x = np.random.default_rng(2).standard_normal((200, 6))
    with_dups = np.concatenate([x, x[:30]])
    res = idim.levina_bickel(with_dups)
    assert res["n_unique"] == 200
    assert res["id"] == pytest.approx(idim.levina_bickel(x)["id"])


@pytest.mark.parametrize("m", [2, 5, 10])
def test_recovers_dimension_of_embedded_gaussian(m):
    rng = np.random.default_rng(m)
    basis = np.linalg.qr(rng.standard_normal((200, m)))[0]  # [200, m] orthonormal
    x = rng.standard_normal((4000, m)) @ basis.T + 5.0
    # the MLE is biased low in higher dimension at finite n; 15% covers m <= 10 at n = 4000
    assert idim.levina_bickel(x)["id"] == pytest.approx(m, rel=0.15)
