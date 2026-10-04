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


def _dadapy_two_nn(x: np.ndarray, mu_fraction: float = 0.9) -> float:
    """DADApy `compute_id_2NN` (algorithm="base") written out: sorted log mu, keep the lowest
    int(N * mu_fraction), regress -log(1 - i/N) on them through the origin."""
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    d = np.sort(d, axis=1)
    n = len(x)
    n_eff = int(n * mu_fraction)
    log_mu = np.sort(np.log(d[:, 1] / d[:, 0]))[:n_eff]
    y = -np.log(1 - np.arange(1, n_eff + 1) / n)
    return float((log_mu @ y) / (log_mu @ log_mu))


def test_two_nn_matches_dadapy_formula():
    x = np.random.default_rng(3).standard_normal((300, 7))
    assert idim.two_nn(x)["id"] == pytest.approx(_dadapy_two_nn(x), rel=1e-10)


def test_two_nn_drops_duplicates():
    x = np.random.default_rng(4).standard_normal((300, 5))
    res = idim.two_nn(np.concatenate([x, x[:20]]))
    assert res["n_unique"] == 300
    assert res["id"] == pytest.approx(idim.two_nn(x)["id"])


@pytest.mark.parametrize("m", [2, 5, 10])
def test_two_nn_recovers_dimension_of_embedded_gaussian(m):
    rng = np.random.default_rng(10 + m)
    basis = np.linalg.qr(rng.standard_normal((200, m)))[0]
    x = rng.standard_normal((4000, m)) @ basis.T + 5.0
    assert idim.two_nn(x)["id"] == pytest.approx(m, rel=0.15)


def test_pca_dim_matches_spectrum():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((500, 30)) * np.linspace(3, 0.1, 30)  # anisotropic spectrum
    lam = np.sort(np.linalg.eigvalsh(np.cov(x, rowvar=False)))[::-1]
    frac = np.cumsum(lam) / lam.sum()
    res = idim.pca_dim(x)
    assert res["n90"] == int(np.argmax(frac >= 0.9)) + 1
    assert res["pr"] == pytest.approx(lam.sum() ** 2 / (lam**2).sum(), rel=1e-8)


def test_pca_dim_of_flat_subspace():
    rng = np.random.default_rng(6)
    basis = np.linalg.qr(rng.standard_normal((100, 5)))[0]
    x = rng.standard_normal((2000, 5)) @ basis.T + 1e-4 * rng.standard_normal((2000, 100))
    res = idim.pca_dim(x)
    assert res["n90"] == 5
    assert res["pr"] == pytest.approx(5, rel=0.1)
