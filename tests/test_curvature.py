import numpy as np
import pytest

from geoprh import curvature as cv
from geoprh import synthetic as syn


def _estimate(manifold, d=2, n=20000, ambient=256):
    rng = np.random.default_rng(0)
    z, truth = syn.MANIFOLDS[manifold](n, d, rng)
    x = syn.embed(z, max(ambient, z.shape[1]), rng)
    mom = cv.neighbourhood_moments(x, d, [10, 20, 40, 80, 160], n_centres=200)
    return cv.curvature_from_moments(mom, d), truth


@pytest.mark.parametrize("manifold", ["flat", "swiss", "torus_rff"])
def test_intrinsically_flat(manifold):
    est, truth = _estimate(manifold)
    assert truth["scalar"] == 0
    assert abs(est["S_norm"]) < 0.05


@pytest.mark.parametrize("manifold", ["sphere", "veronese"])
def test_unit_sphere_metric(manifold):
    est, truth = _estimate(manifold)
    assert truth["scalar"] == 2
    assert est["S_norm"] == pytest.approx(1.0, abs=0.15)


def test_sphere_extrinsic_terms():
    est, _ = _estimate("sphere")
    assert est["H2"] == pytest.approx(4, rel=0.1)  # |H|^2 = d^2
    assert est["II2"] == pytest.approx(2, rel=0.15)  # |II|^2 = d


def test_veronese_is_locally_isometric_to_the_sphere():
    rng = np.random.default_rng(1)
    d = 4
    z0 = rng.standard_normal(d + 1)
    z0 /= np.linalg.norm(z0)
    v = rng.standard_normal(d + 1)
    v -= (v @ z0) * z0  # tangent at z0
    v *= 1e-4 / np.linalg.norm(v)
    iu = np.triu_indices(d + 1)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(2.0)

    def phi(u):
        u = u / np.linalg.norm(u)
        return u[iu[0]] * u[iu[1]] * scale

    assert np.linalg.norm(phi(z0 + v) - phi(z0)) == pytest.approx(1e-4, rel=1e-3)


def test_torus_metric_is_constant():
    # the pullback metric sum_j a_j^2 w_j w_j^T does not depend on theta: distances between two
    # nearby points with the same offset are equal wherever they sit
    rng = np.random.default_rng(2)
    d = 3
    w = rng.integers(-3, 4, size=(64, d))
    w = w[np.abs(w).sum(axis=1) > 0]
    a = np.linalg.norm(w, axis=1) ** -0.5

    def emb(theta):
        ph = w @ theta
        return np.concatenate([a * np.cos(ph), a * np.sin(ph)])

    step = 1e-4 * np.array([1.0, -2.0, 0.5])
    dists = [np.linalg.norm(emb(t + step) - emb(t)) for t in rng.uniform(0, 2 * np.pi, size=(5, d))]
    assert np.ptp(dists) < 1e-3 * np.mean(dists)
