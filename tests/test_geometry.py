import numpy as np
import pytest

from geoprh import geometry as geo
from geoprh import synthetic as syn


def test_delta_four_cycle_and_tree():
    # 4-cycle with unit edges: sums 2, 4, 2 -> delta = (4 - 2) / 2 = 1, diameter 2
    cycle = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]], float)
    out = geo.delta_hyperbolicity(cycle, n_quads=5000)
    assert out["delta"] == pytest.approx(1.0)
    assert out["delta_rel"] == pytest.approx(1.0)
    # points on a path are a tree metric: delta = 0
    p = np.arange(10.0)
    assert geo.delta_hyperbolicity(np.abs(p[:, None] - p[None]), n_quads=5000)["delta"] == 0


def test_geodesic_equals_euclid_on_a_line():
    x = np.column_stack([np.linspace(0, 1, 50), np.zeros(50)])
    d = geo.geodesic_distances(geo.knn_graph(x, 2), sources=[0])
    np.testing.assert_allclose(d[0], x[:, 0], atol=1e-12)


def test_sphere_truth_and_embedding_isometry():
    rng = np.random.default_rng(0)
    z, truth = syn.sphere(300, 3, rng)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0)
    g = truth["geodesic"](np.arange(5))
    assert g.shape == (5, 300) and g.max() <= np.pi + 1e-9
    x = syn.embed(z, 64, rng)
    dz = np.linalg.norm(z[:5, None] - z[None], axis=-1)
    dx = np.linalg.norm(x[:5, None] - x[None], axis=-1)
    np.testing.assert_allclose(dx, dz, atol=1e-10)


def test_swiss_arc_length():
    t = np.linspace(0, 3, 30001)
    numeric = np.concatenate([[0], np.cumsum(np.sqrt(1 + t[:-1] ** 2) * np.diff(t))])
    assert syn._spiral_arc_length(np.array([3.0]))[0] == pytest.approx(numeric[-1], rel=1e-3)


def test_local_pca_flat_vs_sphere():
    rng = np.random.default_rng(0)
    flat = syn.embed(syn.flat(2000, 2, rng)[0], 20, rng)
    prof = geo.local_pca_profile(flat, [10, 200], n_centres=30)
    assert [p["dim"] for p in prof] == pytest.approx([2, 2], abs=0.1)
    circle = syn.embed(syn.sphere(2000, 1, rng)[0], 20, rng)
    prof = geo.local_pca_profile(circle, [5, 1000], n_centres=30)
    assert prof[0]["dim"] == 1 and prof[1]["dim"] == 2  # a large arc needs two directions
