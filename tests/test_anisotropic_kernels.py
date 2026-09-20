"""Anisotropic metric kernels: identity, bounds, contractions, b, neighbours."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prh_replication.anisotropic_kernels import (
    EIG_HI,
    EIG_LO,
    fit_pca,
    identity_deltas,
    knn_from_sq,
    knn_overlap,
    make_pack,
    metric_gram,
    metric_sq_distances,
    reconstruct_b,
    scores_numpy,
    truncation_gram,
)
from prh_replication.kernels import center_gram_o2, extension_stats
from prh_replication.metrics import nearest_neighbors


def _pair(n=20, da=8, db=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    xa = F.normalize(torch.randn(n, da, generator=g), dim=-1)
    xb = F.normalize(torch.randn(n, db, generator=g), dim=-1)
    return xa.numpy().astype(np.float64), xb.numpy().astype(np.float64)


def test_identity_recovers_linear_cka():
    xa, xb = _pair()
    mu_a, mu_b = xa.mean(0), xb.mean(0)
    za, zb = xa - mu_a, xb - mu_b
    pca_a, pca_b = fit_pca(za, q=4), fit_pca(zb, q=4)
    pack = make_pack(za, zb, pca_a["U"], pca_b["U"])
    da, db = identity_deltas(pca_a["q"], pca_b["q"])
    sc = scores_numpy(da, db, pack)
    k = torch.tensor(xa @ xa.T)
    l = torch.tensor(xb @ xb.T)
    direct = extension_stats(k, l)
    assert sc["a"] == pytest.approx(direct["a"], rel=1e-8, abs=1e-8)
    assert sc["ratio"] == pytest.approx(direct["ratio"], rel=1e-8, abs=1e-8)
    k_m = metric_gram(za, pca_a["U"], np.eye(pca_a["q"]))
    kc = center_gram_o2(torch.tensor(xa @ xa.T))
    km = center_gram_o2(torch.tensor(k_m))
    assert torch.allclose(kc, km, atol=1e-8)


def test_eigenvalue_bounds():
    q = 6
    huge = torch.tensor([10.0] * q, dtype=torch.float64)
    b, logb, eig = reconstruct_b("diag", huge, q)
    assert float(eig.max()) == pytest.approx(EIG_HI, rel=1e-12)
    assert float(eig.min()) >= EIG_LO
    tiny = torch.tensor([-10.0] * q, dtype=torch.float64)
    _, _, eig2 = reconstruct_b("diag", tiny, q)
    assert float(eig2.min()) == pytest.approx(EIG_LO, rel=1e-12)
    vech = torch.zeros(q * (q + 1) // 2, dtype=torch.float64)
    vech[0] = 8.0
    b3, _, eig3 = reconstruct_b("full", vech, q)
    assert float(eig3.max()) <= EIG_HI + 1e-10
    assert float(eig3.min()) >= EIG_LO - 1e-10
    assert torch.allclose(b3, b3.T, atol=1e-10)


def test_contraction_matches_direct_gram():
    xa, xb = _pair(n=16, da=7, db=9, seed=2)
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    pa, pb = fit_pca(za, q=3), fit_pca(zb, q=4)
    pack = make_pack(za, zb, pa["U"], pb["U"])
    rng = np.random.default_rng(3)
    ha = rng.uniform(-0.4, 0.4, size=pa["q"])
    hb = rng.uniform(-0.4, 0.4, size=pb["q"])
    ba, bb = np.diag(np.exp(ha)), np.diag(np.exp(hb))
    ka = metric_gram(za, pa["U"], ba)
    kb = metric_gram(zb, pb["U"], bb)
    direct = extension_stats(torch.tensor(ka), torch.tensor(kb))
    sc = scores_numpy(ba - np.eye(pa["q"]), bb - np.eye(pb["q"]), pack)
    assert sc["a"] == pytest.approx(direct["a"], rel=1e-7, abs=1e-7)
    assert sc["b"] == pytest.approx(direct["b"], rel=1e-7, abs=1e-7)
    assert sc["ratio"] == pytest.approx(direct["ratio"], rel=1e-7, abs=1e-7)


def test_gradient_parity():
    xa, xb = _pair(n=14, da=6, db=8, seed=4)
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    pa, pb = fit_pca(za, q=3), fit_pca(zb, q=3)
    pack = make_pack(za, zb, pa["U"], pb["U"])
    from prh_replication.anisotropic_kernels import pack_to_torch, _scores_from_deltas

    p = pack_to_torch(pack)
    ha = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64, requires_grad=True)
    da = torch.diag(torch.exp(ha) - 1.0)
    db = torch.zeros(3, 3, dtype=torch.float64)
    a = _scores_from_deltas(da, db, p)["a"]
    a.backward()
    g_auto = ha.grad.clone()
    g_fd = []
    eps = 1e-6
    with torch.no_grad():
        for i in range(3):
            hp = ha.detach().clone()
            hp[i] += eps
            am = _scores_from_deltas(torch.diag(torch.exp(hp) - 1.0), db, p)["a"]
            hm = ha.detach().clone()
            hm[i] -= eps
            al = _scores_from_deltas(torch.diag(torch.exp(hm) - 1.0), db, p)["a"]
            g_fd.append(float((am - al) / (2 * eps)))
    assert np.allclose(g_auto.numpy(), np.array(g_fd), atol=2e-5, rtol=2e-4)


def test_analytic_shuffle_mean_b():
    xa, xb = _pair(n=18, da=5, db=6, seed=5)
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    pa, pb = fit_pca(za, q=3), fit_pca(zb, q=3)
    pack = make_pack(za, zb, pa["U"], pb["U"])
    ba = np.diag(np.exp(np.array([0.2, 0.0, -0.15])))
    bb = np.diag(np.exp(np.array([-0.1, 0.3, 0.05])))
    ka = torch.tensor(metric_gram(za, pa["U"], ba))
    kb = torch.tensor(metric_gram(zb, pb["U"], bb))
    st = extension_stats(ka, kb)
    n = ka.shape[0]
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(40):
        perm = torch.tensor(rng.permutation(n))
        scores.append(extension_stats(ka, kb[perm][:, perm])["a"])
    mc = float(np.mean(scores))
    assert st["b"] == pytest.approx(mc, rel=0.08, abs=0.02)


def test_identity_neighbours_match_inner_product():
    g = torch.Generator().manual_seed(6)
    x = F.normalize(torch.randn(24, 10, generator=g), dim=-1)
    z = x - x.mean(0)
    pca = fit_pca(z.numpy(), q=5)
    dsq = metric_sq_distances(z.numpy(), pca["U"], np.eye(pca["q"]))
    knn_m = knn_from_sq(dsq, 10)
    knn_ip = nearest_neighbors(x, 10).numpy()
    # Distance ranking on unit vectors equals inner-product ranking (ties excluded by unique noise).
    overlap = knn_overlap(knn_m, knn_ip)
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_truncation_differs_from_identity():
    xa, xb = _pair(n=30, da=12, db=12, seed=7)
    za = xa - xa.mean(0)
    pca = fit_pca(za, q=3)
    kt = truncation_gram(za, pca["U"])
    kfull = za @ za.T
    assert not np.allclose(kt, kfull, atol=1e-6)
    assert pca["variance_fraction"] < 0.999


def test_pca_rank_reduction():
    z = np.zeros((40, 10))
    z[:, 0] = np.linspace(-1, 1, 40)
    z[:, 1] = 2 * z[:, 0]
    rec = fit_pca(z, q=32)
    assert rec["q"] <= 2
    assert rec["reduced"] is True
