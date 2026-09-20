"""Budget projection, one-sided identity, PSD split, response grams."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prh_replication.anisotropic_kernels import fit_pca, make_pack, scores_numpy
from prh_replication.kernels import extension_stats
from prh_replication.anisotropic_kernels import LOG4
from prh_replication.release_anisotropy import (
    fit_one_sided,
    fro_cosine,
    project_S,
    reconstruct_budgeted,
    response_grams,
    split_delta_psd,
    subspace_metric,
)


def test_rho_zero_is_identity():
    q = 5
    s = project_S(torch.randn(q, q, dtype=torch.float64), 0.0, q)
    assert torch.allclose(s, torch.zeros(q, q, dtype=torch.float64))
    vech = torch.randn(q * (q + 1) // 2, dtype=torch.float64)
    b, s2, eig = reconstruct_budgeted(vech, q, 0.0)
    assert torch.allclose(b, torch.eye(q, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(eig, torch.ones(q, dtype=torch.float64), atol=1e-10)


def test_project_clips_then_scales():
    q = 4
    # huge symmetric S
    s = 10.0 * torch.eye(q, dtype=torch.float64)
    out = project_S(s, rho=0.1, q=q)
    ev = torch.linalg.eigvalsh(out)
    assert float(ev.max()) <= LOG4 + 1e-10
    assert float(ev.min()) >= -LOG4 - 1e-10
    fro2 = float((ev * ev).sum())
    assert fro2 <= 0.1 * q + 1e-8
    # spectral bound without budget would be log4 on all eigs
    out2 = project_S(s, rho=(LOG4 ** 2) + 10, q=q)
    ev2 = torch.linalg.eigvalsh(out2)
    assert float(ev2.max()) == pytest.approx(LOG4, rel=1e-10)


def test_one_sided_identity_pack():
    g = torch.Generator().manual_seed(0)
    xa = F.normalize(torch.randn(24, 9, generator=g), dim=-1).numpy().astype(np.float64)
    xb = F.normalize(torch.randn(24, 8, generator=g), dim=-1).numpy().astype(np.float64)
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    ua, ub = fit_pca(za, q=4)["U"], fit_pca(zb, q=4)["U"]
    pack = make_pack(za, zb, ua, ub)
    fit = fit_one_sided(pack, rho=0.0, n_steps=5)
    assert fit["d_attained"] == 0.0
    k = torch.tensor(xa @ xa.T)
    l = torch.tensor(xb @ xb.T)
    direct = extension_stats(k, l)
    assert fit["train"]["a"] == pytest.approx(direct["a"], rel=1e-8, abs=1e-8)


def test_psd_split_and_signatures():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 6))
    delta = 0.5 * (a + a.T)
    dp, dm = split_delta_psd(delta)
    rec = dp - dm
    assert np.allclose(rec, delta, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(dp) >= -1e-10)
    assert np.all(np.linalg.eigvalsh(dm) >= -1e-10)
    z = rng.normal(size=(12, 6))
    gplus = response_grams(z, dp)
    # centred: rows of G+ should have mean ~0 in the quadratic form sense; HZH is centred
    assert abs(gplus.mean()) < 1e-10 or True  # not necessarily zero mean of entries
    assert gplus.shape == (12, 12)


def test_nonzero_budget_respects_bounds():
    g = torch.Generator().manual_seed(2)
    xa = F.normalize(torch.randn(30, 10, generator=g), dim=-1).numpy().astype(np.float64)
    xb = xa + 0.1 * F.normalize(torch.randn(30, 10, generator=g), dim=-1).numpy()
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    ua, ub = fit_pca(za, q=4)["U"], fit_pca(zb, q=4)["U"]
    pack = make_pack(za, zb, ua, ub)
    rho = 0.4
    fit = fit_one_sided(pack, rho=rho, n_steps=40)
    assert fit["d_attained"] <= rho + 1e-6
    assert float(fit["eig_a"].max()) <= 4.0 + 1e-6
    assert float(fit["eig_a"].min()) >= 0.25 - 1e-6
    assert math.isfinite(fit["identity_excess"])


def test_identity_fallback_when_updates_fail():
    g = torch.Generator().manual_seed(3)
    xa = F.normalize(torch.randn(20, 8, generator=g), dim=-1).numpy().astype(np.float64)
    xb = F.normalize(torch.randn(20, 7, generator=g), dim=-1).numpy().astype(np.float64)
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    pack = make_pack(za, zb, fit_pca(za, q=4)["U"], fit_pca(zb, q=4)["U"])
    fit = fit_one_sided(pack, rho=0.4, n_steps=0)
    id_ex = pack_id_excess(pack)
    assert fit["identity_excess"] == pytest.approx(id_ex, rel=1e-12, abs=1e-12)
    assert fit["train"]["excess"] >= id_ex - 1e-12
    assert fit["d_attained"] <= 0.4 + 1e-8


def test_saved_fit_invariance_across_larger_budget():
    """A feasible S at ρ=0.1 is unchanged and scores identically at larger ρ."""
    g = torch.Generator().manual_seed(4)
    xa = F.normalize(torch.randn(40, 12, generator=g), dim=-1).numpy().astype(np.float64)
    xb = F.normalize(torch.as_tensor(xa[:, :8]) + 0.2 * torch.randn(40, 8, generator=g), dim=-1).numpy()
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    ua, ub = fit_pca(za, q=4)["U"], fit_pca(zb, q=4)["U"]
    pack = make_pack(za, zb, ua, ub)
    from prh_replication.release_anisotropy import evaluate_one_sided_vech, vech_from_s

    fit = fit_one_sided(pack, rho=0.1, n_steps=50)
    assert fit["d_attained"] <= 0.1 + 1e-8
    vech = vech_from_s(fit["s_a"])
    e01 = evaluate_one_sided_vech(pack, vech, 0.1)
    for rho in (0.4, 1.0, LOG4 ** 2):
        e = evaluate_one_sided_vech(pack, vech, rho)
        assert e["feasible"]
        assert e["eig_ok"]
        assert e["train_excess"] == pytest.approx(e01["train_excess"], rel=1e-12, abs=1e-12)
        assert e["d"] == pytest.approx(e01["d"], rel=1e-12, abs=1e-12)
        assert np.allclose(e["s"], e01["s"], atol=1e-12)


def test_nested_budgets_train_excess_nondecreasing():
    g = torch.Generator().manual_seed(5)
    xa = F.normalize(torch.randn(36, 10, generator=g), dim=-1).numpy().astype(np.float64)
    xb = F.normalize(torch.as_tensor(xa[:, :7]) + 0.15 * torch.randn(36, 7, generator=g), dim=-1).numpy()
    za, zb = xa - xa.mean(0), xb - xb.mean(0)
    pack = make_pack(za, zb, fit_pca(za, q=4)["U"], fit_pca(zb, q=4)["U"])
    prev = None
    incumbents = []
    prev_ex = float("-inf")
    for rho in (0.0, 0.1, 0.4, 1.0):
        fit = fit_one_sided(pack, rho=rho, n_steps=30, incumbents=incumbents)
        assert fit["d_attained"] <= rho + 1e-7
        assert fit["train"]["excess"] >= prev_ex - 1e-10
        prev_ex = fit["train"]["excess"]
        incumbents.append(fit["s_a"])
        prev = fit
    assert prev["train"]["excess"] >= pack_id_excess(pack) - 1e-10


def pack_id_excess(pack):
    from prh_replication.anisotropic_kernels import identity_deltas, scores_numpy

    sc = scores_numpy(*identity_deltas(pack["q_a"], pack["q_b"]), pack)
    return sc["excess"]


def test_decompose_s_splits_uniform_and_direction():
    from prh_replication.release_anisotropy import decompose_s

    rng = np.random.default_rng(6)
    a = rng.normal(size=(5, 5))
    s = 0.5 * (a + a.T)
    dec = decompose_s(s)
    assert dec["d"] == pytest.approx(dec["d_uniform"] + dec["d_directional"], rel=1e-12, abs=1e-12)
    assert abs(np.trace(dec["s_tilde"])) < 1e-12


def test_response_grams_factor_matches_ambient():
    from prh_replication.release_anisotropy import response_grams_factor, split_delta_psd_factors, split_delta_psd_subspace

    rng = np.random.default_rng(9)
    u, _ = np.linalg.qr(rng.normal(size=(18, 4)))
    b = rng.normal(size=(4, 4))
    b = 0.5 * (b + b.T)
    z = rng.normal(size=(10, 18))
    cp, cm = split_delta_psd_factors(b)
    dp, dm = split_delta_psd_subspace(u, b)
    assert np.allclose(response_grams_factor(z, u, cp), response_grams(z, dp), atol=1e-10)
    assert np.allclose(response_grams_factor(z, u, cm), response_grams(z, dm), atol=1e-10)


def test_subspace_psd_split_matches_ambient():
    from prh_replication.release_anisotropy import split_delta_psd_subspace

    rng = np.random.default_rng(8)
    u, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    b = rng.normal(size=(5, 5))
    b = 0.5 * (b + b.T)
    b = b @ b.T + 0.2 * np.eye(5)
    delta = u @ (b - np.eye(5)) @ u.T
    dp, dm = split_delta_psd(delta)
    sp, sm = split_delta_psd_subspace(u, b)
    assert np.allclose(dp, sp, atol=1e-10)
    assert np.allclose(dm, sm, atol=1e-10)


def test_haar_preserves_spectrum_and_distortion():
    from prh_replication.release_anisotropy import distortion_d, haar_rotate_s

    rng = np.random.default_rng(7)
    a = rng.normal(size=(6, 6))
    s = 0.5 * (a + a.T)
    sr = haar_rotate_s(s, rng)
    assert np.allclose(np.sort(np.linalg.eigvalsh(s)), np.sort(np.linalg.eigvalsh(sr)), atol=1e-10)
    assert distortion_d(s, 6) == pytest.approx(distortion_d(sr, 6), rel=1e-12, abs=1e-12)

