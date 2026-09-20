"""Polynomial / Gegenbauer / spectral-shell kernel checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prh_replication.flex_kernels import (
    NU_GRID,
    P_MAX,
    build_basis_grams,
    cka_from_stats,
    center_stack,
    contract_pair,
    fourier_profile,
    gegenbauer_normalised,
    gegenbauer_scalar,
    mix_grams,
    monomial_grams,
    project_simplex,
)
from prh_replication.kernels import center_gram_o2, extension_stats, linear_gram


def test_degree_one_is_linear():
    torch.manual_seed(0)
    x = F.normalize(torch.randn(20, 8), dim=-1)
    t = x @ x.T
    assert torch.allclose(monomial_grams(t, 3)[0], t, atol=1e-6)
    z = gegenbauer_normalised(t, d=8, p_max=3)
    assert torch.allclose(z[0], t, atol=1e-6)


def test_gegenbauer_at_one():
    for d in (4, 384, 1024):
        for p in range(1, 8):
            assert gegenbauer_scalar(1.0, d, p) == pytest.approx(1.0, abs=1e-5)


def test_gegenbauer_recurrence_independent():
    # Independent check: C_2^{(α)}(t)=2α(α+1)t^2/2 - α wait normalised Z2.
    # For d=4, α=1, Z2(t) = (3 t^2 - 1)/2? Legendre P2.
    # Recurrence p=2,d=4: a=(4+4-4)/(2+4-3)=4/3, b=1/3, Z2= (4/3)t^2 - 1/3
    t = 0.3
    z2 = gegenbauer_scalar(t, 4, 2)
    assert z2 == pytest.approx((4.0 / 3.0) * t * t - 1.0 / 3.0, abs=1e-6)


def test_unit_diagonal_and_symmetry():
    torch.manual_seed(1)
    x = F.normalize(torch.randn(12, 6), dim=-1)
    t = x @ x.T
    dsq = torch.cdist(x, x).pow(2)
    for fam in ("monomial", "spherical", "lin_fourier", "sph_fourier"):
        grams, specs, _ = build_basis_grams(t, dsq, s=1.0, d=6, family=fam, p_max=4, nu_grid=NU_GRID[:3])
        for g, sp in zip(grams, specs):
            assert torch.allclose(g, g.T, atol=1e-5), sp
            assert torch.allclose(g.diag(), torch.ones(12), atol=1e-5), sp


def test_small_psd_tolerance():
    torch.manual_seed(2)
    x = F.normalize(torch.randn(16, 10), dim=-1)
    t = x @ x.T
    dsq = torch.cdist(x, x).pow(2)
    grams, _, _ = build_basis_grams(t, dsq, 1.0, 10, "sph_fourier", p_max=4, nu_grid=NU_GRID[:3])
    for g in grams:
        evals = torch.linalg.eigvalsh(g)
        assert float(evals.min()) > -1e-5


def test_contracted_matches_direct():
    torch.manual_seed(3)
    x = F.normalize(torch.randn(18, 7), dim=-1)
    y = F.normalize(torch.randn(18, 11), dim=-1)
    ta, tb = x @ x.T, y @ y.T
    da, db = torch.cdist(x, x).pow(2), torch.cdist(y, y).pow(2)
    ga, sa, _ = build_basis_grams(ta, da, 1.0, 7, "monomial", p_max=4)
    gb, _, _ = build_basis_grams(tb, db, 1.0, 11, "monomial", p_max=4)
    stats = contract_pair(center_stack(ga), center_stack(gb))
    c = np.array([0.5, 0.2, 0.2, 0.1])
    st = cka_from_stats(c, stats)
    k = mix_grams(ga, c)
    l = mix_grams(gb, c)
    direct = extension_stats(k, l)
    assert st["a"] == pytest.approx(direct["a"], rel=1e-4, abs=1e-5)
    assert st["ratio"] == pytest.approx(direct["ratio"], rel=1e-4, abs=1e-5)


def test_fourier_kappa0_and_gaussian_limit():
    r0 = torch.zeros(4)
    k, _ = fourier_profile(r0, 1.2, d=384)
    assert torch.allclose(k, torch.ones(4), atol=1e-5)
    r = torch.linspace(0, 0.3, 20)
    k, frac = fourier_profile(r, 0.5, d=2048)
    g = torch.exp(-0.5 * 0.25 * r * r)
    assert float((k - g).abs().max()) < 0.02


def test_simplex_projection():
    w = project_simplex(np.array([1.5, -0.2, 0.1]))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= -1e-12)


def test_grids_frozen():
    assert P_MAX == 12
    assert len(NU_GRID) == 12
    assert NU_GRID[0] == 0.1 and NU_GRID[-1] == 8.0
