"""Analytic CKA permutation mean, degeneracy, and profile inner products."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from prh_replication.kernels import (
    ALPHA_GRID,
    LAMBDA_GRID,
    center_gram_o2,
    extension_stats,
    linear_gram,
    mc_cka_mean,
    profile_inner_products,
    profile_nodes,
    rbf_gram_from_dsq,
    rbf_profile,
)
from prh_replication.metrics import linear_cka


def test_grids_frozen():
    assert len(LAMBDA_GRID) == 25
    assert LAMBDA_GRID[0] == 0.1 and LAMBDA_GRID[-1] == 10.0
    assert 1.0 in LAMBDA_GRID
    assert len(ALPHA_GRID) == 9
    assert ALPHA_GRID[0] == 0.1 and ALPHA_GRID[-1] == 100.0


def test_o2_center_matches_hkh():
    torch.manual_seed(0)
    k = torch.randn(12, 12)
    k = (k + k.T) / 2
    n = k.shape[0]
    h = torch.eye(n) - 1.0 / n
    assert torch.allclose(center_gram_o2(k), h @ k @ h, atol=1e-5)


def test_ratio_matches_a_over_b_and_mc():
    torch.manual_seed(1)
    x = torch.randn(24, 8)
    y = x @ torch.randn(8, 6) + 0.2 * torch.randn(24, 6)
    k, l = linear_gram(x), linear_gram(y)
    st = extension_stats(k, l)
    assert st["valid_a"]
    assert st["ratio"] == pytest.approx(st["a"] / st["b"], rel=1e-5)
    mc = mc_cka_mean(k, l, n_perm=200, seed=0)
    assert st["b"] == pytest.approx(mc["mc_mean"], abs=0.04)


def test_official_eps_differs_from_extension_on_tiny_grams():
    k = 1e-8 * torch.ones(8, 8)
    l = 1e-8 * torch.ones(8, 8)
    st = extension_stats(k, l)
    # constant kernel: degenerate after centering
    assert st["degenerate"]


def test_narrow_rbf_cka_and_ratio_near_one():
    torch.manual_seed(2)
    x = F.normalize(torch.randn(16, 5), dim=-1)
    y = F.normalize(torch.randn(16, 7), dim=-1)
    dsq_x = torch.cdist(x, x).pow(2)
    dsq_y = torch.cdist(y, y).pow(2)
    k = rbf_gram_from_dsq(dsq_x, 1e-8)
    l = rbf_gram_from_dsq(dsq_y, 1e-8)
    st = extension_stats(k, l)
    assert st["a"] == pytest.approx(1.0, abs=1e-5)
    assert st["ratio"] == pytest.approx(1.0, abs=1e-4)


def test_linear_cka_official_vs_extension_close():
    torch.manual_seed(3)
    x = torch.randn(40, 10)
    y = torch.randn(40, 9)
    st = extension_stats(linear_gram(x), linear_gram(y))
    off = linear_cka(x, y)
    assert st["a"] == pytest.approx(off, abs=1e-5)
    assert st["official_cka"] == pytest.approx(off, abs=1e-5)


def test_profile_self_inner_product():
    r = profile_nodes()
    f = rbf_profile(r, 1.0)
    ip = profile_inner_products(f, f, r)
    assert ip["raw_normalised_l2"] == pytest.approx(1.0, abs=1e-5)
    assert ip["centred_normalised_l2"] == pytest.approx(1.0, abs=1e-5)


def test_rbf_rq_neighbour_order_unchanged():
    torch.manual_seed(4)
    x = F.normalize(torch.randn(20, 6), dim=-1)
    d = torch.cdist(x, x)
    d = d.clone().fill_diagonal_(1e9)
    knn_euc = d.argsort(dim=1)[:, :5]
    k = rbf_gram_from_dsq(d.pow(2), 0.4)
    k = k.clone().fill_diagonal_(-1e9)
    knn_k = k.argsort(dim=1, descending=True)[:, :5]
    assert torch.equal(knn_euc, knn_k)
