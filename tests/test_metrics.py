"""Focused mathematical verification of metrics and controls."""

from __future__ import annotations

import math

import pytest
import torch

from prh_replication.controls import control_battery, orthogonal_transform
from prh_replication.metrics import (
    center_gram,
    linear_cka,
    linear_gram,
    mutual_knn_score,
    permutation_cka_mean_formula,
    prepare_original,
    rbf_cka,
)


def test_self_mknn_is_one():
    x = torch.randn(32, 16)
    x = torch.nn.functional.normalize(x, dim=-1)
    assert mutual_knn_score(x, x, 10) == pytest.approx(1.0)


def test_linear_cka_invariances():
    x = torch.randn(40, 12)
    y = x @ (torch.randn(12, 8) * 0.3 + torch.eye(12, 8)[:12])
    # orthogonal + scale on x
    q = orthogonal_transform(x, seed=3)
    assert linear_cka(x, y) == pytest.approx(linear_cka(q, y), abs=1e-4)
    assert linear_cka(x, y) == pytest.approx(linear_cka(x * 5.5, y), abs=1e-4)


def test_joint_permutation_invariance():
    x = torch.randn(24, 10)
    y = torch.randn(24, 7)
    perm = torch.randperm(24)
    assert mutual_knn_score(x, y, 5) == pytest.approx(mutual_knn_score(x[perm], y[perm], 5), abs=1e-6)
    assert linear_cka(x, y) == pytest.approx(linear_cka(x[perm], y[perm]), abs=1e-5)


def test_one_sided_perm_near_chance_mknn():
    torch.manual_seed(0)
    x = torch.randn(80, 20)
    y = x[:, :15] + 0.05 * torch.randn(80, 15)
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    actual = mutual_knn_score(x, y, 10)
    perm = torch.randperm(80)
    shuffled = mutual_knn_score(x, y[perm], 10)
    assert actual > shuffled
    assert shuffled < 0.25


def test_random_features_low_cka():
    torch.manual_seed(1)
    a = torch.randn(256, 32)
    b = torch.randn(256, 32)
    assert linear_cka(a, b) < 0.2
    assert linear_cka(a, a) == pytest.approx(1.0, abs=1e-5)


def test_narrow_rbf_near_one():
    x = torch.randn(20, 8)
    y = torch.randn(20, 9)
    assert rbf_cka(x, y, 1e-8, 1e-8) == pytest.approx(1.0, abs=1e-5)


def test_permutation_mean_formula_vs_monte_carlo():
    torch.manual_seed(2)
    x = torch.randn(16, 6)
    y = torch.randn(16, 5)
    kc = center_gram(linear_gram(x))
    lc = center_gram(linear_gram(y))
    formula = permutation_cka_mean_formula(kc, lc)
    scores = []
    for i in range(400):
        perm = torch.randperm(16)
        scores.append(linear_cka(x, y[perm]))
    mc = float(torch.tensor(scores).mean())
    assert formula == pytest.approx(mc, abs=0.04)


def test_control_battery_self():
    x = torch.randn(30, 11)
    y = torch.randn(30, 9)
    out = control_battery(x, y, topk=5)
    assert out["self_mknn"] == pytest.approx(1.0)
    assert out["self_linear_cka"] == pytest.approx(1.0, abs=1e-5)
    assert out["ortho_linear_cka"] == pytest.approx(1.0, abs=1e-4)
    assert out["rescale_linear_cka"] == pytest.approx(1.0, abs=1e-4)
    assert out["narrow_rbf_self"] == pytest.approx(1.0, abs=1e-4)


def test_original_prep_clips():
    x = torch.zeros(4, 10)
    x[0, 0] = 1000
    y = prepare_original(x, q=0.95, normalize=False)
    assert y.abs().max() < 1000
