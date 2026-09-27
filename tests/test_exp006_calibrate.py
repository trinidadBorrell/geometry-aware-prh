"""exp-006's feature-space CKA null must match Aristotelian's Gram-based calibration."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aristotelian.experiments.layerwise_engine import (
    build_gram_cache,
    compute_alignment_gated_cka_cached,
)

_spec = importlib.util.spec_from_file_location(
    "exp006_calibrate",
    Path(__file__).resolve().parents[1] / "experiments/exp-006-sample-size/calibrate.py",
)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def _layers(n, layers, d, shared, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        F.normalize(
            shared @ torch.randn(shared.shape[1], d, generator=g)
            + torch.randn(n, d, generator=g)
            + 1.0,
            dim=-1,
        )
        for _ in range(layers)
    ]


def test_feature_space_cka_null_matches_aristotelian():
    n = 150
    shared = torch.randn(n, 5, generator=torch.Generator().manual_seed(0))
    xs, ys = _layers(n, 3, 24, shared, 1), _layers(n, 2, 40, shared, 2)
    ours = cal.cka_calibrated(xs, ys, permutations=40, alpha=0.05, seed=0)
    ref = compute_alignment_gated_cka_cached(
        build_gram_cache(xs, normalize=False),
        build_gram_cache(ys, normalize=False),
        num_permutations=40,
        alpha=0.05,
        seed=0,
    )
    for key in ("raw_score", "tau_alpha", "g_score", "mu0", "sd0", "p_value"):
        assert ours[key] == pytest.approx(ref[key], rel=1e-4, abs=1e-6), key
    assert tuple(ours["best_indices"]) == tuple(ref["best_indices"])


def test_raw_only_when_no_permutations():
    n = 80
    shared = torch.randn(n, 5, generator=torch.Generator().manual_seed(0))
    xs, ys = _layers(n, 2, 16, shared, 1), _layers(n, 2, 16, shared, 2)
    res = cal.cka_calibrated(xs, ys, permutations=0, alpha=0.05, seed=0)
    assert set(res) == {"raw_score", "best_indices"}
