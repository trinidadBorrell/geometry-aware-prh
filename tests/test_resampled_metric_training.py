"""Sampling-path checks for quarter-gallery metric updates. No cached embeddings."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prh_replication.anisotropic_kernels import make_pack, sample_center, scores_numpy
from prh_replication.metric_stability import (
    b_matrix_exp,
    centred_kernel,
    choose_val_checkpoint,
    constraint_ok,
    draw_quarter_indices,
    fit_basis,
    gradient_at_vech,
    initial_vech,
    run_a_only_stream,
    scores_direct_f64,
)


def _xy(n=32, d=8, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, 2))
    x = np.hstack([lat, 0.3 * rng.normal(size=(n, d - 2))])
    y = np.hstack([lat + 0.05 * rng.normal(size=(n, 2)), 0.3 * rng.normal(size=(n, d - 2))])
    return x, y


def _ids(n):
    return [f"coco-val2017-{i:012d}" for i in range(n)]


def test_quarter_draw_uses_training_ids_only_and_is_unique():
    ids = _ids(16)
    held_out = "coco-val2017-999999999999"
    rng = np.random.Generator(np.random.PCG64(4))
    idx = draw_quarter_indices(len(ids), rng, 0.25)
    chosen = [ids[int(i)] for i in idx]
    assert len(chosen) == 4
    assert len(set(chosen)) == len(chosen)
    assert set(chosen) <= set(ids)
    assert held_out not in chosen
    assert list(idx) == sorted(idx)


def test_sampling_is_deterministic_and_ignores_initialisation_rng():
    first = draw_quarter_indices(20, np.random.Generator(np.random.PCG64(7)))
    _ = initial_vech(8, 2, 0.05)
    torch.manual_seed(12345)
    second = draw_quarter_indices(20, np.random.Generator(np.random.PCG64(7)))
    assert np.array_equal(first, second)
    assert torch.all(initial_vech(6, 0) == 0)


def test_subset_contraction_matches_subset_centred_gram_not_a_global_slice():
    x, y = _xy(n=24, seed=2)
    ba, bb = fit_basis(x, q=3), fit_basis(y, q=3)
    idx = np.array([1, 4, 7, 9, 15, 20])
    eye_a, eye_b = np.eye(ba["q"]), np.eye(bb["q"])
    full = centred_kernel(x, ba["mu"], ba["U"], eye_a)
    sliced = full[np.ix_(idx, idx)]
    subset = centred_kernel(x[idx], ba["mu"], ba["U"], eye_a)
    assert not np.allclose(sliced, subset, atol=1e-6)
    za, zb = x[idx] - ba["mu"], y[idx] - bb["mu"]
    pack = make_pack(za, zb, ba["U"], bb["U"])
    contracted = scores_numpy(np.zeros((ba["q"], ba["q"])), np.zeros((bb["q"], bb["q"])), pack)
    direct = scores_direct_f64(za, zb, ba["U"], bb["U"], eye_a, eye_b)
    assert contracted["excess"] == pytest.approx(direct["excess"], abs=1e-8)
    s = 0.05 * np.eye(ba["q"])
    b = b_matrix_exp(s)
    contracted_s = scores_numpy(b - eye_a, np.zeros((bb["q"], bb["q"])), pack)
    direct_s = scores_direct_f64(za, zb, ba["U"], bb["U"], b, eye_b)
    assert contracted_s["a"] == pytest.approx(direct_s["a"], abs=1e-8)
    assert contracted_s["excess"] == pytest.approx(direct_s["excess"], abs=1e-8)


def test_subset_autograd_matches_float64_finite_differences():
    x, y = _xy(n=24, d=8, seed=3)
    ba, bb = fit_basis(x, q=3), fit_basis(y, q=3)
    idx = np.array([0, 2, 5, 8, 11, 14])
    vech = np.array([0.04, -0.02, 0.01, 0.0, -0.03, 0.02])
    g, _, value = gradient_at_vech(
        x[idx], y[idx], ba, bb, vech, np.zeros(6), mode="a_only", rho=0.4, accounting="equal_total"
    )
    fd = np.zeros_like(g)
    eps = 1e-6
    for i in range(vech.size):
        up, lo = vech.copy(), vech.copy()
        up[i] += eps
        lo[i] -= eps
        fd[i] = (_excess(x, y, ba, bb, up, idx) - _excess(x, y, ba, bb, lo, idx)) / (2 * eps)
    assert np.isfinite(value)
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-4)


def _excess(x, y, ba, bb, vech, idx):
    raw = torch.tensor(vech, dtype=torch.float64)
    from prh_replication.anisotropic_kernels import _unvech
    from prh_replication.metric_stability import project_matrices

    s_a, s_b = project_matrices(
        _unvech(raw, ba["q"]),
        torch.zeros(bb["q"], bb["q"], dtype=torch.float64),
        rho=0.4,
        accounting="equal_total",
        mode="a_only",
    )
    from prh_replication.metric_stability import evaluate_metric

    return evaluate_metric(x[idx], y[idx], ba, bb, s_a.numpy(), s_b.numpy())["excess"]


def test_complete_pool_matches_full_gallery_regardless_of_order():
    x, y = _xy(n=16, seed=5)
    ba, bb = fit_basis(x, q=3), fit_basis(y, q=3)
    vech = np.zeros(6)
    g1, _, v1 = gradient_at_vech(x, y, ba, bb, vech, np.zeros(6), mode="a_only", rho=0.4)
    perm = np.random.default_rng(9).permutation(len(x))
    g2, _, v2 = gradient_at_vech(x[perm], y[perm], ba, bb, vech, np.zeros(6), mode="a_only", rho=0.4)
    assert v1 == pytest.approx(v2, abs=1e-10)
    assert np.allclose(g1, g2, atol=1e-8)
    ids = _ids(len(x))
    full = run_a_only_stream(
        x, y, ids, ba, bb, rho=0.4, init_seed=1, n_updates=1, procedure="full", val_features_a=x[:8], val_features_b=y[:8]
    )
    quarter = run_a_only_stream(
        x,
        y,
        ids,
        ba,
        bb,
        rho=0.4,
        init_seed=1,
        n_updates=1,
        procedure="quarter",
        sampling_seed=0,
        quarter_fraction=1.0,
        val_features_a=x[:8],
        val_features_b=y[:8],
    )
    assert np.allclose(full["s_at_final"], quarter["s_at_final"], atol=1e-8)
    assert full["curves"][1]["full_train_excess"] == pytest.approx(quarter["curves"][1]["full_train_excess"], abs=1e-8)


def test_projected_updates_stay_feasible_and_selection_uses_validation_only():
    rows = [
        {"step": 4, "val_excess": 0.5, "d_total": 0.2, "test_excess": 1.0},
        {"step": 5, "val_excess": 0.5, "d_total": 0.4, "test_excess": 0.01},
        {"step": 0, "val_excess": 0.2, "d_total": 0.0, "test_excess": 0.99},
        {"step": 3, "val_excess": 0.5, "d_total": 0.2, "test_excess": 0.0},
    ]
    chosen = choose_val_checkpoint(rows)
    assert chosen["step"] == 3
    x, y = _xy(n=16, seed=6)
    ba, bb = fit_basis(x, q=3), fit_basis(y, q=3)
    ids = _ids(len(x))
    rec = run_a_only_stream(
        x,
        y,
        ids,
        ba,
        bb,
        rho=0.1,
        init_seed=1,
        n_updates=3,
        procedure="quarter",
        sampling_seed=2,
        val_features_a=x[:8],
        val_features_b=y[:8],
        record_indices=True,
    )
    assert rec["feasible"]
    assert constraint_ok(rec["s_at_final"], np.zeros_like(rec["s_at_final"]), 0.1, "equal_total", "a_only")
    assert rec["selected_final"] == choose_val_checkpoint(
        [{"step": row["step"], "val_excess": row["full_val_excess"], "d_total": row["d_total"]} for row in rec["curves"]]
    )
    seen = []
    for idx in rec["subset_indices"]:
        assert len(idx) == len(set(idx)) == 4
        seen.extend(ids[i] for i in idx)
    assert set(seen) <= set(ids)


def test_state_filenames_keep_budget_and_seed_distinct():
    from pathlib import Path

    from prh_replication.resampled_metric_training import _state_json

    stems = [
        "runs/qwen2-7b__quarter__rho0.1__init0__samp0",
        "runs/qwen2-7b__quarter__rho0.1__init0__samp1",
        "runs/qwen2-7b__quarter__rho0.4__init0__samp0",
        "runs/qwen2-7b__full__rho0.1__init1",
    ]
    paths = [_state_json(Path(stem)) for stem in stems]
    assert len(set(paths)) == len(paths)
    assert paths[0].name == "qwen2-7b__quarter__rho0.1__init0__samp0.state.json"


def test_resumption_restores_optimiser_and_sampling_state():
    x, y = _xy(n=16, seed=8)
    ba, bb = fit_basis(x, q=3), fit_basis(y, q=3)
    ids = _ids(len(x))
    kwargs = dict(
        rho=0.4,
        init_seed=2,
        procedure="quarter",
        sampling_seed=5,
        val_features_a=x[:8],
        val_features_b=y[:8],
        record_indices=True,
    )
    fresh = run_a_only_stream(x, y, ids, ba, bb, n_updates=4, **kwargs)
    partial = run_a_only_stream(x, y, ids, ba, bb, n_updates=2, **kwargs)
    resumed = run_a_only_stream(x, y, ids, ba, bb, n_updates=4, resume=partial["resume_state"], **kwargs)
    assert np.allclose(resumed["s_at_final"], fresh["s_at_final"], atol=1e-10)
    assert resumed["resume_state"]["rng"] == fresh["resume_state"]["rng"]
    assert np.allclose(resumed["resume_state"]["adam"]["exp_avg"], fresh["resume_state"]["adam"]["exp_avg"], atol=1e-10)
    assert [row["full_val_excess"] for row in resumed["curves"]] == pytest.approx(
        [row["full_val_excess"] for row in fresh["curves"]]
    )
    assert resumed["subset_indices"] == fresh["subset_indices"]
