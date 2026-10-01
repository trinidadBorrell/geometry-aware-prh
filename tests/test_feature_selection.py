"""Original-coordinate top-20% selection and ΔM reconstruction."""

from __future__ import annotations

import numpy as np

from prh_replication.feature_selection import (
    permute_freq_then_index,
    select_top_k,
    selection_k,
    tie_seed,
    weight_tolerance,
)
from prh_replication.release_anisotropy import ambient_metric_diag, signed_correction


def test_flat_diagonal_is_uninformative():
    w = np.ones(20)
    rng = np.random.default_rng(0)
    rec = select_top_k(w, selection_k(20), rng=rng)
    assert rec["uninformative"] is True
    assert rec["selected"].sum() == 0
    assert rec["n_tie_pool"] == 20


def test_unique_weights_take_top_k_without_index_bias_on_values():
    d = 10
    k = selection_k(d)
    w = np.linspace(0.0, 1.0, d)
    rec = select_top_k(w, k, rng=np.random.default_rng(1))
    assert rec["uninformative"] is False
    assert rec["tie_break"] is False
    assert rec["selected"].sum() == k
    assert set(np.where(rec["selected"])[0]) == set(range(d - k, d))


def test_boundary_ties_use_seed_not_index_and_are_reproducible():
    w = np.array([5.0, 3.0, 3.0, 3.0, 1.0, 0.0])
    k = 3
    a = select_top_k(w, k, rng=np.random.default_rng(tie_seed("m", "p", 0.1)))
    b = select_top_k(w, k, rng=np.random.default_rng(tie_seed("m", "p", 0.1)))
    c = select_top_k(w, k, rng=np.random.default_rng(tie_seed("m", "q", 0.1)))
    assert a["uninformative"] is False
    assert a["tie_break"] is True
    assert a["selected"].sum() == k
    assert a["selected"][0]  # unique top
    assert not a["selected"][4]
    assert np.array_equal(a["selected"], b["selected"])
    # different partner seed may differ
    assert a["selected"].sum() == c["selected"].sum()


def test_freq_order_breaks_ties_by_original_index():
    freq = np.array([0.2, 0.5, 0.5, 0.1])
    perm = permute_freq_then_index(freq)
    assert list(perm) == [1, 2, 0, 3]


def test_mean_delta_matches_average_of_individuals():
    rng = np.random.default_rng(4)
    u, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    bs = []
    for _ in range(4):
        a = rng.normal(size=(3, 3))
        bs.append(a @ a.T + np.eye(3))
    from prh_replication.feature_selection import _mean_delta

    mean = _mean_delta(u, bs)
    acc = np.mean([signed_correction(u, b) for b in bs], axis=0)
    assert np.allclose(mean, acc, atol=1e-12)


def test_diag_w_matches_full_m_without_building_when_checked():
    rng = np.random.default_rng(5)
    u, _ = np.linalg.qr(rng.normal(size=(9, 3)))
    a = rng.normal(size=(3, 3))
    b = a @ a.T + 0.3 * np.eye(3)
    w = ambient_metric_diag(u, b)
    m = np.eye(9) + signed_correction(u, b)
    assert np.allclose(w, np.diag(m), atol=1e-12)
    assert weight_tolerance(w) >= 1e-10
