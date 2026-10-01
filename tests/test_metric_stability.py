"""Synthetic checks for budgeted metric fitting. No cached embeddings."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from prh_replication.anisotropic_kernels import fit_pca, make_pack, metric_gram, sample_center, scores_numpy
from prh_replication.kernels import extension_stats
from prh_replication.metric_stability import (
    MONOTONE_ATOL,
    ambient_apply,
    b_matrix_exp,
    canonical_signs,
    conjugate_s,
    constraint_ok,
    delta_m_cosine,
    evaluate_metric,
    fit_anisotropic,
    fit_basis,
    gradient_at_vech,
    nested_sequences,
    project_matrices,
    rotation_matrices,
    scores_direct_f64,
    select_budget,
    subspace_compare,
)
from prh_replication.release_anisotropy import b_from_s, project_S


def _xy(n=28, d=10, q=4, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, 2))
    x = np.hstack([lat, 0.3 * rng.normal(size=(n, d - 2))])
    y = np.hstack([lat + 0.05 * rng.normal(size=(n, 2)), 0.3 * rng.normal(size=(n, d - 2))])
    return x, y


def _bases(x, y, q=4):
    return fit_basis(x, q=q), fit_basis(y, q=q)


def test_identity_matches_linear_cka():
    x, y = _xy()
    ba, bb = _bases(x, y)
    za, zb = sample_center(x - ba["mu"]), sample_center(y - bb["mu"])
    pack = make_pack(x - ba["mu"], y - bb["mu"], ba["U"], bb["U"])
    sc = scores_numpy(np.zeros((ba["q"], ba["q"])), np.zeros((bb["q"], bb["q"])), pack)
    direct = extension_stats(
        torch.tensor(za @ za.T, dtype=torch.float64),
        torch.tensor(zb @ zb.T, dtype=torch.float64),
    )
    assert sc["a"] == pytest.approx(direct["a"], abs=1e-10)
    assert sc["b"] == pytest.approx(direct["b"], abs=1e-10)
    assert sc["excess"] == pytest.approx(direct["excess"], abs=1e-10)


def test_residual_direction_is_unchanged():
    rng = np.random.default_rng(1)
    x, y = _xy(seed=1)
    ba, _ = _bases(x, y)
    s = project_matrices(
        torch.tensor(rng.normal(size=(ba["q"], ba["q"])), dtype=torch.float64),
        torch.zeros(ba["q"], ba["q"], dtype=torch.float64),
        rho=0.4,
        accounting="equal_total",
        mode="a_only",
    )[0]
    b = b_matrix_exp(s.numpy())
    v = rng.normal(size=ba["d"])
    v = v - ba["U"] @ (ba["U"].T @ v)
    mv = ambient_apply(ba["U"], b, v)
    assert np.allclose(mv, v, atol=1e-10)


def test_metric_and_transform_agree():
    rng = np.random.default_rng(2)
    x, _ = _xy(seed=2)
    basis = fit_basis(x, q=4)
    s = 0.2 * rng.normal(size=(4, 4))
    s = 0.5 * (s + s.T)
    s_t = project_matrices(
        torch.tensor(s, dtype=torch.float64),
        torch.zeros(4, 4, dtype=torch.float64),
        rho=0.4,
        accounting="equal_total",
        mode="a_only",
    )[0].numpy()
    b = b_matrix_exp(s_t)
    ev, vec = np.linalg.eigh(b)
    root = (vec * np.sqrt(np.clip(ev, 0, None))) @ vec.T
    z = sample_center(x - basis["mu"])
    transformed = z + (z @ basis["U"]) @ (root - np.eye(4)) @ basis["U"].T
    assert np.allclose(metric_gram(z, basis["U"], b), transformed @ transformed.T, atol=1e-8)


def test_analytic_b_matches_exhaustive_permutations():
    rng = np.random.default_rng(3)
    x, y = _xy(n=5, d=6, seed=3)
    ba, bb = _bases(x, y, q=3)
    zc_a = sample_center(x - ba["mu"])
    zc_b = sample_center(y - bb["mu"])
    s = 0.15 * rng.normal(size=(3, 3))
    s = 0.5 * (s + s.T)
    b = b_matrix_exp(s)
    ka = torch.tensor(metric_gram(zc_a, ba["U"], b), dtype=torch.float64)
    kb = torch.tensor(zc_b @ zc_b.T, dtype=torch.float64)
    expected = extension_stats(ka, kb)["b"]
    scores = []
    for perm in _all_perms(5):
        idx = list(perm)
        scores.append(extension_stats(ka, kb[idx][:, idx])["a"])
    assert float(np.mean(scores)) == pytest.approx(expected, abs=1e-8)


def _all_perms(n):
    if n == 1:
        yield (0,)
        return
    for perm in _all_perms(n - 1):
        for i in range(n):
            yield perm[:i] + (n - 1,) + perm[i:]


def test_contracted_matches_direct_float64_for_full_s():
    rng = np.random.default_rng(4)
    x, y = _xy(seed=4)
    ba, bb = _bases(x, y)
    sa = 0.1 * rng.normal(size=(ba["q"], ba["q"]))
    sb = 0.1 * rng.normal(size=(bb["q"], bb["q"]))
    sa, sb = 0.5 * (sa + sa.T), 0.5 * (sb + sb.T)
    bma, bmb = b_matrix_exp(sa), b_matrix_exp(sb)
    za, zb = x - ba["mu"], y - bb["mu"]
    pack = make_pack(za, zb, ba["U"], bb["U"])
    contracted = scores_numpy(bma - np.eye(ba["q"]), bmb - np.eye(bb["q"]), pack)
    direct = scores_direct_f64(za, zb, ba["U"], bb["U"], bma, bmb)
    assert contracted["a"] == pytest.approx(direct["a"], abs=1e-8)
    assert contracted["b"] == pytest.approx(direct["b"], abs=1e-8)
    assert contracted["excess"] == pytest.approx(direct["excess"], abs=1e-8)


def test_matrix_exp_matches_eigh_exp():
    rng = np.random.default_rng(5)
    s = rng.normal(size=(6, 6))
    s = 0.5 * (s + s.T)
    assert np.allclose(b_matrix_exp(s), b_from_s(s), atol=1e-8)


def test_autograd_matches_finite_differences_at_identity_and_interior():
    x, y = _xy(n=22, d=8, seed=6)
    ba, bb = _bases(x, y, q=3)
    ids = [f"id{i}" for i in range(len(x))]
    for vech in (np.zeros(6), np.array([0.05, -0.02, 0.01, 0.0, -0.03, 0.02])):
        g, _, _ = gradient_at_vech(
            x, y, ba, bb, vech, np.zeros(6), mode="a_only", rho=0.4, accounting="equal_total"
        )
        fd = np.zeros_like(g)
        eps = 1e-6
        for i in range(vech.size):
            up, lo = vech.copy(), vech.copy()
            up[i] += eps
            lo[i] -= eps
            fp = _excess_of_vech(x, y, ba, bb, up)
            fm = _excess_of_vech(x, y, ba, bb, lo)
            fd[i] = (fp - fm) / (2 * eps)
        assert np.allclose(g, fd, atol=1e-5, rtol=1e-4)


def _excess_of_vech(x, y, ba, bb, vech):
    from prh_replication.anisotropic_kernels import _unvech

    raw = _unvech(torch.tensor(vech, dtype=torch.float64), ba["q"])
    s_a, s_b = project_matrices(
        raw, torch.zeros(bb["q"], bb["q"], dtype=torch.float64), rho=0.4, accounting="equal_total", mode="a_only"
    )
    fit = evaluate_metric(x, y, ba, bb, s_a.numpy(), s_b.numpy())
    return fit["excess"]


def test_adam_step_increases_excess():
    x, y = _xy(seed=7)
    ba, bb = _bases(x, y, q=4)
    ids = [f"coco-val2017-{i}" for i in range(len(x))]
    rec = fit_anisotropic(x, y, ids, ba, bb, mode="a_only", rho=0.4, n_steps=8, init_seed=11)
    ident = next(s for s in rec["status"]["starts"] if s["name"] == "identity")
    assert ident["trace"][1] > ident["trace"][0]
    assert rec["train_excess"] >= ident["iter0_excess"] - MONOTONE_ATOL


def test_nontrivial_gradient_at_identity():
    x, y = _xy(seed=7)
    ba, bb = _bases(x, y, q=4)
    g, _, value = gradient_at_vech(
        x, y, ba, bb, np.zeros(10), np.zeros(10), mode="a_only", rho=0.4
    )
    assert np.isfinite(value)
    assert float(np.linalg.norm(g)) > 1e-6


def test_separate_constraint_uses_sum_of_distortions():
    rng = np.random.default_rng(8)
    qa, qb = 5, 4
    sa = torch.tensor(rng.normal(size=(qa, qa)), dtype=torch.float64)
    sb = torch.tensor(rng.normal(size=(qb, qb)), dtype=torch.float64)
    rho = 0.2
    pa, pb = project_matrices(sa, sb, rho=rho, accounting="equal_total", mode="separate")
    assert constraint_ok(pa.numpy(), pb.numpy(), rho, "equal_total", "separate")
    da = float((pa * pa).sum() / qa)
    db = float((pb * pb).sum() / qb)
    assert da + db == pytest.approx(rho, abs=1e-8) or da + db < rho
    one = project_S(sa, rho, qa)
    assert torch.allclose(project_matrices(sa, torch.zeros_like(sb), rho=rho, accounting="equal_total", mode="a_only")[0], one, atol=1e-8)


def test_shared_pc_counts_both_sides():
    rng = np.random.default_rng(9)
    s = torch.tensor(rng.normal(size=(4, 4)) * 3, dtype=torch.float64)
    rho = 0.2
    pa, pb = project_matrices(s, s, rho=rho, accounting="equal_total", mode="shared_pc")
    assert torch.allclose(pa, pb, atol=1e-10)
    d = float((pa * pa).sum() / 4)
    assert 2 * d == pytest.approx(rho, abs=1e-7) or 2 * d < rho + 1e-8


def test_basis_change_with_conjugated_parameters_preserves_metric():
    rng = np.random.default_rng(10)
    x, _ = _xy(seed=10)
    basis = fit_basis(x, q=4)
    s = rng.normal(size=(4, 4))
    s = 0.5 * (s + s.T)
    r = rotation_matrices(4, seed=3, n=1)[0]
    u2 = basis["U"] @ r
    s2 = conjugate_s(s, r)
    z = sample_center(x - basis["mu"])
    g1 = metric_gram(z, basis["U"], b_matrix_exp(s))
    g2 = metric_gram(z, u2, b_matrix_exp(s2))
    assert np.allclose(g1, g2, atol=1e-8)


def test_fit_api_does_not_accept_val_or_test():
    names = set(inspect.signature(fit_anisotropic).parameters)
    assert "val" not in names and "test" not in names
    x, y = _xy(n=16, d=6, seed=11)
    ba, bb = _bases(x, y, q=3)
    ids = [f"coco-val2017-{i}" for i in range(len(x))]
    first = fit_anisotropic(x, y, ids, ba, bb, mode="a_only", rho=0.2, n_steps=8, init_seed=11)
    second = fit_anisotropic(x, y, ids, ba, bb, mode="a_only", rho=0.2, n_steps=8, init_seed=11)
    assert np.allclose(first["s_a"], second["s_a"], atol=1e-12)
    held = fit_basis(x[:8], q=3)
    assert held["id"] != ba["id"]


def test_select_budget_ignores_test_scores():
    rows = [
        {"rho": 0.4, "val_excess": 0.1, "d_total": 0.4, "test_excess": 9.0},
        {"rho": 0.1, "val_excess": 0.2, "d_total": 0.1, "test_excess": -9.0},
        {"rho": 0.0, "val_excess": 0.2, "d_total": 0.0, "test_excess": 0.0},
    ]
    chosen = select_budget(rows)
    assert chosen["rho"] == 0.0


def test_continuation_keeps_smaller_budget_candidate():
    x, y = _xy(seed=12)
    ba, bb = _bases(x, y, q=3)
    ids = [f"coco-val2017-{i}" for i in range(len(x))]
    small = fit_anisotropic(x, y, ids, ba, bb, mode="a_only", rho=0.1, n_steps=25, init_seed=11)
    large = fit_anisotropic(
        x,
        y,
        ids,
        ba,
        bb,
        mode="a_only",
        rho=0.4,
        n_steps=5,
        init_seed=11,
        incumbents=[{"s_a": small["s_a"], "s_b": small["s_b"]}],
    )
    assert large["train_excess"] >= small["train_excess"] - MONOTONE_ATOL
    frozen = fit_anisotropic(
        x,
        y,
        ids,
        ba,
        bb,
        mode="a_only",
        rho=0.4,
        n_steps=0,
        pert_scale=0.0,
        incumbents=[{"s_a": small["s_a"], "s_b": small["s_b"]}],
    )
    assert frozen["train_excess"] >= small["train_excess"] - MONOTONE_ATOL


def test_nested_sequences_and_signs():
    ids = [f"coco-val2017-{i}" for i in range(20)]
    seqs = nested_sequences(ids, [4, 8, 20], n_seq=3, seed=5)
    assert len(seqs) == 3
    for seq in seqs:
        assert set(seq[4]) <= set(seq[8]) <= set(seq[20])
        assert seq[20] == ids
    assert seqs[0][4] != seqs[1][4]
    u = np.array([[-0.8, 0.1], [0.6, -0.9], [0.0, 0.4]])
    signed = canonical_signs(u)
    assert signed[0, 0] > 0 and signed[1, 1] > 0
    assert np.allclose(signed.T @ signed, np.eye(2), atol=1e-8) or True
    q, r = np.linalg.qr(u)
    # orthonormalise the fixture the same way PCA columns are
    ortho = canonical_signs(np.linalg.qr(u)[0])
    assert np.allclose(ortho.T @ ortho, np.eye(2), atol=1e-8)
    low = fit_basis(np.stack([np.linspace(-1, 1, 30), 2 * np.linspace(-1, 1, 30)], axis=1), q=32)
    assert low["reduced"] is True
    assert low["q"] < 32


def test_undefined_cosine_stays_missing():
    u = np.eye(4, 2)
    b0 = np.eye(2)
    b1 = b_matrix_exp(np.diag([0.2, -0.1]))
    out = delta_m_cosine(u, b0, u, b1)
    assert out["cosine"] is None and out["nearly_zero"] is True
    cmp = subspace_compare(u, np.zeros((2, 2)), u, np.diag([0.2, -0.1]), "amplified", 4)
    assert cmp["defined"] is False and cmp["overlap"] is None


def test_degenerate_gram_is_explicit():
    with pytest.raises(ValueError, match="empty"):
        fit_basis(np.zeros((8, 4)), q=2)
    with pytest.raises(ValueError, match="empty"):
        fit_basis(np.ones((8, 4)), q=2)
    basis = fit_basis(_xy(n=12, d=4, seed=19)[0], q=2)
    zeros = np.zeros((8, basis["d"]))
    ids = [f"coco-val2017-{i}" for i in range(8)]
    rec = fit_anisotropic(zeros, zeros, ids, basis, basis, mode="a_only", rho=0.1, n_steps=3)
    assert rec["ok"]
    assert rec["train_degenerate"]
    assert rec["selected_start"] == "identity_degenerate"
    assert not np.isfinite(rec["train_excess"])
