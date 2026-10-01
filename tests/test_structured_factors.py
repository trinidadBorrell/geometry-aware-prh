"""Checks for the nonorthogonal factor metric. No cached embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from prh_replication.anisotropic_kernels import metric_gram, scores_numpy
from prh_replication.metric_stability import fit_basis, scores_direct_f64
from prh_replication.structured_factors import (
    build_jobs,
    complexity_total,
    excess_on_pack,
    factor_objective,
    factors_feasible,
    family_block,
    generalized_eigenvalues,
    induced_b,
    penalised_selection,
    project_factors,
    r_off,
    run_factor_stream,
    sorted_log_spectrum,
    spectrum_distance,
    sym_expm,
    total_distortion,
)


def _xy(n=24, d=8, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, 2))
    x = np.hstack([lat, 0.3 * rng.normal(size=(n, d - 2))])
    y = np.hstack([lat + 0.05 * rng.normal(size=(n, 2)), 0.3 * rng.normal(size=(n, d - 2))])
    return x, y


def _basis_pair(n=24, d=8, q=4, seed=0):
    x, y = _xy(n=n, d=d, seed=seed)
    ba, bb = fit_basis(x, q=q), fit_basis(y, q=q)
    assert ba["q"] == q and bb["q"] == q and not ba["reduced"]
    ids = [f"coco-val2017-{i:012d}" for i in range(n)]
    return x, y, ids, ba, bb


def _pack(x, y, ba, bb):
    from prh_replication.anisotropic_kernels import make_pack

    return make_pack(x - ba["mu"], y - bb["mu"], ba["U"], bb["U"])


def test_identity_factors_match_zero_deltas_and_are_not_ada():
    rng = np.random.default_rng(1)
    q = 4
    eye = np.eye(q)
    got = project_factors(eye, np.zeros((q, q)), "oblique_diag")
    assert np.allclose(got["B"], eye, atol=1e-10)
    assert np.allclose(got["D"], eye, atol=1e-10)
    assert got["r_basis"] == pytest.approx(0.0, abs=1e-10)
    assert r_off(got["H"]) == 0.0
    a = eye + 0.15 * rng.normal(size=(q, q))
    h = np.diag(np.array([0.2, -0.1, 0.4, -0.3]))
    d = sym_expm(h)
    b = induced_b(a, d)
    ada = a @ d @ np.linalg.inv(a)
    assert np.allclose(b, b.T, atol=1e-10)
    assert not np.allclose(ada, b, atol=1e-6)


def test_orthogonal_a_matches_conjugate_and_coordinates_match_kernel():
    rng = np.random.default_rng(2)
    q = 4
    raw = rng.normal(size=(q, q))
    ortho, _ = np.linalg.qr(raw)
    h = np.diag(np.array([0.3, -0.2, 0.1, 0.0]))
    d = sym_expm(h)
    b = induced_b(ortho, d)
    assert np.allclose(b, ortho @ d @ ortho.T, atol=1e-10)
    projected = project_factors(ortho, h, "oblique_diag")
    assert np.allclose(projected["B"], ortho @ d @ ortho.T, atol=1e-8)
    u = rng.normal(size=(6, q))
    z = np.linalg.solve(ortho, u.T).T
    assert np.allclose(u @ b @ u.T, z @ d @ z.T, atol=1e-8)
    scaled = ortho * np.array([1.4, 0.7, 1.1, 0.9])
    explicit = induced_b(scaled, d)
    absorbed = project_factors(scaled, h, "oblique_diag")
    assert np.allclose(absorbed["B"], explicit, atol=1e-8)


def test_full_space_matches_reduced_kernel_and_residual_is_identity():
    rng = np.random.default_rng(3)
    x, y, _, ba, bb = _basis_pair()
    q = ba["q"]
    h = np.diag(np.linspace(-0.2, 0.2, q))
    part = project_factors(np.eye(q) + 0.05 * rng.normal(size=(q, q)), h, "oblique_diag")
    za = x - ba["mu"]
    zb = y - bb["mu"]
    eye_b = np.eye(bb["q"])
    gram = metric_gram(za, ba["U"], part["B"])
    coords = za @ ba["U"]
    manual = za @ za.T + coords @ (part["B"] - np.eye(q)) @ coords.T
    assert np.allclose(gram, manual, atol=1e-8)
    pack = _pack(x, y, ba, bb)
    contracted = scores_numpy(part["B"] - np.eye(q), np.zeros_like(eye_b), pack)
    direct = scores_direct_f64(za, zb, ba["U"], bb["U"], part["B"], eye_b)
    assert contracted["excess"] == pytest.approx(direct["excess"], abs=1e-8)
    d = ba["U"].shape[0]
    u_full, _ = np.linalg.qr(np.hstack([ba["U"], rng.normal(size=(d, d - q))]))
    residual = u_full[:, q:]
    metric = np.eye(d) + ba["U"] @ (part["B"] - np.eye(q)) @ ba["U"].T
    assert np.allclose(metric @ residual, residual, atol=1e-8)
    assert np.linalg.eigvalsh(part["B"]).min() > 0.0
    assert np.linalg.eigvalsh(metric).min() > 0.0
    gen = generalized_eigenvalues(part["D"], part["A"])
    assert np.allclose(np.sort(gen), np.sort(np.linalg.eigvalsh(part["B"])), atol=1e-7)


def test_blocks_stay_zero_off_block_and_diagonal_penalty_is_zero():
    rng = np.random.default_rng(4)
    q = 4
    h = rng.normal(size=(q, q))
    h = 0.5 * (h + h.T)
    part = project_factors(np.eye(q), 0.2 * h, "block2")
    for i in range(q):
        for j in range(q):
            if i // 2 != j // 2:
                assert part["D"][i, j] == pytest.approx(0.0, abs=1e-10)
                assert part["H"][i, j] == pytest.approx(0.0, abs=1e-10)
    diag = project_factors(np.eye(q), h, "fixed_diag")
    assert r_off(diag["H"]) == 0.0
    assert np.allclose(diag["A"], np.eye(q))
    shared = [part, part]
    assert np.allclose(shared[0]["B"], shared[1]["B"])
    assert total_distortion([part], "shared_pc") == pytest.approx(2.0 * part["distortion"])
    assert total_distortion([part, diag], "separate") == pytest.approx(part["distortion"] + diag["distortion"])
    assert total_distortion([part], "a_only") == pytest.approx(part["distortion"])


def test_backtracking_keeps_factors_consistent_with_b():
    q = 4
    huge = np.eye(q) * 3.0
    identity = [(np.eye(q), np.zeros((q, q)))]
    proposed = [(np.eye(q) + 0.4 * np.ones((q, q)), huge)]
    from prh_replication.structured_factors import backtrack, identity_factors

    prev = [identity_factors(q, "block4")]
    parts, raw, alpha = backtrack(identity, proposed, prev, "block4", "a_only", 0.1)
    assert alpha < 1.0
    assert factors_feasible([parts[0]], "a_only", 0.1)
    rebuilt = induced_b(parts[0]["A"], parts[0]["D"])
    assert np.allclose(rebuilt, parts[0]["B"], atol=1e-8)
    assert np.allclose(raw[0][0], parts[0]["A"])
    sv = parts[0]["singular_values"]
    assert sv.min() >= 0.5 - 1e-8 and sv.max() <= 2.0 + 1e-8


def test_retraction_clips_the_induced_metric_and_keeps_a_free_direction():
    from prh_replication.structured_factors import accept_step, identity_factors, project_induced_matrices

    q = 4
    rho = 2.0
    prev = [identity_factors(q, "oblique_diag")]
    prev_raw = [(np.eye(q), np.zeros((q, q)))]
    # Eigenvalue 6 is outside [1/4, 4]. The second weight is free and should move.
    proposed = [(np.eye(q), np.diag(np.log(np.array([6.0, 1.8, 1.0, 0.7]))))]
    parts, raw, alpha, how = accept_step(prev_raw, proposed, prev, "oblique_diag", "a_only", rho)
    assert how == "retracted" and alpha == 1.0
    assert factors_feasible([parts[0]], "a_only", rho)
    assert np.allclose(induced_b(parts[0]["A"], parts[0]["D"]), parts[0]["B"], atol=1e-8)
    assert parts[0]["eig"].max() <= 4.0 + 1e-6
    assert parts[0]["eig"].max() > 3.5
    order = np.argsort(parts[0]["eig"])
    assert parts[0]["eig"][order][-2] == pytest.approx(1.8, abs=1e-5)
    gauged = project_factors(*proposed[0], "oblique_diag")
    target = project_induced_matrices([gauged["B"]], "a_only", rho)[0]
    assert np.allclose(parts[0]["B"], target, atol=1e-6)

    rng = np.random.default_rng(2)
    shear = np.eye(q) + 0.15 * rng.normal(size=(q, q))
    base = project_factors(shear, np.diag(rng.normal(size=q) * 0.15), "oblique_diag")
    inflated = (base["A"], base["H"] * 4.0)
    parts_o, _, _, how_o = accept_step(prev_raw, [inflated], prev, "oblique_diag", "a_only", 0.4)
    assert how_o in ("retracted", "accepted")
    assert factors_feasible([parts_o[0]], "a_only", 0.4)
    assert np.allclose(induced_b(parts_o[0]["A"], parts_o[0]["D"]), parts_o[0]["B"], atol=1e-7)
    off = parts_o[0]["D"] - np.diag(np.diag(parts_o[0]["D"]))
    assert np.linalg.norm(off) < 1e-8

    block_prev = [identity_factors(q, "block2")]
    h_block = np.zeros((q, q))
    h_block[0, 1] = h_block[1, 0] = 0.3
    h_block[2, 3] = h_block[3, 2] = -0.2
    h_block += np.diag(np.log(np.array([2.5, 1.4, 0.8, 0.5])))
    block_prop = [(shear, h_block * 3.0)]
    parts_b, _, _, how_b = accept_step([(np.eye(q), np.zeros((q, q)))], block_prop, block_prev, "block2", "a_only", 0.4)
    assert how_b == "retracted"
    assert factors_feasible([parts_b[0]], "a_only", 0.4)
    d = parts_b[0]["D"]
    assert np.linalg.norm(d[:2, 2:]) < 1e-8 and np.linalg.norm(d[2:, :2]) < 1e-8
    gauged_b = project_factors(*block_prop[0], "block2")
    target_b = project_induced_matrices([gauged_b["B"]], "a_only", 0.4)[0]
    assert np.allclose(parts_b[0]["B"], target_b, atol=1e-5)
    assert np.allclose(raw[0][0], parts[0]["A"])


def test_factor_stream_does_not_stall_on_the_constraint_boundary():
    x, y, ids, ba, bb = _basis_pair(n=32, d=8, q=4, seed=7)
    raw = run_factor_stream(
        x, y, ids, ba, bb,
        family="oblique_diag",
        mode="a_only",
        rho=0.1,
        lambda_basis=0.1,
        lambda_off=0.0,
        init_seed=1,
        sampling_seed=1,
        n_updates=30,
        quarter_fraction=0.25,
    )
    assert raw["ok"]
    alphas = [row["alpha"] for row in raw["curves"] if row["step"] > 0]
    assert sum(alpha == 0.0 for alpha in alphas) < 0.4 * len(alphas)
    assert raw["status"]["max_eig_residual"] <= 1e-6
    assert raw["status"]["max_dist_residual"] <= 1e-6
    part = raw["final_parts"][0]
    assert np.allclose(induced_b(part["A"], part["D"]), part["B"], atol=1e-7)


def test_objective_gradient_matches_finite_differences_and_is_finite_at_identity():
    x, y, _, ba, bb = _basis_pair(n=16, d=6, q=4, seed=5)
    pack = _pack(x, y, ba, bb)
    from prh_replication.anisotropic_kernels import pack_to_torch

    pack_t = pack_to_torch(pack)
    rng = np.random.default_rng(5)
    a0 = project_factors(np.eye(4) + 0.04 * rng.normal(size=(4, 4)), 0.05 * rng.normal(size=(4, 4)), "oblique_diag")
    h0 = a0["H"]
    a_np = a0["A"]

    def grads(a_np, h_np, family):
        a = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        h = torch.tensor(h_np, dtype=torch.float64, requires_grad=True)
        loss, excess = factor_objective([a], [h], pack_t, family=family, mode="a_only", lambda_basis=0.1, lambda_off=1.0)
        loss.backward()
        return float(loss.detach()), float(excess.detach()), a.grad.detach().numpy(), h.grad.detach().numpy()

    base, _, g_a, g_h = grads(a_np, h0, "oblique_diag")
    assert np.isfinite(g_a).all() and np.isfinite(g_h).all()
    eps = 1e-6
    a_step = a_np.copy()
    a_step[0, 1] += eps
    stepped, _, _, _ = grads(a_step, h0, "oblique_diag")
    assert (stepped - base) / eps == pytest.approx(g_a[0, 1], rel=1e-4, abs=1e-5)
    h_step = h0.copy()
    h_step[0, 0] += eps
    stepped_h, _, _, _ = grads(a_np, h_step, "oblique_diag")
    assert (stepped_h - base) / eps == pytest.approx(g_h[0, 0], rel=1e-4, abs=1e-5)
    eye = np.eye(4)
    zero = np.zeros((4, 4))
    _, _, g_a0, g_h0 = grads(eye, zero, "block2")
    assert np.isfinite(g_a0).all() and np.isfinite(g_h0).all()
    assert np.linalg.norm(g_a0) + np.linalg.norm(g_h0) > 1e-8


def test_stream_constraints_shared_batches_resume_and_reload(tmp_path: Path):
    x, y, ids, ba, bb = _basis_pair(n=32, d=8, q=4, seed=6)
    kwargs = dict(
        rho=0.1,
        lambda_basis=0.1,
        lambda_off=0.0,
        init_seed=1,
        sampling_seed=3,
        n_updates=4,
        quarter_fraction=0.25,
        val_features_a=x,
        val_features_b=y,
        record_indices=True,
    )
    oblique = run_factor_stream(x, y, ids, ba, bb, family="oblique_diag", mode="a_only", **kwargs)
    other = run_factor_stream(x, y, ids, ba, bb, family="exp_s", mode="separate", **kwargs)
    assert oblique["ok"] and other["ok"]
    assert oblique["subset_indices"] == other["subset_indices"]
    assert oblique["status"]["max_eig_residual"] <= 1e-6
    assert oblique["status"]["max_dist_residual"] <= 1e-6
    part = oblique["final_parts"][0]
    assert np.allclose(induced_b(part["A"], part["D"]), part["B"], atol=1e-8)
    saved = tmp_path / "factors.npz"
    np.savez(saved, A=part["A"], D=part["D"], B=part["B"])
    loaded = np.load(saved)
    pack = _pack(x, y, ba, bb)
    assert excess_on_pack(pack, part["B"], np.eye(4)) == pytest.approx(excess_on_pack(pack, induced_b(loaded["A"], loaded["D"]), np.eye(4)))
    held = {}

    def keep(state):
        if int(state["updates_done"]) == 2:
            held["state"] = state

    short = dict(kwargs)
    short["n_updates"] = 2
    short["lambda_off"] = 1.0
    full = dict(kwargs)
    full["lambda_off"] = 1.0
    first = run_factor_stream(x, y, ids, ba, bb, family="block2", mode="shared_pc", on_checkpoint=keep, **short)
    assert first["ok"]
    resumed = run_factor_stream(x, y, ids, ba, bb, family="block2", mode="shared_pc", resume=held["state"], **full)
    fresh = run_factor_stream(x, y, ids, ba, bb, family="block2", mode="shared_pc", **full)
    assert resumed["ok"] and fresh["ok"]
    assert np.allclose(resumed["final_parts"][0]["B"], fresh["final_parts"][0]["B"], atol=1e-8)
    assert resumed["subset_indices"] == fresh["subset_indices"]
    vals = [row["full_val_excess"] for row in fresh["curves"]]
    assert fresh["selected"]["val_excess"] == max(vals)
    left = sorted_log_spectrum(fresh["final_parts"][0]["B"])
    right = sorted_log_spectrum(np.eye(4))
    assert spectrum_distance(left, right) >= 0.0


def test_job_inventory_deduplicates_diagonal_penalties():
    design = json.loads(Path("configs/structured_metric_factors.json").read_text())
    langs = ["qwen2-7b", "qwen2.5-7b", "qwen3-8b-base", "olmo-7b-0724", "olmo2-1124-7b", "olmo-3-1025-7b"]
    visions = ["dinov2-small", "vit-in21k-small", "clip-laion-base"]
    vl = [(a, v) for a in langs for v in visions]
    ll = [tuple(pair) for pair in design["ll_pairs"]]
    jobs = build_jobs(design, vl, ll)
    assert len(vl) * len(ll) == 0 or len(jobs) == (len(vl) + len(ll)) * 7 * 3 * 2 * 3 + 3 * 2 * 3
    assert len(jobs) == 2664
    profiles = {(job["family"], job["profile"], job["lambda_off"]) for job in jobs if job["correspondence"] == "true"}
    assert ("oblique_diag", "basis", 0.0) in profiles
    assert not any(family in ("exp_s", "fixed_diag", "oblique_diag") and off == 10.0 for family, _, off in profiles)
    assert family_block("block4") == 4
    assert complexity_total(32, "block4", "separate") == 96
    assert complexity_total(32, "block4", "shared_pc") == 48
    assert complexity_total(32, "fixed_diag", "a_only") == 0
    denom = 32 * 31 / 2
    assert penalised_selection(0.4, 48, 32) == pytest.approx(0.4 - 0.01 * 48 / denom)
