"""Core geometry tests.

Reference helpers in this file are copied from the audited angus-prh-replication
formulas (objective_independent.ridge_weight, crosscov reliability weights, and
frozen_partner_cka_audit.scores). They do not call rho_cca.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rho_cca import (
    Endpoint,
    EndpointMismatch,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RhoCCAConfig,
    RhoCCAError,
    RidgeConfig,
    fit_rho_cca,
    fit_ridge,
    linear_cka,
    linear_cka_from_feature_grams,
    load_fit,
    save_fit,
    split_rows,
)


def ref_scores(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    raw = np.square(x.T @ y).sum() / den
    chance = np.square(x).sum() * np.square(y).sum() / ((len(x) - 1) * den)
    return float(raw), float(raw - chance)


def ref_ridge(xc, yc, t):
    n, q = xc.shape
    covariance = (xc.T @ xc) / float(n)
    cross = (xc.T @ yc) / float(n)
    lam = float(t) * float(np.trace(covariance)) / float(q)
    weight = np.linalg.solve(covariance + lam * np.eye(q), cross)
    return weight, lam


def ref_weights(rho, beta=1.0, floor=1e-6):
    clipped = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0)
    weights = np.maximum(clipped * clipped, float(floor)) ** float(beta)
    return weights / float(np.mean(weights))


def endpoint(model, revision, width, layer):
    return Endpoint(model, revision, layer, "pre_block", "token", width, {"dtype": "float64"})


def pair_for(width_a, width_b):
    return ModelPair(endpoint("A", "rev-a", width_a, 1), endpoint("B", "rev-b", width_b, 2))


def table(endpoint_, rows, seed):
    rng = np.random.default_rng(seed)
    ids = tuple(f"s{i}" for i in range(rows))
    groups = tuple(f"g{i}" for i in range(rows))
    values = rng.normal(size=(rows, endpoint_.width))
    return FeatureTable(endpoint_, ids, groups, values)


def dataset_from(source, partner, **split_kwargs):
    bare = PairedDataset(source, partner)
    split = split_rows(bare, seed=0, test_size=0.25, validation_size=0.25, **split_kwargs)
    return PairedDataset(source, partner, split)


def test_rectangular_ridge_matches_the_independent_normal_equation():
    model_pair = pair_for(3, 5)
    rng = np.random.default_rng(1)
    source = table(model_pair.source, 40, 1)
    latent = rng.normal(size=(40, 3))
    partner_values = latent @ rng.normal(size=(3, 5)) + 0.01 * rng.normal(size=(40, 5))
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, partner_values)
    data = dataset_from(source, partner)
    fit = fit_ridge(model_pair, data, config=RidgeConfig(t=0.001))
    train = data.select(data.split.train_ids)
    x = train.source.values
    y = train.partner.values
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    expected, lam = ref_ridge(xc, yc, 0.001)
    assert fit.weight.shape == (3, 5)
    assert fit.metric_factor(dense=True).shape == (3, 3)
    np.testing.assert_allclose(fit.weight, expected, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(fit.lam, lam, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(fit.metric_factor(dense=True), expected @ expected.T, atol=1e-10, rtol=1e-8)
    centred = fit.metric_coordinates(train.source)
    prediction = fit.predict(train.source)
    np.testing.assert_allclose(prediction, centred + fit.mu_partner, atol=1e-10, rtol=1e-8)


def test_fit_ignores_test_rows_and_rejects_a_lookalike_basis():
    model_pair = pair_for(4, 4)
    source = table(model_pair.source, 36, 2)
    partner = table(model_pair.partner, 36, 3)
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, partner.values)
    data = dataset_from(source, partner)
    first = fit_ridge(model_pair, data, config=RidgeConfig(t=0.01))
    edited = data.partner.values.copy()
    test_positions = data.partner.index(data.split.test_ids)
    edited[test_positions] += 10.0
    changed = PairedDataset(
        data.source,
        FeatureTable(data.partner.endpoint, data.partner.sample_ids, data.partner.group_ids, edited),
        data.split,
    )
    second = fit_ridge(model_pair, changed, config=RidgeConfig(t=0.01))
    np.testing.assert_array_equal(first.weight, second.weight)
    chunks = []
    ids = list(data.sample_ids)
    for start in range(0, len(ids), 5):
        part = data.select(ids[start : start + 5])
        chunks.append(first.metric_coordinates(part.source))
    np.testing.assert_allclose(np.vstack(chunks), first.metric_coordinates(data.source), atol=1e-10, rtol=1e-8)
    lookalike = Endpoint("A", "other-revision", 1, "pre_block", "token", 4)
    foreign = FeatureTable(lookalike, data.source.sample_ids, data.source.group_ids, data.source.values)
    with pytest.raises(EndpointMismatch):
        first.metric_coordinates(foreign)


def test_penalty_selection_uses_validation_not_test():
    model_pair = pair_for(3, 3)
    rng = np.random.default_rng(4)
    source_values = rng.normal(size=(48, 3))
    partner_values = source_values @ np.eye(3) + 0.05 * rng.normal(size=(48, 3))
    source = FeatureTable(model_pair.source, tuple(f"s{i}" for i in range(48)), tuple(f"g{i}" for i in range(48)), source_values)
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, partner_values)
    data = dataset_from(source, partner)
    config = RidgeConfig(t=None, t_grid=(1e-6, 1e-2, 10.0))
    chosen = fit_ridge(model_pair, data, config=config)
    noisy = data.partner.values.copy()
    noisy[data.partner.index(data.split.validation_ids)] = rng.normal(size=(len(data.split.validation_ids), 3)) * 20
    noisy_data = PairedDataset(
        data.source,
        FeatureTable(data.partner.endpoint, data.sample_ids, data.group_ids, noisy),
        data.split,
    )
    other = fit_ridge(model_pair, noisy_data, config=config)
    assert other.validation_mse != chosen.validation_mse
    scrambled = data.partner.values.copy()
    scrambled[data.partner.index(data.split.test_ids)] = rng.normal(size=(len(data.split.test_ids), 3)) * 50
    scrambled_data = PairedDataset(
        data.source,
        FeatureTable(data.partner.endpoint, data.sample_ids, data.group_ids, scrambled),
        data.split,
    )
    held = fit_ridge(model_pair, scrambled_data, config=config)
    assert held.t == chosen.t
    np.testing.assert_allclose(held.weight, chosen.weight, atol=1e-10, rtol=1e-8)


def test_rho_cca_weights_and_equal_directions_cover_the_support():
    model_pair = pair_for(5, 4)
    source = table(model_pair.source, 40, 5)
    partner = table(model_pair.partner, 40, 6)
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, partner.values)
    data = dataset_from(source, partner)
    fit = fit_rho_cca(model_pair, data, config=RhoCCAConfig())
    assert fit.n_components == 4
    expected = ref_weights(fit.rho)
    np.testing.assert_allclose(fit.weights, expected, atol=1e-10, rtol=1e-8)
    equal = fit.support_factor(model_pair.source, weights="equal")
    weighted = fit.support_factor(model_pair.source, weights="reliability")
    np.testing.assert_allclose(equal, fit.l_source, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(weighted, fit.l_source * np.sqrt(fit.weights), atol=1e-10, rtol=1e-8)
    projector_equal = equal @ np.linalg.pinv(equal)
    projector_weighted = weighted @ np.linalg.pinv(weighted)
    np.testing.assert_allclose(projector_equal, projector_weighted, atol=1e-8, rtol=1e-8)


def test_one_sided_cka_leaves_the_third_basis_native():
    model_pair = pair_for(4, 4)
    source = table(model_pair.source, 30, 7)
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, table(model_pair.partner, 30, 8).values)
    data = dataset_from(source, partner)
    fit = fit_rho_cca(model_pair, data, config=RhoCCAConfig(eta=0.01))
    third = endpoint("C", "rev-c", 6, 3)
    native = FeatureTable(third, source.sample_ids, source.group_ids, table(third, 30, 9).values)
    score = fit.cka_with_native(data.source, native)
    raw, excess = ref_scores(fit.coordinates(data.source), native.values)
    assert score.raw == pytest.approx(raw, abs=1e-10)
    assert score.excess == pytest.approx(excess, abs=1e-10)
    with pytest.raises(EndpointMismatch):
        fit.coordinates(native)


def test_feature_and_gram_cka_agree_and_match_the_audit_formula():
    rng = np.random.default_rng(10)
    left = rng.normal(size=(25, 6))
    right = left @ rng.normal(size=(6, 5)) + 0.1 * rng.normal(size=(25, 5))
    score = linear_cka(left, right)
    gram = linear_cka_from_feature_grams(left, right)
    raw, excess = ref_scores(left, right)
    assert score.raw == pytest.approx(raw, abs=1e-10)
    assert score.excess == pytest.approx(excess, abs=1e-10)
    assert score.chance == pytest.approx(score.raw - score.excess, abs=1e-12)
    assert gram.raw == pytest.approx(score.raw, abs=1e-8)
    assert gram.excess == pytest.approx(score.excess, abs=1e-8)
    shifted = right + np.array([3.0, -2.0, 0.5, 1.0, -4.0])
    moved = linear_cka(left + 8.0, shifted)
    assert moved.raw == pytest.approx(score.raw, abs=1e-10)
    order = rng.permutation(len(left))
    permuted = linear_cka(left[order], right[order])
    assert permuted.raw == pytest.approx(score.raw, abs=1e-10)
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    rotated = linear_cka(left, right @ rotation)
    assert rotated.raw == pytest.approx(score.raw, abs=1e-10)
    assert rotated.chance == pytest.approx(score.chance, abs=1e-10)


def test_permutation_expectation_matches_every_row_shuffle():
    left = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.7]], dtype=np.float64)
    right = np.array([[1.0, 0.2], [0.0, 1.0], [0.4, 0.5]], dtype=np.float64)
    score = linear_cka(left, right)
    raws = []
    for order in itertools.permutations(range(3)):
        raws.append(ref_scores(left, right[list(order)])[0])
    assert score.chance == pytest.approx(float(np.mean(raws)), abs=1e-10)
    with pytest.raises(RhoCCAError):
        linear_cka(np.ones((4, 3)), right)


def test_save_load_roundtrip_and_schema_shape(tmp_path):
    model_pair = pair_for(4, 3)
    source = table(model_pair.source, 28, 11)
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, table(model_pair.partner, 28, 12).values)
    data = dataset_from(source, partner)
    fit = fit_ridge(model_pair, data, config=RidgeConfig(t=0.001, pca_components=2))
    path = save_fit(fit, tmp_path / "ridge.rho_cca.npz")
    loaded = load_fit(path)
    np.testing.assert_allclose(loaded.weight, fit.weight, atol=1e-12, rtol=0)
    np.testing.assert_allclose(loaded.metric_coordinates(data.source), fit.metric_coordinates(data.source), atol=1e-10, rtol=1e-8)
    blob = np.load(path, allow_pickle=False)
    weight = blob["weight"]
    meta = blob["__meta__"]
    np.savez(tmp_path / "bad.npz", __meta__=meta, weight=weight[:1], mu_source=blob["mu_source"], mu_partner=blob["mu_partner"])
    with pytest.raises(RhoCCAError):
        load_fit(tmp_path / "bad.npz")


def test_clip_threshold_is_frozen_from_training_rows():
    model_pair = pair_for(3, 3)
    source = table(model_pair.source, 30, 13)
    partner = FeatureTable(model_pair.partner, source.sample_ids, source.group_ids, table(model_pair.partner, 30, 14).values)
    data = dataset_from(source, partner)
    fit = fit_ridge(model_pair, data, config=RidgeConfig(t=0.001, clip_quantile=0.95))
    train = data.select(data.split.train_ids).source.values
    flat = np.abs(train).reshape(-1)
    index = min(int(0.95 * flat.size), flat.size - 1)
    assert fit.source_space.clip_threshold == pytest.approx(float(np.sort(flat)[index]), abs=1e-12)
    huge = data.source.values.copy()
    huge[0, 0] = 1e6
    projected = fit.project_native(FeatureTable(data.source.endpoint, data.sample_ids, data.group_ids, huge))
    assert np.max(np.abs(projected)) <= 1.0 + 1e-8
