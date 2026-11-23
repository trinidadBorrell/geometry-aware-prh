"""Splits, providers, and steering transport."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rho_cca import (
    ArrayProvider,
    DifferenceInMeans,
    Endpoint,
    FeatureTable,
    MapExtractor,
    ModelPair,
    NonlinearTransportError,
    PairedDataset,
    RhoCCAConfig,
    RhoCCAError,
    RidgeConfig,
    SteeringConfig,
    attach_steering_target,
    collect_pair,
    fit_rho_cca,
    fit_ridge,
    fit_steering,
    hashed_embedding,
    match_norm,
    split_groups,
    split_rows,
    stratified_group_folds,
)
import rho_cca.extract as extract_module


def endpoint(model, width, layer):
    return Endpoint(model, f"rev-{model}", layer, "pre_block", "mean", width)


def make_pair(n=40, da=4, db=5, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    source_ep = endpoint("A", da, 1)
    partner_ep = endpoint("B", db, 2)
    ids = tuple(f"s{i}" for i in range(n))
    if groups is None:
        groups = tuple(f"g{i}" for i in range(n))
    source = FeatureTable(source_ep, ids, groups, rng.normal(size=(n, da)))
    partner = FeatureTable(partner_ep, ids, groups, rng.normal(size=(n, db)))
    return ModelPair(source_ep, partner_ep), source, partner


def with_split(source, partner, *, groups=False, seed=0):
    bare = PairedDataset(source, partner)
    if groups:
        split = split_groups(bare, test_size=0.3, validation_size=0.5, seed=seed)
    else:
        split = split_rows(bare, test_size=0.25, validation_size=0.25, seed=seed)
    return PairedDataset(source, partner, split)


def test_row_and_group_splits_are_disjoint_and_shared():
    groups = tuple(f"g{i // 2}" for i in range(40))
    _model_pair, source, partner = make_pair(n=40, seed=2, groups=groups)
    data = with_split(source, partner, groups=True, seed=3)
    parts = [set(data.split.train_ids), set(data.split.validation_ids), set(data.split.test_ids)]
    assert parts[0].isdisjoint(parts[1])
    assert parts[0].isdisjoint(parts[2])
    assert parts[1].isdisjoint(parts[2])
    groups = {}
    lookup = dict(zip(data.sample_ids, data.group_ids, strict=True))
    for name, ids in ("train", data.split.train_ids), ("validation", data.split.validation_ids), ("test", data.split.test_ids):
        for sample_id in ids:
            groups.setdefault(lookup[sample_id], set()).add(name)
    assert all(len(names) == 1 for names in groups.values())
    labels = np.array([0 if (index // 2) % 2 == 0 else 1 for index in range(40)])
    folds = stratified_group_folds(PairedDataset(source, partner), labels, n_splits=4, seed=1)
    assert len(folds) == 4
    for fold in folds:
        train_groups = {lookup[sample_id] for sample_id in fold.train_ids}
        test_groups = {lookup[sample_id] for sample_id in fold.test_ids}
        assert train_groups.isdisjoint(test_groups)


def test_alignment_rejects_ambiguous_and_mismatched_ids():
    model_pair, source, partner = make_pair()
    with pytest.raises(RhoCCAError):
        FeatureTable(source.endpoint, ("s0", "s0"), ("g", "g"), np.ones((2, source.endpoint.width)))
    shuffled = partner.take(partner.index(tuple(reversed(partner.sample_ids))))
    aligned = PairedDataset(source, shuffled)
    assert aligned.partner.sample_ids == aligned.source.sample_ids
    foreign = FeatureTable(
        partner.endpoint,
        tuple(f"other{i}" for i in range(len(source.sample_ids))),
        source.group_ids,
        partner.values,
    )
    with pytest.raises(RhoCCAError):
        PairedDataset(source, foreign)
    data = with_split(source, partner)
    other = ModelPair(endpoint("A", source.endpoint.width, 9), model_pair.partner)
    with pytest.raises(RhoCCAError):
        fit_ridge(other, data, config=RidgeConfig())


def test_difference_in_means_ignores_offsets_and_test_labels():
    model_pair, source, partner = make_pair(seed=5)
    data = with_split(source, partner)
    fit = fit_rho_cca(model_pair, data, config=RhoCCAConfig())
    positive = data.split.train_ids[:4]
    negative = data.split.train_ids[4:8]
    objective = DifferenceInMeans(positive, negative)
    steered = fit_steering(fit, model_pair, objective, data, config=SteeringConfig(decoder="pseudoinverse"))
    pos = data.source.values[data.source.index(positive)].mean(0)
    neg = data.source.values[data.source.index(negative)].mean(0)
    np.testing.assert_allclose(steered.source_vector, pos - neg, atol=1e-10, rtol=1e-8)
    offset = data.source.values + 25.0
    shifted = PairedDataset(
        FeatureTable(data.source.endpoint, data.sample_ids, data.group_ids, offset),
        data.partner,
        data.split,
    )
    shifted_fit = fit_steering(fit, model_pair, objective, shifted, config=SteeringConfig(decoder="pseudoinverse"))
    np.testing.assert_allclose(shifted_fit.source_vector, steered.source_vector, atol=1e-10, rtol=1e-8)
    with pytest.raises(RhoCCAError):
        fit_steering(
            fit,
            model_pair,
            DifferenceInMeans(data.split.test_ids[:2], data.split.test_ids[2:4]),
            data,
            config=SteeringConfig(),
        )


def test_pseudoinverse_reproduces_coordinates_and_cancels_full_rank_weights():
    model_pair, source, partner = make_pair(n=48, da=4, db=4, seed=6)
    data = with_split(source, partner)
    fit = fit_rho_cca(model_pair, data, config=RhoCCAConfig())
    objective = DifferenceInMeans(data.split.train_ids[:6], data.split.train_ids[6:12])
    steered = fit_steering(fit, model_pair, objective, data, config=SteeringConfig(decoder="pseudoinverse"))
    f_source = fit.linear_factor(fit.source, weights="reliability")
    f_partner = fit.linear_factor(fit.partner, weights="reliability")
    f_source_eq = fit.linear_factor(fit.source, weights="equal")
    f_partner_eq = fit.linear_factor(fit.partner, weights="equal")
    decoded = steered.source_vector @ f_source @ np.linalg.pinv(f_partner)
    equal = steered.source_vector @ f_source_eq @ np.linalg.pinv(f_partner_eq)
    np.testing.assert_allclose(steered.target_vector, decoded, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(decoded, equal, atol=1e-6, rtol=1e-6)
    assert steered.weights_cancelled is True
    z = np.array([0.2, -0.4, 0.1, 0.3])
    attainable = z @ steered.g_target
    np.testing.assert_allclose(attainable @ steered.f_partner, z, atol=1e-8, rtol=1e-8)
    assert steered.decoder_intercept is None


def test_ridge_decoder_does_not_add_its_intercept_and_attachment_keeps_the_command():
    model_pair, source, partner = make_pair(n=48, da=4, db=5, seed=7)
    data = with_split(source, partner)
    alignment = fit_rho_cca(model_pair, data, config=RhoCCAConfig())
    objective = DifferenceInMeans(data.split.train_ids[:5], data.split.train_ids[5:10])
    steered = fit_steering(alignment, model_pair, objective, data, config=SteeringConfig(decoder="ridge", t=0.01))
    np.testing.assert_allclose(steered.target_vector, steered.command @ steered.g_target, atol=1e-10, rtol=1e-8)
    assert steered.decoder_intercept is not None
    assert float(np.linalg.norm(steered.decoder_intercept)) > 0.0
    intercept_added = steered.command @ steered.g_target + steered.decoder_intercept
    assert float(np.linalg.norm(intercept_added - steered.target_vector)) > 1e-6
    target = endpoint("C", 3, 4)
    rng = np.random.default_rng(8)
    anchor_ids = data.sample_ids
    bridge_source = FeatureTable(model_pair.partner, anchor_ids, data.group_ids, data.partner.values)
    bridge_target = FeatureTable(target, anchor_ids, data.group_ids, rng.normal(size=(len(anchor_ids), 3)) + 2.0)
    bridge = with_split(bridge_source, bridge_target, seed=1)
    attached = attach_steering_target(steered, bridge, anchor="partner", target=target, config=SteeringConfig(t=0.01))
    np.testing.assert_array_equal(attached.command, steered.command)
    np.testing.assert_array_equal(attached.source_vector, steered.source_vector)
    np.testing.assert_allclose(attached.target_vector, attached.command @ attached.g_target, atol=1e-10, rtol=1e-8)
    assert attached.target.identity_key() == target.identity_key()
    assert attached.vector_for(model_pair.partner).shape == steered.partner_vector.shape
    with pytest.raises(Exception):
        attached.vector_for(endpoint("D", 3, 5))


def test_nonlinear_prep_refuses_raw_transport_and_near_zero_norm_is_not_amplified():
    model_pair, source, partner = make_pair(seed=9)
    data = with_split(source, partner)
    fit = fit_rho_cca(model_pair, data, config=RhoCCAConfig(clip_quantile=0.9))
    objective = DifferenceInMeans(data.split.train_ids[:4], data.split.train_ids[4:8])
    with pytest.raises(NonlinearTransportError):
        fit_steering(fit, model_pair, objective, data, config=SteeringConfig())
    matched = match_norm(np.full(4, 1e-12), 2.5, near_zero_norm=1e-8)
    assert matched.degenerate
    assert matched.applied_norm == 0.0
    np.testing.assert_array_equal(matched.applied, np.zeros(4))
    kept = match_norm(np.array([3.0, 4.0]), 2.0)
    assert kept.raw_norm == pytest.approx(5.0, abs=1e-12)
    assert kept.applied_norm == pytest.approx(2.0, abs=1e-12)


def test_provider_and_toy_extractor_feed_the_public_fit():
    model_pair, source, partner = make_pair(n=24, seed=10)
    provider = ArrayProvider([source, partner])
    collected = collect_pair(model_pair, provider, source.sample_ids, source.group_ids)
    data = with_split(collected.source, collected.partner)
    fit = fit_ridge(model_pair, data, config=RidgeConfig(t=0.001))
    assert fit.weight.shape == (4, 5)
    inputs = [[index, index + 1] for index in range(12)]

    def embed(batch, endpoint):
        return hashed_embedding(batch, endpoint.width, seed=endpoint.layer or 0)

    extractor = MapExtractor(
        {model_pair.source.identity_key(): embed, model_pair.partner.identity_key(): embed}
    )
    short_ids = tuple(f"r{i}" for i in range(12))
    short_groups = tuple(f"g{i}" for i in range(12))
    extracted = collect_pair(
        model_pair,
        extractor,
        short_ids,
        short_groups,
        source_inputs=inputs,
        partner_inputs=[[index + 3] for index in range(12)],
    )
    split = split_rows(extracted, test_size=0.25, seed=0)
    fit_rho_cca(model_pair, PairedDataset(extracted.source, extracted.partner, split), config=RhoCCAConfig())
    source_text = inspect.getsource(extract_module)
    assert source_text.split("def preblock_rows")[0].find("import torch") == -1


def test_ridge_alignment_transports_by_the_regression_map():
    model_pair, source, partner = make_pair(n=36, da=3, db=4, seed=11)
    data = with_split(source, partner)
    alignment = fit_ridge(model_pair, data, config=RidgeConfig(t=0.001))
    objective = DifferenceInMeans(data.split.train_ids[:4], data.split.train_ids[4:8])
    steered = fit_steering(alignment, model_pair, objective, data, config=SteeringConfig())
    np.testing.assert_allclose(steered.target_vector, steered.source_vector @ alignment.linear_factor(), atol=1e-8, rtol=1e-8)
    assert steered.decoder == "ridge"
    assert steered.weights_cancelled is None
    with pytest.raises(RhoCCAError):
        fit_steering(alignment, model_pair, objective, data, config=SteeringConfig(decoder="pseudoinverse"))
