"""Synthetic steering correctness. This does not reproduce an IMDb or Moonlight run.

No cached review activations or bridge matrices were available for fitting
parity. The reference vectors below are NumPy means and a pseudoinverse, not
outputs of rho_cca.
"""

import numpy as np

from rho_cca import (
    DifferenceInMeans,
    Endpoint,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RhoCCAFit,
    SteeringConfig,
    attach_steering_target,
    fit_steering,
    load_fit,
    match_norm,
    save_fit,
    split_rows,
)
from rho_cca.preprocess import PreparedSpace

from smoke_case import MAP_ATOL, MAP_RTOL, gap


def _space() -> PreparedSpace:
    return PreparedSpace(None, False, None, None, None, None, None)


def _alignment(source, partner, dataset: PairedDataset) -> RhoCCAFit:
    width = source.width
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(width, width))
    left, _singular, right = np.linalg.svd(raw, full_matrices=False)
    weights = np.linspace(0.4, 1.6, width)
    return RhoCCAFit(
        source=source,
        partner=partner,
        source_space=_space(),
        partner_space=_space(),
        l_source=left,
        l_partner=right.T,
        rho=np.linspace(0.2, 0.9, width),
        weights=weights,
        eta=0.01,
        beta=1.0,
        floor=1e-6,
        tau_source=0.01,
        tau_partner=0.01,
        n_negative_rho=0,
        n_floored=0,
        algebraic_failure=False,
        train_ids=dataset.split.train_ids,
        fingerprint=dataset.fingerprint(),
    )


def test_synthetic_steering_matches_an_independent_decode_and_attachment(tmp_path):
    rng = np.random.default_rng(5)
    source = Endpoint("synthetic-source", "acceptance", 3, "pre_block", "positions", 6)
    partner = Endpoint("synthetic-partner", "acceptance", 5, "pre_block", "positions", 6)
    target = Endpoint("synthetic-target", "acceptance", 7, "pre_block", "positions", 4)
    ids = tuple(f"row-{i}" for i in range(24))
    values = rng.normal(size=(24, 6))
    partner_values = values @ rng.normal(size=(6, 6))
    bare = PairedDataset(FeatureTable(source, ids, ids, values), FeatureTable(partner, ids, ids, partner_values))
    dataset = PairedDataset(bare.source, bare.partner, split_rows(bare, test_size=0.25, validation_size=0.25, seed=5))
    alignment = _alignment(source, partner, dataset)
    frozen_factor = alignment.l_source.copy()
    positive = dataset.split.train_ids[:4]
    negative = dataset.split.train_ids[4:8]
    fitted = fit_steering(
        alignment,
        ModelPair(source, partner),
        DifferenceInMeans(positive, negative),
        dataset,
        config=SteeringConfig(decoder="pseudoinverse", weights="reliability"),
    )
    assert gap(alignment.l_source, frozen_factor) == 0.0

    positions = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    reference_vector = values[[positions[item] for item in positive]].mean(0) - values[
        [positions[item] for item in negative]
    ].mean(0)
    factor = alignment.l_source * np.sqrt(alignment.weights)
    partner_factor = alignment.l_partner * np.sqrt(alignment.weights)
    command = reference_vector @ factor
    decoded = command @ np.linalg.pinv(partner_factor, rcond=1e-8)
    assert gap(fitted.source_vector, reference_vector) <= MAP_ATOL
    assert gap(fitted.command, command) <= MAP_ATOL
    assert gap(fitted.target_vector, decoded) <= MAP_ATOL or gap(fitted.target_vector, decoded) <= MAP_RTOL * max(
        1.0, float(np.max(np.abs(decoded)))
    )
    assert fitted.decoder_intercept is None

    bridge_ids = dataset.sample_ids
    anchor = partner_values + 0.01
    decoded_rows = anchor @ rng.normal(size=(6, 4))
    bridge_bare = PairedDataset(
        FeatureTable(partner, bridge_ids, bridge_ids, anchor),
        FeatureTable(target, bridge_ids, bridge_ids, decoded_rows),
    )
    bridge = PairedDataset(
        bridge_bare.source,
        bridge_bare.partner,
        split_rows(bridge_bare, test_size=0.25, validation_size=0.25, seed=6),
    )
    attached = attach_steering_target(
        fitted,
        bridge,
        anchor="partner",
        target=target,
        config=SteeringConfig(decoder="ridge", t=1e-3),
    )
    train_ids = bridge.split.train_ids
    train_pos = [positions[item] for item in train_ids]
    z_train = anchor[train_pos] @ partner_factor
    y_train = decoded_rows[train_pos]
    zc = z_train - z_train.mean(0)
    yc = y_train - y_train.mean(0)
    width = zc.shape[1]
    covariance = (zc.T @ zc) / zc.shape[0]
    lam = 1e-3 * float(np.trace(covariance)) / width
    reference_g = np.linalg.solve(covariance + lam * np.eye(width), (zc.T @ yc) / zc.shape[0])
    reference_target = command @ reference_g
    intercept = y_train.mean(0) - z_train.mean(0) @ reference_g
    assert gap(attached.command, fitted.command) <= MAP_ATOL
    assert gap(attached.source_vector, fitted.source_vector) <= MAP_ATOL
    assert gap(attached.g_target, reference_g) <= MAP_ATOL or gap(attached.g_target, reference_g) <= MAP_RTOL
    assert gap(attached.target_vector, reference_target) <= MAP_ATOL or gap(attached.target_vector, reference_target) <= MAP_RTOL
    assert gap(attached.displacement(target), reference_target) <= MAP_ATOL or gap(
        attached.displacement(target), reference_target
    ) <= MAP_RTOL
    assert gap(attached.displacement(target), reference_target + intercept) > 1e-8

    restored = load_fit(save_fit(attached, tmp_path / "steering.npz"))
    assert gap(restored.target_vector, attached.target_vector) <= MAP_ATOL
    matched = match_norm(restored.vector_for(target), 1.25)
    assert abs(matched.applied_norm - 1.25) <= MAP_ATOL
    assert gap(matched.raw, restored.target_vector) <= MAP_ATOL
    quiet = match_norm(np.full(4, 1e-12), 3.0)
    assert quiet.degenerate
    assert gap(quiet.applied, np.zeros(4)) == 0.0
