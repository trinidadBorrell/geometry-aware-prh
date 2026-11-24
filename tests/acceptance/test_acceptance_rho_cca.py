"""Real-data ρ-CCA fitting parity on the same smoke triple.

Training correlations, reliability weights, and CKA were computed once by the
legacy sae_crosscov and frozen-partner routines. They were not copied from
rho_cca.
"""

import numpy as np

from rho_cca import RhoCCAConfig, fit_rho_cca, load_fit, save_fit

from smoke_case import CKA_ATOL, MAP_ATOL, MAP_RTOL, gap, load, native_test, paired, source_test


def _rows(payload):
    return (
        {"train": payload["train_olmo"], "val": payload["val_olmo"], "test": payload["test_olmo"]},
        {"train": payload["train_clip"], "val": payload["val_clip"], "test": payload["test_clip"]},
    )


def _fit(payload, source_rows, partner_rows):
    pair, dataset = paired(payload, source_rows=source_rows, partner_rows=partner_rows)
    fitted = fit_rho_cca(pair, dataset, config=RhoCCAConfig())
    return dataset, fitted


def _within(actual, expected) -> bool:
    error = gap(actual, expected)
    scale = max(1.0, float(np.max(np.abs(expected))))
    return error <= MAP_ATOL or error <= MAP_RTOL * scale


def test_rho_cca_matches_the_legacy_fit_and_does_not_read_test_rows(tmp_path):
    payload = load()
    source_rows, partner_rows = _rows(payload)
    dataset, fitted = _fit(payload, source_rows, partner_rows)
    assert fitted.algebraic_failure is False
    assert _within(fitted.rho, payload["cca_rho"])
    assert _within(fitted.weights, payload["cca_weights"])
    reference_metric = (payload["cca_L_source"] * np.sqrt(payload["cca_weights"])) @ (
        payload["cca_L_source"] * np.sqrt(payload["cca_weights"])
    ).T
    equal_metric = payload["cca_L_source"] @ payload["cca_L_source"].T
    assert _within(fitted.metric_factor(fitted.source, weights="reliability", dense=True), reference_metric)
    assert _within(fitted.metric_factor(fitted.source, weights="equal", dense=True), equal_metric)
    if not _within(fitted.l_source, payload["cca_L_source"]):
        projector = fitted.l_source @ np.linalg.pinv(fitted.l_source)
        reference = payload["cca_L_source"] @ np.linalg.pinv(payload["cca_L_source"])
        assert _within(projector, reference)

    before = fitted.l_source.copy()
    native = native_test(payload)
    weighted = fitted.cka_with_native(source_test(dataset, payload), native, weights="reliability")
    equal = fitted.cka_with_native(source_test(dataset, payload), native, weights="equal")
    for score, key in ((weighted, "cka_weighted"), (equal, "cka_equal")):
        reference = payload[key]
        assert abs(score.raw - float(reference[0])) <= CKA_ATOL
        assert abs(score.chance - float(reference[1])) <= CKA_ATOL
        assert abs(score.excess - float(reference[2])) <= CKA_ATOL
    assert gap(fitted.l_source, before) == 0.0

    scrambled_source = {name: rows.copy() for name, rows in source_rows.items()}
    scrambled_partner = {name: rows.copy() for name, rows in partner_rows.items()}
    for name in ("val", "test"):
        scrambled_source[name] = scrambled_source[name][::-1]
        scrambled_partner[name] = scrambled_partner[name][::-1]
    _dataset, again = _fit(payload, scrambled_source, scrambled_partner)
    assert gap(again.weights, fitted.weights) == 0.0
    assert gap(again.l_source @ again.l_source.T, fitted.l_source @ fitted.l_source.T) <= MAP_ATOL

    restored = load_fit(save_fit(fitted, tmp_path / "cca.npz"))
    restored_score = restored.cka_with_native(source_test(dataset, payload), native_test(payload))
    assert abs(restored_score.excess - weighted.excess) <= CKA_ATOL
    assert gap(restored.weights, fitted.weights) <= MAP_ATOL
    assert gap(restored.l_source, fitted.l_source) <= MAP_ATOL
