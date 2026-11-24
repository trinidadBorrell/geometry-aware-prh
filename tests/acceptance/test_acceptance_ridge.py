"""Real-data ridge fitting parity on the cached smoke triple.

The reference map was produced by the unmodified legacy ridge solver, and it
matches frozen_partner_cka_audit_v1 artifacts/smoke_donor_W.npy. Scores are
already q=32 PCA coordinates.
"""

from rho_cca import RidgeConfig, fit_ridge, load_fit, save_fit

from smoke_case import CKA_ATOL, MAP_ATOL, MAP_RTOL, gap, load, native_test, paired, source_test


def _fit(payload, source_rows, partner_rows):
    pair, dataset = paired(payload, source_rows=source_rows, partner_rows=partner_rows)
    fitted = fit_ridge(pair, dataset, config=RidgeConfig(t=None))
    return dataset, fitted


def test_ridge_matches_the_audited_map_and_ignores_heldout_rows(tmp_path):
    payload = load()
    source_rows = {"train": payload["train_olmo"], "val": payload["val_olmo"], "test": payload["test_olmo"]}
    partner_rows = {"train": payload["train_clip"], "val": payload["val_clip"], "test": payload["test_clip"]}
    dataset, fitted = _fit(payload, source_rows, partner_rows)
    weight_error = gap(fitted.weight, payload["ridge_W"])
    metric_error = gap(fitted.metric_factor(dense=True), payload["ridge_W"] @ payload["ridge_W"].T)
    assert weight_error <= MAP_ATOL or weight_error <= MAP_RTOL * max(1.0, float(abs(payload["ridge_W"]).max()))
    assert metric_error <= MAP_ATOL or metric_error <= MAP_RTOL * max(1.0, float(abs(payload["ridge_W"]).max()))
    assert fitted.t == payload["meta"]["selected_t"]
    assert gap(fitted.mu_source, payload["ridge_mu_source"]) <= MAP_ATOL
    assert gap(fitted.mu_partner, payload["ridge_mu_partner"]) <= MAP_ATOL
    assert abs(fitted.validation_mse - payload["meta"]["selected_validation_mse"]) <= CKA_ATOL

    native = native_test(payload)
    score = fitted.cka_with_native(source_test(dataset, payload), native)
    reference = payload["cka_ridge"]
    assert abs(score.raw - float(reference[0])) <= CKA_ATOL
    assert abs(score.chance - float(reference[1])) <= CKA_ATOL
    assert abs(score.excess - float(reference[2])) <= CKA_ATOL
    untouched = native_test(payload)
    native.values[0, 0] += 1.0
    assert gap(untouched.values, payload["test_dinov2"]) == 0.0

    scrambled = {name: rows.copy() for name, rows in source_rows.items()}
    scrambled["test"] = scrambled["test"][::-1]
    partner_scrambled = {name: rows.copy() for name, rows in partner_rows.items()}
    partner_scrambled["test"] = partner_scrambled["test"][::-1]
    _dataset, again = _fit(payload, scrambled, partner_scrambled)
    assert gap(again.weight, fitted.weight) == 0.0
    assert again.t == fitted.t

    path = save_fit(fitted, tmp_path / "ridge.npz")
    restored = load_fit(path)
    assert gap(restored.weight, fitted.weight) <= MAP_ATOL
    restored_score = restored.cka_with_native(source_test(dataset, payload), native_test(payload))
    assert abs(restored_score.excess - score.excess) <= CKA_ATOL
    assert gap(fitted.weight, payload["ridge_W"]) <= MAP_ATOL
