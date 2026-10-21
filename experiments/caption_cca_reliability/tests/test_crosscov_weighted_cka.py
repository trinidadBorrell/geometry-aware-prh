"""Focused checks for cross-covariance weighted CKA."""

from __future__ import annotations

import unittest

import numpy as np

from prh_replication.crosscov_weighted_cka import (
    fit_raw_and_cca,
    linear_cka_agrees_with_gram,
    linear_cka_features,
    raw_power_weights,
    reliability_weights,
    select_alpha,
    transform_cca,
    transform_raw,
    two_sided_cross_spectrum,
)
from prh_replication.crosscov_weighted_cka_study import _run_checks


class WeightedCkaTests(unittest.TestCase):
    def test_verification_suite(self) -> None:
        checks = _run_checks()
        for key, value in checks.items():
            self.assertTrue(value, msg=key)

    def test_unit_svd_preserves_cka(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(200, 12))
        b = a @ rng.normal(size=(12, 12)) + 0.2 * rng.normal(size=(200, 12))
        maps = fit_raw_and_cca(a, b, 0.01, 1e-8)
        c0 = linear_cka_features(a, b)
        c1 = linear_cka_features(a @ maps["svd"]["U"], b @ maps["svd"]["V"])
        self.assertLess(abs(c0 - c1), 1e-10)

    def test_feature_gram_agreement(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.normal(size=(80, 10))
        b = rng.normal(size=(80, 10))
        self.assertTrue(linear_cka_agrees_with_gram(a, b))

    def test_alpha1_is_unit_metric(self) -> None:
        sigma = np.array([4.0, 2.0, 1.0, 0.5])
        pack = raw_power_weights(sigma, alpha=1.0, floor=1e-3, normalize=False)
        self.assertTrue(np.allclose(pack["w"], 1.0))

    def test_select_alpha_tie_prefers_near_one(self) -> None:
        curve = [{"alpha": 0.0, "cka": 0.5}, {"alpha": 1.0, "cka": 0.5}, {"alpha": 2.0, "cka": 0.5}]
        self.assertEqual(select_alpha(curve, 1e-8), 1.0)

    def test_beta0_matches_cca(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(size=(300, 8))
        b = a + 0.1 * rng.normal(size=(300, 8))
        maps = fit_raw_and_cca(a, b, 0.01, 1e-8)
        pack = reliability_weights(maps["rho"], 0.0, 1e-6, normalize=True)
        left, right = transform_cca(a, b, maps["cca"]["L_A"], maps["cca"]["L_B"], pack["w"], "both")
        t_a = a @ maps["cca"]["L_A"]
        t_b = b @ maps["cca"]["L_B"]
        self.assertLess(abs(linear_cka_features(left, right) - linear_cka_features(t_a, t_b)), 1e-10)

    def test_two_sided_scales_sigma_by_w(self) -> None:
        rng = np.random.default_rng(4)
        shared = rng.normal(size=(500, 5))
        a = np.hstack([shared, rng.normal(size=(500, 11))])
        b = np.hstack([shared * 0.8, rng.normal(size=(500, 11))])
        maps = fit_raw_and_cca(a, b, 0.01, 1e-8)
        sigma = maps["svd"]["sigma"]
        pack = raw_power_weights(sigma, alpha=2.0, floor=1e-3, normalize=False)
        diag = two_sided_cross_spectrum(a, b, maps["svd"]["U"], maps["svd"]["V"], pack["w"])
        self.assertLess(float(np.max(np.abs(diag - pack["w"] * sigma))), 1e-8)


if __name__ == "__main__":
    unittest.main()
