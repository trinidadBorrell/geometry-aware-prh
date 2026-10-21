"""Focused checks for truncated metric mKNN."""

from __future__ import annotations

import unittest

import numpy as np

from prh_replication.crosscov_metric_mknn import (
    factorized_distance_sq,
    knn_indices_stable,
    metric_factor,
    quadratic_form_distance_sq,
    select_rank,
)
from prh_replication.crosscov_metric_mknn_study import _run_checks
from prh_replication.spectral_calibration import sq_euclidean_rows


class MetricMknnTests(unittest.TestCase):
    def test_verification_suite(self) -> None:
        checks = _run_checks()
        for key, value in checks.items():
            if key == "notes":
                continue
            self.assertTrue(value, msg=key)

    def test_select_rank_prefers_larger_on_tie(self) -> None:
        curve = [{"rank": 8, "mknn": 0.2}, {"rank": 16, "mknn": 0.2}, {"rank": 4, "mknn": 0.1}]
        self.assertEqual(select_rank(curve, 1e-12), 16)

    def test_factor_shape(self) -> None:
        l = np.eye(10)
        w = np.ones(10)
        f = metric_factor(l, w, 4)
        self.assertEqual(f.shape, (10, 4))

    def test_quadratic_matches_factorized(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 12))
        l = np.linalg.qr(rng.normal(size=(12, 12)))[0]
        w = np.linspace(0.4, 1.6, 12)
        gap = np.max(np.abs(quadratic_form_distance_sq(x, l, w, 5) - factorized_distance_sq(x, l, w, 5)))
        self.assertLess(gap, 1e-9)

    def test_orthogonal_full_equal_matches_euclidean_knn(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        d0 = sq_euclidean_rows(x)
        d1 = factorized_distance_sq(x, q, np.ones(8), 8)
        self.assertLess(np.max(np.abs(d0 - d1)), 1e-9)
        self.assertTrue(np.array_equal(knn_indices_stable(d0, 5), knn_indices_stable(d1, 5)))


if __name__ == "__main__":
    unittest.main()
