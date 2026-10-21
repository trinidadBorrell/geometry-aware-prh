"""Focused checks for CCA metric PCA-axis diagonal ablation."""

from __future__ import annotations

import unittest

import numpy as np

from prh_replication.cca_metric_diagonal_ablation import (
    build_full_metric,
    check_psd,
    diagonal_ablation,
    metric_factor_full,
    weighted_gram,
)
from prh_replication.cca_metric_diagonal_ablation_study import _run_checks


class CcaDiagonalAblationTests(unittest.TestCase):
    def test_verification_suite(self) -> None:
        checks = _run_checks()
        for key, value in checks.items():
            if key == "notes":
                continue
            self.assertTrue(value, msg=f"{key}={value}")

    def test_trace_and_offdiag(self) -> None:
        rng = np.random.default_rng(2)
        l = rng.normal(size=(8, 8))
        w = np.linspace(0.5, 2.0, 8)
        m = build_full_metric(l, w)
        d = diagonal_ablation(m)
        self.assertAlmostEqual(float(np.trace(m)), float(np.trace(d)), places=8)
        self.assertTrue(check_psd(m)["psd_ok"])
        off = m - d
        self.assertGreater(float(np.sum(off * off)), 0.0)

    def test_factor_reconstructs_metric(self) -> None:
        rng = np.random.default_rng(3)
        l = rng.normal(size=(6, 6))
        w = np.abs(rng.normal(size=6)) + 0.1
        m = build_full_metric(l, w)
        f = metric_factor_full(l, w)
        self.assertLess(float(np.max(np.abs(m - 0.5 * (f @ f.T + (f @ f.T).T)))), 1e-10)

    def test_gram_matches_features(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.normal(size=(40, 7))
        l = rng.normal(size=(7, 7))
        w = np.ones(7)
        m = build_full_metric(l, w)
        f = metric_factor_full(l, w)
        self.assertLess(float(np.max(np.abs(weighted_gram(a, m) - (a @ f) @ (a @ f).T))), 1e-9)


if __name__ == "__main__":
    unittest.main()
