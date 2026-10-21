"""Focused checks for feature vs PCA metric comparison."""

from __future__ import annotations

import unittest

import numpy as np

from prh_replication.feature_vs_pca_metric_study import _run_checks


class FeatureVsPcaMetricTests(unittest.TestCase):
    def test_verification_suite(self) -> None:
        checks = _run_checks()
        for key, value in checks.items():
            if key == "notes":
                continue
            self.assertTrue(value, msg=f"{key}={value}")


if __name__ == "__main__":
    unittest.main()
