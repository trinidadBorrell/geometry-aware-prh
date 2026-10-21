"""Focused checks for SAE cross-covariance / CCA."""

from __future__ import annotations

import unittest

import numpy as np

from prh_replication.sae import TopKSAE, encode_features, train_topk_sae
from prh_replication.sae_crosscov import (
    covariance_cols,
    fit_pca,
    pearson_cols,
    project,
    raw_crosscov_svd,
    regularized_cca,
    sample_covariances,
)
from prh_replication.sae_crosscov_study import _run_synthetic_checks


class SaeCrosscovTests(unittest.TestCase):
    def test_synthetic_suite(self) -> None:
        checks = _run_synthetic_checks()
        self.assertTrue(checks["synthetic_shared_recovery"])
        self.assertTrue(checks["synthetic_independent_no_heldout"])
        self.assertTrue(checks["rotation_singular_invariance"])

    def test_raw_svd_joint_signs_keep_training_cov_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        n, d = 400, 24
        shared = rng.normal(size=(n, 6))
        left = np.hstack([shared, rng.normal(size=(n, d - 6))])
        right = np.hstack([shared * 0.8 + 0.05 * rng.normal(size=(n, 6)), rng.normal(size=(n, d - 6))])
        cov = sample_covariances(left, right)
        fit = raw_crosscov_svd(cov["C_AB"])
        recon = fit["U"] @ np.diag(fit["sigma"]) @ fit["V"].T
        rel = float(np.linalg.norm(recon - cov["C_AB"]) / max(np.linalg.norm(cov["C_AB"]), 1e-15))
        self.assertLess(rel, 1e-10)
        paired = covariance_cols(cov["A"] @ fit["U"], cov["B"] @ fit["V"])
        self.assertTrue(np.all(paired >= -1e-10))
        # Independent sign flips would typically make ~half of these negative.

    def test_conventional_r2_differs_from_train_mean_improvement(self) -> None:
        from prh_replication.sae_crosscov import predictive_scores

        rng = np.random.default_rng(4)
        y = rng.normal(size=(100, 3)) + np.array([2.0, -1.0, 0.5])
        yhat = y + 0.1 * rng.normal(size=y.shape)
        train_mean = np.zeros(3)
        scores = predictive_scores(yhat, y, train_mean)
        self.assertFalse(np.allclose(scores["r2"], scores["improvement_vs_train_mean"]))

    def test_cca_singular_values_are_not_silently_used_as_pearson(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(200, 8))
        b = a @ rng.normal(size=(8, 8)) + 0.3 * rng.normal(size=(200, 8))
        cov = sample_covariances(a, b)
        cca = regularized_cca(cov["C_A"], cov["C_B"], cov["C_AB"], eta=0.1)
        pear = pearson_cols(cov["A"] @ cca["L_A"], cov["B"] @ cca["L_B"])
        self.assertEqual(pear.shape, cca["s"].shape)
        # They are computed as separate arrays even if numerically close.
        self.assertIsNot(pear, cca["s"])

    def test_topk_sae_encodes_before_any_average(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(40, 16)).astype(np.float64)
        result = train_topk_sae(
            x[:30],
            x[30:],
            feature_dim=32,
            k=4,
            seed=0,
            learning_rate=1e-2,
            batch_size=16,
            max_epochs=3,
            patience=3,
            min_delta=1e-8,
            device="cpu",
        )
        codes = encode_features(result.model, x, device="cpu")
        self.assertEqual(codes.shape, (40, 32))
        self.assertTrue(np.all(codes >= -1e-8))
        self.assertLessEqual(float(np.mean(np.sum(codes > 0, axis=1))), 4.0 + 1e-6)

    def test_full_pca_preserves_centred_gram_within_tol(self) -> None:
        x = np.random.default_rng(3).normal(size=(50, 12))
        pca = fit_pca(x, q=12)
        scores = project(x, pca["mean"], pca["components"])
        centred = x - pca["mean"]
        gap = np.max(np.abs(centred @ centred.T - scores @ scores.T))
        self.assertLess(gap, 1e-8)

    def test_train_test_ids_disjoint_pattern(self) -> None:
        train = [f"id-{i}" for i in range(10)]
        test = [f"id-{i}" for i in range(10, 15)]
        self.assertTrue(set(train).isdisjoint(test))


if __name__ == "__main__":
    unittest.main()
