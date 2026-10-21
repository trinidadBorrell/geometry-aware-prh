"""Focused checks for caption-reliability weights. No alignment scores."""

from __future__ import annotations

import unittest

import numpy as np

from prh_replication.caption_reliability import (
    apply_scales,
    centred_dot_gap,
    feature_scales,
    fit_pca,
    project,
    regroup,
    variance_components,
    weighted_dot,
)


class VarianceTests(unittest.TestCase):
    def test_within_image_denominator_and_decomposition(self) -> None:
        scores = np.array([[[0.0], [2.0]], [[0.0], [2.0]]], dtype=np.float64)
        parts = variance_components(scores)
        self.assertAlmostEqual(float(parts["W"][0]), 2.0)
        self.assertAlmostEqual(float(parts["Vmean"][0]), 0.0)
        self.assertAlmostEqual(float(parts["B_unclipped"][0]), -1.0)
        self.assertEqual(float(parts["B"][0]), 0.0)
        self.assertEqual(float(parts["w"][0]), 0.0)
        self.assertLess(float(parts["decomposition_abs"][0]), 1e-12)

    def test_random_decomposition_is_exact(self) -> None:
        scores = np.random.default_rng(0).normal(size=(12, 5, 4))
        parts = variance_components(scores)
        self.assertLess(float(np.max(parts["decomposition_abs"])), 1e-8)

    def test_synthetic_signal_plus_noise(self) -> None:
        rng = np.random.default_rng(1)
        n_images, n_captions = 4000, 5
        signal_scale = np.array([2.0, 0.5, 0.0])
        noise_scale = np.array([0.5, 0.5, 1.0])
        signal = rng.normal(size=(n_images, 1, 3)) * signal_scale
        noise = rng.normal(size=(n_images, n_captions, 3)) * noise_scale
        parts = variance_components(signal + noise)
        expected = signal_scale**2 / (signal_scale**2 + noise_scale**2)
        self.assertTrue(np.allclose(parts["w"], expected, atol=0.02))
        self.assertLess(float(parts["w"][2]), 0.05)
        self.assertLess(float(np.max(parts["decomposition_abs"])), 1e-6)

    def test_sqrt_weight_scales_the_dot_product(self) -> None:
        reliability = np.array([0.25, 0.0, 1.0])
        left = np.array([1.0, 2.0, -1.0])
        right = np.array([0.5, 4.0, 3.0])
        scale = feature_scales("RELIABILITY_SQRT", reliability, np.ones(3), tau=1e-3)
        scaled_left = apply_scales(left.reshape(1, -1), scale)[0]
        scaled_right = apply_scales(right.reshape(1, -1), scale)[0]
        self.assertAlmostEqual(float(np.dot(scaled_left, scaled_right)), weighted_dot(left, right, reliability))

    def test_full_pca_preserves_centred_dots(self) -> None:
        prepared = np.random.default_rng(2).normal(size=(30, 8))
        basis = fit_pca(prepared, rank_tol=1e-8)
        self.assertEqual(int(basis["width"]), 8)
        self.assertLess(centred_dot_gap(prepared, basis["mean"], basis["components"]), 1e-8)

    def test_test_rows_do_not_change_training_weights(self) -> None:
        train = np.random.default_rng(3).normal(size=(10, 5, 3))
        before = variance_components(train)["w"].copy()
        test = np.random.default_rng(4).normal(size=train.shape)
        after = variance_components(train)["w"]
        self.assertTrue(np.array_equal(before, after))
        self.assertFalse(np.allclose(variance_components(test)["w"], before))

    def test_one_filter_is_reused(self) -> None:
        reliability = np.array([0.2, 0.8])
        variance = np.array([1.0, 2.0])
        scale = feature_scales("RELIABILITY_SQRT", reliability, variance, tau=1e-3)
        partners = {"dinov2-small": scale, "vit-in21k-small": scale}
        self.assertIs(partners["dinov2-small"], partners["vit-in21k-small"])

    def test_disjoint_image_ids(self) -> None:
        train = ["coco-val2017-1", "coco-val2017-2"]
        test = ["coco-val2017-3"]
        self.assertTrue(set(train).isdisjoint(test))

    def test_regroup_keeps_the_caption_pool(self) -> None:
        scores = np.arange(30, dtype=np.float64).reshape(3, 5, 2)
        shuffled = regroup(scores, seed=11)
        self.assertEqual(shuffled.shape, scores.shape)
        self.assertTrue(np.array_equal(np.sort(scores.reshape(-1)), np.sort(shuffled.reshape(-1))))


if __name__ == "__main__":
    unittest.main()
