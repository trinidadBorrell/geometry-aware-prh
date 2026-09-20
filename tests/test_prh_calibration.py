"""The mask-based mutual-kNN calibration must match the index-based one used in exp-003."""

import pytest
import torch
import torch.nn.functional as F

from aristotelian.experiments.layerwise_engine import (
    build_knn_cache,
    build_knn_cache_with_indices,
    compute_alignment_gated_cached,
    compute_alignment_gated_knn_cached,
)
from geoprh.prh_calibration import Spec, parse_specs


def _feats(n=96, layers=3, d=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(n, 6, generator=g)
    return torch.stack(
        [
            F.normalize(
                shared @ torch.randn(6, d, generator=g) + torch.randn(n, d, generator=g), dim=-1
            )
            for _ in range(layers)
        ],
        dim=1,
    )


def test_mask_and_index_knn_calibration_agree():
    x, y = _feats(seed=0), _feats(seed=1)
    topk, perms = 10, 25
    x_layers, x_masks = build_knn_cache(x, topk=topk, normalize=True)
    y_layers, y_masks = build_knn_cache(y, topk=topk, normalize=True)
    mask_res = compute_alignment_gated_cached(
        x_layers, y_layers, x_masks, y_masks, topk=topk, num_permutations=perms, seed=0
    )
    _, x_knn, _ = build_knn_cache_with_indices(x, topk=topk, normalize=True)
    _, y_knn, _ = build_knn_cache_with_indices(y, topk=topk, normalize=True)
    idx_res = compute_alignment_gated_knn_cached(
        x_knn, y_knn, topk=topk, num_permutations=perms, seed=0
    )
    # the two differ only in float rounding: mask divides by k before averaging, index after`n    assert mask_res["raw_score"] == pytest.approx(idx_res["raw_score"], rel=1e-6)
    assert mask_res["best_indices"] == idx_res["best_indices"]
    # same seed, same permutations -> same null
    assert mask_res["tau_alpha"] == pytest.approx(idx_res["tau_alpha"], rel=1e-6)
    assert mask_res["g_score"] == pytest.approx(idx_res["g_score"], rel=1e-6)


def test_parse_specs_expands_k_for_knn_metrics_only():
    specs = parse_specs("mutual_knn,cknna,cka_lin", "10,200")
    assert specs == [
        Spec("mutual_knn", 10),
        Spec("mutual_knn", 200),
        Spec("cknna", 10),
        Spec("cknna", 200),
        Spec("cka_lin", None),
    ]
    # upstream's default filename is kept for mutual kNN at k=10
    assert specs[0].filename == "prh_alignment.npy"
    assert specs[1].filename == "prh_alignment_mutual_knn_k200.npy"
    assert specs[3].filename == "prh_alignment_cknna_k200.npy"
