"""The cached CPU alignment must reproduce platonic-rep's AlignmentMetrics."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geoprh import prh_alignment as pa

M = pa.M


def _layers(n=160, d=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(n, 8, generator=g)
    a = F.normalize(
        shared @ torch.randn(8, d, generator=g) + torch.randn(n, d, generator=g), dim=-1
    )
    b = F.normalize(
        shared @ torch.randn(8, d + 16, generator=g) + torch.randn(n, d + 16, generator=g), dim=-1
    )
    return a, b


@pytest.mark.parametrize(
    ("name", "upstream", "kwargs"),
    [
        ("mutual_knn", "mutual_knn", {"topk": 10}),
        ("cycle_knn", "cycle_knn", {"topk": 10}),
        ("lcs_knn", "lcs_knn", {"topk": 10}),
        ("edit_distance_knn", "edit_distance_knn", {"topk": 10}),
        ("cka", "cka", {}),
        ("unbiased_cka", "unbiased_cka", {}),
        ("cknna_k10", "cknna", {"topk": 10}),
        ("cknna_k50", "cknna", {"topk": 50}),
    ],
)
def test_metric_matches_upstream(name, upstream, kwargs):
    x, y = _layers()
    ours = pa.CROSS_METRICS[name](pa.Layer(x, 100), pa.Layer(y, 100))
    ref = float(M.AlignmentMetrics.measure(upstream, x, y, **kwargs))
    assert ours == pytest.approx(ref, rel=1e-4, abs=1e-5)


def test_svcca_matches_upstream_with_same_seed():
    x, y = _layers()
    torch.manual_seed(0)
    np.random.seed(0)  # noqa: NPY002 -- upstream svcca uses the global numpy RNG
    ref = float(M.AlignmentMetrics.svcca(x, y, cca_dim=10))
    assert pa.svcca(pa.Layer(x, 20), pa.Layer(y, 20)) == pytest.approx(ref, rel=1e-5)


def test_cknna_large_k_matches_upstream():
    x, y = _layers(n=64)
    ours = pa.cknna(pa.Layer(x, 63), pa.Layer(y, 63), 60)
    ref = M.AlignmentMetrics.cknna(x, y, topk=60)
    assert ours == pytest.approx(ref, rel=1e-4, abs=1e-5)


def test_best_over_layers_matches_upstream_compute_score():
    """End to end: outlier clamp, normalisation, max over layer pairs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "platonic_measure", pa.REPO / "platonic-rep" / "measure_alignment.py"
    )
    measure = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(pa.REPO / "platonic-rep"))
    try:
        spec.loader.exec_module(measure)
    finally:
        sys.path.pop(0)

    g = torch.Generator().manual_seed(1)
    vis = torch.randn(96, 3, 32, generator=g)
    lang = vis[:, :1, :].repeat(1, 4, 1) + 0.5 * torch.randn(96, 4, 32, generator=g)
    q_vis = M.remove_outliers(vis, q=0.95, exact=False)
    q_lang = M.remove_outliers(lang, q=0.95, exact=False)
    for metric in ("mutual_knn", "cka"):
        kwargs = {"topk": 10} if "knn" in metric else {}
        ref_score, ref_idx = measure.compute_score(q_vis, q_lang, metric=metric, topk=10)
        ours, idx = pa.best_over_layers(
            [pa.Layer(x, 50) for x in pa.prepare_layers(vis)],
            [pa.Layer(x, 50) for x in pa.prepare_layers(lang)],
            pa.CROSS_METRICS[metric],
        )
        assert ours == pytest.approx(ref_score, rel=1e-4), (metric, kwargs)
        assert tuple(idx) == tuple(ref_idx)
