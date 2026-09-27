"""exp-005's batched k-sweep must reproduce platonic-rep's per-pair metrics."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from geoprh import prh_alignment as pa

M = pa.M
_spec = importlib.util.spec_from_file_location(
    "cknna_k_sweep",
    Path(__file__).resolve().parents[1] / "experiments/exp-005-cknna-large-k/cknna_k_sweep.py",
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def _feats(n, layers, d, shared, seed):
    g = torch.Generator().manual_seed(seed)
    out = [
        shared @ torch.randn(shared.shape[1], d, generator=g) + torch.randn(n, d, generator=g)
        for _ in range(layers)
    ]
    return torch.stack(out, dim=1) + 2.0  # offset -> anisotropic, like real features


@pytest.mark.parametrize("k", [5, 40, 90, 119])
def test_batched_sweep_matches_upstream(k):
    n = 120
    shared = torch.randn(n, 6, generator=torch.Generator().manual_seed(0))
    va = sweep.Model(_feats(n, 3, 32, shared, 1))
    lb = sweep.Model(_feats(n, 2, 48, shared, 2))
    xa = pa.prepare_layers(_feats(n, 3, 32, shared, 1))
    xb = pa.prepare_layers(_feats(n, 2, 48, shared, 2))

    ma, mb = va.masks(k), lb.masks(k)
    num = sweep.hsic_pairs(ma * va.K, mb, ma, mb * lb.K)
    cknna = sweep._ratio(num, sweep.self_hsic(ma, va.K), sweep.self_hsic(mb, lb.K))
    mknn = (ma.reshape(3, -1) @ mb.reshape(2, -1).T) / (n * k)
    for i in range(3):
        for j in range(2):
            ref = float(M.AlignmentMetrics.measure("cknna", xa[i], xb[j], topk=k))
            assert float(cknna[i, j]) == pytest.approx(ref, rel=1e-4, abs=1e-5)
            if k < n - 1:
                ref = float(M.AlignmentMetrics.measure("mutual_knn", xa[i], xb[j], topk=k))
                assert float(mknn[i, j]) == pytest.approx(ref, rel=1e-5)
            ref = float(M.AlignmentMetrics.measure("unbiased_cka", xa[i], xb[j]))
            if k == n - 1:
                assert float(cknna[i, j]) == pytest.approx(ref, rel=1e-4, abs=1e-5)


def test_hsic_pairs_matches_scalar():
    g = torch.Generator().manual_seed(3)
    a = torch.rand(2, 30, 30, generator=g)
    b = torch.rand(3, 30, 30, generator=g)
    for x in (a, b):
        x.diagonal(dim1=1, dim2=2).zero_()
    got = sweep.hsic_pairs(a, b, a.flip(0), b.flip(0))
    for i in range(2):
        for j in range(3):
            want = sweep.hsic_u(a[i] * b[j], a.flip(0)[i] * b.flip(0)[j])
            assert float(got[i, j]) == pytest.approx(float(want), rel=1e-4, abs=1e-7)


def test_cka_matches_upstream():
    x = F.normalize(torch.randn(80, 16), dim=-1)
    y = F.normalize(torch.randn(80, 24), dim=-1)
    va, lb = sweep.Model(x[:, None]), sweep.Model(y[:, None])
    got = (va.Kc.reshape(1, -1) @ lb.Kc.reshape(1, -1).T) / (
        torch.sqrt(va.hsic_b[:, None] * lb.hsic_b) + 1e-6
    )
    xa, yb = pa.prepare_layers(x[:, None])[0], pa.prepare_layers(y[:, None])[0]
    ref = float(M.AlignmentMetrics.measure("cka", xa, yb))
    assert float(got[0, 0]) == pytest.approx(ref, rel=1e-4)
