"""exp-005's k sweep: CKNNA as published (Huh et al. Eqs. 16-18), mKNN and CKA as upstream."""

import importlib.util
from pathlib import Path

import pytest
import torch

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


def _paper_cknna(x, y, k):
    """Eqs. 12, 16-18 written as literally as possible."""
    n = x.shape[0]
    K, L = x @ x.T, y @ y.T
    Kbar, Lbar = K - K.mean(1, keepdim=True), L - L.mean(1, keepdim=True)

    def knn(G):
        g = G.clone().fill_diagonal_(float("-inf"))
        return [set(torch.topk(g[i], k).indices.tolist()) for i in range(n)]

    nk, nl = knn(K), knn(L)

    def align(A, B, na, nb):
        return sum(float(A[i, j] * B[i, j]) for i in range(n) for j in na[i] & nb[i] if i != j)

    return (
        align(Kbar, Lbar, nk, nl) / (align(Kbar, Kbar, nk, nk) * align(Lbar, Lbar, nl, nl)) ** 0.5
    )


@pytest.fixture(scope="module")
def pair():
    n = 60
    shared = torch.randn(n, 6, generator=torch.Generator().manual_seed(0))
    fa, fb = _feats(n, 3, 32, shared, 1), _feats(n, 2, 48, shared, 2)
    return sweep.Model(fa), sweep.Model(fb), pa.prepare_layers(fa), pa.prepare_layers(fb)


@pytest.mark.parametrize("k", [5, 20, 45, 59])
def test_cknna_matches_paper_equations(pair, k):
    va, lb, xa, xb = pair
    mknn, cknna = sweep.scores(va, lb, k)
    for i in range(3):
        for j in range(2):
            assert float(cknna[i, j]) == pytest.approx(_paper_cknna(xa[i], xb[j], k), rel=1e-4)
            if k < 59:
                ref = float(M.AlignmentMetrics.measure("mutual_knn", xa[i], xb[j], topk=k))
                assert float(mknn[i, j]) == pytest.approx(ref, rel=1e-5)
    assert float(cknna.max()) <= 1 + 1e-5


def test_cka_matches_upstream(pair):
    va, lb, xa, xb = pair
    cka = sweep.linear_cka(va, lb)
    ref = float(M.AlignmentMetrics.measure("cka", xa[1], xb[0]))
    assert float(cka[1, 0]) == pytest.approx(ref, rel=1e-4)
