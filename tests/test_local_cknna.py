"""geoprh.local_cknna against Emily's loop implementation."""

from math import sqrt

import numpy as np
import pytest
import torch

from geoprh import cknna_variants as cv
from geoprh import local_cknna as lc
from geoprh import prh_alignment as pa

M = pa.M  # platonic-rep metrics (same hsic_unbiased as Emily's file)
N = 60


def emily_cknna_local(feats_A, feats_B, topk, min_neighbors=4, per_point=False):
    """Emily's `cknna_local` (branch emily, geometry_aware_convergence/metrics.py#L252-L302),
    unbiased=True, with one fix: the original calls `.item()` on the float returned by
    math.sqrt, which raises AttributeError at the first kept point."""
    n = feats_A.shape[0]
    K, L = feats_A @ feats_A.T, feats_B @ feats_B.T
    K_hat = K.clone().fill_diagonal_(float("-inf"))
    L_hat = L.clone().fill_diagonal_(float("-inf"))
    _, topk_K = torch.topk(K_hat, topk, dim=1)
    _, topk_L = torch.topk(L_hat, topk, dim=1)
    scores = []
    for i in range(n):
        mutual = torch.tensor(
            list(set(topk_K[i].tolist()) & set(topk_L[i].tolist())), dtype=torch.long
        )
        if len(mutual) < min_neighbors:
            scores.append(float("nan"))
            continue
        block = torch.cat([torch.tensor([i]), mutual])
        Kb, Lb = K[block][:, block], L[block][:, block]
        kl, kk, ll = M.hsic_unbiased(Kb, Lb), M.hsic_unbiased(Kb, Kb), M.hsic_unbiased(Lb, Lb)
        scores.append(float(kl / (sqrt(kk * ll) + 1e-6)))  # fixed: no .item() on a float
    scores = np.array(scores)
    return scores if per_point else float(np.nanmean(scores))


def _union_loop(x, y, k):
    n = x.shape[0]
    K, L = x @ x.T, y @ y.T
    tk = torch.topk(K.clone().fill_diagonal_(float("-inf")), k, dim=1).indices
    tl = torch.topk(L.clone().fill_diagonal_(float("-inf")), k, dim=1).indices
    out = []
    for i in range(n):
        block = torch.tensor([i, *sorted(set(tk[i].tolist()) | set(tl[i].tolist()))])
        Kb, Lb = K[block][:, block], L[block][:, block]
        kl, kk, ll = M.hsic_unbiased(Kb, Lb), M.hsic_unbiased(Kb, Kb), M.hsic_unbiased(Lb, Lb)
        out.append(float(kl / (sqrt(kk * ll) + 1e-6)))
    return np.array(out)


@pytest.fixture(scope="module")
def pair():
    g = torch.Generator().manual_seed(0)
    shared = torch.randn(N, 5, generator=g)
    x = shared @ torch.randn(5, 24, generator=g) + torch.randn(N, 24, generator=g) + 2.0
    y = shared @ torch.randn(5, 32, generator=g) + torch.randn(N, 32, generator=g) + 2.0
    return pa.prepare_layers(x[:, None])[0], pa.prepare_layers(y[:, None])[0]


@pytest.mark.parametrize("k", [8, 20, 45, N - 1])
def test_matches_emily_and_union_loop(pair, k):
    x, y = pair
    sx, sy = cv.Stack([x]), cv.Stack([y])
    ax, ay = sx.at_k(k), sy.at_k(k)
    s = lc.local_scores(ax.m[0], ay.m[0], lc.LocalGram(sx.K[0]), lc.LocalGram(sy.K[0]))
    ref = emily_cknna_local(x, y, k, per_point=True)
    np.testing.assert_array_equal(np.isnan(s["mutual"].numpy()), np.isnan(ref))
    np.testing.assert_allclose(s["mutual"].numpy(), ref, rtol=2e-3, atol=1e-4, equal_nan=True)
    red = lc.reduce(s)
    if not np.isnan(ref).all():
        assert float(red["local_mutual"]) == pytest.approx(np.nanmean(ref), rel=2e-3, abs=1e-4)
    np.testing.assert_allclose(s["union"].numpy(), _union_loop(x, y, k), rtol=2e-3, atol=1e-4)


def test_full_k_is_unbiased_cka(pair):
    """At k = n-1 every block is the whole set: each local score is unbiased CKA."""
    x, y = pair
    sx, sy = cv.Stack([x]), cv.Stack([y])
    ax, ay = sx.at_k(N - 1), sy.at_k(N - 1)
    s = lc.local_scores(ax.m[0], ay.m[0], lc.LocalGram(sx.K[0]), lc.LocalGram(sy.K[0]))
    ref = M.AlignmentMetrics.measure("unbiased_cka", x, y)
    np.testing.assert_allclose(s["mutual"].numpy(), ref, rtol=1e-3)
    np.testing.assert_allclose(s["union"].numpy(), ref, rtol=1e-3)


def test_rowwise_sums_to_global(pair):
    """Per-point mutual kNN averages to the global score."""
    x, y = pair
    sx, sy = cv.Stack([x]), cv.Stack([y])
    ax, ay = sx.at_k(10), sy.at_k(10)
    r = lc.rowwise(ax.m[0], ay.m[0], sx.Kc[0], sy.Kc[0], 10)
    glob = cv.per_k_scores(ax, ay, ay.right())
    assert float(r["mknn"].mean()) == pytest.approx(float(glob["mknn"][0, 0]), rel=1e-6)
    assert float(r["centred"].abs().max()) <= 1 + 1e-5
