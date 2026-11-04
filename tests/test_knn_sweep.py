import torch
import torch.nn.functional as F

from geoprh import knn_sweep as ks
from geoprh import layer_calibration as lc


def _feats(seed, n=300, layers=3, d=24, shared=None):
    g = torch.Generator().manual_seed(seed)
    base = shared if shared is not None else torch.randn(n, 8, generator=g)
    out = []
    for _ in range(layers):
        w = torch.randn(base.shape[1], d, generator=g)
        x = torch.tanh(base @ w) + 0.5 * torch.randn(n, d, generator=g)
        out.append(F.normalize(x, dim=-1))
    return out


def test_matches_layer_calibration_at_k10_with_and_without_permutation():
    shared = torch.randn(300, 8, generator=torch.Generator().manual_seed(0))
    xa, xb = _feats(1, shared=shared), _feats(2, shared=shared)
    a, b = ks.build_sweep_cache(xa, 40), ks.build_sweep_cache(xb, 40)
    ca, cb = lc.build_cache(torch.stack(xa, 1), 10), lc.build_cache(torch.stack(xb, 1), 10)
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(3))
    for p in (None, perm):
        ref = lc.scores(ca, cb, p)["mutual_knn"]
        got = ks.scores(a, b, [10], p)[0]
        assert torch.allclose(ref, got, atol=1e-6)


def test_other_k_against_dense_masks():
    xa, xb = _feats(4), _feats(5)
    a, b = ks.build_sweep_cache(xa, 60), ks.build_sweep_cache(xb, 60)
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(6))
    for k in (1, 25, 60):
        ma = lc.build_cache(torch.stack(xa, 1), k)
        mb = lc.build_cache(torch.stack(xb, 1), k)
        ref = lc.scores(ma, mb, perm)["mutual_knn"]
        assert torch.allclose(ref, ks.scores(a, b, [k], perm)[0], atol=1e-6)


def test_identical_models_score_one_and_null_is_small():
    xa = _feats(7)
    a = ks.build_sweep_cache(xa, 30)
    res = ks.calibrated_sweep(a, a, [10, 30], permutations=20, seed=0)
    for k in (10, 30):
        diag = torch.diagonal(res[k]["score"])
        assert torch.allclose(diag, torch.ones_like(diag))
        assert res[k]["mu0"].max() < 3 * k / 300
        assert res[k]["max_p"] <= 1 / 21 + 1e-9
