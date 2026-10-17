import torch

from aristotelian.experiments.layerwise_engine import (
    _cka_from_grams,
    build_gram_cache,
    build_knn_cache,
)
from calibrated_similarity import calibrate
from geoprh import layer_calibration as lc


def _feats(seed, n=60, layers=3, d=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, layers, d, generator=g)


def test_scores_match_aristotelian_cells():
    fa, fb = _feats(0), _feats(1)
    a, b = lc.build_cache(fa, k=5), lc.build_cache(fb, k=5)
    s = lc.scores(a, b)
    _, ma = build_knn_cache(fa, topk=5)
    _, mb = build_knn_cache(fb, topk=5)
    ga, gb = build_gram_cache(fa), build_gram_cache(fb)
    for i in range(3):
        for j in range(3):
            mknn = ((ma[i] & mb[j]).sum(dim=1).float() / 5).mean()
            assert torch.isclose(s["mutual_knn"][i, j], mknn, atol=1e-6)
            cka = _cka_from_grams(ga[i].double(), gb[j].double())
            assert abs(float(s["cka_lin"][i, j]) - cka) < 1e-4


def test_cells_match_algorithm_1():
    fa, fb = _feats(2), _feats(3)
    fb[:, 1] = fa[:, 0] + 0.3 * fb[:, 1]  # one aligned layer pair
    a, b = lc.build_cache(fa, k=5), lc.build_cache(fb, k=5)
    perms = [torch.randperm(60, generator=torch.Generator().manual_seed(s)) for s in range(50)]
    observed = lc.scores(a, b)
    nulls = [lc.scores(a, b, p) for p in perms]
    res = lc.calibrate_cells(observed["cka_lin"], torch.stack([n["cka_lin"] for n in nulls]))
    assert res["g"][0, 1] > 0.3

    xa = torch.nn.functional.normalize(fa[:, 0], dim=-1)
    xb = torch.nn.functional.normalize(fb[:, 1], dim=-1)
    it = iter(perms)

    def cka(x, y):
        kx, ky = x @ x.T, y @ y.T
        return _cka_from_grams(kx.double(), ky.double())

    g, p, tau = calibrate(xa, xb, cka, K=50, perm_fn=lambda n, dev: next(it))
    assert abs(float(res["g"][0, 1]) - float(g)) < 1e-4
    assert abs(float(res["p"][0, 1]) - float(p)) < 1e-6
    assert abs(float(res["tau"][0, 1]) - float(tau)) < 1e-4
