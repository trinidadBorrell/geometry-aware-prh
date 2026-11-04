import torch

from geoprh import layer_calibration as lc
from geoprh import truncation as tr


def _pair(shared_sd, private_sd, seed=0, n=400, s=6, p=18, d=64, layers=2):
    shared = torch.randn(n, s, generator=torch.Generator().manual_seed(seed))
    a = tr.synthetic_model(shared, p, shared_sd, private_sd, d, layers, seed=seed + 1)
    b = tr.synthetic_model(shared, p, shared_sd, private_sd, d, layers, seed=seed + 2)
    return tr.Model(a), tr.Model(b)


def _grams(layers):
    return [x @ x.T for x in layers]


def test_full_size_operators_give_the_untruncated_geometry():
    a, _ = _pair(2.0, 1.0)
    ref = _grams(a.truncate("none", 0))
    gen = torch.Generator().manual_seed(0)
    for op, m in (("top", a.rank), ("drop_top", 0), ("rand_coord", a.d), ("rand_proj", a.d)):
        got = _grams(a.truncate(op, m, gen))
        for r, g in zip(ref, got, strict=False):
            assert torch.allclose(r, g, atol=1e-4), op


def test_top_and_drop_top_split_the_spectrum():
    a, _ = _pair(2.0, 1.0)
    top = a.truncate("top", 5)
    rest = a.truncate("drop_top", 5)
    assert top[0].shape[1] == 5 and rest[0].shape[1] == a.rank - 5
    assert a.size("top", 5) == 5 and a.size("drop_top", 5) == a.rank - 5


def test_batched_scores_match_layer_calibration():
    a, b = _pair(2.0, 1.0)
    ca, cb = tr.build_cache(a.truncate("none", 0)), tr.build_cache(b.truncate("none", 0))
    perms = torch.stack(
        [torch.randperm(a.n, generator=torch.Generator().manual_seed(i)) for i in range(3)]
    )
    ca, cb = ca.dense, cb.dense
    got = tr.batched_scores(ca, cb, perms)
    for i, p in enumerate(perms):
        ref = lc.scores(ca, cb, p)
        for m in ref:
            assert torch.allclose(ref[m], got[m][i], atol=1e-5)
    ref = lc.scores(ca, cb)
    got = tr.batched_scores(ca, cb, None)
    for m in ref:
        assert torch.allclose(ref[m], got[m][0], atol=1e-5)


def _excess(a, b, op, m, perms):
    ca = tr.build_cache(a.truncate(op, m, torch.Generator().manual_seed(0)))
    cb = tr.build_cache(b.truncate(op, m, torch.Generator().manual_seed(0)))
    r = tr.calibrated_grid(ca, cb, perms)["mutual_knn"]
    return r["max_score"] - r["max_mu0"]


def test_truncation_locates_where_the_shared_content_is():
    perms = torch.stack(
        [torch.randperm(400, generator=torch.Generator().manual_seed(i)) for i in range(20)]
    )
    # shared content in the high-variance band: keeping the top 6 PCs removes private variance
    a, b = _pair(3.0, 1.0)
    full = _excess(a, b, "none", 0, perms)
    assert _excess(a, b, "top", 6, perms) > full + 0.1
    assert _excess(a, b, "drop_top", 6, perms) < 0.05
    # shared content in the low-variance band: removing the top 18 PCs removes private variance
    a, b = _pair(1.0, 3.0, seed=10)
    full = _excess(a, b, "none", 0, perms)
    assert _excess(a, b, "drop_top", 18, perms) > full + 0.1
    assert _excess(a, b, "top", 6, perms) < 0.05


def test_self_retention_is_one_at_full_size():
    a, _ = _pair(2.0, 1.0)
    full = tr.build_cache(a.truncate("none", 0))
    r = tr.self_retention(tr.build_cache(a.truncate("top", a.rank)), full)
    assert torch.allclose(r["mutual_knn"], torch.ones(2), atol=1e-6)
    assert torch.allclose(r["cka_lin"], torch.ones(2), atol=1e-4)


def test_calibrated_pairs_matches_dense_permutation_scores():
    a, b = _pair(2.0, 1.0)
    c, _ = _pair(1.0, 2.0, seed=5)
    caches = {k: tr.build_cache(m.truncate("top", 10)) for k, m in (("a", a), ("b", b), ("c", c))}
    perms = torch.stack(
        [torch.randperm(a.n, generator=torch.Generator().manual_seed(i)) for i in range(7)]
    )
    got = tr.calibrated_pairs(caches, [("a", "b"), ("c", "b"), ("a", "c")], perms, chunk=3)
    for x, y in (("a", "b"), ("c", "b"), ("a", "c")):
        ref = tr.batched_scores(caches[x].dense, caches[y].dense, perms)
        obs = tr.batched_scores(caches[x].dense, caches[y].dense, None)
        for m in ("mutual_knn", "cka_lin"):
            r = tr.lc.calibrate_cells(obs[m][0], ref[m])
            assert torch.allclose(got[(x, y)][m]["score"], r["score"], atol=1e-5)
            assert torch.allclose(got[(x, y)][m]["mu0"], r["mu0"], atol=1e-5)
            assert torch.allclose(got[(x, y)][m]["tau"], r["tau"], atol=1e-5)
