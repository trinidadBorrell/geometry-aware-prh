import numpy as np
import scipy.sparse as sp
import torch

from geoprh import layer_calibration as lc
from geoprh import lexical as lex


def _dense(m):
    return m.toarray() if sp.issparse(m) else np.asarray(m)


def test_count_tf_binary_on_toy_captions():
    counts, lengths, vocab = lex.count_matrix([[5, 5, 7], [7, 9]])
    assert vocab.tolist() == [5, 7, 9]
    assert lengths.tolist() == [3, 2]
    assert _dense(counts).tolist() == [[2, 1, 0], [0, 1, 1]]
    assert np.allclose(_dense(lex.tf(counts)), [[2 / 3, 1 / 3, 0], [0, 0.5, 0.5]])
    assert _dense(lex.binary(counts)).tolist() == [[1, 1, 0], [0, 1, 1]]
    assert np.allclose(np.asarray(lex.tf(counts).sum(1)).ravel(), 1)


def test_empty_caption_raises():
    try:
        lex.count_matrix([[1], []])
    except ValueError:
        return
    raise AssertionError("an empty caption must raise")


def test_tfidf_matches_scikit_learn_convention():
    from sklearn.feature_extraction.text import TfidfTransformer

    counts, _, _ = lex.count_matrix([[0, 0, 1], [1, 2], [2, 3, 3, 4]])
    ours = lex.to_unit_rows(lex.tfidf(counts)).numpy()
    theirs = TfidfTransformer(norm="l2", smooth_idf=True).fit_transform(counts).toarray()
    assert np.allclose(ours, theirs, atol=1e-6)


def test_tf_times_table_is_the_mean_of_token_vectors():
    ids = [[0, 2, 2, 3], [1, 3]]
    table = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    counts, _, vocab = lex.count_matrix(ids)
    manual = np.stack([table[x].mean(0) for x in ids])
    assert np.allclose(lex.tf(counts) @ table[vocab], manual, atol=1e-6)


def test_token_order_does_not_change_the_kernel():
    rng = np.random.default_rng(0)
    ids = [rng.integers(0, 50, 30).tolist() for _ in range(20)]
    shuffled = [rng.permutation(x).tolist() for x in ids]
    a = lex.to_unit_rows(lex.tf(lex.count_matrix(ids)[0]))
    b = lex.to_unit_rows(lex.tf(lex.count_matrix(shuffled)[0]))
    assert torch.allclose(a @ a.T, b @ b.T)


def test_truncate_modes():
    ids = [list(range(10)), list(range(4))]
    assert lex.truncate(ids, 3) == [[0, 1, 2], [0, 1, 2]]
    assert lex.truncate(ids, [2, 9]) == [[0, 1], [0, 1, 2, 3]]
    sampled = lex.truncate(ids, 3, "random", seed=1)
    assert [len(x) for x in sampled] == [3, 3]
    assert all(x == sorted(x) and set(x) <= set(y) for x, y in zip(sampled, ids, strict=True))


def test_mask_top_removes_most_frequent_tokens():
    counts, _, vocab = lex.count_matrix([[0, 0, 0, 1], [0, 2]])
    assert vocab.tolist() == [0, 1, 2]
    assert _dense(lex.mask_top(counts, 1)).tolist() == [[1, 0], [0, 1]]


def test_feature_and_kernel_cka_agree():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 16, generator=g).double()
    y = x @ torch.randn(16, 8, generator=g).double() + torch.randn(200, 8, generator=g).double()
    gx, gy = lex.centred_unit_gram(x @ x.T), lex.centred_unit_gram(y @ y.T)
    assert abs(float((gx * gy).sum()) - lex.cka_features(x, y)) < 1e-10


def test_debiased_cka_is_near_zero_for_independent_kernels():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 300, generator=g)
    y = torch.randn(1000, 300, generator=g)
    biased = float((lex.centred_unit_gram(x @ x.T) * lex.centred_unit_gram(y @ y.T)).sum())
    debiased = lex.HsicU(x @ x.T).cka(lex.HsicU(y @ y.T))
    assert biased > 0.2
    assert abs(debiased) < 0.02


def test_debiased_cka_of_a_kernel_with_itself_is_one():
    x = torch.randn(300, 20, generator=torch.Generator().manual_seed(0))
    u = lex.HsicU(x @ x.T)
    assert abs(u.cka(u) - 1) < 1e-9


def test_analytic_permutation_null_of_centred_kernels():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(150, 10, generator=g), torch.randn(150, 10, generator=g)
    a, b = lex.centred_unit_gram(x @ x.T), lex.centred_unit_gram(y @ y.T)
    perms = [torch.randperm(150, generator=g) for _ in range(3000)]
    empirical = np.mean([float((a * b[p][:, p]).sum()) for p in perms])
    assert abs(empirical - float(a.trace() * b.trace()) / 149) < 2e-3


def test_build_cache_cka_matches_exact_value_at_large_n():
    """float32 Gram norms drift on the CPU; at n = 4096 the error was ~0.14%."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4096, 32, generator=g)
    x = z @ torch.randn(32, 128, generator=g) + 0.5 * torch.randn(4096, 128, generator=g)
    y = z @ torch.randn(32, 128, generator=g) + 0.5 * torch.randn(4096, 128, generator=g)
    x, y = (torch.nn.functional.normalize(v, dim=-1) for v in (x, y))
    got = lc.scores(lc.build_cache(x[:, None], k=5), lc.build_cache(y[:, None], k=5))["cka_lin"]
    assert abs(float(got) - lex.cka_features(x, y)) < 2e-4
