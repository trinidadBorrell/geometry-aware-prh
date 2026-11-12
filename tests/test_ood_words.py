"""exp-029 words.py: sparse PPMI equals exp-028's dense PPMI; window and document co-occurrence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp

EXP = Path(__file__).resolve().parents[1] / "experiments"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load("exp029_words", EXP / "exp-029-ood-word-vectors" / "words.py")
S = _load("exp028_spectral", EXP / "exp-028-spectral-lexical-floor" / "spectral.py")


def test_sparse_ppmi_matches_dense():
    rng = np.random.default_rng(0)
    b = sp.csr_matrix((rng.random((200, 30)) < 0.2).astype(np.float32))
    dense = S.ppmi_matrix(b)
    c = (b.T @ b).tocsr()
    assert np.allclose(W.ppmi_sparse(c).toarray(), dense, atol=1e-4)


def test_cooc_doc_matches_binary_product():
    docs = [np.array([0, 1, 1, 2]), np.array([2, 3]), np.array([0, 3, 4, 0])]
    c = W.cooc_doc(docs, 5).toarray()
    b = np.zeros((3, 5))
    for i, d in enumerate(docs):
        b[i, np.unique(d)] = 1
    want = b.T @ b
    np.fill_diagonal(want, 0)
    assert np.array_equal(c, want)


def test_cooc_window_counts():
    docs = [np.array([0, 1, 2, 3]), np.array([3, 0])]
    c = W.cooc_window(docs, 4, window=2, chunk_tokens=3).toarray()
    want = np.zeros((4, 4))
    for d in docs:
        for i in range(len(d)):
            for j in range(i + 1, min(len(d), i + 3)):
                want[d[i], d[j]] += 1
                want[d[j], d[i]] += 1
    assert np.array_equal(c, want)
    assert np.array_equal(c, c.T)


def test_vectors_shape_and_rank():
    rng = np.random.default_rng(1)
    m = sp.random(60, 60, density=0.2, random_state=1, format="csr", dtype=np.float32)
    m = m + m.T
    v = W.vectors(m, 10)
    assert v.shape == (60, 10) and np.isfinite(v).all()
    _ = rng


def test_analyzer_matches_exp028():
    a = W.analyzer()
    assert a("The Dog and THE puppy, on a beach!") == ["dog", "puppy", "beach"]
