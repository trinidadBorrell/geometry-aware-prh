"""exp-029: word co-occurrence from token-id documents, sparse PPMI, word vectors.

The analyzer is scikit-learn's word analyzer (lowercase, token pattern \\b\\w\\w+\\b, English stopwords
removed), the same one exp-028 used for its captions. OOV words are dropped before windowing.

- `cooc_window`: symmetric counts of word pairs within +-`window` positions in one document.
- `cooc_doc`: number of documents containing both words (exp-028's within-caption co-occurrence).
- `ppmi_sparse`: exp-028's `ppmi_matrix` on a sparse matrix (tested to be identical).
- `vectors`: U S^power of `randomized_svd`, as exp-028's `ppmi_vectors`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp


def analyzer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(lowercase=True, stop_words="english").build_analyzer()


def build_vocab(docs: Iterable[list[str]], size: int, min_count: int) -> list[str]:
    counts = Counter()
    for d in docs:
        counts.update(d)
    words = [w for w, c in counts.most_common() if c >= min_count]
    return words[:size]


def to_ids(docs: Iterable[list[str]], index: dict[str, int]) -> Iterable[np.ndarray]:
    for d in docs:
        yield np.fromiter((index[w] for w in d if w in index), dtype=np.int32)


def cooc_window(
    id_docs: Iterable[np.ndarray], v: int, window: int = 5, chunk_tokens: int = 20_000_000
) -> sp.csr_matrix:
    """Symmetric counts of pairs (a, b) at distance 1..window within a document."""
    acc = sp.csr_matrix((v, v), dtype=np.float64)
    buf, n = [], 0

    def flush(parts):
        ids = np.concatenate(parts)
        doc = np.concatenate([np.full(len(p), i, dtype=np.int32) for i, p in enumerate(parts)])
        rows, cols = [], []
        for d in range(1, window + 1):
            same = doc[:-d] == doc[d:]
            a, b = ids[:-d][same], ids[d:][same]
            rows += [a, b]
            cols += [b, a]
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(v, v)).tocsr()

    for ids in id_docs:
        if len(ids) < 2:
            continue
        buf.append(ids)
        n += len(ids)
        if n >= chunk_tokens:
            acc = acc + flush(buf)
            buf, n = [], 0
    if buf:
        acc = acc + flush(buf)
    return acc


def cooc_doc(id_docs: Iterable[np.ndarray], v: int) -> sp.csr_matrix:
    """C[a, b] = number of documents containing both a and b (a != b)."""
    rows, cols = [], []
    for i, ids in enumerate(id_docs):
        u = np.unique(ids)
        rows.append(np.full(len(u), i, dtype=np.int64))
        cols.append(u)
    r, c = np.concatenate(rows), np.concatenate(cols)
    b = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(int(r.max()) + 1, v))
    out = (b.T @ b).tocsr()
    out.setdiag(0)
    out.eliminate_zeros()
    return out


def ppmi_sparse(c: sp.spmatrix, alpha: float = 0.75) -> sp.csr_matrix:
    """max(0, log(n(w, c) * sum_c' n(c')^a / (n(w) * n(c)^a))), diagonal removed."""
    c = sp.csr_matrix(c, dtype=np.float64)
    c.setdiag(0)
    c.eliminate_zeros()
    row = np.asarray(c.sum(1)).ravel()
    ctx = np.asarray(c.sum(0)).ravel() ** alpha
    coo = c.tocoo()
    pmi = np.log(coo.data * ctx.sum() / (row[coo.row] * ctx[coo.col]))
    keep = pmi > 0
    return sp.csr_matrix(
        (pmi[keep].astype(np.float32), (coo.row[keep], coo.col[keep])), shape=c.shape
    )


def vectors(ppmi: sp.spmatrix, k: int, power: float = 0.5, seed: int = 0) -> np.ndarray:
    from sklearn.utils.extmath import randomized_svd

    k = min(k, min(ppmi.shape) - 1)
    u, s, _ = randomized_svd(ppmi, n_components=k, random_state=seed)
    return (u * s[None, :] ** power).astype(np.float32)


def coverage(docs: Iterable[list[str]], vocab: set[str]) -> float:
    tot = hit = 0
    for d in docs:
        tot += len(d)
        hit += sum(w in vocab for w in d)
    return hit / max(tot, 1)
