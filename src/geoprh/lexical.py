"""Caption features from token identity alone, and exact CKA estimators (exp-022).

A mean-pooled embedding layer is `tf(counts) @ table`, so every feature here is a sparse
[n, V] token matrix, with V the tokens present in the captions. Rows are l2-normalised before use,
as PRH features are, so raw counts and token frequencies give the same kernel.

CKA estimators accumulate in float64: float32 norms over n^2 entries drift on the CPU (CKA +2.3% at
n = 10240).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
import torch


def count_matrix(ids: Sequence[Sequence[int]]) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Token ids per caption -> ([n, V] counts, caption lengths, the V token ids)."""
    lengths = np.array([len(x) for x in ids])
    if lengths.min() == 0:
        raise ValueError(f"{int((lengths == 0).sum())} empty captions")
    rows = np.repeat(np.arange(len(ids)), lengths)
    vocab, cols = np.unique(np.concatenate([np.asarray(x) for x in ids]), return_inverse=True)
    counts = sp.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(ids), len(vocab))
    )
    counts.sum_duplicates()
    return counts, lengths, vocab


def tf(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Token frequencies: what a mean over the caption's tokens multiplies the table by."""
    lengths = np.asarray(counts.sum(1)).ravel()
    return sp.diags(1 / lengths.astype(np.float32)) @ counts


def binary(counts: sp.csr_matrix) -> sp.csr_matrix:
    out = counts.copy()
    out.data = np.ones_like(out.data)
    return out


def idf(counts: sp.csr_matrix) -> np.ndarray:
    """Smoothed idf, the scikit-learn convention: log((1 + n) / (1 + df)) + 1."""
    df = np.asarray((counts > 0).sum(0)).ravel()
    return (np.log((1 + counts.shape[0]) / (1 + df)) + 1).astype(np.float32)


def tfidf(
    counts: sp.csr_matrix, weights: np.ndarray | None = None, sublinear: bool = False
) -> sp.csr_matrix:
    """Term weights x idf. `sublinear` replaces a count c by 1 + log(c)."""
    weights = idf(counts) if weights is None else weights
    if sublinear:
        term = counts.copy()
        term.data = 1 + np.log(term.data)
    else:
        term = tf(counts)
    return term @ sp.diags(weights)


def mask_top(counts: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Drop the k tokens with the largest corpus count. Rows left empty raise."""
    total = np.asarray(counts.sum(0)).ravel()
    keep = np.sort(np.argsort(total)[::-1][k:])
    out = counts[:, keep]
    empty = int((np.asarray(out.sum(1)).ravel() == 0).sum())
    if empty:
        raise ValueError(f"masking the top {k} tokens empties {empty} captions")
    return out


def truncate(
    ids: Sequence[Sequence[int]], length: int | Sequence[int], mode: str = "first", seed: int = 0
) -> list[list[int]]:
    """Keep `length` tokens of each caption (a scalar or one per caption): the first ones, or a
    random sample kept in order."""
    n = len(ids)
    lengths = [length] * n if isinstance(length, int | np.integer) else list(length)
    if mode == "first":
        return [list(x[: int(m)]) for x, m in zip(ids, lengths, strict=True)]
    if mode != "random":
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    out = []
    for x, m in zip(ids, lengths, strict=True):
        m = min(int(m), len(x))
        keep = np.sort(rng.choice(len(x), size=m, replace=False))
        out.append([x[i] for i in keep])
    return out


def to_unit_rows(x: sp.spmatrix | np.ndarray) -> torch.Tensor:
    """Dense float32 l2-normalised rows; all-zero rows stay zero."""
    dense = x.toarray() if sp.issparse(x) else np.asarray(x)
    return torch.nn.functional.normalize(torch.from_numpy(dense.astype(np.float32)), dim=-1)


def centred_unit_gram(k: torch.Tensor) -> torch.Tensor:
    """H K H with unit Frobenius norm, float64."""
    c = k.double()
    c = c - c.mean(0, keepdim=True)
    c = c - c.mean(1, keepdim=True)
    return c / c.norm()


def participation_ratio(k: torch.Tensor) -> float:
    """(tr C)^2 / ||C||_F^2 of the centred kernel."""
    c = centred_unit_gram(k)
    return float(c.trace() ** 2)


def cka_features(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA from features: ||X^T Y||^2 / (||X^T X|| ||Y^T Y||), centred, float64."""
    x, y = x.double(), y.double()
    x, y = x - x.mean(0), y - y.mean(0)
    return float((x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm()))


class HsicU:
    """Precomputed parts of the unbiased HSIC (Song et al. 2012) of one kernel."""

    def __init__(self, k: torch.Tensor):
        self.n = k.shape[0]
        self.k0 = k.clone()
        self.k0.fill_diagonal_(0)
        self.row = self.k0.sum(1, dtype=torch.float64)
        self.total = self.row.sum()
        self.self_hsic = self.hsic(self)

    def hsic(self, other: HsicU) -> torch.Tensor:
        n = self.n
        trace = (self.k0 * other.k0).sum(1, dtype=torch.float64).sum()
        return (
            trace
            + self.total * other.total / ((n - 1) * (n - 2))
            - 2 * (self.row @ other.row) / (n - 2)
        ) / (n * (n - 3))

    def cka(self, other: HsicU) -> float:
        denom = (self.self_hsic * other.self_hsic).clamp_min(0).sqrt()
        return float(self.hsic(other) / denom.clamp_min(1e-30))
