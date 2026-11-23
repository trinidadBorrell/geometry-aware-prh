"""Centred linear CKA with the analytical permutation expectation separated out.

``linear_cka`` is the feature formula used by ``frozen_partner_cka_audit.scores``
in the angus-prh-replication working tree (the audit was not part of commit
``8fd211a``). After column centring,

    raw = ||XᵀY||_F² / (||XᵀX||_F ||YᵀY||_F)
    chance = ||X||_F² ||Y||_F² / ((n - 1) ||XᵀX||_F ||YᵀY||_F)
    excess = raw - chance

``chance`` is the expectation of ``raw`` when one side's rows are exchanged
uniformly and the other side, and the centred Gram, stay fixed. It is not
itself a centred-Gram cosine. Zero-variance inputs make the denominator zero
and raise ``RhoCCAError`` rather than returning a quiet NaN.

``shared_neighbour_fraction`` is the optional neighbour score from that audit:
the mean, over rows, of ``|N_k(left) ∩ N_k(right)| / k``. It is set overlap of
the two k-nearest-neighbour lists, not a count of mutual kNN edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rho_cca.records import RhoCCAError


@dataclass(frozen=True)
class CKAScore:
    raw: float
    chance: float
    excess: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"raw": self.raw, "chance": self.chance, "excess": self.excess, "n": self.n}


def _centred(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise RhoCCAError("CKA features must be row matrices")
    if values.shape[0] < 2:
        raise RhoCCAError("CKA needs at least two rows")
    return values - values.mean(axis=0, keepdims=True)


def linear_cka(left: np.ndarray, right: np.ndarray) -> CKAScore:
    x = _centred(left)
    y = _centred(right)
    if x.shape[0] != y.shape[0]:
        raise RhoCCAError("CKA sides need the same number of rows")
    denominator = float(np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro"))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RhoCCAError("undefined CKA: a side has zero centred variance")
    raw = float(np.sum((x.T @ y) ** 2) / denominator)
    chance = float(np.sum(x * x) * np.sum(y * y) / ((x.shape[0] - 1) * denominator))
    return CKAScore(raw=raw, chance=chance, excess=raw - chance, n=int(x.shape[0]))


def double_centre(gram: np.ndarray) -> np.ndarray:
    values = np.asarray(gram, dtype=np.float64)
    return values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()


def linear_cka_from_feature_grams(left: np.ndarray, right: np.ndarray) -> CKAScore:
    """CKA from double-centred feature Grams ``XXᵀ`` and ``YYᵀ`` of uncentred rows.

    Centring is inside the Gram, so the rows must be the same uncentred matrices
    that ``linear_cka`` would centre itself.
    """
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise RhoCCAError("Gram CKA needs matching row matrices with at least two rows")
    left_gram = double_centre(x @ x.T)
    right_gram = double_centre(y @ y.T)
    denominator = float(np.linalg.norm(left_gram, ord="fro") * np.linalg.norm(right_gram, ord="fro"))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RhoCCAError("undefined CKA: a side has zero centred variance")
    raw = float(np.sum(left_gram * right_gram) / denominator)
    chance = float(np.trace(left_gram) * np.trace(right_gram) / ((left_gram.shape[0] - 1) * denominator))
    return CKAScore(raw=raw, chance=chance, excess=raw - chance, n=int(x.shape[0]))


def shared_neighbour_fraction(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each row's ``k`` neighbours that also sit in the other list."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        raise RhoCCAError("neighbour overlap needs matching rows")
    width = int(min(int(k), a.shape[0] - 1))
    if width < 1:
        raise RhoCCAError("k must leave at least one neighbour")

    def neighbours(features: np.ndarray) -> np.ndarray:
        squared = np.sum(features * features, axis=1)
        distance = squared[:, None] + squared[None, :] - 2.0 * (features @ features.T)
        np.fill_diagonal(distance, np.inf)
        part = np.argpartition(distance, kth=width - 1, axis=1)[:, :width]
        order = np.argsort(distance[np.arange(features.shape[0])[:, None], part], axis=1)
        return np.sort(part[np.arange(features.shape[0])[:, None], order], axis=1)

    left_k = neighbours(a)
    right_k = neighbours(b)
    hits = left_k[:, :, None] == right_k[:, None, :]
    return float(hits.any(axis=2).sum() / (left_k.shape[0] * left_k.shape[1]))
