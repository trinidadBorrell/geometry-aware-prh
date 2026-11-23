"""Clipping, row normalisation, and PCA.

Clipping plus row normalisation is the caption-feature prep from
``feature_prep.py`` on ``angus-prh-replication`` (commit ``8fd211a``): the
threshold is the training absolute-value order statistic, then each row is
L2-normalised. That map is nonlinear, so a metric learned after it is a
constant matrix only in the prepared coordinates.

PCA follows ``fit_pca`` / ``project`` in ``sae_crosscov.py`` at the same
commit. Components are sign-canonicalised, but callers that compare bases
should still use subspaces when a sign convention may differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rho_cca.records import RhoCCAError


def clip_threshold(values: np.ndarray, quantile: float) -> float:
    """Training absolute order statistic. ``index = int(quantile * numel)``."""
    flat = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if flat.size == 0:
        raise RhoCCAError("cannot clip an empty feature matrix")
    if not 0.0 <= float(quantile) <= 1.0:
        raise RhoCCAError("clip quantile must be in [0, 1]")
    index = int(float(quantile) * flat.size)
    if index >= flat.size:
        index = flat.size - 1
    return float(np.sort(flat)[index])


def clip_and_normalise(values: np.ndarray, threshold: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -float(threshold), float(threshold))
    norms = np.linalg.norm(clipped, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RhoCCAError("a clipped row has zero norm")
    return clipped / norms


def row_normalise(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise RhoCCAError("a feature row has zero norm")
    return matrix / norms


def _column_signs(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=np.float64, copy=True)
    for column in range(out.shape[1]):
        col = out[:, column]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0.0:
            out[:, column] *= -1.0
    return out


def fit_pca(features: np.ndarray, components: int | None, rank_tol: float = 1e-8) -> dict:
    """Economy SVD of column-centred rows. ``components=None`` keeps every supported mode."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise RhoCCAError("PCA needs at least two rows")
    mean = values.mean(axis=0)
    centred = values - mean
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    leading = float(singular[0]) if singular.size else 0.0
    threshold = float(rank_tol) * leading if leading > 0.0 else float(rank_tol)
    supported = int(np.sum(singular > threshold))
    requested = supported if components is None else int(components)
    width = int(min(requested, supported, values.shape[0] - 1, values.shape[1]))
    if width < 1:
        raise RhoCCAError("PCA retained no mode")
    basis = _column_signs(vt[:width].T.copy())
    total = float(np.sum(singular * singular))
    covered = float(np.sum(singular[:width] ** 2) / total) if total > 0.0 else float("nan")
    return {
        "mean": mean,
        "components": basis,
        "singular_values": np.asarray(singular, dtype=np.float64),
        "width": width,
        "numerical_rank": supported,
        "variance_fraction": covered,
    }


def project(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    centred = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64).reshape(1, -1)
    return centred @ np.asarray(components, dtype=np.float64)


@dataclass(frozen=True)
class PreparedSpace:
    """Training-fitted prep. ``components`` is ``(ambient, support)`` or absent."""

    clip_threshold: float | None
    row_normalise: bool
    mean: np.ndarray | None
    components: np.ndarray | None
    pca_width: int | None
    numerical_rank: int | None
    variance_fraction: float | None

    @property
    def nonlinear(self) -> bool:
        return self.clip_threshold is not None or self.row_normalise

    @property
    def support_width(self) -> int | None:
        if self.components is None:
            return None
        return int(self.components.shape[1])

    def apply(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2:
            raise RhoCCAError("features must be a row matrix")
        if self.clip_threshold is not None:
            matrix = clip_and_normalise(matrix, self.clip_threshold)
        elif self.row_normalise:
            matrix = row_normalise(matrix)
        if self.components is not None:
            if matrix.shape[1] != self.components.shape[0]:
                raise RhoCCAError("features do not match the prepared width")
            matrix = project(matrix, self.mean, self.components)
        return matrix


def fit_space(
    features: np.ndarray,
    *,
    clip_quantile: float | None,
    row_normalise_rows: bool,
    pca_components: int | None,
    pca_rank_tol: float,
) -> PreparedSpace:
    matrix = np.asarray(features, dtype=np.float64)
    threshold = None
    if clip_quantile is not None:
        threshold = clip_threshold(matrix, clip_quantile)
        matrix = clip_and_normalise(matrix, threshold)
    elif row_normalise_rows:
        matrix = row_normalise(matrix)
    mean = None
    components = None
    width = None
    numerical_rank = None
    variance_fraction = None
    if pca_components is not None:
        fitted = fit_pca(matrix, pca_components, pca_rank_tol)
        mean = fitted["mean"]
        components = fitted["components"]
        width = int(fitted["width"])
        numerical_rank = int(fitted["numerical_rank"])
        variance_fraction = float(fitted["variance_fraction"])
    return PreparedSpace(
        clip_threshold=threshold,
        row_normalise=bool(row_normalise_rows or clip_quantile is not None),
        mean=None if mean is None else np.array(mean, copy=True),
        components=None if components is None else np.array(components, copy=True),
        pca_width=width,
        numerical_rank=numerical_rank,
        variance_fraction=variance_fraction,
    )
