"""Regularised CCA and correlation weights.

Covariances use the ``n - 1`` divisor in ``sae_crosscov.sample_covariances``.
Regularised CCA whitens ``C + τ I`` with ``τ = η trace(C) / d`` and takes the
SVD of the whitened cross-covariance (``regularized_cca`` in that module,
``angus-prh-replication`` commit ``8fd211a``).

Reliability weights are not those singular values. They are the archived
transform of the training Pearson correlations of the paired canonical scores
(``reliability_weights`` in ``crosscov_weighted_cka.py``):

    w = mean_normalise(max(ρ², floor) ** β)

``β = 1`` and ``floor = 1e-6`` are the frozen one-sided audit settings. The
default keeps every finite canonical coordinate in the prepared support; it
does not silently truncate to 16. Equal-weight coordinates use the same
directions with weight 1. One-sided scoring transforms the requested endpoint
and leaves the other side native.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rho_cca.cka import CKAScore, linear_cka
from rho_cca.preprocess import PreparedSpace, fit_space
from rho_cca.records import (
    Endpoint,
    EndpointMismatch,
    FeatureTable,
    ModelPair,
    NonlinearTransportError,
    PairedDataset,
    RhoCCAError,
)


@dataclass(frozen=True)
class RhoCCAConfig:
    eta: float = 0.01
    support_tol: float = 1e-8
    beta: float = 1.0
    floor: float = 1e-6
    components: int | None = None
    pca_components: int | None = None
    clip_quantile: float | None = None
    row_normalise: bool = False
    pca_rank_tol: float = 1e-8

    def __post_init__(self) -> None:
        if float(self.eta) < 0.0:
            raise RhoCCAError("CCA eta must be non-negative")
        if float(self.floor) < 0.0:
            raise RhoCCAError("correlation floor must be non-negative")
        if self.components is not None and int(self.components) < 1:
            raise RhoCCAError("component count must be positive")


def _joint_signs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.array(left, dtype=np.float64, copy=True)
    v = np.array(right, dtype=np.float64, copy=True)
    if u.shape[1] != v.shape[1]:
        raise RhoCCAError("paired factors need the same number of columns")
    for column in range(u.shape[1]):
        col = u[:, column]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0.0:
            u[:, column] *= -1.0
            v[:, column] *= -1.0
    return u, v


def _sym_inv_sqrt(matrix: np.ndarray, ridge: float, support_tol: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T) + ridge * np.eye(matrix.shape[0]))
    keep = values > float(support_tol) * max(float(values[-1]), float(support_tol))
    safe = np.where(keep, values, 1.0)
    inverse = vectors * (1.0 / np.sqrt(safe))
    return inverse @ vectors.T, keep


def regularized_cca(
    left: np.ndarray,
    right: np.ndarray,
    eta: float,
    support_tol: float,
) -> dict[str, np.ndarray | float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        raise RhoCCAError("CCA needs matching rows")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = a.shape[0] - 1
    c_a = (a.T @ a) / denom
    c_b = (b.T @ b) / denom
    c_ab = (a.T @ b) / denom
    tau_a = float(eta) * float(np.trace(c_a)) / float(c_a.shape[0])
    tau_b = float(eta) * float(np.trace(c_b)) / float(c_b.shape[0])
    ra, keep_a = _sym_inv_sqrt(c_a, tau_a, support_tol)
    rb, keep_b = _sym_inv_sqrt(c_b, tau_b, support_tol)
    p, s, qt = np.linalg.svd(ra @ c_ab @ rb, full_matrices=False)
    p, q = _joint_signs(p, qt.T)
    return {
        "L_A": ra @ p,
        "L_B": rb @ q,
        "s": np.asarray(s, dtype=np.float64),
        "tau_a": tau_a,
        "tau_b": tau_b,
        "support_a": keep_a,
        "support_b": keep_b,
    }


def pearson_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    numerator = np.sum(a * b, axis=0)
    denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    ok = denominator > 0.0
    out[ok] = numerator[ok] / denominator[ok]
    return out


def reliability_weights(rho: np.ndarray, beta: float, floor: float) -> dict[str, np.ndarray | int | bool]:
    rho = np.asarray(rho, dtype=np.float64)
    if rho.size == 0 or not np.all(np.isfinite(rho)):
        raise RhoCCAError("training correlations must be finite")
    clamp_tol = 1e-8
    n_neg = int(np.sum(rho < -clamp_tol))
    clipped = np.clip(rho, 0.0, 1.0)
    base = np.maximum(clipped * clipped, float(floor))
    weights = np.power(base, float(beta))
    if weights.size and float(np.mean(weights)) > 0.0:
        weights = weights / float(np.mean(weights))
    return {
        "w": weights,
        "n_neg_train_rho": n_neg,
        "n_floored": int(np.sum((clipped * clipped) < float(floor))),
        "algebraic_failure": n_neg > 0,
    }


@dataclass
class RhoCCAFit:
    source: Endpoint
    partner: Endpoint
    source_space: PreparedSpace
    partner_space: PreparedSpace
    l_source: np.ndarray
    l_partner: np.ndarray
    rho: np.ndarray
    weights: np.ndarray
    eta: float
    beta: float
    floor: float
    tau_source: float
    tau_partner: float
    n_negative_rho: int
    n_floored: int
    algebraic_failure: bool
    train_ids: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        self.l_source = np.array(self.l_source, dtype=np.float64, copy=True)
        self.l_partner = np.array(self.l_partner, dtype=np.float64, copy=True)
        self.rho = np.array(self.rho, dtype=np.float64, copy=True)
        self.weights = np.array(self.weights, dtype=np.float64, copy=True)
        self.train_ids = tuple(self.train_ids)

    @property
    def kind(self) -> str:
        return "rho_cca"

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    def _space(self, endpoint: Endpoint) -> tuple[str, PreparedSpace, np.ndarray]:
        key = endpoint.identity_key()
        if key == self.source.identity_key():
            return "source", self.source_space, self.l_source
        if key == self.partner.identity_key():
            return "partner", self.partner_space, self.l_partner
        raise EndpointMismatch(
            "CCA coordinates only accept the fitted source or partner basis; matching width is not enough"
        )

    def project_native(self, table: FeatureTable) -> np.ndarray:
        _role, space, _factor = self._space(table.endpoint)
        return space.apply(table.values)

    def support_factor(self, endpoint: Endpoint, *, weights: str = "reliability") -> np.ndarray:
        _role, _space, factor = self._space(endpoint)
        if weights == "equal":
            scale = np.ones(self.n_components, dtype=np.float64)
        elif weights == "reliability":
            scale = np.sqrt(self.weights)
        else:
            raise RhoCCAError("weights must be 'reliability' or 'equal'")
        return factor * scale.reshape(1, -1)

    def linear_factor(self, endpoint: Endpoint, *, weights: str = "reliability") -> np.ndarray:
        role, space, _factor = self._space(endpoint)
        if space.nonlinear:
            raise NonlinearTransportError(
                "clipping or row normalisation is not a constant factor in raw activations"
            )
        factor = self.support_factor(endpoint, weights=weights)
        if space.components is not None:
            factor = space.components @ factor
        if role not in {"source", "partner"}:
            raise EndpointMismatch(role)
        return factor

    def coordinates(self, table: FeatureTable, *, weights: str = "reliability") -> np.ndarray:
        """One endpoint, ``support @ L @ diag(scale)``. The other endpoint is not touched."""
        return self.project_native(table) @ self.support_factor(table.endpoint, weights=weights)

    def metric_factor(self, endpoint: Endpoint, *, weights: str = "reliability", dense: bool = False) -> np.ndarray:
        factor = self.support_factor(endpoint, weights=weights)
        if dense:
            return factor @ factor.T
        return factor

    def cka_with_native(
        self,
        source: FeatureTable,
        native: FeatureTable,
        *,
        weights: str = "reliability",
    ) -> CKAScore:
        """Transform the fitted source and score it against native features of C.

        C is not given the source factor, even when its width matches.
        """
        if source.endpoint.identity_key() != self.source.identity_key():
            raise EndpointMismatch("one-sided CKA transforms the fitted source only")
        if native.endpoint.identity_key() == self.source.identity_key():
            raise EndpointMismatch("the native side must be a different feature basis")
        if source.sample_ids != native.sample_ids:
            raise RhoCCAError("CKA rows must be joined on the same sample ids")
        return linear_cka(self.coordinates(source, weights=weights), np.asarray(native.values, dtype=np.float64))


def _bind_pair(model_pair: ModelPair, dataset: PairedDataset) -> None:
    if dataset.source.endpoint.identity_key() != model_pair.source.identity_key():
        raise EndpointMismatch("dataset source is not the model pair source")
    if dataset.partner.endpoint.identity_key() != model_pair.partner.identity_key():
        raise EndpointMismatch("dataset partner is not the model pair partner")


def fit_rho_cca(model_pair: ModelPair, dataset: PairedDataset, *, config: RhoCCAConfig) -> RhoCCAFit:
    """Fit source and partner directions on training rows. Evaluation partners stay out."""
    _bind_pair(model_pair, dataset)
    if dataset.split is None:
        raise RhoCCAError("fitting requires a split")
    dataset.split.validate_against(dataset)
    train = dataset.select(dataset.split.train_ids)
    if len(train.sample_ids) < 2:
        raise RhoCCAError("CCA needs at least two training rows")
    source_space = fit_space(
        train.source.values,
        clip_quantile=config.clip_quantile,
        row_normalise_rows=config.row_normalise,
        pca_components=config.pca_components,
        pca_rank_tol=config.pca_rank_tol,
    )
    partner_space = fit_space(
        train.partner.values,
        clip_quantile=config.clip_quantile,
        row_normalise_rows=config.row_normalise,
        pca_components=config.pca_components,
        pca_rank_tol=config.pca_rank_tol,
    )
    fitted = regularized_cca(
        source_space.apply(train.source.values),
        partner_space.apply(train.partner.values),
        float(config.eta),
        float(config.support_tol),
    )
    l_source = np.asarray(fitted["L_A"], dtype=np.float64)
    l_partner = np.asarray(fitted["L_B"], dtype=np.float64)
    if config.components is not None:
        keep = min(int(config.components), l_source.shape[1])
        l_source = l_source[:, :keep]
        l_partner = l_partner[:, :keep]
    source_scores = source_space.apply(train.source.values) @ l_source
    partner_scores = partner_space.apply(train.partner.values) @ l_partner
    rho = pearson_columns(source_scores, partner_scores)
    packed = reliability_weights(rho, config.beta, config.floor)
    return RhoCCAFit(
        source=model_pair.source,
        partner=model_pair.partner,
        source_space=source_space,
        partner_space=partner_space,
        l_source=l_source,
        l_partner=l_partner,
        rho=rho,
        weights=np.asarray(packed["w"], dtype=np.float64),
        eta=float(config.eta),
        beta=float(config.beta),
        floor=float(config.floor),
        tau_source=float(fitted["tau_a"]),
        tau_partner=float(fitted["tau_b"]),
        n_negative_rho=int(packed["n_neg_train_rho"]),
        n_floored=int(packed["n_floored"]),
        algebraic_failure=bool(packed["algebraic_failure"]),
        train_ids=train.sample_ids,
        fingerprint=dataset.fingerprint(),
    )
