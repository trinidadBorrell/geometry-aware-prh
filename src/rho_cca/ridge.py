"""Ridge-induced source metrics.

The normal equation is ``ridge_weight`` from ``objective_independent.py`` on
``angus-prh-replication`` commit ``8fd211a``. Rows are observations:

    (C_X + λ I) W = Xcᵀ Yc / n
    λ = t · trace(C_X) / q
    C_X = Xcᵀ Xc / n

``W`` has shape ``(source support, partner support)``. The induced source
metric is ``W @ W.T``. Predictions put the partner training mean back:

    Y_hat = (X - μ_X) W + μ_Y

Metric coordinates omit that intercept. Penalty selection, when requested,
minimises validation prediction MSE and breaks ties toward the larger
penalty (``choose_index`` in the same module). Test rows are never scored
for that choice. The spectral column-convention ridge is not used.
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

DEFAULT_T_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
ARCHIVAL_T = 1e-3


@dataclass(frozen=True)
class RidgeConfig:
    """``t=None`` selects the penalty on the validation split. A fixed ``t`` does not."""

    t: float | None = ARCHIVAL_T
    t_grid: tuple[float, ...] = DEFAULT_T_GRID
    tie_tol: float = 1e-8
    pca_components: int | None = None
    clip_quantile: float | None = None
    row_normalise: bool = False
    pca_rank_tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.t is None and not self.t_grid:
            raise RhoCCAError("penalty selection needs a t grid")
        if self.t is not None and float(self.t) < 0.0:
            raise RhoCCAError("ridge t must be non-negative")
        if float(self.tie_tol) < 0.0:
            raise RhoCCAError("tie tolerance must be non-negative")


def ridge_weight(xc: np.ndarray, yc: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    n_rows, width = xc.shape
    if n_rows < 2 or yc.shape[0] != n_rows:
        raise RhoCCAError("ridge needs paired rows")
    covariance = (xc.T @ xc) / float(n_rows)
    cross = (xc.T @ yc) / float(n_rows)
    lam = float(t) * float(np.trace(covariance)) / float(width)
    weight = np.linalg.solve(covariance + lam * np.eye(width, dtype=np.float64), cross)
    return weight, lam


def choose_penalty(penalties: np.ndarray, scores: np.ndarray, tie_tol: float) -> int:
    """Lowest validation MSE. Ties keep the larger penalty."""
    penalties = np.asarray(penalties, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if penalties.shape != scores.shape or penalties.size == 0:
        raise RhoCCAError("penalty selection needs matching scores")
    if not np.all(np.isfinite(scores)):
        raise RhoCCAError("validation MSE is undefined")
    best = float(np.min(scores))
    thresh = best * (1.0 + float(tie_tol)) + 1e-12
    eligible = np.flatnonzero(scores <= thresh)
    return int(eligible[np.argmax(penalties[eligible])])


def _training(dataset: PairedDataset) -> PairedDataset:
    if dataset.split is None:
        raise RhoCCAError("fitting requires a split; test rows are not a default holdout")
    dataset.split.validate_against(dataset)
    train = dataset.select(dataset.split.train_ids)
    if len(train.sample_ids) < 2:
        raise RhoCCAError("ridge needs at least two training rows")
    return train


@dataclass
class RidgeFit:
    """Source metric induced by predicting the partner. The partner has no learned metric."""

    source: Endpoint
    partner: Endpoint
    source_space: PreparedSpace
    partner_space: PreparedSpace
    mu_source: np.ndarray
    mu_partner: np.ndarray
    weight: np.ndarray
    t: float
    lam: float
    tie_tol: float
    selected: bool
    validation_mse: float | None
    train_ids: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        self.mu_source = np.array(self.mu_source, dtype=np.float64, copy=True)
        self.mu_partner = np.array(self.mu_partner, dtype=np.float64, copy=True)
        self.weight = np.array(self.weight, dtype=np.float64, copy=True)
        self.train_ids = tuple(self.train_ids)

    @property
    def kind(self) -> str:
        return "ridge"

    def require_source(self, endpoint: Endpoint) -> None:
        if endpoint.identity_key() != self.source.identity_key():
            raise EndpointMismatch(
                "this ridge metric was fit on a different source basis; matching width is not enough"
            )

    def require_partner(self, endpoint: Endpoint) -> None:
        if endpoint.identity_key() != self.partner.identity_key():
            raise EndpointMismatch("partner features do not match the basis this ridge map predicts")

    def project_native(self, table: FeatureTable) -> np.ndarray:
        key = table.endpoint.identity_key()
        if key == self.source.identity_key():
            return self.source_space.apply(table.values)
        if key == self.partner.identity_key():
            return self.partner_space.apply(table.values)
        raise EndpointMismatch(
            "native projection only accepts the fitted source or partner basis"
        )

    def metric_factor(self, *, dense: bool = False) -> np.ndarray:
        """Source factor ``F`` with ``X @ F`` in the support, or the dense ``F @ F.T``."""
        factor = self.weight
        if dense:
            return factor @ factor.T
        return factor

    def linear_factor(self) -> np.ndarray:
        """Factor in the pre-PCA feature space. Rejects clipping and row normalisation."""
        if self.source_space.nonlinear:
            raise NonlinearTransportError(
                "clipping or row normalisation is not a constant factor in raw activations"
            )
        factor = self.weight
        if self.source_space.components is not None:
            factor = self.source_space.components @ factor
        return factor

    def metric_coordinates(self, table: FeatureTable) -> np.ndarray:
        """Centred coordinates ``(support - μ_source) @ W``. No partner intercept."""
        self.require_source(table.endpoint)
        support = self.source_space.apply(table.values)
        return (support - self.mu_source) @ self.weight

    def predict(self, table: FeatureTable) -> np.ndarray:
        """``(support - μ_source) @ W + μ_partner``."""
        self.require_source(table.endpoint)
        return self.metric_coordinates(table) + self.mu_partner

    def cka_with_native(self, source: FeatureTable, native: FeatureTable) -> CKAScore:
        """Apply the source metric and leave ``native`` on its own supplied features.

        ``native`` is not projected by this fit. Passing it through
        ``metric_coordinates`` is rejected unless it really is the source basis,
        which is the wrong role for a held-out partner.
        """
        if native.endpoint.identity_key() == self.source.identity_key():
            raise EndpointMismatch("the native side must be a different feature basis")
        if source.sample_ids != native.sample_ids:
            raise RhoCCAError("CKA rows must be joined on the same sample ids")
        return linear_cka(self.metric_coordinates(source), np.asarray(native.values, dtype=np.float64))


def _bind_pair(model_pair: ModelPair, dataset: PairedDataset) -> None:
    if dataset.source.endpoint.identity_key() != model_pair.source.identity_key():
        raise EndpointMismatch("dataset source is not the model pair source")
    if dataset.partner.endpoint.identity_key() != model_pair.partner.identity_key():
        raise EndpointMismatch("dataset partner is not the model pair partner")


def fit_ridge(model_pair: ModelPair, dataset: PairedDataset, *, config: RidgeConfig) -> RidgeFit:
    """Fit a source metric on training rows only. ``model_pair`` selects the bases."""
    _bind_pair(model_pair, dataset)
    train = _training(dataset)
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
    x_train = source_space.apply(train.source.values)
    y_train = partner_space.apply(train.partner.values)
    mu_x = x_train.mean(axis=0)
    mu_y = y_train.mean(axis=0)
    xc = x_train - mu_x
    yc = y_train - mu_y
    selected = config.t is None
    validation_mse = None
    if selected:
        if dataset.split is None or not dataset.split.validation_ids:
            raise RhoCCAError("penalty selection requires a non-empty validation split")
        held = dataset.select(dataset.split.validation_ids)
        x_val = source_space.apply(held.source.values)
        y_val = partner_space.apply(held.partner.values)
        penalties = np.asarray(config.t_grid, dtype=np.float64)
        scores = np.empty(penalties.shape, dtype=np.float64)
        weights = []
        for index, t_value in enumerate(penalties):
            weight, _lam = ridge_weight(xc, yc, float(t_value))
            weights.append(weight)
            prediction = (x_val - mu_x) @ weight + mu_y
            gap = prediction - y_val
            scores[index] = float(np.mean(gap * gap))
        chosen = choose_penalty(penalties, scores, config.tie_tol)
        t_value = float(penalties[chosen])
        weight = weights[chosen]
        validation_mse = float(scores[chosen])
        lam = float(t_value) * float(np.trace((xc.T @ xc) / xc.shape[0])) / float(xc.shape[1])
    else:
        t_value = float(config.t)
        weight, lam = ridge_weight(xc, yc, t_value)
    return RidgeFit(
        source=model_pair.source,
        partner=model_pair.partner,
        source_space=source_space,
        partner_space=partner_space,
        mu_source=mu_x,
        mu_partner=mu_y,
        weight=weight,
        t=t_value,
        lam=lam,
        tie_tol=float(config.tie_tol),
        selected=selected,
        validation_mse=validation_mse,
        train_ids=train.sample_ids,
        fingerprint=dataset.fingerprint(),
    )
