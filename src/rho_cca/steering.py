"""Contrastive steering directions and explicit transport.

Row convention: a source factor ``F_A`` has shape ``(d_A, k)`` and a target
decoder ``G_B`` has shape ``(k, d_B)``. A displacement moves as

    v_B = v_A @ F_A @ G_B

``v_A`` is a difference of source training means. It is not clipped, row-normalised,
or recentred, and the decoder intercept is not added. Those are activation
transforms. A fit that clipped or row-normalised its features refuses this
raw transport.

Minimum-norm decoding sets ``G_B = pinv(F_B)``. Learned ridge decoding fits a
separate map from paired coordinates to partner features and is named
``ridge``. A ridge alignment already is that map: ``F_A`` is the regression
weight in the linear feature space and ``G_B`` is the identity. It is not
reported as a pseudoinverse.

Correlation weights are part of ``F`` when requested. Full-rank symmetric
pseudoinverse decoding cancels a shared positive diagonal; the fit records
whether the weighted and equal-weight displacements actually agreed. Norm
matching is :func:`match_norm`, not part of fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rho_cca.cca import RhoCCAFit
from rho_cca.records import (
    Endpoint,
    EndpointMismatch,
    ModelPair,
    NonlinearTransportError,
    PairedDataset,
    RhoCCAError,
)
from rho_cca.ridge import DEFAULT_T_GRID, RidgeFit, choose_penalty, ridge_weight

Alignment = RidgeFit | RhoCCAFit


@dataclass(frozen=True)
class DifferenceInMeans:
    """Labels name source training rows. They are not applied to the partner."""

    positive_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        positive = tuple(str(item) for item in self.positive_ids)
        negative = tuple(str(item) for item in self.negative_ids)
        if not positive or not negative:
            raise RhoCCAError("difference-in-means needs both classes")
        if set(positive) & set(negative):
            raise RhoCCAError("a sample cannot be both positive and negative")
        object.__setattr__(self, "positive_ids", positive)
        object.__setattr__(self, "negative_ids", negative)


@dataclass(frozen=True)
class SteeringConfig:
    """``decoder='auto'`` is ``ridge`` for a ridge alignment and ``pseudoinverse`` for ρ-CCA."""

    decoder: str = "auto"
    weights: str = "reliability"
    t: float | None = 1e-3
    t_grid: tuple[float, ...] = DEFAULT_T_GRID
    tie_tol: float = 1e-8
    pinv_tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.decoder not in {"auto", "pseudoinverse", "ridge"}:
            raise RhoCCAError("decoder must be auto, pseudoinverse, or ridge")
        if self.weights not in {"reliability", "equal"}:
            raise RhoCCAError("weights must be reliability or equal")


@dataclass(frozen=True)
class NormMatch:
    raw: np.ndarray
    applied: np.ndarray
    raw_norm: float
    applied_norm: float
    degenerate: bool


def match_norm(vector: np.ndarray, target_norm: float, *, near_zero_norm: float = 1e-8) -> NormMatch:
    """Rescale a displacement. A near-zero vector becomes zero rather than a unit noise vector."""
    raw = np.array(np.asarray(vector, dtype=np.float64).reshape(-1), copy=True)
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm < float(near_zero_norm):
        return NormMatch(raw, np.zeros_like(raw), raw_norm, 0.0, True)
    applied = raw * (float(target_norm) / raw_norm)
    return NormMatch(raw, applied, raw_norm, float(np.linalg.norm(applied)), False)


def _mean_rows(table_values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    chosen = np.asarray(table_values, dtype=np.float64)[positions]
    if chosen.shape[0] < 1:
        raise RhoCCAError("a steering class has no rows")
    return chosen.mean(axis=0)


def _require_train(dataset: PairedDataset, sample_ids: tuple[str, ...]) -> None:
    if dataset.split is None:
        raise RhoCCAError("steering examples must come from a training split")
    dataset.split.validate_against(dataset)
    train = set(dataset.split.train_ids)
    outside = [sample_id for sample_id in sample_ids if sample_id not in train]
    if outside:
        raise RhoCCAError("steering labels must be training sample ids")


def _pinv(factor: np.ndarray, tol: float) -> np.ndarray:
    return np.linalg.pinv(np.asarray(factor, dtype=np.float64), rcond=float(tol))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 1.0 if float(np.linalg.norm(left - right)) == 0.0 else 0.0
    return float(np.dot(left, right) / denom)


@dataclass
class SteeringFit:
    source: Endpoint
    partner: Endpoint
    target: Endpoint
    anchor: str
    decoder: str
    weights: str
    source_vector: np.ndarray
    command: np.ndarray
    f_source: np.ndarray
    f_partner: np.ndarray
    g_target: np.ndarray
    partner_vector: np.ndarray
    target_vector: np.ndarray
    decoder_intercept: np.ndarray | None
    weights_cancelled: bool | None
    raw_norm: float
    train_ids: tuple[str, ...]
    fingerprint: str
    alignment_kind: str

    def __post_init__(self) -> None:
        self.source_vector = np.array(self.source_vector, dtype=np.float64, copy=True)
        self.command = np.array(self.command, dtype=np.float64, copy=True)
        self.f_source = np.array(self.f_source, dtype=np.float64, copy=True)
        self.f_partner = np.array(self.f_partner, dtype=np.float64, copy=True)
        self.g_target = np.array(self.g_target, dtype=np.float64, copy=True)
        self.partner_vector = np.array(self.partner_vector, dtype=np.float64, copy=True)
        self.target_vector = np.array(self.target_vector, dtype=np.float64, copy=True)
        if self.decoder_intercept is not None:
            self.decoder_intercept = np.array(self.decoder_intercept, dtype=np.float64, copy=True)
        self.train_ids = tuple(self.train_ids)

    @property
    def kind(self) -> str:
        return "steering"

    def vector_for(self, endpoint: Endpoint) -> np.ndarray:
        key = endpoint.identity_key()
        if key == self.source.identity_key():
            return self.source_vector.copy()
        if key == self.target.identity_key():
            return self.target_vector.copy()
        if key == self.partner.identity_key():
            return self.partner_vector.copy()
        raise EndpointMismatch("no steering vector was decoded for this feature basis")

    def displacement(self, endpoint: Endpoint) -> np.ndarray:
        """The stored displacement. Means and decoder intercepts are not applied."""
        return self.vector_for(endpoint)


def _factors(alignment: Alignment, weights: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(alignment, RidgeFit):
        if weights not in {"reliability", "none"}:
            raise RhoCCAError("a ridge alignment has no correlation weights to turn off")
        return alignment.linear_factor(), np.eye(alignment.weight.shape[1], dtype=np.float64)
    return (
        alignment.linear_factor(alignment.source, weights=weights),
        alignment.linear_factor(alignment.partner, weights=weights),
    )


def _resolve_decoder(alignment: Alignment, requested: str) -> str:
    if requested == "auto":
        return "ridge" if isinstance(alignment, RidgeFit) else "pseudoinverse"
    if isinstance(alignment, RidgeFit) and requested == "pseudoinverse":
        raise RhoCCAError(
            "ridge transport uses the regression map; pseudoinverse decoding requires a CCA partner factor"
        )
    if isinstance(alignment, RidgeFit) and requested != "ridge":
        raise RhoCCAError("unsupported ridge decoder")
    return requested


def _fit_ridge_decoder(
    commands: np.ndarray,
    target: np.ndarray,
    commands_val: np.ndarray | None,
    target_val: np.ndarray | None,
    config: SteeringConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``G``, intercept, and the penalty. Displacement application ignores the intercept."""
    z = np.asarray(commands, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mu_z = z.mean(axis=0)
    mu_y = y.mean(axis=0)
    zc = z - mu_z
    yc = y - mu_y
    if config.t is None:
        if commands_val is None or target_val is None:
            raise RhoCCAError("ridge decoder selection requires validation pairs")
        penalties = np.asarray(config.t_grid, dtype=np.float64)
        scores = np.empty(penalties.shape, dtype=np.float64)
        weights = []
        for index, t_value in enumerate(penalties):
            weight, _lam = ridge_weight(zc, yc, float(t_value))
            weights.append(weight)
            prediction = (commands_val - mu_z) @ weight + mu_y
            scores[index] = float(np.mean((prediction - target_val) ** 2))
        chosen = choose_penalty(penalties, scores, config.tie_tol)
        weight = weights[chosen]
        t_value = float(penalties[chosen])
    else:
        t_value = float(config.t)
        weight, _lam = ridge_weight(zc, yc, t_value)
    intercept = mu_y - mu_z @ weight
    return weight, intercept, t_value


def fit_steering(
    alignment: Alignment,
    model_pair: ModelPair,
    objective: DifferenceInMeans,
    dataset: PairedDataset,
    *,
    config: SteeringConfig,
) -> SteeringFit:
    """Build a source difference-in-means direction and decode it for the partner."""
    if alignment.source.identity_key() != model_pair.source.identity_key():
        raise EndpointMismatch("alignment source does not match the model pair")
    if alignment.partner.identity_key() != model_pair.partner.identity_key():
        raise EndpointMismatch("alignment partner does not match the model pair")
    if dataset.source.endpoint.identity_key() != model_pair.source.identity_key():
        raise EndpointMismatch("steering dataset source does not match the model pair")
    if dataset.partner.endpoint.identity_key() != model_pair.partner.identity_key():
        raise EndpointMismatch("steering dataset partner does not match the model pair")
    _require_train(dataset, objective.positive_ids + objective.negative_ids)
    decoder = _resolve_decoder(alignment, config.decoder)
    source_vector = _mean_rows(dataset.source.values, dataset.source.index(objective.positive_ids))
    source_vector = source_vector - _mean_rows(dataset.source.values, dataset.source.index(objective.negative_ids))
    f_source, f_partner = _factors(alignment, config.weights)
    command = source_vector @ f_source
    cancelled: bool | None = None
    intercept = None
    if isinstance(alignment, RhoCCAFit) and decoder == "pseudoinverse":
        g_target = _pinv(f_partner, config.pinv_tol)
        f_source_eq, f_partner_eq = _factors(alignment, "equal")
        equal_vector = (source_vector @ f_source_eq) @ _pinv(f_partner_eq, config.pinv_tol)
        weighted_vector = command @ _pinv(
            _factors(alignment, "reliability")[1],
            config.pinv_tol,
        )
        cancelled = _cosine(equal_vector, weighted_vector) > 1.0 - 1e-6 and abs(
            float(np.linalg.norm(equal_vector) - np.linalg.norm(weighted_vector))
        ) <= 1e-6 * max(1.0, float(np.linalg.norm(equal_vector)))
        if config.weights == "equal":
            command = source_vector @ f_source_eq
            f_source = f_source_eq
            f_partner = f_partner_eq
            g_target = _pinv(f_partner, config.pinv_tol)
    elif decoder == "ridge" and isinstance(alignment, RhoCCAFit):
        train = dataset.select(dataset.split.train_ids)
        z_train = train.source.values @ f_source
        validation = None
        if dataset.split.validation_ids:
            validation = dataset.select(dataset.split.validation_ids)
        g_target, intercept, _t = _fit_ridge_decoder(
            z_train,
            train.partner.values,
            None if validation is None else validation.source.values @ f_source,
            None if validation is None else validation.partner.values,
            config,
        )
    elif isinstance(alignment, RidgeFit):
        g_target = np.eye(f_source.shape[1], dtype=np.float64)
    else:
        raise RhoCCAError("unsupported steering decoder")
    partner_vector = command @ g_target
    return SteeringFit(
        source=alignment.source,
        partner=alignment.partner,
        target=alignment.partner,
        anchor="source",
        decoder=decoder if not isinstance(alignment, RidgeFit) else "ridge",
        weights=config.weights if isinstance(alignment, RhoCCAFit) else "none",
        source_vector=source_vector,
        command=command,
        f_source=f_source,
        f_partner=f_partner,
        g_target=g_target,
        partner_vector=partner_vector,
        target_vector=partner_vector,
        decoder_intercept=intercept,
        weights_cancelled=cancelled,
        raw_norm=float(np.linalg.norm(partner_vector)),
        train_ids=dataset.split.train_ids,
        fingerprint=dataset.fingerprint(),
        alignment_kind=alignment.kind,
    )


def attach_steering_target(
    fit: SteeringFit,
    bridge: PairedDataset,
    *,
    anchor: str,
    target: Endpoint,
    config: SteeringConfig,
) -> SteeringFit:
    """Fit only a decoder from frozen anchor coordinates to a new model.

    ``bridge.source`` must be the anchor basis and ``bridge.partner`` the new
    target. Labels, target-native steering vectors, and behavioural scores are
    not arguments and are not read.
    """
    if anchor not in {"source", "partner"}:
        raise RhoCCAError("anchor must be source or partner")
    anchor_endpoint = fit.source if anchor == "source" else fit.partner
    if bridge.source.endpoint.identity_key() != anchor_endpoint.identity_key():
        raise EndpointMismatch("bridge anchor features are not the requested anchor basis")
    if bridge.partner.endpoint.identity_key() != target.identity_key():
        raise EndpointMismatch("bridge target features are not the requested target basis")
    if target.identity_key() == fit.source.identity_key():
        raise EndpointMismatch("the new target must be a different feature basis from the source")
    if bridge.split is None or not bridge.split.train_ids:
        raise RhoCCAError("attachment requires a training split of bridge pairs")
    bridge.split.validate_against(bridge)
    factor = fit.f_source if anchor == "source" else fit.f_partner
    train = bridge.select(bridge.split.train_ids)
    z_train = train.source.values @ factor
    if z_train.shape[1] != fit.command.shape[0]:
        raise RhoCCAError("anchor coordinates do not match the stored command")
    validation_z = None
    validation_y = None
    if bridge.split.validation_ids:
        held = bridge.select(bridge.split.validation_ids)
        validation_z = held.source.values @ factor
        validation_y = held.partner.values
    g_target, intercept, _t = _fit_ridge_decoder(z_train, train.partner.values, validation_z, validation_y, config)
    target_vector = fit.command @ g_target
    return SteeringFit(
        source=fit.source,
        partner=fit.partner,
        target=target,
        anchor=anchor,
        decoder="ridge",
        weights=fit.weights,
        source_vector=fit.source_vector,
        command=fit.command,
        f_source=fit.f_source,
        f_partner=fit.f_partner,
        g_target=g_target,
        partner_vector=fit.partner_vector,
        target_vector=target_vector,
        decoder_intercept=intercept,
        weights_cancelled=fit.weights_cancelled,
        raw_norm=float(np.linalg.norm(target_vector)),
        train_ids=fit.train_ids,
        fingerprint=fit.fingerprint,
        alignment_kind=fit.alignment_kind,
    )
