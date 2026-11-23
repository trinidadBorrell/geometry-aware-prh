"""NPZ payloads with a JSON header. No pickle and no model objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from rho_cca.cca import RhoCCAFit
from rho_cca.preprocess import PreparedSpace
from rho_cca.records import Endpoint, RhoCCAError
from rho_cca.ridge import RidgeFit
from rho_cca.steering import SteeringFit

SCHEMA_VERSION = 1


def _meta_array(meta: Mapping) -> np.ndarray:
    return np.asarray(json.dumps(meta, sort_keys=True))


def _read(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as blob:
        if "__meta__" not in blob.files:
            raise RhoCCAError("fit file has no JSON header")
        meta = json.loads(str(blob["__meta__"].item()))
        arrays = {key: np.array(blob[key], copy=True) for key in blob.files if key != "__meta__"}
    if int(meta.get("version", -1)) != SCHEMA_VERSION:
        raise RhoCCAError(f"unsupported fit schema version {meta.get('version')}")
    return meta, arrays


def _space_arrays(space: PreparedSpace, prefix: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if space.mean is not None:
        arrays[f"{prefix}_mean"] = space.mean
    if space.components is not None:
        arrays[f"{prefix}_components"] = space.components
    return arrays


def _space_meta(space: PreparedSpace) -> dict:
    return {
        "clip_threshold": space.clip_threshold,
        "row_normalise": space.row_normalise,
        "pca_width": space.pca_width,
        "numerical_rank": space.numerical_rank,
        "variance_fraction": space.variance_fraction,
    }


def _space_from(meta: Mapping, arrays: Mapping[str, np.ndarray], prefix: str) -> PreparedSpace:
    block = meta[prefix]
    mean = arrays.get(f"{prefix}_mean")
    components = arrays.get(f"{prefix}_components")
    if (mean is None) != (components is None):
        raise RhoCCAError("PCA mean and components must be saved together")
    return PreparedSpace(
        clip_threshold=block["clip_threshold"],
        row_normalise=bool(block["row_normalise"]),
        mean=mean,
        components=components,
        pca_width=block["pca_width"],
        numerical_rank=block["numerical_rank"],
        variance_fraction=block["variance_fraction"],
    )


def _check_matrix(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if tuple(array.shape) != tuple(shape):
        raise RhoCCAError(f"{name} has shape {array.shape}, expected {shape}")


def save_fit(fit: RidgeFit | RhoCCAFit | SteeringFit, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fit, RidgeFit):
        arrays, meta = _ridge_payload(fit)
    elif isinstance(fit, RhoCCAFit):
        arrays, meta = _cca_payload(fit)
    elif isinstance(fit, SteeringFit):
        arrays, meta = _steering_payload(fit)
    else:
        raise RhoCCAError("unsupported fit type")
    meta["version"] = SCHEMA_VERSION
    arrays["__meta__"] = _meta_array(meta)
    np.savez(destination, **arrays)
    return destination


def load_fit(path: str | Path) -> RidgeFit | RhoCCAFit | SteeringFit:
    meta, arrays = _read(Path(path))
    kind = meta.get("kind")
    if kind == "ridge":
        return _ridge_from(meta, arrays)
    if kind == "rho_cca":
        return _cca_from(meta, arrays)
    if kind == "steering":
        return _steering_from(meta, arrays)
    raise RhoCCAError(f"unknown fit kind {kind}")


def _ridge_payload(fit: RidgeFit) -> tuple[dict[str, np.ndarray], dict]:
    arrays = {
        "mu_source": fit.mu_source,
        "mu_partner": fit.mu_partner,
        "weight": fit.weight,
        **_space_arrays(fit.source_space, "source_space"),
        **_space_arrays(fit.partner_space, "partner_space"),
    }
    meta = {
        "kind": "ridge",
        "source": fit.source.to_dict(),
        "partner": fit.partner.to_dict(),
        "source_space": _space_meta(fit.source_space),
        "partner_space": _space_meta(fit.partner_space),
        "t": fit.t,
        "lam": fit.lam,
        "tie_tol": fit.tie_tol,
        "selected": fit.selected,
        "validation_mse": fit.validation_mse,
        "train_ids": list(fit.train_ids),
        "fingerprint": fit.fingerprint,
    }
    return arrays, meta


def _ridge_from(meta: Mapping, arrays: Mapping[str, np.ndarray]) -> RidgeFit:
    source = Endpoint.from_dict(meta["source"])
    partner = Endpoint.from_dict(meta["partner"])
    source_space = _space_from(meta, arrays, "source_space")
    partner_space = _space_from(meta, arrays, "partner_space")
    weight = arrays["weight"]
    source_width = source_space.support_width or source.width
    partner_width = partner_space.support_width or partner.width
    _check_matrix(weight, (source_width, partner_width), "weight")
    _check_matrix(arrays["mu_source"], (source_width,), "mu_source")
    _check_matrix(arrays["mu_partner"], (partner_width,), "mu_partner")
    return RidgeFit(
        source=source,
        partner=partner,
        source_space=source_space,
        partner_space=partner_space,
        mu_source=arrays["mu_source"],
        mu_partner=arrays["mu_partner"],
        weight=weight,
        t=float(meta["t"]),
        lam=float(meta["lam"]),
        tie_tol=float(meta["tie_tol"]),
        selected=bool(meta["selected"]),
        validation_mse=meta["validation_mse"],
        train_ids=tuple(meta["train_ids"]),
        fingerprint=str(meta["fingerprint"]),
    )


def _cca_payload(fit: RhoCCAFit) -> tuple[dict[str, np.ndarray], dict]:
    arrays = {
        "l_source": fit.l_source,
        "l_partner": fit.l_partner,
        "rho": fit.rho,
        "weights": fit.weights,
        **_space_arrays(fit.source_space, "source_space"),
        **_space_arrays(fit.partner_space, "partner_space"),
    }
    meta = {
        "kind": "rho_cca",
        "source": fit.source.to_dict(),
        "partner": fit.partner.to_dict(),
        "source_space": _space_meta(fit.source_space),
        "partner_space": _space_meta(fit.partner_space),
        "eta": fit.eta,
        "beta": fit.beta,
        "floor": fit.floor,
        "tau_source": fit.tau_source,
        "tau_partner": fit.tau_partner,
        "n_negative_rho": fit.n_negative_rho,
        "n_floored": fit.n_floored,
        "algebraic_failure": fit.algebraic_failure,
        "train_ids": list(fit.train_ids),
        "fingerprint": fit.fingerprint,
    }
    return arrays, meta


def _cca_from(meta: Mapping, arrays: Mapping[str, np.ndarray]) -> RhoCCAFit:
    source = Endpoint.from_dict(meta["source"])
    partner = Endpoint.from_dict(meta["partner"])
    source_space = _space_from(meta, arrays, "source_space")
    partner_space = _space_from(meta, arrays, "partner_space")
    l_source = arrays["l_source"]
    l_partner = arrays["l_partner"]
    source_width = source_space.support_width or source.width
    partner_width = partner_space.support_width or partner.width
    k = int(l_source.shape[1])
    _check_matrix(l_source, (source_width, k), "l_source")
    _check_matrix(l_partner, (partner_width, k), "l_partner")
    _check_matrix(arrays["rho"], (k,), "rho")
    _check_matrix(arrays["weights"], (k,), "weights")
    return RhoCCAFit(
        source=source,
        partner=partner,
        source_space=source_space,
        partner_space=partner_space,
        l_source=l_source,
        l_partner=l_partner,
        rho=arrays["rho"],
        weights=arrays["weights"],
        eta=float(meta["eta"]),
        beta=float(meta["beta"]),
        floor=float(meta["floor"]),
        tau_source=float(meta["tau_source"]),
        tau_partner=float(meta["tau_partner"]),
        n_negative_rho=int(meta["n_negative_rho"]),
        n_floored=int(meta["n_floored"]),
        algebraic_failure=bool(meta["algebraic_failure"]),
        train_ids=tuple(meta["train_ids"]),
        fingerprint=str(meta["fingerprint"]),
    )


def _steering_payload(fit: SteeringFit) -> tuple[dict[str, np.ndarray], dict]:
    arrays = {
        "source_vector": fit.source_vector,
        "command": fit.command,
        "f_source": fit.f_source,
        "f_partner": fit.f_partner,
        "g_target": fit.g_target,
        "partner_vector": fit.partner_vector,
        "target_vector": fit.target_vector,
    }
    if fit.decoder_intercept is not None:
        arrays["decoder_intercept"] = fit.decoder_intercept
    meta = {
        "kind": "steering",
        "source": fit.source.to_dict(),
        "partner": fit.partner.to_dict(),
        "target": fit.target.to_dict(),
        "anchor": fit.anchor,
        "decoder": fit.decoder,
        "weights": fit.weights,
        "weights_cancelled": fit.weights_cancelled,
        "raw_norm": fit.raw_norm,
        "train_ids": list(fit.train_ids),
        "fingerprint": fit.fingerprint,
        "alignment_kind": fit.alignment_kind,
        "has_intercept": fit.decoder_intercept is not None,
    }
    return arrays, meta


def _steering_from(meta: Mapping, arrays: Mapping[str, np.ndarray]) -> SteeringFit:
    f_source = arrays["f_source"]
    g_target = arrays["g_target"]
    command = arrays["command"]
    if f_source.ndim != 2 or g_target.ndim != 2:
        raise RhoCCAError("steering factors must be matrices")
    if f_source.shape[1] != command.shape[0] or g_target.shape[0] != command.shape[0]:
        raise RhoCCAError("command width does not match the saved factors")
    if g_target.shape[1] != arrays["target_vector"].shape[0]:
        raise RhoCCAError("target vector width does not match the decoder")
    intercept = arrays.get("decoder_intercept")
    if bool(meta["has_intercept"]) != (intercept is not None):
        raise RhoCCAError("decoder intercept flag does not match the payload")
    return SteeringFit(
        source=Endpoint.from_dict(meta["source"]),
        partner=Endpoint.from_dict(meta["partner"]),
        target=Endpoint.from_dict(meta["target"]),
        anchor=str(meta["anchor"]),
        decoder=str(meta["decoder"]),
        weights=str(meta["weights"]),
        source_vector=arrays["source_vector"],
        command=command,
        f_source=f_source,
        f_partner=arrays["f_partner"],
        g_target=g_target,
        partner_vector=arrays["partner_vector"],
        target_vector=arrays["target_vector"],
        decoder_intercept=intercept,
        weights_cancelled=meta["weights_cancelled"],
        raw_norm=float(meta["raw_norm"]),
        train_ids=tuple(meta["train_ids"]),
        fingerprint=str(meta["fingerprint"]),
        alignment_kind=str(meta["alignment_kind"]),
    )
