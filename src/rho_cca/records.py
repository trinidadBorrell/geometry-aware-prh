"""Endpoints, paired tables, and the errors shared by the public fitters.

Feature identity is the checkpoint, site, pooling, width, and extraction
settings. A role name such as ``source`` is not an identity: the same basis
stays the same basis when it is reused as a later evaluation partner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


class RhoCCAError(ValueError):
    """Invalid geometry input or an operation the fitted object cannot support."""


class EndpointMismatch(RhoCCAError):
    """A factor was aimed at a feature basis other than the one it was fit on."""


class NonlinearTransportError(RhoCCAError):
    """A raw displacement cannot travel through clipping or row normalisation."""


def _text_map(values: Mapping[str, str] | None) -> dict[str, str]:
    if not values:
        return {}
    out: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise RhoCCAError("extraction settings must be strings")
        out[key] = value
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class Endpoint:
    """One side of a model pair, with the settings that define its features."""

    model: str
    revision: str
    layer: int | None
    site: str
    pooling: str
    width: int
    extraction: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not self.revision or not self.site or not self.pooling:
            raise RhoCCAError("model, revision, site, and pooling are required")
        if self.layer is not None and int(self.layer) < 0:
            raise RhoCCAError("layer must be non-negative")
        if int(self.width) < 1:
            raise RhoCCAError("feature width must be positive")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "extraction", _text_map(self.extraction))
        if self.layer is not None:
            object.__setattr__(self, "layer", int(self.layer))

    def identity_key(self) -> str:
        payload = {
            "model": self.model,
            "revision": self.revision,
            "layer": self.layer,
            "site": self.site,
            "pooling": self.pooling,
            "width": int(self.width),
            "extraction": dict(self.extraction),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict:
        return json.loads(self.identity_key())

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Endpoint":
        return cls(
            model=str(payload["model"]),
            revision=str(payload["revision"]),
            layer=payload["layer"],
            site=str(payload["site"]),
            pooling=str(payload["pooling"]),
            width=int(payload["width"]),
            extraction=dict(payload.get("extraction") or {}),
        )


@dataclass(frozen=True)
class ModelPair:
    """Source and partner endpoints. The high-level fitters read both."""

    source: Endpoint
    partner: Endpoint

    def __post_init__(self) -> None:
        if self.source.identity_key() == self.partner.identity_key():
            raise RhoCCAError("source and partner feature identities must differ")


def _ids(values: Sequence[str], *, what: str) -> tuple[str, ...]:
    out = tuple(str(item) for item in values)
    if len(out) != len(set(out)):
        raise RhoCCAError(f"ambiguous duplicate {what}")
    if any(item == "" for item in out):
        raise RhoCCAError(f"empty {what}")
    return out


@dataclass
class FeatureTable:
    """Row features for one endpoint. Row ``i`` is sample ``sample_ids[i]``."""

    endpoint: Endpoint
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        self.sample_ids = _ids(self.sample_ids, what="sample ids")
        self.group_ids = tuple(str(item) for item in self.group_ids)
        self.values = np.array(self.values, dtype=np.float64, copy=True)
        if self.values.ndim != 2:
            raise RhoCCAError("features must be a row matrix")
        n, width = self.values.shape
        if n != len(self.sample_ids) or n != len(self.group_ids):
            raise RhoCCAError("sample ids, group ids, and rows must align")
        if width != self.endpoint.width:
            raise RhoCCAError(
                f"{self.endpoint.model} width is {self.endpoint.width}, features have {width} columns"
            )
        if n < 1:
            raise RhoCCAError("feature table is empty")

    def index(self, sample_ids: Sequence[str]) -> np.ndarray:
        lookup = {sample_id: position for position, sample_id in enumerate(self.sample_ids)}
        missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
        if missing:
            raise RhoCCAError(f"unknown sample ids: {missing[:3]}")
        return np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)

    def take(self, positions: Sequence[int]) -> "FeatureTable":
        chosen = np.asarray(positions, dtype=np.int64)
        return FeatureTable(
            self.endpoint,
            tuple(self.sample_ids[int(i)] for i in chosen),
            tuple(self.group_ids[int(i)] for i in chosen),
            self.values[chosen],
        )


@dataclass
class PairedDataset:
    """Caption/image (or text/text) rows joined on sample id.

    Partner rows are reordered to the source ids. Group ids must then agree.
    A split, when present, names sample ids in this aligned order.
    """

    source: FeatureTable
    partner: FeatureTable
    split: "Split | None" = None

    def __post_init__(self) -> None:
        self.source, self.partner = align_tables(self.source, self.partner)
        if self.split is not None:
            self.split.validate_against(self)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.source.sample_ids

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self.source.group_ids

    def select(self, sample_ids: Sequence[str]) -> "PairedDataset":
        positions = self.source.index(sample_ids)
        return PairedDataset(self.source.take(positions), self.partner.take(positions), split=None)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.source.endpoint.identity_key().encode())
        digest.update(b"\0")
        digest.update(self.partner.endpoint.identity_key().encode())
        digest.update(b"\0")
        digest.update("\n".join(self.sample_ids).encode())
        digest.update(b"\0")
        digest.update("\n".join(self.group_ids).encode())
        digest.update(b"\0")
        digest.update(str(self.source.values.shape).encode())
        digest.update(str(self.partner.values.shape).encode())
        return digest.hexdigest()


def align_tables(source: FeatureTable, partner: FeatureTable) -> tuple[FeatureTable, FeatureTable]:
    if source.endpoint.identity_key() == partner.endpoint.identity_key():
        raise RhoCCAError("paired endpoints must have different feature identities")
    if set(source.sample_ids) != set(partner.sample_ids):
        raise RhoCCAError("source and partner sample ids do not match")
    order = {sample_id: position for position, sample_id in enumerate(partner.sample_ids)}
    positions = [order[sample_id] for sample_id in source.sample_ids]
    if positions != list(range(len(positions))):
        partner = partner.take(positions)
    if source.group_ids != partner.group_ids:
        raise RhoCCAError("group ids disagree after joining on sample id")
    return source, partner


@dataclass(frozen=True)
class Split:
    """Sample and group assignments produced by scikit-learn.

    This object stores the assignment. It does not draw a new split on load.
    """

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    group_by_sample: tuple[tuple[str, str], ...]
    seed: int | None
    fingerprint: str
    method: str

    def __post_init__(self) -> None:
        parts = {
            "train": self.train_ids,
            "validation": self.validation_ids,
            "test": self.test_ids,
        }
        seen: dict[str, str] = {}
        for name, ids in parts.items():
            for sample_id in ids:
                if sample_id in seen:
                    raise RhoCCAError(f"sample {sample_id} is in both {seen[sample_id]} and {name}")
                seen[sample_id] = name
        if not self.train_ids:
            raise RhoCCAError("training split is empty")
        groups: dict[str, str] = {}
        known = set(seen)
        for sample_id, group_id in self.group_by_sample:
            if sample_id not in known:
                raise RhoCCAError(f"split assignment has unknown sample {sample_id}")
            previous = groups.get(group_id)
            if previous is not None and previous != seen[sample_id]:
                raise RhoCCAError(f"group {group_id} crosses {previous} and {seen[sample_id]}")
            groups[group_id] = seen[sample_id]
        assigned = {sample_id for sample_id, _group in self.group_by_sample}
        if assigned != known:
            raise RhoCCAError("split assignments must name every train, validation, and test id")

    def validate_against(self, dataset: PairedDataset) -> None:
        known = set(dataset.sample_ids)
        named = set(self.train_ids) | set(self.validation_ids) | set(self.test_ids)
        if not named <= known:
            raise RhoCCAError("split names sample ids that are not in the dataset")
        lookup = dict(self.group_by_sample)
        for sample_id, group_id in zip(dataset.sample_ids, dataset.group_ids, strict=True):
            if sample_id in lookup and lookup[sample_id] != group_id:
                raise RhoCCAError(f"group id for {sample_id} does not match the saved split")
        if dataset.fingerprint() != self.fingerprint:
            raise RhoCCAError("dataset fingerprint does not match the split")

    def to_dict(self) -> dict:
        return {
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "group_by_sample": [list(item) for item in self.group_by_sample],
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Split":
        return cls(
            train_ids=tuple(payload["train_ids"]),
            validation_ids=tuple(payload["validation_ids"]),
            test_ids=tuple(payload["test_ids"]),
            group_by_sample=tuple((str(a), str(b)) for a, b in payload["group_by_sample"]),
            seed=payload["seed"],
            fingerprint=str(payload["fingerprint"]),
            method=str(payload["method"]),
        )
