"""Load the committed smoke-triple fixture. This is not a second fitter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rho_cca import Endpoint, FeatureTable, ModelPair, PairedDataset, Split

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_olmo_clip_dinov2_q32.npz"
MAP_ATOL = 1e-10
MAP_RTOL = 1e-8
CKA_ATOL = 1e-6


def load():
    with np.load(FIXTURE, allow_pickle=False) as blob:
        payload = {key: blob[key] for key in blob.files}
    payload["meta"] = json.loads(str(payload["meta"].item()))
    for key in ("train_ids", "val_ids", "test_ids"):
        payload[key] = tuple(str(item) for item in payload[key].tolist())
    return payload


def endpoint(spec: dict) -> Endpoint:
    return Endpoint(
        model=spec["model"],
        revision=spec["revision"],
        layer=spec["layer"],
        site=spec["site"],
        pooling=spec["pooling"],
        width=int(spec["width"]),
        extraction=dict(spec["extraction"]),
    )


def paired(payload, *, source_rows: dict[str, np.ndarray], partner_rows: dict[str, np.ndarray]) -> tuple[ModelPair, PairedDataset]:
    meta = payload["meta"]["endpoints"]
    source = endpoint(meta["source"])
    partner = endpoint(meta["partner"])
    order = ("train", "val", "test")
    ids = payload["train_ids"] + payload["val_ids"] + payload["test_ids"]
    groups = ids
    source_table = FeatureTable(source, ids, groups, np.vstack([source_rows[name] for name in order]))
    partner_table = FeatureTable(partner, ids, groups, np.vstack([partner_rows[name] for name in order]))
    bare = PairedDataset(source_table, partner_table)
    split = Split(
        train_ids=payload["train_ids"],
        validation_ids=payload["val_ids"],
        test_ids=payload["test_ids"],
        group_by_sample=tuple((sample_id, sample_id) for sample_id in ids),
        seed=None,
        fingerprint=bare.fingerprint(),
        method="coco-val2017 manifest; one image is one group",
    )
    return ModelPair(source, partner), PairedDataset(source_table, partner_table, split)


def native_test(payload) -> FeatureTable:
    spec = endpoint(payload["meta"]["endpoints"]["native"])
    return FeatureTable(spec, payload["test_ids"], payload["test_ids"], payload["test_dinov2"])


def source_test(dataset: PairedDataset, payload) -> FeatureTable:
    return dataset.select(payload["test_ids"]).source


def gap(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))
