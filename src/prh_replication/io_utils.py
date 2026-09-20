"""JSON/torch IO with sample-id checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_key(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_features(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_features(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if "feats" not in obj or "sample_ids" not in obj:
        raise ValueError(f"Feature cache missing feats/sample_ids: {path}")
    feats = obj["feats"]
    if not torch.isfinite(feats.float()).all():
        raise ValueError(f"Non-finite features in {path}")
    return obj


def assert_ids_match(left: list[str], right: list[str], *, what: str) -> None:
    if list(left) != list(right):
        raise ValueError(f"Sample-order mismatch ({what}): {len(left)} vs {len(right)}")
