"""JSON/torch IO with sample-id checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def skip_if_complete(out: Path, *, force: bool = False, smoke: bool = False, label: str | None = None) -> bool:
    """True if a completed summary.json exists and this is not an explicit rerun."""
    if (out / "summary.json").exists() and not force and not smoke:
        name = label or out.name
        print(f"{name} already complete; pass --force to rerun", flush=True)
        return True
    return False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(x: Any) -> Any:
    """JSON-safe values: numpy/torch/Path; non-finite floats become null."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return jsonable(x.detach().cpu().tolist())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


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
