"""Clip / L2-prep helpers shared by caption and CCA reliability studies."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F


def clip_threshold(feats: torch.Tensor, q: float = 0.95) -> float:
    """Same order statistic as ``remove_outliers`` on one gallery."""
    flat = feats.detach().float().abs().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("cannot clip an empty feature matrix")
    index = int(q * flat.numel())
    if index >= flat.numel():
        index = flat.numel() - 1
    return float(flat.sort().values[index].item())


def prepare_with_threshold(feats: torch.Tensor, threshold: float) -> torch.Tensor:
    """Clip at a supplied threshold, then L2-normalise rows. The threshold is not reestimated."""
    clipped = feats.detach().float().clamp(-float(threshold), float(threshold))
    return F.normalize(clipped, p=2, dim=-1)


def sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(repr(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()
