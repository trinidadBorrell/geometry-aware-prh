"""Optional activation helpers.

Importing this module does not import PyTorch. ``preblock_rows`` is the
forward-pre-hook used by the IMDb matched-pair runs on ``angus-prh-replication``
(commit ``8fd211a``): the residual entering the chosen block, at explicit token
positions, with no generation. ``hashed_embedding`` is a deterministic stand-in
so examples can exercise the extractor interface without a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rho_cca.records import RhoCCAError


def hashed_embedding(token_ids: Sequence[Sequence[int]], width: int, seed: int) -> np.ndarray:
    """Mean of seeded token vectors. Same ids and width always give the same rows."""
    if int(width) < 1:
        raise RhoCCAError("embedding width must be positive")
    rows = []
    for ids in token_ids:
        if len(ids) == 0:
            raise RhoCCAError("a token row is empty")
        accum = np.zeros(int(width), dtype=np.float64)
        for token in ids:
            generator = np.random.default_rng(int(seed) + int(token) * 1_000_003)
            accum += generator.normal(size=int(width))
        rows.append(accum / float(len(ids)))
    return np.vstack(rows)


def preblock_rows(model, input_ids, block, positions: Sequence[int]):
    """Return the pre-block residual at ``positions``. Requires PyTorch and a loaded module."""
    import torch

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.view(1, -1)
    box: dict = {}

    def hook(_module, inputs):
        box["h"] = inputs[0].detach()
        return None

    handle = block.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()
    if "h" not in box:
        raise RhoCCAError("pre-block hook did not run")
    hidden = box["h"][0].float().cpu().numpy()
    missing = [int(position) for position in positions if position is None or int(position) >= hidden.shape[0]]
    if missing:
        raise RhoCCAError(f"token positions are outside the residual sequence: {missing[:3]}")
    return np.stack([hidden[int(position)] for position in positions]).astype(np.float64)
