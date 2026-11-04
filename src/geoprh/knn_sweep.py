"""Calibrated mutual kNN at several neighbourhood sizes k, for every layer pair (exp-017 / exp-018).

`layer_calibration` stores one dense 0/1 kNN mask per layer, which is fixed to a single k. For a
k sweep at n = 10240 that would be one [n, n] mask per layer *and* k. Here each layer of A keeps
the rank matrix R_A[i, j] = rank of j among i's neighbours (0 = nearest; >= k_max if beyond), and
each layer of B its neighbour lists I_B [n, k_max] (nearest first). Neighbours are cosine
neighbours on the PRH features, self excluded, as `build_knn_cache` (top-k of x x^T).

With B's samples paired to A's through the permutation pi (A's sample i <-> B's sample pi[i]),
B's neighbours of pi[i], mapped back to A's indices, are pi^-1[I_B[pi[i]]], and the mutual-kNN
overlap of sample i at size k is

    o_k(i) = #{ j in the first k of pi^-1[I_B[pi[i], :]] : R_A[i, j] < k } / k,

so one gather of R_A per (layer pair, permutation) gives every k at once. The score is the mean of
o_k over samples, identical to `layer_calibration.scores` (<M_A, M_B>_F / (n k)) at the same k.
Each cell is calibrated against its own permutation null and the max over cells against the null
of the max (Algorithm 1 / 2 of Groger et al.), the same permutations for every k.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geoprh.layer_calibration import calibrate_cells


@dataclass
class SweepCache:
    """Per-model neighbour data for every layer: ranks [L, n, n] (int16) and lists [L, n, k_max]."""

    ranks: torch.Tensor
    idx: torch.Tensor

    @property
    def k_max(self) -> int:
        return self.idx.shape[2]


def neighbours(x: torch.Tensor, k_max: int, chunk: int = 2048) -> torch.Tensor:
    """[n, d] l2-normalised features -> [n, k_max] cosine neighbours, nearest first, self excluded."""
    n = x.shape[0]
    out = torch.empty(n, k_max, dtype=torch.long, device=x.device)
    for s in range(0, n, chunk):
        sim = x[s : s + chunk] @ x.T
        rows = torch.arange(s, min(s + chunk, n), device=x.device)
        sim[rows - s, rows] = float("-inf")
        out[s : s + chunk] = sim.topk(k_max, dim=1).indices
    return out


def build_sweep_cache(layers: list[torch.Tensor], k_max: int) -> SweepCache:
    """layers: list of [n, d] PRH-prepared (l2-normalised) features, all on one device."""
    n = layers[0].shape[0]
    assert k_max < 32767 and k_max < n
    idx = torch.stack([neighbours(x, k_max) for x in layers])
    ranks = torch.full((len(layers), n, n), k_max, dtype=torch.int16, device=idx.device)
    order = torch.arange(k_max, dtype=torch.int16, device=idx.device).expand(n, k_max)
    for layer in range(len(layers)):
        ranks[layer].scatter_(1, idx[layer], order)
    return SweepCache(ranks=ranks, idx=idx)


def _mapped(b: SweepCache, perm: torch.Tensor | None) -> torch.Tensor:
    """[Lb, n, k_max]: B's neighbour lists of each A sample's partner, in A's indices."""
    if perm is None:
        return b.idx
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv[b.idx[:, perm]]


def overlaps(a: SweepCache, b: SweepCache, ks: list[int], perm=None) -> torch.Tensor:
    """Per-sample overlaps [len(ks), La, Lb, n] (float32)."""
    assert max(ks) <= min(a.k_max, b.k_max)
    mapped = _mapped(b, perm)
    la, lb, n = a.ranks.shape[0], mapped.shape[0], mapped.shape[1]
    out = torch.empty(len(ks), la, lb, n, device=mapped.device)
    for j in range(lb):
        for i in range(la):
            r = a.ranks[i].gather(1, mapped[j])  # [n, k_max]: A-rank of each mapped B neighbour
            for t, k in enumerate(ks):
                out[t, i, j] = (r[:, :k] < k).sum(1).float() / k
    return out


def scores(a: SweepCache, b: SweepCache, ks: list[int], perm=None) -> torch.Tensor:
    """Mean overlaps [len(ks), La, Lb]."""
    return overlaps(a, b, ks, perm).mean(-1)


def calibrated_sweep(
    a: SweepCache,
    b: SweepCache,
    ks: list[int],
    permutations: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """{k: calibrate_cells(...) plus max_* entries (null of the max)}; tensors on the CPU."""
    n = a.idx.shape[1]
    gen = torch.Generator(device=a.idx.device).manual_seed(seed)
    observed = scores(a, b, ks)
    null = torch.stack(
        [
            scores(a, b, ks, torch.randperm(n, generator=gen, device=a.idx.device))
            for _ in range(permutations)
        ],
        1,
    )  # [len(ks), P, La, Lb]
    out = {}
    for t, k in enumerate(ks):
        cells = {m: v.cpu() for m, v in calibrate_cells(observed[t], null[t], alpha).items()}
        mx = calibrate_cells(
            observed[t].max().view(1, 1), null[t].flatten(1).max(1).values.view(-1, 1, 1), alpha
        )
        out[k] = {**cells, **{f"max_{m}": float(v) for m, v in mx.items()}}
    return out
