"""Calibrated alignment for every layer pair at once (exp-012, WP1).

The PRH and Aristotelian code calibrate the *max* over the layer x layer score matrix (Algorithm 2
of Groger et al.). Relating alignment to per-layer geometry needs every cell calibrated, so here
each cell (i, j) gets its own permutation null: the same row permutation of model B is applied to
all of B's layers (as in Algorithm 2), and each cell is calibrated against its own null samples
exactly as `calibrated_similarity.calibrate` (Algorithm 1) does for one pair: tau = the
ceil((1 - alpha)(P + 1))-th order statistic of {observed} + null, the add-one p-value, and
g = clip((S - tau) / (1 - tau), 0, 1).

Both metrics are computed for all layer pairs by one matrix product per permutation, on the PRH
features (q = 0.95 clamp, l2 norm), with Aristotelian's caches:
- mutual kNN (k = 10, cosine neighbours): with M_l the n x n 0/1 kNN mask of layer l
  (`build_knn_cache`), the mean overlap of layers i and j is <M_i, M_j>_F / (n k);
- linear CKA (biased HSIC): with Kc_l = H K_l H the centred Gram matrix (`build_gram_cache`),
  CKA = <Kc_i, Kc_j>_F / (||Kc_i||_F ||Kc_j||_F).
Permuting B's rows by pi maps M to M[pi][:, pi] (and Kc likewise), so each permutation is one
gather and one product. Masks and Gram matrices are stored as float32 [L, n, n].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from aristotelian.experiments.layerwise_engine import build_gram_cache, build_knn_cache

METRICS = ("mutual_knn", "cka_lin")


@dataclass
class ModelCache:
    """Per-model operands: kNN masks [L, n, n] and unit-norm centred Gram matrices [L, n, n]."""

    masks: torch.Tensor
    grams: torch.Tensor
    k: int

    @property
    def num_layers(self) -> int:
        return self.masks.shape[0]


def build_cache(feats: torch.Tensor, k: int = 10) -> ModelCache:
    """feats: [n, L, d] PRH-prepared features (any device)."""
    _, masks = build_knn_cache(feats, topk=k, normalize=True)
    masks = torch.stack([m.float() for m in masks])
    grams = torch.stack(build_gram_cache(feats, normalize=True, kernel="linear")).float()
    grams = grams - grams.mean(dim=1, keepdim=True)
    grams = grams - grams.mean(dim=2, keepdim=True)  # H K H
    grams = grams / grams.flatten(1).norm(dim=1).view(-1, 1, 1)
    return ModelCache(masks=masks, grams=grams, k=k)


def scores(a: ModelCache, b: ModelCache, perm: torch.Tensor | None = None) -> dict:
    """{metric: [La, Lb]} for A against B (B's rows permuted by `perm` if given)."""
    n = a.masks.shape[1]
    out = {}
    for metric, xa, xb in (("mutual_knn", a.masks, b.masks), ("cka_lin", a.grams, b.grams)):
        if perm is not None:
            xb = xb[:, perm][:, :, perm]
        s = xa.flatten(1) @ xb.flatten(1).T
        out[metric] = s / (n * a.k) if metric == "mutual_knn" else s
    return out


def calibrate_cells(observed: torch.Tensor, null: torch.Tensor, alpha: float = 0.05) -> dict:
    """Per-cell Algorithm 1. observed [La, Lb], null [P, La, Lb] -> dict of [La, Lb] tensors."""
    p_count = null.shape[0]
    combined = torch.cat([observed[None], null]).sort(dim=0).values
    idx = min(max(math.ceil((1 - alpha) * (p_count + 1)) - 1, 0), p_count)
    tau = combined[idx]
    pval = (1 + (null >= observed[None]).sum(dim=0)) / (p_count + 1)
    g = ((observed - tau) / (1 - tau).clamp_min(1e-12)).clamp(0, 1)
    return {
        "score": observed,
        "mu0": null.mean(dim=0),
        "sd0": null.std(dim=0),
        "tau": tau,
        "p": pval,
        "g": g,
    }


def calibrated_pair(
    a: ModelCache, b: ModelCache, permutations: int = 200, alpha: float = 0.05, seed: int = 0
) -> dict:
    """{metric: calibrate_cells(...)} for every layer pair of A x B."""
    n = a.masks.shape[1]
    gen = torch.Generator(device=a.masks.device).manual_seed(seed)
    observed = scores(a, b)
    nulls = {m: [] for m in METRICS}
    for _ in range(permutations):
        perm = torch.randperm(n, generator=gen, device=a.masks.device)
        for m, s in scores(a, b, perm).items():
            nulls[m].append(s)
    return {m: calibrate_cells(observed[m], torch.stack(nulls[m]), alpha) for m in METRICS}
