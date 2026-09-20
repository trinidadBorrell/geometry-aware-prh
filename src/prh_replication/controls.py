"""Metric controls and shuffle nulls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prh_replication.metrics import (
    center_gram,
    linear_cka,
    linear_gram,
    mutual_knn_score,
    permutation_cka_mean_formula,
    rbf_cka,
    rbf_gram,
)


@dataclass
class ShuffleResult:
    actual: float
    shuffle_mean: float
    shuffle_std: float
    excess: float
    n_perm: int
    formula_mean: float | None = None


def one_sided_permute(feats: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return feats[perm]


def shuffle_null(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    score_fn,
    n_perm: int = 100,
    seed: int = 0,
) -> ShuffleResult:
    g = torch.Generator(device="cpu").manual_seed(seed)
    actual = float(score_fn(feats_a, feats_b))
    scores = []
    n = feats_b.shape[0]
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=g)
        scores.append(float(score_fn(feats_a, feats_b[perm])))
    t = torch.tensor(scores)
    mean = float(t.mean())
    std = float(t.std(unbiased=True)) if n_perm > 1 else 0.0
    return ShuffleResult(actual=actual, shuffle_mean=mean, shuffle_std=std, excess=actual - mean, n_perm=n_perm)


def mknn_shuffle(feats_a, feats_b, topk=10, n_perm=100, seed=0) -> ShuffleResult:
    return shuffle_null(feats_a, feats_b, lambda x, y: mutual_knn_score(x, y, topk), n_perm, seed)


def linear_cka_shuffle(feats_a, feats_b, n_perm=100, seed=0) -> ShuffleResult:
    res = shuffle_null(feats_a, feats_b, linear_cka, n_perm, seed)
    kc = center_gram(linear_gram(feats_a.float()))
    lc = center_gram(linear_gram(feats_b.float()))
    res.formula_mean = permutation_cka_mean_formula(kc, lc)
    return res


def rbf_cka_shuffle(feats_a, feats_b, sigma_a, sigma_b, n_perm=100, seed=0) -> ShuffleResult:
    return shuffle_null(
        feats_a,
        feats_b,
        lambda x, y: rbf_cka(x, y, sigma_a, sigma_b),
        n_perm,
        seed,
    )


def orthogonal_transform(feats: torch.Tensor, seed: int = 1) -> torch.Tensor:
    d = feats.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, r = torch.linalg.qr(a)
    signs = torch.diag(r).sign()
    signs[signs == 0] = 1
    q = q * signs
    return feats.float() @ q


def control_battery(feats_a: torch.Tensor, feats_b: torch.Tensor, topk: int = 10) -> dict:
    n, da = feats_a.shape
    g = torch.Generator().manual_seed(0)
    rand_a = torch.randn(n, da, generator=g)
    rand_b = torch.randn(n, feats_b.shape[1], generator=g)
    perm = torch.randperm(n, generator=g)
    joint = torch.randperm(n, generator=g)
    scale = 3.7
    ortho = orthogonal_transform(feats_a, seed=2)
    tiny_sigma = 1e-6
    return {
        "self_mknn": mutual_knn_score(feats_a, feats_a, topk),
        "self_linear_cka": linear_cka(feats_a, feats_a),
        "ortho_linear_cka": linear_cka(feats_a, ortho),
        "rescale_linear_cka": linear_cka(feats_a, feats_a * scale),
        "joint_perm_mknn": mutual_knn_score(feats_a[joint], feats_b[joint], topk),
        "one_sided_perm_mknn": mutual_knn_score(feats_a, feats_b[perm], topk),
        "random_mknn": mutual_knn_score(rand_a, rand_b, topk),
        "random_linear_cka": linear_cka(rand_a, rand_b),
        "narrow_rbf_cka": rbf_cka(feats_a, feats_b, tiny_sigma, tiny_sigma),
        "narrow_rbf_self": rbf_cka(feats_a, feats_a, tiny_sigma, tiny_sigma),
        "unbiased_linear_cka": linear_cka(feats_a, feats_b, unbiased=True),
    }
