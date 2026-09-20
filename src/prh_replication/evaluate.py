"""Pairwise alignment evaluation with frozen splits."""

from __future__ import annotations

from itertools import combinations, product

import torch

from prh_replication.controls import ShuffleResult, linear_cka_shuffle, mknn_shuffle, rbf_cka_shuffle
from prh_replication.io_utils import load_features, write_json
from prh_replication.metrics import (
    apply_layer_prep,
    linear_cka,
    max_cka_over_layers,
    max_mknn_over_layers,
    median_offdiag_distance,
    mutual_knn_score,
    rbf_cka,
    relative_layer_indices,
)
from prh_replication.registry import (
    MKNN_K,
    MODERN_LAYER_FRACTIONS,
    N_PERM,
    RBF_MULTIPLIERS,
    SEED,
    ModelSpec,
    Paths,
)


def _layers(obj, protocol: str, indices: list[int] | None) -> torch.Tensor:
    feats = obj["feats"].float()
    if indices is not None:
        feats = feats[:, indices]
    prepped = torch.stack([apply_layer_prep(feats[:, i], protocol) for i in range(feats.shape[1])], dim=1)
    return prepped


def _score_pair(x: torch.Tensor, y: torch.Tensor, metric: str, **kw) -> float:
    if metric == "mknn":
        return mutual_knn_score(x, y, kw.get("topk", MKNN_K))
    if metric == "linear_cka":
        return linear_cka(x, y)
    if metric == "rbf_cka":
        return rbf_cka(x, y, kw["sigma_a"], kw["sigma_b"])
    raise ValueError(metric)


def evaluate_pair(
    a: dict,
    b: dict,
    protocol: str,
    split: str,
    sigma_a: dict[float, float] | None = None,
    sigma_b: dict[float, float] | None = None,
    layer_mode: str = "max",
    val_choice: tuple[int, int] | None = None,
) -> dict:
    ids_a, ids_b = list(a["sample_ids"]), list(b["sample_ids"])
    if ids_a != ids_b:
        raise ValueError("sample_ids mismatch between feature caches")
    if protocol == "original" or layer_mode == "max":
        xa, xb = _layers(a, protocol, None), _layers(b, protocol, None)
        results = {}
        score, ij = max_mknn_over_layers(xa, xb, MKNN_K)
        sh = mknn_shuffle(xa[:, ij[0]], xb[:, ij[1]], MKNN_K, N_PERM, SEED)
        results["mknn"] = _pack(score, ij, sh, layer_mode="max_over_layers")
        score, ij = max_cka_over_layers(xa, xb, kind="linear")
        sh = linear_cka_shuffle(xa[:, ij[0]], xb[:, ij[1]], N_PERM, SEED)
        results["linear_cka"] = _pack(score, ij, sh, layer_mode="max_over_layers")
        if sigma_a and sigma_b:
            rbf = {}
            for m in RBF_MULTIPLIERS:
                score, ij = max_cka_over_layers(xa, xb, kind="rbf", sigma_a=sigma_a[m], sigma_b=sigma_b[m])
                sh = rbf_cka_shuffle(xa[:, ij[0]], xb[:, ij[1]], sigma_a[m], sigma_b[m], N_PERM, SEED)
                rbf[str(m)] = _pack(score, ij, sh, layer_mode="max_over_layers")
            results["rbf_cka"] = rbf
        results["n"] = len(ids_a)
        results["split"] = split
        results["protocol"] = protocol
        return results

    # modern: relative-depth grid; optional frozen val layer pair
    ia = relative_layer_indices(a["feats"].shape[1], MODERN_LAYER_FRACTIONS)
    ib = relative_layer_indices(b["feats"].shape[1], MODERN_LAYER_FRACTIONS)
    xa, xb = _layers(a, protocol, ia), _layers(b, protocol, ib)
    results = {"layer_indices_a": ia, "layer_indices_b": ib}
    grid = {}
    for i, j in product(range(len(ia)), range(len(ib))):
        grid[f"{ia[i]}-{ib[j]}"] = {
            "mknn": mutual_knn_score(xa[:, i], xb[:, j], MKNN_K),
            "linear_cka": linear_cka(xa[:, i], xb[:, j]),
        }
    results["grid"] = grid
    if val_choice is not None:
        i, j = val_choice
        # map original layer ids to tensor columns
        ci, cj = ia.index(i), ib.index(j)
        xa_best, xb_best = xa[:, ci], xb[:, cj]
        sh = mknn_shuffle(xa_best, xb_best, MKNN_K, N_PERM, SEED)
        results["mknn"] = _pack(sh.actual, (i, j), sh, layer_mode="frozen_val_choice")
        shc = linear_cka_shuffle(xa_best, xb_best, N_PERM, SEED)
        results["linear_cka"] = _pack(shc.actual, (i, j), shc, layer_mode="frozen_val_choice")
        if sigma_a and sigma_b:
            rbf = {}
            for m in RBF_MULTIPLIERS:
                shr = rbf_cka_shuffle(xa_best, xb_best, sigma_a[m], sigma_b[m], N_PERM, SEED)
                rbf[str(m)] = _pack(shr.actual, (i, j), shr, layer_mode="frozen_val_choice")
            results["rbf_cka"] = rbf
    results["n"] = len(ids_a)
    results["split"] = split
    results["protocol"] = protocol
    return results


def _pack(score: float, ij: tuple[int, int], sh: ShuffleResult, layer_mode: str) -> dict:
    return {
        "score": score,
        "layers": list(ij),
        "shuffle_mean": sh.shuffle_mean,
        "shuffle_std": sh.shuffle_std,
        "excess": sh.excess,
        "n_perm": sh.n_perm,
        "formula_mean": sh.formula_mean,
        "layer_mode": layer_mode,
        "null_label": "one-sided correspondence permutation null (not a CI for the observed score)",
    }


def train_sigmas(feat_obj: dict, protocol: str) -> dict[float, float]:
    last = feat_obj["feats"].shape[1] - 1
    x = apply_layer_prep(feat_obj["feats"][:, last].float(), protocol)
    med = median_offdiag_distance(x)
    return {m: m * med for m in RBF_MULTIPLIERS}


def choose_val_layers(val_grid: dict) -> tuple[int, int]:
    best = None
    best_s = float("-inf")
    for key, vals in val_grid.items():
        if vals["mknn"] > best_s:
            best_s = vals["mknn"]
            a, b = key.split("-")
            best = (int(a), int(b))
    assert best is not None
    return best
