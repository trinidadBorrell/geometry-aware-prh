"""Train/val/eval pipeline for the learned-kernel extension."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any

import torch

from prh_replication.extract import feature_path
from prh_replication.io_utils import load_features, write_json
from prh_replication.kernels import (
    ALPHA_GRID,
    LAMBDA_GRID,
    extension_stats,
    gram_diagnostics,
    is_boundary,
    kernel_from_spec,
    linear_gram,
    mass_outside_interval,
    mc_cka_mean,
    median_offdiag_from_dsq,
    objective_value,
    pairwise_sq_distances,
    profile_inner_products,
    profile_nodes,
    rbf_profile,
    rq_profile,
    u_centred_cka,
)
from prh_replication.prh_ref import stack_prepared
from prh_replication.registry import MODELS, Paths


LANG_KEYS = [
    "bloomz-560m",
    "bloomz-1b1",
    "qwen3-0.6b-base",
    "qwen3-1.7b-base",
    "qwen2.5-0.5b",
    "olmo-1b-0724",
]
VIS_KEYS = ["dinov2-small", "vit-in21k-small", "clip-laion-base"]
OBJECTIVES = ("cka", "ratio", "excess")
FAMILIES = ("rbf", "rq")


def vl_pairs() -> list[tuple[str, str]]:
    return [(a, b) for a in LANG_KEYS for b in VIS_KEYS]


def pair_name(a: str, b: str) -> str:
    return f"{a}__{b}"


def load_split_features(paths: Paths, key: str, split: str, sample_ids: list[str]) -> dict:
    spec = MODELS[key]
    p = feature_path(paths, spec, "coco_val2017", split, "original", sample_ids)
    if not p.exists():
        raise FileNotFoundError(p)
    obj = load_features(p)
    if list(obj["sample_ids"]) != list(sample_ids):
        raise ValueError(f"id mismatch {key} {split}")
    return obj


def prepared_layers(obj: dict) -> torch.Tensor:
    return stack_prepared(obj["feats"])


def select_train_layers(xa: torch.Tensor, xb: torch.Tensor) -> dict:
    grams_a = [center_and_norm(linear_gram(xa[:, i])) for i in range(xa.shape[1])]
    grams_b = [center_and_norm(linear_gram(xb[:, j])) for j in range(xb.shape[1])]
    best, best_ij = float("-inf"), (0, 0)
    best_a = float("nan")
    for i, (ka, nka) in enumerate(grams_a):
        if nka <= 0:
            continue
        for j, (kb, nkb) in enumerate(grams_b):
            if nkb <= 0:
                continue
            a = float((ka * kb).sum() / (nka * nkb))
            if a > best or (a == best and (i, j) < best_ij):
                best, best_ij, best_a = a, (i, j), a
    return {"layers": list(best_ij), "train_linear_cka": best_a}


def center_and_norm(k: torch.Tensor) -> tuple[torch.Tensor, float]:
    from prh_replication.kernels import center_gram_o2

    kc = center_gram_o2(k)
    return kc, float(torch.linalg.norm(kc, ord="fro"))


def frozen_xy(xa_all: torch.Tensor, xb_all: torch.Tensor, layers: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return xa_all[:, layers[0]], xb_all[:, layers[1]]


def candidate_list(lambda_grid=LAMBDA_GRID, alpha_grid=ALPHA_GRID) -> list[dict]:
    out = []
    for lam in lambda_grid:
        out.append({"family": "rbf", "lambda": float(lam), "alpha": None})
        for alpha in alpha_grid:
            out.append({"family": "rq", "lambda": float(lam), "alpha": float(alpha)})
    return out


def score_candidate(xa, xb, dsq_a, dsq_b, sa, sb, cand) -> dict:
    fam = cand["family"]
    lam = cand["lambda"]
    alpha = cand["alpha"]
    if not math.isfinite(sa) or sa <= 0 or not math.isfinite(sb) or sb <= 0:
        return {**cand, "valid_a": False, "degenerate": True, "a": float("nan"), "b": float("nan"),
                "ratio": float("nan"), "excess": float("nan"), "zero_scale": True}
    sigma_a, sigma_b = lam * sa, lam * sb
    ka = kernel_from_spec(dsq_a, xa, fam, sigma_a, alpha)
    kb = kernel_from_spec(dsq_b, xb, fam, sigma_b, alpha)
    st = extension_stats(ka, kb)
    st.update(cand)
    st["sigma_a"] = sigma_a
    st["sigma_b"] = sigma_b
    st["zero_scale"] = False
    st["boundary"] = is_boundary(fam, lam, alpha)
    return st


def score_fixed(xa, xb, dsq_a, dsq_b, sa, sb, name: str) -> dict:
    if name == "linear":
        st = extension_stats(linear_gram(xa), linear_gram(xb))
        st.update({"family": "linear", "lambda": None, "alpha": None, "name": name})
        return st
    if name == "rbf_raw_sigma_1":
        ka = kernel_from_spec(dsq_a, xa, "rbf", 1.0, None)
        kb = kernel_from_spec(dsq_b, xb, "rbf", 1.0, None)
        st = extension_stats(ka, kb)
        st.update({"family": "rbf", "lambda": None, "alpha": None, "sigma_a": 1.0, "sigma_b": 1.0, "name": name})
        return st
    if name == "rbf_lambda_1":
        cand = {"family": "rbf", "lambda": 1.0, "alpha": None}
        st = score_candidate(xa, xb, dsq_a, dsq_b, sa, sb, cand)
        st["name"] = name
        return st
    raise ValueError(name)


def select_from_landscape(rows: list[dict], family: str, objective: str) -> dict:
    ranked = []
    for r in rows:
        if r.get("family") != family:
            continue
        val = objective_value(r, objective)
        if not math.isfinite(val):
            continue
        lam = r["lambda"]
        alpha = r["alpha"] if r["alpha"] is not None else 0.0
        ranked.append((val, lam, alpha, r))
    if not ranked:
        return {"family": family, "objective": objective, "valid": False}
    ranked.sort(key=lambda t: (-t[0], t[1], t[2]))
    best = ranked[0][3]
    return {
        "family": family,
        "objective": objective,
        "valid": True,
        "lambda": best["lambda"],
        "alpha": best["alpha"],
        "train_objective": ranked[0][0],
        "train_a": best["a"],
        "train_b": best["b"],
        "train_ratio": best["ratio"],
        "train_excess": best["excess"],
        "boundary": best.get("boundary", False),
        "n_valid_candidates": len(ranked),
        "n_total_family": sum(1 for r in rows if r.get("family") == family),
    }


def score_grams(xa, xb, dsq_a, dsq_b, sa, sb, family, lam, alpha, name=None):
    if family == "linear":
        st = extension_stats(linear_gram(xa), linear_gram(xb))
        st.update({"family": "linear", "lambda": None, "alpha": None})
        return st
    if name == "rbf_raw_sigma_1":
        sigma_a = sigma_b = 1.0
    else:
        if not math.isfinite(sa) or sa <= 0 or not math.isfinite(sb) or sb <= 0:
            return {"valid_a": False, "zero_scale": True, "family": family, "a": float("nan")}
        sigma_a, sigma_b = lam * sa, lam * sb
    ka = kernel_from_spec(dsq_a, xa, family, sigma_a, alpha)
    kb = kernel_from_spec(dsq_b, xb, family, sigma_b, alpha)
    st = extension_stats(ka, kb)
    st.update({"family": family, "lambda": lam, "alpha": alpha, "sigma_a": sigma_a, "sigma_b": sigma_b})
    return st


def pack_eval(st: dict, extra: dict | None = None) -> dict:
    out = {
        "a": st.get("a"),
        "b": st.get("b"),
        "ratio": st.get("ratio"),
        "excess": st.get("excess"),
        "official_cka": st.get("official_cka"),
        "valid_a": st.get("valid_a"),
        "degenerate": st.get("degenerate"),
        "ratio_undefined": st.get("ratio_undefined"),
    }
    if extra:
        out.update(extra)
    return out


def evaluate_kernel_full(xa, xb, dsq_a, dsq_b, sa, sb, family, lam, alpha, n_perm, seed, name=None, diagnostics=True):
    if family == "linear":
        k, l = linear_gram(xa), linear_gram(xb)
        st = extension_stats(k, l)
        spec = {"family": "linear", "lambda": None, "alpha": None}
    else:
        if name == "rbf_raw_sigma_1":
            sigma_a = sigma_b = 1.0
        else:
            if not math.isfinite(sa) or sa <= 0 or not math.isfinite(sb) or sb <= 0:
                return {"valid_a": False, "zero_scale": True, "family": family}
            sigma_a, sigma_b = lam * sa, lam * sb
        k = kernel_from_spec(dsq_a, xa, family, sigma_a, alpha)
        l = kernel_from_spec(dsq_b, xb, family, sigma_b, alpha)
        st = extension_stats(k, l)
        spec = {"family": family, "lambda": lam, "alpha": alpha, "sigma_a": sigma_a, "sigma_b": sigma_b}
    extra = {**spec, "boundary": is_boundary(family, lam, alpha)}
    if diagnostics:
        extra.update(gram_diagnostics(k))
        extra.update(u_centred_cka(k, l))
        extra.update(mc_cka_mean(k, l, n_perm, seed))
    out = pack_eval(st, extra)
    if name:
        out["name"] = name
    return out


def profile_of(family: str, lam: float, alpha: float | None) -> torch.Tensor:
    r = profile_nodes()
    if family == "rbf":
        return rbf_profile(r, lam)
    return rq_profile(r, lam, float(alpha))


def subset_indices(n: int, subset_size: int, n_subsets: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    out = []
    for i in range(n_subsets):
        sl = perm[i * subset_size : (i + 1) * subset_size]
        out.append(sl)
    return out


def share_tag(src: tuple[str, str], tgt: tuple[str, str]) -> str:
    sa, sb = src
    ta, tb = tgt
    share_l = sa == ta
    share_v = sb == tb
    if share_l and share_v:
        return "same_pair"
    if share_l:
        return "shared_language"
    if share_v:
        return "shared_vision"
    return "neither"


def jsonable(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x
