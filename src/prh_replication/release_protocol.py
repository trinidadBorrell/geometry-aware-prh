"""Shared load/eval helpers for the release-alignment runners.

Scoring here is read-only with respect to fitting: ``eval_onesided`` builds
ambient Grams and calls ``extension_stats`` (float32 Gram path). Contracted
``scores_numpy`` is float64. They can differ at ~1e-5; that is a known
precision gap, not a second estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from prh_replication.anisotropic_kernels import (
    centred_diag_energy,
    knn_from_sq,
    knn_overlap,
    make_pack,
    metric_gram,
    metric_sq_distances,
    mnn_from_knn,
)
from prh_replication.datasets import load_manifest
from prh_replication.io_utils import jsonable
from prh_replication.kernels import center_gram_o2, extension_stats, linear_gram, mc_cka_mean
from prh_replication.metrics import mutual_knn_score
from prh_replication.registry import Paths

VIS = ["dinov2-small", "vit-in21k-small", "clip-laion-base"]


def load_splits(paths: Paths, repo: Path):
    man_path = paths.data / "manifests" / "coco_val2017.json"
    slim_path = repo / "data" / "manifests" / "coco_val2017_splits.json"
    try:
        man = load_manifest(man_path)
        if "splits" not in man:
            raise ValueError("missing splits")
    except (OSError, ValueError, json.JSONDecodeError):
        man = load_manifest(slim_path)
    for split in ("train", "val", "test"):
        ids = list(man["splits"][split])
        assert len(ids) == len(set(ids)), split
    train, val, test = [list(man["splits"][s]) for s in ("train", "val", "test")]
    assert not (set(train) & set(val) | set(train) & set(test) | set(val) & set(test))
    return man, {"train": train, "val": val, "test": test}


def dump_fit(fit: dict) -> dict:
    keep = dict(fit)
    for k in ("b_a", "s_a", "eig_a", "b_b", "s_b", "eig_b"):
        if k in keep and isinstance(keep[k], np.ndarray):
            keep[k] = keep[k].tolist()
    return jsonable(keep)


def pack_ab(cache, pca, a, b, split):
    xa, xb = cache[a][split], cache[b][split]
    za = xa - pca[a]["mu"]
    zb = xb - pca[b]["mu"]
    return make_pack(za, zb, pca[a]["U"], pca[b]["U"]), za, zb


def eval_onesided(za, zb, ua, ub, ba, n_perm: int, seed: int, k: int = 10):
    qa = ua.shape[1]
    eye_b = np.eye(ub.shape[1])
    ka = torch.tensor(metric_gram(za, ua, ba), dtype=torch.float64)
    kb = torch.tensor(metric_gram(zb, ub, eye_b), dtype=torch.float64)
    st = extension_stats(ka.float(), kb.float())
    kca = center_gram_o2(ka)
    nkc = float(torch.linalg.norm(kca, ord="fro"))
    trk = float(kca.trace())
    r_eff = (trk * trk / (nkc * nkc)) if nkc > 0 else float("nan")
    mc = mc_cka_mean(ka.float(), kb.float(), n_perm, seed) if n_perm else {}
    dsq_a = metric_sq_distances(za, ua, ba)
    dsq_b = metric_sq_distances(zb, ub, eye_b)
    knn_a, knn_b = knn_from_sq(dsq_a, k), knn_from_sq(dsq_b, k)
    knn_id_a = knn_from_sq(metric_sq_distances(za, ua, np.eye(qa)), k)
    return {
        **{kk: st[kk] for kk in ("a", "b", "ratio", "excess", "official_cka", "valid_a", "degenerate", "ratio_undefined")},
        "r_eff_k": r_eff,
        "centred_diag_energy_frac": centred_diag_energy(kca.numpy()),
        "mnn_k10": mnn_from_knn(knn_a, knn_b),
        "knn_overlap_identity_a": knn_overlap(knn_a, knn_id_a),
        **mc,
    }


def native_pair(xa: np.ndarray, xb: np.ndarray, n_perm: int, seed: int):
    ta = torch.tensor(xa, dtype=torch.float32)
    tb = torch.tensor(xb, dtype=torch.float32)
    k = linear_gram(ta.double())
    l = linear_gram(tb.double())
    st = extension_stats(k, l)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(tb.shape[0], generator=g)
    mnn = mutual_knn_score(ta, tb, 10)
    mnn_shuffle = mutual_knn_score(ta, tb[perm], 10)
    mc = mc_cka_mean(k.float(), l.float(), n_perm, seed) if n_perm else {}
    return {
        "cka_a": st["a"],
        "cka_b": st["b"],
        "cka_ratio": st["ratio"],
        "cka_excess": st["excess"],
        "mnn_k10": mnn,
        "mnn_shuffle": mnn_shuffle,
        **mc,
    }


def partner_sets(base_qwen, base_olmo, supp):
    base = base_qwen + base_olmo
    out = {}
    for a in base_qwen:
        out[a] = [x for x in base if x != a] + VIS
    for a in base_olmo:
        out[a] = [x for x in base if x != a] + VIS
    for a in supp:
        out[a] = list(base_olmo) + [x for x in supp if x != a] + VIS
    return out


def native_pairs(base_qwen, base_olmo, supp):
    pairs = []
    for a in base_qwen:
        for b in base_olmo:
            pairs.append((a, b, "cross_family_base"))
    for fam in (base_qwen, base_olmo, supp):
        for i, a in enumerate(fam):
            for b in fam[i + 1 :]:
                tag = "within_supp" if fam is supp else "within_family_base"
                pairs.append((a, b, tag))
    for a in base_qwen + base_olmo + supp:
        for v in VIS:
            panel = "vl_supp" if a in supp else "vl_base"
            pairs.append((a, v, panel))
    for a in supp:
        for b in base_olmo:
            pairs.append((a, b, "supp_qwen_x_olmo_base"))
    seen = set()
    uniq = []
    for a, b, t in pairs:
        key = tuple(sorted((a, b))) + (t,)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((a, b, t))
    return uniq
