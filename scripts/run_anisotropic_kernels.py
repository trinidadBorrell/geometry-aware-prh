#!/usr/bin/env python3
"""Bounded anisotropic metrics on frozen PRH layers.

Writes results/anisotropic_kernels/ only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.anisotropic_kernels import (
    FAMILIES,
    OBJECTIVES,
    TAU_GRID,
    centred_diag_energy,
    fit_metrics,
    fit_pca,
    fro_cosine,
    knn_from_sq,
    knn_overlap,
    make_pack,
    metric_gram,
    metric_sq_distances,
    mnn_from_knn,
    principal_angles,
    scores_numpy,
    select_tau,
    truncation_gram,
)
from prh_replication.datasets import load_manifest
from prh_replication.io_utils import jsonable, skip_if_complete, write_json
from prh_replication.kernel_experiment import (
    frozen_xy,
    load_split_features,
    pair_name,
    prepared_layers,
    subset_indices,
    vl_pairs,
)
from prh_replication.kernels import center_gram_o2, extension_stats, linear_gram, mc_cka_mean
from prh_replication.metrics import mutual_knn_score
from prh_replication.plots import lines
from prh_replication.registry import Paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--pairs-limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_layers(repo: Path):
    return json.loads((repo / "results" / "learned_kernels" / "layers.json").read_text())


def np_xy(xa: torch.Tensor, xb: torch.Tensor):
    return xa.detach().cpu().numpy().astype(np.float64), xb.detach().cpu().numpy().astype(np.float64)


def centre_train(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return x - mu


def pack_split(xa, xb, ua, ub, mu_a, mu_b):
    return make_pack(centre_train(xa, mu_a), centre_train(xb, mu_b), ua, ub)


def score_b(ba, bb, pack):
    qa, qb = ba.shape[0], bb.shape[0]
    return scores_numpy(ba - np.eye(qa), bb - np.eye(qb), pack)


def dump_fit(fit: dict) -> dict:
    skip = {"pa", "pb"}
    out = {}
    for k, v in fit.items():
        if k in skip:
            continue
        if k in ("b_a", "b_b") and isinstance(v, np.ndarray):
            out[k] = v.tolist()
            continue
        if k in ("eig_a", "eig_b") and isinstance(v, np.ndarray):
            out[k] = v.tolist()
            continue
        out[k] = v
    return jsonable(out)


def eval_selected(za, zb, ua, ub, ba, bb, n_perm: int, seed: int, k: int = 10):
    ka = torch.tensor(metric_gram(za, ua, ba), dtype=torch.float64)
    kb = torch.tensor(metric_gram(zb, ub, bb), dtype=torch.float64)
    st = extension_stats(ka.float(), kb.float())
    kca = center_gram_o2(ka)
    nkc = float(torch.linalg.norm(kca, ord="fro"))
    trk = float(kca.trace())
    r_eff = (trk * trk / (nkc * nkc)) if nkc > 0 else float("nan")
    diag_e = centred_diag_energy(kca.numpy())
    mc = mc_cka_mean(ka.float(), kb.float(), n_perm, seed) if n_perm else {}
    dsq_a = metric_sq_distances(za, ua, ba)
    dsq_b = metric_sq_distances(zb, ub, bb)
    knn_a = knn_from_sq(dsq_a, k)
    knn_b = knn_from_sq(dsq_b, k)
    dsq_id_a = metric_sq_distances(za, ua, np.eye(ua.shape[1]))
    dsq_id_b = metric_sq_distances(zb, ub, np.eye(ub.shape[1]))
    knn_id_a = knn_from_sq(dsq_id_a, k)
    knn_id_b = knn_from_sq(dsq_id_b, k)
    return {
        **{kk: st[kk] for kk in ("a", "b", "ratio", "excess", "official_cka", "valid_a", "degenerate", "ratio_undefined")},
        "r_eff_k": r_eff,
        "centred_diag_energy_frac": diag_e,
        "mnn_k10": mnn_from_knn(knn_a, knn_b),
        "knn_overlap_identity_a": knn_overlap(knn_a, knn_id_a),
        "knn_overlap_identity_b": knn_overlap(knn_b, knn_id_b),
        **mc,
    }


def main():
    args = parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out = paths.results / "anisotropic_kernels"
    out.mkdir(parents=True, exist_ok=True)
    if skip_if_complete(out, force=args.force, smoke=args.smoke):
        return
    design = json.loads((ROOT / "configs" / "anisotropic_kernels.json").read_text())
    write_json(out / "design.json", design)

    q = 8 if args.smoke else int(design["q"])
    taus = [0.0, 0.1] if args.smoke else list(TAU_GRID)
    families = FAMILIES
    n_steps = 40 if args.smoke else 120
    n_perm = 8 if args.smoke else args.n_perm
    k_mnn = int(design["mnn_k"])

    man_path = paths.data / "manifests" / "coco_val2017.json"
    slim_path = ROOT / "data" / "manifests" / "coco_val2017_splits.json"
    try:
        man = load_manifest(man_path)
        if "splits" not in man:
            raise ValueError("manifest missing splits")
    except (OSError, ValueError, json.JSONDecodeError):
        man = load_manifest(slim_path)
    layers_tbl = load_layers(ROOT)
    pairs = vl_pairs()
    if args.smoke:
        pairs = pairs[:1]
    if args.pairs_limit:
        pairs = pairs[: args.pairs_limit]

    lk_eval = {}
    lk_path = ROOT / "results" / "learned_kernels" / "pair_eval.json"
    if lk_path.exists():
        for rec in json.loads(lk_path.read_text()):
            lk_eval[pair_name(rec["a"], rec["b"])] = rec
    prh_tbl = {}
    prh_path = ROOT / "results" / "prh_released_code" / "pair_table.json"
    if prh_path.exists():
        for rec in json.loads(prh_path.read_text()):
            prh_tbl[pair_name(rec["a"], rec["b"])] = rec

    t0 = time.time()
    cache = {}
    pca_tbl = {}
    for a, b in pairs:
        name = pair_name(a, b)
        ly = layers_tbl[name]["layers"]
        splits = {}
        for split in ("train", "val", "test"):
            ids = list(man["splits"][split])
            fa = load_split_features(paths, a, split, ids)
            fb = load_split_features(paths, b, split, ids)
            xa, xb = frozen_xy(prepared_layers(fa), prepared_layers(fb), ly)
            splits[split] = np_xy(xa, xb)
            print(f"load {name} {split} n={xa.shape[0]} d={xa.shape[1]},{xb.shape[1]}", flush=True)
        xa_tr, xb_tr = splits["train"]
        mu_a, mu_b = xa_tr.mean(0), xb_tr.mean(0)
        za_tr, zb_tr = xa_tr - mu_a, xb_tr - mu_b
        pca_a = fit_pca(za_tr, q=q)
        pca_b = fit_pca(zb_tr, q=q)
        pca_tbl[name] = {
            "layers": ly,
            "a": a,
            "b": b,
            "mu_a": mu_a,
            "mu_b": mu_b,
            "pca_a": {k: pca_a[k] for k in ("q", "q_requested", "numerical_rank", "variance_fraction", "d", "reduced")},
            "pca_b": {k: pca_b[k] for k in ("q", "q_requested", "numerical_rank", "variance_fraction", "d", "reduced")},
            "U_a": pca_a["U"],
            "U_b": pca_b["U"],
        }
        packs = {}
        for split in ("train", "val", "test"):
            packs[split] = pack_split(splits[split][0], splits[split][1], pca_a["U"], pca_b["U"], mu_a, mu_b)
        cache[name] = {"splits": splits, "packs": packs, "meta": pca_tbl[name]}
        print("elapsed", name, time.time() - t0, flush=True)

    write_json(
        out / "pca.json",
        jsonable({n: {k: v for k, v in rec.items() if k not in ("U_a", "U_b", "mu_a", "mu_b")} for n, rec in pca_tbl.items()}),
    )

    fits = {}
    pair_eval = []
    t1 = time.time()
    for a, b in pairs:
        name = pair_name(a, b)
        meta = cache[name]["meta"]
        packs = cache[name]["packs"]
        ua, ub = meta["U_a"], meta["U_b"]
        mu_a, mu_b = meta["mu_a"], meta["mu_b"]
        lin_tr = score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), packs["train"])
        lin_va = score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), packs["val"])
        lin_te = score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), packs["test"])
        if not (lin_tr["valid_a"] and lin_tr["a"] > 1e-8 and math.isfinite(lin_tr["ratio"]) and lin_tr["ratio"] > 1e-8):
            print("skip unsound linear baseline", name, lin_tr, flush=True)
            continue
        xa_te, xb_te = cache[name]["splits"]["test"]
        za_te, zb_te = xa_te - mu_a, xb_te - mu_b
        xa_tr, xb_tr = cache[name]["splits"]["train"]
        rec = {
            "a": a,
            "b": b,
            "layers": meta["layers"],
            "pca_a": pca_tbl[name]["pca_a"],
            "pca_b": pca_tbl[name]["pca_b"],
            "identity": {
                "train": lin_tr,
                "val": lin_va,
                "test": eval_selected(za_te, zb_te, ua, ub, np.eye(ua.shape[1]), np.eye(ub.shape[1]), n_perm, 0, k_mnn),
            },
            "truncation_q": {},
            "families": {},
            "prh_max_mnn": prh_tbl.get(name, {}).get("mknn"),
            "prh_max_mnn_layers": prh_tbl.get(name, {}).get("mknn_layers"),
            "prh_max_linear_cka": prh_tbl.get(name, {}).get("linear_cka"),
            "learned_kernel_linear_a": None,
            "learned_kernel_rbf_ratio_a": None,
            "learned_kernel_rbf_excess_a": None,
        }
        ktr = torch.tensor(truncation_gram(za_te, ua))
        ltr = torch.tensor(truncation_gram(zb_te, ub))
        rec["truncation_q"]["test"] = jsonable(extension_stats(ktr.float(), ltr.float()))
        rec["identity"]["test_contracted"] = lin_te
        lk = lk_eval.get(name)
        te_lk = (lk or {}).get("splits", {}).get("test", {})
        rec["learned_kernel_linear_a"] = (te_lk.get("fixed") or {}).get("linear", {}).get("a")
        rec["learned_kernel_rbf_ratio_a"] = (te_lk.get("selected") or te_lk).get("rbf_ratio", {}).get("a") if isinstance((te_lk.get("selected") or te_lk).get("rbf_ratio"), dict) else None
        rec["learned_kernel_rbf_excess_a"] = (te_lk.get("selected") or te_lk).get("rbf_excess", {}).get("a") if isinstance((te_lk.get("selected") or te_lk).get("rbf_excess"), dict) else None
        rec["learned_kernel_rbf_sigma1_a"] = (te_lk.get("fixed") or {}).get("rbf_raw_sigma_1", {}).get("a") or (te_lk.get("fixed") or {}).get("rbf_lambda_1", {}).get("a")

        lin_ip_mnn = mutual_knn_score(torch.tensor(xa_te), torch.tensor(xb_te), k_mnn)
        rec["identity"]["test"]["mnn_inner_product"] = lin_ip_mnn

        fam_fits = {}
        for fam in families:
            fam_fits[fam] = {}
            rec["families"][fam] = {}
            for obj in OBJECTIVES:
                cands = {}
                for tau in taus:
                    fit = fit_metrics(packs["train"], fam, obj, tau, lin_tr, n_steps=n_steps)
                    fit["val"] = score_b(fit["b_a"], fit["b_b"], packs["val"])
                    cands[float(tau)] = fit
                    print(
                        f"fit {name} {fam} {obj} tau={tau} train_a={fit['train']['a']:.4f} val_{obj}={fit['val'].get(obj)} R={fit['r']:.4f} start={fit['selected_start']}",
                        flush=True,
                    )
                tau_star, picked = select_tau(cands, packs["val"], obj)
                te = eval_selected(za_te, zb_te, ua, ub, picked["b_a"], picked["b_b"], n_perm, 1, k_mnn)
                te_c = score_b(picked["b_a"], picked["b_b"], packs["test"])
                id_te = rec["identity"]["test"]
                rec["families"][fam][obj] = {
                    "selected_tau": tau_star,
                    "selected_start": picked["selected_start"],
                    "r": picked["r"],
                    "frac_at_bounds": picked["frac_at_bounds"],
                    "cond_a": picked["cond_a"],
                    "cond_b": picked["cond_b"],
                    "eig_a": picked["eig_a"].tolist(),
                    "eig_b": picked["eig_b"].tolist(),
                    "train": picked["train"],
                    "val": picked["val"],
                    "test": te,
                    "test_contracted": te_c,
                    "delta_a_vs_identity": (te["a"] - id_te["a"]) if te.get("a") is not None else None,
                    "delta_ratio_vs_identity": (te["ratio"] - id_te["ratio"]) if te.get("ratio") is not None else None,
                    "delta_excess_vs_identity": (te["excess"] - id_te["excess"]) if te.get("excess") is not None else None,
                    "delta_mnn_vs_identity": (te["mnn_k10"] - id_te["mnn_k10"]) if te.get("mnn_k10") is not None else None,
                    "ok": picked["ok"],
                    "start_spread": picked["start_spread"],
                    "tau_panel_val": {str(t): cands[t]["val"].get(obj) for t in cands},
                    "tau_panel_r": {str(t): cands[t]["r"] for t in cands},
                }
                fam_fits[fam][obj] = {str(t): dump_fit(cands[t]) for t in cands}
                fam_fits[fam][obj]["selected"] = dump_fit(picked)
        fits[name] = fam_fits
        pair_eval.append(rec)
        write_json(out / "fits.json", jsonable(fits))
        write_json(out / "pair_eval.json", jsonable(pair_eval))
        print("eval", name, flush=True)
    print("fit minutes", (time.time() - t1) / 60, flush=True)

    # shuffle-fit
    shuffle_pairs = [tuple(p) for p in design["shuffle_pairs"]]
    if args.smoke:
        shuffle_pairs = shuffle_pairs[:1]
    seeds = design["shuffle_seeds"][: (1 if args.smoke else 3)]
    shuffle_fit = []
    for a, b in shuffle_pairs:
        if (a, b) not in pairs:
            continue
        name = pair_name(a, b)
        meta = cache[name]["meta"]
        ua, ub = meta["U_a"], meta["U_b"]
        mu_a, mu_b = meta["mu_a"], meta["mu_b"]
        xa_tr, xb_tr = cache[name]["splits"]["train"]
        xa_va, xb_va = cache[name]["splits"]["val"]
        xa_te, xb_te = cache[name]["splits"]["test"]
        za_te, zb_te = xa_te - mu_a, xb_te - mu_b
        id_true = score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), cache[name]["packs"]["test"])
        for seed in seeds:
            g = np.random.default_rng(int(seed))
            xb_tr_s = xb_tr[g.permutation(len(xb_tr))]
            g2 = np.random.default_rng(1000 + int(seed))
            xb_va_s = xb_va[g2.permutation(len(xb_va))]
            g3 = np.random.default_rng(2000 + int(seed))
            xb_te_s = xb_te[g3.permutation(len(xb_te))]
            pack_tr = pack_split(xa_tr, xb_tr_s, ua, ub, mu_a, mu_b)
            pack_va = pack_split(xa_va, xb_va_s, ua, ub, mu_a, mu_b)
            pack_te_s = pack_split(xa_te, xb_te_s, ua, ub, mu_a, mu_b)
            lin_tr = score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), pack_tr)
            rec = {"a": a, "b": b, "seed": int(seed), "families": {}, "identity_true_test_a": id_true["a"]}
            for fam in families:
                rec["families"][fam] = {}
                for obj in OBJECTIVES:
                    cands = {}
                    for tau in taus:
                        fit = fit_metrics(pack_tr, fam, obj, tau, lin_tr, n_steps=n_steps)
                        cands[float(tau)] = fit
                    _, picked = select_tau(cands, pack_va, obj)
                    true_te = eval_selected(za_te, zb_te, ua, ub, picked["b_a"], picked["b_b"], 0, 0, k_mnn)
                    shuf_te = score_b(picked["b_a"], picked["b_b"], pack_te_s)
                    rec["families"][fam][obj] = {
                        "selected_tau": picked["selected_tau"],
                        "r": picked["r"],
                        "true_test": true_te,
                        "shuffled_test": shuf_te,
                        "delta_a_true_vs_identity": true_te["a"] - id_true["a"],
                        "delta_a_shuffled_vs_identity_shuffled": shuf_te["a"]
                        - score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), pack_te_s)["a"],
                        "nl_mass_proxy_r": picked["r"],
                    }
                    print("shuffle-fit", a, b, seed, fam, obj, "true_a", true_te["a"], flush=True)
            shuffle_fit.append(rec)
            write_json(out / "shuffle_fit.json", jsonable(shuffle_fit))

    # stability on freeze-PCA subsets
    stability = []
    n_sub = 1 if args.smoke else int(design["stability_subsets"])
    sub_n = int(design["subset_size"])
    for a, b in (shuffle_pairs if not args.smoke else pairs[:1]):
        if (a, b) not in pairs:
            continue
        name = pair_name(a, b)
        meta = cache[name]["meta"]
        ua, ub = meta["U_a"], meta["U_b"]
        mu_a, mu_b = meta["mu_a"], meta["mu_b"]
        xa, xb = cache[name]["splits"]["train"]
        xva, xvb = cache[name]["splits"]["val"]
        pack_va = cache[name]["packs"]["val"]
        idxs = subset_indices(len(xa), min(sub_n, len(xa) // n_sub if n_sub else len(xa)), n_sub, int(design["subset_seed"]))
        rec = {"a": a, "b": b, "fits": []}
        # selected tau from full-data pair_eval
        parent = next(r for r in pair_eval if r["a"] == a and r["b"] == b)
        for si, idx in enumerate(idxs):
            ii = idx.numpy()
            pack_tr = pack_split(xa[ii], xb[ii], ua, ub, mu_a, mu_b)
            lin_tr = score_b(np.eye(ua.shape[1]), np.eye(ub.shape[1]), pack_tr)
            slot = {"subset": si, "families": {}}
            for fam in families:
                slot["families"][fam] = {}
                for obj in OBJECTIVES:
                    tau = parent["families"][fam][obj]["selected_tau"]
                    fit = fit_metrics(pack_tr, fam, obj, tau, lin_tr, n_steps=n_steps)
                    val = score_b(fit["b_a"], fit["b_b"], pack_va)
                    slot["families"][fam][obj] = {
                        "r": fit["r"],
                        "tau": tau,
                        "val": val,
                        "train": fit["train"],
                        "b_a": fit["b_a"].tolist(),
                        "b_b": fit["b_b"].tolist(),
                        "eig_a": fit["eig_a"].tolist(),
                        "eig_b": fit["eig_b"].tolist(),
                    }
            rec["fits"].append(slot)
        # comparisons
        rec["comparisons"] = {}
        for fam in families:
            rec["comparisons"][fam] = {}
            for obj in OBJECTIVES:
                mats_a = [np.array(f["families"][fam][obj]["b_a"]) for f in rec["fits"]]
                grams = []
                for f in rec["fits"]:
                    ba = np.array(f["families"][fam][obj]["b_a"])
                    bb = np.array(f["families"][fam][obj]["b_b"])
                    za = xva - mu_a
                    ka = center_gram_o2(torch.tensor(metric_gram(za, ua, ba)))
                    grams.append(ka.numpy())
                cos_b = [fro_cosine(mats_a[i], mats_a[j]) for i in range(len(mats_a)) for j in range(i + 1, len(mats_a))]
                cos_g = [fro_cosine(grams[i], grams[j]) for i in range(len(grams)) for j in range(i + 1, len(grams))]
                # dominant distorted subspace: |log eig| top 4
                angles = []
                for i in range(len(mats_a)):
                    for j in range(i + 1, len(mats_a)):
                        ei, vi = np.linalg.eigh(mats_a[i])
                        ej, vj = np.linalg.eigh(mats_a[j])
                        ki = np.argsort(np.abs(np.log(np.clip(ei, 1e-12, None))))[-min(4, len(ei)) :]
                        kj = np.argsort(np.abs(np.log(np.clip(ej, 1e-12, None))))[-min(4, len(ej)) :]
                        ang = principal_angles(vi[:, ki], vj[:, kj])
                        angles.append(float(np.mean(ang)))
                rec["comparisons"][fam][obj] = {
                    "mean_fro_cosine_B_a": float(np.mean(cos_b)) if cos_b else None,
                    "mean_val_gram_cosine": float(np.mean(cos_g)) if cos_g else None,
                    "mean_principal_angle_top4_log": float(np.mean(angles)) if angles else None,
                    "r": [f["families"][fam][obj]["r"] for f in rec["fits"]],
                    "val_a": [f["families"][fam][obj]["val"]["a"] for f in rec["fits"]],
                }
        stability.append(rec)
        write_json(out / "stability.json", jsonable(stability))
        print("stability", name, flush=True)

    summary = summarise(pair_eval, families)
    write_json(out / "summary.json", jsonable(summary))
    make_figures(pair_eval, out / "figures", families)
    copy_repo(out, ROOT / "results" / "anisotropic_kernels")
    print(json.dumps(summary, indent=2, default=str))


def summarise(pair_eval, families):
    out = {"n_pairs": len(pair_eval), "evaluation": "exploratory test n=1024"}

    def mean_path(getter):
        xs = []
        for rec in pair_eval:
            v = getter(rec)
            if isinstance(v, (int, float)) and math.isfinite(v):
                xs.append(v)
        return float(np.mean(xs)) if xs else None

    out["identity_a"] = mean_path(lambda r: r["identity"]["test"]["a"])
    out["identity_ratio"] = mean_path(lambda r: r["identity"]["test"]["ratio"])
    out["identity_excess"] = mean_path(lambda r: r["identity"]["test"]["excess"])
    out["identity_mnn"] = mean_path(lambda r: r["identity"]["test"]["mnn_k10"])
    out["identity_r_eff"] = mean_path(lambda r: r["identity"]["test"]["r_eff_k"])
    out["truncation_a"] = mean_path(lambda r: r["truncation_q"]["test"]["a"])
    out["mnn_inner_product"] = mean_path(lambda r: r["identity"]["test"].get("mnn_inner_product"))
    out["prh_max_mnn"] = mean_path(lambda r: r.get("prh_max_mnn"))
    for fam in families:
        out[fam] = {}
        for obj in OBJECTIVES:
            block = {
                "a": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["a"]),
                "ratio": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["ratio"]),
                "excess": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["excess"]),
                "delta_a": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["delta_a_vs_identity"]),
                "r": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["r"]),
                "mnn": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["mnn_k10"]),
                "r_eff": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["r_eff_k"]),
                "knn_overlap": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["knn_overlap_identity_a"]),
                "frac_at_bounds": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["frac_at_bounds"]),
                "tau_counts": dict(Counter(str(r["families"][fam][obj]["selected_tau"]) for r in pair_eval)),
            }
            out[fam][obj] = block
    return out


def make_figures(pair_eval, figdir: Path, families):
    figdir.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(pair_eval)))
    series = {"identity": [r["identity"]["test"]["a"] for r in pair_eval]}
    for fam in families:
        series[f"{fam}_ratio"] = [r["families"][fam]["ratio"]["test"]["a"] for r in pair_eval]
        series[f"{fam}_excess"] = [r["families"][fam]["excess"]["test"]["a"] for r in pair_eval]
    lines(xs, series, "Test CKA a (exploratory n=1024)", "pair index", "CKA a", figdir / "test_a_by_pair.png")
    for obj in OBJECTIVES:
        ser_d, ser_r, ser_e = {}, {}, {}
        for fam in families:
            ser_d[fam] = [r["families"][fam][obj]["delta_a_vs_identity"] for r in pair_eval]
            ser_r[fam] = [r["families"][fam][obj]["r"] for r in pair_eval]
            ser_e[fam] = [r["families"][fam][obj]["test"]["r_eff_k"] for r in pair_eval]
        lines(xs, ser_d, f"Test Δa vs identity ({obj})", "pair index", "Δa", figdir / f"delta_a_{obj}.png")
        lines(xs, ser_r, f"Metric distortion R ({obj})", "pair index", "R", figdir / f"distortion_{obj}.png")
        lines(
            xs,
            {"identity": [r["identity"]["test"]["r_eff_k"] for r in pair_eval], **ser_e},
            f"Centred Gram r_eff ({obj})",
            "pair index",
            "tr(Kc)²/||Kc||_F²",
            figdir / f"rank_{obj}.png",
        )


def copy_repo(out: Path, repo_out: Path):
    repo_out.mkdir(parents=True, exist_ok=True)
    for name in ["design.json", "summary.json", "fits.json", "pair_eval.json", "shuffle_fit.json", "stability.json", "pca.json"]:
        p = out / name
        if p.exists():
            (repo_out / name).write_bytes(p.read_bytes())
    if (out / "figures").exists():
        (repo_out / "figures").mkdir(exist_ok=True)
        for p in (out / "figures").glob("*.png"):
            (repo_out / "figures" / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()
