#!/usr/bin/env python3
"""Learned-kernel extension on frozen PRH released-code representations.

Writes results/learned_kernels/ only. Does not modify original/modern/prh_released_code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.datasets import load_manifest
from prh_replication.io_utils import write_json
from prh_replication.kernel_experiment import (
    FAMILIES,
    LANG_KEYS,
    OBJECTIVES,
    VIS_KEYS,
    candidate_list,
    evaluate_kernel_full,
    frozen_xy,
    jsonable,
    load_split_features,
    pack_eval,
    pair_name,
    prepared_layers,
    profile_of,
    score_candidate,
    score_fixed,
    select_from_landscape,
    select_train_layers,
    share_tag,
    subset_indices,
    vl_pairs,
)
from prh_replication.kernels import (
    ALPHA_GRID,
    LAMBDA_GRID,
    mass_outside_interval,
    median_offdiag_from_dsq,
    pairwise_sq_distances,
    profile_inner_products,
    profile_nodes,
)
from prh_replication.plots import heatmap, lines
from prh_replication.registry import MODELS, Paths


FIXED = ("linear", "rbf_raw_sigma_1", "rbf_lambda_1")


def ids_for(man, split):
    return list(man["splits"][split])


def load_xy(paths, a, b, split, man, layers):
    ids = ids_for(man, split)
    fa = load_split_features(paths, a, split, ids)
    fb = load_split_features(paths, b, split, ids)
    xa, xb = frozen_xy(prepared_layers(fa), prepared_layers(fb), layers)
    return xa, xb


def dist_pack(xa, xb):
    dsq_a = pairwise_sq_distances(xa)
    dsq_b = pairwise_sq_distances(xb)
    return dsq_a, dsq_b


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-hooks", action="store_true")
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--pairs-limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out = paths.results / "learned_kernels"
    out.mkdir(parents=True, exist_ok=True)
    repo_out = ROOT / "results" / "learned_kernels"
    repo_out.mkdir(parents=True, exist_ok=True)
    design = json.loads((ROOT / "configs" / "learned_kernels.json").read_text())
    write_json(out / "design.json", design)

    man = load_manifest(paths.data / "manifests" / "coco_val2017.json")
    n_held = len(man["splits"].get("heldout", []))
    write_json(
        out / "split_sizes.json",
        {
            "train": len(man["splits"]["train"]),
            "val": len(man["splits"]["val"]),
            "test": len(man["splits"]["test"]),
            "heldout": n_held,
            "evaluation_status": "exploratory",
            "reason": "heldout has 904 images; no unused 1024 confirmatory gallery",
        },
    )

    pairs = vl_pairs()
    if args.smoke:
        pairs = pairs[:1]
    if args.pairs_limit:
        pairs = pairs[: args.pairs_limit]

    lam_grid = LAMBDA_GRID
    alpha_grid = ALPHA_GRID
    if args.smoke:
        lam_grid = LAMBDA_GRID[::6]
        alpha_grid = ALPHA_GRID[::4]
        print("SMOKE grids", len(lam_grid), "lambda", len(alpha_grid), "alpha", flush=True)

    cands = candidate_list(lam_grid, alpha_grid)
    print(f"{len(pairs)} VL pairs, {len(cands)} learned candidates", flush=True)

    # --- layer selection on train ---
    layer_table = {}
    t0 = time.time()
    for a, b in pairs:
        name = pair_name(a, b)
        print(f"layers {name}", flush=True)
        ids = ids_for(man, "train")
        fa = load_split_features(paths, a, "train", ids)
        fb = load_split_features(paths, b, "train", ids)
        xa, xb = prepared_layers(fa), prepared_layers(fb)
        sel = select_train_layers(xa, xb)
        sel.update({"a": a, "b": b, "n_train": xa.shape[0], "n_layers": [xa.shape[1], xb.shape[1]]})
        layer_table[name] = sel
        write_json(out / "layers.json", jsonable(layer_table))
    print("layer selection minutes", (time.time() - t0) / 60, flush=True)

    # --- fit landscapes ---
    selections = {}
    landscapes_meta = {}
    t1 = time.time()
    for a, b in pairs:
        name = pair_name(a, b)
        layers = layer_table[name]["layers"]
        xa, xb = load_xy(paths, a, b, "train", man, layers)
        dsq_a, dsq_b = dist_pack(xa, xb)
        sa, sb = median_offdiag_from_dsq(dsq_a), median_offdiag_from_dsq(dsq_b)
        scales = {
            "s_a": sa,
            "s_b": sb,
            "mass_r_gt_3_a": mass_outside_interval(dsq_a, sa),
            "mass_r_gt_3_b": mass_outside_interval(dsq_b, sb),
            "n": xa.shape[0],
        }
        rows = []
        for cand in cands:
            st = score_candidate(xa, xb, dsq_a, dsq_b, sa, sb, cand)
            rows.append(
                {
                    "family": st.get("family"),
                    "lambda": st.get("lambda"),
                    "alpha": st.get("alpha"),
                    "a": st.get("a"),
                    "b": st.get("b"),
                    "ratio": st.get("ratio"),
                    "excess": st.get("excess"),
                    "official_cka": st.get("official_cka"),
                    "valid_a": st.get("valid_a"),
                    "degenerate": st.get("degenerate"),
                    "ratio_undefined": st.get("ratio_undefined"),
                    "boundary": st.get("boundary"),
                }
            )
        write_json(out / "landscapes" / f"{name}.json", {"pair": [a, b], "scales": scales, "rows": jsonable(rows)})
        fixed = {fn: jsonable(pack_eval(score_fixed(xa, xb, dsq_a, dsq_b, sa, sb, fn), {"name": fn})) for fn in FIXED}
        sel = {f"{fam}_{obj}": select_from_landscape(rows, fam, obj) for fam in FAMILIES for obj in OBJECTIVES}
        selections[name] = {"a": a, "b": b, "layers": layers, "train_scales": scales, "fixed_train": fixed, "selected": sel}
        landscapes_meta[name] = scales
        write_json(out / "selections.json", jsonable(selections))
        print(f"fit {name} sa={sa:.4f} sb={sb:.4f}", flush=True)
        del xa, xb, dsq_a, dsq_b
    print("fit minutes", (time.time() - t1) / 60, flush=True)

    # --- val + exploratory test eval ---
    pair_eval = []
    t2 = time.time()
    for a, b in pairs:
        name = pair_name(a, b)
        layers = layer_table[name]["layers"]
        rec = {"a": a, "b": b, "layers": layers, "splits": {}}
        train_s = selections[name]["train_scales"]
        sa_t, sb_t = train_s["s_a"], train_s["s_b"]
        for split in ("train", "val", "test"):
            xa, xb = load_xy(paths, a, b, split, man, layers)
            dsq_a, dsq_b = dist_pack(xa, xb)
            n_perm = args.n_perm if split == "test" else 0
            diag = split == "test"
            split_out = {"n": xa.shape[0], "fixed": {}, "learned": {}}
            for fn in FIXED:
                fam = "linear" if fn == "linear" else "rbf"
                lam = 1.0 if fn == "rbf_lambda_1" else None
                alpha = None
                split_out["fixed"][fn] = jsonable(
                    evaluate_kernel_full(
                        xa, xb, dsq_a, dsq_b, sa_t, sb_t, fam, lam, alpha, n_perm, 0, name=fn, diagnostics=diag
                    )
                )
            for fam in FAMILIES:
                for obj in OBJECTIVES:
                    key = f"{fam}_{obj}"
                    sel = selections[name]["selected"][key]
                    if not sel.get("valid"):
                        split_out["learned"][key] = {"valid": False}
                        continue
                    split_out["learned"][key] = jsonable(
                        evaluate_kernel_full(
                            xa,
                            xb,
                            dsq_a,
                            dsq_b,
                            sa_t,
                            sb_t,
                            fam,
                            sel["lambda"],
                            sel["alpha"],
                            n_perm,
                            0,
                            diagnostics=diag,
                        )
                    )
                    split_out["learned"][key]["selection"] = sel
            rec["splits"][split] = split_out
            del xa, xb, dsq_a, dsq_b
        pair_eval.append(rec)
        write_json(out / "pair_eval.json", jsonable(pair_eval))
        print(f"eval {name}", flush=True)
    print("eval minutes", (time.time() - t2) / 60, flush=True)

    # --- stability ---
    stability = []
    t3 = time.time()
    n_train = len(man["splits"]["train"])
    subsets = subset_indices(n_train, 682, 3, 7)
    for a, b in pairs:
        name = pair_name(a, b)
        layers = layer_table[name]["layers"]
        xa, xb = load_xy(paths, a, b, "train", man, layers)
        fits = []
        for si, idx in enumerate(subsets):
            xas, xbs = xa[idx], xb[idx]
            dsq_a, dsq_b = dist_pack(xas, xbs)
            sa, sb = median_offdiag_from_dsq(dsq_a), median_offdiag_from_dsq(dsq_b)
            rows = [score_candidate(xas, xbs, dsq_a, dsq_b, sa, sb, cand) for cand in cands]
            sel = {f"{fam}_{obj}": select_from_landscape(rows, fam, obj) for fam in FAMILIES for obj in OBJECTIVES}
            fits.append({"subset": si, "n": int(idx.numel()), "s_a": sa, "s_b": sb, "selected": jsonable(sel)})
            del dsq_a, dsq_b
        stability.append({"a": a, "b": b, "layers": layers, "fits": fits})
        write_json(out / "stability.json", jsonable(stability))
        print(f"stability {name}", flush=True)
        del xa, xb
    print("stability minutes", (time.time() - t3) / 60, flush=True)

    # --- profiles / transfer ---
    r = profile_nodes()
    keys = [pair_name(a, b) for a, b in pairs]
    transfer = {f"{fam}_{obj}": {"labels": keys, "a": [], "ratio": [], "excess": [], "share": []} for fam in FAMILIES for obj in OBJECTIVES}
    profile_sims = {}
    for fam in FAMILIES:
        for obj in OBJECTIVES:
            key = f"{fam}_{obj}"
            feats = []
            for a, b in pairs:
                sel = selections[pair_name(a, b)]["selected"][key]
                if sel.get("valid"):
                    feats.append(profile_of(fam, sel["lambda"], sel["alpha"]))
                else:
                    feats.append(None)
            n = len(pairs)
            raw = np.full((n, n), np.nan)
            cen = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    if feats[i] is None or feats[j] is None:
                        continue
                    ip = profile_inner_products(feats[i], feats[j], r)
                    raw[i, j] = ip["raw_normalised_l2"]
                    cen[i, j] = ip["centred_normalised_l2"]
            profile_sims[key] = {"raw": raw.tolist(), "centred": cen.tolist(), "labels": keys}

    # transfer eval on exploratory test
    t4 = time.time()
    target_cache = {}
    for a, b in pairs:
        layers = layer_table[pair_name(a, b)]["layers"]
        xa, xb = load_xy(paths, a, b, "test", man, layers)
        dsq_a, dsq_b = dist_pack(xa, xb)
        target_cache[pair_name(a, b)] = (xa, xb, dsq_a, dsq_b, selections[pair_name(a, b)]["train_scales"])

    transfer_rows = []
    for src in pairs:
        for tgt in pairs:
            sn, tn = pair_name(*src), pair_name(*tgt)
            xa, xb, dsq_a, dsq_b, sc = target_cache[tn]
            sa, sb = sc["s_a"], sc["s_b"]
            share = share_tag(src, tgt)
            for fam in FAMILIES:
                for obj in OBJECTIVES:
                    key = f"{fam}_{obj}"
                    sel = selections[sn]["selected"][key]
                    if not sel.get("valid"):
                        continue
                    st = evaluate_kernel_full(
                        xa, xb, dsq_a, dsq_b, sa, sb, fam, sel["lambda"], sel["alpha"], n_perm=0, seed=0, diagnostics=False
                    )
                    row = {
                        "source": list(src),
                        "target": list(tgt),
                        "share": share,
                        "family": fam,
                        "objective": obj,
                        "source_lambda": sel["lambda"],
                        "source_alpha": sel["alpha"],
                        "target_a": st.get("a"),
                        "target_ratio": st.get("ratio"),
                        "target_excess": st.get("excess"),
                    }
                    tgt_sel = pair_eval[keys.index(tn)]["splits"]["test"]["learned"][key]
                    row["target_fitted_a"] = tgt_sel.get("a")
                    row["gap_fitted_minus_transfer_a"] = (
                        None
                        if tgt_sel.get("a") is None or st.get("a") is None
                        else tgt_sel["a"] - st["a"]
                    )
                    transfer_rows.append(row)
    write_json(out / "transfer_rows.json", jsonable(transfer_rows))
    write_json(out / "profile_sims.json", jsonable(profile_sims))
    print("transfer minutes", (time.time() - t4) / 60, flush=True)

    # --- shuffle-fit control ---
    shuffle_pairs = [tuple(p) for p in design["shuffle_fit_control"]["pairs"]]
    if args.smoke:
        shuffle_pairs = shuffle_pairs[:1]
    shuffle_fit = []
    for a, b in shuffle_pairs:
        if (a, b) not in pairs:
            continue
        for seed in design["shuffle_fit_control"]["seeds"][: (1 if args.smoke else 2)]:
            print(f"shuffle-fit {a} {b} seed {seed}", flush=True)
            g = torch.Generator().manual_seed(int(seed))
            ids_tr = ids_for(man, "train")
            fa = load_split_features(paths, a, "train", ids_tr)
            fb = load_split_features(paths, b, "train", ids_tr)
            xa_all, xb_all = prepared_layers(fa), prepared_layers(fb)
            perm_tr = torch.randperm(xa_all.shape[0], generator=g)
            xb_shuf = xb_all[perm_tr]
            sel_layers = select_train_layers(xa_all, xb_shuf)
            xa, xb = frozen_xy(xa_all, xb_shuf, sel_layers["layers"])
            dsq_a, dsq_b = dist_pack(xa, xb)
            sa, sb = median_offdiag_from_dsq(dsq_a), median_offdiag_from_dsq(dsq_b)
            rows = [score_candidate(xa, xb, dsq_a, dsq_b, sa, sb, cand) for cand in cands]
            sel = {f"{fam}_{obj}": select_from_landscape(rows, fam, obj) for fam in FAMILIES for obj in OBJECTIVES}
            evals = {}
            for split in ("val", "test"):
                g2 = torch.Generator().manual_seed(1000 + int(seed) + (0 if split == "val" else 1))
                xs, ys = load_xy(paths, a, b, split, man, sel_layers["layers"])
                perm = torch.randperm(xs.shape[0], generator=g2)
                ys = ys[perm]
                d1, d2 = dist_pack(xs, ys)
                evals[split] = {}
                for fam in FAMILIES:
                    for obj in OBJECTIVES:
                        key = f"{fam}_{obj}"
                        s = sel[key]
                        if not s.get("valid"):
                            evals[split][key] = {"valid": False}
                            continue
                        evals[split][key] = jsonable(
                            evaluate_kernel_full(xs, ys, d1, d2, sa, sb, fam, s["lambda"], s["alpha"], n_perm=20, seed=0)
                        )
            shuffle_fit.append(
                {
                    "a": a,
                    "b": b,
                    "seed": seed,
                    "layers": sel_layers,
                    "selected": jsonable(sel),
                    "eval_shuffled": evals,
                }
            )
            write_json(out / "shuffle_fit.json", jsonable(shuffle_fit))

    summary = summarise(pair_eval, selections, pairs)
    write_json(out / "summary.json", jsonable(summary))
    write_json(out / "profile_sims.json", jsonable(profile_sims))
    figdir = out / "figures"
    make_figures(pair_eval, profile_sims, transfer_rows, figdir, keys)
    copy_to_repo(out, repo_out)
    print(json.dumps(summary, indent=2, default=str))


def summarise(pair_eval, selections, pairs):
    def mean_key(split, bucket, name, field):
        xs = []
        for rec in pair_eval:
            item = rec["splits"][split][bucket].get(name, {})
            v = item.get(field)
            if isinstance(v, (int, float)) and math.isfinite(v):
                xs.append(v)
        return float(np.mean(xs)) if xs else None

    out = {"n_pairs": len(pairs), "evaluation_split": "test (exploratory)"}
    for split in ("train", "val", "test"):
        out[split] = {
            "fixed": {fn: {f: mean_key(split, "fixed", fn, f) for f in ("a", "b", "ratio", "excess")} for fn in FIXED},
            "learned": {
                f"{fam}_{obj}": {f: mean_key(split, "learned", f"{fam}_{obj}", f) for f in ("a", "b", "ratio", "excess")}
                for fam in FAMILIES
                for obj in OBJECTIVES
            },
        }
    # objective disagreement
    disagree = 0
    for rec in pair_eval:
        name = pair_name(rec["a"], rec["b"])
        sel = selections[name]["selected"]
        for fam in FAMILIES:
            lams = [sel[f"{fam}_{o}"]["lambda"] for o in OBJECTIVES if sel[f"{fam}_{o}"].get("valid")]
            if len(set(round(x, 12) for x in lams)) > 1:
                disagree += 1
    out["n_family_slots_where_objectives_disagree_on_lambda"] = disagree
    out["n_family_slots"] = len(pairs) * len(FAMILIES)
    return out


def make_figures(pair_eval, profile_sims, transfer_rows, figdir, keys):
    figdir.mkdir(parents=True, exist_ok=True)
    labels = [r["a"][:8] + "/" + r["b"][:8] for r in pair_eval]
    for name, title in [("linear", "linear"), ("rbf_raw_sigma_1", "RBF σ=1"), ("rbf_ratio", "RBF ratio-learned")]:
        actual, null = [], []
        for rec in pair_eval:
            te = rec["splits"]["test"]
            item = te["fixed"][name] if name in te["fixed"] else te["learned"].get(name.replace("rbf_ratio", "rbf_ratio"), {})
            if name == "rbf_ratio":
                item = te["learned"]["rbf_ratio"]
            actual.append(item.get("a") or 0)
            null.append(item.get("b") or 0)
        from prh_replication.plots import bars

        bars(labels, actual, null, f"Exploratory test CKA vs analytic shuffle mean ({title})", "CKA", figdir / f"cka_{name}.png")
    mat = np.array(profile_sims["rbf_ratio"]["centred"])
    heatmap(mat, keys, keys, "Centred radial-profile similarity (RBF, ratio obj, ν=U[0,3])", figdir / "profile_rbf_ratio.png", "S")
    # transfer mean by share for ratio rbf
    shares = ["shared_language", "shared_vision", "neither", "same_pair"]
    means = []
    for sh in shares:
        xs = [r["target_a"] for r in transfer_rows if r["family"] == "rbf" and r["objective"] == "ratio" and r["share"] == sh and r["target_a"] is not None]
        means.append(float(np.mean(xs)) if xs else float("nan"))
    lines(list(range(len(shares))), {"RBF ratio transfer a": means}, "Transfer CKA by model overlap", "share class (see report)", "CKA a", figdir / "transfer_share.png")


def copy_to_repo(out: Path, repo_out: Path):
    repo_out.mkdir(parents=True, exist_ok=True)
    for name in [
        "design.json",
        "split_sizes.json",
        "layers.json",
        "selections.json",
        "summary.json",
        "pair_eval.json",
        "stability.json",
        "transfer_rows.json",
        "profile_sims.json",
        "shuffle_fit.json",
    ]:
        src = out / name
        if src.exists():
            (repo_out / name).write_bytes(src.read_bytes())
    fig = out / "figures"
    if fig.exists():
        (repo_out / "figures").mkdir(parents=True, exist_ok=True)
        for p in fig.glob("*.png"):
            (repo_out / "figures" / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()
