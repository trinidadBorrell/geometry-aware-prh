#!/usr/bin/env python3
"""Polynomial + Fourier mixture kernels on frozen PRH layers.

Writes results/flex_kernels/ only.
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
from prh_replication.flex_kernels import (
    FAMILIES,
    NU_GRID,
    OBJECTIVES,
    P_MAX,
    REGIMES,
    build_basis_grams,
    cka_from_stats,
    center_stack,
    contract_pair,
    family_basis_names,
    fourier_profile,
    gaussian_limit_error,
    gegenbauer_normalised,
    mix_grams,
    objective_value,
    penalty_and_parts,
    fit_simplex,
)
from prh_replication.io_utils import write_json
from prh_replication.kernel_experiment import (
    frozen_xy,
    jsonable,
    load_split_features,
    pair_name,
    prepared_layers,
    share_tag,
    subset_indices,
    vl_pairs,
)
from prh_replication.kernels import (
    extension_stats,
    gram_diagnostics,
    linear_gram,
    mc_cka_mean,
    median_offdiag_from_dsq,
    pairwise_sq_distances,
    rbf_gram_from_dsq,
    u_centred_cka,
)
from prh_replication.metrics import mutual_knn_score, nearest_neighbors
from prh_replication.plots import heatmap, lines
from prh_replication.registry import Paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--pairs-limit", type=int, default=0)
    p.add_argument(
        "--shuffle-only",
        action="store_true",
        help="Load shuffle-pair features and rewrite shuffle_fit.json for all families; skip other stages.",
    )
    return p.parse_args()


def load_frozen(repo: Path):
    layers = json.loads((repo / "results" / "learned_kernels" / "layers.json").read_text())
    sel = json.loads((repo / "results" / "learned_kernels" / "selections.json").read_text())
    scales = {k: v["train_scales"] for k, v in sel.items()}
    return layers, scales


def linear_vertex_stats(stats):
    c = np.zeros(stats["M"].shape[0])
    c[0] = 1.0
    return c, cka_from_stats(c, stats)


def best_vertex(stats, obj):
    m = stats["M"].shape[0]
    best, bi = float("-inf"), 0
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        j = objective_value(cka_from_stats(e, stats), obj)
        if math.isfinite(j) and j > best:
            best, bi = j, i
    e = np.zeros(m)
    e[bi] = 1.0
    return e, bi, best


def compute_stats(xa, xb, sa, sb, family, p_max, nu_grid):
    da, db = xa.shape[1], xb.shape[1]
    ta, tb = linear_gram(xa), linear_gram(xb)
    dsq_a, dsq_b = pairwise_sq_distances(xa), pairwise_sq_distances(xb)
    ga, specs, fb_a = build_basis_grams(ta, dsq_a, sa, da, family, p_max, nu_grid)
    gb, _, fb_b = build_basis_grams(tb, dsq_b, sb, db, family, p_max, nu_grid)
    stats = contract_pair(center_stack(ga), center_stack(gb))
    meta = {"d_a": da, "d_b": db, "fallback_a": fb_a, "fallback_b": fb_b, "specs": specs, "n": xa.shape[0]}
    return stats, meta, ga, gb, dsq_a, dsq_b


def save_npz(path: Path, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def run_shuffle_fit(*, design, pairs, stats_cache, families, regimes, p_max, nu_grid, n_steps, smoke, out):
    """Coefficient + regulariser search on independently shuffled correspondences."""
    shuffle_pairs = [tuple(p) for p in design["shuffle_pairs"]]
    if smoke:
        shuffle_pairs = shuffle_pairs[:1]
    pair_set = set(pairs)
    shuffle_fit = []
    for a, b in shuffle_pairs:
        if (a, b) not in pair_set:
            continue
        name = pair_name(a, b)
        sa, sb = stats_cache[name]["s_a"], stats_cache[name]["s_b"]
        for seed in design["shuffle_seeds"][: (1 if smoke else 2)]:
            g = torch.Generator().manual_seed(int(seed))
            xa = stats_cache[name]["splits"]["train"]["xa"]
            xb = stats_cache[name]["splits"]["train"]["xb"][torch.randperm(xa.shape[0], generator=g)]
            xva = stats_cache[name]["splits"]["val"]["xa"]
            g2 = torch.Generator().manual_seed(1000 + int(seed))
            xvb = stats_cache[name]["splits"]["val"]["xb"][torch.randperm(xva.shape[0], generator=g2)]
            g3 = torch.Generator().manual_seed(2000 + int(seed))
            xta = stats_cache[name]["splits"]["test"]["xa"]
            xtb = stats_cache[name]["splits"]["test"]["xb"][torch.randperm(xta.shape[0], generator=g3)]
            rec = {"a": a, "b": b, "seed": int(seed), "families": {}}
            for fam in families:
                st_tr, meta, _, _, _, _ = compute_stats(xa, xb, sa, sb, fam, p_max, nu_grid)
                st_va, _, _, _, _, _ = compute_stats(xva, xvb, sa, sb, fam, p_max, nu_grid)
                st_te, _, _, _, _, _ = compute_stats(xta, xtb, sa, sb, fam, p_max, nu_grid)
                _, jlin = linear_vertex_stats(st_tr)
                rec["families"][fam] = {}
                for obj in OBJECTIVES:
                    cand = {}
                    for rg in regimes:
                        fit = fit_simplex(st_tr, meta["specs"], obj, jlin[obj], REGIMES[rg], n_steps=n_steps)
                        c = np.array(fit["c"])
                        cand[rg] = {"c": fit["c"], "val": cka_from_stats(c, st_va), "parts": fit["parts"]}
                    pick = max(
                        regimes,
                        key=lambda r: cand[r]["val"].get(obj) if math.isfinite(cand[r]["val"].get(obj) or float("nan")) else -1e18,
                    )
                    c = np.array(cand[pick]["c"])
                    rec["families"][fam][obj] = {
                        "selected_reg": pick,
                        "test": cka_from_stats(c, st_te),
                        "nl_mass": cand[pick]["parts"]["nl_mass"],
                    }
            shuffle_fit.append(rec)
            write_json(out / "shuffle_fit.json", jsonable(shuffle_fit))
            print("shuffle-fit", a, b, seed, flush=True)
    return shuffle_fit


def main():
    args = parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out = paths.results / "flex_kernels"
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads((ROOT / "configs" / "flex_kernels.json").read_text())
    write_json(out / "design.json", design)
    p_max = 4 if args.smoke else P_MAX
    nu_grid = NU_GRID[:4] if args.smoke else NU_GRID
    regimes = ["unreg", "nl"] if args.smoke else list(REGIMES)
    families = ("monomial", "lin_fourier") if args.smoke else FAMILIES
    n_steps = 60 if args.smoke else 180

    man = load_manifest(paths.data / "manifests" / "coco_val2017.json")
    layers_tbl, scales_tbl = load_frozen(ROOT)
    pairs = vl_pairs()
    if args.smoke:
        pairs = pairs[:1]
    if args.pairs_limit:
        pairs = pairs[: args.pairs_limit]
    if args.shuffle_only:
        allowed = set(vl_pairs())
        pairs = [tuple(p) for p in design["shuffle_pairs"] if tuple(p) in allowed]
        if args.smoke:
            pairs = pairs[:1]

    if not args.shuffle_only:
        # verification bundle (fixtures + gaussian limit)
        ver = {
            "gaussian_limit_d384": gaussian_limit_error(384, nu_grid),
            "gaussian_limit_d1024": gaussian_limit_error(1024, nu_grid),
            "gaussian_limit_d2048": gaussian_limit_error(2048, nu_grid),
            "d_ref": 1024,
            "s_ref": 1.0,
        }
        write_json(out / "verification.json", jsonable(ver))

    t0 = time.time()
    stats_cache = {}
    grams_keep = {}  # only test grams for selected eval later rebuilt
    for a, b in pairs:
        name = pair_name(a, b)
        ly = layers_tbl[name]["layers"]
        sa, sb = scales_tbl[name]["s_a"], scales_tbl[name]["s_b"]
        stats_cache[name] = {"layers": ly, "s_a": sa, "s_b": sb, "splits": {}}
        for split in ("train", "val", "test"):
            ids = list(man["splits"][split])
            fa = load_split_features(paths, a, split, ids)
            fb = load_split_features(paths, b, split, ids)
            xa, xb = frozen_xy(prepared_layers(fa), prepared_layers(fb), ly)
            stats_cache[name]["splits"][split] = {"xa": xa, "xb": xb}
            print(f"stats {name} {split} n={xa.shape[0]} d={xa.shape[1]},{xb.shape[1]}", flush=True)
        print("elapsed stats pair", name, time.time() - t0, flush=True)

    if args.shuffle_only:
        run_shuffle_fit(
            design=design,
            pairs=pairs,
            stats_cache=stats_cache,
            families=families,
            regimes=regimes,
            p_max=p_max,
            nu_grid=nu_grid,
            n_steps=n_steps,
            smoke=args.smoke,
            out=out,
        )
        copy_repo(out, ROOT / "results" / "flex_kernels")
        print("shuffle-only done", flush=True)
        return

    # fits
    fits = {}
    t1 = time.time()
    for a, b in pairs:
        name = pair_name(a, b)
        sa, sb = stats_cache[name]["s_a"], stats_cache[name]["s_b"]
        xa, xb = stats_cache[name]["splits"]["train"]["xa"], stats_cache[name]["splits"]["train"]["xb"]
        xva, xvb = stats_cache[name]["splits"]["val"]["xa"], stats_cache[name]["splits"]["val"]["xb"]
        fits[name] = {"a": a, "b": b, "families": {}}
        for fam in families:
            st_tr, meta_tr, ga_tr, gb_tr, _, _ = compute_stats(xa, xb, sa, sb, fam, p_max, nu_grid)
            st_va, _, ga_va, gb_va, _, _ = compute_stats(xva, xvb, sa, sb, fam, p_max, nu_grid)
            npz_dir = out / "cache" / name
            save_npz(npz_dir / f"{fam}_train.npz", st_tr)
            save_npz(npz_dir / f"{fam}_val.npz", st_va)
            _, jlin_tr = linear_vertex_stats(st_tr)
            _, jlin_va = linear_vertex_stats(st_va)
            specs = meta_tr["specs"]
            fam_rec = {"meta": {k: meta_tr[k] for k in ("d_a", "d_b", "fallback_a", "fallback_b", "n")}, "objectives": {}}
            # basis correlations
            A = st_tr["A"]
            dA = np.sqrt(np.clip(np.diag(A), 1e-30, None))
            corr = A / np.outer(dA, dA)
            fam_rec["basis_corr_cond"] = {
                "cond_A": float(np.linalg.cond(st_tr["A"])),
                "cond_B": float(np.linalg.cond(st_tr["B"])),
                "max_offdiag_corr": float(np.max(np.abs(corr - np.eye(len(corr))))),
            }
            for obj in OBJECTIVES:
                e_best, bi, bj = best_vertex(st_tr, obj)
                obj_rec = {"jlin_train": jlin_tr[obj], "jlin_val": jlin_va[obj], "best_vertex": bi, "best_vertex_train": bj, "regs": {}}
                for rg in regimes:
                    lams = REGIMES[rg]
                    fit = fit_simplex(st_tr, specs, obj, jlin_tr[obj], lams, n_steps=n_steps)
                    c = np.array(fit["c"])
                    val_st = cka_from_stats(c, st_va)
                    obj_rec["regs"][rg] = {
                        "c": fit["c"],
                        "train": fit["unpenalised"],
                        "train_penalised": fit["penalised"],
                        "val": val_st,
                        "parts": fit["parts"],
                        "penalty": fit["penalty"],
                        "start_spread": fit.get("start_spread"),
                        "best_start": fit.get("best_start"),
                        "ok": fit["ok"],
                    }
                    print(f"fit {name} {fam} {obj} {rg} val_{obj}={val_st.get(obj)} nl={fit['parts']['nl_mass']:.3f}", flush=True)
                # select reg on val unpenalised
                pick = max(
                    regimes,
                    key=lambda r: (obj_rec["regs"][r]["val"].get(obj) if math.isfinite(obj_rec["regs"][r]["val"].get(obj) or float("nan")) else -1e18),
                )
                obj_rec["selected_reg"] = pick
                fam_rec["objectives"][obj] = obj_rec
            fits[name]["families"][fam] = fam_rec
            del ga_tr, gb_tr, ga_va, gb_va, st_tr, st_va
        write_json(out / "fits.json", jsonable(fits))
    print("fit minutes", (time.time() - t1) / 60, flush=True)

    # evaluation on test
    pair_eval = []
    t2 = time.time()
    for a, b in pairs:
        name = pair_name(a, b)
        sa, sb = stats_cache[name]["s_a"], stats_cache[name]["s_b"]
        xta, xtb = stats_cache[name]["splits"]["test"]["xa"], stats_cache[name]["splits"]["test"]["xb"]
        rec = {"a": a, "b": b, "families": {}, "baselines": {}}
        tlin = linear_gram(xta)
        dsq_a, dsq_b = pairwise_sq_distances(xta), pairwise_sq_distances(xtb)
        rec["baselines"]["linear"] = jsonable(extension_stats(tlin, linear_gram(xtb)))
        rec["baselines"]["rbf_sigma1"] = jsonable(
            extension_stats(rbf_gram_from_dsq(dsq_a, 1.0), rbf_gram_from_dsq(dsq_b, 1.0))
        )
        rec["baselines"]["linear"].update(gram_diagnostics(tlin))
        rec["baselines"]["linear"].update(u_centred_cka(tlin, linear_gram(xtb)))
        rec["baselines"]["linear"].update(mc_cka_mean(tlin, linear_gram(xtb), args.n_perm if not args.smoke else 20, 0))
        knn_lin_a = nearest_neighbors(xta, 10)
        knn_lin_b = nearest_neighbors(xtb, 10)
        rec["mnn_linear"] = mutual_knn_score(xta, xtb, 10)
        for fam in families:
            st_te, meta, ga, gb, _, _ = compute_stats(xta, xtb, sa, sb, fam, p_max, nu_grid)
            save_npz(out / "cache" / name / f"{fam}_test.npz", st_te)
            fam_out = {}
            for obj in OBJECTIVES:
                fr = fits[name]["families"][fam]["objectives"][obj]
                rg = fr["selected_reg"]
                c = np.array(fr["regs"][rg]["c"])
                # also unreg counterpart
                c_un = np.array(fr["regs"]["unreg"]["c"])
                k, l = mix_grams(ga, c), mix_grams(gb, c)
                ev = extension_stats(k, l)
                ev.update(gram_diagnostics(k))
                ev.update(u_centred_cka(k, l))
                ev.update(mc_cka_mean(k, l, args.n_perm if not args.smoke else 20, 0))
                _, parts = penalty_and_parts(c, meta["specs"], *REGIMES[rg])
                kk = k.clone()
                kk.fill_diagonal_(-1e8)
                knn_ka = kk.argsort(dim=1, descending=True)[:, :10]
                ll = l.clone()
                ll.fill_diagonal_(-1e8)
                knn_kb = ll.argsort(dim=1, descending=True)[:, :10]
                eq_m = knn_ka.unsqueeze(2) == knn_kb.unsqueeze(1)
                mnn_k = float((eq_m.any(dim=2).float().sum(1) / 10).mean())
                eq_o = knn_lin_a.unsqueeze(2) == knn_ka.unsqueeze(1)
                overlap_a = float((eq_o.any(dim=2).float().sum(1) / 10).mean())
                ev_un = cka_from_stats(c_un, st_te)
                tr_stats = dict(load_npz(out / "cache" / name / f"{fam}_train.npz"))
                tr_stats["n"] = int(np.array(tr_stats["n"]).reshape(()))
                e_tr, bidx, _ = best_vertex(tr_stats, obj)
                ev_vert = cka_from_stats(e_tr, st_te)
                fam_out[obj] = {
                    "selected_reg": rg,
                    "c": c.tolist(),
                    "test": jsonable(ev),
                    "test_unreg": jsonable(ev_un),
                    "test_best_vertex": jsonable(ev_vert),
                    "best_vertex_index": bidx,
                    "parts": parts,
                    "mnn_k10": mnn_k,
                    "knn_overlap_with_linear_A": overlap_a,
                    "train": fr["regs"][rg]["train"],
                    "val": fr["regs"][rg]["val"],
                }
            rec["families"][fam] = fam_out
            del ga, gb
        pair_eval.append(rec)
        write_json(out / "pair_eval.json", jsonable(pair_eval))
        print(f"eval {name}", flush=True)
    print("eval minutes", (time.time() - t2) / 60, flush=True)

    # transfer (val-selected c, target stats on test)
    transfer = []
    for src in pairs:
        sn = pair_name(*src)
        for tgt in pairs:
            tn = pair_name(*tgt)
            sa, sb = stats_cache[tn]["s_a"], stats_cache[tn]["s_b"]
            xta, xtb = stats_cache[tn]["splits"]["test"]["xa"], stats_cache[tn]["splits"]["test"]["xb"]
            share = share_tag(src, tgt)
            for fam in families:
                st_te = dict(load_npz(out / "cache" / tn / f"{fam}_test.npz"))
                st_te["n"] = int(np.array(st_te["n"]).reshape(()))
                specs = family_basis_names(fam, p_max, nu_grid)
                for obj in OBJECTIVES:
                    fr = fits[sn]["families"][fam]["objectives"][obj]
                    c = np.array(fr["regs"][fr["selected_reg"]]["c"])
                    if len(c) != st_te["M"].shape[0]:
                        continue
                    ev = cka_from_stats(c, st_te)
                    fitted = None
                    for rec in pair_eval:
                        if rec["a"] == tgt[0] and rec["b"] == tgt[1]:
                            fitted = rec["families"][fam][obj]["test"]["a"]
                    transfer.append(
                        {
                            "source": list(src),
                            "target": list(tgt),
                            "share": share,
                            "family": fam,
                            "objective": obj,
                            "nl_mass": penalty_and_parts(c, specs, 0, 0, 0)[1]["nl_mass"],
                            "a": ev["a"],
                            "ratio": ev["ratio"],
                            "excess": ev["excess"],
                            "target_fitted_a": fitted,
                            "gap_fitted_minus_transfer_a": None if fitted is None or ev["a"] is None else fitted - ev["a"],
                        }
                    )
    write_json(out / "transfer_rows.json", jsonable(transfer))

    # common kernel on all pairs (train), per family/obj using moderate if present else unreg
    common = {}
    for fam in families:
        common[fam] = {}
        for obj in OBJECTIVES:
            rg = "moderate" if "moderate" in regimes else "unreg"
            stats_list = []
            jlins = []
            specs = family_basis_names(fam, p_max, nu_grid)
            for a, b in pairs:
                name = pair_name(a, b)
                st = dict(load_npz(out / "cache" / name / f"{fam}_train.npz"))
                st["n"] = int(np.array(st["n"]).reshape(()))
                stats_list.append(st)
                jlins.append(fits[name]["families"][fam]["objectives"][obj]["jlin_train"])

            def mean_j(c):
                xs = []
                for st, jl in zip(stats_list, jlins):
                    xs.append(cka_from_stats(c, st)[obj] / jl if jl and jl > 1e-8 else float("nan"))
                return float(np.nanmean(xs))

            m = stats_list[0]["M"].shape[0]
            c = np.zeros(m)
            c[0] = 1.0
            best, bestc = mean_j(c) - penalty_and_parts(c, specs, *REGIMES[rg])[0], c.copy()
            rng = np.random.default_rng(4)
            starts = [c.copy(), np.ones(m) / m]
            for _ in range(3):
                starts.append(rng.dirichlet(np.ones(m)))
            from prh_replication.flex_kernels import project_simplex

            for c0 in starts:
                c = project_simplex(c0)
                for step in range(n_steps):
                    val = mean_j(c) - penalty_and_parts(c, specs, *REGIMES[rg])[0]
                    g = np.zeros(m)
                    for i in range(m):
                        cp = c.copy()
                        cp[i] += 1e-5
                        cp = project_simplex(cp)
                        g[i] = (mean_j(cp) - penalty_and_parts(cp, specs, *REGIMES[rg])[0] - val) / 1e-5
                    c = project_simplex(c + (0.15 / math.sqrt(1 + step / 40)) * g)
                    val = mean_j(c) - penalty_and_parts(c, specs, *REGIMES[rg])[0]
                    if val > best:
                        best, bestc = val, c.copy()
            test_as = []
            for rec, (pa, pb) in zip(pair_eval, pairs):
                tn = pair_name(pa, pb)
                st_te = dict(load_npz(out / "cache" / tn / f"{fam}_test.npz"))
                st_te["n"] = int(np.array(st_te["n"]).reshape(()))
                ev = cka_from_stats(bestc, st_te)
                test_as.append(ev["a"])
                rec.setdefault("common", {}).setdefault(fam, {})[obj] = jsonable(ev)
            common[fam][obj] = {
                "reg": rg,
                "c": bestc.tolist(),
                "parts": penalty_and_parts(bestc, specs, *REGIMES[rg])[1],
                "mean_test_a": float(np.nanmean(test_as)),
            }
    write_json(out / "common_kernel.json", jsonable(common))
    write_json(out / "pair_eval.json", jsonable(pair_eval))

    # LOPO for sph_fourier or first family, ratio, selected typical reg
    lopo = []
    fam_lopo = "sph_fourier" if "sph_fourier" in families else families[0]
    for hi, (ha, hb) in enumerate(pairs):
        others = [p for p in pairs if p != (ha, hb)]
        if not others:
            lopo.append({"held_pair": [ha, hb], "skipped": "need ≥2 pairs for leave-one-pair-out"})
            continue
        obj = "ratio"
        rg = "moderate" if "moderate" in regimes else "unreg"
        specs = family_basis_names(fam_lopo, p_max, nu_grid)
        stats_list, jlins = [], []
        for a, b in others:
            name = pair_name(a, b)
            st = dict(load_npz(out / "cache" / name / f"{fam_lopo}_train.npz"))
            st["n"] = int(np.array(st["n"]).reshape(()))
            stats_list.append(st)
            jlins.append(fits[name]["families"][fam_lopo]["objectives"][obj]["jlin_train"])
        m = stats_list[0]["M"].shape[0]
        from prh_replication.flex_kernels import project_simplex

        def mean_j(c):
            return float(np.nanmean([cka_from_stats(c, st)[obj] / jl if jl > 1e-8 else np.nan for st, jl in zip(stats_list, jlins)]))

        c = np.ones(m) / m
        for step in range(n_steps):
            val = mean_j(c) - penalty_and_parts(c, specs, *REGIMES[rg])[0]
            g = np.zeros(m)
            for i in range(min(m, 12) if args.smoke else m):
                cp = project_simplex(c + np.eye(m)[i] * 1e-5)
                g[i] = (mean_j(cp) - penalty_and_parts(cp, specs, *REGIMES[rg])[0] - val) / 1e-5
            c = project_simplex(c + 0.1 * g)
        tn = pair_name(ha, hb)
        st_te = dict(load_npz(out / "cache" / tn / f"{fam_lopo}_test.npz"))
        st_te["n"] = int(np.array(st_te["n"]).reshape(()))
        ev = cka_from_stats(c, st_te)
        lopo.append({"held_pair": [ha, hb], "family": fam_lopo, "a": ev["a"], "ratio": ev["ratio"], "nl_mass": penalty_and_parts(c, specs, 0, 0, 0)[1]["nl_mass"]})
        if args.smoke:
            break
    write_json(out / "lopo.json", jsonable(lopo))

    # stability on 3 subsets, monomial+ratio only if smoke else all families ratio
    stability = []
    n_train = len(man["splits"]["train"])
    subsets = subset_indices(n_train, min(682, n_train // 3), 3, 7)
    stab_fams = families[:1] if args.smoke else families
    for a, b in pairs:
        name = pair_name(a, b)
        sa, sb = stats_cache[name]["s_a"], stats_cache[name]["s_b"]
        xa, xb = stats_cache[name]["splits"]["train"]["xa"], stats_cache[name]["splits"]["train"]["xb"]
        rec = {"a": a, "b": b, "fits": []}
        for si, idx in enumerate(subsets):
            slot = {"subset": si, "n": int(idx.numel()), "families": {}}
            xas, xbs = xa[idx], xb[idx]
            for fam in stab_fams:
                st, meta, _, _, _, _ = compute_stats(xas, xbs, sa, sb, fam, p_max, nu_grid)
                _, jlin = linear_vertex_stats(st)
                slot["families"][fam] = {}
                for obj in OBJECTIVES:
                    rg = fits[name]["families"][fam]["objectives"][obj]["selected_reg"]
                    fit = fit_simplex(st, meta["specs"], obj, jlin[obj], REGIMES[rg], n_steps=n_steps)
                    slot["families"][fam][obj] = {"c": fit["c"], "parts": fit["parts"], "train": fit["unpenalised"], "reg": rg}
            rec["fits"].append(slot)
        stability.append(rec)
        write_json(out / "stability.json", jsonable(stability))
        print("stability", name, flush=True)
        if args.smoke:
            break

    run_shuffle_fit(
        design=design,
        pairs=pairs,
        stats_cache=stats_cache,
        families=families,
        regimes=regimes,
        p_max=p_max,
        nu_grid=nu_grid,
        n_steps=n_steps,
        smoke=args.smoke,
        out=out,
    )

    summary = summarise(pair_eval, families)
    write_json(out / "summary.json", jsonable(summary))
    make_figures(pair_eval, out / "figures", families)
    copy_repo(out, ROOT / "results" / "flex_kernels")
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

    out["linear_a"] = mean_path(lambda r: r["baselines"]["linear"]["a"])
    out["linear_ratio"] = mean_path(lambda r: r["baselines"]["linear"]["ratio"])
    out["rbf_sigma1_a"] = mean_path(lambda r: r["baselines"]["rbf_sigma1"]["a"])
    for fam in families:
        out[fam] = {}
        for obj in OBJECTIVES:
            out[fam][obj] = {
                "a": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["a"]),
                "ratio": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["ratio"]),
                "excess": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["test"]["excess"]),
                "nl_mass": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["parts"]["nl_mass"]),
                "mnn": mean_path(lambda r, f=fam, o=obj: r["families"][f][o]["mnn_k10"]),
                "reg_counts": {},
            }
            from collections import Counter
            c = Counter(r["families"][fam][obj]["selected_reg"] for r in pair_eval)
            out[fam][obj]["reg_counts"] = dict(c)
    out["mnn_linear"] = mean_path(lambda r: r["mnn_linear"])
    return out


def make_figures(pair_eval, figdir: Path, families):
    figdir.mkdir(parents=True, exist_ok=True)
    labels = [r["a"][:10] + "/" + r["b"][:8] for r in pair_eval]
    series = {"linear": [r["baselines"]["linear"]["a"] for r in pair_eval]}
    for fam in families:
        series[fam + "_ratio"] = [r["families"][fam]["ratio"]["test"]["a"] for r in pair_eval]
        series[fam + "_excess"] = [r["families"][fam]["excess"]["test"]["a"] for r in pair_eval]
    lines(list(range(len(labels))), series, "Test CKA a (val-selected regulariser)", "pair index", "CKA a", figdir / "test_a_by_pair.png")
    for fam in families:
        ys_r = [r["families"][fam]["ratio"]["parts"]["nl_mass"] for r in pair_eval]
        ys_e = [r["families"][fam]["excess"]["parts"]["nl_mass"] for r in pair_eval]
        lines(
            list(range(len(labels))),
            {"ratio": ys_r, "excess": ys_e},
            f"Nonlinear mass ({fam})",
            "pair index",
            "1-c_linear",
            figdir / f"nl_mass_{fam}.png",
        )


def copy_repo(out: Path, repo_out: Path):
    repo_out.mkdir(parents=True, exist_ok=True)
    for name in ["design.json", "summary.json", "fits.json", "pair_eval.json", "transfer_rows.json", "common_kernel.json", "lopo.json", "stability.json", "shuffle_fit.json", "verification.json"]:
        p = out / name
        if p.exists():
            (repo_out / name).write_bytes(p.read_bytes())
    if (out / "figures").exists():
        (repo_out / "figures").mkdir(exist_ok=True)
        for p in (out / "figures").glob("*.png"):
            (repo_out / "figures" / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()
