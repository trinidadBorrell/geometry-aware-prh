#!/usr/bin/env python3
"""Recompute PRH released-code protocol from existing COCO 1024 caches.

Writes results/prh_released_code/ without modifying results/original or results/modern.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.extract import feature_path
from prh_replication.io_utils import load_features, skip_if_complete, write_json
from prh_replication.metrics import linear_cka, mutual_knn_score
from prh_replication.plots import bars, heatmap, lines
from prh_replication.prh_ref import (
    PRH_RBF_SIGMA,
    PRH_TOPK,
    centred_gram_diag_share,
    concat_prepared,
    fixed_layer_mknn_null,
    max_linear_cka,
    max_mknn,
    max_rbf_cka,
    selection_aware_cka_null,
    selection_aware_mknn_null,
    stack_prepared,
)
from prh_replication.registry import MODELS, Paths
import importlib.util

spec = importlib.util.spec_from_file_location("run_phase1", ROOT / "scripts" / "run_phase1.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def pack_metric(score, layers, extra: dict) -> dict:
    out = {"score": float(score), "layers": list(layers)}
    out.update(extra)
    return out


def evaluate_released(fa, fb, n_perm_select: int, n_perm_fixed: int, include_concat: bool, cka_sel_null: bool):
    xa = stack_prepared(fa["feats"])
    xb = stack_prepared(fb["feats"])
    ids = list(fa["sample_ids"])
    if ids != list(fb["sample_ids"]):
        raise ValueError("sample id mismatch")

    mnn, mij = max_mknn(xa, xb)
    cka, cij = max_linear_cka(xa, xb)
    rbf, rij = max_rbf_cka(xa, xb, PRH_RBF_SIGMA)

    concat_info = None
    if include_concat:
        xcat, ycat = concat_prepared(xa), concat_prepared(xb)
        m_cat = mutual_knn_score(xcat, ycat, PRH_TOPK)
        c_cat = linear_cka(xcat, ycat)
        concat_info = {
            "mknn_concat": m_cat,
            "linear_cka_concat": c_cat,
            "mknn_concat_beats_max_layer": m_cat > mnn,
            "cka_concat_beats_max_layer": c_cat > cka,
        }

    xa_m, xb_m = xa[:, mij[0]], xb[:, mij[1]]
    xa_c, xb_c = xa[:, cij[0]], xb[:, cij[1]]
    xa_r, xb_r = xa[:, rij[0]], xb[:, rij[1]]

    # selection-aware nulls are expensive for CKA; mNN uses index mapping
    sel_m = selection_aware_mknn_null(xa, xb, n_perm=n_perm_select, seed=0)
    fix_m = fixed_layer_mknn_null(xa_m, xb_m, n_perm=n_perm_fixed, seed=0)
    # CKA selection-aware: subsample perms if requested via n_perm_select
    sel_c = {"shuffle_mean": None, "shuffle_std": None, "null_label": "not computed (cost; mNN is the PRH primary statistic)"}
    if cka_sel_null:
        sel_c = selection_aware_cka_null(xa, xb, n_perm=n_perm_select, seed=0)
    from prh_replication.controls import linear_cka_shuffle, rbf_cka_shuffle

    fix_c = linear_cka_shuffle(xa_c, xb_c, n_perm_fixed, 0)
    fix_r = rbf_cka_shuffle(xa_r, xb_r, PRH_RBF_SIGMA, PRH_RBF_SIGMA, n_perm_fixed, 0)

    rbf_diag = centred_gram_diag_share(xa_r)

    return {
        "n": len(ids),
        "protocol": "prh_released_code",
        "mknn": pack_metric(
            mnn,
            mij,
            {
                "fixed_layer_shuffle_mean": fix_m["shuffle_mean"],
                "fixed_layer_shuffle_std": fix_m["shuffle_std"],
                "selection_aware_shuffle_mean": sel_m["shuffle_mean"],
                "selection_aware_shuffle_std": sel_m["shuffle_std"],
                "excess_vs_fixed_layer_null": mnn - fix_m["shuffle_mean"],
                "excess_vs_selection_aware_null": mnn - sel_m["shuffle_mean"],
                "null_notes": {
                    "fixed_layer": fix_m["null_label"],
                    "selection_aware": sel_m["null_label"],
                    "k_over_n_minus_1": PRH_TOPK / (len(ids) - 1),
                },
            },
        ),
        "linear_cka": pack_metric(
            cka,
            cij,
            {
                "fixed_layer_shuffle_mean": fix_c.shuffle_mean,
                "fixed_layer_shuffle_std": fix_c.shuffle_std,
                "selection_aware_shuffle_mean": sel_c["shuffle_mean"],
                "selection_aware_shuffle_std": sel_c["shuffle_std"],
                "excess_vs_fixed_layer_null": cka - fix_c.shuffle_mean,
                "excess_vs_selection_aware_null": None if sel_c["shuffle_mean"] is None else cka - sel_c["shuffle_mean"],
            },
        ),
        "rbf_cka_sigma1": pack_metric(
            rbf,
            rij,
            {
                "sigma": PRH_RBF_SIGMA,
                "fixed_layer_shuffle_mean": fix_r.shuffle_mean,
                "fixed_layer_shuffle_std": fix_r.shuffle_std,
                "excess_vs_fixed_layer_null": rbf - fix_r.shuffle_mean,
                "rbf_diag_diagnostic_on_selected_A": rbf_diag,
            },
        ),
        "concat_paper_extra": concat_info,
        "sample_ids_head": ids[:3],
        "layer_counts": [xa.shape[1], xb.shape[1]],
    }


def kind_of(a, b):
    vis = {"dinov2-small", "vit-in21k-small", "clip-laion-base"}
    da, db = a in vis, b in vis
    if da and db:
        return "vision-vision"
    if not da and not db:
        return "language-language"
    return "vision-language"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--n-perm-select", type=int, default=100, help="selection-aware perms for max-over-layers mNN")
    p.add_argument("--n-perm-fixed", type=int, default=100)
    p.add_argument("--concat", action="store_true")
    p.add_argument("--cka-sel-null", action="store_true")
    p.add_argument("--pairs-limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    # rebuild a fake run_phase1 args
    class A:
        split_limit = 0
        dataset = "coco_val2017"
        protocol = "original"
        models = None
        work = args.work

    a = A()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out_root = paths.results / "prh_released_code"
    out_root.mkdir(parents=True, exist_ok=True)
    if skip_if_complete(out_root, force=args.force):
        return
    keys = rp.selected_models(a)
    keys = [k for k in keys if rp._load(paths, k, "test", a) is not None]
    pair_rows = []
    combos = list(combinations(keys, 2))
    if args.pairs_limit:
        combos = combos[: args.pairs_limit]
    for i, (ka, kb) in enumerate(combos):
        print(f"[{i+1}/{len(combos)}] {ka} vs {kb}", flush=True)
        fa, fb = rp._load(paths, ka, "test", a), rp._load(paths, kb, "test", a)
        res = evaluate_released(fa, fb, args.n_perm_select, args.n_perm_fixed, args.concat, args.cka_sel_null)
        res["model_a"], res["model_b"] = ka, kb
        write_json(out_root / "test_pairs" / f"{ka}__{kb}.json", res)
        pair_rows.append(
            {
                "a": ka,
                "b": kb,
                "kinds": kind_of(ka, kb),
                "mknn": res["mknn"]["score"],
                "mknn_layers": res["mknn"]["layers"],
                "mknn_fixed_layer_shuffle": res["mknn"]["fixed_layer_shuffle_mean"],
                "mknn_selection_aware_shuffle": res["mknn"]["selection_aware_shuffle_mean"],
                "mknn_excess_fixed": res["mknn"]["excess_vs_fixed_layer_null"],
                "mknn_excess_sel": res["mknn"]["excess_vs_selection_aware_null"],
                "linear_cka": res["linear_cka"]["score"],
                "linear_cka_layers": res["linear_cka"]["layers"],
                "linear_cka_fixed_layer_shuffle": res["linear_cka"]["fixed_layer_shuffle_mean"],
                "linear_cka_selection_aware_shuffle": res["linear_cka"]["selection_aware_shuffle_mean"],
                "rbf_cka_sigma1": res["rbf_cka_sigma1"]["score"],
                "rbf_cka_layers": res["rbf_cka_sigma1"]["layers"],
                "rbf_fixed_layer_shuffle": res["rbf_cka_sigma1"]["fixed_layer_shuffle_mean"],
                "rbf_diag_frac": res["rbf_cka_sigma1"]["rbf_diag_diagnostic_on_selected_A"]["centred_diag_energy_frac"],
                "rbf_mean_offdiag_k": res["rbf_cka_sigma1"]["rbf_diag_diagnostic_on_selected_A"]["mean_offdiag_kernel"],
                "concat": res["concat_paper_extra"],
                "n_params_a": MODELS[ka].n_params,
                "n_params_b": MODELS[kb].n_params,
                "family_a": MODELS[ka].family,
                "family_b": MODELS[kb].family,
            }
        )
        write_json(out_root / "pair_table.json", pair_rows)

    # summaries
    def grp(kind):
        return [r for r in pair_rows if r["kinds"] == kind]

    summary = {}
    for kind in ["vision-language", "language-language", "vision-vision"]:
        rs = grp(kind)
        if not rs:
            continue
        summary[kind] = {
            "n_pairs": len(rs),
            "mknn_mean": float(np.mean([r["mknn"] for r in rs])),
            "mknn_fixed_shuffle_mean": float(np.mean([r["mknn_fixed_layer_shuffle"] for r in rs])),
            "mknn_sel_shuffle_mean": float(np.mean([r["mknn_selection_aware_shuffle"] for r in rs])),
            "linear_cka_mean": float(np.mean([r["linear_cka"] for r in rs])),
            "linear_cka_fixed_shuffle_mean": float(np.mean([r["linear_cka_fixed_layer_shuffle"] for r in rs])),
            "linear_cka_sel_shuffle_mean": _nanmean([r["linear_cka_selection_aware_shuffle"] for r in rs]),
            "rbf_sigma1_mean": float(np.mean([r["rbf_cka_sigma1"] for r in rs])),
            "rbf_sigma1_fixed_shuffle_mean": float(np.mean([r["rbf_fixed_layer_shuffle"] for r in rs])),
            "rbf_centred_diag_energy_frac_mean": float(np.mean([r["rbf_diag_frac"] for r in rs])),
            "rbf_mean_offdiag_k_mean": float(np.mean([r["rbf_mean_offdiag_k"] for r in rs])),
        }
    write_json(out_root / "summary.json", summary)
    write_json(out_root / "protocol.json", json.loads((ROOT / "configs" / "prh_released_code.json").read_text()))

    keys_sorted = sorted({r["a"] for r in pair_rows} | {r["b"] for r in pair_rows})
    idx = {k: i for i, k in enumerate(keys_sorted)}
    mat = np.full((len(keys_sorted), len(keys_sorted)), np.nan)
    for r in pair_rows:
        i, j = idx[r["a"]], idx[r["b"]]
        mat[i, j] = mat[j, i] = r["mknn"]
        mat[i, i] = mat[j, j] = 1.0
    figdir = out_root / "figures"
    heatmap(mat, keys_sorted, keys_sorted, "PRH released-code protocol on COCO test: mutual kNN k=10", figdir / "mknn_heatmap.png", "mNN")
    vl = [r for r in pair_rows if r["kinds"] == "vision-language"]
    labels = [f"{r['a']} vs {r['b']}" for r in vl]
    bars(
        labels,
        [r["mknn"] for r in vl],
        [r["mknn_selection_aware_shuffle"] for r in vl],
        "V–L mNN vs selection-aware max-over-layers shuffle mean",
        "mutual kNN",
        figdir / "mknn_vs_sel_shuffle.png",
    )
    bars(
        labels,
        [r["mknn"] for r in vl],
        [r["mknn_fixed_layer_shuffle"] for r in vl],
        "V–L mNN vs fixed-layer shuffle mean (~k/(n-1))",
        "mutual kNN",
        figdir / "mknn_vs_fixed_shuffle.png",
    )
    _family_size_plot(pair_rows, "Qwen3", "dinov2-small", figdir / "qwen3_size_vs_dinov2.png")
    _family_size_plot(pair_rows, "BLOOMZ", "dinov2-small", figdir / "bloomz_size_vs_dinov2.png")
    repo_out = ROOT / "results" / "prh_released_code"
    repo_out.mkdir(parents=True, exist_ok=True)
    (repo_out / "summary.json").write_text((out_root / "summary.json").read_text())
    (repo_out / "pair_table.json").write_text((out_root / "pair_table.json").read_text())
    (repo_out / "protocol.json").write_text((out_root / "protocol.json").read_text())
    fig_repo = repo_out / "figures"
    fig_repo.mkdir(parents=True, exist_ok=True)
    for pth in figdir.glob("*.png"):
        (fig_repo / pth.name).write_bytes(pth.read_bytes())
    print(json.dumps(summary, indent=2))


def _nanmean(xs):
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def _family_size_plot(pair_rows, family, vis_key, path):
    rows = []
    for r in pair_rows:
        if r["kinds"] != "vision-language":
            continue
        if vis_key not in (r["a"], r["b"]):
            continue
        lang = r["a"] if r["b"] == vis_key else r["b"]
        if r["family_a"] != family and r["family_b"] != family:
            continue
        n = r["n_params_a"] if r["a"] != vis_key else r["n_params_b"]
        rows.append((n / 1e9, r["mknn"], lang))
    if len(rows) < 2:
        return
    rows.sort()
    lines(
        [x for x, _, _ in rows],
        {f"mNN vs {vis_key}": [y for _, y, _ in rows]},
        f"{family} within-family size vs {vis_key} (PRH released-code on COCO)",
        "language params (B)",
        "mutual kNN",
        path,
    )


if __name__ == "__main__":
    main()
