#!/usr/bin/env python3
"""End-to-end runner: manifests, extract, metrics, plots, report fragments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.controls import control_battery
from prh_replication.datasets import build_coco_manifest, load_manifest, split_records
from prh_replication.evaluate import choose_val_layers, evaluate_pair, train_sigmas
from prh_replication.extract import extract_model, feature_path
from prh_replication.io_utils import load_features, write_json
from prh_replication.plots import bars, heatmap, lines
from prh_replication.registry import MODELS, Paths, default_paths


PANELS = {
    "A": ["bloomz-560m", "bloomz-1b1", "dinov2-small", "vit-in21k-small", "clip-laion-base"],
    "B": [
        "qwen3-0.6b-base",
        "qwen3-1.7b-base",
        "qwen2.5-0.5b",
        "olmo-1b-0724",
        "dinov2-small",
        "vit-in21k-small",
        "clip-laion-base",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["manifest", "smoke", "extract", "eval", "plots", "all"], default="all")
    p.add_argument("--split-limit", type=int, default=0, help="If >0, truncate each split (smoke).")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--work", type=str, default=None)
    p.add_argument("--dataset", default="coco_val2017")
    p.add_argument("--protocol", default="modern", choices=["modern", "original"])
    return p.parse_args()


def paths_from_args(args) -> Paths:
    repo = ROOT
    if args.work:
        return Paths(work=Path(args.work), repo=repo)
    return default_paths(repo)


def prepare_env(paths: Paths):
    os.environ.setdefault("HF_HOME", str(paths.hf))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(paths.hf / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths.hf / "transformers"))
    os.environ.setdefault("TIMM_CACHE", str(paths.hf / "timm"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def get_records(paths: Paths, split: str, limit: int):
    man_path = paths.data / "manifests" / "coco_val2017.json"
    if not man_path.exists():
        build_coco_manifest(paths)
    man = load_manifest(man_path)
    recs = split_records(man, split)
    if limit:
        recs = recs[:limit]
    return man, recs


def selected_models(args) -> list[str]:
    if args.models:
        return args.models
    keys = []
    for panel in PANELS.values():
        for k in panel:
            if k not in keys:
                keys.append(k)
    return keys


def run_extract(args, paths: Paths, splits: list[str]):
    keys = selected_models(args)
    written = []
    for split in splits:
        _, recs = get_records(paths, split, args.split_limit)
        for key in keys:
            spec = MODELS[key]
            try:
                path = extract_model(spec, recs, paths, args.dataset, split, args.protocol)
                written.append(str(path))
            except Exception as e:
                err = {
                    "model": key,
                    "split": split,
                    "error": repr(e),
                }
                write_json(paths.results / "errors" / f"{key}_{split}.json", err)
                print("FAILED", key, split, e)
    return written


def _load(paths, key, split, args):
    spec = MODELS[key]
    _, recs = get_records(paths, split, args.split_limit)
    ids = [r["sample_id"] for r in recs]
    p = feature_path(spec=spec, paths=paths, dataset=args.dataset, split=split, protocol=args.protocol, sample_ids=ids)
    if not p.exists():
        return None
    return load_features(p)


def run_eval(args, paths: Paths):
    keys = [k for k in selected_models(args) if _load(paths, k, "train", args) is not None]
    sigmas = {}
    for k in keys:
        sigmas[k] = train_sigmas(_load(paths, k, "train", args), args.protocol)
    write_json(paths.results / args.protocol / "rbf_scales.json", {k: {str(a): b for a, b in v.items()} for k, v in sigmas.items()})

    pair_rows = []
    val_choices = {}
    available_test = [k for k in keys if _load(paths, k, "test", args)]
    available_val = [k for k in keys if _load(paths, k, "val", args)]

    for a, b in combinations(available_val, 2):
        fa, fb = _load(paths, a, "val", args), _load(paths, b, "val", args)
        if args.protocol == "modern":
            val_res = evaluate_pair(fa, fb, args.protocol, "val", sigmas[a], sigmas[b], layer_mode="grid")
            choice = choose_val_layers(val_res["grid"])
            val_choices[f"{a}||{b}"] = choice
            val_res["chosen_layers"] = list(choice)
            write_json(paths.results / args.protocol / "val_pairs" / f"{a}__{b}.json", val_res)
        else:
            val_res = evaluate_pair(fa, fb, args.protocol, "val", sigmas[a], sigmas[b], layer_mode="max")
            write_json(paths.results / args.protocol / "val_pairs" / f"{a}__{b}.json", val_res)

    for a, b in combinations(available_test, 2):
        fa, fb = _load(paths, a, "test", args), _load(paths, b, "test", args)
        if args.protocol == "modern":
            choice = val_choices.get(f"{a}||{b}") or val_choices.get(f"{b}||{a}")
            if choice is None:
                continue
            res = evaluate_pair(fa, fb, args.protocol, "test", sigmas[a], sigmas[b], layer_mode="grid", val_choice=choice)
        else:
            res = evaluate_pair(fa, fb, args.protocol, "test", sigmas[a], sigmas[b], layer_mode="max")
        res["model_a"] = a
        res["model_b"] = b
        write_json(paths.results / args.protocol / "test_pairs" / f"{a}__{b}.json", res)
        if "mknn" in res:
            pair_rows.append(
                {
                    "a": a,
                    "b": b,
                    "mknn": res["mknn"]["score"],
                    "mknn_excess": res["mknn"]["excess"],
                    "mknn_shuffle_mean": res["mknn"]["shuffle_mean"],
                    "mknn_shuffle_std": res["mknn"]["shuffle_std"],
                    "linear_cka": res["linear_cka"]["score"],
                    "linear_cka_excess": res["linear_cka"]["excess"],
                    "layers": res["mknn"]["layers"],
                    "kinds": f"{MODELS[a].kind}-{MODELS[b].kind}",
                    "language_sup_a": MODELS[a].language_supervised,
                    "language_sup_b": MODELS[b].language_supervised,
                    "n_params_a": MODELS[a].n_params,
                    "n_params_b": MODELS[b].n_params,
                    "family_a": MODELS[a].family,
                    "family_b": MODELS[b].family,
                }
            )
    write_json(paths.results / args.protocol / "pair_table.json", pair_rows)

    # controls on one representative pair if possible
    vis = [k for k in available_test if MODELS[k].kind == "vision"]
    lang = [k for k in available_test if MODELS[k].kind == "language"]
    if vis and lang:
        fa, fb = _load(paths, lang[0], "test", args), _load(paths, vis[0], "test", args)
        from prh_replication.metrics import apply_layer_prep

        xa = apply_layer_prep(fa["feats"][:, -1].float(), args.protocol)
        xb = apply_layer_prep(fb["feats"][:, -1].float(), args.protocol)
        write_json(
            paths.results / args.protocol / "controls_last_layer.json",
            {"a": lang[0], "b": vis[0], **control_battery(xa, xb)},
        )
    return pair_rows


def run_plots(args, paths: Paths):
    table = json.loads((paths.results / args.protocol / "pair_table.json").read_text())
    if not table:
        return
    keys = sorted({r["a"] for r in table} | {r["b"] for r in table})
    idx = {k: i for i, k in enumerate(keys)}
    mat = np.full((len(keys), len(keys)), np.nan)
    excess = np.full_like(mat, np.nan)
    for r in table:
        i, j = idx[r["a"]], idx[r["b"]]
        mat[i, j] = mat[j, i] = r["mknn"]
        excess[i, j] = excess[j, i] = r["mknn_excess"]
        mat[i, i] = 1.0
        mat[j, j] = 1.0
    figdir = paths.results / args.protocol / "figures"
    heatmap(mat, keys, keys, f"mutual kNN k=10 ({args.protocol}, test)", figdir / "mknn_heatmap.png", "mNN")
    heatmap(excess, keys, keys, "mNN excess over shuffle mean", figdir / "mknn_excess_heatmap.png", "actual - shuffle mean")
    labels = [f"{r['a']} vs {r['b']}" for r in table]
    bars(labels, [r["mknn"] for r in table], [r["mknn_shuffle_mean"] for r in table],
         "Actual mNN vs shuffle-null mean", "mutual kNN", figdir / "mknn_vs_shuffle.png")
    # size trend within Qwen family if present
    lang_vis = [r for r in table if r["kinds"] == "language-vision" or r["kinds"] == "vision-language"]
    qwen = [r for r in table if "Qwen3" in r["family_a"] + r["family_b"] and "dinov2" in r["a"] + r["b"]]
    if qwen:
        qwen_sorted = sorted(qwen, key=lambda r: r["n_params_a"] if "qwen" in r["a"] else r["n_params_b"])
        xs = [ (r["n_params_a"] if "qwen" in r["a"] else r["n_params_b"]) / 1e9 for r in qwen_sorted]
        lines(xs, {"mNN vs DINOv2-S": [r["mknn"] for r in qwen_sorted]},
              "Qwen3 size vs DINOv2-S alignment (not a scaling law)",
              "language params (B)", "mNN", figdir / "qwen_size_trend.png")
    # copy figures into repo
    repo_fig = paths.repo / "results" / args.protocol / "figures"
    repo_fig.mkdir(parents=True, exist_ok=True)
    for p in figdir.glob("*.png"):
        (repo_fig / p.name).write_bytes(p.read_bytes())
    write_json(paths.repo / "results" / args.protocol / "pair_table.json", table)


def main():
    args = parse_args()
    paths = paths_from_args(args)
    prepare_env(paths)
    paths.work.mkdir(parents=True, exist_ok=True)
    if args.stage in ("manifest", "all", "smoke"):
        build_coco_manifest(paths)
    splits = ["train", "val", "test"]
    if args.stage == "smoke":
        args.split_limit = args.split_limit or 32
        args.models = args.models or ["dinov2-small", "qwen3-0.6b-base"]
        run_extract(args, paths, splits)
        run_eval(args, paths)
        run_plots(args, paths)
        return
    if args.stage in ("extract", "all"):
        run_extract(args, paths, splits)
    if args.stage in ("eval", "all"):
        proto = args.protocol
        for protocol in ([proto] if args.stage == "eval" else ["modern", "original"]):
            args.protocol = protocol
            run_eval(args, paths)
            if args.stage in ("plots", "all"):
                run_plots(args, paths)
        args.protocol = proto
    elif args.stage == "plots":
        run_plots(args, paths)


if __name__ == "__main__":
    main()
