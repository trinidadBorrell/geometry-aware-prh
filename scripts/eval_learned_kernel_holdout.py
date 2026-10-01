#!/usr/bin/env python3
"""CKA of learned RBF/RQ kernels on a disjoint 80/20 holdout.

Pools the cached COCO train+val+test gallery (held-out images were never
extracted). Layer choice, distance scales, and kernel hyperparameters are
fit on 80% only. CKA is scored on the other 20%. Does not overwrite
results/learned_kernels/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.datasets import load_manifest  # noqa: E402
from prh_replication.io_utils import write_json  # noqa: E402
from prh_replication.kernel_experiment import (  # noqa: E402
    FAMILIES,
    OBJECTIVES,
    candidate_list,
    center_and_norm,
    evaluate_kernel_full,
    jsonable,
    load_split_features,
    pair_name,
    score_candidate,
    select_from_landscape,
    split_fit_holdout,
    vl_pairs,
)
from prh_replication.kernels import linear_gram, median_offdiag_from_dsq, pairwise_sq_distances  # noqa: E402
from prh_replication.plots import _save  # noqa: E402
from prh_replication.prh_ref import stack_prepared  # noqa: E402
from prh_replication.registry import Paths  # noqa: E402

OUT_NAME = "learned_kernel_cka_80_20"
POOL_SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-frac", type=float, default=0.2)
    p.add_argument("--pairs-limit", type=int, default=0)
    return p.parse_args()


def _ids(man, split):
    return list(man["splits"][split])


def load_pooled(paths, key, man):
    chunks = []
    ids = []
    for split in POOL_SPLITS:
        sid = _ids(man, split)
        obj = load_split_features(paths, key, split, sid)
        chunks.append(obj["feats"])
        ids.extend(sid)
    feats = torch.cat(chunks, dim=0)
    if feats.shape[0] != len(ids):
        raise RuntimeError(key)
    return feats, ids


def layer_grams(x_nld: torch.Tensor):
    return [center_and_norm(linear_gram(x_nld[:, i])) for i in range(x_nld.shape[1])]


def best_layers(grams_a, grams_b) -> dict:
    best, best_ij, best_a = float("-inf"), (0, 0), float("nan")
    for i, (ka, nka) in enumerate(grams_a):
        if nka <= 0:
            continue
        for j, (kb, nkb) in enumerate(grams_b):
            if nkb <= 0:
                continue
            a = float((ka * kb).sum() / (nka * nkb))
            if a > best or (a == best and (i, j) < best_ij):
                best, best_ij, best_a = a, (i, j), a
    return {"layers": list(best_ij), "fit_linear_cka": best_a}


def plot_means(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["linear"] + [f"{fam}/{obj}" for fam in FAMILIES for obj in OBJECTIVES]
    keys = ["linear"] + [f"{fam}_{obj}" for fam in FAMILIES for obj in OBJECTIVES]
    ys = [summary["holdout_mean_cka_a"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(np.arange(len(labels)), ys, color="C0")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("mean holdout CKA a")
    ax.set_title("Learned kernels: CKA on 20% holdout\n(hyperparameters chosen on the other 80%)")
    ax.set_ylim(0, 1.05)
    fig.text(
        0.01,
        0.01,
        "18 vision–language pairs. CKA-objective selection can push a near 1. Ratio/excess stay near linear.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, path)


def main():
    args = parse_args()
    t0 = time.time()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out = paths.results / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    # Work-disk coco_val2017.json is a truncated copy. Split ids live in the repo slim manifest.
    man = load_manifest(ROOT / "data" / "manifests" / "coco_val2017_splits.json")

    pairs = vl_pairs()
    if args.pairs_limit:
        pairs = pairs[: args.pairs_limit]
    keys = []
    for a, b in pairs:
        if a not in keys:
            keys.append(a)
        if b not in keys:
            keys.append(b)

    pooled = {}
    id_ref = None
    for key in keys:
        feats, ids = load_pooled(paths, key, man)
        if id_ref is None:
            id_ref = ids
        elif ids != id_ref:
            raise RuntimeError(f"sample-id order mismatch {key}")
        pooled[key] = feats
        print(f"loaded {key} {tuple(feats.shape)}", flush=True)

    n = len(id_ref)
    fit_idx, hold_idx = split_fit_holdout(n, args.hold_frac, args.seed)
    if set(fit_idx.tolist()) & set(hold_idx.tolist()):
        raise RuntimeError("split overlap")
    print(f"n={n} fit={fit_idx.numel()} hold={hold_idx.numel()}", flush=True)

    prepared = {}
    for key in list(pooled):
        feats = pooled.pop(key)
        fit_x = stack_prepared(feats[fit_idx])
        hold_x = stack_prepared(feats[hold_idx])
        prepared[key] = {"fit": fit_x, "hold": hold_x}
        print(f"prepared {key}", flush=True)

    grams = {key: layer_grams(prepared[key]["fit"]) for key in keys}
    rows = []
    cands = candidate_list()
    for a, b in pairs:
        layers = best_layers(grams[a], grams[b])
        ia, ib = layers["layers"]
        xa = prepared[a]["fit"][:, ia]
        xb = prepared[b]["fit"][:, ib]
        dsq_a, dsq_b = pairwise_sq_distances(xa), pairwise_sq_distances(xb)
        sa, sb = median_offdiag_from_dsq(dsq_a), median_offdiag_from_dsq(dsq_b)
        landscape = [score_candidate(xa, xb, dsq_a, dsq_b, sa, sb, c) for c in cands]
        selected = {f"{fam}_{obj}": select_from_landscape(landscape, fam, obj) for fam in FAMILIES for obj in OBJECTIVES}
        ya = prepared[a]["hold"][:, ia]
        yb = prepared[b]["hold"][:, ib]
        hdsq_a, hdsq_b = pairwise_sq_distances(ya), pairwise_sq_distances(yb)
        hold = {
            "linear": evaluate_kernel_full(
                ya, yb, hdsq_a, hdsq_b, sa, sb, "linear", None, None, 0, 0, name="linear", diagnostics=False
            )
        }
        for fam in FAMILIES:
            for obj in OBJECTIVES:
                sel = selected[f"{fam}_{obj}"]
                hold[f"{fam}_{obj}"] = evaluate_kernel_full(
                    ya,
                    yb,
                    hdsq_a,
                    hdsq_b,
                    sa,
                    sb,
                    fam,
                    sel.get("lambda"),
                    sel.get("alpha"),
                    0,
                    0,
                    diagnostics=False,
                )
                hold[f"{fam}_{obj}"]["selected_on_fit"] = {
                    "lambda": sel.get("lambda"),
                    "alpha": sel.get("alpha"),
                    "fit_a": sel.get("train_a"),
                    "boundary": sel.get("boundary"),
                    "valid": sel.get("valid"),
                }
        rec = {
            "a": a,
            "b": b,
            "name": pair_name(a, b),
            "layers": layers,
            "fit_scales": {"s_a": sa, "s_b": sb},
            "n_fit": int(xa.shape[0]),
            "n_hold": int(ya.shape[0]),
            "holdout_cka_a": {k: hold[k].get("a") for k in hold},
            "holdout": jsonable(hold),
        }
        rows.append(rec)
        print(f"{rec['name']} holdout CKA {rec['holdout_cka_a']}", flush=True)
        del dsq_a, dsq_b, hdsq_a, hdsq_b, landscape

    keys_score = ["linear"] + [f"{fam}_{obj}" for fam in FAMILIES for obj in OBJECTIVES]
    means = {}
    for k in keys_score:
        vals = [r["holdout_cka_a"][k] for r in rows if r["holdout_cka_a"][k] is not None]
        means[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    summary = {
        "pool": list(POOL_SPLITS),
        "n_pool": n,
        "n_fit": int(fit_idx.numel()),
        "n_hold": int(hold_idx.numel()),
        "hold_frac": args.hold_frac,
        "seed": args.seed,
        "n_pairs": len(rows),
        "holdout_mean_cka_a": means,
        "note": (
            "Clip-then-L2 is computed separately on the fit and holdout tensors. "
            "Distance scales and kernel parameters are frozen from the fit split. "
            "CKA-objective selection is known to drive centred-Gram CKA near 1."
        ),
        "did_not_overwrite": "results/learned_kernels",
    }
    write_json(out / "summary.json", jsonable(summary))
    write_json(out / "pairs.json", jsonable(rows))
    fig = out / "figures" / "holdout_cka_means.png"
    plot_means(summary, fig)
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Learned-kernel CKA, 80/20 holdout",
                "",
                f"- Pool: cached COCO {', '.join(POOL_SPLITS)} (n={n}). The 904-image manifest heldout has no features.",
                f"- Fit n={int(fit_idx.numel())}, holdout n={int(hold_idx.numel())}, seed={args.seed}.",
                "- Layers, median scales, and RBF/RQ hyperparameters are chosen on the fit split only.",
                "- Reported number is holdout centred-Gram CKA `a`.",
                "",
                "Mean holdout CKA:",
                "",
                *[f"- `{k}`: {means[k]:.4f}" for k in keys_score],
                "",
                "```bash",
                "PYTHONPATH=src python scripts/eval_learned_kernel_holdout.py --work /mnt/sdb1/prh-replication-work",
                "```",
                "",
            ]
        )
    )
    print(json.dumps(means, indent=2), flush=True)
    print(f"wrote {out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
