"""Figures and summary for exp-011 from run.py output.

    uv run python experiments/exp-011-synthetic-geometry/plot.py --root results/exp-011-synthetic-geometry

* fig1_id            estimated vs true ID (Levina-Bickel, TwoNN), per manifold, n = 1k and 10k
* fig2_geodesic      Spearman of graph geodesics (k = 10, 30, 100) and of Euclidean distances
                     with the true geodesics (flat, sphere, Swiss roll)
* fig3_curvature     delta_rel from graph geodesics, and mean edge Ollivier-Ricci curvature
                     (k = 10, 30), per manifold and ID
* fig4_local_pca     local PCA dimension / ID vs neighbourhood size (n = 10k)
* summary.txt        the numbers behind the figures, noise-free and noisy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
DIMS = {5: "#86b6ef", 20: "#2a78d6", 40: "#104281"}  # blue 250/450/650, ordinal in ID
MANIFOLDS = ["flat", "sphere", "swiss", "tree", "clusters"]
TRUTH = ["flat", "sphere", "swiss"]  # manifolds with closed-form geodesics

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 2,
    }
)


def get(recs, manifold, d, n, noise=0.0):
    for r in recs:
        if (r["manifold"], r["d"], r["n"], r["noise"]) == (manifold, d, n, noise):
            return r
    return None


def save(fig, root: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def dim_legend(ax):
    for d, c in DIMS.items():
        ax.plot([], [], "o", color=c, label=f"ID {d}")
    ax.legend(frameon=False, fontsize=8)


def fig_id(recs, root, sizes):
    fig, axes = plt.subplots(2, len(sizes), figsize=(4 * len(sizes), 6.4), squeeze=False)
    for row, (key, label) in enumerate((("id_lb", "Levina-Bickel"), ("id_twonn", "TwoNN"))):
        for col, n in enumerate(sizes):
            ax = axes[row, col]
            for x, m in enumerate(MANIFOLDS):
                for d, c in DIMS.items():
                    r = get(recs, m, d, n)
                    if r:
                        ax.plot(x, r[key] / d, "o", color=c, ms=7, mec=SURFACE, mew=1)
            ax.axhline(1, color=INK_MUTED, lw=0.8)
            ax.set_xticks(range(len(MANIFOLDS)), MANIFOLDS)
            ax.set_ylabel(f"{label} estimate / true ID")
            ax.set_title(f"{label}, n = {n}, no noise")
            ax.set_ylim(0, 1.3)
    dim_legend(axes[0, 0])
    save(fig, root, "fig1_id")


def fig_geodesic(recs, root, sizes, ks):
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.5 * len(sizes), 3.6), squeeze=False)
    for col, n in enumerate(sizes):
        ax = axes[0, col]
        labels = [f"graph k={k}" for k in ks] + ["Euclidean"]
        for gi, m in enumerate(TRUTH):
            for d, c in DIMS.items():
                r = get(recs, m, d, n)
                if not r:
                    continue
                vals = [r["graph"][str(k)].get("spearman", np.nan) for k in ks]
                vals.append(r["euclid_vs_true"]["spearman"])
                xs = gi * (len(labels) + 1) + np.arange(len(labels))
                ax.plot(xs, vals, "o-", color=c, ms=5, lw=1.2)
        ticks = [gi * (len(labels) + 1) + i for gi in range(len(TRUTH)) for i in range(len(labels))]
        ax.set_xticks(ticks, labels * len(TRUTH), rotation=60, fontsize=7)
        for gi, m in enumerate(TRUTH):
            ax.annotate(m, (gi * (len(labels) + 1) + 1.5, 1.02), ha="center", fontsize=9)
        ax.set_ylabel("Spearman with true geodesic")
        ax.set_title(f"n = {n}, no noise", pad=14)
        ax.set_ylim(None, 1.05)
    dim_legend(axes[0, 0])
    save(fig, root, "fig2_geodesic")


def fig_curvature(recs, root, sizes, orc_ks):
    panels = [("delta", "graph", "delta_rel, graph geodesics (k=30)")]
    panels += [("orc", str(k), f"mean edge Ollivier-Ricci, k={k}") for k in orc_ks]
    fig, axes = plt.subplots(
        len(sizes), len(panels), figsize=(4 * len(panels), 3.2 * len(sizes)), squeeze=False
    )
    for row, n in enumerate(sizes):
        for col, (kind, key, title) in enumerate(panels):
            ax = axes[row, col]
            for x, m in enumerate(MANIFOLDS):
                for j, (d, c) in enumerate(DIMS.items()):
                    r = get(recs, m, d, n)
                    if not r or key not in r[kind]:
                        continue
                    v = (
                        r[kind][key]["delta_rel"]
                        if kind == "delta"
                        else r[kind][key]["edge"]["mean"]
                    )
                    ax.plot(x + (j - 1) * 0.18, v, "o", color=c, ms=7, mec=SURFACE, mew=1)
            if kind == "orc":
                ax.axhline(0, color=INK_MUTED, lw=0.8)
            ax.set_xticks(range(len(MANIFOLDS)), MANIFOLDS)
            ax.set_title(f"{title}\nn = {n}, no noise")
    dim_legend(axes[0, 0])
    save(fig, root, "fig3_curvature")


def fig_local_pca(recs, root, n):
    fig, axes = plt.subplots(1, len(DIMS), figsize=(4 * len(DIMS), 3.4), sharey=True)
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical 1-5
    for ax, d in zip(axes, DIMS, strict=True):
        for m, c in zip(MANIFOLDS, colours, strict=True):
            r = get(recs, m, d, n)
            if r:
                ks = [p["k"] for p in r["local_pca"]]
                ax.plot(ks, [p["dim"] / d for p in r["local_pca"]], "o-", color=c, ms=4, label=m)
        ax.set_xscale("log")
        ax.axhline(1, color=INK_MUTED, lw=0.8)
        ax.set_xlabel("neighbourhood size k")
        ax.set_title(f"ID {d}, n = {n}, no noise")
    axes[0].set_ylabel("local PCA dimension (90%) / ID")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, root, "fig4_local_pca")


def summary(recs, root, ks, orc_ks):
    head = (
        f"{'manifold':<9} {'d':>3} {'n':>6} {'noise':>5} {'LB':>6} {'2NN':>6} "
        + " ".join(f"{'geo' + str(k):>6}" for k in ks)
        + f" {'euclid':>6} {'comp30':>6} {'dG':>6} {'dE':>6} {'dT':>6} "
        + " ".join(f"{'orc' + str(k):>7}" for k in orc_ks)
        + f" {'pca10':>6} {'pca500':>6} {'sec':>5}"
    )
    lines = [head]
    for r in sorted(
        recs, key=lambda r: (r["n"], r["d"], MANIFOLDS.index(r["manifold"]), r["noise"])
    ):
        geo = [r["graph"][str(k)].get("spearman", np.nan) for k in ks]
        eu = r.get("euclid_vs_true", {}).get("spearman", np.nan)
        dl = r["delta"]
        orc = [r["orc"].get(str(k), {}).get("edge", {}).get("mean", np.nan) for k in orc_ks]
        pca = {p["k"]: p["dim"] for p in r["local_pca"]}
        lines.append(
            f"{r['manifold']:<9} {r['d']:>3} {r['n']:>6} {r['noise']:>5} {r['id_lb']:>6.1f} "
            f"{r['id_twonn']:>6.1f} "
            + " ".join(f"{v:>6.3f}" for v in geo)
            + f" {eu:>6.3f} {r['graph'].get('30', {}).get('n_components', -1):>6} "
            f"{dl['graph']['delta_rel']:>6.3f} {dl['euclid']['delta_rel']:>6.3f} "
            f"{dl.get('true', {}).get('delta_rel', np.nan):>6.3f} "
            + " ".join(f"{v:>+7.3f}" for v in orc)
            + f" {pca.get(10, np.nan):>6.1f} {pca.get(500, np.nan):>6.1f} {r['seconds']:>5.0f}"
        )
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recs = [json.loads(line) for line in (args.root / "synthetic.jsonl").read_text().splitlines()]
    sizes = sorted({r["n"] for r in recs})
    ks = sorted({int(k) for r in recs for k in r["graph"]})
    orc_ks = sorted({int(k) for r in recs for k in r["orc"]})
    summary(recs, args.root, ks, orc_ks)
    fig_id(recs, args.root, sizes)
    fig_geodesic(recs, args.root, sizes, ks)
    fig_curvature(recs, args.root, sizes, orc_ks)
    fig_local_pca(recs, args.root, max(sizes))


if __name__ == "__main__":
    main()
