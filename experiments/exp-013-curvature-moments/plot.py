"""Figures and summary for exp-013 from run.py output.

    uv run python experiments/exp-013-curvature-moments/plot.py --root results/exp-013-curvature-moments/v2

* fig1_synthetic     estimated scalar curvature S / (d(d-1)) per manifold and ID, n = 10k / 100k,
                     noise 0.1: filled = d chosen from the data, open = true ID; dashed = truth
* fig2_profile       scale profile at the chosen d: the estimate from consecutive neighbourhood
                     sizes vs the neighbourhood radius, ID 20, n = 10k and 100k
* fig3_map           linear vs intrinsic: PCA90 / chosen d against S / (d(d-1)); synthetic
                     (ID 20, 40, n = 10k) and every model layer
* fig4_models        per model layer: S / (d(d-1)) and chosen d vs relative depth
* fig5_d_sensitivity S / (d(d-1)) as a function of the d used, relative to the true ID (ID 20,
                     n = 10k), with the chosen d marked
* summary.txt        the numbers behind the figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
SIZES = {10000: "#2a78d6", 100000: "#104281"}  # blue ramp, ordinal in n
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MANIFOLDS = ["flat", "swiss", "torus_rff", "sphere", "veronese", "clusters", "tree"]

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


def save(fig, root: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def get(recs, m, d, n, noise=0.1):
    for r in recs:
        if (r["manifold"], r["d"], r["n"], r["noise"]) == (m, d, n, noise):
            return r
    return None


def truth_norm(r):
    s, d = r["scalar_true"], r["d"]
    return None if s is None or d < 2 else s / (d * (d - 1))


def fig_synthetic(recs, root):
    dims = sorted({r["d"] for r in recs if r["d"] > 1})
    fig, axes = plt.subplots(1, len(dims), figsize=(3.6 * len(dims), 3.6), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dims, strict=True):
        for x, m in enumerate(MANIFOLDS):
            for j, (n, c) in enumerate(SIZES.items()):
                r = get(recs, m, d, n)
                if r:
                    xx = x + (j - 0.5) * 0.3
                    v = r["curv"]["chosen"]["pooled"]["S_norm"]
                    ax.plot(xx, np.clip(v, -1.5, 4), "o", color=c, ms=6)
                    if "true" in r["curv"]:
                        v = r["curv"]["true"]["pooled"]["S_norm"]
                        ax.plot(xx, np.clip(v, -1.5, 4), "o", mfc="none", mec=c, ms=9)
            t = truth_norm(get(recs, m, d, 10000) or {"scalar_true": None, "d": d})
            if t is not None:
                ax.plot([x - 0.35, x + 0.35], [t, t], color=INK_MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(MANIFOLDS)), MANIFOLDS, rotation=40, fontsize=8)
        ax.set_title(f"ID {d}, noise 0.1")
        ax.axhline(0, color=GRID, lw=0.8)
    np.atleast_1d(axes)[0].set_ylabel("estimated S / d(d-1)  (clipped to [-1.5, 4])")
    for n, c in SIZES.items():
        np.atleast_1d(axes)[0].plot([], [], "o", color=c, label=f"n = {n}")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    save(fig, root, "fig1_synthetic")


def fig_profile(recs, root, d=20):
    shown = ["flat", "torus_rff", "sphere", "veronese"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, n in zip(axes, (10000, 100000), strict=True):
        for m, c in zip(shown, CAT, strict=False):
            r = get(recs, m, d, n)
            if not r:
                continue
            prof = r["curv"]["chosen"]["profile"]
            ax.plot(
                [p["radius"] for p in prof],
                np.clip([p["S_norm"] for p in prof], -2, 5),
                "o-",
                color=c,
                ms=4,
                label=m,
            )
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.axhline(1, color=INK_MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("neighbourhood radius (data units)")
        ax.set_title(f"ID {d}, n = {n}: estimate between consecutive k")
    axes[0].set_ylabel("S / d(d-1)  (clipped to [-2, 5])")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, root, "fig2_profile")


def fig_map(recs, models, root):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m, c in zip(MANIFOLDS, CAT, strict=False):
        pts = [r for r in recs if r["manifold"] == m and r["n"] == 10000 and r["d"] in (20, 40)]
        pts = [r for r in pts if r["noise"] == 0.1]
        if pts:
            ax.scatter(
                [r["pca"]["n90"] / r["curv"]["chosen"]["d"] for r in pts],
                np.clip([r["curv"]["chosen"]["pooled"]["S_norm"] for r in pts], -2, 10),
                marker="s",
                s=40,
                color=c,
                label=f"synthetic: {m}",
            )
    for modality, marker in (("language", "o"), ("vision", "^")):
        ms = [r for r in models if r.get("modality") == modality]
        if ms:
            ax.scatter(
                [r["pca"]["n90"] / r["curv"]["chosen"]["d"] for r in ms],
                np.clip([r["curv"]["chosen"]["pooled"]["S_norm"] for r in ms], -2, 10),
                marker=marker,
                s=14,
                color=INK_MUTED,
                alpha=0.6,
                label=f"model layers: {'LLM' if modality == 'language' else 'ViT'}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("PCA90 / chosen d  (linear dimension relative to intrinsic)")
    ax.set_ylabel("estimated S / d(d-1)  (intrinsic, clipped to [-2, 10])")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_title("Nonlinear in the linear sense need not mean intrinsically curved")
    save(fig, root, "fig3_map")


def fig_models(models, root):
    names = list(dict.fromkeys(r["model"] for r in models if "layer" in r))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, c in zip(names, CAT, strict=False):
        rs = sorted([r for r in models if r.get("model") == name], key=lambda r: r["layer"])
        depth = [r["layer"] / (r["num_layers"] - 1) for r in rs]
        ls = "-" if rs[0]["modality"] == "language" else "--"
        label = name.split("/")[-1]
        s_norm = [r["curv"]["chosen"]["pooled"]["S_norm"] for r in rs]
        axes[0].plot(depth, s_norm, ls, color=c, label=label)
        axes[1].plot(depth, [r["curv"]["chosen"]["d"] for r in rs], ls, color=c)
        axes[1].plot(depth, [r["id_twonn"] for r in rs], ":", color=c, lw=1)
    axes[0].set_ylabel("S / d(d-1)  (intrinsic)")
    axes[1].set_ylabel("chosen d (solid / dashed), TwoNN ID (dotted)")
    for ax in axes:
        ax.set_xlabel("relative depth")
        ax.axhline(0, color=INK_MUTED, lw=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "PRH features, n = 10240, d chosen from the data; solid = LLM, dashed = ViT", fontsize=9
    )
    save(fig, root, "fig4_models")


def fig_d_sensitivity(recs, root, d=20, n=10000):
    shown = ["flat", "swiss", "torus_rff", "sphere", "veronese"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, c in zip(shown, CAT, strict=False):
        r = get(recs, m, d, n)
        if not r:
            continue
        sbd = r["curv"]["S_by_d"]
        ds = sorted(int(k) for k in sbd)
        ys = np.clip([sbd[str(k)] for k in ds], -5, 10)
        ax.plot([k / d for k in ds], ys, "o-", color=c, ms=3, label=m)
        ch = r["curv"]["chosen"]["d"]
        ax.plot(ch / d, np.clip(sbd[str(ch)], -5, 10), "s", color=c, ms=9, mfc="none")
    ax.axvline(1, color=INK_MUTED, lw=0.8, ls="--")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("d used / true ID")
    ax.set_ylabel("S / d(d-1)  (clipped to [-5, 10])")
    ax.set_title(f"Sensitivity to d: ID {d}, n = {n}, noise 0.1 (square = chosen d)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, root, "fig5_d_sensitivity")


def summary(recs, models, root):
    lines = ["synthetic (pooled over the 3 smallest usable k): S/d(d-1)"]
    lines.append(
        f"{'manifold':<10} {'d':>3} {'n':>6} {'noise':>5} {'true':>5} {'2NN':>6} {'chosen':>6} "
        f"{'S@chosen':>9} {'S@true':>8} {'S@2NN':>8} {'PCA90/d':>7}"
    )
    order = lambda r: (r["d"], r["n"], MANIFOLDS.index(r["manifold"]), r["noise"])  # noqa: E731
    for r in sorted(recs, key=order):
        t = truth_norm(r)
        c = r["curv"]
        s_true = c["true"]["pooled"]["S_norm"] if "true" in c else float("nan")
        tt = "-" if t is None else f"{t:.0f}"
        lines.append(
            f"{r['manifold']:<10} {r['d']:>3} {r['n']:>6} {r['noise']:>5} {tt:>5} "
            f"{r['id_twonn']:>6.1f} {c['chosen']['d']:>6} "
            f"{c['chosen']['pooled']['S_norm']:>+9.3f} {s_true:>+8.3f} "
            f"{c['twonn']['pooled']['S_norm']:>+8.3f} {r['pca']['n90'] / r['d']:>7.1f}"
        )
    if models:
        lines += ["", "models (PRH, n = 10240, d chosen from the data):"]
        lines.append(
            f"{'model':<45} {'layer':>5} {'2NN':>5} {'d':>3} {'PCA90/d':>7} {'S@chosen':>9} "
            f"{'S@2NN':>8}"
        )
        for r in sorted(models, key=lambda r: (r["model"], r["layer"])):
            c = r["curv"]
            lines.append(
                f"{r['model']:<45} {r['layer']:>5} {r['id_twonn']:>5.1f} {c['chosen']['d']:>3} "
                f"{r['pca']['n90'] / c['chosen']['d']:>7.1f} "
                f"{c['chosen']['pooled']['S_norm']:>+9.3f} {c['twonn']['pooled']['S_norm']:>+8.3f}"
            )
    (root / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    read = lambda f: [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []  # noqa: E731
    recs = read(args.root / "synthetic.jsonl")
    models = read(args.root / "models.jsonl")
    summary(recs, models, args.root)
    if recs:
        fig_synthetic(recs, args.root)
        fig_profile(recs, args.root)
        fig_d_sensitivity(recs, args.root)
    fig_map(recs, models, args.root)
    if models:
        fig_models(models, args.root)


if __name__ == "__main__":
    main()
