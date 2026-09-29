"""Figures and summary table for exp-007 from sweep.py output.

    uv run python experiments/exp-007-cknna-variants/plot.py --root results/exp-007-cknna-variants

Reads <root>/grid and <root>/vision_vision; writes PNG + PDF to <root>/figures:

* fig1_variants       cross-modal: every CKNNA variant, mutual kNN and linear CKA vs k, mean over
                      all (LLM, ViT) pairs of the max over layer pairs (band = min..max)
* fig2_vision_vision  last-block ViT x ViT (PRH Fig. 12 setting), MAE x MAE vs other pairs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
# categorical slots in fixed order (reference palette): blue, orange, aqua, yellow, magenta
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METRICS = {
    "mknn": ("mutual kNN", SERIES[0]),
    "paper": ("CKNNA paper (row-centred)", SERIES[1]),
    "centred": ("CKNNA centred (HKH)", SERIES[2]),
    "code": ("CKNNA code (platonic-rep)", SERIES[3]),
    "eq36": ("CKNNA Groger Eq. 36", SERIES[4]),
}
XTICKS = ([10, 200, 400, 600, 800, 1023], ["10", "200", "400", "600", "800", "1024"])
TABLE_KS = [10, 100, 200, 400, 500, 600, 800, 1000, 1023]

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


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}.png")


def xaxis(ax) -> None:
    ax.set_xticks(XTICKS[0])
    ax.set_xticklabels(XTICKS[1])
    ax.set_xlabel("k (n = 1024)")


def legend(fig, ax, ncol=3) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.12)
    )


def panel(ax, ks, curves: dict[str, np.ndarray], cka: np.ndarray) -> None:
    """curves: {metric: [n_items, len(ks)]}; cka: [n_items]."""
    ax.axhline(1.0, color=INK_MUTED, lw=0.8)
    for name, y in curves.items():
        label, color = METRICS[name]
        ax.fill_between(ks, y.min(0), y.max(0), color=color, alpha=0.12, lw=0)
        ax.plot(ks, y.mean(0), color=color, label=label)
    # CKA does not depend on k, but it varies across pairs: same min..max band, constant in k
    ax.axhspan(cka.min(), cka.max(), color=INK, alpha=0.07, lw=0)
    ax.axhline(cka.mean(), color=INK, ls="--", lw=1.5, label="linear CKA")
    xaxis(ax)


def load_grid(root: Path) -> list[dict]:
    path = root / "grid" / "sweep_pairs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def grid_curves(recs: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    ks = np.array([p["k"] for p in recs[0]["per_k"]])
    curves = {m: np.array([[p[m] for p in r["per_k"]] for r in recs]) for m in METRICS}
    return ks, curves, np.array([r["cka"] for r in recs])


def fig1_variants(recs: list[dict], out: Path) -> None:
    ks, curves, cka = grid_curves(recs)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    panel(ax, ks, curves, cka)
    ax.set_ylabel("score (max over layer pairs)")
    legend(fig, ax)
    fig.suptitle(
        f"CKNNA definitions vs k, {len(recs)} (LLM, ViT) pairs (band = min..max)",
        y=1.2,
        color=INK,
    )
    save(fig, out, "fig1_variants")


def fig2_vision_vision(path: Path, out: Path) -> None:
    d = np.load(path)
    ks, lvms = d["ks"], [str(m) for m in d["lvms"]]
    mae = np.array([".mae" in m for m in lvms])
    iu = np.triu_indices(len(lvms), 1)
    groups = {
        "MAE x MAE": mae[iu[0]] & mae[iu[1]],
        "all other pairs": ~(mae[iu[0]] & mae[iu[1]]),
    }
    groups = {name: sel for name, sel in groups.items() if sel.any()}
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, sel) in zip(axes, groups.items(), strict=True):
        curves = {m: d[m][:, iu[0], iu[1]][:, sel].T for m in METRICS}
        panel(ax, ks, curves, d["cka"][iu][sel])
        ax.set_title(f"{name} ({int(sel.sum())} pairs)", color=INK)
    axes[0].set_ylabel("score, last-block CLS")
    legend(fig, axes[0])
    fig.suptitle("Vision-vision (PRH Fig. 12 setting)", y=1.22, color=INK)
    save(fig, out, "fig2_vision_vision")


def table(recs: list[dict]) -> None:
    """Markdown table: mean over pairs at selected k."""
    ks, curves, cka = grid_curves(recs)
    cols = [i for i, k in enumerate(ks) if k in TABLE_KS]
    print("| k | " + " | ".join(str(ks[i]) for i in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, (label, _) in METRICS.items():
        vals = curves[name].mean(0)
        print(f"| {label} | " + " | ".join(f"{vals[i]:.2f}" for i in cols) + " |")
    print(f"\nlinear CKA: {cka.mean():.2f}")
    for name in METRICS:
        peak = curves[name].mean(0).argmax()
        print(f"{name}: max {curves[name].max():.2f}, mean-curve peak at k={ks[peak]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "figures"
    if (args.root / "grid" / "sweep_pairs.jsonl").exists():
        recs = load_grid(args.root)
        fig1_variants(recs, out)
        table(recs)
    vv = args.root / "vision_vision" / "vision_vision.npz"
    if vv.exists():
        fig2_vision_vision(vv, out)


if __name__ == "__main__":
    main()
