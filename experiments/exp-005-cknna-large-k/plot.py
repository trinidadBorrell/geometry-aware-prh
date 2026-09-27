"""Figures for exp-005 from cknna_k_sweep.py output.

    uv run python experiments/exp-005-cknna-large-k/plot.py --root results/exp-005-cknna-large-k

Reads <root>/grid and <root>/vision_vision; writes PNG + PDF to <root>/figures:

* fig1_convergence    mutual kNN(k) and CKNNA(k) with linear CKA overlaid, one panel per vision
                      family (largest model), mean over LLMs (band = min..max)
* fig2_families       CKNNA(k) for every ViT, mean over LLMs, coloured by vision family
* fig3_vision_vision  last-block ViT x ViT (PRH Fig. 12 setting), MAE vs other pairs
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
METRIC_COLOR = {"mutual_knn": SERIES[0], "cknna": SERIES[1], "cka": SERIES[2]}
FAMILIES = [
    ("ImageNet21K", lambda n: "augreg" in n),
    ("MAE", lambda n: ".mae" in n),
    ("DINOv2", lambda n: "dinov2" in n),
    ("CLIP", lambda n: "clip" in n and "ft_in12k" not in n),
    ("CLIP (I12K ft)", lambda n: "ft_in12k" in n),
]
SIZE_ORDER = ["tiny", "small", "base", "large", "huge", "giant"]
XTICKS = ([10, 200, 400, 600, 800, 1023], ["10", "200", "400", "600", "800", "1024"])

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


def family(lvm: str) -> int:
    return next(i for i, (_, match) in enumerate(FAMILIES) if match(lvm))


def size(lvm: str) -> int:
    return next((i for i, s in enumerate(SIZE_ORDER) if f"_{s}_" in lvm), 0)


def short(lvm: str) -> str:
    return f"{FAMILIES[family(lvm)][0]} {SIZE_ORDER[size(lvm)]}"


def series(rec: dict, key: str) -> np.ndarray:
    return np.array([p[key] for p in rec["per_k"]])


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}.png")


def _xaxis(ax, rotate: bool = True) -> None:
    ax.set_xticks(XTICKS[0])
    ax.set_xticklabels(XTICKS[1], rotation=45 if rotate else 0)
    ax.set_xlabel("k (n = 1024)")


def fig1_convergence(recs: list[dict], out: Path) -> None:
    lvms = sorted({r["lvm"] for r in recs}, key=lambda m: (family(m), size(m)))
    # the largest model of each vision family present
    fams = sorted({family(m) for m in lvms})
    panels = [max((m for m in lvms if family(m) == f), key=size) for f in fams]
    n_llms = len({r["llm"] for r in recs})
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lvm in zip(axes, panels, strict=True):
        rs = [r for r in recs if r["lvm"] == lvm]
        ks = series(rs[0], "k")
        for key, label in (("mutual_knn", "mutual kNN(k)"), ("cknna", "CKNNA(k)")):
            y = np.stack([series(r, key) for r in rs])
            ax.fill_between(ks, y.min(0), y.max(0), color=METRIC_COLOR[key], alpha=0.15, lw=0)
            ax.plot(ks, y.mean(0), color=METRIC_COLOR[key], label=label)
        cka = np.mean([r["cka"] for r in rs])
        ax.axhline(cka, color=METRIC_COLOR["cka"], ls="--", lw=1.5, label="linear CKA")
        ax.plot(ks, ks / ks[-1], color=INK_MUTED, ls=":", lw=1, label="mutual kNN chance k/(n-1)")
        ax.set_title(short(lvm), color=INK)
        _xaxis(ax)
    axes[0].set_ylabel("score (max over layer pairs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1)
    )
    fig.suptitle(f"Local to global: mean over {n_llms} LLMs (band = min..max)", y=1.17, color=INK)
    save(fig, out, "fig1_convergence")


def fig2_families(recs: list[dict], out: Path) -> None:
    lvms = sorted({r["lvm"] for r in recs}, key=lambda m: (family(m), size(m)))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    seen = set()
    for lvm in lvms:
        rs = [r for r in recs if r["lvm"] == lvm]
        f = family(lvm)
        ax.plot(
            series(rs[0], "k"),
            np.mean([series(r, "cknna") for r in rs], axis=0),
            color=SERIES[f],
            lw=1.2,
            label=None if f in seen else FAMILIES[f][0],
        )
        seen.add(f)
    _xaxis(ax, rotate=False)
    ax.set_ylabel("CKNNA (max over layer pairs)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"CKNNA vs k for {len(lvms)} ViTs, mean over LLMs", color=INK)
    save(fig, out, "fig2_families")


def fig3_vision_vision(path: Path, out: Path) -> None:
    d = np.load(path)
    ks, lvms = d["ks"], [str(m) for m in d["lvms"]]
    mae = np.array([".mae" in m for m in lvms])
    iu = np.triu_indices(len(lvms), 1)
    groups = {
        "MAE x MAE": mae[iu[0]] & mae[iu[1]],
        "MAE x other": mae[iu[0]] ^ mae[iu[1]],
        "other x other": ~mae[iu[0]] & ~mae[iu[1]],
    }
    groups = {name: sel for name, sel in groups.items() if sel.any()}
    fig, axes = plt.subplots(1, len(groups), figsize=(2.8 * len(groups), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, sel) in zip(axes, groups.items(), strict=True):
        for key, label in (("mknn", "mutual kNN(k)"), ("cknna", "CKNNA(k)")):
            y = d[key][:, iu[0], iu[1]][:, sel]
            color = METRIC_COLOR["mutual_knn" if key == "mknn" else key]
            ax.fill_between(ks, y.min(1), y.max(1), color=color, alpha=0.15, lw=0)
            ax.plot(ks, y.mean(1), color=color, label=label)
        cka = d["cka"][iu][sel].mean()
        ax.axhline(cka, color=METRIC_COLOR["cka"], ls="--", lw=1.5, label="linear CKA")
        ax.set_title(f"{name} ({int(sel.sum())} pairs)", color=INK)
        _xaxis(ax)
    axes[0].set_ylabel("score, last-block CLS")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1)
    )
    fig.suptitle("Vision-vision (PRH Fig. 12 setting)", y=1.17, color=INK)
    save(fig, out, "fig3_vision_vision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "figures"
    path = args.root / "grid" / "sweep_pairs.jsonl"
    if path.exists():
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        fig1_convergence(recs, out)
        fig2_families(recs, out)
    vv = args.root / "vision_vision" / "vision_vision.npz"
    if vv.exists():
        fig3_vision_vision(vv, out)


if __name__ == "__main__":
    main()
