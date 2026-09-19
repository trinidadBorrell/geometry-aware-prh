"""Plot small-model PRH alignment: raw (platonic-rep) vs null-calibrated (Aristotelian).

Example:
    uv run python experiments/prh_small/plot.py --root results/prh_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aristotelian.prh.prh_models import get_models

PAIR_DIR = "minhuh/prh/{modelset}/language_pool-avg_prompt-False_vision_pool-cls_prompt-False"
# (label, platonic-rep file, Aristotelian file)
METRICS = [
    ("mutual kNN (k=10)", "mutual_knn_k10.npy", "mutual_knn_k10.npy"),
    ("linear CKA", "cka.npy", "cka_lin.npy"),
]

# Reference categorical palette, fixed order (slots 1-4), plus markers as a second cue.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"

VIT_LABELS = {
    "vit_tiny_patch16_224.augreg_in21k": "ViT-Ti in21k",
    "vit_base_patch16_224.augreg_in21k": "ViT-B in21k",
    "vit_small_patch14_dinov2.lvd142m": "DINOv2-S",
    "vit_base_patch16_clip_224.laion2b": "CLIP-B",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def _direct_labels(ax, x: float, ys: np.ndarray, lvms: list[str], min_gap: float = 0.06) -> None:
    """Label line ends, nudging labels apart so they keep `min_gap` (axes fraction)."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    order = np.argsort(ys)
    placed = ys[order].astype(float)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    for rank, j in enumerate(order):
        ax.annotate(
            VIT_LABELS.get(lvms[j], lvms[j]),
            xy=(x, ys[j]),
            xytext=(x * 1.06, placed[rank]),
            textcoords="data",
            va="center",
            fontsize=9,
            color=INK_MUTED,
            annotation_clip=False,
        )


def _llm_params(features_dir: Path, llms: list[str]) -> np.ndarray:
    params = []
    for name in llms:
        payload = torch.load(
            features_dir / f"{name.replace('/', '_')}_pool-avg.pt", map_location="cpu"
        )
        params.append(payload["num_params"])
    return np.array(params, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="results/prh_small")
    parser.add_argument("--modelset", default="small")
    args = parser.parse_args()

    root = Path(args.root)
    pair_dir = PAIR_DIR.format(modelset=args.modelset)
    llms, lvms = get_models(args.modelset)
    params = _llm_params(root / "platonic/features/minhuh/prh/wit_1024", llms)
    out_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    results = {}
    for label, p_file, a_file in METRICS:
        plat = np.load(root / "platonic/alignment" / pair_dir / p_file, allow_pickle=True).item()
        arist = np.load(
            root / "aristotelian/alignment" / pair_dir / a_file, allow_pickle=True
        ).item()
        results[label] = (plat["scores"], arist["gated"])

    # Figure 1: score vs LLM size, raw vs calibrated, shared y within each metric row.
    scores_max = [max(raw.max(), cal.max()) for raw, cal in results.values()]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    for row, (label, (raw, cal)) in enumerate(results.items()):
        for col, (title, scores) in enumerate(
            [("raw (platonic-rep)", raw), ("null-calibrated (Aristotelian)", cal)]
        ):
            ax = axes[row, col]
            for j, vit in enumerate(lvms):
                ax.plot(
                    params / 1e9,
                    scores[:, j],
                    color=SERIES_COLORS[j],
                    marker=MARKERS[j],
                    markersize=8,
                    linewidth=2,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    label=VIT_LABELS.get(vit, vit),
                )
            ax.set_xscale("log")
            ax.set_xticks(params / 1e9, [f"{p / 1e9:.2g}B" for p in params])
            ax.minorticks_off()
            ax.set_ylim(bottom=0, top=scores_max[row] * 1.08)
            if col == 1:
                _direct_labels(ax, params[-1] / 1e9, scores[-1], lvms)
            ax.set_title(f"{label}: {title}", fontsize=10, loc="left")
            if col == 0:
                ax.set_ylabel("max alignment over layer pairs")
            if row == 1:
                ax.set_xlabel("language model size (bloomz, params)")
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(
        "Language-vision alignment on WIT-1024, small models", x=0.06, ha="left", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_vs_llm_size.png", dpi=160, bbox_inches="tight")

    # Figure 2: how much calibration removes (raw - calibrated), per metric.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    for ax, (label, (raw, cal)) in zip(axes, results.items(), strict=True):
        drop = raw - cal
        im = ax.imshow(drop, cmap=cmap, vmin=0, aspect="auto")
        ax.set_xticks(
            range(len(lvms)), [VIT_LABELS.get(v, v) for v in lvms], rotation=20, ha="right"
        )
        ax.set_yticks(range(len(llms)), [m.split("/")[-1] for m in llms])
        ax.grid(False)
        for i in range(drop.shape[0]):
            for j in range(drop.shape[1]):
                dark = drop[i, j] > 0.6 * drop.max()
                ax.text(
                    j,
                    i,
                    f"{drop[i, j]:.3f}\n({drop[i, j] / raw[i, j]:.0%})",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=SURFACE if dark else INK,
                )
        ax.set_title(f"{label}: raw - calibrated", fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_drop.png", dpi=160, bbox_inches="tight")
    print(f"saved plots to {out_dir}")


if __name__ == "__main__":
    main()
