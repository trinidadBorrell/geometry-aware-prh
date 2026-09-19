"""Reproduce PRH (Huh et al. 2024) Fig. 3, 10, 12, 13 and 14 from geoprh.prh_alignment output.

Example:
    uv run python experiments/prh_val/plot_figures.py --root results/prh_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
# ordinal blue ramp (reference palette steps), light -> dark = small -> large
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
MARKERS = ["o", "s", "^", "D"]

# (title, matcher, size order) -- PRH Fig. 3/13 column order
FAMILIES = [
    ("ImageNet21K", lambda n: "augreg" in n, ["tiny", "small", "base", "large"]),
    ("MAE", lambda n: ".mae" in n, ["base", "large", "huge"]),
    ("DINOv2", lambda n: "dinov2" in n, ["small", "base", "large", "giant"]),
    ("CLIP", lambda n: "clip" in n and "ft_in12k" not in n, ["base", "large", "huge"]),
    ("CLIP (I12K ft)", lambda n: "ft_in12k" in n, ["base", "large", "huge"]),
]
LLM_SHORT = {
    "bigscience/bloomz-560m": "bloom0.56b",
    "bigscience/bloomz-1b1": "bloom1.1b",
    "bigscience/bloomz-1b7": "bloom1.7b",
    "bigscience/bloomz-3b": "bloom3b",
    "bigscience/bloomz-7b1": "bloom7b",
    "openlm-research/open_llama_3b": "openllama3b",
    "openlm-research/open_llama_7b": "openllama7b",
    "openlm-research/open_llama_13b": "openllama13b",
    "huggyllama/llama-7b": "llama7b",
    "huggyllama/llama-13b": "llama13b",
    "huggyllama/llama-30b": "llama33b",
    "huggyllama/llama-65b": "llama65b",
}
CROSS_PANELS = {
    "fig13_cross_modal_metrics": [
        ("cka", "(a) CKA"),
        ("unbiased_cka", "(b) Unbiased CKA"),
        ("svcca", "(c) SVCCA"),
        ("mutual_knn", "(d) Mutual k-NN (k = 10)"),
    ],
    "fig14_cross_modal_metrics": [
        ("cknna_k10", "(a) CKNNA (k = 10)"),
        ("cycle_knn", "(b) Cycle k-NN (k = 10)"),
        ("edit_distance_knn", "(c) Edit-distance k-NN (k = 10)"),
        ("lcs_knn", "(d) LCS k-NN (k = 10), normalised by k"),
    ],
}
CKNNA_KS = (1000, 800, 500, 200, 100, 50, 20, 10)


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
            "font.size": 9,
        }
    )


def _families(lvms: list[str]) -> list[tuple[str, list[tuple[int, str]]]]:
    """[(family title, [(column index, size label), ...small->large]), ...]"""
    out = []
    for title, match, sizes in FAMILIES:
        members = [(j, s) for s in sizes for j, n in enumerate(lvms) if match(n) and s in n]
        if members:
            out.append((title, members))
    return out


def _scores(npz, metric: str) -> np.ndarray:
    s = npz[f"{metric}__scores"].copy()
    return s / 10 if metric == "lcs_knn" else s  # upstream returns LCS length; paper plots /k


def fig3(npz, meta, out: Path) -> None:
    llms, lvms = meta["llms"], meta["lvms"]
    x = np.array([1 - meta["models"][m]["bpb"] for m in llms])
    order = np.argsort(x)
    scores = _scores(npz, "mutual_knn")
    fams = _families(lvms)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.1 * len(fams), 3.0))
    for ax, (title, members) in zip(axes, fams, strict=True):
        for c, (j, size) in enumerate(members):
            ax.plot(
                x[order],
                scores[order, j],
                color=ORDINAL[c],
                marker=MARKERS[c],
                markersize=5,
                linewidth=1.8,
                markeredgecolor=SURFACE,
                label=size,
            )
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("1 - bits-per-byte (WIT captions*)")
    axes[0].set_ylabel("alignment (mutual k-NN, k = 10)")
    fig.suptitle(
        "PRH Fig. 3 - language and vision models align  (*paper: bpb on OpenWebText)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "fig3_language_vision_alignment.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig10(npz, meta, out: Path) -> None:
    llms, lvms = meta["llms"], meta["lvms"]
    ppl = np.array([np.exp(meta["models"][m]["loss"]) for m in llms])
    order = np.argsort(-ppl)  # worse (higher perplexity) language models first
    smallest = int(np.argmin([meta["models"][m]["num_params"] for m in llms]))
    ramp = LinearSegmentedColormap.from_list("ord", SEQUENTIAL[1:])(
        np.linspace(0, 1, len(CKNNA_KS))
    )
    fams = _families(lvms)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.1 * len(fams), 3.0), sharey=True)
    for ax, (title, members) in zip(axes, fams, strict=True):
        cols = [j for j, _ in members]
        for c, k in enumerate(CKNNA_KS):
            a = npz[f"cknna_k{k}__scores"][:, cols].mean(axis=1)
            y = (a - a[smallest]) / (a.std() + 1e-12)
            ax.plot(ppl[order], y[order], color=ramp[c], linewidth=1.8, label=f"K={k}")
        ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("perplexity (WIT captions*, log)")
    axes[0].set_ylabel("alignment trend (CKNNA, centred, / std)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("PRH Fig. 10 - cross-modal alignment increases locally", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig10_cknna_k_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_cross(npz, meta, out: Path, name: str, panels) -> None:
    llms, lvms = meta["llms"], meta["lvms"]
    fams = _families(lvms)
    fig, axes = plt.subplots(len(panels), len(fams), figsize=(3.1 * len(fams), 2.6 * len(panels)))
    xs = np.arange(len(llms))
    for r, (metric, label) in enumerate(panels):
        scores = _scores(npz, metric)
        for c_ax, (title, members) in enumerate(fams):
            ax = axes[r, c_ax]
            for c, (j, size) in enumerate(members):
                ax.plot(
                    xs,
                    scores[:, j],
                    color=ORDINAL[c],
                    marker=MARKERS[c],
                    markersize=4,
                    linewidth=1.6,
                    markeredgecolor=SURFACE,
                    label=size,
                )
            ax.set_title(f"Alignment to {title}", loc="left", fontsize=9)
            ax.set_xticks(xs, [LLM_SHORT.get(m, m) for m in llms], rotation=90, fontsize=7)
            if c_ax == 0:
                ax.set_ylabel(label)
            if r == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"PRH {name.split('_')[0].replace('fig', 'Fig. ')} - cross-modal alignment "
        "for various metrics",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig12(root: Path, out: Path) -> None:
    data = np.load(root / "vision_vision.npz")["scores"]
    meta = json.loads((root / "vision_vision.json").read_text())
    rho = spearmanr(data, axis=1).statistic
    labels = meta["metric_labels"]
    fig, ax = plt.subplots(figsize=(11, 10))
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    ax.imshow(rho, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{rho[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=SURFACE if rho[i, j] > 0.6 else INK,
            )
    ax.set_title(
        f"PRH Fig. 12 - Spearman correlation between metrics over "
        f"{len(meta['pairs'])} vision-vision pairs ({len(meta['lvms'])} ViTs, "
        f"{meta['layer']})",
        loc="left",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "fig12_vision_vision_metric_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("results/prh_val"))
    args = parser.parse_args()
    out = args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    _style()
    if (args.root / "cross_modal.npz").exists():
        npz = np.load(args.root / "cross_modal.npz")
        meta = json.loads((args.root / "cross_modal.json").read_text())
        fig3(npz, meta, out)
        if "cknna_k1000__scores" in npz:
            fig10(npz, meta, out)
        for name, panels in CROSS_PANELS.items():
            if all(f"{m}__scores" in npz for m, _ in panels):
                fig_cross(npz, meta, out, name, panels)
    if (args.root / "vision_vision.npz").exists():
        fig12(args.root, out)
    print(f"saved figures to {out}")


if __name__ == "__main__":
    main()
