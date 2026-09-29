"""Figures and summary table for exp-007 from sweep.py output.

    uv run python experiments/exp-007-cknna-variants/plot.py --root results/exp-007-cknna-variants

Reads <root>/grid and <root>/vision_vision; writes PNG + PDF to <root>/figures:

* fig1_variants       cross-modal: every CKNNA variant, mutual kNN and linear CKA vs k, mean over
                      all (LLM, ViT) pairs of the max over layer pairs (band = min..max)
* fig2_vision_vision  last-block ViT x ViT (PRH Fig. 12 setting), MAE x MAE vs other pairs
* fig3_by_llm_k<k>    Groger et al. Fig. 19 layout: alignment vs LLM (grouped by family, ordered by
                      size), one panel per vision family, one line per ViT size; rows are the four
                      CKNNA definitions; raw (dotted) and null-corrected (solid). Needs exp-008's
                      calibrated_pairs.jsonl (--calibrated). Null-corrected is Groger's
                      g = (raw - tau) / (1 - tau) for the definitions bounded by 1 and the
                      unscaled excess max(raw - tau, 0) (their s_max = None case) for `code` and
                      `eq36`, whose null exceeds 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
BY_LLM_KS = "10,200,500,1024"
VISION_FAMILIES = [
    ("ImageNet21K", lambda n: "augreg" in n),
    ("MAE", lambda n: ".mae" in n),
    ("DINOv2", lambda n: "dinov2" in n),
    ("CLIP", lambda n: "clip" in n and "ft_in12k" not in n),
    ("CLIP (I12K ft)", lambda n: "ft_in12k" in n),
]
VIT_SIZES = ["tiny", "small", "base", "large", "huge", "giant"]
LLM_FAMILIES = [
    ("BLOOMZ", "bigscience/bloomz-"),
    ("OpenLLaMA", "openlm-research/open_llama_"),
    ("LLaMA", "huggyllama/llama-"),
]
# ordinal blue ramp (reference palette steps 250..700), small ViT light -> large ViT dark
SIZE_RAMP = {
    1: ["#1c5cab"],
    2: ["#6da7ec", "#104281"],
    3: ["#86b6ef", "#2a78d6", "#0d366b"],
    4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
}
CKNNA_ROWS = ("paper", "centred", "code", "eq36")
CORR_LABEL = {"paper": "g", "centred": "g", "code": "raw - tau", "eq36": "raw - tau"}

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


def llm_params(name: str) -> float:
    """Parameter count in billions from the model name (bloomz-560m, -1b1, open_llama_13b)."""
    tag = next(name[len(pre) :] for _, pre in LLM_FAMILIES if name.startswith(pre))
    if tag.endswith("m"):
        return float(tag[:-1]) / 1000
    whole, _, frac = tag.partition("b")
    return float(f"{whole}.{frac or 0}")


def llm_order(name: str) -> tuple[int, float]:
    fam = next(i for i, (_, pre) in enumerate(LLM_FAMILIES) if name.startswith(pre))
    return fam, llm_params(name)


def vit_size(lvm: str) -> int:
    return next(i for i, s in enumerate(VIT_SIZES) if f"_{s}_" in lvm)


def vision_family(lvm: str) -> int:
    return next(i for i, (_, match) in enumerate(VISION_FAMILIES) if match(lvm))


def null_corrected(res: dict, name: str) -> float:
    """Groger's g for the bounded definitions, max(raw - tau, 0) for code / eq36."""
    if name in ("code", "eq36"):
        return max(res["raw_score"] - res["tau_alpha"], 0.0)
    return res["g_score"]


def fig3_by_llm(recs: list[dict], k: int, out: Path) -> None:
    """Groger Fig. 19 layout at one k: rows = CKNNA definitions, columns = vision families."""
    t = next(i for i, p in enumerate(recs[0]["per_k"]) if p["k"] == k)
    llms = sorted({r["llm"] for r in recs}, key=llm_order)
    # x positions: LLM families as groups separated by a gap
    x, groups, pos = {}, [], 0.0
    for fam, pre in LLM_FAMILIES:
        members = [m for m in llms if m.startswith(pre)]
        if members:
            groups.append((fam, members))
            x.update({m: pos + i for i, m in enumerate(members)})
            pos += len(members) + 0.8
    lookup = {(r["llm"], r["lvm"]): r["per_k"][t] for r in recs}
    lvms = sorted({r["lvm"] for r in recs}, key=lambda m: (vision_family(m), vit_size(m)))
    fams = [f for f in range(len(VISION_FAMILIES)) if any(vision_family(m) == f for m in lvms)]

    fig, axes = plt.subplots(
        len(CKNNA_ROWS),
        len(fams),
        figsize=(2.3 * len(fams), 2.0 * len(CKNNA_ROWS)),
        sharey="row",
        squeeze=False,
    )
    for r, name in enumerate(CKNNA_ROWS):
        for c, f in enumerate(fams):
            ax = axes[r, c]
            members = [m for m in lvms if vision_family(m) == f]
            for lvm, color in zip(members, SIZE_RAMP[len(members)], strict=True):
                label = VIT_SIZES[vit_size(lvm)]
                for _, fam_llms in groups:
                    sel = [m for m in fam_llms if (m, lvm) in lookup]
                    xx = [x[m] for m in sel]
                    raw = [lookup[(m, lvm)][name]["raw_score"] for m in sel]
                    corr = [null_corrected(lookup[(m, lvm)][name], name) for m in sel]
                    ax.plot(xx, raw, "d:", color=color, lw=1.2, ms=3.5)
                    ax.plot(xx, corr, "o-", color=color, lw=1.5, ms=3.5, label=label)
                    label = None
            if r == 0:
                ax.set_title(VISION_FAMILIES[f][0], color=INK)
                ax.legend(frameon=False, fontsize=6, handlelength=1.2)
            if c == 0:
                short = METRICS[name][0].replace("CKNNA ", "")
                ax.set_ylabel(f"{short}\nnull-corr.: {CORR_LABEL[name]}", fontsize=8)
            ax.set_xticks([x[m] for m in llms])
            ax.grid(axis="x", visible=False)
            if r < len(CKNNA_ROWS) - 1:
                ax.set_xticklabels([])
                continue
            ax.set_xticklabels([f"{llm_params(m):g}B" for m in llms], rotation=90, fontsize=6)
            for fam, fam_llms in groups:
                ax.annotate(
                    fam,
                    xy=((x[fam_llms[0]] + x[fam_llms[-1]]) / 2, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -32),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=INK_MUTED,
                )
    style = [
        Line2D([], [], color=INK_MUTED, ls="-", marker="o", ms=3.5, label="null-corrected"),
        Line2D([], [], color=INK_MUTED, ls=":", marker="d", ms=3.5, label="raw"),
    ]
    fig.legend(handles=style, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    kk = "n - 1 = 1023" if t == len(recs[0]["per_k"]) - 1 else str(k)
    fig.suptitle(
        f"CKNNA alignment vs LLM at k = {kk}, by vision family (max over layer pairs, "
        f"{len(recs)} pairs)",
        y=1.025,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, out, f"fig3_by_llm_k{k}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--calibrated",
        type=Path,
        default=None,
        help="exp-008 calibrated_pairs.jsonl (default: <root>/../exp-008-cknna-null/)",
    )
    parser.add_argument("--by-llm-ks", default=BY_LLM_KS, help="k values for fig3_by_llm")
    args = parser.parse_args()
    out = args.root / "figures"
    if (args.root / "grid" / "sweep_pairs.jsonl").exists():
        recs = load_grid(args.root)
        fig1_variants(recs, out)
        table(recs)
    vv = args.root / "vision_vision" / "vision_vision.npz"
    if vv.exists():
        fig2_vision_vision(vv, out)
    cal = args.calibrated or args.root.parent / "exp-008-cknna-null" / "calibrated_pairs.jsonl"
    if cal.exists():
        recs8 = [json.loads(line) for line in cal.read_text().splitlines()]
        n_1 = recs8[0]["per_k"][-1]["k"]
        for k in args.by_llm_ks.split(","):
            fig3_by_llm(recs8, min(int(k), n_1), out)


if __name__ == "__main__":
    main()
