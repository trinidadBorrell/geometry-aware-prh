"""exp-010 stage 2: compare layer ID profiles across models over relative depth.

    uv run python experiments/exp-010-intrinsic-dim/compare.py \\
        --root results/exp-010-intrinsic-dim/prh1024 --check results/exp-010-intrinsic-dim

Relative depth = layer / (L - 1). LLM layer 0 is the token embedding; ViT layer 0 is the output of
block 0 (the cache has no patch-embedding layer).

* fig2_profiles_<prep>   ID vs relative depth, one panel per family, one line per model size
                         (darker = larger)
* fig3_family_means      mean profile per family on a common depth grid (band = min..max over the
                         family's sizes); columns LLM | ViT, rows raw | prh
* fig4_estimators_<prep> with --other: family mean profiles of the two estimators
* summary_models.txt     per model: ID at depth 0, 0.5 and 1, peak depth, min..max; and, with
                         --check, Spearman of the model ordering at n=1000 vs n=10000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order
# blue 250..700: ordinal, small -> large model
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
# (name, test on the model name); order matters: CLIP-ft before CLIP
FAMILIES = [
    ("BLOOMZ", lambda m: "bloomz" in m),
    ("OpenLLaMA", lambda m: "open_llama" in m),
    ("LLaMA", lambda m: "huggyllama" in m),
    ("ViT IN21k", lambda m: "augreg_in21k" in m),
    ("MAE", lambda m: ".mae" in m),
    ("DINOv2", lambda m: "dinov2" in m),
    ("CLIP ft-IN12k", lambda m: "laion2b_ft" in m),
    ("CLIP", lambda m: "clip" in m),
]
VIT_SIZES = ["tiny", "small", "base", "large", "huge", "giant"]
GRID_DEPTH = np.linspace(0, 1, 21)
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
PREP_TITLE = {"raw": "raw activations", "prh": "PRH features"}

ESTIMATOR_LABEL = {"lb": "Levina-Bickel, k=10..20", "twonn": "TwoNN"}
YLABEL = [ESTIMATOR_LABEL["lb"]]  # set from the data in main()


def estimator_of(path: Path) -> str:
    with path.open() as f:
        return json.loads(f.readline()).get("estimator", "lb")


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


def family(model: str) -> str:
    return next(name for name, test in FAMILIES if test(model))


def size_label(model: str) -> str:
    if "/" in model:
        return model.split("-")[-1].split("_")[-1]
    return next(s for s in VIT_SIZES if f"_{s}_" in model)


def load(path: Path, n: int | None = None) -> dict:
    """{(model, prep): {"id": [sets, L] array, "dim": d, "modality": str}} at sample size n."""
    acc = defaultdict(dict)
    info = {}
    for line in path.open():
        r = json.loads(line)
        if n is not None and r["n"] != n:
            continue
        acc[(r["model"], r["prep"])][(r["set"], r["layer"])] = r["id"]
        info[(r["model"], r["prep"])] = (r["dim"], r["num_layers"], r["modality"])
    out = {}
    for key, d in acc.items():
        sets = sorted({s for s, _ in d})
        n_layers = info[key][1]
        ids = np.array([[d[(s, la)] for la in range(n_layers)] for s in sets])
        out[key] = {"id": ids, "dim": info[key][0], "modality": info[key][2]}
    return out


def profile(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """(relative depth, mean ID over sets)."""
    m = entry["id"].mean(0)
    return np.linspace(0, 1, len(m)), m


def by_family(data: dict, prep: str) -> dict[str, list[str]]:
    """{family: models sorted small -> large} for one geometry, in FAMILIES order."""
    fams = defaultdict(list)
    for model, p in data:
        if p == prep:
            fams[family(model)].append(model)
    order = [name for name, _ in FAMILIES]
    return {
        f: sorted(ms, key=lambda m: (data[(m, prep)]["dim"], data[(m, prep)]["id"].shape[1]))
        for f, ms in sorted(fams.items(), key=lambda kv: order.index(kv[0]))
    }


def fig_profiles(data: dict, prep: str, out: Path) -> None:
    fams = by_family(data, prep)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True, sharey=True)
    ymax = max(np.nanmax(data[(m, prep)]["id"]) for ms in fams.values() for m in ms)
    for ax, (fam, models) in zip(axes.flat, fams.items(), strict=False):
        ramp = RAMP[-len(models) :] if len(models) > 1 else RAMP[-2:-1]
        for c, model in zip(ramp, models, strict=True):
            x, y = profile(data[(model, prep)])
            ax.plot(x, y, color=c, label=f"{size_label(model)} ({len(y)} layers)")
        ax.set_title(fam, color=INK)
        ax.set_ylim(0, ymax * 1.05)
        ax.legend(frameon=False, fontsize=7, loc="best")
    for ax in axes.flat[len(fams) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("relative depth (layer / (L-1))")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"intrinsic dimension ({YLABEL[0]})")
    fig.suptitle(f"ID profiles, n = 1024 (PRH wit_1024) - {PREP_TITLE[prep]}", color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_profiles_{prep}.{ext}", dpi=160)
    plt.close(fig)


def fig_family_means(data: dict, out: Path) -> None:
    preps = [p for p in ("raw", "prh") if any(k[1] == p for k in data)]
    fig, axes = plt.subplots(len(preps), 2, figsize=(10.5, 3.4 * len(preps)), squeeze=False)
    for i, prep in enumerate(preps):
        fams = by_family(data, prep)
        for j, modality in enumerate(("language", "vision")):
            ax = axes[i, j]
            k = 0
            for fam, models in fams.items():
                if data[(models[0], prep)]["modality"] != modality:
                    continue
                ys = np.array([np.interp(GRID_DEPTH, *profile(data[(m, prep)])) for m in models])
                c = SERIES[k]
                k += 1
                ax.fill_between(GRID_DEPTH, ys.min(0), ys.max(0), color=c, alpha=0.18, lw=0)
                ax.plot(GRID_DEPTH, ys.mean(0), color=c, label=f"{fam} ({len(models)} models)")
            ax.set_title(f"{'LLMs' if modality == 'language' else 'ViTs'} - {PREP_TITLE[prep]}")
            ax.set_xlabel("relative depth")
            ax.set_ylabel("intrinsic dimension")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3_family_means.{ext}", dpi=160)
    plt.close(fig)


def fig_estimators(data: dict, other: dict, other_label: str, out: Path) -> None:
    """Family mean profile of this estimator vs another, one panel per family, both geometries."""
    for prep in ("raw", "prh"):
        fams = by_family(data, prep)
        if not fams:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True, sharey=True)
        for ax, (fam, models) in zip(axes.flat, fams.items(), strict=False):
            for c, src, label in ((SERIES[0], data, YLABEL[0]), (SERIES[1], other, other_label)):
                ys = np.array([np.interp(GRID_DEPTH, *profile(src[(m, prep)])) for m in models])
                ax.fill_between(GRID_DEPTH, ys.min(0), ys.max(0), color=c, alpha=0.18, lw=0)
                ax.plot(GRID_DEPTH, ys.mean(0), color=c, label=label)
            ax.set_title(f"{fam} ({len(models)} models)", color=INK)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=7, loc="best")
        for ax in axes.flat[len(fams) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("relative depth")
        for ax in axes[:, 0]:
            ax.set_ylabel("intrinsic dimension")
        fig.suptitle(
            f"Estimators compared, n = 1024 - {PREP_TITLE[prep]} (line = family mean, "
            "band = min..max over sizes)",
            color=INK,
        )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig4_estimators_{prep}.{ext}", dpi=160)
        plt.close(fig)


def at_depth(entry: dict, depth: float) -> float:
    return float(np.interp(depth, *profile(entry)))


def summary(data: dict, check: Path | None) -> str:
    lines = []
    for prep in ("raw", "prh"):
        fams = by_family(data, prep)
        if not fams:
            continue
        lines.append(f"[{prep}] n = 1024")
        lines.append(
            f"  {'model':<46} {'L':>3} {'dim':>5}  {'d=0':>6} {'d=0.5':>6} {'d=1':>6}  "
            f"{'peak d':>6}  {'min..max':>13}"
        )
        for fam, models in fams.items():
            for m in models:
                x, y = profile(data[(m, prep)])
                lines.append(
                    f"  {m:<46} {len(y):>3} {data[(m, prep)]['dim']:>5}  "
                    f"{y[0]:6.1f} {at_depth(data[(m, prep)], 0.5):6.1f} {y[-1]:6.1f}  "
                    f"{x[y.argmax()]:6.2f}  {y.min():5.1f}..{y.max():5.1f}   {fam}"
                )
        lines.append("")
    if check:
        small, large = (
            load(check / "intrinsic_dim.jsonl", 1000),
            load(check / "intrinsic_dim.jsonl", 10000),
        )
        lines.append("Model ordering at n=1000 (mean of 10 sets) vs n=10000, wit1m10k models:")
        for prep in ("raw", "prh"):
            models = sorted(m for m, p in small if p == prep and (m, prep) in large)
            if len(models) < 3:
                continue
            rhos = []
            for depth in DEPTHS:
                a = [at_depth(small[(m, prep)], depth) for m in models]
                b = [at_depth(large[(m, prep)], depth) for m in models]
                rhos.append(f"d={depth:.2f}: {spearmanr(a, b).statistic:+.2f}")
            lines.append(f"  [{prep}] {len(models)} models  " + "  ".join(rhos))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="stage-2 output dir")
    parser.add_argument("--check", type=Path, default=None, help="wit1m10k output dir (n check)")
    parser.add_argument(
        "--other", type=Path, default=None, help="stage-2 dir of another estimator (fig4)"
    )
    args = parser.parse_args()
    YLABEL[0] = ESTIMATOR_LABEL[estimator_of(args.root / "intrinsic_dim.jsonl")]
    data = load(args.root / "intrinsic_dim.jsonl")
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    for prep in ("raw", "prh"):
        if any(k[1] == prep for k in data):
            fig_profiles(data, prep, out)
    fig_family_means(data, out)
    if args.other:
        other = args.other / "intrinsic_dim.jsonl"
        fig_estimators(data, load(other), ESTIMATOR_LABEL[estimator_of(other)], out)
    text = summary(data, args.check)
    (args.root / "summary_models.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
