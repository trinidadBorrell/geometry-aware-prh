"""Figure and summary for exp-010 from run.py output.

    uv run python experiments/exp-010-intrinsic-dim/plot.py --root results/exp-010-intrinsic-dim

* fig1_id_by_layer   Levina-Bickel ID per layer, one line per n (mean over the disjoint sample
                     sets, band = min..max); one row per model, columns raw | prh geometry
* summary.txt        per model, geometry and n: peak layer, ID range over layers, typical
                     spread across sets, Spearman of the layer profile against the largest n
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
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # blue 250/400/550/700, small -> large n
PREP_TITLE = {"raw": "raw activations", "prh": "PRH features (q=0.95 clamp, l2 norm)"}

ESTIMATOR_LABEL = {
    "lb": "Levina-Bickel, k=10..20",
    "twonn": "TwoNN",
    "pca": "PCA: PCs for 90% variance",
}
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


def load(root: Path) -> dict:
    """{(model, prep, n): array [sets, layers]}."""
    acc = defaultdict(dict)
    for line in (root / "intrinsic_dim.jsonl").open():
        r = json.loads(line)
        acc[(r["model"], r["prep"], r["n"])][(r["set"], r["layer"])] = r["id"]
    out = {}
    for key, d in acc.items():
        sets = sorted({s for s, _ in d})
        n_layers = max(layer for _, layer in d) + 1
        out[key] = np.array([[d[(s, la)] for la in range(n_layers)] for s in sets])
    return out


def fig_by_layer(data: dict, out: Path) -> None:
    models = list(dict.fromkeys(m for m, _, _ in data))
    preps = [p for p in ("raw", "prh") if any(k[1] == p for k in data)]
    sizes = sorted({n for _, _, n in data})
    fig, axes = plt.subplots(
        len(models), len(preps), figsize=(5.2 * len(preps), 3.4 * len(models)), squeeze=False
    )
    for i, model in enumerate(models):
        for j, prep in enumerate(preps):
            ax = axes[i, j]
            for c, n in zip(ORDINAL[-len(sizes) :], sizes, strict=True):
                a = data.get((model, prep, n))
                if a is None:
                    continue
                x = np.arange(a.shape[1])
                ax.fill_between(x, a.min(0), a.max(0), color=c, alpha=0.25, lw=0)
                ax.plot(x, a.mean(0), color=c, label=f"n = {n:,} ({a.shape[0]} sets)")
                ax.annotate(
                    f"{n // 1000}k",
                    (x[-1], a.mean(0)[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    color=INK_MUTED,
                    fontsize=8,
                )
            ax.set_title(f"{model} - {PREP_TITLE[prep]}", color=INK)
            ax.set_xlabel("layer (0 = embeddings)" if "/" in model else "block")
            ax.set_ylabel(f"intrinsic dimension ({YLABEL[0]})")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_id_by_layer.{ext}", dpi=160)
    plt.close(fig)


def summary(data: dict) -> str:
    lines = []
    for model, prep in dict.fromkeys((m, p) for m, p, _ in data):
        sizes = sorted(n for m, p, n in data if (m, p) == (model, prep))
        ref = data[(model, prep, sizes[-1])].mean(0)
        lines.append(f"{model} [{prep}]")
        lines.append(
            "  n       sets  peak layer  ID min..max over layers  ID at peak  "
            "mean spread across sets  Spearman vs n=" + f"{sizes[-1]}"
        )
        for n in sizes:
            a = data[(model, prep, n)]
            m = a.mean(0)
            spread = (a.max(0) - a.min(0)).mean() if a.shape[0] > 1 else float("nan")
            rho = spearmanr(m, ref).statistic
            lines.append(
                f"  {n:<7} {a.shape[0]:<5} {int(m.argmax()):<11} "
                f"{m.min():6.1f} .. {m.max():<14.1f} {m.max():<11.1f} {spread:<24.2f} {rho:.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    YLABEL[0] = ESTIMATOR_LABEL[estimator_of(args.root / "intrinsic_dim.jsonl")]
    data = load(args.root)
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    fig_by_layer(data, out)
    text = summary(data)
    (args.root / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
