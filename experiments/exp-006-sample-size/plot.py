"""Figures and summary for exp-006 from calibrate.py output.

    uv run python experiments/exp-006-sample-size/plot.py --root results/exp-006-sample-size \\
        --prh results/prh_val_calibrated/calibrated_pairs.jsonl

* fig1_variability   raw and calibrated score of every pair on 5 disjoint 1024-sample sets
                     (dots), with the PRH wit_1024 set (exp-003) as a reference marker
* fig2_sample_size   raw score, null mean and calibrated score vs n (1024 -> 10240), mean over
                     pairs (band = min..max); the 1024 point shows the 5 disjoint sets' spread
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order
METRICS = [("mutual_knn_k10", "mutual kNN (k=10)"), ("cka_lin", "linear CKA")]
PRH_KEY = {"mutual_knn_k10": "mutual_knn", "cka_lin": "cka_lin"}
FAMILY = [("augreg", "IN21k"), (".mae", "MAE"), ("dinov2", "DINOv2"), ("clip", "CLIP")]

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


def short(llm: str, lvm: str) -> str:
    fam = next(name for key, name in FAMILY if key in lvm)
    size = next(s for s in ("tiny", "small", "base") if f"_{s}_" in lvm)
    return f"{llm.split('-')[-1]} x {fam}-{size}"


def load(root: Path, prh: Path | None) -> tuple[dict, list, dict]:
    recs = [json.loads(line) for line in (root / "calibrated.jsonl").open()]
    by = {(r["llm"], r["lvm"], r["set"]): r for r in recs}
    pairs = list(dict.fromkeys((r["llm"], r["lvm"]) for r in recs))
    ref = {}
    if prh and prh.exists():
        for r in map(json.loads, prh.open()):
            ref[(r["llm"], r["lvm"])] = r
    return by, pairs, ref


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}.png")


def fig1_variability(by, pairs, ref, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    x = np.arange(len(pairs))
    for col, (key, label) in enumerate(METRICS):
        for row, (q, qlabel) in enumerate((("raw_score", "raw"), ("g_score", "calibrated g"))):
            ax = axes[row, col]
            for s in range(5):
                y = [by[(*p, f"disjoint{s}")]["results"][key][q] for p in pairs]
                ax.scatter(
                    x,
                    y,
                    s=18,
                    color=SERIES[0],
                    alpha=0.7,
                    lw=0,
                    label="5 disjoint WIT-1M sets (n=1024)" if s == 0 else None,
                )
            if ref:
                y = [ref[p][PRH_KEY[key]][q] if p in ref else np.nan for p in pairs]
                ax.scatter(
                    x,
                    y,
                    s=40,
                    marker="D",
                    facecolor="none",
                    edgecolor=SERIES[1],
                    lw=1.5,
                    label="PRH wit_1024 (exp-003)",
                )
            ax.set_title(f"{label}: {qlabel}", color=INK)
            ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xticks(x, [short(*p) for p in pairs], rotation=60, ha="right")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Variability across disjoint 1024-sample sets", color=INK)
    fig.tight_layout()
    save(fig, out, "fig1_variability")


def fig2_sample_size(by, pairs, out: Path) -> None:
    ns = [1024, 2048, 4096, 10240]
    sets = ["disjoint0", "first2048", "first4096", "first10240"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (key, label) in zip(axes, METRICS, strict=True):
        for c, (q, qlabel) in enumerate(
            (("raw_score", "raw"), ("mu0", "null mean"), ("g_score", "calibrated g"))
        ):
            ys = np.array(
                [[by[(*p, s)]["results"][key].get(q, np.nan) for s in sets] for p in pairs]
            )
            mean = np.nanmean(ys, 0) if np.isfinite(ys).any() else ys[0]
            ax.fill_between(ns, np.nanmin(ys, 0), np.nanmax(ys, 0), color=SERIES[c], alpha=0.15)
            ax.plot(ns, mean, "o-", color=SERIES[c], ms=5, label=qlabel)
            # spread over the 5 disjoint 1024 sets (mean over pairs of each set)
            d = np.array(
                [
                    np.mean(
                        [by[(*p, f"disjoint{s}")]["results"][key].get(q, np.nan) for p in pairs]
                    )
                    for s in range(5)
                ]
            )
            ax.errorbar(
                [1024],
                [d.mean()],
                yerr=[[d.mean() - d.min()], [d.max() - d.mean()]],
                color=SERIES[c],
                capsize=4,
                lw=1.2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("number of samples n")
        ax.set_title(label, color=INK)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("score (max over layer pairs)")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Alignment vs sample size, mean over {len(pairs)} pairs (band = min..max; "
        "CKA null not computed at n=10240)",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "fig2_sample_size")


def summary(by, pairs, ref) -> None:
    print("\nper metric, mean over pairs")
    for key, label in METRICS:
        for q in ("raw_score", "mu0", "g_score"):
            dis = np.array(
                [
                    [by[(*p, f"disjoint{s}")]["results"][key].get(q, np.nan) for s in range(5)]
                    for p in pairs
                ]
            )
            spread = (dis.max(1) - dis.min(1)).mean()
            cv = np.nanmean(dis.std(1) / np.abs(dis.mean(1)))
            nested = [
                np.nanmean([by[(*p, s)]["results"][key].get(q, np.nan) for p in pairs])
                for s in ("disjoint0", "first2048", "first4096", "first10240")
            ]
            prh = np.mean([ref[p][PRH_KEY[key]][q] for p in pairs if p in ref]) if ref else np.nan
            print(
                f"{label:<18} {q:<9} disjoint mean {dis.mean():.3f}  per-pair range "
                f"{spread:.3f}  CV {cv:.2f}  | PRH {prh:.3f} | n=1k/2k/4k/10k "
                + " ".join(f"{v:.3f}" for v in nested)
            )
        # does the pair ranking survive a change of sample set?
        raw = np.array(
            [
                [by[(*p, f"disjoint{s}")]["results"][key]["raw_score"] for s in range(5)]
                for p in pairs
            ]
        )

        rho = [spearmanr(raw[:, a], raw[:, b]).statistic for a in range(5) for b in range(a + 1, 5)]
        print(
            f"{label:<18} pair-ranking Spearman between disjoint sets: "
            f"mean {np.mean(rho):.2f}, min {np.min(rho):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prh", type=Path, default=None, help="exp-003 calibrated_pairs.jsonl")
    args = parser.parse_args()
    by, pairs, ref = load(args.root, args.prh)
    summary(by, pairs, ref)
    fig1_variability(by, pairs, ref, args.root / "figures")
    fig2_sample_size(by, pairs, args.root / "figures")


if __name__ == "__main__":
    main()
