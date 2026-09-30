"""Figures and summary tables for exp-009 from run.py output.

    uv run python experiments/exp-009-local-cknna/plot.py --root results/exp-009-local-cknna

Reads <root>/pairs.jsonl and <root>/pairs/*.npz; writes PNG + PDF to <root>/figures:

* fig1_variants       left: every metric vs k (mean over pairs of the max over layer pairs,
                      band = min..max); right: the same minus its permutation-null mean at the
                      selected layer pair (excess over chance)
* fig2_coverage       fraction of points kept by the mutual local CKA (>= 4 mutual neighbours)
                      vs k, observed and under the null
* fig3_distributions  per-point scores at the best layer pair, pooled over all pairs, for the four
                      local scores (mKNN, centred CKNNA row, local mutual, local union) at several k
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "plot7", Path(__file__).parents[1] / "exp-007-cknna-variants" / "plot.py"
)
p7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p7)
INK, INK_MUTED = p7.INK, p7.INK_MUTED

# categorical slots in fixed order (reference palette, eight hues)
METRICS = {
    "mknn": ("mutual kNN", "#2a78d6"),
    "paper": ("CKNNA paper (row-centred)", "#eb6834"),
    "centred": ("CKNNA centred (HKH)", "#1baf7a"),
    "code": ("CKNNA code (platonic-rep)", "#eda100"),
    "eq36": ("CKNNA Groger Eq. 36", "#e87ba4"),
    "local_mutual": ("local CKA, mutual nbrs (Emily)", "#008300"),
    "local_union": ("local CKA, union nbrs", "#4a3aa7"),
}
POINT_METRICS = ("mknn", "centred", "local_mutual", "local_union")


def load(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / "pairs.jsonl").read_text().splitlines()]


def curves(recs: list[dict], field: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ks = np.array([p["k"] for p in recs[0]["per_k"]])
    return ks, {m: np.array([[p[m][field] for p in r["per_k"]] for r in recs]) for m in METRICS}


def band(ax, ks, y, color, label, ls="-"):
    ax.fill_between(ks, np.nanmin(y, 0), np.nanmax(y, 0), color=color, alpha=0.1, lw=0)
    ax.plot(ks, np.nanmean(y, 0), color=color, ls=ls, label=label)


def fig1_variants(recs: list[dict], out: Path) -> None:
    ks, raw = curves(recs, "raw")
    _, null = curves(recs, "null_mean")
    cka = np.array([r["cka"]["raw"] for r in recs])
    cka_null = np.array([r["cka"]["null_mean"] for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=False)
    for ax, (title, data, ref) in zip(
        axes,
        (
            ("observed", raw, cka),
            (
                "observed minus null mean (chance at the selected layer pair)",
                {m: raw[m] - null[m] for m in METRICS},
                cka - cka_null,
            ),
        ),
        strict=True,
    ):
        ax.axhline(1.0, color=INK_MUTED, lw=0.8)
        for name, (label, color) in METRICS.items():
            band(ax, ks, data[name], color, label)
        ax.axhspan(ref.min(), ref.max(), color=INK, alpha=0.07, lw=0)
        ax.axhline(ref.mean(), color=INK, ls="--", lw=1.5, label="linear CKA")
        ax.set_title(title, color=INK)
        p7.xaxis(ax)
    axes[0].set_ylabel("score (max over layer pairs)")
    p7.legend(fig, axes[0], ncol=4)
    fig.suptitle(
        f"Local CKA vs the CKNNA definitions, {len(recs)} (LLM, ViT) pairs (band = min..max)",
        y=1.2,
        color=INK,
    )
    p7.save(fig, out, "fig1_variants")


def fig2_coverage(recs: list[dict], root: Path, out: Path) -> None:
    ks = np.array([p["k"] for p in recs[0]["per_k"]])
    obs = np.array([[p["coverage"] for p in r["per_k"]] for r in recs])
    nulls = []
    for r in recs:
        stem = f"{r['llm'].replace('/', '__')}__x__{r['lvm']}"
        d = np.load(root / "pairs" / f"{stem}.npz")
        nulls.append([d[f"null_coverage_k{k}"].mean() for k in ks])
    nulls = np.array(nulls)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    color = METRICS["local_mutual"][1]
    band(ax, ks, obs, color, "observed (best layer pair)")
    band(ax, ks, nulls, color, "permutation null", ls=":")
    ax.set_ylabel("fraction of points kept")
    ax.set_title("Coverage of the mutual local CKA (>= 4 mutual neighbours)", color=INK)
    ax.legend(frameon=False, loc="lower right")
    p7.xaxis(ax)
    p7.save(fig, out, "fig2_coverage")


def fig3_distributions(recs: list[dict], root: Path, out: Path, dist_ks: list[int]) -> None:
    pooled = {(m, k): [] for m in POINT_METRICS for k in dist_ks}
    for r in recs:
        stem = f"{r['llm'].replace('/', '__')}__x__{r['lvm']}"
        d = np.load(root / "pairs" / f"{stem}.npz")
        for m in POINT_METRICS:
            for k in dist_ks:
                v = d[f"points_{m}_k{k}"]
                pooled[(m, k)].append(v[np.isfinite(v)])
    fig, axes = plt.subplots(
        1, len(POINT_METRICS), figsize=(3.0 * len(POINT_METRICS), 3.2), sharey=False
    )
    for ax, m in zip(axes, POINT_METRICS, strict=True):
        label, color = METRICS[m]
        data = [np.concatenate(pooled[(m, k)]) for k in dist_ks]
        keep = [i for i, v in enumerate(data) if v.size]
        parts = ax.violinplot([data[i] for i in keep], positions=keep, showmedians=True, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(color)
        ax.set_xticks(range(len(dist_ks)))
        ax.set_xticklabels([str(k) for k in dist_ks])
        ax.set_xlabel("k")
        ax.set_title(label, color=INK, pad=14 if m == "local_mutual" else 6)
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        if m == "local_mutual":
            for i in keep:
                ax.annotate(
                    f"{len(data[i]) / (len(recs) * recs[0]['n']):.0%}",
                    xy=(i, 1.0),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK_MUTED,
                )
    axes[0].set_ylabel("per-point score")
    fig.suptitle(
        "Per-point scores at the best layer pair, pooled over all pairs "
        "(local mutual: % of points kept above each violin)",
        y=1.05,
        color=INK,
    )
    fig.tight_layout()
    p7.save(fig, out, "fig3_distributions")


def table(recs: list[dict]) -> None:
    ks, raw = curves(recs, "raw")
    _, null = curves(recs, "null_mean")
    cols = [i for i, k in enumerate(ks) if k in p7.TABLE_KS]
    cov = np.array([[p["coverage"] for p in r["per_k"]] for r in recs]).mean(0)
    for title, vals in (("raw", raw), ("null mean", null)):
        print(f"\n{title}")
        print("| k | " + " | ".join(str(ks[i]) for i in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for name, (label, _) in METRICS.items():
            row = np.nanmean(vals[name], 0)
            print(f"| {label} | " + " | ".join(f"{row[i]:.2f}" for i in cols) + " |")
    print("| coverage (mutual) | " + " | ".join(f"{cov[i]:.2f}" for i in cols) + " |")
    cka = np.array([r["cka"]["raw"] for r in recs])
    print(
        f"\nlinear CKA raw {cka.mean():.3f}, null {np.mean([r['cka']['null_mean'] for r in recs]):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recs = load(args.root)
    meta = (
        json.loads((args.root / "meta.json").read_text())
        if (args.root / "meta.json").exists()
        else {}
    )
    dist_ks = [min(k, recs[0]["n"] - 1) for k in meta.get("dist_ks", [10, 50, 200, 500, 1000])]
    ks = {p["k"] for p in recs[0]["per_k"]}
    dist_ks = [k for k in dist_ks if k in ks]
    out = args.root / "figures"
    fig1_variants(recs, out)
    fig2_coverage(recs, args.root, out)
    fig3_distributions(recs, args.root, out, dist_ks)
    table(recs)


if __name__ == "__main__":
    main()
