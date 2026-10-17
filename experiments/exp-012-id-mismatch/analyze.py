"""exp-012 analysis: does a mismatch in intrinsic dimension predict misalignment?

    uv run python experiments/exp-012-id-mismatch/analyze.py --root <exp-012 dir> \\
        --id-root <exp-010 dir>

For every model pair and metric, over all layer cells (i, j):
- alignment = null-corrected score S - mu0 (the permutation-null mean of that cell);
- ID mismatch = |log ID_a(i) - log ID_b(j)|, with TwoNN and with Levina-Bickel (exp-010, PRH
  features, n = 1024); and a linearity mismatch |log(PCA90/ID)_a(i) - log(PCA90/ID)_b(j)|
  (PCA90 = PCs for 90% of the variance, ID = TwoNN);
- Spearman rho(alignment, mismatch), and the partial Spearman controlling for depth: ranks of
  both regressed on the relative depths r_a = i/(La-1), r_b = j/(Lb-1), |r_a - r_b| and r_a r_b,
  then the correlation of the residuals;
- best-cell percentile: the fraction of cells with a smaller mismatch than the best-aligned
  cell (small = alignment peaks where IDs match).

Outputs in --root: analysis.jsonl (one line per pair x metric), summary.txt, and figures
fig1_partial_rho, fig2_examples, fig3_best_cell (.png, .pdf).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata, spearmanr

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
POINT = "#2a78d6"  # categorical slot 1
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
TRACKS = {"T1": "LLM x LLM", "T2": "ViT x ViT", "T3": "ViT x LLM"}
METRICS = {"mutual_knn": "mutual kNN (k=10)", "cka_lin": "linear CKA"}
MISMATCH = {
    "id_twonn": "|log ID ratio|, TwoNN",
    "id_lb": "|log ID ratio|, Levina-Bickel",
    "linearity": "|log (PCA90/ID) ratio|",
}
EXAMPLES = [
    ("T1", "bigscience/bloomz-7b1", "huggyllama/llama-13b"),
    ("T2", "vit_giant_patch14_dinov2.lvd142m", "vit_huge_patch14_clip_224.laion2b"),
    ("T3", "vit_giant_patch14_dinov2.lvd142m", "huggyllama/llama-13b"),
]

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
    }
)


def load_ids(id_root: Path) -> dict:
    """{(estimator, model): per-layer array} from exp-010 (PRH features, n = 1024, set 0)."""
    files = {
        "lb": id_root / "prh1024" / "intrinsic_dim.jsonl",
        "twonn": id_root / "twonn" / "prh1024" / "intrinsic_dim.jsonl",
        "pca": id_root / "pca" / "prh1024" / "intrinsic_dim.jsonl",
    }
    rows = defaultdict(dict)
    for est, f in files.items():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["prep"] == "prh" and r["n"] == 1024 and r["set"] == 0:
                rows[(est, r["model"])][r["layer"]] = r["id"]
    return {k: np.array([v[i] for i in sorted(v)]) for k, v in rows.items()}


def mismatches(ids: dict, a: str, b: str) -> dict:
    out = {}
    for name, est in (("id_twonn", "twonn"), ("id_lb", "lb")):
        la, lb = np.log(ids[(est, a)]), np.log(ids[(est, b)])
        out[name] = np.abs(la[:, None] - lb[None, :])
    lin_a = np.log(ids[("pca", a)] / ids[("twonn", a)])
    lin_b = np.log(ids[("pca", b)] / ids[("twonn", b)])
    out["linearity"] = np.abs(lin_a[:, None] - lin_b[None, :])
    return out


def partial_spearman(y: np.ndarray, x: np.ndarray, covariates: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), covariates])

    def resid(v):
        r = rankdata(v)
        return r - design @ np.linalg.lstsq(design, r, rcond=None)[0]

    return float(np.corrcoef(resid(y), resid(x))[0, 1])


def analyse_pair(arrays, ids, track, a, b) -> list[dict]:
    mm = mismatches(ids, a, b)
    la, lb = next(iter(mm.values())).shape
    ra, rb = np.meshgrid(np.arange(la) / (la - 1), np.arange(lb) / (lb - 1), indexing="ij")
    cov = np.column_stack([ra.ravel(), rb.ravel(), np.abs(ra - rb).ravel(), (ra * rb).ravel()])
    out = []
    for metric in METRICS:
        score = arrays[f"{metric}_score"]
        if score.shape != (la, lb):
            raise ValueError(f"{a} x {b}: layers {score.shape} vs IDs {(la, lb)}")
        align = (score - arrays[f"{metric}_mu0"]).ravel()
        best = int(align.argmax())
        rec = {"track": track, "a": a, "b": b, "metric": metric}
        for name, m in mm.items():
            m = m.ravel()
            rec[name] = {
                "rho": float(spearmanr(align, m).statistic),
                "partial_rho": partial_spearman(align, m, cov),
                "best_percentile": float((m < m[best]).mean()),
            }
        out.append(rec)
    return out


def strip(ax, groups: dict, ylabel: str, rng):
    for x, vals in enumerate(groups.values()):
        vals = np.asarray(vals)
        ax.scatter(
            x + rng.uniform(-0.18, 0.18, len(vals)), vals, s=10, color=POINT, alpha=0.45, lw=0
        )
        med = np.median(vals)
        ax.plot([x - 0.3, x + 0.3], [med, med], color=INK, lw=2)
        ax.annotate(f"{med:+.2f}", (x + 0.32, med), va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)), [f"{t}\n{TRACKS[t]}" for t in groups])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(groups) - 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--id-root", type=Path, required=True)
    args = parser.parse_args()
    ids = load_ids(args.id_root)
    pairs = [json.loads(line) for line in (args.root / "pairs.jsonl").read_text().splitlines()]

    recs = []
    for p in pairs:
        f = (
            args.root
            / "pairs"
            / f"{p['track']}__{p['a'].replace('/', '__')}__{p['b'].replace('/', '__')}.npz"
        )
        recs += analyse_pair(np.load(f), ids, p["track"], p["a"], p["b"])
    with (args.root / "analysis.jsonl").open("w") as fout:
        for r in recs:
            fout.write(json.dumps(r) + "\n")

    tracks = [t for t in TRACKS if any(r["track"] == t for r in recs)]
    lines = [
        "track  metric      mismatch   pairs  rho_med  partial_med [q25, q75]  "
        "frac_partial<0  best_pct_med"
    ]
    for metric in METRICS:
        for name in MISMATCH:
            for t in tracks:
                rs = [r for r in recs if r["track"] == t and r["metric"] == metric]
                rho = np.array([r[name]["rho"] for r in rs])
                pr = np.array([r[name]["partial_rho"] for r in rs])
                bp = np.array([r[name]["best_percentile"] for r in rs])
                lines.append(
                    f"{t:<6} {metric:<11} {name:<10} {len(rs):>5}  {np.median(rho):+.3f}   "
                    f"{np.median(pr):+.3f} [{np.quantile(pr, 0.25):+.3f}, "
                    f"{np.quantile(pr, 0.75):+.3f}]   {np.mean(pr < 0):.2f}           "
                    f"{np.median(bp):.2f}"
                )
    (args.root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 2, figsize=(8, 9), sharey=True)
    for row, name in enumerate(MISMATCH):
        for col, metric in enumerate(METRICS):
            groups = {
                t: [
                    r[name]["partial_rho"]
                    for r in recs
                    if r["track"] == t and r["metric"] == metric
                ]
                for t in tracks
            }
            ax = axes[row, col]
            ax.axhline(0, color=INK_MUTED, lw=0.8)
            strip(ax, groups, f"partial Spearman vs\n{MISMATCH[name]}" if col == 0 else "", rng)
            if row == 0:
                ax.set_title(METRICS[metric])
    fig.suptitle(
        "Null-corrected alignment vs ID mismatch across layer pairs, depth partialled out\n"
        "(one point per model pair; bar = median; negative = alignment is lower where IDs differ)",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.root / f"fig1_partial_rho.{ext}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        groups = {
            t: [
                r["id_twonn"]["best_percentile"]
                for r in recs
                if r["track"] == t and r["metric"] == metric
            ]
            for t in tracks
        }
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls="--")
        strip(
            ax,
            groups,
            "percentile of the best cell's\nID gap among all cells" if ax is axes[0] else "",
            rng,
        )
        ax.set_title(METRICS[metric])
        ax.set_ylim(-0.03, 1.03)
    fig.suptitle(
        "Where alignment peaks, relative to the ID gap (TwoNN); 0.5 = no relation, 0 = smallest gap",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.root / f"fig3_best_cell.{ext}", dpi=200)
    plt.close(fig)

    examples = [e for e in EXAMPLES if e[0] in tracks]
    if examples:
        fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3.2 * len(examples)), squeeze=False)
        for row, (t, a, b) in enumerate(examples):
            f = args.root / "pairs" / f"{t}__{a.replace('/', '__')}__{b.replace('/', '__')}.npz"
            if not f.exists():
                continue
            arr = np.load(f)
            panels = [
                (arr["mutual_knn_score"] - arr["mutual_knn_mu0"], "mutual kNN, S - null mean"),
                (arr["cka_lin_score"] - arr["cka_lin_mu0"], "linear CKA, S - null mean"),
                (mismatches(ids, a, b)["id_twonn"], "|log ID ratio| (TwoNN)"),
            ]
            for ax, (m, title) in zip(axes[row], panels, strict=True):
                im = ax.imshow(m, cmap=SEQ, aspect="auto", origin="lower")
                ax.grid(False)
                ax.set_title(f"{title}", fontsize=8)
                ax.set_xlabel(f"{b.split('/')[-1]} layer", fontsize=8)
                ax.set_ylabel(f"{a.split('/')[-1]} layer", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046)
            axes[row, 0].annotate(
                f"{t} {TRACKS[t]}", (0, 1.18), xycoords="axes fraction", fontsize=9, color=INK
            )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(args.root / f"fig2_examples.{ext}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
