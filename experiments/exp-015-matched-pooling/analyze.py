"""exp-015 analysis: intrinsic dimension and cross-modal alignment under matched pooling.

Inputs (all PRH wit_1024, n = 1024, PRH features):
- per-layer ORC / density / TwoNN ID: exp-014 `models/` (LLM avg, ViT cls) and exp-015 `orc/models/`
  (LLM last, ViT avg)
- calibrated mutual kNN (k = 10) and linear CKA per layer pair: exp-012 `pairs/` (LLM avg x ViT
  cls), exp-015 `align_token/pairs/` (LLM last x ViT cls, and T1 on last), `align_mean/pairs/`
  (LLM avg x ViT avg, and T2 on avg)

Pairings: mixed = LLM avg x ViT cls (the PRH default), token = LLM last x ViT cls, mean = LLM avg
x ViT avg. Outputs in --out: summary.txt, fig1_id (TwoNN ID vs depth per pooling), fig2_alignment
(per ViT x LLM pair, best-layer calibrated mutual kNN and CKA under each pairing), fig3_orc
(per pair, best-layer ORC-given-density agreement under each pairing, with T1 / T2 references).

    uv run python experiments/exp-015-matched-pooling/analyze.py --e14 results/exp-014-orc-paired \\
        --e12 results/exp-012-id-mismatch --e15 results/exp-015-matched-pooling
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoprh import paired

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
COND_COLOURS = {"mixed": "#52514e", "token": "#2a78d6", "mean": "#eb6834"}
COND_LABEL = {
    "mixed": "mixed: LLM mean x ViT CLS (PRH)",
    "token": "token: LLM last x ViT CLS",
    "mean": "mean: LLM mean x ViT patch mean",
}
DROP = [493, 531]

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


def _exp012():
    path = Path(__file__).resolve().parents[1] / "exp-012-id-mismatch" / "run.py"
    spec = importlib.util.spec_from_file_location("exp012_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def slug(m: str) -> str:
    return m.replace("/", "__")


def load_models(d: Path) -> dict:
    return {p.stem: dict(np.load(p)) for p in d.glob("*.npz")}


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def fill_degenerate(orc: np.ndarray, dens: np.ndarray) -> tuple:
    """Per layer: undefined ORC (points whose neighbours all coincide with them, e.g. LLM last
    token at layer 0, where captions ending in the same token are identical) -> layer median;
    infinite density (zero distance to the k-th neighbour) -> layer maximum. These samples become
    ties in the rank statistics."""
    orc, dens = orc.copy(), dens.copy()
    for i in range(orc.shape[0]):
        bad = ~np.isfinite(orc[i])
        if bad.any():
            orc[i, bad] = np.nanmedian(np.where(bad, np.nan, orc[i]))
        inf = ~np.isfinite(dens[i])
        if inf.any():
            dens[i, inf] = dens[i, ~inf].max() if (~inf).any() else 1.0
    return orc, dens


def orc_inputs(z: dict, k: int, keep: np.ndarray) -> tuple:
    orc, dens = fill_degenerate(z[f"orc_k{k}"][:, keep], z[f"dens_k{k}"][:, keep])
    return paired.residualize(orc, dens), paired.standardized_ranks(dens)


def best_align(f: Path, metric: str) -> dict | None:
    if not f.exists():
        return None
    z = np.load(f)
    return {
        "max_score": float(z[f"{metric}_score"].max()),
        "max_excess": float((z[f"{metric}_score"] - z[f"{metric}_mu0"]).max()),
        "max_g": float(z[f"{metric}_g"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--e14", type=Path, required=True)
    parser.add_argument("--e12", type=Path, required=True)
    parser.add_argument("--e15", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    e12 = _exp012()
    out = args.e15
    old, new = load_models(args.e14 / "models"), load_models(args.e15 / "orc" / "models")
    llm = {"avg": {m: old[slug(m)] for m in e12.LLMS}, "last": {m: new[slug(m)] for m in e12.LLMS}}
    vit = {"cls": {m: old[slug(m)] for m in e12.VITS}, "avg": {m: new[slug(m)] for m in e12.VITS}}
    lines = []

    # 1. TwoNN ID per pooling, family means at relative depth 0 / 0.5 / 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, group, title in ((axes[0], llm, "LLMs"), (axes[1], vit, "ViTs")):
        for pool, colour in zip(group, ("#2a78d6", "#eb6834"), strict=True):
            vals = {0.0: [], 0.5: [], 1.0: []}
            for i, z in enumerate(group[pool].values()):
                t = z["twonn"]
                depth = np.linspace(0, 1, len(t))
                ax.plot(depth, t, color=colour, alpha=0.5, lw=1.2, label=pool if i == 0 else None)
                for d in vals:
                    vals[d].append(float(np.interp(d, depth, t)))
            lines.append(
                f"TwoNN ID, {title} pool={pool}: depth 0 / 0.5 / 1 = "
                + " / ".join(f"{np.mean(v):.1f}" for v in vals.values())
            )
        ax.set_title(f"{title}: TwoNN ID per layer (n = 1024, PRH features)")
        ax.set_xlabel("relative depth")
        ax.legend(frameon=False)
    axes[0].set_ylabel("TwoNN ID")
    save(fig, out, "fig1_id")

    # 2. cross-modal mutual kNN / CKA under the three pairings
    pair_dirs = {
        "mixed": args.e12 / "pairs",
        "token": args.e15 / "align_token" / "pairs",
        "mean": args.e15 / "align_mean" / "pairs",
    }
    t3 = [(v, m) for v in e12.VITS for m in e12.LLMS]
    align = {c: {} for c in pair_dirs}
    for c, d in pair_dirs.items():
        for v, m in t3:
            for metric in ("mutual_knn", "cka_lin"):
                r = best_align(d / f"T3__{slug(v)}__{slug(m)}.npz", metric)
                if r:
                    align[c].setdefault(metric, []).append(r)
    for metric in ("mutual_knn", "cka_lin"):
        for c in pair_dirs:
            rs = align[c].get(metric, [])
            if rs:
                lines.append(
                    f"T3 {metric:<10} {c:<6} ({len(rs)} pairs): median best raw "
                    f"{np.median([r['max_score'] for r in rs]):.3f}, best S - null "
                    f"{np.median([r['max_excess'] for r in rs]):.3f}, max g "
                    f"{np.median([r['max_g'] for r in rs]):.3f}"
                )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, title in (
        (axes[0], "mutual_knn", "mutual kNN (k = 10)"),
        (axes[1], "cka_lin", "linear CKA"),
    ):
        for x, c in enumerate(pair_dirs):
            vals = [r["max_excess"] for r in align[c].get(metric, [])]
            if vals:
                ax.scatter(
                    x + np.random.default_rng(x).uniform(-0.15, 0.15, len(vals)),
                    vals,
                    s=8,
                    alpha=0.5,
                    color=COND_COLOURS[c],
                    lw=0,
                )
                ax.plot([x - 0.3, x + 0.3], [np.median(vals)] * 2, color=INK, lw=2)
        ax.set_xticks(range(len(pair_dirs)), [COND_LABEL[c].split(":")[0] for c in pair_dirs])
        ax.set_title(f"ViT x LLM, best layer pair: {title}, S - null mean")
    fig.suptitle("; ".join(COND_LABEL.values()), fontsize=8)
    save(fig, out, "fig2_alignment")

    # 3. per-sample ORC agreement (given density) under the three pairings, with references
    n = next(iter(llm["avg"].values()))[f"orc_k{args.k}"].shape[1]
    bad = np.zeros(n, bool)
    for group in (*llm.values(), *vit.values()):
        for z in group.values():
            bad |= (z[f"orc_k{args.k}"] < -5).any(axis=0)
    bad[DROP] = True
    for pool, group in (("LLM last", llm["last"]), ("ViT avg", vit["avg"])):
        counts = {
            m.split("/")[-1]: int((~np.isfinite(z[f"orc_k{args.k}"])).sum(axis=1).max())
            for m, z in group.items()
        }
        counts = {m: c for m, c in counts.items() if c}
        if counts:
            lines.append(
                f"{pool}: max samples per layer with undefined ORC (identical points): {counts}"
            )
    keep = np.flatnonzero(~bad)
    lines.append(
        f"ORC: dropped {int(bad.sum())} samples (duplicates / degenerate): {np.flatnonzero(bad).tolist()}"
    )
    conds = {
        "mixed": [(vit["cls"][v], llm["avg"][m]) for v, m in t3],
        "token": [(vit["cls"][v], llm["last"][m]) for v, m in t3],
        "mean": [(vit["avg"][v], llm["avg"][m]) for v, m in t3],
        "T1 avg": [(llm["avg"][a], llm["avg"][b]) for a, b in itertools.combinations(e12.LLMS, 2)],
        "T1 last": [
            (llm["last"][a], llm["last"][b]) for a, b in itertools.combinations(e12.LLMS, 2)
        ],
        "T2 cls": [(vit["cls"][a], vit["cls"][b]) for a, b in itertools.combinations(e12.VITS, 2)],
        "T2 avg": [(vit["avg"][a], vit["avg"][b]) for a, b in itertools.combinations(e12.VITS, 2)],
    }
    orc_res = {}
    for c, prs in conds.items():
        rs = []
        for i, (za, zb) in enumerate(prs):
            ra, da = orc_inputs(za, args.k, keep)
            rb, db = orc_inputs(zb, args.k, keep)
            r = paired.calibrated_rank_corr(ra, rb, args.permutations, seed=i)
            rd = paired.calibrated_rank_corr(da, db, args.permutations, seed=i)
            bi = np.unravel_index(int(r["score"].argmax()), r["score"].shape)
            rs.append(
                {
                    "orc_dens": float(r["score"].max()),
                    "tau": float(r["tau"][bi]),
                    "frac_sig": float((r["p"] < 0.05).mean()),
                    "density": float(rd["score"].max()),
                }
            )
        orc_res[c] = rs
        lines.append(
            f"ORC|density, best layer pair, {c:<8} ({len(rs)} pairs): median {np.median([r['orc_dens'] for r in rs]):+.3f} "
            f"[tau {np.median([r['tau'] for r in rs]):.3f}], frac cells p<0.05 {np.median([r['frac_sig'] for r in rs]):.2f}; "
            f"density {np.median([r['density'] for r in rs]):+.3f}"
        )
    (out / "orc_conditions.json").write_text(json.dumps(orc_res))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for x, c in enumerate(conds):
        vals = [r["orc_dens"] for r in orc_res[c]]
        colour = COND_COLOURS.get(c, INK_MUTED)
        ax.scatter(
            x + np.random.default_rng(x).uniform(-0.15, 0.15, len(vals)),
            vals,
            s=8,
            alpha=0.5,
            color=colour,
            lw=0,
        )
        ax.plot([x - 0.3, x + 0.3], [np.median(vals)] * 2, color=INK, lw=2)
    ax.axhline(
        np.median([r["tau"] for rs in orc_res.values() for r in rs]), color=INK_MUTED, ls="--", lw=1
    )
    ax.set_xticks(range(len(conds)), ["ViT x LLM\n" + c if c in COND_LABEL else c for c in conds])
    ax.set_ylabel("best layer-pair Spearman, ORC | density")
    ax.set_title(f"Per-sample ORC agreement (k = {args.k}); dashed = null 95% threshold")
    save(fig, out, "fig3_orc")

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
