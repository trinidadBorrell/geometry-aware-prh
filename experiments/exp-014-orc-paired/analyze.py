"""exp-014 step 3: paired per-sample ORC statistics for T1 (LLM x LLM) and T2 (ViT x ViT).

For every model pair, k and layer pair (i, j), across the 1024 shared samples:
- orc:       Spearman of node ORC in A layer i vs B layer j
- density:   Spearman of local density (the control)
- orc|dens:  Spearman of ORC after removing each layer's own density (rank residuals)
each calibrated cell by cell against a 200-permutation null of the pairing (`geoprh.paired`).
Per pair: the best cell, the fraction of cells with p < 0.05 (5% expected by chance), and the
Spearman across cells between the orc|dens map and the null-corrected mutual kNN map of exp-012.

    uv run python experiments/exp-014-orc-paired/analyze.py --root results/exp-014-orc-paired \\
        --exp012 results/exp-012-id-mismatch
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from geoprh import paired

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
STATS = {"orc": "node ORC", "density": "local density", "orc_dens": "ORC | density"}
STAT_COLOURS = {"orc": "#2a78d6", "density": "#eb6834", "orc_dens": "#1baf7a"}
TRACKS = {"T1": "LLM x LLM", "T2": "ViT x ViT"}
EXAMPLES = [
    ("T1", "bigscience/bloomz-7b1", "huggyllama/llama-13b"),
    ("T2", "vit_giant_patch14_dinov2.lvd142m", "vit_huge_patch14_clip_224.laion2b"),
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


def stat_inputs(npz, k: int) -> dict:
    orc, dens = npz[f"orc_k{k}"], npz[f"dens_k{k}"]
    return {
        "orc": paired.standardized_ranks(orc),
        "density": paired.standardized_ranks(dens),
        "orc_dens": paired.residualize(orc, dens),
    }


def save(fig, root: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exp012", type=Path, required=True)
    parser.add_argument("--ks", default="10,30")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    e12 = _exp012()
    ks = [int(k) for k in args.ks.split(",")]
    models = {p.stem: dict(np.load(p)) for p in (args.root / "models").glob("*.npz")}
    # Exact-duplicate captions (one pair in the PRH wit_1024 set) give near-zero LLM edge lengths
    # and kappa = 1 - W1/d ~ -1e5 for those samples; drop them from every model.
    n_samples = next(iter(models.values()))[f"orc_k{ks[0]}"].shape[1]
    bad = np.zeros(n_samples, bool)
    for z in models.values():
        for k in ks:
            bad |= (z[f"orc_k{k}"] < -5).any(axis=0)
    for z in models.values():
        for key in list(z):
            if key.startswith(("orc_", "dens_")):
                z[key] = z[key][:, ~bad]
    print(
        f"dropped {int(bad.sum())} samples with degenerate ORC (duplicates): {np.flatnonzero(bad).tolist()}"
    )
    pairs = [("T1", a, b) for a, b in itertools.combinations(e12.LLMS, 2)]
    pairs += [("T2", a, b) for a, b in itertools.combinations(e12.VITS, 2)]
    pairs = [p for p in pairs if slug(p[1]) in models and slug(p[2]) in models]
    (args.root / "pairs").mkdir(exist_ok=True)

    recs = []
    for i, (track, a, b) in enumerate(pairs):
        mk = args.exp012 / "pairs" / f"{track}__{slug(a)}__{slug(b)}.npz"
        mk = np.load(mk) if mk.exists() else None
        out = {}
        for k in ks:
            ia, ib = stat_inputs(models[slug(a)], k), stat_inputs(models[slug(b)], k)
            for s in STATS:
                res = paired.calibrated_rank_corr(ia[s], ib[s], args.permutations, seed=i)
                for key in ("score", "mu0", "tau", "p", "g"):
                    out[f"{s}_k{k}_{key}"] = res[key]
        np.savez(args.root / "pairs" / f"{track}__{slug(a)}__{slug(b)}.npz", **out)
        rec = {"track": track, "a": a, "b": b}
        for k in ks:
            for s in STATS:
                sc, p = out[f"{s}_k{k}_score"], out[f"{s}_k{k}_p"]
                bi = np.unravel_index(int(sc.argmax()), sc.shape)
                rec[f"{s}_k{k}"] = {
                    "max": float(sc.max()),
                    "best": [int(bi[0]), int(bi[1])],
                    "tau_at_best": float(out[f"{s}_k{k}_tau"][bi]),
                    "frac_sig": float((p < 0.05).mean()),
                    "max_g": float(out[f"{s}_k{k}_g"].max()),
                }
            if mk is not None and mk["mutual_knn_score"].shape == out[f"orc_dens_k{k}_score"].shape:
                m = (mk["mutual_knn_score"] - mk["mutual_knn_mu0"]).ravel()
                for s in STATS:
                    rec[f"{s}_k{k}"]["rho_vs_mknn_map"] = float(
                        spearmanr(out[f"{s}_k{k}_score"].ravel(), m).statistic
                    )
        recs.append(rec)
    with (args.root / "pairs.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    lines = [
        "median over pairs: best-cell Spearman [null tau at that cell] | frac cells p<0.05 | "
        "max calibrated g | Spearman of the layer-pair map with mutual kNN (exp-012)"
    ]
    for k in ks:
        for t in TRACKS:
            rs = [r for r in recs if r["track"] == t]
            if not rs:
                continue
            lines.append(f"{t} {TRACKS[t]}, k = {k}, {len(rs)} pairs")
            for s, label in STATS.items():

                def v(key, s=s, k=k, rs=rs):
                    return np.median([r[f"{s}_k{k}"].get(key, np.nan) for r in rs])

                lines.append(
                    f"  {label:<14} best {v('max'):+.3f} [tau {v('tau_at_best'):+.3f}]  "
                    f"frac_sig {v('frac_sig'):.2f}  max_g {v('max_g'):.3f}  "
                    f"vs mKNN map {v('rho_vs_mknn_map'):+.2f}"
                )
    (args.root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # fig1: best-cell correlation per pair, per stat and track, k = ks[0]
    k = ks[0]
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(9, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, t in zip(axes, TRACKS, strict=True):
        rs = [r for r in recs if r["track"] == t]
        for x, s in enumerate(STATS):
            vals = np.array([r[f"{s}_k{k}"]["max"] for r in rs])
            taus = np.array([r[f"{s}_k{k}"]["tau_at_best"] for r in rs])
            ax.scatter(
                x + rng.uniform(-0.15, 0.15, len(vals)),
                vals,
                s=8,
                alpha=0.5,
                color=STAT_COLOURS[s],
                lw=0,
            )
            ax.plot([x - 0.3, x + 0.3], [np.median(vals)] * 2, color=INK, lw=2)
            ax.plot([x - 0.3, x + 0.3], [np.median(taus)] * 2, color=INK_MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(STATS)), list(STATS.values()))
        ax.set_title(f"{t} {TRACKS[t]} ({len(rs)} pairs), k = {k}")
    axes[0].set_ylabel("best layer-pair Spearman across samples")
    fig.suptitle(
        "Do the same samples get high / low values in both models? (bar = median, "
        "dashed = null 95% threshold at the best cell)",
        fontsize=9,
    )
    save(fig, args.root, "fig1_paired")

    # fig2: example layer-pair maps
    ex = [
        e
        for e in EXAMPLES
        if (args.root / "pairs" / f"{e[0]}__{slug(e[1])}__{slug(e[2])}.npz").exists()
    ]
    if ex:
        fig, axes = plt.subplots(len(ex), 3, figsize=(11, 3.3 * len(ex)), squeeze=False)
        for row, (t, a, b) in enumerate(ex):
            arr = np.load(args.root / "pairs" / f"{t}__{slug(a)}__{slug(b)}.npz")
            mk = np.load(args.exp012 / "pairs" / f"{t}__{slug(a)}__{slug(b)}.npz")
            panels = [
                (arr[f"orc_k{k}_score"], f"node ORC, Spearman (k = {k})"),
                (arr[f"orc_dens_k{k}_score"], "ORC | density, Spearman"),
                (mk["mutual_knn_score"] - mk["mutual_knn_mu0"], "mutual kNN, S - null (exp-012)"),
            ]
            for ax, (m, title) in zip(axes[row], panels, strict=True):
                im = ax.imshow(m, cmap="Blues", aspect="auto", origin="lower")
                ax.grid(False)
                ax.set_title(title, fontsize=8)
                ax.set_xlabel(f"{b.split('/')[-1]} layer", fontsize=8)
                ax.set_ylabel(f"{a.split('/')[-1]} layer", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046)
        save(fig, args.root, "fig2_examples")

    # fig3: ORC level per layer vs the flat-torus reference at the layer's ID (descriptive)
    refs = json.loads((args.root / "torus_ref.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, group, title in ((axes[0], e12.LLMS, "LLMs"), (axes[1], e12.VITS, "ViTs")):
        for m in group:
            if slug(m) not in models:
                continue
            z = models[slug(m)]
            orc = np.median(z[f"orc_k{k}"], axis=1)
            ids = np.rint(z["twonn"]).astype(int)
            ref = np.array([refs.get(f"d{d}_k{k}", np.nan) for d in ids])
            depth = np.linspace(0, 1, len(orc))
            ax.plot(depth, orc - ref, color="#2a78d6", alpha=0.6, lw=1.2)
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.set_title(f"{title}: median node ORC minus flat-torus mean at the layer's ID (k = {k})")
        ax.set_xlabel("relative depth")
    axes[0].set_ylabel("ORC - reference")
    save(fig, args.root, "fig3_levels")


if __name__ == "__main__":
    main()
