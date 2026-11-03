"""exp-014 / exp-015: distributions of node ORC per layer, against a flat-torus reference.

Reference: for each layer, a uniformly sampled flat torus T^d (Clifford, (S^1)^d in R^{2d}) with
the same n and k as the data and d = the layer's TwoNN ID (rounded). It answers "what node ORC
would this layer show if its points were uniformly spread on a flat, boundary-free manifold of
the same dimension": ORC's level depends strongly on n, k and the ID (exp-011), and the torus
holds those fixed while having zero curvature, uniform density, no clusters and no boundary
(a cube's boundary raises ORC, exp-011 correction). Deviations from it therefore mix curvature,
density heterogeneity and clustering; they are not curvature alone.

The reference distribution is computed here in full (`torus_dist_k{k}.json`: quantiles, median,
mean), so data and reference are compared distribution to distribution (median to median), not
median to mean as in the first exp-014 level figure.

Figures (in --out):
- fig4_orc_distributions[_<pooling>]: per model family, each model's per-layer node-ORC median
  (line) and 10-90% band, with the reference median and 10-90% band at each layer's ID (grey)
- fig3_levels_median: per model, median node ORC minus the reference median, per layer

    uv run python experiments/exp-014-orc-paired/distributions.py --out results/exp-014-orc-paired \\
        --models results/exp-014-orc-paired/models --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
DROP = [493, 531]  # exact-duplicate caption pair in wit_1024

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
        "lines.linewidth": 1.5,
    }
)


def _exp012():
    path = Path(__file__).resolve().parents[1] / "exp-012-id-mismatch" / "run.py"
    spec = importlib.util.spec_from_file_location("exp012_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family(m: str) -> str:
    if "/" in m:
        return {"bigscience": "BLOOMZ", "openlm-research": "OpenLLaMA", "huggyllama": "LLaMA"}[
            m.split("/")[0]
        ]
    for key, name in (
        ("ft_in12k", "CLIP-ft"),
        ("laion2b", "CLIP"),
        ("dinov2", "DINOv2"),
        ("mae", "MAE"),
        ("augreg", "IN21k"),
    ):
        if key in m:
            return name
    return m


def torus_task(args):
    d, k, n = args
    from threadpoolctl import threadpool_limits

    from geoprh import geometry as geo

    threadpool_limits(1)
    rng = np.random.default_rng(d)
    t = rng.uniform(0, 2 * np.pi, size=(n, d))
    x = np.concatenate([np.cos(t), np.sin(t)], axis=1)
    node = geo.ollivier_ricci_exact(geo.knn_graph(x, k))["node"]
    return d, {
        "quantiles": dict(zip(map(str, QS), np.quantile(node, QS).tolist(), strict=True)),
        "median": float(np.median(node)),
        "mean": float(node.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="dirs of model npz")
    parser.add_argument("--labels", nargs="+", default=None, help="pooling label per --models dir")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    e12 = _exp012()
    labels = args.labels or [""] * len(args.models)
    sets = {
        lab: {p.stem: dict(np.load(p)) for p in d.glob("*.npz")}
        for lab, d in zip(labels, args.models, strict=True)
    }
    n = next(iter(next(iter(sets.values())).values()))[f"orc_k{args.k}"].shape[1]

    ref_path = args.out / f"torus_dist_k{args.k}.json"
    refs = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    need = sorted(
        {int(d) for s in sets.values() for z in s.values() for d in np.rint(z["twonn"])}
        - {int(x) for x in refs}
    )
    if need:
        with ProcessPoolExecutor(args.workers) as pool:
            for d, r in pool.map(torus_task, [(d, args.k, n) for d in need]):
                refs[str(d)] = r
                print(f"torus d={d}: median {r['median']:+.3f} mean {r['mean']:+.3f}", flush=True)
        ref_path.write_text(json.dumps(refs, indent=1))
    keep = np.setdiff1d(np.arange(n), DROP)

    for lab, models in sets.items():
        fams = {}
        for m in e12.LLMS + e12.VITS:
            if m.replace("/", "__") in models:
                fams.setdefault(family(m), []).append(m)
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey=True)
        for ax, (fam, ms) in zip(axes.ravel(), fams.items(), strict=False):
            for m, c in zip(ms, CAT, strict=False):
                z = models[m.replace("/", "__")]
                orc = z[f"orc_k{args.k}"][:, keep]
                depth = np.linspace(0, 1, orc.shape[0])
                q = np.nanquantile(orc, [0.1, 0.5, 0.9], axis=1)
                ax.fill_between(depth, q[0], q[2], color=c, alpha=0.15, lw=0)
                ax.plot(depth, q[1], color=c, label=m.split("/")[-1].split(".")[0])
                ids = np.rint(z["twonn"]).astype(int)
                r = np.array(
                    [[refs[str(d)]["quantiles"][str(p)] for p in (0.1, 0.5, 0.9)] for d in ids]
                ).T
                ax.fill_between(depth, r[0], r[2], color=INK_MUTED, alpha=0.08, lw=0)
                ax.plot(depth, r[1], color=INK_MUTED, lw=1, ls="--")
            ax.set_title(f"{fam} ({lab})" if lab else fam)
            ax.legend(frameon=False, fontsize=6.5, loc="lower right")
            ax.set_ylim(-1.3, 0.6)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"node ORC (k = {args.k})")
        for ax in axes[1]:
            ax.set_xlabel("relative depth")
        fig.suptitle(
            "Node ORC per layer: median and 10-90% band per model (colour); flat-torus reference at "
            "each layer's TwoNN ID, median (dashed) and 10-90% band (grey)",
            fontsize=9,
        )
        fig.tight_layout()
        suffix = f"_{lab}" if lab else ""
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"fig4_orc_distributions{suffix}.{ext}", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, group, title in ((axes[0], e12.LLMS, "LLMs"), (axes[1], e12.VITS, "ViTs")):
            for m in group:
                z = models.get(m.replace("/", "__"))
                if z is None:
                    continue
                orc = np.nanmedian(z[f"orc_k{args.k}"][:, keep], axis=1)
                ref = np.array([refs[str(d)]["median"] for d in np.rint(z["twonn"]).astype(int)])
                ax.plot(np.linspace(0, 1, len(orc)), orc - ref, color="#2a78d6", alpha=0.6, lw=1.2)
            ax.axhline(0, color=INK_MUTED, lw=0.8)
            ax.set_title(f"{title}: median node ORC - flat-torus median (k = {args.k}) {lab}")
            ax.set_xlabel("relative depth")
        axes[0].set_ylabel("ORC - reference (medians)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"fig3_levels_median{suffix}.{ext}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
