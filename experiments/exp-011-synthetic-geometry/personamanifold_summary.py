"""Summary and figure for the PersonaManifold replication (personamanifold_orc.py output).

    uv run python experiments/exp-011-synthetic-geometry/personamanifold_summary.py \\
        --root results/exp-011-synthetic-geometry/personamanifold

* personamanifold_summary.txt  node ORC per manifold / type, both constructions; Spearman with
                               density; their decoupling test per density quintile
* fig6_pm_decoupling           their Fig. E.1(b) on synthetic data: mean node ORC per type within
                               density quintiles (whitened graph, density at k = 5d), one panel
                               per scenario and ID
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
TYPE_COLOURS = {
    "flat": "#2a78d6",
    "sphere": "#eb6834",
    "tree": "#1baf7a",
    "clusters": "#eda100",
    "spread": "#2a78d6",
    "grouped": "#eb6834",
    "a": "#2a78d6",
    "b": "#eb6834",
    "c": "#1baf7a",
}

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


def fmt_p(p):
    return "  -   " if p is None else f"{p:.0e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recs = [
        json.loads(x) for x in (args.root / "personamanifold_orc.jsonl").read_text().splitlines()
    ]
    out = ["Suite: mean node ORC (frac < 0); Spearman(density at k=5d, node ORC)"]
    out.append(f"{'manifold':<9} {'d':>3} {'whitened':>16} {'euclidean':>16} {'rho':>6} {'p':>7}")
    for r in recs:
        if r["scenario"] != "suite":
            continue
        w, e = r["whitened"], r["euclidean"]
        sp = w["spearman_density"][f"k{r['k']}"]
        out.append(
            f"{r['manifold']:<9} {r['d']:>3} {w['node']['mean']:>+8.3f} ({w['node']['frac_neg']:.2f}) "
            f"{e['node']['mean']:>+8.3f} ({e['node']['frac_neg']:.2f}) {sp['rho']:>+6.2f} {sp['p']:>7.0e}"
        )
    typed = [r for r in recs if r["scenario"] != "suite"]
    for r in typed:
        w = r["whitened"]
        dec = w["decoupling"][f"k{r['k']}"]
        out += ["", f"{r['scenario']} d={r['d']} (truth {r['truth']['curvature_by_type']})"]
        out.append(
            "  mean node ORC by type, whitened: "
            + ", ".join(f"{t} {v['mean']:+.3f}" for t, v in w["by_type"].items())
        )
        out.append(
            "  mean node ORC by type, euclidean: "
            + ", ".join(f"{t} {v['mean']:+.3f}" for t, v in r["euclidean"]["by_type"].items())
        )
        out.append(
            f"  Spearman(density k=5d, ORC) rho {dec['spearman_rho']:+.3f} p {dec['spearman_p']:.0e}"
        )
        for i, b in enumerate(dec["bins"]):
            means = ", ".join(f"{t} {m:+.3f}" for t, m in b["mean_by_type"].items())
            out.append(
                f"  quintile {i + 1}: KW p {fmt_p(b.get('kruskal_p'))}  n {b['count_by_type']}  {means}"
            )
    (args.root / "personamanifold_summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))

    dims = sorted({r["d"] for r in typed})
    scen = ["mixture", "grouped", "null"]
    fig, axes = plt.subplots(
        len(dims), len(scen), figsize=(3.6 * len(scen), 2.9 * len(dims)), squeeze=False
    )
    for i, d in enumerate(dims):
        for j, s in enumerate(scen):
            ax = axes[i, j]
            r = next((r for r in typed if r["d"] == d and r["scenario"] == s), None)
            if r is None:
                ax.set_visible(False)
                continue
            bins = r["whitened"]["decoupling"][f"k{r['k']}"]["bins"]
            types = sorted({t for b in bins for t in b["mean_by_type"]})
            for t in types:
                xs = [q + 1 for q, b in enumerate(bins) if t in b["mean_by_type"]]
                ys = [b["mean_by_type"][t] for b in bins if t in b["mean_by_type"]]
                truth = r["truth"]["curvature_by_type"].get(t)
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=TYPE_COLOURS.get(t, INK_MUTED),
                    ms=4,
                    label=f"{t} (true {'+' if truth == 1 else '-' if truth == -1 else '0'})",
                )
            ps = [b.get("kruskal_p") for b in bins if b.get("kruskal_p") is not None]
            sig = sum(p < 0.001 for p in ps)
            ax.axhline(0, color=INK_MUTED, lw=0.8)
            ax.set_title(f"{s}, ID {d}: KW p<0.001 in {sig}/{len(ps)} quintiles")
            ax.set_xticks(range(1, 6))
            if i == len(dims) - 1:
                ax.set_xlabel("density quintile (low -> high, k = 5d)")
            if j == 0:
                ax.set_ylabel("mean node ORC")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "PersonaManifold's density-decoupling test on synthetic data with known curvature "
        "(their graph and ORC conventions)",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.root / f"fig6_pm_decoupling.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
