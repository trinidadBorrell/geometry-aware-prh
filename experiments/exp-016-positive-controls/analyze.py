"""exp-016: tables and figures from `pairs.jsonl` (written by `measure.py --stages pairs`).

Every value is an excess over the permutation null of the pairing, score - mu0, so measures with
different null levels (mutual kNN ~k/n, CKA ~0.02, rank agreements ~0) read on one scale. Per pair:
the best layer pair (largest excess) and, for dual encoders, the final embeddings; `sig` is the
max-over-layers test (Algorithm 2, p < 0.05).

Outputs in --out: `summary.md` (groups, arm A per tower, B3 Gemini - original per LLM x ViT pair),
`fig_groups.png/pdf`, `fig_b3.png/pdf`.

    uv run python experiments/exp-016-positive-controls/analyze.py \\
        --pairs results/exp-016-positive-controls/pairs.jsonl --out results/exp-016-positive-controls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MEASURES = [
    ("mutual_knn", "mutual kNN (k = 10)"),
    ("cka_lin", "linear CKA"),
    ("orc_k10", "ORC given density (k = 10)"),
    ("dens_k10", "density (k = 10)"),
]
# reading order: positive control, ceilings, references, negative controls
GROUPS = ["A_orig", "A_gemini", "A_cross", "B1", "B2", "B3_orig", "B3_gemini", "X", "W_llm",
          "W_vit", "D_shuffle", "D_random"]  # fmt: skip
TAGS = ["d1", "d2s", "d2l"]


def excess(row: dict, m: str, which: str = "best") -> float:
    c = row.get(m, {}).get(which)
    return np.nan if c is None else c["score"] - c["mu0"]


def threshold(row: dict, m: str, which: str = "best") -> float:
    """Null 95% threshold as an excess. The best cell is a maximum over all layer pairs, so it is
    compared with the null of the maximum (Algorithm 2), not with its own cell's null."""
    c = row.get(m, {}).get(which)
    if c is None:
        return np.nan
    return (row[m]["max_tau"] if which == "best" else c["tau"]) - c["mu0"]


def n_sig(sub: list[dict], m: str) -> str:
    v = [r[m]["max_p"] < 0.05 for r in sub if m in r]
    return f"{sum(v)}/{len(v)}"


def short(name: str) -> str:
    for cut in ("__text-orig", "__text-gemini", "__image", "__orig", "__gemini", "__cls"):
        name = name.replace(cut, "")
    return name.split("__")[-1].split(".")[0]


def fmt(v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if not len(v):
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]"


def group_table(rows: list[dict]) -> list[str]:
    out = [
        "| group | data | pairs | " + " | ".join(t for _, t in MEASURES) + " |",
        "|---|---|---|" + "---|" * len(MEASURES),
    ]
    for g in GROUPS:
        for tag in TAGS:
            sub = [r for r in rows if r["group"] == g and r["tag"] == tag]
            if not sub:
                continue
            cells = []
            for m, _ in MEASURES:
                v = np.array([excess(r, m) for r in sub])
                if not np.isfinite(v).any():
                    cells.append("-")
                    continue
                null = np.nanmedian([threshold(r, m) for r in sub])
                cells.append(f"{fmt(v)} ({n_sig(sub, m)}; null {null:.3f})")
            out.append(f"| {g} | {tag} | {len(sub)} | " + " | ".join(cells) + " |")
    return out


def arm_a_table(rows: list[dict]) -> list[str]:
    out = [
        "| pair | data | captions | mkNN final (95% CI) | mkNN best | CKA final | CKA best "
        "| ORC final | ORC best | density final |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["group"] not in ("A_orig", "A_gemini", "A_cross"):
            continue
        fin = r["mutual_knn"].get("final")
        ci = (
            f"{excess(r, 'mutual_knn', 'final'):.3f} ({fin['excess_ci'][1]:.3f}-{fin['excess_ci'][2]:.3f})"
            if fin
            else "-"
        )
        vals = [
            excess(r, "mutual_knn"),
            excess(r, "cka_lin", "final"),
            excess(r, "cka_lin"),
            excess(r, "orc_k10", "final"),
            excess(r, "orc_k10"),
            excess(r, "dens_k10", "final"),
        ]
        caps = "gemini" if r["group"] == "A_gemini" else "orig"
        pair = short(r["a"]) if r["group"] != "A_cross" else f"{short(r['a'])} x {short(r['b'])}"
        out.append(
            f"| {pair} | {r['tag']} | {caps} | {ci} | "
            + " | ".join("-" if not np.isfinite(v) else f"{v:.3f}" for v in vals)
            + " |"
        )
    return out


def b3_pairs(rows: list[dict]) -> list[tuple[str, str, str, dict, dict]]:
    """(llm, vit, tag, original row, Gemini row) for every LLM x ViT pair measured with both."""
    orig = {
        (r["a"].rsplit("__", 1)[0], r["b"], r["tag"]): r for r in rows if r["group"] == "B3_orig"
    }
    out = []
    for r in rows:
        if r["group"] == "B3_gemini":
            key = (r["a"].rsplit("__", 1)[0], r["b"], r["tag"])
            if key in orig:
                out.append((short(key[0]), short(key[1]), key[2], orig[key], r))
    return out


def b3_table(rows: list[dict]) -> list[str]:
    out = [
        "| LLM | ViT | data | mkNN orig | mkNN Gemini | diff | CKA orig | CKA Gemini | diff "
        "| ORC orig | ORC Gemini | diff |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for llm, vit, tag, ro, rg in b3_pairs(rows):
        cells = []
        for m in ("mutual_knn", "cka_lin", "orc_k10"):
            a, b = excess(ro, m), excess(rg, m)
            cells += (
                ["-", "-", "-"] if not np.isfinite(a) else [f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"]
            )
        out.append(f"| {llm} | {vit} | {tag} | " + " | ".join(cells) + " |")
    return out


def fig_groups(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [
        (g, tag)
        for g in GROUPS
        for tag in TAGS
        if tag != "d2l" and any(r["group"] == g and r["tag"] == tag for r in rows)
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(
        len(MEASURES), 1, figsize=(max(8, 0.55 * len(labels)), 11), sharex=True
    )
    rng = np.random.default_rng(0)
    for ax, (m, title) in zip(axes, MEASURES, strict=True):
        for i, (g, tag) in enumerate(labels):
            sub = [r for r in rows if r["group"] == g and r["tag"] == tag]
            v = np.array([excess(r, m) for r in sub])
            th = np.array([threshold(r, m) for r in sub])
            if np.isfinite(th).any():  # null 95% threshold, min..max over the group's pairs
                ax.fill_between(
                    [i - 0.4, i + 0.4],
                    np.nanmin(th),
                    np.nanmax(th),
                    color="#52514e",
                    alpha=0.25,
                    lw=0,
                )
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            colour = "#2a78d6" if g.startswith(("A", "B")) else "#d6662a" if g == "X" else "#52514e"
            ax.vlines(i, v.min(), v.max(), color=colour, alpha=0.35, lw=6)  # spread over pairs
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=colour, zorder=3)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", lw=1.2, zorder=4)
            if m == "mutual_knn":
                fin = np.array([excess(r, m, "final") for r in sub])
                fin = fin[np.isfinite(fin)]
                if len(fin):
                    ax.scatter(
                        np.full(len(fin), i + 0.3), fin, marker="D", s=14, color="#7a1fa2", zorder=5
                    )
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("excess over null", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(x, [f"{g}\n{t}" for g, t in labels], fontsize=7, rotation=90)
    fig.suptitle(
        "exp-016: best layer pair per pair of models (dots; bar = min..max, line = median)\n"
        "grey = null 95% threshold of the max over layer pairs (min..max); diamonds = final embeddings",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_groups.{ext}", dpi=200)
    plt.close(fig)


def fig_b3(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = b3_pairs(rows)
    tags = [t for t in ("d2s", "d2l") if any(p[2] == t for p in pairs)]
    if not tags:
        return
    ms = MEASURES[:3]
    fig, axes = plt.subplots(
        len(tags), len(ms), figsize=(4 * len(ms), 3.4 * len(tags)), squeeze=False
    )
    for row_axes, tag in zip(axes, tags, strict=True):
        sub = [p for p in pairs if p[2] == tag]
        a_rows = {
            g: [r for r in rows if r["group"] == g and r["tag"] == tag]
            for g in ("A_orig", "A_gemini")
        }
        for ax, (m, title) in zip(row_axes, ms, strict=True):
            for _, _, _, ro, rg in sub:
                a, b = excess(ro, m), excess(rg, m)
                if np.isfinite(a):
                    ax.plot([0, 1], [a, b], color="#d6662a", alpha=0.5, marker="o", ms=3)
            for ro, rg in zip(a_rows["A_orig"], a_rows["A_gemini"], strict=False):
                a, b = excess(ro, m, "final"), excess(rg, m, "final")
                if np.isfinite(a):
                    ax.plot([0, 1], [a, b], color="#2a78d6", alpha=0.8, marker="D", ms=4)
            ax.set_xticks([0, 1], ["original", "Gemini"])
            ax.set_xlim(-0.3, 1.3)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{title}, {tag}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if not ax.lines[:-1]:  # only the zero line: not measured at this n
                ax.text(0.5, 0.5, "not computed", ha="center", transform=ax.transAxes, fontsize=8)
    axes[0, 0].plot([], [], color="#d6662a", marker="o", label="LLM x ViT, best layer pair")
    axes[0, 0].plot([], [], color="#2a78d6", marker="D", label="dual encoder, final")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("B3: excess over null with original vs Gemini captions", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_b3.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    text = [
        f"# exp-016 summary ({len(rows)} pairs)",
        "",
        "Excess over the permutation null (score - mu0) at the best layer pair: median [min, max] "
        "over the group's pairs; in brackets: pairs significant under the max-over-layers test "
        "(Algorithm 2, p < 0.05) and the median null 95% threshold of that maximum, as an excess.",
        "",
        *group_table(rows),
        "",
        "## Arm A: dual encoders",
        "",
        *arm_a_table(rows),
        "",
        "## B3: Gemini minus original captions, LLM x ViT (best layer pair)",
        "",
        *b3_table(rows),
        "",
    ]
    (args.out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    fig_groups(rows, args.out)
    fig_b3(rows, args.out)


if __name__ == "__main__":
    main()
