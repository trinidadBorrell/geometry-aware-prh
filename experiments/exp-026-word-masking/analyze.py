"""exp-026: alignment across LM depth when word classes are masked. Tables and figures from `rows.jsonl`.

Excess is the score minus its exact null. At each LM layer we take the best of the ViT's 7 stored
layers. Curves are mean and min..max over LM x ViT pairs, on a common relative-depth grid, since the
LMs differ in depth. Each condition has its own TF-IDF floor. Writes `summary.md`, `fig_masking_*`.

    uv run python experiments/exp-026-word-masking/analyze.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {
    "intact": "Intact",
    "shuffled": "All tokens shuffled",
    "nofunc": "No function words",
    "nofunc_shuf": "No function words, shuffled",
    "nouns": "Nouns only",
    "nouns_shuf": "Nouns only, shuffled",
    "nonouns": "No nouns",
    "ctrl_nofunc": "Random removal (control)",
    "ctrl_nonouns": "Random removal (control)",
    "ctrl_nouns": "Random words kept (control)",
}
COLOR = {
    "intact": "#2a78d6",
    "shuffled": "#eb6834",
    "masked": "#1baf7a",
    "masked_shuf": "#8a63d2",
    "control": "#52514e",
}
PANELS = {
    "gemini": [
        ("No function words", "nofunc", "nofunc_shuf", "ctrl_nofunc"),
        ("Nouns only", "nouns", "nouns_shuf", "ctrl_nouns"),
        ("No nouns", "nonouns", None, "ctrl_nonouns"),
    ],
    "orig": [
        ("No function words", "nofunc", "nofunc_shuf", "ctrl_nofunc"),
        ("No nouns", "nonouns", None, "ctrl_nonouns"),
    ],
}
MEASURES = {"10": "Mutual kNN, k = 10", "1000": "Mutual kNN, k = 1000", "cka": "Linear CKA"}
GRID = np.linspace(0, 1, 25)
DEPTHS = np.array([0, 0.25, 0.5, 0.75, 1])


def excess(r: dict, m: str) -> np.ndarray:
    """[L_lm, L_vit]."""
    if m == "cka":
        return np.array(r["cka"]) - np.array(r["cka_mu0"])
    return np.array(r["knn"][m]) - r["knn_mu0"][m]


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def table(head, rows) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows],
        "",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in (a.out / "rows.jsonl").read_text().splitlines()]
    data = {(r["lm"], r["caption"], r["condition"], r["kind"], r["target"]): r for r in rows}
    lms = sorted({r["lm"] for r in rows})
    targets = ("dino", "clip")
    pairs = [(m, t) for m in lms for t in targets]

    def curve(lm, cap, cond, t, m):
        """Best ViT layer at each LM layer."""
        return excess(data[(lm, cap, cond, "lm", t)], m).max(1)

    def floor(lm, cap, cond, t, m):
        base = {"shuffled": "intact", "nofunc_shuf": "nofunc", "nouns_shuf": "nouns"}.get(
            cond, cond
        )
        return float(excess(data[(lm, cap, base, "floor", t)], m).max())

    def present(cap, cond):
        return all((lm, cap, cond, "lm", t) in data for lm, t in pairs)

    lines = [
        "# exp-026 word masking",
        "",
        "n = 10240. Excess over the exact null. `floor`: TF-IDF of the same masked tokens. `gain`: best "
        "layer minus floor. Values are means over LM x ViT pairs, and each row gives min-max for the best "
        "layer.",
        "",
    ]
    for cap in ("gemini", "orig"):
        for m, mname in MEASURES.items():
            t = []
            for cond in LABEL:
                if not present(cap, cond):
                    continue
                best, l0, last, fl, depth = [], [], [], [], []
                for lm, tg in pairs:
                    c = curve(lm, cap, cond, tg, m)
                    best.append(float(c.max()))
                    l0.append(float(c[0]))
                    last.append(float(c[-1]))
                    fl.append(floor(lm, cap, cond, tg, m))
                    depth.append(float(c.argmax() / (len(c) - 1)))
                t.append(
                    [
                        cond,
                        float(np.mean(fl)),
                        float(np.mean(l0)),
                        float(np.mean(last)),
                        float(np.mean(best)),
                        f"{min(best):.3f}-{max(best):.3f}",
                        float(np.mean(best) - np.mean(fl)),
                        f"{min(depth):.2f}-{max(depth):.2f}",
                    ]
                )
            lines += [f"## {cap}, {mname}", ""]
            lines += table(
                [
                    "condition",
                    "floor",
                    "layer 0",
                    "last",
                    "best",
                    "best range",
                    "gain",
                    "best depth",
                ],
                t,
            )
    # By pair, for the main comparison: is a class worse than removing the same number of random words?
    lines += ["## Masked vs its random control, best layer, per pair", ""]
    t = []
    for cap in ("gemini", "orig"):
        for cond, ctrl in (
            ("nofunc", "ctrl_nofunc"),
            ("nonouns", "ctrl_nonouns"),
            ("nouns", "ctrl_nouns"),
        ):
            if not (present(cap, cond) and present(cap, ctrl)):
                continue
            for m in MEASURES:
                d = [
                    float(curve(lm, cap, cond, tg, m).max() - curve(lm, cap, ctrl, tg, m).max())
                    for lm, tg in pairs
                ]
                t.append([cap, cond, m, *d])
    lines += table(
        ["caption", "class", "measure", *[f"{m.split('/')[1]} {t_}" for m, t_ in pairs]], t
    )
    lines += ["## Contrasts between conditions, best layer, per pair", ""]
    t = []
    for cap in ("gemini", "orig"):
        for x, y in (
            ("intact", "shuffled"),
            ("nofunc", "intact"),
            ("nofunc", "nofunc_shuf"),
            ("nouns", "nouns_shuf"),
            ("nouns", "intact"),
            ("nonouns", "intact"),
        ):
            if not (present(cap, x) and present(cap, y)):
                continue
            for m in MEASURES:
                d = [
                    float(curve(lm, cap, x, tg, m).max() - curve(lm, cap, y, tg, m).max())
                    for lm, tg in pairs
                ]
                t.append([cap, f"{x} - {y}", m, *d])
    lines += table(
        ["caption", "contrast", "measure", *[f"{m.split('/')[1]} {t_}" for m, t_ in pairs]], t
    )
    lines += ["## By LM depth, mean over pairs", ""]
    t = []
    for cap in ("gemini", "orig"):
        for cond in (
            "intact",
            "shuffled",
            "nofunc",
            "nofunc_shuf",
            "nouns",
            "nouns_shuf",
            "nonouns",
        ):
            if not present(cap, cond):
                continue
            for m in MEASURES:
                ys = [
                    np.interp(DEPTHS, np.linspace(0, 1, len(c := curve(lm, cap, cond, tg, m))), c)
                    for lm, tg in pairs
                ]
                t.append([cap, cond, m, *[float(v) for v in np.mean(ys, 0)]])
    lines += table(["caption", "condition", "measure", *[f"{d:.2f}" for d in DEPTHS]], t)
    # A shuffle leaves layer 0 unchanged, up to the clamp, whose threshold is computed over all layers.
    chk = [
        abs(curve(lm, c, cond, tg, m)[0] - curve(lm, c, base, tg, m)[0])
        for c in ("gemini", "orig")
        for cond, base in (
            ("shuffled", "intact"),
            ("nofunc_shuf", "nofunc"),
            ("nouns_shuf", "nouns"),
        )
        if present(c, cond) and present(c, base)
        for lm, tg in pairs
        for m in MEASURES
    ]
    lines += [f"Shuffle vs its base at layer 0: largest difference {max(chk):.5f}.", ""]
    (a.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})

    def on_grid(cap, cond, m):
        return np.array(
            [
                np.interp(
                    GRID,
                    np.linspace(0, 1, len(c := curve(lm, cap, cond, tg, m))),
                    c,
                )
                for lm, tg in pairs
            ]
        )

    csvrows = []
    for cap, panels in PANELS.items():
        if not all(present(cap, x) for p_ in panels for x in p_[1:] if x):
            print(f"{cap}: incomplete, no figure")
            continue
        fig, axes = plt.subplots(
            len(MEASURES), len(panels), figsize=(4.1 * len(panels), 9.0), sharex=True, squeeze=False
        )
        for r, (m, mname) in enumerate(MEASURES.items()):
            for c, (title, mask, shuf, ctrl) in enumerate(panels):
                ax = axes[r, c]
                series = [
                    ("intact", "intact", "-"),
                    ("shuffled", "shuffled", "-"),
                    (mask, "masked", "-"),
                    (shuf, "masked_shuf", "-"),
                    (ctrl, "control", "-"),
                ]
                for cond, tone, ls in series:
                    if cond is None:
                        continue
                    y = on_grid(cap, cond, m)
                    color = COLOR[tone]
                    ax.fill_between(GRID, y.min(0), y.max(0), color=color, alpha=0.14, lw=0)
                    ax.plot(GRID, y.mean(0), color=color, ls=ls, lw=2, label=LABEL[cond])
                    csvrows += [
                        [cap, mname, title, LABEL[cond], float(d), float(v), float(lo), float(hi)]
                        for d, v, lo, hi in zip(GRID, y.mean(0), y.min(0), y.max(0), strict=True)
                    ]
                fl = np.array([floor(lm, cap, mask, tg, m) for lm, tg in pairs])
                ax.axhspan(fl.min(), fl.max(), color=COLOR["masked"], alpha=0.14, lw=0)
                ax.axhline(
                    fl.mean(), color=COLOR["masked"], ls="--", lw=1.1, label="TF-IDF, masked tokens"
                )
                fi = np.array([floor(lm, cap, "intact", tg, m) for lm, tg in pairs])
                ax.axhspan(fi.min(), fi.max(), color=COLOR["intact"], alpha=0.12, lw=0)
                ax.axhline(
                    fi.mean(), color=COLOR["intact"], ls="--", lw=1.1, label="TF-IDF, intact tokens"
                )
                ax.set_ylim(bottom=0)
                ax.tick_params(colors="#52514e", length=0)
                if r == 0:
                    ax.set_title(title, fontsize=10, loc="left")
                if r == len(MEASURES) - 1:
                    ax.set_xlabel("LM relative depth (0 = embedding layer)")
                if c == 0:
                    ax.set_ylabel(f"{mname}\nexcess over null")
                if r == len(MEASURES) - 1:
                    ax.legend(
                        frameon=False,
                        fontsize=7.5,
                        ncol=2,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.2),
                    )
        name = "Gemini captions" if cap == "gemini" else "original captions"
        fig.suptitle(
            f"Masking word classes, {name}. Lines: mean over {len(pairs)} LM x ViT pairs. "
            "Bands: min..max. n = 10240.",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(a.out / f"fig_masking_{cap}.{ext}", dpi=200)
        plt.close(fig)
    with (a.out / "fig_masking.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caption", "measure", "panel", "series", "depth", "mean", "min", "max"])
        w.writerows(csvrows)
    print("done")


if __name__ == "__main__":
    main()
