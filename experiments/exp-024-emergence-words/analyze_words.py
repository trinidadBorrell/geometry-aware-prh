"""exp-024: which word classes carry the TF-IDF floor? Tables and a figure from `words.jsonl`.

Excess is the score minus its exact null, at the target layer where it is largest. A class counts only
if it beats frequency-matched random words. Writes `words_summary.md`, `fig_words.*`.

    uv run python experiments/exp-024-emergence-words/analyze_words.py --out <dir>
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

CLASSES = [
    "noun",
    "proper",
    "adj",
    "verb",
    "adv",
    "number",
    "function",
    "medium",
    "color",
    "spatial",
    "lighting",
]
NAMES = {
    "noun": "Nouns",
    "proper": "Proper nouns",
    "adj": "Adjectives",
    "verb": "Verbs",
    "adv": "Adverbs",
    "number": "Numbers",
    "function": "Function words",
    "medium": "Medium (lexicon)",
    "color": "Colour (lexicon)",
    "spatial": "Spatial (lexicon)",
    "lighting": "Lighting (lexicon)",
}
CAPTIONS = {"gemini": "Gemini captions", "orig": "Original captions"}
ONLY, CONTROL, ALL = "#2a78d6", "#52514e", "#eb6834"


def excess(r: dict, measure: str) -> np.ndarray:
    """One excess per target layer."""
    if measure == "cka":
        return (np.array(r["cka"]) - np.array(r["cka_mu0"])).ravel()
    k = measure[1:]
    return (np.array(r["knn"][k]) - r["knn_mu0"][k]).ravel()


def best(r: dict, measure: str) -> float:
    return float(excess(r, measure).max())


def table(head, rows) -> list[str]:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
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
    rows = [json.loads(x) for x in (a.out / "words.jsonl").read_text().splitlines()]
    stats = json.loads((a.out / "words_stats.json").read_text())

    def get(caption, target, mode, cls, seed=None):
        return next(
            (
                r
                for r in rows
                if (r["caption"], r["target"], r["mode"], r["class"], r["seed"])
                == (caption, target, mode, cls, seed)
            ),
            None,
        )

    lines = [
        "# exp-024 word classes",
        "",
        "n = 10240 WIT images. Excess over the exact null, best of the target's last 3 stored layers",
        "for ViTs. `controls` is the two frequency-matched random word sets (mean; both shown in the",
        "last column). Empty = captions with no word of the class.",
        "",
    ]
    csvrows = []
    for caption in CAPTIONS:
        for target in ("dino", "clip"):
            for measure, mname in (("k10", "k = 10"), ("k1000", "k = 1000"), ("cka", "CKA")):
                base = best(get(caption, target, "all", "all"), measure)
                t = []
                for cls in CLASSES:
                    o, w = get(caption, target, "only", cls), get(caption, target, "without", cls)
                    cs = [get(caption, target, "control", cls, s) for s in (0, 1)]
                    cs = [c for c in cs if c]
                    if not o or not w:
                        continue
                    oe, we = best(o, measure), best(w, measure)
                    ce = [best(c, measure) for c in cs]
                    cm = float(np.mean(ce)) if ce else float("nan")
                    t.append(
                        [
                            NAMES[cls],
                            float(stats[caption][cls]["token_share"]),
                            o["empty_rows"],
                            oe,
                            cm,
                            oe - cm,
                            we,
                            base - we,
                            " / ".join(f"{x:.3f}" for x in ce),
                        ]
                    )
                    csvrows.append(
                        [
                            caption,
                            target,
                            measure,
                            cls,
                            stats[caption][cls]["token_share"],
                            o["empty_rows"],
                            oe,
                            cm,
                            min(ce, default=np.nan),
                            max(ce, default=np.nan),
                            we,
                            base,
                        ]
                    )
                lines += [f"## {CAPTIONS[caption]}, {target}, {mname}: all words {base:.3f}", ""]
                lines += table(
                    [
                        "class",
                        "token share",
                        "empty",
                        "only",
                        "controls",
                        "only - controls",
                        "without",
                        "all - without",
                        "each control",
                    ],
                    t,
                )
    lines += ["## What LMs track: only-class kernel vs LM layer (first, last)", ""]
    for caption in CAPTIONS:
        t = []
        for cls in ["all", *CLASSES]:
            mode = "all" if cls == "all" else "only"
            row = [NAMES.get(cls, "All words")]
            for target in ("pythia-410m", "llama-13b"):
                r = get(caption, target, mode, cls)
                if r is None:
                    row += ["-", "-", "-", "-"]
                    continue
                k, c = excess(r, "k10"), excess(r, "cka")
                row += [float(k[0]), float(k[-1]), float(c[0]), float(c[-1])]
            t.append(row)
        lines += [CAPTIONS[caption], ""]
        lines += table(
            [
                "class",
                "Pythia k10 first",
                "last",
                "Pythia CKA first",
                "last",
                "LLaMA k10 first",
                "last",
                "LLaMA CKA first",
                "last",
            ],
            t,
        )
    (a.out / "words_summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (a.out / "fig_words.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "caption",
                "target",
                "measure",
                "class",
                "token_share",
                "empty",
                "only",
                "control_mean",
                "control_min",
                "control_max",
                "without",
                "all",
            ]
        )
        w.writerows(csvrows)

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    for r, (measure, mname) in enumerate((("k10", "Mutual kNN, k = 10"), ("cka", "Linear CKA"))):
        for c, caption in enumerate(CAPTIONS):
            ax = axes[r, c]
            x = np.arange(len(CLASSES))
            for tgt, off, alpha in (("dino", -0.14, 1.0), ("clip", 0.14, 0.55)):
                only, cm, lo, hi = [], [], [], []
                for cls in CLASSES:
                    only.append(best(get(caption, tgt, "only", cls), measure))
                    ce = [
                        best(z, measure)
                        for z in (get(caption, tgt, "control", cls, s) for s in (0, 1))
                        if z
                    ]
                    cm.append(np.mean(ce))
                    lo.append(min(ce))
                    hi.append(max(ce))
                cm, lo, hi = map(np.array, (cm, lo, hi))
                ax.bar(
                    x + off, only, 0.26, color=ONLY, alpha=alpha, label=f"Only this class, {tgt}"
                )
                ax.errorbar(
                    x + off,
                    cm,
                    yerr=[cm - lo, hi - cm],
                    fmt="_",
                    color=CONTROL,
                    ms=9,
                    capsize=2,
                    alpha=alpha,
                    label=f"Matched random words, {tgt} (range of 2)",
                )
                ax.axhline(
                    best(get(caption, tgt, "all", "all"), measure),
                    color=ALL,
                    lw=1.2,
                    ls="-" if tgt == "dino" else ":",
                    label=f"All words, {tgt}",
                )
            ax.set_xticks(
                x, [NAMES[k].replace(" (lexicon)", "") for k in CLASSES], rotation=40, ha="right"
            )
            ax.set_title(f"{mname}, {CAPTIONS[caption].lower()}", fontsize=9, loc="left")
            ax.tick_params(colors="#52514e", length=0)
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
        axes[r, 0].set_ylabel("excess over null")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"fig_words.{ext}", dpi=200)
    print("done")


if __name__ == "__main__":
    main()
