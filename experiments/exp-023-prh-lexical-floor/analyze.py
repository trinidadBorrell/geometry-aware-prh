"""exp-023: lexical floor vs layer 0 vs best layer in the PRH pool; capability trends and figure.

uv run python experiments/exp-023-prh-lexical-floor/analyze.py --out <results dir> --stored <prh_val>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEST, LAYER0, FLOOR = "#2a78d6", "#eb6834", "#52514e"
MEASURES = [
    ("mknn", "Mutual kNN, k = 10 (raw)"),
    ("cka", "Linear CKA (raw, biased)"),
    ("ucka", "Unbiased CKA"),
]
LEX = ["tf", "tfidf", "binary", "tfidf_sublinear", "word_tfidf", "word_tfidf_nostop"]
FAMILIES = {"bloomz": "bigscience/bloomz", "open_llama": "openlm-research/", "llama": "huggyllama/"}
VIT_FAMILIES = {
    "AugReg": "augreg",
    "MAE": ".mae",
    "DINOv2": "dinov2",
    "CLIP": ".laion2b",
    "CLIP ft": "ft_in12k",
}
SIZES = ["tiny", "small", "base", "large", "huge", "giant"]


def vit_family(v: str) -> str:
    if "ft_in12k" in v:
        return "CLIP ft"
    for name, key in VIT_FAMILIES.items():
        if key in v:
            return name
    raise ValueError(v)


def vit_size(v: str) -> int:
    return next(i for i, s in enumerate(SIZES) if s in v)


def short(m: str) -> str:
    return m.split("/")[-1]


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
    p.add_argument("--stored", type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.stored / "cross_modal.json").read_text())
    rows = [json.loads(x) for x in (args.out / "pairs.jsonl").read_text().splitlines()]
    llms = sorted({r["llm"] for r in rows}, key=lambda m: -meta["models"][m]["bpb"])
    vits = sorted(
        {r["vit"] for r in rows},
        key=lambda v: (list(VIT_FAMILIES).index(vit_family(v)), vit_size(v)),
    )
    idx = {(r["llm"], r["vit"]): r for r in rows}
    assert len(rows) == len(llms) * len(vits)

    def arr(fn):
        return np.array([[fn(idx[(m, v)]) for v in vits] for m in llms])

    cube = {}
    for m, _ in MEASURES:
        cube[m] = {
            "best": arr(lambda r, m=m: float(max(r["lm"][m]))),
            "layer0": arr(lambda r, m=m: float(r["lm"][m][0])),
            "deep": arr(lambda r, m=m: float(max(r["lm"][m][1:]))),
            "best_layer": arr(lambda r, m=m: int(np.argmax(r["lm"][m]))),
            "depth": arr(lambda r, m=m: float(np.argmax(r["lm"][m]) / (r["n_llm_layers"] - 1))),
            **{x: arr(lambda r, m=m, x=x: float(r["lexical"][x][m])) for x in LEX},
        }
    params = np.log10([meta["models"][m]["num_params"] for m in llms])
    cap = np.array([1 - meta["models"][m]["bpb"] for m in llms])
    lines = [
        "# exp-023 summary",
        "",
        f"{len(llms)} LLMs x {len(vits)} ViTs, n = 1024, raw scores.",
        "",
    ]

    # 1. Levels
    lines += ["## Levels over all pairs: median (min-max)", ""]
    head = ["measure", "chance", *LEX, "layer 0", "best layer"]
    t = []
    for m, name in MEASURES:
        c = cube[m]
        chance = "0.010" if m == "mknn" else ("see note" if m == "cka" else "0")
        row = [name, chance]
        for x in [*LEX, "layer0", "best"]:
            a = c[x]
            row.append(f"{np.median(a):.3f} ({a.min():.3f}-{a.max():.3f})")
        t.append(row)
    lines += table(head, t)

    # 2. Comparisons, per pair
    lines += ["## Per pair: word statistics against the model", ""]
    t = []
    for m, name in MEASURES:
        c = cube[m]
        for x in ("tf", "tfidf", "tfidf_sublinear", "word_tfidf_nostop"):
            t.append(
                [
                    name,
                    x,
                    float(np.median(c[x] / c["best"])),
                    float((c[x] / c["best"]).min()),
                    float((c[x] / c["best"]).max()),
                    float(np.median(c[x] / c["layer0"])),
                    f"{(c[x] >= c['layer0']).mean():.0%}",
                    f"{(c[x] >= c['best']).mean():.0%}",
                ]
            )
    lines += table(
        [
            "measure",
            "word kernel",
            "median ratio to best",
            "min",
            "max",
            "median ratio to layer 0",
            "pairs at or above layer 0",
            "pairs at or above best",
        ],
        t,
    )
    lines += ["Ratios are descriptive; they are not shares of variance.", ""]

    t = []
    for m, name in MEASURES:
        c = cube[m]
        t.append(
            [
                name,
                float(np.median(c["layer0"] / c["best"])),
                float(np.median(c["depth"])),
                f"{(c['best_layer'] == 0).mean():.0%}",
            ]
        )
    lines += table(
        [
            "measure",
            "median layer 0 / best",
            "median relative depth of best layer",
            "pairs where layer 0 is best",
        ],
        t,
    )

    # 3. Language capability
    lines += [
        "## Language capability: Spearman across the 10 LLMs, one value per ViT, median (min-max)",
        "",
    ]
    t = []
    for m, name in MEASURES:
        c = cube[m]
        for x in ("layer0", "best", "tfidf", "word_tfidf_nostop"):
            for cname, cv in (("1 - bpb", cap), ("log params", params)):
                rhos = np.array([spearmanr(cv, c[x][:, j]).statistic for j in range(len(vits))])
                t.append(
                    [
                        name,
                        x,
                        cname,
                        f"{np.nanmedian(rhos):.2f} ({np.nanmin(rhos):.2f} to {np.nanmax(rhos):.2f})",
                    ]
                )
    lines += table(["measure", "series", "capability", "Spearman"], t)
    lines += [
        "Word kernels vary across LLMs only through the tokenizer; word-level TF-IDF does not vary at all.",
        "",
    ]

    lines += ["## By LLM, mean over 17 ViTs", ""]
    t = []
    for i, mdl in enumerate(llms):
        row = [short(mdl), float(cap[i]), f"{10 ** params[i] / 1e9:.2f}B"]
        for m, _ in MEASURES:
            c = cube[m]
            row += [
                float(c["layer0"][i].mean()),
                float(c["best"][i].mean()),
                float(c["tfidf"][i].mean()),
            ]
        t.append(row)
    lines += table(
        [
            "LLM",
            "1 - bpb",
            "params",
            "mKNN L0",
            "mKNN best",
            "mKNN TF-IDF",
            "CKA L0",
            "CKA best",
            "CKA TF-IDF",
            "uCKA L0",
            "uCKA best",
            "uCKA TF-IDF",
        ],
        t,
    )

    lines += ["## Within LLM family (mean over ViTs): Spearman with params", ""]
    t = []
    for fam, key in FAMILIES.items():
        ii = [i for i, m in enumerate(llms) if m.startswith(key)]
        for m, name in MEASURES:
            c = cube[m]
            for x in ("layer0", "best"):
                y = c[x][ii].mean(1)
                t.append(
                    [
                        fam,
                        len(ii),
                        name,
                        x,
                        f"{spearmanr(params[ii], y).statistic:.2f}" if len(ii) > 2 else "-",
                        " ".join(f"{v:.3f}" for v in y[np.argsort(params[ii])]),
                    ]
                )
    lines += table(
        ["family", "models", "measure", "series", "Spearman", "values, small to large"], t
    )

    # 4. Vision capability
    lines += ["## Vision capability: mean over 10 LLMs, by ViT", ""]
    t = []
    for j, v in enumerate(vits):
        row = [v.replace("vit_", "")]
        for m, _ in MEASURES:
            c = cube[m]
            row += [
                float(c["layer0"][:, j].mean()),
                float(c["best"][:, j].mean()),
                float(c["tfidf"][:, j].mean()),
            ]
        t.append(row)
    lines += table(
        [
            "ViT",
            "mKNN L0",
            "mKNN best",
            "mKNN TF-IDF",
            "CKA L0",
            "CKA best",
            "CKA TF-IDF",
            "uCKA L0",
            "uCKA best",
            "uCKA TF-IDF",
        ],
        t,
    )

    lines += ["## Within ViT family: Spearman with size, across sizes, mean over LLMs", ""]
    t = []
    for fam in VIT_FAMILIES:
        jj = [j for j, v in enumerate(vits) if vit_family(v) == fam]
        sz = [vit_size(vits[j]) for j in jj]
        for m, name in MEASURES:
            c = cube[m]
            row = [fam, len(jj), name]
            for x in ("layer0", "best", "tfidf"):
                y = c[x][:, jj].mean(0)
                row.append(f"{spearmanr(sz, y).statistic:.2f}" if len(jj) > 2 else "-")
            t.append(row)
    lines += table(["family", "sizes", "measure", "layer 0", "best", "TF-IDF"], t)
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # 5. Figure: every series with its band
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.6), sharey="row")
    csvrows = []

    def draw(ax, labels, data, axis_name):
        x = np.arange(len(labels))
        for label, color, a, ls in data:
            band = 1 if axis_name == "LLM" else 0
            ax.fill_between(x, a.min(band), a.max(band), color=color, alpha=0.16, lw=0)
            mean = a.mean(band)
            ax.plot(
                x,
                mean,
                color=color,
                lw=2,
                marker="o" if ls == "-" else None,
                ms=4,
                ls=ls,
                label=label,
            )
            for i, lab in enumerate(labels):
                csvrows.append(
                    [axis_name, cur, label, lab, mean[i], a.min(band)[i], a.max(band)[i]]
                )
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7.5)
        ax.tick_params(colors="#52514e", length=0)
        ax.set_ylim(bottom=0)

    for r, (m, name) in enumerate(MEASURES):
        cur = m
        c = cube[m]
        series = [
            ("Best layer", BEST, c["best"], "-"),
            ("Layer 0", LAYER0, c["layer0"], "-"),
            ("TF-IDF, no model", FLOOR, c["tfidf"], "--"),
        ]
        draw(axes[r, 0], [short(x) for x in llms], series, "LLM")
        draw(
            axes[r, 1],
            [v.replace("vit_", "").split("_patch")[0] + " " + vit_family(v) for v in vits],
            series,
            "ViT",
        )
        axes[r, 0].set_ylabel(name)
    axes[0, 0].set_title(
        "By LLM, ordered by 1 - bpb on captions (band: 17 ViTs)", fontsize=9, loc="left"
    )
    axes[0, 1].set_title("By ViT, family and size (band: 10 LLMs)", fontsize=9, loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.suptitle(
        "PRH pool on WIT-1024, original captions: uncalibrated scores, best ViT layer",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig_floor.{ext}", dpi=200)
    with (args.out / "fig_floor.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "measure", "series", "label", "mean", "min", "max"])
        w.writerows(csvrows)
    print("done")


if __name__ == "__main__":
    main()
