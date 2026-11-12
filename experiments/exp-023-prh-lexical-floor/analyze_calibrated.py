"""exp-023: calibrated TF-IDF floor, layer 0 and best layer on the PRH pool (Algorithm 2).

Excess is the raw max minus the mean of the null of the max, over the cells each series chooses from.
g is Gröger's gated score, (raw - tau) / (1 - tau). Writes `calibrated_summary.md`, `fig_calibrated.*`.

    uv run python experiments/exp-023-prh-lexical-floor/analyze_calibrated.py --out <dir> \\
        --stored <prh_val> --stored-cal <prh_val_calibrated>
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
KEYS = {"mutual_knn_k10": "Mutual kNN, k = 10", "cka_lin": "Linear CKA"}
STORED = {"mutual_knn_k10": "mutual_knn", "cka_lin": "cka_lin"}
FAMS = ["augreg", ".mae", "dinov2", ".laion2b", "ft_in12k"]
FAM_NAME = ["AugReg", "MAE", "DINOv2", "CLIP", "CLIP ft"]
SIZES = ["tiny", "small", "base", "large", "huge", "giant"]


def fam(v: str) -> int:
    if "ft_in12k" in v:
        return 4
    return next(i for i, k in enumerate(FAMS) if k in v)


def size(v: str) -> int:
    return next(i for i, s in enumerate(SIZES) if s in v)


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def table(head, rows) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows],
        "",
    ]


def med(a) -> str:
    return f"{np.median(a):.3f} ({a.min():.3f}-{a.max():.3f})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stored", type=Path, required=True)
    p.add_argument("--stored-cal", type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.stored / "cross_modal.json").read_text())
    rows = [json.loads(x) for x in (a.out / "calibrated.jsonl").read_text().splitlines()]
    cal = {}
    for line in (a.stored_cal / "calibrated_pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        cal[(r["llm"], r["lvm"])] = r
    llms = sorted(meta["llms"], key=lambda m: -meta["models"][m]["bpb"])
    vits = sorted(meta["lvms"], key=lambda v: (fam(v), size(v)))
    data = {}
    for r in rows:
        for m in r["llms"]:
            if r["kind"] == "layer0" and r["name"] != m:
                continue
            data[(r["kind"], m, r["vit"])] = r

    def cube(kind, key, field):
        out = np.zeros((len(llms), len(vits)))
        for i, m in enumerate(llms):
            for j, v in enumerate(vits):
                if kind == "best":
                    out[i, j] = cal[(m, v)][STORED[key]][field]
                else:
                    out[i, j] = data[(kind, m, v)][key][field]
        return out

    kinds = {
        "tfidf": "TF-IDF, no model",
        "word_tfidf_nostop": "Word TF-IDF, no stopwords",
        "layer0": "Layer 0",
        "best": "Best layer",
    }
    lines = [
        "# exp-023 calibrated summary",
        "",
        "n = 1024, 200 permutations (p floor 0.005). Median (min-max) over 170 pairs.",
        "",
    ]
    t = []
    res = {}
    for key, name in KEYS.items():
        for k, label in kinds.items():
            raw, mu, _tau, pv, g = (
                cube(k, key, f) for f in ("raw_score", "mu0", "tau_alpha", "p_value", "g_score")
            )
            ex = raw - mu
            res[(key, k)] = {"raw": raw, "mu": mu, "ex": ex, "g": g, "p": pv}
            t.append([name, label, med(raw), med(mu), med(ex), med(g), f"{(pv < 0.05).mean():.0%}"])
    lines += table(
        ["measure", "series", "raw", "null of max", "excess", "g", "pairs with p < 0.05"], t
    )
    t = []
    for key, name in KEYS.items():
        for k in ("tfidf", "word_tfidf_nostop", "layer0"):
            x, b = res[(key, k)], res[(key, "best")]
            t.append(
                [
                    name,
                    kinds[k],
                    float(np.median(x["ex"] / b["ex"])),
                    float((x["ex"] / b["ex"]).min()),
                    float((x["ex"] / b["ex"]).max()),
                    float(np.median(x["g"] / b["g"])),
                    f"{(x['ex'] >= b['ex']).mean():.0%}",
                    float(np.median(b["ex"] - x["ex"])),
                ]
            )
    lines += table(
        [
            "measure",
            "series",
            "median excess / best",
            "min",
            "max",
            "median g / best",
            "pairs at or above best",
            "median best - series",
        ],
        t,
    )
    lines += ["Ratios and differences are descriptive; they are not shares of variance.", ""]

    cap = np.array([1 - meta["models"][m]["bpb"] for m in llms])
    params = np.log10([meta["models"][m]["num_params"] for m in llms])
    t = []
    for key, name in KEYS.items():
        for k in ("layer0", "best"):
            for cname, c in (("1 - bpb", cap), ("log params", params)):
                ex = res[(key, k)]["ex"]
                rho = np.array([spearmanr(c, ex[:, j]).statistic for j in range(len(vits))])
                t.append(
                    [
                        name,
                        kinds[k],
                        cname,
                        f"{np.median(rho):.2f} ({rho.min():.2f} to {rho.max():.2f})",
                    ]
                )
        gap = res[(key, "best")]["ex"] - res[(key, "tfidf")]["ex"]
        for cname, c in (("1 - bpb", cap), ("log params", params)):
            rho = np.array([spearmanr(c, gap[:, j]).statistic for j in range(len(vits))])
            t.append(
                [
                    name,
                    "Best - TF-IDF",
                    cname,
                    f"{np.median(rho):.2f} ({rho.min():.2f} to {rho.max():.2f})",
                ]
            )
    lines += [
        "## LLM capability: Spearman across LLMs, one value per ViT, on calibrated excess",
        "",
    ]
    lines += table(["measure", "series", "capability", "Spearman, median (min to max)"], t)

    t = []
    for i, m in enumerate(llms):
        row = [m.split("/")[-1], float(cap[i])]
        for key in KEYS:
            row += [float(res[(key, x)]["ex"][i].mean()) for x in ("tfidf", "layer0", "best")]
        t.append(row)
    lines += ["## By LLM: mean calibrated excess over 17 ViTs", ""]
    lines += table(
        [
            "LLM",
            "1 - bpb",
            "mKNN TF-IDF",
            "mKNN L0",
            "mKNN best",
            "CKA TF-IDF",
            "CKA L0",
            "CKA best",
        ],
        t,
    )
    t = []
    for j, v in enumerate(vits):
        row = [v.replace("vit_", "").split("_patch")[0] + " " + FAM_NAME[fam(v)]]
        for key in KEYS:
            row += [float(res[(key, x)]["ex"][:, j].mean()) for x in ("tfidf", "layer0", "best")]
        t.append(row)
    lines += ["## By ViT: mean calibrated excess over 10 LLMs", ""]
    lines += table(
        ["ViT", "mKNN TF-IDF", "mKNN L0", "mKNN best", "CKA TF-IDF", "CKA L0", "CKA best"], t
    )
    t = []
    for f, fname in enumerate(FAM_NAME):
        jj = [j for j, v in enumerate(vits) if fam(v) == f]
        sz = [size(vits[j]) for j in jj]
        for key, name in KEYS.items():
            row = [fname, name]
            for x in ("tfidf", "layer0", "best"):
                y = res[(key, x)]["ex"][:, jj].mean(0)
                row.append(f"{spearmanr(sz, y).statistic:.2f}")
            gap = (res[(key, "best")]["ex"] - res[(key, "tfidf")]["ex"])[:, jj].mean(0)
            row.append(f"{spearmanr(sz, gap).statistic:.2f}")
            t.append(row)
    lines += ["## ViT size, within family: Spearman on mean excess", ""]
    lines += table(["family", "measure", "TF-IDF", "layer 0", "best", "best - TF-IDF"], t)
    (a.out / "calibrated_summary.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharey="row")
    csvrows = []
    llm_labels = [m.split("/")[-1] for m in llms]
    vit_labels = [v.replace("vit_", "").split("_patch")[0] + " " + FAM_NAME[fam(v)] for v in vits]
    for r, (key, name) in enumerate(KEYS.items()):
        for c, (labels, axis, ax_name) in enumerate(
            ((llm_labels, 1, "LLM"), (vit_labels, 0, "ViT"))
        ):
            ax = axes[r, c]
            x = np.arange(len(labels))
            for label, color, k, ls in (
                ("Best layer", BEST, "best", "-"),
                ("Layer 0", LAYER0, "layer0", "-"),
                ("TF-IDF, no model", FLOOR, "tfidf", "--"),
            ):
                ex = res[(key, k)]["ex"]
                ax.fill_between(x, ex.min(axis), ex.max(axis), color=color, alpha=0.16, lw=0)
                ax.plot(
                    x,
                    ex.mean(axis),
                    color=color,
                    lw=2,
                    ls=ls,
                    marker="o" if ls == "-" else None,
                    ms=4,
                    label=label,
                )
                csvrows += [
                    [ax_name, name, label, lab, ex.mean(axis)[i], ex.min(axis)[i], ex.max(axis)[i]]
                    for i, lab in enumerate(labels)
                ]
            ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7.5)
            ax.tick_params(colors="#52514e", length=0)
            ax.set_ylim(bottom=0)
        axes[r, 0].set_ylabel(f"{name}\nexcess over null of max")
    axes[0, 0].set_title(
        "By LLM, ordered by 1 - bpb on captions (band: 17 ViTs)", fontsize=9, loc="left"
    )
    axes[0, 1].set_title("By ViT, family and size (band: 10 LLMs)", fontsize=9, loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.suptitle(
        "PRH pool on WIT-1024, original captions: null-calibrated", fontsize=10, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"fig_calibrated.{ext}", dpi=200)
    with (a.out / "fig_calibrated.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "measure", "series", "label", "mean", "min", "max"])
        w.writerows(csvrows)
    print("done")


if __name__ == "__main__":
    main()
