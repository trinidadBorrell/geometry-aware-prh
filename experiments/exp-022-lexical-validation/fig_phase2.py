"""exp-022 phase 2 figure: alignment across LM layers on WIT and DOCCI, with shuffle and word floors.

Mean and min..max over LM x ViT pairs at each LM layer (best ViT layer); floors are TF-IDF and
sublinear TF-IDF with no model, min..max over ViTs. Writes `phase2/fig_phase2.{png,pdf,csv}`.

    uv run python experiments/exp-022-lexical-validation/fig_phase2.py --out <results dir>
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

DINO = "vit_giant_patch14_dinov2.lvd142m__cls"
CLIP = "vit_huge_patch14_clip_224.laion2b__cls"
MAE = "vit_huge_patch14_224.mae__cls"
AUGREG = "vit_large_patch16_224.augreg_in21k__cls"
INTACT, SHUFFLED, FLOOR = "#2a78d6", "#eb6834", "#52514e"
FINAL = "step143000"
LMS = {
    "EleutherAI__pythia-410m": f"@{FINAL}",
    "openlm-research__open_llama_7b": "",
    "bigscience__bloomz-7b1": "",
}
MEASURES = [("knn10", "10", "Mutual kNN, k = 10"), ("cka", None, "Linear CKA")]


def prof(g: dict, m: str) -> np.ndarray:
    return (np.array(g[m]) - np.array(g[f"{m}_mu0"])).max(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    p2 = args.out / "phase2"
    new = json.loads((p2 / "phase2_grids.json").read_text())
    fixed = {g["id"]: g for g in json.loads((p2 / "big_shuffle_grids.json").read_text())}
    grids = {g["id"]: fixed.get(g["id"], g) for g in new}
    old = {g["id"]: g for g in json.loads((args.out / "grids_full.json").read_text())}
    rows1 = {}
    for line in (args.out / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["stage"] == "lexical":  # other stages reuse these names on subsets of the rows
            rows1[(r["caption"], r["variant"], r["target"])] = r
    doc = {}
    for line in (p2 / "docci_lexical.jsonl").read_text().splitlines():
        r = json.loads(line)
        doc[(r["variant"], r["target"])] = r

    def lm_id(lm, tail):
        return f"{lm}{LMS[lm]}{tail}"

    def intact(lm, cap, t):
        if lm.startswith("Eleuther"):
            if cap == "gemini":
                return old[f"LV__{lm_id(lm, '__gemini')}__{t}__d2l"]
            return grids[f"LV__{lm_id(lm, '__orig')}__{t}__d2l"]
        return old[f"B3_{cap}__{lm}__{cap}__{t}__d2l"]

    def shuffled(lm, cap, t):
        return grids[f"SHUF__{lm_id(lm, f'__{cap}-shuf0')}__{t}__d2l"]

    def lex(row, k):
        if k is None:
            return float((np.array(row["cka"]) - np.array(row["cka_mu0"])).max())
        return float((np.array(row["knn"][k]) - row["knn_mu0"][k]).max())

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex=True, sharey="row")
    x = np.linspace(0, 1, 7)
    table = []
    panels = [
        ("WIT, Gemini captions", "gemini"),
        ("WIT, original captions", "orig"),
        ("DOCCI, human long captions (other images)", "docci"),
    ]
    for r, (m, k, name) in enumerate(MEASURES):
        for c, (title, cap) in enumerate(panels):
            ax = axes[r, c]
            if cap == "docci":
                d = np.stack(
                    [
                        prof(grids[f"DOCCI__{lm_id(lm, '__docci')}__{t}__d2l"], m)
                        for lm in LMS
                        for t in (DINO, CLIP, MAE, AUGREG)
                    ]
                )
                f = np.array(
                    [
                        lex(doc[(v, t)], k)
                        for v in ("tfidf", "tfidf_sublinear")
                        for t in (DINO, CLIP, MAE, AUGREG)
                    ]
                )
                series = [("Intact, 12 pairs", INTACT, d)]
            else:
                d = np.stack([prof(intact(lm, cap, t), m) for lm in LMS for t in (DINO, CLIP)])
                s = np.stack([prof(shuffled(lm, cap, t), m) for lm in LMS for t in (DINO, CLIP)])
                f = np.array(
                    [
                        lex(rows1[(cap, v, t)], k)
                        for v in ("tfidf", "tfidf_sublinear")
                        for t in (DINO, CLIP)
                    ]
                )
                series = [("Intact, 6 pairs", INTACT, d), ("Tokens shuffled, 6 pairs", SHUFFLED, s)]
            for label, color, data in series:
                ax.fill_between(x, data.min(0), data.max(0), color=color, alpha=0.16, lw=0)
                ax.plot(x, data.mean(0), color=color, lw=2, marker="o", ms=4.5, label=label)
                table += [
                    [title, name, label, i, data[:, i].mean(), data[:, i].min(), data[:, i].max()]
                    for i in range(7)
                ]
            ax.axhspan(f.min(), f.max(), color=FLOOR, alpha=0.13, lw=0)
            ax.axhline(f.mean(), color=FLOOR, lw=1.2, ls="--", label="TF-IDF variants, no model")
            table.append([title, name, "TF-IDF variants", "", f.mean(), f.min(), f.max()])
            if r == 0:
                ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"{name}\nexcess over null")
            if r == 1:
                ax.set_xlabel("LM relative depth (0 = embedding layer)")
            ax.set_ylim(bottom=0)
            ax.tick_params(colors="#52514e", length=0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h,
        [x_.replace(", 6 pairs", "") for x_ in l],
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="lower center",
    )
    fig.suptitle(
        "Pythia-410M, open_llama_7b, bloomz-7b1. Lines: mean. Bands: min..max over pairs "
        "(WIT: DINOv2, CLIP; DOCCI: 4 ViTs). n = 10240.",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(p2 / f"fig_phase2.{ext}", dpi=200)
    with (p2 / "fig_phase2.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "measure", "series", "layer_index", "mean", "min", "max"])
        w.writerows(table)
    print(f"wrote fig_phase2, {len(table)} rows")


if __name__ == "__main__":
    main()
