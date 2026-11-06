"""exp-022: alignment across LM layers, mean and min..max over LM x ViT pairs, with lexical floors.

Uses stored grids only (`grids_full.json`, `rows.jsonl`). Writes `fig_layers.{png,pdf}` and the
plotted values to `fig_layers.csv`.

    uv run python experiments/exp-022-lexical-validation/fig_layers.py --out <results dir>
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
VITS = [DINO, CLIP, MAE, AUGREG]
GEMINI, ORIG, RANDOM = "#2a78d6", "#eb6834", "#52514e"
MEASURES = [("knn10", "10", "Mutual kNN, k = 10"), ("cka", None, "Linear CKA")]
FINAL = "step143000"
PY = ["pythia-160m", "pythia-160m-seed1", "pythia-410m", "pythia-410m-seed1"]


def excess_grid(g: dict, m: str) -> np.ndarray:
    return np.array(g[m]) - np.array(g[f"{m}_mu0"])


def profile(grids: list[dict], m: str, fixed: bool) -> np.ndarray:
    """[pairs, LM layers]: excess at each LM layer, at the best or the last target layer."""
    e = [excess_grid(g, m) for g in grids]
    return np.stack([x[:, -1] if fixed else x.max(1) for x in e])


def floors(rows: dict, caption: str, vits: list[str], k: str | None, fixed: bool) -> np.ndarray:
    out = []
    for t in vits:
        r = rows.get(("lexical", caption, "tfidf", t)) or rows.get(("targets", caption, "tfidf", t))
        s = (
            np.array(r["cka"]) - np.array(r["cka_mu0"])
            if k is None
            else np.array(r["knn"][k]) - r["knn_mu0"][k]
        )
        out.append(s[-1, 0] if fixed else s.max())
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    grids = json.loads((args.out / "grids_full.json").read_text())
    rows = {}
    for line in (args.out / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["stage"], r["caption"], r["variant"], r["target"])] = r
    b3 = {c: [g for g in grids if g["group"] == f"B3_{c}"] for c in ("gemini", "orig")}
    py = {
        rev: [
            g
            for g in grids
            if g["id"].startswith(tuple(f"LV__EleutherAI__{s}@{rev}__gemini__" for s in PY))
            and (DINO in g["id"] or CLIP in g["id"])
        ]
        for rev in ("step0", FINAL)
    }
    assert len(b3["gemini"]) == len(b3["orig"]) == 16 and len(py[FINAL]) == 8 == len(py["step0"])

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex=True, sharey="row")
    x = np.linspace(0, 1, 7)
    table = []

    def series(ax, data, color, label):
        mean = data.mean(0)
        ax.fill_between(x, data.min(0), data.max(0), color=color, alpha=0.16, lw=0)
        ax.plot(x, mean, color=color, lw=2, marker="o", ms=4.5, label=label)
        return mean

    def floor(ax, values, color, label):
        ax.axhspan(values.min(), values.max(), color=color, alpha=0.12, lw=0)
        ax.axhline(values.mean(), color=color, lw=1.2, ls="--", label=label)

    columns = [
        ("exp-016 LMs x 4 ViTs (16 pairs), best ViT layer", False, "b3"),
        ("exp-016 LMs x 4 ViTs (16 pairs), last ViT layer", True, "b3"),
        ("Pythia x DINOv2, CLIP (8 pairs), best ViT layer", False, "py"),
    ]
    for r, (m, k, name) in enumerate(MEASURES):
        for c, (title, fixed, kind) in enumerate(columns):
            ax = axes[r, c]
            if kind == "b3":
                for cap, color, lab in ((("gemini"), GEMINI, "Gemini"), ("orig", ORIG, "Original")):
                    d = profile(b3[cap], m, fixed)
                    series(ax, d, color, f"{lab} captions")
                    f = floors(rows, cap, VITS, k, fixed)
                    floor(ax, f, color, f"TF-IDF, {lab}, no model")
                    table += [[title, name, f"{lab} deep", i, *v] for i, v in enumerate(_stats(d))]
                    table.append([title, name, f"{lab} TF-IDF", "", f.mean(), f.min(), f.max()])
            else:
                d = profile(py[FINAL], m, fixed)
                series(ax, d, GEMINI, "Final checkpoint, Gemini")
                d0 = profile(py["step0"], m, fixed)
                series(ax, d0, RANDOM, "Random init, Gemini")
                f = floors(rows, "gemini", [DINO, CLIP], k, fixed)
                floor(ax, f, GEMINI, "TF-IDF, Gemini, no model")
                table += [[title, name, "final", i, *v] for i, v in enumerate(_stats(d))]
                table += [[title, name, "random init", i, *v] for i, v in enumerate(_stats(d0))]
                table.append([title, name, "TF-IDF", "", f.mean(), f.min(), f.max()])
            if r == 0:
                ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"{name}\nexcess over null")
            if r == 1:
                ax.set_xlabel("LM relative depth (0 = embedding layer)")
            ax.set_ylim(bottom=0)
            ax.tick_params(colors="#52514e", length=0)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        [*h0, *h2[:2]], [*l0, *l2[:2]], frameon=False, fontsize=8, ncol=4, loc="lower center"
    )
    fig.suptitle(
        "Alignment across LM layers. Lines: mean over pairs. Bands: min..max over pairs "
        "(floors: over ViTs). n = 10240.",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig_layers.{ext}", dpi=200)
    with (args.out / "fig_layers.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "measure", "series", "layer_index", "mean", "min", "max"])
        w.writerows(table)
    print(f"wrote fig_layers, {len(table)} rows")


def _stats(d: np.ndarray) -> list[tuple[float, float, float]]:
    return list(zip(d.mean(0).tolist(), d.min(0).tolist(), d.max(0).tolist(), strict=True))


if __name__ == "__main__":
    main()
