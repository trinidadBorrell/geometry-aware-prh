"""exp-024: image-type structure over training. Tables and a figure from `typed.jsonl`.

Series: 4 Pythia runs (160M and 410M, two seeds each), each scored on DINOv2 and on CLIP with the
other ViT's types. Bands are min..max over those 8 curves. t50 is the first step reaching halfway from
step 0 to the final checkpoint, linearly interpolated, and is given only when the change exceeds
`MIN_CHANGE`. Writes `typed_summary.md`, `fig_types.*`.

    uv run python experiments/exp-024-emergence-words/analyze_typed.py --out <dir>
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

PY = ["pythia-160m", "pythia-410m", "pythia-160m-seed1", "pythia-410m-seed1"]
LAYERS = {0: "Layer 0", 3: "Middle", 6: "Last"}
COLORS = {0: "#eb6834", 3: "#1baf7a", 6: "#2a78d6"}
MEASURES = {
    "between_share": "Between-type share of variance",
    "purity": "Neighbours in the same type",
    "within_knn": "Mutual kNN within type",
    "cka_between": "CKA, between-type part",
    "cka_within": "CKA, within-type part",
}
TABLE = [*MEASURES, "knn_same", "knn_other", "knn", "cka"]
WARMUP = 1430
MIN_CHANGE = 0.02


def t50(steps: np.ndarray, y: np.ndarray) -> float:
    half = y[0] + (y[-1] - y[0]) / 2
    sign = np.sign(y[-1] - y[0])
    for i in range(1, len(y)):
        if sign * (y[i] - half) >= 0:
            frac = (half - y[i - 1]) / (y[i] - y[i - 1])
            return float(steps[i - 1] + frac * (steps[i] - steps[i - 1]))
    return float("nan")


def table(head, rows) -> list[str]:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows],
        "",
    ]


def span(a) -> str:
    a = np.asarray(a, dtype=float)
    return f"{np.nanmedian(a):.3f} ({np.nanmin(a):.3f}-{np.nanmax(a):.3f})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in (a.out / "typed.jsonl").read_text().splitlines()]
    nulls = []
    path = a.out / "typed_null.jsonl"
    if path.exists():
        nulls = [json.loads(x) for x in path.read_text().splitlines()]
    data = {}
    for r in rows:
        data[
            (r["series"], r["caption"], r["step"], r["layer_index"], r["target"], r["k_types"])
        ] = r

    def curves(caption, layer, k, field):
        """[8, steps] and the steps."""
        steps = sorted({r["step"] for r in rows if r["caption"] == caption and r["step"] >= 0})
        out = [
            [data[(s, caption, st, layer, t, k)][field] for st in steps]
            for s in PY
            for t in ("dino", "clip")
        ]
        return np.array(out), np.array(steps, dtype=float)

    lines = [
        "# exp-024 image types",
        "",
        "n = 10240. Median (min-max) over 4 Pythia runs x 2 ViTs. Each ViT uses the other's types.",
        "Everything fast happens inside the 1430-step warmup. Original captions have 6 checkpoints.",
        "",
    ]
    for caption in ("gemini", "orig"):
        for k in (5, 20, 100):
            t = []
            for field in TABLE:
                for layer, name in LAYERS.items():
                    y, steps = curves(caption, layer, k, field)
                    change = np.abs(y[:, -1] - y[:, 0])
                    t50s = [t50(steps, c) for c in y]
                    t.append(
                        [
                            field,
                            name,
                            span(y[:, 0]),
                            span(y[:, -1]),
                            span(y[:, -1] - y[:, 0]),
                            f"{np.nanmedian(t50s):.0f} ({np.nanmin(t50s):.0f}-{np.nanmax(t50s):.0f})"
                            if np.median(change) >= MIN_CHANGE
                            else "flat",
                        ]
                    )
            lines += [f"## {caption}, K = {k}", ""]
            lines += table(["measure", "layer", "step 0", "final", "change", "t50, steps"], t)

    # References: pairing broken, word kernels, finished larger LMs
    if nulls:
        t = []
        keys = sorted({(r["set"], r["layer_index"]) for r in nulls})
        for caption in ("gemini", "orig"):
            for k in (5, 20, 100):
                for field in MEASURES:
                    for st, li in ((0, 0), (143000, 0), (143000, 6)):
                        name = f"EleutherAI__pythia-410m@step{st}__{caption}"
                        if (name, li) not in keys:
                            continue
                        for tgt in ("dino", "clip"):
                            nv = [
                                r[field]
                                for r in nulls
                                if r["set"] == name
                                and r["layer_index"] == li
                                and r["target"] == tgt
                                and r["k_types"] == k
                            ]
                            real = data[("pythia-410m", caption, st, li, tgt, k)][field]
                            t.append(
                                [
                                    caption,
                                    k,
                                    field,
                                    f"step {st}, L{li}",
                                    tgt,
                                    real,
                                    float(np.mean(nv)),
                                    float(np.max(nv)),
                                ]
                            )
        lines += ["## Pairing broken, 10 permutations, Pythia-410M", ""]
        lines += table(
            ["caption", "K", "measure", "cell", "ViT", "real", "null mean", "null max"], t
        )
    refs = sorted({r["series"] for r in rows if r["step"] == -1})
    t = []
    for caption in ("gemini", "orig"):
        for k in (5, 20, 100):
            for field in MEASURES:
                for s in refs:
                    layers = sorted(
                        {
                            r["layer_index"]
                            for r in rows
                            if r["series"] == s and r["caption"] == caption
                        }
                    )
                    for li in layers:
                        vals = [
                            data[(s, caption, -1, li, tt, k)][field]
                            for tt in ("dino", "clip")
                            if (s, caption, -1, li, tt, k) in data
                        ]
                        if vals:
                            t.append([caption, k, field, s, li, *map(float, vals)])
    lines += ["## References: word kernels and finished LMs", ""]
    lines += table(["caption", "K", "measure", "set", "layer", "DINOv2", "CLIP"], t)

    # Excess over the broken pairing. Larger LMs have no null of their own, so they use Pythia-410M's.
    def null_mean(name, li, tgt, k, field):
        v = [
            r[field]
            for r in nulls
            if r["set"] == name
            and r["layer_index"] == li
            and r["target"] == tgt
            and r["k_types"] == k
        ]
        return float(np.mean(v))

    if nulls:
        fields = ("between_share", "purity", "within_knn", "cka_between", "cka_within")
        for caption in ("gemini", "orig"):
            for k in (5, 20, 100):
                t = []
                pn = f"EleutherAI__pythia-410m@step143000__{caption}"
                p0 = pn.replace("143000", "0")
                cells = [
                    ("TF-IDF", "tfidf", -1, 0, f"tfidf__{caption}", 0),
                    ("Pythia-410M step 0, L0", "pythia-410m", 0, 0, p0, 0),
                    ("Pythia-410M step 0, last", "pythia-410m", 0, 6, p0, 6),
                    ("Pythia-410M final, L0", "pythia-410m", 143000, 0, pn, 0),
                    ("Pythia-410M final, last", "pythia-410m", 143000, 6, pn, 6),
                ]
                cells += [(f"{s}, L0", s, -1, 0, pn, 0) for s in refs if s != "tfidf"]
                cells += [(f"{s}, last", s, -1, 6, pn, 6) for s in refs if s != "tfidf"]
                for label, s, st, li, nname, nli in cells:
                    row = [label]
                    for tgt in ("dino", "clip"):
                        for f in fields:
                            real = data.get((s, caption, st, li, tgt, k))
                            row.append(real[f] - null_mean(nname, nli, tgt, k, f) if real else "-")
                    t.append(row)
                lines += [f"## Excess over broken pairing: {caption}, K = {k}", ""]
                lines += table(
                    ["cell", *[f"{v} {x}" for v in ("DINOv2", "CLIP") for x in fields]], t
                )
    (a.out / "typed_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Figure: K = 20
    plt.rcParams.update(
        {"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    k = 20
    fig, axes = plt.subplots(2, len(MEASURES), figsize=(13.5, 5.6), sharex="row")
    csvrows = []
    for r, caption in enumerate(("gemini", "orig")):
        for c, (field, name) in enumerate(MEASURES.items()):
            ax = axes[r, c]
            for layer, lname in LAYERS.items():
                y, steps = curves(caption, layer, k, field)
                x = steps + 8  # so step 0 sits on a log axis
                ax.fill_between(x, y.min(0), y.max(0), color=COLORS[layer], alpha=0.16, lw=0)
                ax.plot(x, y.mean(0), color=COLORS[layer], lw=2, marker="o", ms=2.5, label=lname)
                csvrows += [
                    [caption, field, lname, int(st), m, lo, hi]
                    for st, m, lo, hi in zip(steps, y.mean(0), y.min(0), y.max(0), strict=True)
                ]
            vals = [
                data[("tfidf", caption, -1, 0, tt, k)][field]
                for tt in ("dino", "clip")
                if ("tfidf", caption, -1, 0, tt, k) in data
            ]
            if vals:
                ax.axhspan(min(vals), max(vals), color="#52514e", alpha=0.18, lw=0)
                ax.axhline(np.mean(vals), color="#52514e", lw=1, ls="--", label="TF-IDF, no model")
            nv = [
                x[field]
                for x in nulls
                if x["set"] == f"EleutherAI__pythia-410m@step143000__{caption}"
                and x["layer_index"] == 6
                and x["k_types"] == k
            ]
            if nv:
                ax.axhline(np.mean(nv), color="#52514e", lw=1, ls=":", label="Pairing broken")
            if field == "purity":
                ax.axhline(
                    data[("pythia-410m", caption, 0, 0, "dino", k)]["purity_base"],
                    color="#52514e",
                    lw=0.8,
                    ls="-.",
                    label="Base rate",
                )
            ax.axvspan(8, WARMUP + 8, color="#f0efea", lw=0, zorder=0)
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            ax.tick_params(colors="#52514e", length=0)
            if r == 0:
                ax.set_title(name, fontsize=8.5, loc="left")
            else:
                ax.set_xlabel("training step + 8")
        axes[r, 0].set_ylabel("Gemini captions" if r == 0 else "Original captions")
    h = {}
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels(), strict=True):
            h[ll] = hh
    fig.legend(list(h.values()), list(h), frameon=False, fontsize=8, ncol=6, loc="lower center")
    fig.suptitle(
        "K = 20 image types. Bands: 4 Pythia runs x 2 ViTs. Shaded: warmup.",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"fig_types.{ext}", dpi=200)
    with (a.out / "fig_types.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caption", "measure", "layer", "step", "mean", "min", "max"])
        w.writerows(csvrows)
    print("done")


if __name__ == "__main__":
    main()
