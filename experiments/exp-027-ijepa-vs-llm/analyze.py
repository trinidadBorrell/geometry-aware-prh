"""exp-027: tables and figures from `d2_rows.jsonl` (score_d2.py) and `wit_rows.jsonl` (wit.py).

All numbers are excess over the null: score minus the exact null mean (D2) or minus the mean of the
null of the max (WIT-1024, Gröger's Algorithm 2). Writes `summary.md`, `fig_wit.png`, `fig_d2.png`.

    uv run python experiments/exp-027-ijepa-vs-llm/analyze.py --out results/exp-027-ijepa-vs-llm
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLORS = {  # reference categorical palette, fixed order
    "ijepa1k": "#2a78d6",
    "ijepa22k": "#eb6834",
    "mae_avg": "#1baf7a",
    "dino_avg": "#eda100",
}
LABEL = {
    "ijepa1k": "I-JEPA H/14, IN-1k",
    "ijepa22k": "I-JEPA H/14, IN-22k",
    "mae_avg": "MAE H/14, IN-1k",
    "dino_avg": "DINOv2 g/14",
    "mae_cls": "MAE H/14 (CLS)",
    "dino_cls": "DINOv2 g/14 (CLS)",
}
WIT_KEY = {
    "facebook/ijepa_vith14_1k": "ijepa1k",
    "facebook/ijepa_vith14_22k": "ijepa22k",
    "vit_huge_patch14_224.mae": "mae_avg",
    "vit_giant_patch14_dinov2.lvd142m": "dino_avg",
}
MAIN = ["ijepa1k", "ijepa22k", "mae_avg", "dino_avg"]
MEASURES = [("knn", "10"), ("knn", "100"), ("knn", "1000"), ("cka", None)]
MNAME = {"10": "k = 10", "100": "k = 100", "1000": "k = 1000", None: "CKA"}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def excess(r: dict, m: str, k: str | None) -> np.ndarray:
    """[L_set, L_vit] excess over the exact null."""
    if m == "cka":
        return np.array(r["cka"]) - np.array(r["cka_mu0"])
    return np.array(r["knn"][k]) - r["knn_mu0"][k]


def fmt(x: float) -> str:
    return f"{x:.3f}"


def d2_tables(rows: list[dict]) -> tuple[list[str], dict]:
    by = {(r["target"], r["set"]): r for r in rows}
    targets = [t for t in [*MAIN, "mae_cls", "dino_cls"] if any(k[0] == t for k in by)]
    llm_sets = sorted({r["set"] for r in rows if r["kind"] == "llm"})
    out, stats = [], {}
    measures = [(m, k) for m, k in MEASURES if m == "cka" or k in rows[0]["knn"]]
    for cap in ("orig", "gemini"):
        sets = [s for s in llm_sets if s.endswith(f"__{cap}")]
        out += [f"\n### D2, {cap} captions (n = 10240), mean over {len(sets)} LLMs\n"]
        head = "| ViT | " + " | ".join(f"{MNAME[k]} best | {MNAME[k]} last" for _, k in measures)
        head += " | Word floor k = 10 | Word floor CKA | TF-IDF floor k = 10 | TF-IDF floor CKA |"
        out += [head, "|" + "---|" * (2 * len(measures) + 5)]
        for t in targets:
            cells = []
            for m, k in measures:
                ex = [excess(by[(t, s)], m, k) for s in sets if (t, s) in by]
                best = np.mean([e.max() for e in ex])
                last = np.mean([e[-1, -1] for e in ex])
                stats[(t, cap, m, k)] = {
                    "best": [float(e.max()) for e in ex],
                    "last": [float(e[-1, -1]) for e in ex],
                    "sets": sets,
                }
                cells += [fmt(best), fmt(last)]
            for fl in ("word", "tfidf"):
                r = by.get((t, f"{cap}__{fl}"))
                for m, k in (("knn", "10"), ("cka", None)):
                    v = excess(r, m, k).max() if r else np.nan
                    stats[(t, cap, f"floor_{fl}", m, k)] = float(v)
                    cells.append(fmt(v))
            out.append(f"| {LABEL[t]} | " + " | ".join(cells) + " |")
    return out, stats


def d2_vitvit(rows: list[dict]) -> list[str]:
    names = {
        "facebook__ijepa_vith14_1k__avg": "ijepa1k",
        "facebook__ijepa_vith14_22k__avg": "ijepa22k",
        "vit_huge_patch14_224.mae__avg": "mae_avg",
        "vit_giant_patch14_dinov2.lvd142m__avg": "dino_avg",
        "vit_huge_patch14_224.mae__cls": "mae_cls",
        "vit_giant_patch14_dinov2.lvd142m__cls": "dino_cls",
    }
    vv = {(r["target"], names[r["set"]]): r for r in rows if r["kind"] == "vit"}
    keys = [t for t in [*MAIN, "mae_cls", "dino_cls"] if any(k[0] == t for k in vv)]
    out = [
        "\n### D2, ViT x ViT (best cell, excess): mutual 10-NN / CKA\n",
        "| | " + " | ".join(LABEL[k] for k in keys) + " |",
        "|" + "---|" * (len(keys) + 1),
    ]
    for a in keys:
        cells = []
        for b in keys:
            r = vv.get((a, b))
            cells.append(
                "-"
                if r is None
                else f"{excess(r, 'knn', '10').max():.3f} / {excess(r, 'cka', None).max():.3f}"
            )
        out.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
    return out


def wit_tables(rows: list[dict]) -> tuple[list[str], dict]:
    def ex(r, key):
        return r[key]["raw_score"] - r[key]["mu0"]

    llm = defaultdict(dict)  # (vit, pool) -> llm -> row
    floor = defaultdict(dict)
    for r in rows:
        (llm if r["kind"] == "llm" else floor)[(r["vit"], r["pool"])][r["name"]] = r
    keys = {"mknn": "mutual_knn_k10", "cka": "cka_lin"}
    summary = []
    for (vit, pool), d in llm.items():
        rec = {"vit": vit, "pool": pool, "n": len(d)}
        for short, key in keys.items():
            v = np.array([ex(r, key) for r in d.values()])
            rec[short] = (v.mean(), v.min(), v.max())
            rec[f"{short}_by_llm"] = {m: ex(r, key) for m, r in d.items()}
            rec[f"{short}_p"] = max(r[key]["p_value"] for r in d.values())
            for fl, fr in floor.get((vit, pool), {}).items():
                rec[f"{short}_{fl}"] = ex(fr, key)
        summary.append(rec)
    summary.sort(key=lambda r: -r["mknn"][0])
    out = [
        "\n### WIT-1024, every ViT mean-pooled: excess over the null of the max, mean (min-max) over LLMs\n",
        "| ViT | pool | LLMs | mutual 10-NN | CKA | word floor 10-NN | word floor CKA | "
        "10-NN above floor | CKA above floor | max p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        mk, ck = r["mknn"], r["cka"]
        fk, fc = r.get("mknn_word_tfidf_nostop", np.nan), r.get("cka_word_tfidf_nostop", np.nan)
        out.append(
            f"| {r['vit']} | {r['pool']} | {r['n']} | {mk[0]:.3f} ({mk[1]:.3f}-{mk[2]:.3f}) | "
            f"{ck[0]:.3f} ({ck[1]:.3f}-{ck[2]:.3f}) | {fk:.3f} | {fc:.3f} | {mk[0] - fk:.3f} | "
            f"{ck[0] - fc:.3f} | {max(r['mknn_p'], r['cka_p']):.3f} |"
        )
    from scipy.stats import spearmanr

    avg = [r for r in summary if r["pool"] == "avg"]
    out.append("")
    for short, name in (("mknn", "mutual 10-NN"), ("cka", "CKA")):
        x = [r[short][0] for r in avg]
        f = [r[f"{short}_word_tfidf_nostop"] for r in avg]
        rho = spearmanr(x, f).correlation
        out.append(
            f"- {name}: Spearman between LLM alignment and the word floor across the "
            f"{len(avg)} mean-pooled ViTs = {rho:.2f}"
        )
    return out, {(r["vit"], r["pool"]): r for r in summary}


def d2_depth(rows: list[dict]) -> list[str]:
    """Mean over LLMs (best LLM layer) at each stored ViT layer, and the word floor, k = 10."""
    by = {(r["target"], r["set"]): r for r in rows}
    out = []
    for cap in ("orig", "gemini"):
        sets = sorted({r["set"] for r in rows if r["kind"] == "llm" and r["set"].endswith(cap)})
        out += [
            f"\n### D2, {cap} captions, mutual 10-NN by ViT depth: LLMs (mean of best LLM layer) / word floor\n",
            "| ViT | " + " | ".join(f"{d:.2f}" for d in np.linspace(0, 1, 7)) + " |",
            "|" + "---|" * 8,
        ]
        for t in [*MAIN, "mae_cls", "dino_cls"]:
            ex = [excess(by[(t, s)], "knn", "10").max(0) for s in sets if (t, s) in by]
            fl = by.get((t, f"{cap}__word"))
            if not ex or fl is None:
                continue
            m, f = np.mean(ex, 0), excess(fl, "knn", "10")[0]
            out.append(
                f"| {LABEL[t]} | "
                + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in zip(m, f, strict=True))
                + " |"
            )
    return out


def count_wins(a: dict, b: dict) -> tuple[int, int, float]:
    common = sorted(set(a) & set(b))
    d = np.array([a[m] - b[m] for m in common])
    return int((d > 0).sum()), len(common), float(d.mean()) if len(d) else np.nan


def predictions(d2: dict, wit: dict) -> list[str]:
    out = [
        "\n## Predictions\n",
        "| | Test | Result | Verdict |",
        "|---|---|---|---|",
    ]

    def w(vit, short):
        return wit.get((vit, "avg"), {}).get(f"{short}_by_llm", {})

    ij1, ij22 = "facebook/ijepa_vith14_1k", "facebook/ijepa_vith14_22k"
    mae, dino = "vit_huge_patch14_224.mae", "vit_giant_patch14_dinov2.lvd142m"

    def d2_cells(a, b, m, k):
        wins = tot = 0
        diffs = []
        for cap in ("orig", "gemini"):
            sa, sb = d2.get((a, cap, m, k)), d2.get((b, cap, m, k))
            if not (sa and sb):
                continue
            for x, y in zip(sa["best"], sb["best"], strict=True):
                wins += x > y
                tot += 1
                diffs.append(x - y)
        return wins, tot, float(np.mean(diffs)) if diffs else np.nan

    for pid, short, m, k, label in (
        ("P1", "mknn", "knn", "10", "mutual 10-NN"),
        ("P2", "cka", "cka", None, "CKA"),
    ):
        wn, wt, wd = count_wins(w(ij1, short), w(mae, short))
        dn, dt, dd = d2_cells("ijepa1k", "mae_avg", m, k)
        ok = wn >= 8 and dn >= 7
        out.append(
            f"| {pid} | I-JEPA 1k > MAE, {label} | WIT {wn}/{wt} LLMs (mean diff {wd:+.3f}); "
            f"D2 {dn}/{dt} cells (mean diff {dd:+.3f}) | {'holds' if ok else '**fails**'} |"
        )
    n, t, dmean = count_wins(w(ij1, "mknn"), w(dino, "mknn"))
    out.append(
        f"| P3 | I-JEPA 1k < DINOv2-g, mutual 10-NN | I-JEPA above DINOv2 for {n}/{t} LLMs "
        f"(mean diff {dmean:+.3f}) | {'holds' if n < 6 else '**fails**'} |"
    )
    n, t, dmean = count_wins(w(ij1, "mknn"), w(ij22, "mknn"))
    out.append(
        f"| P4 | I-JEPA 22k >= 1k, mutual 10-NN | 1k above 22k for {n}/{t} LLMs "
        f"(mean diff 1k - 22k {dmean:+.3f}) | {'holds' if n < 6 else '**fails**'} |"
    )
    sa, sb = d2.get(("ijepa1k", "gemini", "knn", "10")), d2.get(("mae_avg", "gemini", "knn", "10"))
    if sa and sb:
        llm_gap = np.mean(sa["best"]) - np.mean(sb["best"])
        fl = [
            d2[("ijepa1k", "gemini", f"floor_{f}", "knn", "10")]
            - d2[("mae_avg", "gemini", f"floor_{f}", "knn", "10")]
            for f in ("word", "tfidf")
        ]
        ok = llm_gap > max(fl)
        out.append(
            f"| P5 | LLM gap > floor gap, D2 Gemini k = 10 | LLM gap {llm_gap:+.3f}; floor gaps "
            f"word {fl[0]:+.3f}, TF-IDF {fl[1]:+.3f} | {'holds' if ok else '**fails**'} |"
        )
    hf = wit.get((ij1, "avg-hfproc"))
    tm = wit.get((ij1, "avg"))
    if hf and tm:
        out.append(
            f"| check | HF processor vs timm transform, I-JEPA 1k | mutual 10-NN {hf['mknn'][0]:.3f} vs "
            f"{tm['mknn'][0]:.3f}; CKA {hf['cka'][0]:.3f} vs {tm['cka'][0]:.3f} | - |"
        )
    return out


def fig_wit(wit: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    recs = [r for (v, p), r in wit.items() if p == "avg"]
    recs.sort(key=lambda r: r["mknn"][0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(recs) + 1.4), sharey=True)
    for ax, short, title in zip(axes, ("mknn", "cka"), ("Mutual 10-NN", "Linear CKA"), strict=True):
        for i, r in enumerate(recs):
            key = WIT_KEY.get(r["vit"])
            c = COLORS.get(key, "#9a9890")
            mean, lo, hi = r[short]
            ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.plot(mean, i, "o", ms=8, color=c, mec="#fcfcfb", mew=2)
            fl = r.get(f"{short}_word_tfidf_nostop")
            if fl is not None:
                ax.plot(fl, i, "|", ms=10, mew=2, color="#52514e")
        ax.set_title(f"{title}, excess over the null", loc="left", fontsize=11)
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(recs)))
    axes[0].set_yticklabels(
        [LABEL.get(WIT_KEY.get(r["vit"], ""), r["vit"]) for r in recs], fontsize=9
    )
    fig.suptitle(
        "WIT-1024: 10 PRH LLMs x mean-pooled ViTs (dot = mean, bar = min..max over LLMs, "
        "tick = word TF-IDF floor)",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_scatter(wit: dict, path: Path) -> None:
    """LLM alignment against the ViT's own word floor, every mean-pooled ViT, WIT-1024."""
    import matplotlib.pyplot as plt

    recs = [r for (v, p), r in wit.items() if p == "avg"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, short, title in zip(axes, ("mknn", "cka"), ("Mutual 10-NN", "Linear CKA"), strict=True):
        f = np.array([r[f"{short}_word_tfidf_nostop"] for r in recs])
        y = np.array([r[short][0] for r in recs])
        slope, icpt = np.polyfit(f, y, 1)
        r2 = 1 - ((y - (slope * f + icpt)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        xs = np.linspace(f.min(), f.max(), 10)
        ax.plot(xs, slope * xs + icpt, color="#9a9890", lw=1.2, ls="--")
        for r, fx, yy in zip(recs, f, y, strict=True):
            key = WIT_KEY.get(r["vit"])
            lo, hi = r[short][1], r[short][2]
            c = COLORS.get(key, "#9a9890")
            ax.plot([fx, fx], [lo, hi], color=c, lw=1.5, alpha=0.6)
            ax.plot(fx, yy, "o", ms=8, color=c, mec="#fcfcfb", mew=2, zorder=3)
            if key:
                ax.annotate(
                    LABEL[key],
                    (fx, yy),
                    xytext=(6, -3),
                    textcoords="offset points",
                    fontsize=8,
                    color="#52514e",
                )
        ax.set_title(f"{title}: linear fit R² = {r2:.2f}", loc="left", fontsize=10)
        ax.set_xlabel("word TF-IDF floor (excess)")
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("LLM alignment (excess), mean over 10 LLMs")
    fig.suptitle(
        "WIT-1024, one dot per mean-pooled ViT (bar = min..max over 10 LLMs; gray = PRH pool)",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_d2(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    by = {(r["target"], r["set"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, cap in enumerate(("orig", "gemini")):
        sets = sorted({r["set"] for r in rows if r["kind"] == "llm" and r["set"].endswith(cap)})
        for row, (m, k) in enumerate((("knn", "10"), ("cka", None))):
            ax = axes[row, col]
            for t in MAIN:
                ex = [excess(by[(t, s)], m, k).max(0) for s in sets if (t, s) in by]
                if not ex:
                    continue
                ex = np.array(ex)  # [LLMs, L_vit], best over LLM layers
                x = np.linspace(0, 1, ex.shape[1])
                ax.fill_between(x, ex.min(0), ex.max(0), color=COLORS[t], alpha=0.15, lw=0)
                ax.plot(x, ex.mean(0), color=COLORS[t], lw=2, label=LABEL[t])
                fl = by.get((t, f"{cap}__word"))
                if fl is not None:
                    ax.plot(x, excess(fl, m, k)[0], color=COLORS[t], lw=1.2, ls="--")
            ax.set_title(f"{cap} captions, {MNAME[k]}", loc="left", fontsize=10)
            ax.grid(color="#e6e5e0", lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == 1:
                ax.set_xlabel("ViT relative depth")
            if col == 0:
                ax.set_ylabel("excess over the null")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "D2 (n = 10240): best LLM layer at each ViT layer; line = mean, band = min..max over 4 LLMs; "
        "dashed = word TF-IDF floor",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    d2_rows = load(args.out / "d2_rows.jsonl")
    wit_rows = load(args.out / "wit_rows.jsonl")
    lines = ["# exp-027 summary", ""]
    d2, wit = {}, {}
    if wit_rows:
        t, wit = wit_tables(wit_rows)
        lines += t
    if d2_rows:
        t, d2 = d2_tables(d2_rows)
        lines += t + d2_depth(d2_rows) + d2_vitvit(d2_rows)
        fig_d2(d2_rows, args.out / "fig_d2.png")
    lines += predictions(d2, wit)
    if wit:
        fig_wit(wit, args.out / "fig_wit.png")
        fig_scatter(wit, args.out / "fig_floor_scatter.png")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
