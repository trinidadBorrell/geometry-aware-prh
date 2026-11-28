"""exp-031: does PRH's capability trend survive shuffling the captions' words?

Reads the intact calibrated pairs (`prh_val_calibrated`), the shuffled ones (`<out>/shufw*/`) and
the model info (`prh_val/cross_modal.json`: params, layers, bpb on intact captions). Every score is
excess over chance: raw max over layer pairs minus the mean of the null of the max (Gröger's
Algorithm 2). Intact and shuffled trends are compared as trends; no score is subtracted from
another. Writes `summary.md`, `trends.json` and `fig_trend.{png,pdf}` to `<out>`.

    uv run python experiments/exp-031-shuffled-capability/analyze.py --out <dir> \\
        --stored <prh_val> --stored-cal <prh_val_calibrated>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEASURES = {"mutual_knn_k10": "Mutual kNN, k = 10", "cka_lin": "Linear CKA"}
STORED_KEY = {"mutual_knn_k10": "mutual_knn", "cka_lin": "cka_lin"}  # prh_val_calibrated names
FAMILIES = {"BLOOMZ": "bigscience/bloomz", "OpenLLaMA": "openlm-research/open_llama"}
COLORS = {"I": "#2a78d6", "Ist": "#7aa7e0", "S0": "#eb6834", "S1": "#b8462a"}
LABELS = {
    "I": "intact",
    "Ist": "intact, stored (other GPU)",
    "S0": "words shuffled (seed 0)",
    "S1": "words shuffled (seed 1)",
}

# Reading rules fixed in the README before the run.
P1_MAX_SHUF_RHO = 0.6
P2_MIN_WIN_SHARE = 160 / 170  # README: at least 160 of PRH's 170 pairs
P3_MAX_RHO_DIFF = 0.15
SEED_MAX_RHO_DIFF = 0.1


def load_pairs(path: Path, key_map: dict[str, str] | None = None) -> dict:
    """(llm, lvm) -> {measure: result} from a calibrated_pairs.jsonl."""
    out = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        res = rec.get("results", rec)
        out.setdefault((rec["llm"], rec["lvm"]), {})
        for m in MEASURES:
            src = (key_map or {}).get(m, m)
            if src in res:
                out[(rec["llm"], rec["lvm"])][m] = res[src]
    return out


def cube(pairs: dict, llms, lvms, measure: str, field: str) -> np.ndarray:
    out = np.full((len(llms), len(lvms)), np.nan)
    for i, lm in enumerate(llms):
        for j, v in enumerate(lvms):
            r = pairs.get((lm, v), {}).get(measure)
            if r is not None:
                out[i, j] = r[field] if field != "excess" else r["raw_score"] - r["mu0"]
    return out


def rho_per_vit(cap: np.ndarray, ex: np.ndarray, rows=None) -> np.ndarray:
    """Spearman(capability, excess) across LLMs (optionally a subset of rows), one per ViT."""
    rows = np.arange(len(cap)) if rows is None else np.asarray(rows)
    return np.array([spearmanr(cap[rows], ex[rows, j]).statistic for j in range(ex.shape[1])])


def summarise(rho: np.ndarray) -> dict:
    return {"median": float(np.median(rho)), "min": float(rho.min()), "max": float(rho.max())}


def fmt_rho(s: dict) -> str:
    return f"{s['median']:.2f} ({s['min']:.2f} to {s['max']:.2f})"


def table(head: list[str], rows: list[list]) -> list[str]:
    def f(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(f(x) for x in r) + " |" for r in rows],
        "",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stored", type=Path, required=True, help="prh_val (cross_modal.json)")
    p.add_argument("--stored-cal", type=Path, required=True, help="prh_val_calibrated")
    p.add_argument("--seeds", default="0,1")
    a = p.parse_args()

    meta = json.loads((a.stored / "cross_modal.json").read_text())
    info = meta["models"]
    stored = load_pairs(a.stored_cal / "calibrated_pairs.jsonl", STORED_KEY)
    reext = a.out / "reext" / "calibrated_pairs.jsonl"
    # Primary intact condition: re-extracted on the same GPU as the shuffles when available;
    # the stored PRH scores (extracted on another GPU) are then a reproducibility check.
    conds = {"I": load_pairs(reext), "Ist": stored} if reext.exists() else {"I": stored}
    for s in a.seeds.split(","):
        path = a.out / f"shufw{s}" / "calibrated_pairs.jsonl"
        if path.exists():
            conds[f"S{s}"] = load_pairs(path)
    shuf = [c for c in conds if c.startswith("S")]
    if not shuf:
        raise SystemExit(f"no shuffled calibrated pairs under {a.out}")

    have = set.intersection(*(set(lm for lm, _ in c) for c in conds.values()))
    llms = sorted((m for m in meta["llms"] if m in have), key=lambda m: -info[m]["bpb"])
    lvms = [v for v in meta["lvms"] if all((llms[0], v) in c for c in conds.values())]
    cap = np.array([1 - info[m]["bpb"] for m in llms])
    print(f"{len(llms)} LLMs x {len(lvms)} ViTs; conditions {list(conds)}")

    ex = {(c, m): cube(conds[c], llms, lvms, m, "excess") for c in conds for m in MEASURES}
    res: dict = {
        "llms": llms,
        "lvms": lvms,
        "capability": cap.tolist(),
        "trends": {},
        "within": {},
        "wins": {},
        "levels": {},
        "depth": {},
    }

    lines = [
        "# exp-031 summary: PRH capability trend, intact vs word-shuffled captions",
        "",
        f"WIT-1024, {len(llms)} LLMs x {len(lvms)} ViTs. Excess = raw max over layer pairs minus "
        "the mean of the null of the max (200 permutations). Capability = 1 - bpb on the intact "
        "captions. Spearman across LLMs, one value per ViT: median (min to max) over ViTs.",
        "",
        "## 1. Capability trend",
        "",
    ]
    rows = []
    for m, name in MEASURES.items():
        for c in conds:
            s = summarise(rho_per_vit(cap, ex[(c, m)]))
            res["trends"][f"{c}/{m}"] = s
            rows.append([name, LABELS[c], fmt_rho(s)])
    lines += table(["measure", "captions", "Spearman(1 - bpb, excess)"], rows)

    lines += ["Within families (sizes ordered by bpb):", ""]
    rows = []
    for fam, prefix in FAMILIES.items():
        idx = [i for i, lm in enumerate(llms) if lm.startswith(prefix)]
        if len(idx) < 3:
            continue
        for m, name in MEASURES.items():
            for c in conds:
                s = summarise(rho_per_vit(cap, ex[(c, m)], idx))
                res["within"][f"{fam}/{c}/{m}"] = s
                rows.append([fam, len(idx), name, LABELS[c], fmt_rho(s)])
    lines += table(["family", "LLMs", "measure", "captions", "Spearman"], rows)

    if "Ist" in conds:
        rows = []
        for m, name in MEASURES.items():
            d = np.abs(ex[("I", m)] - ex[("Ist", m)])
            res.setdefault("reproduction", {})[m] = {
                "median_abs_diff": float(np.median(d)),
                "max_abs_diff": float(d.max()),
            }
            rows.append([name, float(np.median(d)), float(d.max())])
        lines += [
            "Reproduction: intact re-extracted on this GPU vs the stored PRH features "
            "(another GPU), |excess difference| over pairs:",
            "",
            *table(["measure", "median", "max"], rows),
        ]

    lines += ["## 2. Levels", ""]
    rows = []
    for m, name in MEASURES.items():
        i_ex = ex[("I", m)]
        for c in shuf:
            wins = int((i_ex > ex[(c, m)]).sum())
            res["wins"][f"{c}/{m}"] = wins
            rows.append(
                [
                    name,
                    LABELS[c],
                    f"{wins}/{i_ex.size}",
                    float(np.median(i_ex)),
                    float(np.median(ex[(c, m)])),
                ]
            )
    lines += table(
        [
            "measure",
            "shuffled",
            "pairs with intact > shuffled",
            "median excess, intact",
            "median excess, shuffled",
        ],
        rows,
    )

    rows = []
    for i, lm in enumerate(llms):
        row = [lm.split("/")[-1], float(cap[i])]
        for m in MEASURES:
            row += [float(ex[(c, m)][i].mean()) for c in conds]
        rows.append(row)
        res["levels"][lm] = {
            f"{c}/{m}": float(ex[(c, m)][i].mean()) for c in conds for m in MEASURES
        }
    head = ["LLM", "1 - bpb"] + [
        f"{'mKNN' if m.startswith('mutual') else 'CKA'} {c}" for m in MEASURES for c in conds
    ]
    lines += ["Mean excess over ViTs, per LLM:", "", *table(head, rows)]

    lines += ["## 3. Depth of the best LLM layer (relative), median over ViTs", ""]
    rows = []
    for lm in llms:
        L = info[lm]["num_layers"]
        row = [lm.split("/")[-1]]
        for m in MEASURES:
            for c in conds:
                d = [conds[c][(lm, v)][m]["best_indices"][1] / (L - 1) for v in lvms]
                res["depth"][f"{lm}/{c}/{m}"] = float(np.median(d))
                row.append(float(np.median(d)))
        rows.append(row)
    lines += table(
        ["LLM"]
        + [f"{'mKNN' if m.startswith('mutual') else 'CKA'} {c}" for m in MEASURES for c in conds],
        rows,
    )

    # Predictions and checks (README).
    t = res["trends"]
    verdicts = []
    for c in shuf:
        r_i, r_s = t["I/mutual_knn_k10"]["median"], t[f"{c}/mutual_knn_k10"]["median"]
        verdicts.append(
            [
                "P1",
                c,
                f"mKNN Spearman {r_s:.2f} (intact {r_i:.2f}) <= {P1_MAX_SHUF_RHO}",
                "holds" if r_s <= P1_MAX_SHUF_RHO else "fails",
            ]
        )
        w = res["wins"][f"{c}/mutual_knn_k10"]
        verdicts.append(
            [
                "P2",
                c,
                f"intact > shuffled in {w}/{len(llms) * len(lvms)} pairs",
                "holds" if w >= P2_MIN_WIN_SHARE * len(llms) * len(lvms) else "fails",
            ]
        )
        d = abs(t[f"{c}/cka_lin"]["median"] - t["I/cka_lin"]["median"])
        verdicts.append(
            [
                "P3",
                c,
                f"CKA Spearman differs by {d:.2f} (limit {P3_MAX_RHO_DIFF})",
                "holds" if d <= P3_MAX_RHO_DIFF else "fails",
            ]
        )
    if {"S0", "S1"} <= set(conds):
        for m in MEASURES:
            d = abs(t[f"S0/{m}"]["median"] - t[f"S1/{m}"]["median"])
            ok = d <= SEED_MAX_RHO_DIFF
            verdicts.append(
                [
                    "seeds",
                    m,
                    f"S0 vs S1 median Spearman differ by {d:.2f}",
                    "ok" if ok else "TOO NOISY",
                ]
            )
    res["verdicts"] = verdicts
    lines += ["## Predictions and checks (fixed before the run)", ""]
    lines += table(["", "condition", "result", "verdict"], verdicts)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.md").write_text("\n".join(lines))
    (a.out / "trends.json").write_text(json.dumps(res, indent=2))
    plot(a.out, llms, cap, ex, conds)
    print("\n".join(lines))


def plot(out: Path, llms, cap, ex, conds) -> None:
    """Per LLM: mean excess over ViTs against 1 - bpb, band = min..max over ViTs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    order = np.argsort(cap)
    for ax, (m, name) in zip(axes, MEASURES.items(), strict=True):
        for c in conds:
            e = ex[(c, m)][order]
            ax.fill_between(cap[order], e.min(1), e.max(1), color=COLORS[c], alpha=0.15, lw=0)
            ax.plot(cap[order], e.mean(1), "o-", color=COLORS[c], ms=3.5, lw=1.4, label=LABELS[c])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("1 - bits per byte (intact captions)")
        ax.set_ylabel("excess over chance")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("WIT-1024: mean over ViTs, band min..max over ViTs", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_trend.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
