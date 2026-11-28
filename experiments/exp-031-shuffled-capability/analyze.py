"""exp-031: does PRH's capability trend survive shuffling the captions' words?

Reads, under `--out`:
  - `reext/calibrated_pairs.jsonl`: intact captions, extracted on the same GPU as the shuffles
    (primary intact condition I); without it, the stored `prh_val_calibrated` rows are I;
  - `shufw<seed>/calibrated_pairs.jsonl`: word-shuffled captions (S0, S1);
  - `extract.jsonl`: bpb and depth of LLMs outside PRH's pool (their intact re-extraction).
Model info for PRH's 10 LLMs comes from `prh_val/cross_modal.json` (params, layers, intact bpb).

Every score is excess over chance: raw max over layer pairs minus the mean of the null of the max
(Gröger's Algorithm 2). Derived condition G = I - mean(S0, S1), the alignment that needs word
order. G is a difference of scores, so its level is not "image-relevant order information" (the
exp-030 toy shows I - S > 0 with none); its trend with capability is what is read.

Writes `summary.md`, `trends.json` and `fig_trend.{png,pdf}` to `<out>`.

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
FAMILIES = [  # (name, model-id prefixes)
    ("BLOOMZ", ("bigscience/bloomz",)),
    ("OpenLLaMA", ("openlm-research/open_llama",)),
    ("LLaMA-1", ("huggyllama/llama",)),
    ("Llama-3", ("NousResearch/Meta-Llama-3", "unsloth/Llama-3", "meta-llama/")),
    ("Qwen2.5", ("Qwen/Qwen2.5",)),
    ("Gemma-2", ("unsloth/gemma-2", "google/gemma-2")),
]
MARKERS = {
    "BLOOMZ": "o",
    "OpenLLaMA": "s",
    "LLaMA-1": "D",
    "Llama-3": "^",
    "Qwen2.5": "v",
    "Gemma-2": "P",
}
COLORS = {"I": "#2a78d6", "S0": "#eb6834", "S1": "#b8462a", "S": "#eb6834", "G": "#2f9e5b"}
LABELS = {
    "I": "intact",
    "S0": "words shuffled (seed 0)",
    "S1": "words shuffled (seed 1)",
    "S": "words shuffled (mean of seeds)",
    "G": "order gain, intact - shuffled",
}

# Reading rules fixed in the README before the run (PRH's 10 LLMs).
P1_MAX_SHUF_RHO = 0.6
P2_MIN_WIN_SHARE = 160 / 170  # README: at least 160 of PRH's 170 pairs
P3_MAX_RHO_DIFF = 0.15
SEED_MAX_RHO_DIFF = 0.1


NO_BOS_FAMILIES = {"BLOOMZ", "Qwen2.5"}  # tokenizers that add no BOS (checked 2026-11-28)


def family(model: str) -> str:
    for name, prefixes in FAMILIES:
        if model.startswith(prefixes):
            return name
    return model.split("/")[-1]


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


def load_info(stored: Path, out: Path) -> tuple[list[str], list[str], dict]:
    """PRH's LLMs and ViTs, and model info; LLMs outside PRH's pool from extract.jsonl."""
    meta = json.loads((stored / "cross_modal.json").read_text())
    info = {m: dict(v) for m, v in meta["models"].items()}
    ext = out / "extract.jsonl"
    if ext.exists():
        for rec in map(json.loads, ext.read_text().splitlines()):
            if rec["transform"] == "reext" and rec["model"] not in info:
                info[rec["model"]] = {"bpb": rec["bpb"], "num_layers": rec["shape"][1]}
    return meta["llms"], meta["lvms"], info


def cube(pairs: dict, llms, lvms, measure: str) -> np.ndarray:
    out = np.full((len(llms), len(lvms)), np.nan)
    for i, lm in enumerate(llms):
        for j, v in enumerate(lvms):
            r = pairs.get((lm, v), {}).get(measure)
            if r is not None:
                out[i, j] = r["raw_score"] - r["mu0"]
    return out


def rho_per_vit(cap: np.ndarray, ex: np.ndarray, rows=None) -> np.ndarray:
    """Spearman(capability, excess) across LLMs (optionally a subset of rows), one per ViT."""
    rows = np.arange(len(cap)) if rows is None else np.asarray(rows)
    return np.array([spearmanr(cap[rows], ex[rows, j]).statistic for j in range(ex.shape[1])])


def slope_per_vit(cap: np.ndarray, ex: np.ndarray, rows=None) -> np.ndarray:
    rows = np.arange(len(cap)) if rows is None else np.asarray(rows)
    return np.array([np.polyfit(cap[rows], ex[rows, j], 1)[0] for j in range(ex.shape[1])])


def within_between(cap: np.ndarray, ex: np.ndarray, fams: list[str]) -> tuple[float, float]:
    """Median over ViTs of the pooled within-family slope (family-demeaned, families with >= 2
    LLMs) and of the between-family slope (over family means)."""
    names = sorted(set(fams))
    multi = [f for f in names if fams.count(f) >= 2]
    within, between = [], []
    for j in range(ex.shape[1]):
        num = den = 0.0
        for f in multi:
            idx = [i for i, g in enumerate(fams) if g == f]
            dc = cap[idx] - cap[idx].mean()
            num += float(dc @ (ex[idx, j] - ex[idx, j].mean()))
            den += float(dc @ dc)
        within.append(num / den if den else np.nan)
        fc = [cap[[i for i, g in enumerate(fams) if g == f]].mean() for f in names]
        fe = [ex[[i for i, g in enumerate(fams) if g == f], j].mean() for f in names]
        between.append(np.polyfit(fc, fe, 1)[0] if len(names) >= 2 else np.nan)
    return float(np.nanmedian(within)), float(np.nanmedian(between))


def summarise(x: np.ndarray) -> dict:
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def fmt(s: dict, nd: int = 2) -> str:
    return f"{s['median']:.{nd}f} ({s['min']:.{nd}f} to {s['max']:.{nd}f})"


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

    prh_llms, all_lvms, info = load_info(a.stored, a.out)
    stored = load_pairs(a.stored_cal / "calibrated_pairs.jsonl", STORED_KEY)
    reext = a.out / "reext" / "calibrated_pairs.jsonl"
    # Primary intact condition: re-extracted on the same GPU as the shuffles when available.
    conds = {"I": load_pairs(reext) if reext.exists() else stored}
    for s in a.seeds.split(","):
        path = a.out / f"shufw{s}" / "calibrated_pairs.jsonl"
        if path.exists():
            conds[f"S{s}"] = load_pairs(path)
    shuf = [c for c in conds if c.startswith("S")]
    if not shuf:
        raise SystemExit(f"no shuffled calibrated pairs under {a.out}")

    # LLMs need I and the first seed; a further seed is used only if it covers all of them (new
    # families were run with seed 0 only), so S is the same seeds for every LLM.
    have = {lm for lm, _ in conds["I"]} & {lm for lm, _ in conds[shuf[0]]}
    for c in shuf[1:]:
        if not have <= {lm for lm, _ in conds[c]}:
            print(f"[note] {c} does not cover every LLM; shuffled = {shuf[0]} only")
            del conds[c]
    shuf = [c for c in conds if c.startswith("S")]
    if len(shuf) == 1:
        LABELS["S"] = LABELS[shuf[0]]
    llms = sorted((m for m in have if m in info), key=lambda m: -info[m]["bpb"])
    lvms = [v for v in all_lvms if all((lm, v) in c for c in conds.values() for lm in llms)]
    cap = np.array([1 - info[m]["bpb"] for m in llms])
    fams = [family(m) for m in llms]
    prh = [i for i, m in enumerate(llms) if m in prh_llms]
    pools = {f"PRH {len(prh)}": prh}
    if len(prh) < len(llms):
        pools[f"all {len(llms)}"] = list(range(len(llms)))
    print(f"{len(llms)} LLMs x {len(lvms)} ViTs; conditions {list(conds)}; pools {list(pools)}")

    ex = {(c, m): cube(conds[c], llms, lvms, m) for c in conds for m in MEASURES}
    for m in MEASURES:
        ex[("S", m)] = np.mean([ex[(c, m)] for c in shuf], axis=0)
        ex[("G", m)] = ex[("I", m)] - ex[("S", m)]
    series = ["I", *shuf, "G"]
    res: dict = {
        "llms": llms,
        "lvms": lvms,
        "families": fams,
        "capability": cap.tolist(),
        "trends": {},
        "slopes": {},
        "within_between": {},
        "within": {},
        "families_table": {},
        "wins": {},
        "levels": {},
        "depth": {},
    }

    lines = [
        "# exp-031 summary: PRH capability trend, intact vs word-shuffled captions",
        "",
        f"WIT-1024, {len(llms)} LLMs x {len(lvms)} ViTs. Excess = raw max over layer pairs minus "
        "the mean of the null of the max (200 permutations). Capability = 1 - bpb on the intact "
        "captions. Per ViT across LLMs, then median (min to max) over ViTs. G = intact minus the "
        "mean of the shuffled seeds: the alignment that needs word order (its trend is the "
        "readout; its level is not image-relevant order information, see the README).",
        "",
        "## 1. Capability trend",
        "",
    ]
    rows = []
    for pool, idx in pools.items():
        for m, name in MEASURES.items():
            for c in series:
                rho = summarise(rho_per_vit(cap, ex[(c, m)], idx))
                sl = summarise(slope_per_vit(cap, ex[(c, m)], idx))
                res["trends"][f"{pool}/{c}/{m}"] = rho
                res["slopes"][f"{pool}/{c}/{m}"] = sl
                rows.append([pool, name, LABELS[c], fmt(rho), fmt(sl, 3)])
    lines += table(["LLMs", "measure", "condition", "Spearman", "slope on 1 - bpb"], rows)

    rows = []
    for pool, idx in pools.items():
        for m, name in MEASURES.items():
            s_i = slope_per_vit(cap, ex[("I", m)], idx)
            s_s = slope_per_vit(cap, ex[("S", m)], idx)
            share = float(np.median(s_s / s_i))
            res["slopes"][f"{pool}/S_share_of_I/{m}"] = share
            rows.append([pool, name, share, float(np.median(1 - s_s / s_i))])
    lines += [
        "Share of the intact slope kept by shuffled captions (per ViT, median):",
        "",
        *table(["LLMs", "measure", "shuffled / intact", "order gain / intact"], rows),
    ]

    bpb_path = a.out / "bpb.jsonl"
    if bpb_path.exists():
        full = {}
        for rec in map(json.loads, bpb_path.read_text().splitlines()):
            full[rec["model"]] = rec
        missing = [m for m in llms if m not in full and family(m) in NO_BOS_FAMILIES]
        if missing:
            lines += [f"(All-token capability skipped: no bpb.jsonl entry for {missing}.)", ""]
        else:
            cap_full = np.array(
                [1 - full[m]["bpb_all_tokens"] if m in full else cap[i] for i, m in enumerate(llms)]
            )
            res["capability_all_tokens"] = cap_full.tolist()
            rows = []
            for pool, idx in pools.items():
                for m, name in MEASURES.items():
                    for c in ("I", "S", "G"):
                        rho = summarise(rho_per_vit(cap_full, ex[(c, m)], idx))
                        sl = summarise(slope_per_vit(cap_full, ex[(c, m)], idx))
                        res["trends"][f"all_tokens/{pool}/{c}/{m}"] = rho
                        res["slopes"][f"all_tokens/{pool}/{c}/{m}"] = sl
                        rows.append([pool, name, LABELS[c], fmt(rho), fmt(sl, 3)])
            lines += [
                "Capability with every caption token scored (`bpb.jsonl`). PRH's bpb leaves the "
                "first caption token unscored for tokenizers without a BOS (BLOOMZ, Qwen2.5); "
                "here their EOS is prepended. Other families already score every token.",
                "",
                *table(["LLMs", "measure", "condition", "Spearman", "slope on 1 - bpb"], rows),
            ]
            rows = [
                [m.split("/")[-1], str(r["adds_bos"]), r["bpb_default"], r["bpb_all_tokens"]]
                for m, r in full.items()
            ]
            lines += table(["LLM", "adds BOS", "bpb, PRH default", "bpb, all tokens"], rows)

    lines += [
        "## 2. Within vs between families",
        "",
        "Pooled within-family slope (families with >= 2 LLMs, family means removed) and "
        "between-family slope (over family means), median over ViTs.",
        "",
    ]
    rows = []
    for m, name in MEASURES.items():
        for c in ("I", "S", "G"):
            w, b = within_between(cap, ex[(c, m)], fams)
            res["within_between"][f"{c}/{m}"] = {"within": w, "between": b}
            rows.append([name, LABELS[c], w, b])
    lines += table(["measure", "condition", "within-family slope", "between-family slope"], rows)

    rows = []
    for f in dict.fromkeys(fams):
        idx = [i for i, g in enumerate(fams) if g == f]
        row = [f, len(idx), float(cap[idx].mean())]
        for m in MEASURES:
            vals = {c: float(ex[(c, m)][idx].mean()) for c in ("I", "S", "G")}
            res["families_table"][f"{f}/{m}"] = {"n": len(idx), "capability": row[2], **vals}
            row += list(vals.values())
        rows.append(row)
    head = ["family", "LLMs", "1 - bpb"] + [
        f"{'mKNN' if m.startswith('mutual') else 'CKA'} {c}" for m in MEASURES for c in "ISG"
    ]
    lines += ["Family means (excess, mean over LLMs and ViTs):", "", *table(head, rows)]

    rows = []
    for f in dict.fromkeys(fams):
        idx = [i for i, g in enumerate(fams) if g == f]
        if len(idx) < 3:
            continue
        for m, name in MEASURES.items():
            for c in ("I", "S", "G"):
                s = summarise(rho_per_vit(cap, ex[(c, m)], idx))
                res["within"][f"{f}/{c}/{m}"] = s
                rows.append([f, len(idx), name, LABELS[c], fmt(s)])
    lines += [
        "Within families with >= 3 LLMs (sizes ordered by bpb), Spearman:",
        "",
        *table(["family", "LLMs", "measure", "condition", "Spearman"], rows),
    ]

    if reext.exists():
        rows = []
        common = [i for i, lm in enumerate(llms) if all((lm, v) in stored for v in lvms)]
        for m, name in MEASURES.items():
            st = cube(stored, [llms[i] for i in common], lvms, m)
            d = np.abs(ex[("I", m)][common] - st)
            res.setdefault("reproduction", {})[m] = {
                "median_abs_diff": float(np.median(d)),
                "max_abs_diff": float(d.max()),
                "n_llms": len(common),
            }
            rows.append([name, len(common), float(np.median(d)), float(d.max())])
        lines += [
            "Reproduction: intact re-extracted on this GPU vs the stored PRH features "
            "(another GPU), |excess difference| over pairs:",
            "",
            *table(["measure", "LLMs", "median", "max"], rows),
        ]

    lines += ["## 3. Levels", ""]
    rows = []
    for m, name in MEASURES.items():
        i_ex = ex[("I", m)]
        for c in shuf:
            wins = int((i_ex > ex[(c, m)]).sum())
            wins_prh = int((i_ex[prh] > ex[(c, m)][prh]).sum())
            res["wins"][f"{c}/{m}"] = wins
            res["wins"][f"PRH/{c}/{m}"] = wins_prh
            rows.append(
                [
                    name,
                    LABELS[c],
                    f"{wins}/{i_ex.size}",
                    float(np.median(i_ex)),
                    float(np.median(ex[(c, m)])),
                    float(np.median(ex[(c, m)] / i_ex)),
                ]
            )
    lines += table(
        [
            "measure",
            "shuffled",
            "pairs with intact > shuffled",
            "median excess, intact",
            "median excess, shuffled",
            "median shuffled / intact",
        ],
        rows,
    )

    rows = []
    for i, lm in enumerate(llms):
        row = [lm.split("/")[-1], fams[i], float(cap[i])]
        for m in MEASURES:
            row += [float(ex[(c, m)][i].mean()) for c in series]
        rows.append(row)
        res["levels"][lm] = {
            f"{c}/{m}": float(ex[(c, m)][i].mean()) for c in series for m in MEASURES
        }
    head = ["LLM", "family", "1 - bpb"] + [
        f"{'mKNN' if m.startswith('mutual') else 'CKA'} {c}" for m in MEASURES for c in series
    ]
    lines += ["Mean excess over ViTs, per LLM:", "", *table(head, rows)]

    lines += ["## 4. Depth of the best LLM layer (relative), median over ViTs", ""]
    rows = []
    for lm in llms:
        L = info[lm]["num_layers"]
        row = [lm.split("/")[-1]]
        for m in MEASURES:
            for c in ["I", *shuf]:
                d = [conds[c][(lm, v)][m]["best_indices"][1] / (L - 1) for v in lvms]
                res["depth"][f"{lm}/{c}/{m}"] = float(np.median(d))
                row.append(float(np.median(d)))
        rows.append(row)
    lines += table(
        ["LLM"]
        + [
            f"{'mKNN' if m.startswith('mutual') else 'CKA'} {c}"
            for m in MEASURES
            for c in ["I", *shuf]
        ],
        rows,
    )

    # Predictions and checks (README), on PRH's 10 LLMs.
    t = res["trends"]
    pk = next(iter(pools))
    n_prh = len(prh) * len(lvms)
    verdicts = []
    for c in shuf:
        r_i, r_s = t[f"{pk}/I/mutual_knn_k10"]["median"], t[f"{pk}/{c}/mutual_knn_k10"]["median"]
        verdicts.append(
            [
                "P1",
                c,
                f"mKNN Spearman {r_s:.2f} (intact {r_i:.2f}) <= {P1_MAX_SHUF_RHO}",
                "holds" if r_s <= P1_MAX_SHUF_RHO else "fails",
            ]
        )
        w = res["wins"][f"PRH/{c}/mutual_knn_k10"]
        verdicts.append(
            [
                "P2",
                c,
                f"intact > shuffled in {w}/{n_prh} pairs",
                "holds" if w >= P2_MIN_WIN_SHARE * n_prh else "fails",
            ]
        )
        d = abs(t[f"{pk}/{c}/cka_lin"]["median"] - t[f"{pk}/I/cka_lin"]["median"])
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
            d = abs(t[f"{pk}/S0/{m}"]["median"] - t[f"{pk}/S1/{m}"]["median"])
            verdicts.append(
                [
                    "seeds",
                    m,
                    f"S0 vs S1 median Spearman differ by {d:.2f}",
                    "ok" if d <= SEED_MAX_RHO_DIFF else "TOO NOISY",
                ]
            )
    res["verdicts"] = verdicts
    lines += [f"## Predictions and checks (fixed before the run; {pk} LLMs)", ""]
    lines += table(["", "condition", "result", "verdict"], verdicts)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.md").write_text("\n".join(lines))
    (a.out / "trends.json").write_text(json.dumps(res, indent=2))
    # x axis: all-token bpb when available (comparable across tokenizers), else PRH's bpb
    if "capability_all_tokens" in res:
        x, xlab = (
            np.array(res["capability_all_tokens"]),
            "1 - bits per byte, every caption token scored",
        )
    else:
        x, xlab = cap, "1 - bits per byte (intact captions)"
    plot(a.out, llms, x, fams, ex, xlab)
    print("\n".join(lines))


def plot(out: Path, llms, cap, fams, ex, xlabel: str) -> None:
    """Per LLM: mean excess over ViTs against 1 - bpb; band = min..max over ViTs for every
    series; marker = family."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    order = np.argsort(cap)
    for ax, (m, name) in zip(axes, MEASURES.items(), strict=True):
        for c in ("I", "S", "G"):
            e = ex[(c, m)][order]
            ax.fill_between(cap[order], e.min(1), e.max(1), color=COLORS[c], alpha=0.13, lw=0)
            ax.plot(cap[order], e.mean(1), "-", color=COLORS[c], lw=1.2, label=LABELS[c])
            for i in order:
                ax.plot(
                    cap[i],
                    ex[(c, m)][i].mean(),
                    MARKERS.get(fams[i], "P"),
                    color=COLORS[c],
                    ms=4,
                )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("excess over chance")
        ax.grid(alpha=0.25, lw=0.5)
    handles = [
        plt.Line2D([], [], color="0.3", marker=MARKERS.get(f, "P"), ls="", label=f)
        for f in dict.fromkeys(fams)
    ]
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    axes[1].legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    fig.suptitle("WIT-1024: mean over ViTs, band min..max over ViTs", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_trend.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
