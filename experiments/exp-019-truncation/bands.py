"""exp-019 D2 analysis on `truncate.py d2s` output: band widths, caption and size effects, and the
null of the max over truncation sizes. Writes bands.md next to the results.

Per pair and metric, from the excess e(m) = max_score - max_mu0 (log2-interpolated):
- m90: top-m PCs needed to reach 90% of the untruncated excess;
- d50: leading PCs whose removal halves it.

Null of the max over m: permutations are the same at every setting, so for permutation p the excess
at setting m is null_max[p](m) - mu0(m), and its max over the top-m settings (including the
untruncated one) is the null of a data-driven choice of m. Gain = max_m e(m) - e(full) is compared
with the same difference under each permutation. This calibrates selection noise at zero alignment.

The max-cell excess subtracts the null of the max over all layer pairs, which truncation also moves
(the CKA null grows with effective dimension), so a gain in it need not be a gain in agreement at any
cell. `decomposition` separates the two, and the fixed-cell gain (best layer pair untruncated, per-cell
null) is the one to read as a change in agreement.

    uv run python experiments/exp-019-truncation/bands.py --root results/exp-019-truncation/d2s
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from plot import curves, load

METRICS = ("mutual_knn", "cka_lin")
TOP_M = (2, 4, 8, 16, 32, 64, 128, 256, 512)
DROP_M = (1, 2, 4, 8, 16, 32, 64, 128, 256)
GROUPS = (
    "B3_orig", "B3_gemini", "W_llm_orig", "W_llm_gemini", "W_vit", "A_orig", "A_gemini", "B1",
)  # fmt: skip
SHORT = {
    "bigscience__bloomz-1b7": "bloomz-1b7",
    "bigscience__bloomz-7b1": "bloomz-7b1",
    "openlm-research__open_llama_7b": "open_llama_7b",
    "huggyllama__llama-13b": "llama-13b",
}


def crossing(xs, ys, level, rising):
    """First log2-interpolated x where ys crosses level; x is in log2 units."""
    pts = list(zip(xs, ys, strict=True))
    for (x0, y0), (x1, y1) in pairwise(pts):
        if (rising and y0 < level <= y1) or (not rising and y0 > level >= y1):
            return x0 + (x1 - x0) * (level - y0) / (y1 - y0)
    return np.nan


def band_widths(rows):
    """{(metric, a, b): (group, e_full, m90, d50)}."""
    e = defaultdict(dict)
    groups = {}
    for r in rows:
        if r["op"] in ("none", "top", "drop_top"):
            e[(r["metric"], r["a"], r["b"])][(r["op"], r["m"])] = r["max_score"] - r["max_mu0"]
            groups[(r["metric"], r["a"], r["b"])] = r["group"]
    out = {}
    for key, v in e.items():
        full = v[("none", 0)]
        top = [v[("top", m)] for m in TOP_M]
        drop = [full, *[v[("drop_top", m)] for m in DROP_M]]
        logs_top = [float(np.log2(m)) for m in TOP_M]
        logs_drop = [-1.0, *[float(np.log2(m)) for m in DROP_M]]  # "none" one step left of m = 1
        m90 = crossing([*logs_top, 10.0], [*top, full], 0.9 * full, True) if full > 0.02 else np.nan
        if full > 0.02 and top[0] >= 0.9 * full:
            m90 = logs_top[0]
        d50 = crossing(logs_drop, drop, 0.5 * full, False) if full > 0.02 else np.nan
        out[key] = (groups[key], full, 2**m90, 2**d50)
    return out


def null_of_max_over_m(rows):
    """{(metric, a, b): dict} from the `null_max` of the top-m and untruncated settings."""
    data = defaultdict(dict)
    groups = {}
    for r in rows:
        if r["op"] in ("none", "top"):
            data[(r["metric"], r["a"], r["b"])][r["m"]] = (
                r["max_score"] - r["max_mu0"],
                np.asarray(r["null_max"]) - r["max_mu0"],
            )
            groups[(r["metric"], r["a"], r["b"])] = r["group"]
    out = {}
    for key, v in data.items():
        ms = sorted(v)
        obs = np.array([v[m][0] for m in ms])
        null = np.stack([v[m][1] for m in ms])  # [settings, P]
        full = ms.index(0)
        gain = obs.max() - obs[full]
        null_gain = null.max(0) - null[full]
        out[key] = {
            "group": groups[key],
            "m_star": ms[int(obs.argmax())],
            "gain": float(gain),
            "null_gain_q95": float(np.quantile(null_gain, 0.95)),
            "p_gain": float((1 + (null_gain >= gain).sum()) / (1 + null.shape[1])),
            "max_excess": float(obs.max()),
            "p_max": float((1 + (null.max(0) >= obs.max()).sum()) / (1 + null.shape[1])),
        }
    return out


def fixed_m_gain(rows, m=16):
    """{(metric, group): [e(m) - e(full) per pair]}: a choice of m made once for all pairs."""
    e = defaultdict(dict)
    for r in rows:
        if (r["op"] == "top" and r["m"] == m) or r["op"] == "none":
            e[(r["metric"], r["group"], r["a"], r["b"])][r["op"]] = r["max_score"] - r["max_mu0"]
    out = defaultdict(list)
    for (metric, group, _, _), v in e.items():
        out[(metric, group)].append(v["top"] - v["none"])
    return out


def decomposition(rows, m=16):
    """{(metric, group): per-pair [max_score, max_mu0, fixed_score, fixed_mu0] at none and top-m}."""
    e = defaultdict(dict)
    for r in rows:
        if (r["op"] == "top" and r["m"] == m) or r["op"] == "none":
            e[(r["metric"], r["group"], r["a"], r["b"])][r["op"]] = [
                r["max_score"], r["max_mu0"], r["fixed_score"], r["fixed_mu0"],
            ]  # fmt: skip
    out = defaultdict(list)
    for (metric, group, _, _), v in e.items():
        out[(metric, group)].append((v["none"], v["top"]))
    return {k: np.array(v) for k, v in out.items()}  # [pairs, none/top, 4]


def drop_gain(rows, ms=(1, 2)):
    """Pairs whose fixed-cell excess rises by more than 0.01 after removing the top PCs."""
    base, out = {}, []
    for r in rows:
        if r["op"] == "none":
            base[(r["metric"], r["a"], r["b"])] = r["fixed_score"] - r["fixed_mu0"]
    for r in rows:
        if r["op"] == "drop_top" and r["m"] in ms:
            b = base[(r["metric"], r["a"], r["b"])]
            v = r["fixed_score"] - r["fixed_mu0"]
            if v - b > 0.01:
                out.append((r["metric"], r["group"], r["a"], r["b"], r["m"], b, v))
    return out


def _fmt(values):
    v = np.asarray(values, float)
    return f"{np.nanmean(v):.1f} [{np.nanmin(v):.1f}, {np.nanmax(v):.1f}]"


def llm_of(name):
    return name.split("__orig")[0].split("__gemini")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    rows, selfrows = load(args.root)
    pairs = {(r["a"], r["b"]) for r in rows}
    bands = band_widths(rows)
    nulls = null_of_max_over_m(rows)
    cur = curves(rows, selfrows)
    out = [f"# exp-019 D2 bands ({len(pairs)} pairs)", ""]

    out += ["## Band widths per group, mean [min, max] over pairs", ""]
    out += [
        "| metric | group | pairs | untruncated excess | m90 | d50 |",
        "|---|---|---|---|---|---|",
    ]
    for metric in METRICS:
        for g in GROUPS:
            b = [v for k, v in bands.items() if k[0] == metric and v[0] == g]
            ex = np.array([x[1] for x in b])
            out.append(
                f"| {metric} | {g} | {len(b)} | {ex.mean():.3f} [{ex.min():.3f}, {ex.max():.3f}] "
                f"| {_fmt([x[2] for x in b])} | {_fmt([x[3] for x in b])} |"
            )
    out.append("")

    out += ["## Gemini vs original captions, same model pair (gemini - orig)", ""]
    out += ["| metric | family | pairs | excess | m90 | d50 | d50 larger with Gemini |"]
    out += ["|---|---|---|---|---|---|---|"]
    for metric in METRICS:
        for fam in ("B3", "W_llm", "A"):
            dif = []
            for (m_, a, b), v in bands.items():
                if m_ != metric or v[0] != f"{fam}_gemini":
                    continue
                oa, ob = a.replace("gemini", "orig"), b.replace("gemini", "orig")
                w = bands[(metric, oa, ob)]
                dif.append((v[1] - w[1], v[2] - w[2], v[3] - w[3]))
            d = np.array(dif)
            out.append(
                f"| {metric} | {fam} | {len(d)} | {d[:, 0].mean():+.3f} | {d[:, 1].mean():+.1f} "
                f"[{d[:, 1].min():+.1f}, {d[:, 1].max():+.1f}] | {d[:, 2].mean():+.1f} "
                f"[{d[:, 2].min():+.1f}, {d[:, 2].max():+.1f}] | {(d[:, 2] > 0).sum()}/{len(d)} |"
            )
    out.append("")

    out += ["## Model size, B3 pairs: means by LLM and by ViT", ""]
    out += [
        "| metric | captions | model | untruncated excess | m90 | d50 |",
        "|---|---|---|---|---|---|",
    ]
    for metric in METRICS:
        for cap in ("orig", "gemini"):
            by = defaultdict(list)
            for (m_, a, b), v in bands.items():
                if m_ == metric and v[0] == f"B3_{cap}":
                    by[SHORT[llm_of(a)]].append(v)
                    by[b.split("__")[0]].append(v)
            for name, b in by.items():
                out.append(
                    f"| {metric} | {cap} | {name} | {np.mean([x[1] for x in b]):.3f} "
                    f"| {np.nanmean([x[2] for x in b]):.1f} | {np.nanmean([x[3] for x in b]):.1f} |"
                )
    out.append("")

    out += ["## Null of the max over m (top-m and untruncated)", ""]
    out += [
        "| metric | group | pairs | k* median | gain | gain > null q95 | p(gain) median "
        "| p(max over m) max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric in METRICS:
        for g in (*GROUPS, "D_shuffle"):
            v = [x for k, x in nulls.items() if k[0] == metric and x["group"] == g]
            gain = np.array([x["gain"] for x in v])
            above = sum(x["gain"] > x["null_gain_q95"] for x in v)
            ks = [x["m_star"] if x["m_star"] else 2048 for x in v]
            out.append(
                f"| {metric} | {g} | {len(v)} | {np.median(ks):.0f} | {gain.mean():.3f} "
                f"[{gain.min():.3f}, {gain.max():.3f}] | {above}/{len(v)} "
                f"| {np.median([x['p_gain'] for x in v]):.3f} "
                f"| {max(x['p_max'] for x in v):.3f} |"
            )
    out += ["", "(k* = 2048 stands for untruncated.)", ""]

    out += ["## Choice of m fixed in advance: e(16) - e(full)", ""]
    out += ["| metric | group | mean [min, max] | pairs with gain |", "|---|---|---|---|"]
    fixed = fixed_m_gain(rows)
    for metric in METRICS:
        for g in (*GROUPS, "D_shuffle"):
            d = np.array(fixed[(metric, g)])
            out.append(
                f"| {metric} | {g} | {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}] "
                f"| {(d > 0).sum()}/{len(d)} |"
            )
    out.append("")

    out += ["## What moves at top-16: agreement or the null (means over pairs, none -> top-16)", ""]
    out += [
        "| metric | group | max score | null of the max | max-cell excess | fixed-cell score "
        "| cell null | fixed-cell excess | pairs with fixed-cell gain |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    dec = decomposition(rows)
    for metric in METRICS:
        for g in (*GROUPS, "D_shuffle"):
            d = dec[(metric, g)]
            m = d.mean(0)
            fx = d[:, :, 2] - d[:, :, 3]
            mx = d[:, :, 0] - d[:, :, 1]
            out.append(
                f"| {metric} | {g} | {m[0, 0]:.3f} -> {m[1, 0]:.3f} | {m[0, 1]:.3f} -> {m[1, 1]:.3f} "
                f"| {mx[:, 0].mean():.3f} -> {mx[:, 1].mean():.3f} "
                f"| {m[0, 2]:.3f} -> {m[1, 2]:.3f} | {m[0, 3]:.3f} -> {m[1, 3]:.3f} "
                f"| {fx[:, 0].mean():.3f} -> {fx[:, 1].mean():.3f} "
                f"| {(fx[:, 1] > fx[:, 0]).sum()}/{len(fx)} |"
            )
    out.append("")

    out += ["## Fixed-cell excess that rises by > 0.01 after removing the top 1-2 PCs", ""]
    out += ["| metric | group | a | b | removed | none | after |", "|---|---|---|---|---|---|---|"]
    for metric, g, a, b, m, v0, v1 in drop_gain(rows):
        out.append(f"| {metric} | {g} | {a} | {b} | {m} | {v0:.3f} | {v1:.3f} |")
    out.append("")

    out += ["## Relative excess / self-retention at the fixed cell, mean over pairs", ""]
    for metric in METRICS:
        for op, ms in (
            ("top", (8, 16, 32, 64, 128)),
            ("drop_top", (2, 8, 16, 32, 64, 128)),
            ("rand_coord", (16, 64, 256)),
            ("rand_proj", (16, 64, 256)),
        ):
            out += [f"**{metric}, {op}**", ""]
            out += ["| group | " + " | ".join(str(m) for m in ms) + " |"]
            out += ["|---|" + "---|" * len(ms)]
            for g in GROUPS:
                c = cur[(g, metric, op)]
                cells = []
                for m in ms:
                    v = np.array([c[pr][m] for pr in c])
                    cells.append(f"{np.nanmean(v[:, 1]):.2f} / {np.nanmean(v[:, 2]):.2f}")
                out.append(f"| {g} | " + " | ".join(cells) + " |")
            out.append("")

    (args.root / "bands.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
