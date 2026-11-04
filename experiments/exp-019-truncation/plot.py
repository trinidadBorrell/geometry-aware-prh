"""exp-019 figures and summary from `truncate.py` output.

fig_excess: alignment excess over the null of the max (max_score - max_mu0) against the truncation
size m, one panel per (metric, operator); mean over pairs of a group with a min..max band.
fig_retention: the excess at the pair's fixed layer cell (best untruncated), relative to its
untruncated value, next to the
group's self-retention (geometric mean of the two models' truncated-vs-full agreement at the
pair's fixed layer cell), which is what truncation would leave if it acted on each model alone.

    uv run python experiments/exp-019-truncation/plot.py --root results/exp-019-truncation/pilot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

OPS = ("top", "drop_top", "rand_coord", "rand_proj")
OP_LABEL = {
    "top": "keep top-m PCs",
    "drop_top": "remove top-m PCs",
    "rand_coord": "keep m random coordinates",
    "rand_proj": "keep m random directions",
}
METRICS = {"mutual_knn": "mutual kNN (k = 10)", "cka_lin": "linear CKA"}
# fixed categorical order (reference palette slots 1-4), never cycled
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# colour follows the group family (the caption variant is the figure, not the colour)
FAMILY_SLOT = {"X": 0, "B3": 0, "W_llm": 1, "W_vit": 2, "D_shuffle": 3, "A": 4, "B1": 5}
D2_SETS = {
    "orig": ["B3_orig", "W_llm_orig", "W_vit", "A_orig", "D_shuffle"],
    "gemini": ["B3_gemini", "W_llm_gemini", "W_vit", "A_gemini", "B1", "D_shuffle"],
}


def color(group: str) -> str:
    fam = group if group in FAMILY_SLOT else group.rsplit("_", 1)[0]
    return COLORS[FAMILY_SLOT.get(fam, 0)]


def load(root: Path):
    rows = [json.loads(x) for x in (root / "pairs.jsonl").read_text().splitlines()]
    selfrows = [json.loads(x) for x in (root / "self.jsonl").read_text().splitlines()]
    return rows, selfrows


def curves(rows, selfrows):
    """{(group, metric, op): {pair: {m: (excess, relative, self)}}} averaged over draws."""
    base = {}
    for r in rows:
        if r["op"] == "none":
            base[(r["a"], r["b"], r["metric"])] = r["fixed_score"] - r["fixed_mu0"]
    selfmap = {}
    for s in selfrows:
        for met in METRICS:
            selfmap.setdefault((s["model"], s["op"], s["m"], met), []).append(s[met])
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["op"] in ("prh",):
            continue
        key = (r["a"], r["b"], r["metric"])
        exc = r["max_score"] - r["max_mu0"]
        # relative excess at the fixed cell (best layer pair untruncated), like self-retention
        rel = (r["fixed_score"] - r["fixed_mu0"]) / base[key] if base[key] > 0.02 else np.nan
        i, j = r["fixed_cell"]
        sa = np.mean([v[i] for v in selfmap[(r["a"], r["op"], r["m"], r["metric"])]])
        sb = np.mean([v[j] for v in selfmap[(r["b"], r["op"], r["m"], r["metric"])]])
        acc[(r["group"], r["metric"], r["op"])][(r["a"], r["b"])][r["m"]].append(
            (exc, rel, np.sqrt(max(sa, 0) * max(sb, 0)))
        )
    out = {}
    for k, pairs in acc.items():
        out[k] = {p: {m: tuple(np.mean(v, 0)) for m, v in ms.items()} for p, ms in pairs.items()}
    # the untruncated point belongs to every operator's curve
    for (g, met, op), pairs in list(out.items()):
        if op == "none":
            continue
        for p in pairs:
            pairs[p][0] = out[(g, met, "none")][p][0]
    return out


FULL_X = 10  # untruncated point, right of m = 512 (left of m = 1 when removing PCs)


def _xpos(op, m):
    if m == 0:
        return -1 if op == "drop_top" else FULL_X
    return np.log2(m)


def _ticks(ax, op):
    if op == "drop_top":
        xs = [-1, 1, 3, 5, 7, 8]
        ax.set_xticks(xs, ["none", "2", "8", "32", "128", "256"])
    else:
        xs = [1, 3, 5, 7, 9, FULL_X]
        ax.set_xticks(xs, ["2", "8", "32", "128", "512", "full"])


def _band(ax, xs, ys, color, label, ls="-"):
    ys = np.array(ys, float)
    mean = np.nanmean(ys, 0)
    ax.fill_between(xs, np.nanmin(ys, 0), np.nanmax(ys, 0), color=color, alpha=0.15, lw=0)
    ax.plot(xs, mean, ls, color=color, lw=2, marker="o", ms=3, label=label)


def figure(cur, groups, out: Path, which: str, tag: str = "") -> None:
    fig, axes = plt.subplots(
        len(METRICS), len(OPS), figsize=(4 * len(OPS), 3.2 * len(METRICS)), sharex="col"
    )
    for r, met in enumerate(METRICS):
        for c, op in enumerate(OPS):
            ax = axes[r, c]
            for g in groups:
                pairs = cur.get((g, met, op))
                if not pairs or (which == "retention" and g == "D_shuffle"):
                    continue
                ms = sorted(
                    set.intersection(*[set(v) for v in pairs.values()]), key=lambda m: _xpos(op, m)
                )
                xs = [_xpos(op, m) for m in ms]
                idx = 0 if which == "excess" else 1
                _band(ax, xs, [[pairs[p][m][idx] for m in ms] for p in pairs], color(g), g)
                if which == "retention":
                    _band(
                        ax,
                        xs,
                        [[pairs[p][m][2] for m in ms] for p in pairs],
                        color(g),
                        f"{g} self",
                        ls="--",
                    )
            ax.axhline(0 if which == "excess" else 1, color="0.6", lw=0.8)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_title(f"{OP_LABEL[op]}", fontsize=10)
            if r == len(METRICS) - 1:
                ax.set_xlabel("m kept" if op != "drop_top" else "leading PCs removed")
            _ticks(ax, op)
            if c == 0:
                ax.set_ylabel(
                    METRICS[met]
                    + (
                        "\nexcess over null of max"
                        if which == "excess"
                        else "\nrelative to untruncated"
                    )
                )
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2 if which == "retention" else 1)
    fig.tight_layout()
    fig.savefig(out / f"fig_{which}{tag}.png", dpi=150)
    fig.savefig(out / f"fig_{which}{tag}.pdf")
    plt.close(fig)


def summary(rows, cur, groups, out: Path) -> None:
    lines = ["# exp-019 summary (generated by plot.py)", ""]
    prh = {
        (r["a"], r["b"], r["metric"]): r["max_score"] - r["max_mu0"]
        for r in rows
        if r["op"] == "prh"
    }
    for met, mlabel in METRICS.items():
        lines += [f"## {mlabel}: excess over null of max, mean [min, max] over pairs", ""]
        for op in ("none", *OPS):
            for g in groups:
                pairs = cur.get((g, met, op))
                if not pairs:
                    continue
                ms = sorted(set.intersection(*[set(v) for v in pairs.values()]))
                cells = []
                for m in ms:
                    v = np.array([pairs[p][m][0] for p in pairs])
                    cells.append(f"{m}: {v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]")
                lines.append(f"- {op} / {g}: " + "; ".join(cells))
        lines.append("")
        for g in groups:
            v = [prh[(a, b, met)] for (a, b) in cur.get((g, met, "none"), {})]
            if v:
                lines.append(
                    f"- uncentred PRH reference / {g}: {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]"
                )
        lines.append("")
        lines += [f"### {mlabel}: best m under top-m PCA (k*), per pair", ""]
        for g in groups:
            pairs = cur.get((g, met, "top"))
            if not pairs:
                continue
            for p, ms in pairs.items():
                best = max(ms, key=lambda m: ms[m][0])
                lines.append(
                    f"- {g} {p[0]} x {p[1]}: k* = {best or 'full'}, excess {ms[best][0]:.3f} vs untruncated {ms[0][0]:.3f}"
                )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows, selfrows = load(root)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    cur = curves(rows, selfrows)
    if len(groups) <= len(COLORS):
        sets = {"": groups}
    else:
        sets = {f"_{k}": [g for g in v if g in groups] for k, v in D2_SETS.items()}
    for tag, gs in sets.items():
        figure(cur, gs, root, "excess", tag)
        figure(cur, gs, root, "retention", tag)
    summary(rows, cur, groups, root)


if __name__ == "__main__":
    main()
