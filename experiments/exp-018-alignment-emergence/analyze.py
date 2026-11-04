"""exp-018: emergence curves, onset tokens per neighbourhood size, alignment vs LM loss.

Reads `pairs.jsonl`, `pairs/*.npz` and `bpb/*.json` of the output folder. For every series (model,
seed), ViT or reference LM, caption set and measure: the calibrated excess (score - null mean)
at the best cell of each checkpoint, against training tokens (Pythia: 2,097,152 tokens per step;
step 0 = random initialisation).

Onsets: progress(t) = (e(t) - e(0)) / (e(final) - e(0)), and t_q = the first tokens at which
progress reaches q (q = 0.25, 0.5, 0.75; log-linear interpolation between checkpoints). For mutual
kNN at each k, a 95% CI by bootstrap over samples: the per-sample overlaps saved at each
checkpoint's best cell are resampled with the same indices for every checkpoint (neighbourhoods
and best cells fixed). The ordering test asks whether t50 increases or decreases with k.

    uv run python experiments/exp-018-alignment-emergence/analyze.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TOKENS_PER_STEP = {"pythia": 2_097_152}
KEY = re.compile(r"^(?P<repo>.+)@(?P<rev>[^_]+)__(?P<caption>orig|gemini)$")
QS = (0.25, 0.5, 0.75)


def tokens(repo: str, rev: str) -> float:
    m = re.search(r"tokens(\d+)B", rev)  # OLMo-2: stage1-step1000-tokens2B
    if m:
        return float(m.group(1)) * 1e9
    step = int(re.search(r"step(\d+)", rev).group(1))
    per = next((v for k, v in TOKENS_PER_STEP.items() if k in repo), None)
    assert per is not None, f"tokens per step unknown for {repo}"
    return step * per


def onset(t: np.ndarray, e: np.ndarray, q: float) -> float:
    """First tokens at which (e - e[0]) / (e[-1] - e[0]) reaches q; log-linear interpolation."""
    span = e[-1] - e[0]
    if abs(span) < 1e-9:
        return np.nan
    prog = (e - e[0]) / span
    lt = np.log10(np.maximum(t, t[t > 0].min() / 2))  # step 0 placed half a step before the first
    for i in range(1, len(prog)):
        if prog[i] >= q:
            f = (q - prog[i - 1]) / max(prog[i] - prog[i - 1], 1e-12)
            return float(10 ** (lt[i - 1] + f * (lt[i] - lt[i - 1])))
    return np.nan


def collect(out: Path):
    """{(series, target, caption, measure): [(tokens, excess, npz path, field, mu0)]}."""
    curves = defaultdict(list)
    for line in (out / "pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        m = KEY.match(r["a"])
        if not m:
            continue
        series, cap = m["repo"].split("__")[-1], m["caption"]
        target = (
            r["b"].rsplit("__", 1)[0].split(".")[0]
            if r["group"] == "LV"
            else r["b"].rsplit("__", 1)[0]
        )
        t = tokens(m["repo"], m["rev"])
        npz = out / "pairs" / f"{r['id']}.npz"
        n = r["n"]

        entries = [
            (f"mkNN k=10 n={n}", r["mutual_knn"], "o_k10_best"),
            (f"CKA n={n}", r["cka_lin"], None),
        ]
        entries += [
            (f"mkNN k={k} n={n}", d, f"o_k{k}_best")
            for k, d in r.get("knn_sweep", {}).items()
            if k != "10"
        ]
        if "orc_k10" in r:
            entries.append((f"ORC|dens n={n}", r["orc_k10"], None))
        for name, d, field in entries:
            ex = d["best"]["score"] - d["best"]["mu0"]
            curves[(series, target, cap, name)].append((t, ex, npz, field, d["best"]["mu0"]))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in curves.items()}


def boot_onsets(pts, reps: int, seed: int = 0) -> np.ndarray:
    """[reps, len(QS) + 1] onsets and total change from per-sample overlaps resampled with common indices."""
    o = np.stack([np.load(p)[f] for _, _, p, f, _ in pts])  # [T, n]
    mu0 = np.array([x[4] for x in pts])
    t = np.array([x[0] for x in pts])
    rng = np.random.default_rng(seed)
    out = np.empty((reps, len(QS) + 1))  # onsets, then the change final - step 0
    for b in range(reps):
        e = o[:, rng.integers(0, o.shape[1], o.shape[1])].mean(1) - mu0
        out[b] = [*(onset(t, e, q) for q in QS), e[-1] - e[0]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=200)
    args = p.parse_args()
    curves = collect(args.out)
    rows, lines = (
        [],
        [
            "# exp-018 summary",
            "",
            "Onset tokens (progress from step 0 to the final checkpoint; 95% bootstrap CI for mutual kNN)",
            "",
        ],
    )
    lines += [
        "| series | target | captions | measure | excess step 0 | excess final [CI of change] | t25 | t50 [CI] | t75 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (series, target, cap, meas), pts in sorted(curves.items()):
        t = np.array([x[0] for x in pts])
        e = np.array([x[1] for x in pts])
        on = [onset(t, e, q) for q in QS]
        ci, change_ci = [np.nan, np.nan], [np.nan, np.nan]
        if pts[0][3] is not None and len(pts) > 2:
            b = boot_onsets(pts, args.reps)
            if np.isfinite(b[:, 1]).any():
                ci = np.nanpercentile(b[:, 1], [2.5, 97.5]).tolist()
            change_ci = np.percentile(b[:, -1], [2.5, 97.5]).tolist()
        # an onset is only meaningful if the curve changed beyond sampling noise
        moved = not (change_ci[0] <= 0 <= change_ci[1])
        rows.append(
            {
                "series": series,
                "target": target,
                "caption": cap,
                "measure": meas,
                "n": int(meas.split("n=")[1]),
                "tokens": t.tolist(),
                "excess": e.tolist(),
                "onsets": on,
                "t50_ci": ci,
                "change_ci": change_ci,
                "moved": moved,
            }
        )
        lines.append(
            f"| {series} | {target} | {cap} | {meas} | {e[0]:.3f} | {e[-1]:.3f} [{change_ci[0]:+.3f}, {change_ci[1]:+.3f}] | {on[0]:.2g} | {on[1]:.2g} [{ci[0]:.2g}, {ci[1]:.2g}] | {on[2]:.2g} |"
        )
    lines += [
        "",
        "Ordering test: Spearman of t50 against k (mutual kNN at the largest n; curves whose change is within noise left out);",
        "negative = large neighbourhoods first (coarse first), positive = small first (local first).",
        "",
    ]
    from scipy.stats import spearmanr

    groups = defaultdict(list)
    n_max = max((r["n"] for r in rows), default=0)
    for r in rows:
        m = re.match(rf"mkNN k=(\d+) n={n_max}$", r["measure"])
        if m and r["moved"] and np.isfinite(r["onsets"][1]):
            groups[(r["series"], r["target"], r["caption"])].append(
                (int(m.group(1)), r["onsets"][1])
            )
    lines += ["| series | target | captions | k: t50 | Spearman(k, t50) |", "|---|---|---|---|---|"]
    for key, v in sorted(groups.items()):
        v.sort()
        rho = spearmanr([k for k, _ in v], [x for _, x in v])[0] if len(v) > 2 else np.nan
        lines.append(
            f"| {' | '.join(key)} | {', '.join(f'{k}: {x:.2g}' for k, x in v)} | {rho:+.2f} |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    (args.out / "curves.json").write_text(json.dumps(rows))
    print("\n".join(lines))
    plot(rows, args.out)


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    bpb = {}
    for f in (out / "bpb").glob("*.json"):
        d = json.loads(f.read_text())
        bpb[(d["model"].split("/")[-1], d["revision"])] = d["generic"]["bpb"]
    n_max = max(r["n"] for r in rows)
    panels = sorted({(r["target"], r["caption"]) for r in rows})
    series = sorted({r["series"] for r in rows})
    fig, axes = plt.subplots(
        len(series),
        max(len(panels), 1),
        figsize=(4 * len(panels), 3.2 * len(series)),
        squeeze=False,
    )
    for i, s in enumerate(series):
        for j, (tg, cap) in enumerate(panels):
            ax = axes[i, j]
            for r in rows:
                if (r["series"], r["target"], r["caption"]) == (s, tg, cap) and r["n"] == n_max:
                    t = np.maximum(np.array(r["tokens"]), 1e6)
                    ax.plot(t, r["excess"], marker="o", ms=2, label=r["measure"].split(" n=")[0])
            ax.set_xscale("log")
            ax.set_title(
                f"{s} x {tg.split('_patch')[0] if 'patch' in tg else tg}, {cap}", fontsize=8
            )
            ax.grid(alpha=0.3)
            ax.set_xlabel("training tokens (step 0 at 1e6)", fontsize=7)
    axes[0, 0].set_ylabel("calibrated excess (best cell)")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_curves.{ext}", dpi=200)
    if bpb:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        for r in rows:
            if (
                r["measure"] == f"mkNN k=10 n={n_max}"
                and r["caption"] == "gemini"
                and r["target"] == panels[0][0]
            ):
                b = [bpb.get((r["series"], rev)) for rev in revisions(out, r)]
                if all(x is not None for x in b):
                    ax.plot(b, r["excess"], marker="o", ms=2, label=r["series"])
        ax.invert_xaxis()
        ax.set_xlabel("generic bits per byte (lower = better LM)")
        ax.set_ylabel(f"mutual kNN k=10 excess, x {panels[0][0]}, Gemini")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_vs_bpb.{ext}", dpi=200)


def revisions(out: Path, row: dict) -> list[str]:
    """Revisions of a curve's checkpoints, in token order (from the feature file names)."""
    revs = set()
    for f in (out / "features").glob(f"*{row['series']}@*__{row['caption']}__d2l.pt"):
        revs.add(f.stem.split("@")[1].split("__")[0])
    return sorted(revs, key=lambda r: tokens(row["series"], r) if "pythia" in row["series"] else r)


if __name__ == "__main__":
    main()
