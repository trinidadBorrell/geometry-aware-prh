"""Figures and summary table for exp-008 from calibrate.py output.

    uv run python experiments/exp-008-cknna-null/plot.py --root results/exp-008-cknna-null

Reads <root>/calibrated_pairs.jsonl; writes PNG + PDF to <root>/figures:

* fig1_null        per metric: raw score and null mean (max over layer pairs) vs k, mean over
                   pairs; linear CKA as horizontal lines
* fig2_calibrated  calibrated g = (raw - tau) / (1 - tau) vs k for the metrics bounded by 1
                   (mutual kNN, `paper`, `centred`) and CKA; g is undefined for `code` and
                   `eq36`, whose null exceeds 1 (fig1 shows them against their null)
* fig3_families    calibrated g vs k per vision family (mean over LLMs and ViTs of the family)
                   for mutual kNN and the two bounded CKNNA definitions
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "plot7", Path(__file__).parents[1] / "exp-007-cknna-variants" / "plot.py"
)
p7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p7)
METRICS, INK, INK_MUTED, SERIES = p7.METRICS, p7.INK, p7.INK_MUTED, p7.SERIES

BOUNDED = ("mknn", "paper", "centred")
FAMILIES = [
    ("ImageNet21K", lambda n: "augreg" in n),
    ("MAE", lambda n: ".mae" in n),
    ("DINOv2", lambda n: "dinov2" in n),
    ("CLIP", lambda n: "clip" in n and "ft_in12k" not in n),
    ("CLIP (I12K ft)", lambda n: "ft_in12k" in n),
]


def family(lvm: str) -> int:
    return next(i for i, (_, match) in enumerate(FAMILIES) if match(lvm))


def curves(recs: list[dict], key: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """ks, {metric: [pairs, ks]} of the summary field `key`, and CKA's [pairs]."""
    ks = np.array([p["k"] for p in recs[0]["per_k"]])
    out = {m: np.array([[p[m][key] for p in r["per_k"]] for r in recs]) for m in METRICS}
    return ks, out, np.array([r["cka"][key] for r in recs])


def fig1_null(recs: list[dict], out: Path) -> None:
    ks, raw, cka_raw = curves(recs, "raw_score")
    _, mu0, cka_mu0 = curves(recs, "mu0")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.5 * len(METRICS), 2.9), sharey=True)
    for ax, (name, (label, color)) in zip(axes, METRICS.items(), strict=True):
        ax.axhline(1.0, color=INK_MUTED, lw=0.8)
        ax.plot(ks, raw[name].mean(0), color=color, label="raw")
        ax.plot(ks, mu0[name].mean(0), color=color, ls=":", label="null mean")
        ax.fill_between(ks, mu0[name].mean(0), raw[name].mean(0), color=color, alpha=0.15, lw=0)
        ax.axhline(cka_raw.mean(), color=INK, ls="--", lw=1.2, label="CKA raw")
        ax.axhline(cka_mu0.mean(), color=INK, ls=":", lw=1.2, label="CKA null mean")
        ax.set_title(label, color=INK)
        p7.xaxis(ax)
    axes[0].set_ylabel("max over layer pairs")
    p7.legend(fig, axes[0], ncol=4)
    fig.suptitle(
        f"Observed score and permutation-null mean, mean over {len(recs)} pairs",
        y=1.2,
        color=INK,
    )
    p7.save(fig, out, "fig1_null")


def fig2_calibrated(recs: list[dict], out: Path) -> None:
    ks, g, cka_g = curves(recs, "g_score")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    p7.panel(ax, ks, {m: g[m] for m in BOUNDED}, cka_g)
    ax.set_ylabel("calibrated g")
    p7.legend(fig, ax)
    fig.suptitle(
        f"Null-calibrated scores vs k, {len(recs)} pairs (band = min..max)", y=1.2, color=INK
    )
    p7.save(fig, out, "fig2_calibrated")


def fig3_families(recs: list[dict], out: Path) -> None:
    ks, g, _ = curves(recs, "g_score")
    fams = np.array([family(r["lvm"]) for r in recs])
    shown = BOUNDED
    fig, axes = plt.subplots(1, len(shown), figsize=(3.0 * len(shown), 3.0), sharey=True)
    for ax, name in zip(axes, shown, strict=True):
        for f, (fam, _) in enumerate(FAMILIES):
            if (fams == f).any():
                ax.plot(ks, g[name][fams == f].mean(0), color=SERIES[f], lw=1.5, label=fam)
        ax.set_title(METRICS[name][0], color=INK)
        p7.xaxis(ax)
    axes[0].set_ylabel("calibrated g")
    p7.legend(fig, axes[0], ncol=5)
    fig.suptitle("Calibrated g by vision family (mean over LLMs)", y=1.15, color=INK)
    p7.save(fig, out, "fig3_families")


def table(recs: list[dict]) -> None:
    """Markdown tables at selected k: raw, null mean and calibrated g, mean over pairs."""
    ks = np.array([p["k"] for p in recs[0]["per_k"]])
    cols = [i for i, k in enumerate(ks) if k in p7.TABLE_KS]
    for key, title in (("raw_score", "raw"), ("mu0", "null mean"), ("g_score", "calibrated g")):
        _, vals, cka = curves(recs, key)
        print(f"\n{title} (CKA {cka.mean():.3f})")
        print("| k | " + " | ".join(str(ks[i]) for i in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for name, (label, _) in METRICS.items():
            row = vals[name].mean(0)
            print(f"| {label} | " + " | ".join(f"{row[i]:.2f}" for i in cols) + " |")
    _, p, cka_p = curves(recs, "p_value")
    print("\nfraction of pairs with p < 0.05")
    for name, (label, _) in METRICS.items():
        frac = (p[name] < 0.05).mean(0)
        print(f"| {label} | " + " | ".join(f"{frac[i]:.2f}" for i in cols) + " |")
    print(f"CKA: {(cka_p < 0.05).mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    path = args.root / "calibrated_pairs.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    out = args.root / "figures"
    fig1_null(recs, out)
    fig2_calibrated(recs, out)
    fig3_families(recs, out)
    table(recs)


if __name__ == "__main__":
    main()
