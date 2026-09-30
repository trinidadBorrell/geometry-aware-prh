"""One-panel figure for the Aristotelian / platonic-rep CKNNA issue, from exp-008 output.

    uv run python experiments/exp-008-cknna-null/issue_figure.py --root results/exp-008-cknna-null

Mean over all (LLM, ViT) pairs of the max over layer pairs vs k: observed (solid) and
permutation-null mean (dotted) for the released implementation, Eq. 36 and the proposed
centre-then-mask CKNNA. Band = min..max over pairs of the observed score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SHOWN = {
    "code": ("released implementation (platonic-rep / Aristotelian)", "#eda100"),
    "eq36": ("Aristotelian paper, Eq. 36", "#e87ba4"),
    "centred": ("proposed: centre (HKH), then mask", "#1baf7a"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recs = [json.loads(x) for x in (args.root / "calibrated_pairs.jsonl").read_text().splitlines()]
    ks = np.array([p["k"] for p in recs[0]["per_k"]])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(1.0, color="#52514e", lw=0.8)
    for name, (label, color) in SHOWN.items():
        raw = np.array([[p[name]["raw_score"] for p in r["per_k"]] for r in recs])
        null = np.array([[p[name]["mu0"] for p in r["per_k"]] for r in recs])
        ax.fill_between(ks, raw.min(0), raw.max(0), color=color, alpha=0.15, lw=0)
        ax.plot(ks, raw.mean(0), color=color, lw=2, label=label)
        ax.plot(ks, null.mean(0), color=color, lw=1.5, ls=":")
    ax.plot([], [], color="#52514e", ls=":", label="permutation null (mean of 200)")
    ax.set_xlabel("k (n = 1024)")
    ax.set_ylabel("CKNNA (max over layer pairs)")
    ax.set_title(f"CKNNA vs k, {len(recs)} (LLM, ViT) pairs of the PRH val set", fontsize=10)
    ax.grid(color="#e4e3dd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    out = args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "issue_cknna.png", dpi=200, bbox_inches="tight")
    print(f"saved {out / 'issue_cknna.png'}")


if __name__ == "__main__":
    main()
