"""exp-031 follow-up readout: capability slopes and the family gap, intact vs shuffled.

Spearman only sees the ranking, and a flat but monotone trend keeps a high Spearman. This
reports the size of the trend instead, per ViT and then the median over the 17 ViTs:
  - slope of excess on 1 - bpb over all 10 LLMs;
  - the same slope within BLOOMZ (5 sizes);
  - the gap: mean excess of the LLaMA-type LLMs minus mean excess of the BLOOMZ LLMs.

  uv run python experiments/exp-031-shuffled-capability/slopes.py \
      --out results/exp-031-shuffled-capability --stored results/prh_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("reext", "shufw0", "shufw1")
MEASURES = ("mutual_knn_k10", "cka_lin")


def load(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for m, v in r["results"].items():
            d[(m, r["llm"], r["lvm"])] = v["raw_score"] - v["mu0"]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stored", type=Path, required=True)
    args = ap.parse_args()
    info = json.loads((args.stored / "cross_modal.json").read_text())["models"]
    cond = {tf: load(args.out / tf / "calibrated_pairs.jsonl") for tf in CONDITIONS}
    llms = sorted({k[1] for k in cond["reext"]}, key=lambda m: -info[m]["bpb"])
    lvms = sorted({k[2] for k in cond["reext"]})
    cap = np.array([1 - info[m]["bpb"] for m in llms])
    bloomz = np.array(["bloomz" in m for m in llms])
    res: dict = {}
    for m in MEASURES:
        for tf, d in cond.items():
            x = np.array([[d[(m, lm, v)] for v in lvms] for lm in llms])
            slope = [np.polyfit(cap, x[:, j], 1)[0] for j in range(len(lvms))]
            slope_b = [np.polyfit(cap[bloomz], x[bloomz, j], 1)[0] for j in range(len(lvms))]
            gap = x[~bloomz].mean(0) - x[bloomz].mean(0)
            res[f"{m}/{tf}"] = {
                "slope_all": float(np.median(slope)),
                "slope_bloomz": float(np.median(slope_b)),
                "family_gap": float(np.median(gap)),
            }
            print(
                f"{m:15s} {tf:7s} slope all {np.median(slope):7.4f}  "
                f"slope BLOOMZ {np.median(slope_b):7.4f}  gap {np.median(gap):7.4f}"
            )
    (args.out / "slopes.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
