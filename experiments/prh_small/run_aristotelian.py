"""Run Aristotelian's null-calibrated PRH alignment on the small modelset.

Thin wrapper around `aristotelian.prh.run_prh_experiment`; the upstream CLI section
(`scripts.experiments.cli --sections prh_alignment`) is hard-wired to the full "val"
modelset, which is too large for quick replication.

Example:
    uv run python experiments/prh_small/run_aristotelian.py --device cuda --metrics mutual_knn,cka_lin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aristotelian.prh import run_prh_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modelset", default="small")
    parser.add_argument("--metrics", default="mutual_knn,cka_lin")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-permutations", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--q-outlier", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="results/prh_small/aristotelian")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        payload = run_prh_experiment(
            dataset="minhuh/prh",
            subset="wit_1024",
            modelset=args.modelset,
            modality_x="language",
            pool_x="avg",
            modality_y="vision",
            pool_y="cls",
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=str(out_dir),
            k=args.k,
            metric=metric,
            num_permutations=args.num_permutations,
            alpha=args.alpha,
            q_outlier=args.q_outlier,
            force_features=args.force_features,
        )
        with np.printoptions(precision=3, suppress=True):
            print(f"\n== {metric} (rows: language models, cols: vision models) ==")
            print("raw:\n", payload["scores"])
            print("calibrated (gated):\n", payload["gated"])
            print("p-values:\n", payload["pvalues"])


if __name__ == "__main__":
    main()
