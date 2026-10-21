#!/usr/bin/env python3
"""PCA-axis diagonal ablation of full-rank CCA reliability metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from prh_replication.cca_metric_diagonal_ablation_study import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Work root containing features/ and results/")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run(work=args.work.resolve(), repo=args.repo.resolve(), force=args.force)
    print(summary.get("answers", summary), flush=True)


if __name__ == "__main__":
    main()
