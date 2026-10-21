#!/usr/bin/env python3
"""Feature vs PCA vs full CCA reliability metric comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from prh_replication.feature_vs_pca_metric_study import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run(work=args.work.resolve(), repo=args.repo.resolve(), force=args.force)
    print(summary.get("answers", summary), flush=True)


if __name__ == "__main__":
    main()
