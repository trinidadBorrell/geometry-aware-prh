#!/usr/bin/env python3
"""Fit and report structured anisotropic metrics. Historical result directories are not written."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--stage", choices=("prepare", "smoke", "fit", "eval", "report", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    repo = args.repo or Path(__file__).resolve().parents[1]
    from prh_replication.structured_metric_study import run

    run(args.work, repo, args.stage, force=args.force, workers=max(1, int(args.workers)))


if __name__ == "__main__":
    main()
