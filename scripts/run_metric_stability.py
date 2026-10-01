#!/usr/bin/env python3
"""Bounded metric-stability study. Does not modify historical result directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.metric_stability_study import run


def parse_args():
    p = argparse.ArgumentParser(description="Fitting-mode, stability, and transfer study")
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--stage", default="all", choices=("audit", "smoke", "run", "eval", "report", "all"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(work=Path(args.work), repo=Path(args.repo), stage=args.stage, force=args.force)
