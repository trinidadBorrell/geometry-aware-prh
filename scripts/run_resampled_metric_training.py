#!/usr/bin/env python3
"""Full-gallery versus quarter-gallery anisotropic CKA. Does not rerun metric stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.resampled_metric_training import run


def parse_args():
    parser = argparse.ArgumentParser(description="Resampled one-sided metric training")
    parser.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument("--stage", default="all", choices=("smoke", "fit", "eval", "report", "all"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(work=Path(args.work), repo=Path(args.repo), stage=args.stage, force=args.force)
