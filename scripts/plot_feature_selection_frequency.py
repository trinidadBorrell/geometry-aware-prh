#!/usr/bin/env python3
"""Plot original-coordinate selection frequency and ΔM from frozen repaired fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.feature_selection import OUT_NAME, run_feature_selection  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.work)
    out = Path(args.out) if args.out else None
    summary = run_feature_selection(work=work, repo=ROOT, out=out)
    dest = out or work / "results" / OUT_NAME
    print(f"wrote {dest}", flush=True)
    print(f"unavailable={len(summary.get('unavailable') or [])}", flush=True)


if __name__ == "__main__":
    main()
