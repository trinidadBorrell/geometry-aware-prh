#!/usr/bin/env python3
"""Plot identity, one-sided, shared, and separate holdout excess for each model pair.

Reads results/structured_metric_factors/joined.json. Does not refit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.condition_pair_plots import write_condition_pair_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    parser.add_argument("--joined", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    joined = Path(args.joined) if args.joined else Path(args.work) / "results" / "structured_metric_factors" / "joined.json"
    out = Path(args.out) if args.out else joined.parent / "figures" / "by_pair"
    rows = json.loads(joined.read_text())
    written = write_condition_pair_figures(rows, out)
    for path in written:
        print(path, flush=True)


if __name__ == "__main__":
    main()
