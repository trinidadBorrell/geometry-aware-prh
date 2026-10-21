#!/usr/bin/env python3
"""Bounded CCA reliability procedure replication."""

from __future__ import annotations

import argparse
from pathlib import Path

from prh_replication.cca_reliability_replication_study import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sections", nargs="+", default=None, help="Subset of A B C D")
    args = parser.parse_args()
    summary = run(
        work=args.work.resolve(),
        repo=args.repo.resolve(),
        force=args.force,
        sections=args.sections,
    )
    print(summary.get("answers", summary), flush=True)


if __name__ == "__main__":
    main()
