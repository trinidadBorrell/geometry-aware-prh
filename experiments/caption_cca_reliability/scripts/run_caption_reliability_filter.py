#!/usr/bin/env python3
"""Caption-reliability language filters. Does not train the underlying models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--stage", choices=("extract", "pilot", "full"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = args.repo or Path(__file__).resolve().parents[3]
    os.environ.setdefault("HF_HOME", str(args.work / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.work / "hf" / "hub"))
    from prh_replication.caption_reliability_study import run

    run(args.work, repo, args.stage, force=args.force)


if __name__ == "__main__":
    main()
