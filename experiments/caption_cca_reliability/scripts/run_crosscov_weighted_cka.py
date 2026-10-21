#!/usr/bin/env python3
"""Cross-covariance spectral reweighting evaluated by held-out linear CKA."""

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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = args.repo or Path(__file__).resolve().parents[3]
    os.environ.setdefault("HF_HOME", str(args.work / "hf"))
    from prh_replication.crosscov_weighted_cka_study import run

    run(args.work, repo, force=args.force)


if __name__ == "__main__":
    main()
