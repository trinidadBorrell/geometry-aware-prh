"""Print platonic-rep (raw) vs Aristotelian (raw / calibrated) PRH alignment side by side.

Example:
    uv run python experiments/prh_small/compare.py --root results/prh_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aristotelian.prh.prh_models import get_models

PAIR_DIR = Path("minhuh/prh/{modelset}/language_pool-avg_prompt-False_vision_pool-cls_prompt-False")
# platonic-rep metric name -> Aristotelian metric name
METRICS = {"mutual_knn": "mutual_knn", "cka": "cka_lin"}


def _filename(metric: str, k: int) -> str:
    return f"{metric}_k{k}.npy" if "knn" in metric else f"{metric}.npy"


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  (missing: {path})")
        return None
    return np.load(path, allow_pickle=True).item()


def _short(name: str) -> str:
    return name.split("/")[-1].replace("_patch", "_p").replace("_224", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="results/prh_small")
    parser.add_argument("--modelset", default="small")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    llms, lvms = get_models(args.modelset)
    pair_dir = Path(str(PAIR_DIR).format(modelset=args.modelset))
    root = Path(args.root)

    for p_metric, a_metric in METRICS.items():
        print(f"\n=== {p_metric} (platonic) / {a_metric} (aristotelian) ===")
        plat = _load(root / "platonic/alignment" / pair_dir / _filename(p_metric, args.k))
        arist = _load(root / "aristotelian/alignment" / pair_dir / _filename(a_metric, args.k))
        if plat is None and arist is None:
            continue
        header = f"{'language':<14}{'vision':<28}"
        header += f"{'platonic':>10}" if plat else ""
        header += f"{'arist_raw':>11}{'calibrated':>11}{'p':>8}" if arist else ""
        print(header)
        for i, llm in enumerate(llms):
            for j, lvm in enumerate(lvms):
                row = f"{_short(llm):<14}{_short(lvm):<28}"
                if plat:
                    row += f"{plat['scores'][i, j]:>10.3f}"
                if arist:
                    row += (
                        f"{arist['scores'][i, j]:>11.3f}{arist['gated'][i, j]:>11.3f}"
                        f"{arist['pvalues'][i, j]:>8.3f}"
                    )
                print(row)


if __name__ == "__main__":
    main()
