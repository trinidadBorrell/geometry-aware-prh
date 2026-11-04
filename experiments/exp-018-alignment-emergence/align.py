"""exp-018: calibrated alignment of LM training checkpoints with fixed ViTs and a final LLM.

Uses exp-020's alignment engine (`exp-020-lm-data-pilot/align.py`): features are searched in
`--roots` (this experiment's checkpoint features from exp-020 `extract_lm.py --revisions`,
exp-016's ViT features and geometry, exp-020's final-LM features for the references).

Pairs, for every checkpoint feature set `<repo>@<revision>__<caption>`:
- LV: checkpoint x each ViT in `--vits` (default DINOv2-g and CLIP ViT-H), both caption sets; at
  n = 10240 mutual kNN at every k in `--ks` (default 10, 30, 100, 300, 1000: 0.1-10% of the data)
  and linear CKA; at n = 1024 mutual kNN, CKA, ORC given density and density;
- LL: checkpoint x each reference LM in `--refs` (final models, same captions): within-modality
  emergence for comparison.

    uv run python experiments/exp-018-alignment-emergence/align.py \\
        --roots <exp018 out>,<exp016 out>,<exp020 out> --stages geometry,pairs
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "align020", HERE.parent / "exp-020-lm-data-pilot" / "align.py"
)
A20 = importlib.util.module_from_spec(_spec)
sys.modules["align020"] = A20
_spec.loader.exec_module(A20)

VITS = ["vit_giant_patch14_dinov2.lvd142m", "vit_huge_patch14_clip_224.laion2b"]
REFS = ["EleutherAI__pythia-1.4b"]
CKPT = re.compile(r"^(?P<model>.+)@(?P<rev>[^_]+)__(?P<caption>orig|gemini)__(?P<tag>d2s|d2l)$")


def specs_018(avail: set[str], vits: list[str], refs: list[str], captions: list[str]) -> list[dict]:
    ckpts = sorted({m.group(0).rsplit("__", 1)[0] for k in avail if (m := CKPT.match(k))})
    specs = []

    def add(group, a, b, tag, **extra):
        if f"{a}__{tag}" in avail and f"{b}__{tag}" in avail:
            pid = A20.pair_id(group, a, b, tag)
            specs.append({"id": pid, "group": group, "a": a, "b": b, "tag": tag, **extra})

    for ck, tag in itertools.product(ckpts, A20.TAGS):
        cap = ck.rsplit("__", 1)[1]
        if cap not in captions:
            continue
        for v in vits:
            add("LV", ck, f"{v}__cls", tag, caption=cap)
        for r in refs:
            add("LL", ck, f"{r}__{cap}", tag, caption=cap)
    return specs


def main() -> None:
    def extra(p):
        p.add_argument("--vits", default=",".join(VITS))
        p.add_argument("--refs", default=",".join(REFS), help="final LMs ('' for none)")
        p.add_argument("--captions", default="orig,gemini")

    args = A20.parse(__doc__.splitlines()[0], "10,30,100,300,1000", extra)
    roots = [Path(r) for r in args.roots.split(",")]
    store, out = A20.Store(roots), roots[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stages = args.stages.split(",")
    if "geometry" in stages:
        A20.M16.stage_geometry(out, args.workers, args.tmp)
    refs = [r for r in args.refs.split(",") if r]
    specs = specs_018(store.available(), args.vits.split(","), refs, args.captions.split(","))
    if args.only:
        specs = [s for s in specs if args.only in s["id"]]
    if "pairs" in stages:
        ks = [int(k) for k in args.ks.split(",")]
        A20.stage_pairs(
            specs, store, out, device, args.permutations, args.alpha, ks, args.keep_caches
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
