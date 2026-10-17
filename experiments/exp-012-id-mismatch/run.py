"""exp-012 (WP1): calibrated alignment of every layer pair, for every model pair of the PRH set.

Data: the PRH `wit_1024` activations (LLM: mean-pooled caption, every hidden state; ViT: CLS
token, every block), PRH preprocessing (q = 0.95 clamp, l2 norm). Model pairs, by track:
- T1: LLM x LLM (confound-free: same captions, same pooling)
- T2: ViT x ViT (confound-free: CLS throughout)
- T3: ViT x LLM (the PRH pairs; mean pooling vs CLS)

For each pair, mutual kNN (k = 10) and linear CKA on all layer pairs, each cell calibrated
against a 200-permutation null (`geoprh.layer_calibration`). One .npz per pair under
`pairs/`, plus a summary line per pair in `pairs.jsonl`. GPU recommended (all models' masks and
Gram matrices are kept on the device, ~6 GB at n = 1024).

    uv run python experiments/exp-012-id-mismatch/run.py --out <dir>
    uv run python experiments/exp-012-id-mismatch/run.py --out <dir> --tracks T1 --permutations 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from aristotelian.prh.preprocess import prepare_features
from geoprh import activation_cache as ac
from geoprh import layer_calibration as lc

LLMS = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
    "openlm-research/open_llama_13b",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",
]
VITS = [
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b",
    "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k",
    "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
]


def variant(model: str) -> ac.Variant:
    language = model in LLMS
    return ac.Variant(
        dataset="minhuh/prh",
        subset="wit_1024",
        pool="avg" if language else "cls",
        caption_idx=0 if language else None,
        modality="language" if language else "vision",
    )


def pairs(tracks: list[str]) -> list[tuple[str, str, str]]:
    out = []
    if "T1" in tracks:
        out += [("T1", a, b) for a, b in itertools.combinations(LLMS, 2)]
    if "T2" in tracks:
        out += [("T2", a, b) for a, b in itertools.combinations(VITS, 2)]
    if "T3" in tracks:
        out += [("T3", v, m) for v in VITS for m in LLMS]
    return out


def slug(model: str) -> str:
    return model.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--tracks", default="T1,T2,T3")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root or ac.default_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (args.out / "pairs").mkdir(parents=True, exist_ok=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    meta = {**{k: str(v) for k, v in vars(args).items()}, "git_sha": sha.strip(), "device": device}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    todo = [p for p in pairs(args.tracks.split(",")) if not (args.out / "pairs" / name(p)).exists()]
    caches = {}
    for model in sorted({m for _, a, b in todo for m in (a, b)}):
        payload = ac.load_model(root, model, variant(model))
        if payload is None:
            print(f"[skip] {model}: not cached")
            continue
        feats = prepare_features(payload["feats"], q=0.95, exact=False, device=device)
        caches[model] = lc.build_cache(feats, k=args.k)
        print(f"{model}: {tuple(payload['feats'].shape)} cached", flush=True)
        del payload, feats

    with (args.out / "pairs.jsonl").open("a") as fout:
        for track, a, b in todo:
            if a not in caches or b not in caches:
                continue
            t0 = time.time()
            res = lc.calibrated_pair(
                caches[a], caches[b], args.permutations, args.alpha, seed=args.seed
            )
            arrays = {f"{m}_{s}": v.cpu().numpy() for m, r in res.items() for s, v in r.items()}
            np.savez(args.out / "pairs" / name((track, a, b)), **arrays)
            rec = {"track": track, "a": a, "b": b, "La": caches[a].num_layers}
            rec["Lb"] = caches[b].num_layers
            for m, r in res.items():
                i, j = np.unravel_index(int(r["score"].argmax()), r["score"].shape)
                rec[m] = {
                    "max_score": float(r["score"].max()),
                    "best": [int(i), int(j)],
                    "max_g": float(r["g"].max()),
                    "frac_sig": float((r["p"] < args.alpha).float().mean()),
                }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(
                f"{track} {a} x {b}: {time.time() - t0:.1f}s, "
                f"mkNN max {rec['mutual_knn']['max_score']:.3f}, "
                f"CKA max {rec['cka_lin']['max_score']:.3f}",
                flush=True,
            )


def name(pair: tuple[str, str, str]) -> str:
    track, a, b = pair
    return f"{track}__{slug(a)}__{slug(b)}.npz"


if __name__ == "__main__":
    main()
