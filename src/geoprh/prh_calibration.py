"""Null-calibrated PRH alignment: permutation calibration of the max-over-layers score.

Uses Aristotelian's calibration directly (`aristotelian.experiments.layerwise_engine`), which
implements Algorithm 2 of Gröger et al.: permute the pairing between modalities, recompute the
whole layer x layer score matrix, take its max, and compare the observed max against that null.
Reads activations from the shared cache and writes payloads in the format Aristotelian's own
plotting expects (`prh_alignment*.npy`), so its figures work unchanged:

    uv run python -m geoprh.prh_calibration --out <assets dir> --permutations 200
    cd Aristotelian && uv run python -m scripts.plots.experiments \\
        --sections prh_alignment --assets-dir <assets dir>

Rows are padded to the full 12-model "val" LLM list (NaN for models we have no activations
for), because the upstream plotting takes its labels from `get_models("val")`.

GPU is used when available: the null loop is dense elementwise work over n x n Grams.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aristotelian.experiments.layerwise_engine import (
    build_gram_cache,
    build_knn_cache_with_indices,
    compute_alignment_gated_cka_cached,
    compute_alignment_gated_knn_cached,
)
from aristotelian.experiments.multiple import bh_fdr
from aristotelian.prh.preprocess import prepare_features
from aristotelian.prh.prh_models import get_models
from geoprh import activation_cache as ac
from geoprh.prh_alignment import _git_sha, available

# payload file name -> (metric key in our json, k)
METRICS = {
    "mutual_knn": ("prh_alignment.npy", 10),
    "cka_lin": ("prh_alignment_cka_lin.npy", None),
}


def _layers(model: str, variant: ac.Variant, root: Path, q: float, device: str):
    payload = ac.load_model(root, model, variant)
    if payload is None:
        raise FileNotFoundError(f"{model} not fully cached under {root}")
    return prepare_features(payload["feats"], q=q, exact=False, device=device)


def run(
    *,
    root: Path,
    out: Path,
    modelset: str,
    permutations: int,
    alpha: float,
    q_outlier: float,
    topk: int,
    seed: int,
    device: str,
) -> None:
    llms, lvms, info = available(modelset, root)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "calibrated_pairs.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            done[(rec["llm"], rec["lvm"])] = rec

    text_variant = ac.Variant(pool="avg", caption_idx=0)
    vis_variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    total = len(llms) * len(lvms)
    with partial.open("a") as fh:
        for i, llm in enumerate(llms):
            lang = _layers(llm, text_variant, root, q_outlier, device)
            lang_layers, lang_knn, _ = build_knn_cache_with_indices(lang, topk=topk, normalize=True)
            lang_grams = build_gram_cache(lang, normalize=True, kernel="linear")
            for j, lvm in enumerate(lvms):
                if (llm, lvm) in done:
                    continue
                t0 = time.time()
                vis = _layers(lvm, vis_variant, root, q_outlier, device)
                _, vis_knn, _ = build_knn_cache_with_indices(vis, topk=topk, normalize=True)
                vis_grams = build_gram_cache(vis, normalize=True, kernel="linear")
                # metric(vision, language), as in platonic-rep's compute_alignment
                knn_res = compute_alignment_gated_knn_cached(
                    vis_knn,
                    lang_knn,
                    topk=topk,
                    num_permutations=permutations,
                    alpha=alpha,
                    seed=seed,
                )
                cka_res = compute_alignment_gated_cka_cached(
                    vis_grams,
                    lang_grams,
                    num_permutations=permutations,
                    alpha=alpha,
                    seed=seed,
                    unbiased=False,
                )
                rec = {
                    "llm": llm,
                    "lvm": lvm,
                    "mutual_knn": {k: v for k, v in knn_res.items()},
                    "cka_lin": {k: v for k, v in cka_res.items()},
                    "seconds": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[(llm, lvm)] = rec
                idx = i * len(lvms) + j + 1
                print(
                    f"[{idx}/{total}] {llm} x {lvm} "
                    f"mNN raw={knn_res['raw_score']:.3f} gated={knn_res['g_score']:.3f} "
                    f"p={knn_res['p_value']:.4f} | "
                    f"CKA raw={cka_res['raw_score']:.3f} gated={cka_res['g_score']:.3f} "
                    f"({rec['seconds']}s)",
                    flush=True,
                )
                del vis, vis_knn, vis_grams
            del lang, lang_layers, lang_knn, lang_grams
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    _write_payloads(done, llms, lvms, info, out, permutations, alpha, q_outlier, topk, seed, root)


def _write_payloads(
    done, llms, lvms, info, out: Path, permutations, alpha, q_outlier, topk, seed, root: Path
) -> None:
    """Write one Aristotelian-format payload per metric, padded to the full val LLM list."""
    all_llms, all_lvms = get_models("val", modality="all")
    rows = [all_llms.index(m) if m in all_llms else None for m in llms]
    cols = [all_lvms.index(m) if m in all_lvms else None for m in lvms]
    shape = (len(all_llms), len(all_lvms))

    for metric, (fname, k) in METRICS.items():
        arrays = {
            key: np.full(shape, np.nan)
            for key in ("scores", "gated", "pvalues", "taus", "mu0", "sd0")
        }
        indices = np.full((*shape, 2), -1, dtype=int)
        for (llm, lvm), rec in done.items():
            r, c = rows[llms.index(llm)], cols[lvms.index(lvm)]
            if r is None or c is None:
                continue
            res = rec[metric]
            arrays["scores"][r, c] = res["raw_score"]
            arrays["gated"][r, c] = res["g_score"]
            arrays["pvalues"][r, c] = res["p_value"]
            arrays["taus"][r, c] = res["tau_alpha"]
            arrays["mu0"][r, c] = res["mu0"]
            arrays["sd0"][r, c] = res["sd0"]
            indices[r, c] = res["best_indices"]

        finite = np.isfinite(arrays["pvalues"])
        fdr_threshold, mask_flat = bh_fdr(arrays["pvalues"][finite].ravel(), alpha=alpha)
        fdr_mask = np.zeros(shape, dtype=bool)
        fdr_mask[finite] = mask_flat
        payload = {
            **arrays,
            "indices": indices,
            "metric": metric,
            "fdr_threshold": fdr_threshold,
            "fdr_mask": fdr_mask,
            "k": k,
            "llms": llms,
            "lvms": lvms,
            "models": info,
            "num_permutations": permutations,
            "alpha": alpha,
            "q_outlier": q_outlier,
            "topk": topk,
            "seed": seed,
            "git_sha": _git_sha(),
            "activations_root": str(root),
            "note": "rows/cols are the full val modelset; NaN where activations are missing",
        }
        np.save(out / fname, payload)
        print(f"saved {out / fname}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--q-outlier", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"device: {args.device}, permutations: {args.permutations}")
    run(
        root=args.root or ac.default_root(),
        out=args.out,
        modelset=args.modelset,
        permutations=args.permutations,
        alpha=args.alpha,
        q_outlier=args.q_outlier,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
