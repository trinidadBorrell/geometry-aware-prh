"""Null-calibrated PRH alignment: permutation calibration of the max-over-layers score.

Uses Aristotelian's calibration directly (`aristotelian.experiments.layerwise_engine`), which
implements Algorithm 2 of Gröger et al.: permute the pairing between modalities, recompute the
whole layer x layer score matrix, take its max, and compare the observed max against that null.
Reads activations from the shared cache and writes payloads in the format Aristotelian's own
plotting expects (`prh_alignment*.npy`), so its figures work unchanged:

    uv run python -m geoprh.prh_calibration --out <dir> --metrics mutual_knn,cka_lin
    uv run python -m geoprh.prh_calibration --out <dir> --metrics mutual_knn,cknna --k 200 \\
        --models reduced
    cd Aristotelian && uv run python -m scripts.plots.experiments \\
        --sections prh_alignment --assets-dir <dir>

Rows are padded to the full 12-model "val" LLM list (NaN for models we have no activations
for, or that a reduced grid skips), because the upstream plotting takes its labels from
`get_models("val")`.

Mutual kNN uses the mask-based calibration, whose cost does not depend on k; the index-based
variant allocates an n x k x k tensor (256M entries at k=500). GPU is used when available.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aristotelian.experiments.layerwise_engine import (
    build_gram_cache,
    build_knn_cache,
    compute_alignment_gated_cached,
    compute_alignment_gated_cka_cached,
    compute_alignment_gated_cknna_cached,
)
from aristotelian.experiments.multiple import bh_fdr
from aristotelian.prh.preprocess import prepare_features
from aristotelian.prh.prh_models import get_models
from geoprh import activation_cache as ac
from geoprh.prh_alignment import _git_sha, available

# Reduced grid: LLMs spanning 0.56B -> 13B across three families, ViTs covering every
# vision family at two sizes each.
REDUCED_LLMS = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_13b",
    "huggyllama/llama-13b",
]
REDUCED_LVMS = [
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_huge_patch14_clip_224.laion2b",
]


@dataclass(frozen=True)
class Spec:
    """One calibrated metric: its name, neighbourhood size and payload file."""

    metric: str
    k: int | None

    @property
    def key(self) -> str:
        return f"{self.metric}_k{self.k}" if self.k else self.metric

    @property
    def filename(self) -> str:
        if self.metric == "mutual_knn" and self.k == 10:
            return "prh_alignment.npy"  # upstream's default name
        return f"prh_alignment_{self.key}.npy"


def parse_specs(metrics: str, ks: str) -> list[Spec]:
    k_values = [int(k) for k in ks.split(",") if k.strip()]
    specs = []
    for metric in (m.strip() for m in metrics.split(",") if m.strip()):
        if metric in ("mutual_knn", "cknna"):
            specs.extend(Spec(metric, k) for k in k_values)
        elif metric == "cka_lin":
            specs.append(Spec(metric, None))
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return specs


def _layers(model: str, variant: ac.Variant, root: Path, q: float, device: str):
    payload = ac.load_model(root, model, variant)
    if payload is None:
        raise FileNotFoundError(f"{model} not fully cached under {root}")
    return prepare_features(payload["feats"], q=q, exact=False, device=device)


def _caches(feats, specs: list[Spec]) -> dict:
    """Per-model caches: kNN masks per requested k, Gram matrices when needed."""
    cache = {"masks": {}, "grams": None}
    for k in {s.k for s in specs if s.metric == "mutual_knn"}:
        layers, masks = build_knn_cache(feats, topk=k, normalize=True)
        cache["masks"][k] = (layers, masks)
    if any(s.metric in ("cka_lin", "cknna") for s in specs):
        cache["grams"] = build_gram_cache(feats, normalize=True, kernel="linear")
    return cache


def _calibrate(spec: Spec, vis: dict, lang: dict, *, permutations: int, alpha: float, seed: int):
    """metric(vision, language), matching platonic-rep's compute_alignment argument order."""
    if spec.metric == "mutual_knn":
        vis_layers, vis_masks = vis["masks"][spec.k]
        lang_layers, lang_masks = lang["masks"][spec.k]
        return compute_alignment_gated_cached(
            vis_layers,
            lang_layers,
            vis_masks,
            lang_masks,
            topk=spec.k,
            num_permutations=permutations,
            alpha=alpha,
            seed=seed,
        )
    if spec.metric == "cknna":
        return compute_alignment_gated_cknna_cached(
            vis["grams"],
            lang["grams"],
            topk=spec.k,
            num_permutations=permutations,
            alpha=alpha,
            seed=seed,
        )
    return compute_alignment_gated_cka_cached(
        vis["grams"],
        lang["grams"],
        num_permutations=permutations,
        alpha=alpha,
        seed=seed,
        unbiased=False,
    )


def run(
    *,
    root: Path,
    out: Path,
    modelset: str,
    models: str,
    specs: list[Spec],
    permutations: int,
    alpha: float,
    q_outlier: float,
    seed: int,
    device: str,
) -> None:
    llms, lvms, info = available(modelset, root)
    if models == "reduced":
        llms = [m for m in llms if m in REDUCED_LLMS]
        lvms = [m for m in lvms if m in REDUCED_LVMS]
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "calibrated_pairs.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            done.setdefault((rec["llm"], rec["lvm"]), {}).update(rec["results"])

    text_variant = ac.Variant(pool="avg", caption_idx=0)
    vis_variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    wanted = [s.key for s in specs]
    total = len(llms) * len(lvms)
    print(f"{len(llms)} LLMs x {len(lvms)} ViTs = {total} pairs, metrics: {', '.join(wanted)}")

    with partial.open("a") as fh:
        for i, llm in enumerate(llms):
            if all(set(wanted) <= set(done.get((llm, v), {})) for v in lvms):
                continue
            lang_feats = _layers(llm, text_variant, root, q_outlier, device)
            lang = _caches(lang_feats, specs)
            for j, lvm in enumerate(lvms):
                missing = [s for s in specs if s.key not in done.get((llm, lvm), {})]
                if not missing:
                    continue
                t0 = time.time()
                vis_feats = _layers(lvm, vis_variant, root, q_outlier, device)
                vis = _caches(vis_feats, missing)
                results = {}
                for spec in missing:
                    results[spec.key] = _calibrate(
                        spec, vis, lang, permutations=permutations, alpha=alpha, seed=seed
                    )
                fh.write(
                    json.dumps(
                        {
                            "llm": llm,
                            "lvm": lvm,
                            "results": results,
                            "seconds": round(time.time() - t0, 1),
                        }
                    )
                    + "\n"
                )
                fh.flush()
                done.setdefault((llm, lvm), {}).update(results)
                summary = "  ".join(
                    f"{key}: raw={r['raw_score']:.3f} gated={r['g_score']:.3f} "
                    f"tau={r['tau_alpha']:.3f} p={r['p_value']:.4f}"
                    for key, r in results.items()
                )
                print(
                    f"[{i * len(lvms) + j + 1}/{total}] {llm} x {lvm} {summary} "
                    f"({time.time() - t0:.1f}s)",
                    flush=True,
                )
                del vis_feats, vis
            del lang_feats, lang
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    _write_payloads(
        done,
        llms,
        lvms,
        info,
        specs,
        out,
        permutations=permutations,
        alpha=alpha,
        q_outlier=q_outlier,
        seed=seed,
        root=root,
    )


def _write_payloads(
    done, llms, lvms, info, specs, out: Path, *, permutations, alpha, q_outlier, seed, root: Path
) -> None:
    """One Aristotelian-format payload per metric, padded to the full val model lists."""
    all_llms, all_lvms = get_models("val", modality="all")
    shape = (len(all_llms), len(all_lvms))

    for spec in specs:
        arrays = {
            key: np.full(shape, np.nan)
            for key in ("scores", "gated", "pvalues", "taus", "mu0", "sd0")
        }
        indices = np.full((*shape, 2), -1, dtype=int)
        for (llm, lvm), results in done.items():
            res = results.get(spec.key)
            if res is None or llm not in all_llms or lvm not in all_lvms:
                continue
            r, c = all_llms.index(llm), all_lvms.index(lvm)
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
            "metric": spec.metric,
            "fdr_threshold": fdr_threshold,
            "fdr_mask": fdr_mask,
            "k": spec.k,
            "llms": llms,
            "lvms": lvms,
            "models": info,
            "num_permutations": permutations,
            "alpha": alpha,
            "q_outlier": q_outlier,
            "seed": seed,
            "git_sha": _git_sha(),
            "activations_root": str(root),
            "note": "rows/cols are the full val modelset; NaN where a pair was not computed",
        }
        np.save(out / spec.filename, payload)
        print(f"saved {out / spec.filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--models", default="all", choices=["all", "reduced"])
    parser.add_argument("--metrics", default="mutual_knn,cka_lin")
    parser.add_argument("--k", default="10", help="comma-separated k for kNN-based metrics")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--q-outlier", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    specs = parse_specs(args.metrics, args.k)
    print(f"device: {args.device}, permutations: {args.permutations}")
    run(
        root=args.root or ac.default_root(),
        out=args.out,
        modelset=args.modelset,
        models=args.models,
        specs=specs,
        permutations=args.permutations,
        alpha=args.alpha,
        q_outlier=args.q_outlier,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
