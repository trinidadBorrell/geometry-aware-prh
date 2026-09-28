"""exp-008: permutation-null calibration of the CKNNA variants, mutual kNN and CKA at every k.

Groger et al.'s Algorithm 2, as `geoprh.prh_calibration`: permute the image-caption pairing,
recompute the whole layer x layer score matrix, take its max, and compare the observed max with
that null. Reports the raw score, null mean mu0 and sd sd0, the 95% cutoff tau, the p-value and
the calibrated score g = (raw - tau) / (1 - tau) (0 when raw <= tau), from Aristotelian's
`_build_gated_summary`. The same permutations (Aristotelian `batched_perms`, CPU, seed) are used
for every metric and every k. Metrics as exp-007 (`geoprh.cknna_variants`); g assumes a maximum
of 1, which `code` and `eq36` do not respect.

    uv run python experiments/exp-008-cknna-null/calibrate.py --out <dir> --permutations 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from aristotelian.experiments.layerwise_engine import _build_gated_summary
from aristotelian.metrics.utils import batched_perms
from geoprh import activation_cache as ac
from geoprh import cknna_variants as cv
from geoprh.prh_alignment import _git_sha, available

_spec = importlib.util.spec_from_file_location(
    "sweep", Path(__file__).parents[1] / "exp-007-cknna-variants" / "sweep.py"
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def make_perms(n: int, permutations: int, seed: int) -> torch.Tensor:
    return torch.cat(list(batched_perms(n, permutations, device="cpu", seed=seed, chunk_size=8)))


def _summary(obs: torch.Tensor, null: list[float], alpha: float) -> dict:
    raw, layers = sweep.best(obs)
    res = _build_gated_summary(null, T_obs=raw, best_indices=tuple(layers), alpha=alpha)
    return {**res, "best_indices": layers}


def calibrate_pair(
    a: cv.Stack, b: cv.Stack, ks: list[int], perms: torch.Tensor, alpha: float
) -> tuple[dict, dict[str, np.ndarray]]:
    """Calibrated max-over-layers scores of metric(a, b); perms is [P, n] on a's device."""
    nulls = {name: np.zeros((len(ks), len(perms)), np.float32) for name in cv.PER_K}
    per_k = []
    for t, k in enumerate(ks):
        ak, bk = a.at_k(k), b.at_k(k)
        obs = cv.per_k_scores(ak, bk, bk.right())
        for p, perm in enumerate(perms):
            for name, mat in cv.per_k_scores(ak, bk, bk.permuted(perm)).items():
                nulls[name][t, p] = float(mat.max())
        per_k.append(
            {"k": k, **{name: _summary(obs[name], nulls[name][t].tolist(), alpha) for name in obs}}
        )
        del ak, bk
    cka_null = np.array([float(cv.cka(a, b, perm).max()) for perm in perms], np.float32)
    nulls["cka"] = cka_null
    return {"per_k": per_k, "cka": _summary(cv.cka(a, b), cka_null.tolist(), alpha)}, nulls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--llms", default="all", help="'all' (cached modelset) or a list")
    parser.add_argument("--lvms", default="all", help="'all' (cached modelset) or a list")
    parser.add_argument("--ks", default=",".join(map(str, sweep.KS)))
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    cached_llms, cached_lvms, _ = available(args.modelset, root)
    llms = cached_llms if args.llms == "all" else args.llms.split(",")
    lvms = cached_lvms if args.lvms == "all" else args.lvms.split(",")

    (args.out / "nulls").mkdir(parents=True, exist_ok=True)
    partial = args.out / "calibrated_pairs.jsonl"
    done = set()
    if partial.exists():
        done = {(r["llm"], r["lvm"]) for r in map(json.loads, partial.read_text().splitlines())}
    todo = [(lm, v) for lm in llms for v in lvms if (lm, v) not in done]
    print(f"{len(todo)} pairs to go, {args.permutations} permutations, k = {ks}", flush=True)
    perms = None
    with partial.open("a") as fh:
        for c, llm in enumerate(llms):
            if all((llm, v) in done for v in lvms):
                continue
            lang = sweep.load_stack(root, llm, sweep.TEXT, args.device)
            for lvm in lvms:
                if (llm, lvm) in done:
                    continue
                t0 = time.time()
                vis = sweep.load_stack(root, lvm, sweep.VISION, args.device)
                if perms is None:
                    perms = make_perms(vis.n, args.permutations, args.seed).to(args.device)
                ks_n = [min(k, vis.n - 1) for k in ks]
                res, nulls = calibrate_pair(vis, lang, ks_n, perms, args.alpha)
                stem = f"{llm.replace('/', '__')}__x__{lvm}"
                np.savez_compressed(args.out / "nulls" / f"{stem}.npz", ks=np.array(ks_n), **nulls)
                rec = {
                    "llm": llm,
                    "lvm": lvm,
                    "n": vis.n,
                    **res,
                    "seconds": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                k500 = next(r for r in res["per_k"] if r["k"] >= min(500, vis.n - 1))
                info = "  ".join(
                    f"{name} {k500[name]['raw_score']:.2f}/{k500[name]['mu0']:.2f}"
                    for name in cv.CKNNA_VARIANTS
                )
                print(
                    f"[{c + 1}/{len(llms)}] {llm} x {lvm}: k={k500['k']} raw/null {info} "
                    f"({rec['seconds']}s)",
                    flush=True,
                )
                del vis
            del lang
    meta = {
        "ks": ks,
        "permutations": args.permutations,
        "alpha": args.alpha,
        "seed": args.seed,
        "perms": "aristotelian batched_perms, cpu, chunk 8",
        "git_sha": _git_sha(),
        "root": str(root),
        "device": args.device,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
