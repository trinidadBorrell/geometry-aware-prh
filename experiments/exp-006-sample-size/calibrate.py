"""exp-006: null-calibrated CKA and mutual kNN on sample sets of different size.

For each (LLM, ViT) pair and each sample set, scores every layer pair, takes the max over layer
pairs, and calibrates that max with Groger et al.'s permutation null (Algorithm 2: permute the
image-caption pairing, recompute the whole layer x layer matrix, take its max). Reports the raw
score, the null mean mu0, the null 95% quantile tau and the calibrated score
g = (raw - tau) / (1 - tau) (0 when raw <= tau), exactly as `geoprh.prh_calibration`.

Sample sets, all drawn from the same 10240 extracted samples (see extract.py):
- disjoint:  rows [1024 s, 1024 (s+1)) for s = 0..4 (five non-overlapping 1024-sample sets)
- nested:    the first 2048, 4096 and 10240 rows (the first 1024 is disjoint set 0)

The CKA null is skipped above --cka-null-max-n (default 4096): at n = 10240 only raw CKA is
reported for now.

Preprocessing per sample set as platonic-rep / exp-003: q=0.95 outlier clamp, l2 norm.
Mutual kNN (k=10) uses Aristotelian's index-based calibration. Linear CKA is computed in feature
space, CKA = ||Xc^T Yc||^2 / (||Xc^T Xc|| ||Yc^T Yc||) with column-centred features, which
equals the Gram-matrix form and lets a permutation act on the rows of Y. That avoids n x n Grams
(400 MB each at n = 10240). Permutations come from Aristotelian's `batched_perms` with the same
seed, so the null matches `compute_alignment_gated_cka_cached` (checked in tests).

    uv run python experiments/exp-006-sample-size/calibrate.py --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from aristotelian import compute_nearest_neighbors
from aristotelian.experiments.layerwise_engine import (
    _build_gated_summary,
    compute_alignment_gated_knn_cached,
)
from aristotelian.metrics.aggregation import agg_max
from aristotelian.metrics.utils import batched_perms
from aristotelian.prh.preprocess import prepare_features
from geoprh import activation_cache as ac
from geoprh.prh_alignment import _git_sha

_spec = importlib.util.spec_from_file_location("extract", Path(__file__).with_name("extract.py"))
_extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract)
LLMS, LVMS, variant = _extract.LLMS, _extract.LVMS, _extract.variant

N_TOTAL = 10240
SETS = {f"disjoint{s}": (1024 * s, 1024 * (s + 1)) for s in range(5)}
SETS.update({f"first{n}": (0, n) for n in (2048, 4096, 10240)})
KNN_K = 10


def _layers(feats: torch.Tensor, rows: tuple[int, int], device: str) -> list[torch.Tensor]:
    x = prepare_features(feats[rows[0] : rows[1]], q=0.95, exact=False, device=device)
    return [F.normalize(x[:, i], dim=-1) for i in range(x.shape[1])]


def cka_calibrated(
    xs: list[torch.Tensor], ys: list[torch.Tensor], *, permutations: int, alpha: float, seed: int
) -> dict:
    """Groger-calibrated max-over-layers linear CKA, metric(x, y), in feature space."""
    xc = [x - x.mean(0) for x in xs]
    yc = [y - y.mean(0) for y in ys]
    x_norm = torch.stack([torch.linalg.matrix_norm(x.T @ x) for x in xc])
    y_norm = torch.stack([torch.linalg.matrix_norm(y.T @ y) for y in yc])
    x_all = torch.cat(xc, dim=1)  # [n, sum dx]
    bounds = torch.tensor([0] + [x.shape[1] for x in xc], device=x_all.device).cumsum(0)

    def scores(y_perm: torch.Tensor, j: int) -> torch.Tensor:
        """CKA of every x layer with y layer j (rows already permuted); [Lx]."""
        m = x_all.T @ y_perm  # [sum dx, dy]
        sq = (m * m).sum(1)
        hsic = torch.stack([sq[bounds[i] : bounds[i + 1]].sum() for i in range(len(xc))])
        return hsic / (x_norm * y_norm[j])

    S = torch.stack([scores(y, j) for j, y in enumerate(yc)], dim=1)  # [Lx, Ly]
    agg = agg_max(S, return_indices=True)
    best = (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    if permutations == 0:  # raw score only
        return {"raw_score": float(agg.value), "best_indices": best}
    null = []
    n = xs[0].shape[0]
    for perms in batched_perms(n, permutations, device=xs[0].device, seed=seed, chunk_size=16):
        for p in perms:
            s_perm = torch.stack([scores(y[p], j) for j, y in enumerate(yc)], dim=1)
            null.append(float(s_perm.max()))
    return _build_gated_summary(null, T_obs=float(agg.value), best_indices=best, alpha=alpha)


def knn_calibrated(xs, ys, *, permutations: int, alpha: float, seed: int) -> dict:
    kx = [compute_nearest_neighbors(x, topk=KNN_K) for x in xs]
    ky = [compute_nearest_neighbors(y, topk=KNN_K) for y in ys]
    return compute_alignment_gated_knn_cached(
        kx, ky, topk=KNN_K, num_permutations=permutations, alpha=alpha, seed=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--sets", default=",".join(SETS))
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cka-null-max-n", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.root or ac.default_root()
    sets = args.sets.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "calibrated.jsonl"
    done = set()
    if path.exists():
        done = {(r["llm"], r["lvm"], r["set"]) for r in map(json.loads, path.open())}
    meta = {"git_sha": _git_sha(), "sets": {s: SETS[s] for s in sets}, **vars(args)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    kw = {"permutations": args.permutations, "alpha": args.alpha, "seed": args.seed}
    with path.open("a") as fh:
        for llm in LLMS:
            lang = ac.load_model(root, llm, variant(N_TOTAL, "language"))["feats"]
            for lvm in LVMS:
                vis = ac.load_model(root, lvm, variant(N_TOTAL, "vision"))["feats"]
                for name in sets:
                    if (llm, lvm, name) in done:
                        continue
                    t0 = time.time()
                    xs = _layers(vis, SETS[name], args.device)
                    ys = _layers(lang, SETS[name], args.device)
                    n = xs[0].shape[0]
                    cka_kw = {**kw, "permutations": kw["permutations"] * (n <= args.cka_null_max_n)}
                    res = {
                        "mutual_knn_k10": knn_calibrated(xs, ys, **kw),
                        "cka_lin": cka_calibrated(xs, ys, **cka_kw),
                    }
                    rec = {
                        "llm": llm,
                        "lvm": lvm,
                        "set": name,
                        "n": n,
                        "results": res,
                        "seconds": round(time.time() - t0, 1),
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(
                        f"{llm} x {lvm} [{name}] "
                        + "  ".join(
                            f"{k}: raw={r['raw_score']:.3f} mu0={r.get('mu0', float('nan')):.3f} "
                            f"g={r.get('g_score', float('nan')):.3f}"
                            for k, r in res.items()
                        )
                        + f" ({rec['seconds']}s)",
                        flush=True,
                    )
                    del xs, ys
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
