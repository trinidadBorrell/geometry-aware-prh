"""Diagnostics for the mutual-neighbour local CKA (exp-009), from run.py output.

    uv run python experiments/exp-009-local-cknna/diagnostics.py --root results/exp-009-local-cknna

1. small k: across the layer x layer grid, how the score relates to the number of kept points
2. larger k: is the metric anything but unbiased CKA? (per-pair ratio to its k = n-1 value, and
   rank agreement across pairs)
3. rank agreement across pairs with the other metrics at k = 10 and k = 200
4. toy: two models that agree on the neighbourhoods of 5% of the points only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from geoprh import cknna_variants as cv
from geoprh import local_cknna as lc


def small_k(recs, root: Path, k: int) -> None:
    print(f"\n1. k = {k}: score vs kept points over all layer pairs")
    rows = []
    for r in recs:
        stem = f"{r['llm'].replace('/', '__')}__x__{r['lvm']}"
        d = np.load(root / "pairs" / f"{stem}.npz")
        s, c = d[f"grid_local_mutual_k{k}"].ravel(), d[f"grid_coverage_k{k}"].ravel() * r["n"]
        ok = c > 0
        best = int(np.nanargmax(np.where(ok, s, -np.inf)))
        rows.append((spearmanr(s[ok], c[ok])[0], c[best], np.median(c[ok]), ok.mean(), s[ok].std()))
    rho, kept_best, kept_med, frac, sd = map(np.array, zip(*rows, strict=True))
    print(f"   layer pairs with any kept point: {frac.mean():.0%}")
    print(
        f"   kept points at the argmax layer pair: median {np.median(kept_best):.0f} "
        f"(median over layer pairs: {np.median(kept_med):.0f}) of n = {recs[0]['n']}"
    )
    print(f"   Spearman(score, kept points) over layer pairs: mean {rho.mean():.2f}")
    print(f"   sd of the score across layer pairs: mean {sd.mean():.2f}")


def vs_unbiased_cka(recs) -> None:
    ks = [p["k"] for p in recs[0]["per_k"]]
    print("\n2. local_mutual relative to its k = n-1 value (= unbiased CKA of the best layer pair)")
    full = np.array([r["per_k"][-1]["local_mutual"]["raw"] for r in recs])
    for t, k in enumerate(ks):
        v = np.array([r["per_k"][t]["local_mutual"]["raw"] for r in recs])
        u = np.array([r["per_k"][t]["local_union"]["raw"] for r in recs])
        print(
            f"   k={k:5d}: mutual/UCKA {np.mean(v / full):.2f}  rank vs UCKA {spearmanr(v, full)[0]:.2f}"
            f"  | union/UCKA {np.mean(u / full):.2f}  rank {spearmanr(u, full)[0]:.2f}"
        )


def ranks(recs) -> None:
    names = ["mknn", "centred", "paper", "local_mutual", "local_union"]
    cka = np.array([r["cka"]["raw"] for r in recs])
    for k in (10, 50, 200):
        t = [p["k"] for p in recs[0]["per_k"]].index(k)
        v = {m: np.array([r["per_k"][t][m]["raw"] for r in recs]) for m in names}
        print(f"\n3. k = {k}: Spearman across {len(recs)} pairs")
        for m in names:
            print(
                f"   {m:13s} vs mKNN(k) {spearmanr(v[m], v['mknn'])[0]:5.2f}  "
                f"vs centred(k) {spearmanr(v[m], v['centred'])[0]:5.2f}  vs CKA {spearmanr(v[m], cka)[0]:5.2f}"
            )


def toy(n: int = 1000, k: int = 10, frac: float = 0.05) -> None:
    """Model B equals model A on a small cluster of points, and is unrelated elsewhere."""
    g = torch.Generator().manual_seed(0)
    xa = F.normalize(torch.randn(n, 64, generator=g), dim=-1)
    xb = F.normalize(torch.randn(n, 64, generator=g), dim=-1)
    m = int(frac * n)
    centre = torch.randn(64, generator=g)
    cluster = F.normalize(centre + 0.3 * torch.randn(m, 64, generator=g), dim=-1)
    xa[:m], xb[:m] = cluster, cluster  # the same tight cluster in both models
    sa, sb = cv.Stack([xa]), cv.Stack([xb])
    ak, bk = sa.at_k(k), sb.at_k(k)
    glob = cv.per_k_scores(ak, bk, bk.right())
    red = lc.reduce(lc.local_scores(ak.m[0], bk.m[0], lc.LocalGram(sa.K[0]), lc.LocalGram(sb.K[0])))
    print(
        f"\n4. toy: models identical on {frac:.0%} of points (one cluster), unrelated elsewhere, k = {k}"
    )
    print(
        f"   mKNN {float(glob['mknn']):.2f}  centred CKNNA {float(glob['centred']):.2f}  "
        f"CKA {float(cv.cka(sa, sb)):.2f}  |  local_mutual {float(red['local_mutual']):.2f} "
        f"(coverage {float(red['coverage']):.2f})  local_union {float(red['local_union']):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recs = [json.loads(x) for x in (args.root / "pairs.jsonl").read_text().splitlines()]
    print(f"{len(recs)} pairs")
    small_k(recs, args.root, 10)
    small_k(recs, args.root, 25)
    vs_unbiased_cka(recs)
    ranks(recs)
    toy()


if __name__ == "__main__":
    main()
