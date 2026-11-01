"""exp-014 step 1: does the paired per-sample ORC statistic behave as expected?

Two copies A, B of the same n = 1024 samples of a manifold (ID 10), each embedded by its own random
isometry into 1024 dimensions with independent noise (sd = --noise x the median NN distance).
Same geometry and same samples, so per-sample ORC and density should agree across copies up to
noise; with the pairing shuffled they should not. Reported per manifold and k: Spearman across
samples of node ORC (A vs B), of local density, and of ORC with density removed in each copy
(partial, `geoprh.paired.residualize`); and the same three with B's samples permuted.

    uv run python experiments/exp-014-orc-paired/synthetic_check.py
"""

from __future__ import annotations

import argparse

import numpy as np

from geoprh import geometry as geo
from geoprh import paired
from geoprh import synthetic as syn


def copy(z, rng, noise):
    x = syn.embed(z, 1024, rng)
    nn = geo.knn_distance(x, 1)
    return x + noise * np.median(nn) / np.sqrt(x.shape[1]) * rng.standard_normal(x.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--noise", type=float, default=0.3)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    print("manifold  k  | ORC   density  ORC|density | shuffled: ORC  density  ORC|density")
    for name in ("flat", "sphere", "clusters"):
        z, _ = syn.MANIFOLDS[name](1024, 10, rng)
        xa, xb = copy(z, rng, args.noise), copy(z, rng, args.noise)
        for k in (10, 30):
            q, dens = [], []
            for x in (xa, xb):
                q.append(geo.ollivier_ricci_exact(geo.knn_graph(x, k))["node"][None])
                dens.append(1 / geo.knn_distance(x, k)[None])
            perm = rng.permutation(1024)
            vals = []
            for p in (None, perm):
                za, zb = paired.standardized_ranks(q[0]), paired.standardized_ranks(q[1])
                da, db = paired.standardized_ranks(dens[0]), paired.standardized_ranks(dens[1])
                ra, rb = paired.residualize(q[0], dens[0]), paired.residualize(q[1], dens[1])
                vals += [
                    paired.rank_corr_matrix(a, b, p)[0, 0]
                    for a, b in ((za, zb), (da, db), (ra, rb))
                ]
            print(
                f"{name:<9} {k:<3}| "
                + " ".join(f"{v:+.2f}  " for v in vals[:3])
                + "| "
                + " ".join(f"{v:+.2f}  " for v in vals[3:]),
                flush=True,
            )


if __name__ == "__main__":
    main()
