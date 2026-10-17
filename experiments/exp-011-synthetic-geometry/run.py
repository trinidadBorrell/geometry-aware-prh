"""exp-011 (WP0): geometry estimators on synthetic manifolds with known geometry.

For every manifold in `geoprh.synthetic` x intrinsic dimension x n x noise level, embedded by a
random isometry into 1024 dimensions, record:

- ID: Levina-Bickel (k = 10..20), TwoNN, and TwoNN by decimation (ID vs scale)
- local PCA dimension of growing kNN neighbourhoods (the flatness profile)
- geodesics: for kNN graphs with k in --graph-ks, how well graph distances from 200 random
  sources match the true geodesics (Spearman, median relative error), against plain Euclidean
  distances; and the number of connected components
- delta-hyperbolicity (relative to the diameter) on 1000 points of the largest component, from
  the graph geodesics (k = 30), from Euclidean distances, and from the true geodesics
- Ollivier-Ricci curvature of the kNN graph (k in --orc-ks) on the first --orc-n points

Noise: isotropic Gaussian in all ambient dimensions, with norm --noise times the median
nearest-neighbour distance of the clean data.

    uv run python experiments/exp-011-synthetic-geometry/run.py --out <dir> --threads 16
    uv run python experiments/exp-011-synthetic-geometry/run.py --out <dir> --quick
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from geoprh import geometry as geo
from geoprh import intrinsic_dim as idim
from geoprh import synthetic as syn

AMBIENT = 1024
N_SOURCES = 200
N_DELTA = 1000
DELTA_K = 30
PCA_KS = [10, 20, 50, 100, 200, 500]


def _q(a: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q10": float(np.quantile(a, 0.1)),
        "q90": float(np.quantile(a, 0.9)),
        "frac_neg": float(np.mean(a < 0)),
    }


def _compare(est: np.ndarray, true: np.ndarray, sources: np.ndarray, rng) -> dict:
    mask = np.isfinite(est) & (true > 0)
    mask[np.arange(len(sources)), sources] = False
    e, t = est[mask], true[mask]
    pick = rng.choice(len(e), min(len(e), 100_000), replace=False)
    return {
        "spearman": float(spearmanr(e[pick], t[pick]).statistic),
        "median_rel_err": float(np.median(np.abs(e - t) / t)),
        "frac_finite": float(np.isfinite(est).mean()),
    }


def run_one(manifold, d, n, noise, args, seed) -> dict:
    rng = np.random.default_rng(seed)
    z, truth = syn.MANIFOLDS[manifold](n, d, rng)
    x = syn.embed(z, AMBIENT, rng)
    if noise > 0:
        nn = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x[: min(n, 2000)])[0][:, 1]
        x = x + noise * np.median(nn) / np.sqrt(AMBIENT) * rng.standard_normal(x.shape)
    rec = {"manifold": manifold, "d": d, "n": n, "noise": noise, "seed": seed}
    rec["true_curvature"] = truth["curvature"]
    t0 = time.time()

    rec["id_lb"] = idim.levina_bickel(x, n_jobs=args.threads)["id"]
    rec["id_twonn"] = idim.two_nn(x, n_jobs=args.threads)["id"]
    rec["twonn_scales"] = geo.twonn_scales(x, seed=seed)
    rec["local_pca"] = geo.local_pca_profile(x, PCA_KS, seed=seed)

    sources = rng.choice(n, N_SOURCES, replace=False)
    true_d = truth["geodesic"](sources) if truth["geodesic"] else None
    if true_d is not None:
        rec["euclid_vs_true"] = _compare(cdist(x[sources], x), true_d, sources, rng)
    rec["graph"] = {}
    graphs = {}
    for k in args.graph_ks:
        g = geo.knn_graph(x, k, n_jobs=args.threads)
        graphs[k] = g
        comp = geo.components(g)
        entry = {"n_components": int(comp.max() + 1), "largest": int(np.bincount(comp).max())}
        if true_d is not None:
            entry.update(_compare(geo.geodesic_distances(g, sources), true_d, sources, rng))
        rec["graph"][str(k)] = entry

    g = graphs[DELTA_K] if DELTA_K in graphs else geo.knn_graph(x, DELTA_K, n_jobs=args.threads)
    comp = geo.components(g)
    main = np.flatnonzero(comp == np.bincount(comp).argmax())
    sub = rng.choice(main, min(N_DELTA, len(main)), replace=False)
    rec["delta"] = {
        "graph": geo.delta_hyperbolicity(geo.geodesic_distances(g, sub)[:, sub], seed=seed),
        "euclid": geo.delta_hyperbolicity(cdist(x[sub], x[sub]), seed=seed),
    }
    if truth["geodesic"]:
        rec["delta"]["true"] = geo.delta_hyperbolicity(truth["geodesic"](sub)[:, sub], seed=seed)

    rec["orc"] = {}
    xs = x[: args.orc_n]
    for k in args.orc_ks:
        o = geo.ollivier_ricci(geo.knn_graph(xs, k, n_jobs=args.threads), proc=args.threads)
        rec["orc"][str(k)] = {"edge": _q(o["edge"]), "node": _q(o["node"][np.isfinite(o["node"])])}
    rec["seconds"] = time.time() - t0
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifolds", default=",".join(syn.MANIFOLDS))
    parser.add_argument("--dims", default="5,20,40")
    parser.add_argument("--sizes", default="1000,10000")
    parser.add_argument("--noise", default="0,0.1")
    parser.add_argument("--graph-ks", default="10,30,100")
    parser.add_argument("--orc-ks", default="10,30")
    parser.add_argument("--orc-n", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    args = parser.parse_args()
    if args.quick:
        args.dims, args.sizes, args.noise, args.orc_n = "5", "400", "0.1", 300
        args.graph_ks = "10,30"
    args.graph_ks = [int(k) for k in args.graph_ks.split(",")]
    # GraphRicciCurvature needs the "fork" start method: Linux only; "none" skips it
    args.orc_ks = [int(k) for k in args.orc_ks.split(",") if k not in ("", "none")]
    args.out.mkdir(parents=True, exist_ok=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    meta = {**{k: str(v) for k, v in vars(args).items()}, "git_sha": sha.strip()}
    meta.update(ambient=AMBIENT, n_sources=N_SOURCES, n_delta=N_DELTA, delta_k=DELTA_K)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    done = set()
    path = args.out / "synthetic.jsonl"
    if path.exists():  # resume
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["manifold"], r["d"], r["n"], r["noise"]))
    grid = [
        (m, int(d), int(n), float(nz))
        for n in args.sizes.split(",")
        for d in args.dims.split(",")
        for m in args.manifolds.split(",")
        for nz in args.noise.split(",")
    ]
    with path.open("a") as fout:
        for i, (m, d, n, nz) in enumerate(grid):
            if (m, d, n, nz) in done:
                continue
            rec = run_one(m, d, n, nz, args, seed=args.seed + i)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(
                f"{m:<9} d={d:<3} n={n:<6} noise={nz}: {rec['seconds']:.0f}s "
                f"LB {rec['id_lb']:.1f} 2NN {rec['id_twonn']:.1f} "
                f"delta_rel {rec['delta']['graph']['delta_rel']:.3f} "
                f"ORC {[round(o['edge']['mean'], 3) for o in rec['orc'].values()]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
