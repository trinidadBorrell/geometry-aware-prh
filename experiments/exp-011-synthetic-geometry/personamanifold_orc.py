"""exp-011 addendum: PersonaManifold's Ollivier-Ricci pipeline and density-decoupling test on
synthetic data with known geometry.

Graph: PersonaManifold's own `PersonaManifold.fit` (package pinned to commit 33e181c): PCA to 99%
of the variance, local metric g_i = sum_a v_a v_a^T / (lambda_a + eps) on the top-d local PCA
directions (eps = 1e-4, Table A.1), edge weight (|delta|_{g_i} + |delta|_{g_j}) / 2, union-
symmetrised kNN graph with k = 5 d (Table A.1), d = the true ID (their d*).

Curvature: kappa(x, y) = 1 - W1(mu_x, mu_y) / d_G(x, y) on every edge, mu uniform on the graph
neighbours with no mass on the node (idleness 0, their code default), exact transport with
graph shortest-path costs, i.e. their `edge_curvature`; computed for all edges with
GraphRicciCurvature (method "OTD", alpha 0, exp_power 0), checked against `edge_curvature` in
tests/test_personamanifold_orc.py. Node curvature = mean over the node's own k kNN edges (§3.2).
Comparison construction: the same union-kNN topology with plain Euclidean edge lengths.

Density test (their Appendix E): local kNN density = 1 / (distance to the k-th neighbour) in the
PCA space (k = 5d; and k = 10); points binned into density quintiles; within each quintile a
Kruskal-Wallis test of node curvature across "types"; Spearman rho between density and
curvature over all points.

Scenarios (n = 2000, embedded by a random isometry into 1024 dimensions, noise 0.1 x the median
nearest-neighbour distance), at ID 5, 20 and 40:
- suite: each exp-011 manifold alone (flat, sphere, swiss, tree, clusters); no types
- mixture: four components, 500 points each, as types: flat (true curvature 0), sphere (+),
  tree (hierarchical, -), clusters (flat pieces, 0); each in its own random subspace, translated
  random subspace and 3 radii from the others, so the graph does not connect them
- grouped: one flat cube [-1, 1]^d, all curvature 0; type "spread" = 1000 uniform points, type
  "grouped" = 100 tight groups of 10 (sd 0.02) at uniform positions. Grouped points have many
  shared neighbours but, at the k = 5d density scale, about the same density as spread points
- null: one flat cube, 2000 uniform points, 3 random type labels

    uv run python experiments/exp-011-synthetic-geometry/personamanifold_orc.py --out <dir> --threads 28
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import kruskal, spearmanr
from sklearn.neighbors import NearestNeighbors

from geoprh import synthetic as syn

N = 2000
AMBIENT = 1024
NOISE = 0.1
EPS = 1e-4  # their metric regularisation (Table A.1)


def their_graph(x: np.ndarray, d: int):
    from personamanifold import PersonaManifold

    return PersonaManifold(n_neighbors=5 * d, variance=0.99, tangent_dim=d, ridge=EPS).fit(x)


def euclidean_twin(pm):
    """Same union-kNN topology, Euclidean edge lengths in the PCA space."""
    g = pm.graph_.tocoo()
    w = np.linalg.norm(pm.points_[g.row] - pm.points_[g.col], axis=1)
    from scipy.sparse import csr_matrix

    return csr_matrix((w, (g.row, g.col)), shape=g.shape)


def node_orc(graph, neighbors: np.ndarray, proc: int) -> dict:
    """Edge ORC (their convention) and node curvature = mean over the node's own kNN edges."""
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    g = nx.from_scipy_sparse_array(graph, edge_attribute="weight")
    orc = OllivierRicci(g, alpha=0.0, method="OTD", exp_power=0, proc=proc, verbose="ERROR")
    orc.compute_ricci_curvature()
    kap = orc.G
    node = np.array(
        [
            np.mean([kap[i][int(j)]["ricciCurvature"] for j in row])
            for i, row in enumerate(neighbors)
        ]
    )
    edge = np.array([c for _, _, c in kap.edges(data="ricciCurvature")])
    return {"node": node, "edge": edge}


def density(points: np.ndarray, k: int) -> np.ndarray:
    r = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)[0][:, k]
    return 1.0 / r


def decoupling_test(curv: np.ndarray, dens: np.ndarray, types: np.ndarray) -> dict:
    """Their Appendix E: Kruskal-Wallis across types within density quintiles; Spearman."""
    rho = spearmanr(dens, curv)
    edges = np.quantile(dens, [0.2, 0.4, 0.6, 0.8])
    q = np.digitize(dens, edges)
    bins = []
    for b in range(5):
        m = q == b
        groups = {t: curv[m & (types == t)] for t in np.unique(types)}
        groups = {t: v for t, v in groups.items() if len(v) >= 5}
        entry = {
            "n": int(m.sum()),
            "mean_by_type": {str(t): float(v.mean()) for t, v in groups.items()},
        }
        entry["count_by_type"] = {str(t): len(v) for t, v in groups.items()}
        if len(groups) >= 2:
            entry["kruskal_p"] = float(kruskal(*groups.values()).pvalue)
        bins.append(entry)
    return {"spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue), "bins": bins}


def summary(v: np.ndarray) -> dict:
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "q10": float(np.quantile(v, 0.1)),
        "q90": float(np.quantile(v, 0.9)),
        "frac_neg": float((v < 0).mean()),
    }


def add_noise(x: np.ndarray, rng) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x)[0][:, 1]
    return x + NOISE * np.median(nn) / np.sqrt(x.shape[1]) * rng.standard_normal(x.shape)


def make(scenario: str, d: int, rng, manifold: str | None = None):
    """(x [N, AMBIENT], types [N] or None, truth dict)."""
    if scenario == "suite":
        z, truth = syn.MANIFOLDS[manifold](N, d, rng)
        return add_noise(syn.embed(z, AMBIENT, rng), rng), None, {"curvature": truth["curvature"]}
    if scenario == "mixture":
        parts, types = [], []
        for name in ("flat", "sphere", "tree", "clusters"):
            z, _ = syn.MANIFOLDS[name](N // 4, d, rng)
            z = z - z.mean(axis=0)
            z /= np.sqrt((z**2).sum(axis=1).mean())  # unit RMS radius
            x = syn.embed(z, AMBIENT, rng)
            u = rng.standard_normal(AMBIENT)
            # 3 RMS radii apart: no kNN edges between components, yet the offsets do not take
            # over their 99%-variance PCA (1000 units apart left only the 3 offset directions)
            parts.append(x + 3 * u / np.linalg.norm(u))
            types += [name] * len(z)
        x = np.concatenate(parts)
        truth = {"flat": 0, "sphere": 1, "tree": -1, "clusters": 0}
        return add_noise(x, rng), np.array(types), {"curvature_by_type": truth}
    if scenario == "grouped":
        spread = rng.uniform(-1, 1, size=(N // 2, d))
        centres = rng.uniform(-1, 1, size=(N // 20, d))
        grouped = np.repeat(centres, 10, axis=0) + 0.02 * rng.standard_normal((N // 2, d))
        z = np.concatenate([spread, grouped])
        types = np.array(["spread"] * (N // 2) + ["grouped"] * (N // 2))
        x = add_noise(syn.embed(z, AMBIENT, rng), rng)
        return x, types, {"curvature_by_type": {"spread": 0, "grouped": 0}}
    if scenario == "null":
        z, _ = syn.flat(N, d, rng)
        types = rng.choice(np.array(["a", "b", "c"]), N)
        x = add_noise(syn.embed(z, AMBIENT, rng), rng)
        return x, types, {"curvature_by_type": {"a": 0, "b": 0, "c": 0}}
    raise ValueError(scenario)


def run_one(scenario, d, manifold, threads, seed) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x, types, truth = make(scenario, d, rng, manifold)
    pm = their_graph(x, d)
    rec = {"scenario": scenario, "manifold": manifold, "d": d, "n": N, "k": 5 * d, "seed": seed}
    rec.update(truth=truth, pca_dim=int(pm.points_.shape[1]))
    for name, graph in (("whitened", pm.graph_), ("euclidean", euclidean_twin(pm))):
        orc = node_orc(graph, pm.neighbors_, threads)
        res = {"node": summary(orc["node"]), "edge": summary(orc["edge"])}
        dens = {f"k{k}": density(pm.points_, k) for k in (5 * d, 10)}
        res["spearman_density"] = {
            key: dict(zip(("rho", "p"), map(float, spearmanr(v, orc["node"])), strict=True))
            for key, v in dens.items()
        }
        if types is not None:
            res["by_type"] = {str(t): summary(orc["node"][types == t]) for t in np.unique(types)}
            res["decoupling"] = {
                key: decoupling_test(orc["node"], v, types) for key, v in dens.items()
            }
        rec[name] = res
    rec["seconds"] = time.time() - t0
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dims", default="5,20,40")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    meta = {**{k: str(v) for k, v in vars(args).items()}, "git_sha": sha.strip()}
    meta.update(n=N, ambient=AMBIENT, noise=NOISE, eps=EPS, personamanifold="33e181c")
    (args.out / "meta_personamanifold.json").write_text(json.dumps(meta, indent=2))
    jobs = [("suite", m) for m in ("flat", "sphere", "swiss", "tree", "clusters")]
    jobs += [("mixture", None), ("grouped", None), ("null", None)]
    with (args.out / "personamanifold_orc.jsonl").open("a") as fout:
        i = 0
        for d in (int(v) for v in args.dims.split(",")):
            for scenario, manifold in jobs:
                rec = run_one(scenario, d, manifold, args.threads, args.seed + i)
                i += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                w = rec["whitened"]
                line = f"{scenario:<8} {manifold or '':<9} d={d:<3} {rec['seconds']:.0f}s "
                line += f"node ORC whitened {w['node']['mean']:+.3f} "
                line += f"euclid {rec['euclidean']['node']['mean']:+.3f}"
                if "by_type" in w:
                    line += " | " + " ".join(
                        f"{t}:{v['mean']:+.3f}" for t, v in w["by_type"].items()
                    )
                print(line, flush=True)


if __name__ == "__main__":
    main()
