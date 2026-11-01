"""exp-011 check: can graph Ollivier-Ricci curvature tell a sphere from a boundary-free flat torus?

The flat control in exp-011 and in the PersonaManifold replication is a cube [-1, 1]^d, which has
boundaries; at ID 20 almost every point of a cube is near one. Here the flat control is the
Clifford torus T^d = (S^1)^d in R^{2d} (each circle of radius 1): intrinsically flat, no boundary,
uniform density. The curved control is the unit sphere S^d. Both are sampled uniformly.

ORC: exact, as PersonaManifold's `edge_curvature` (checked against it below): union-symmetrised
kNN graph with Euclidean edge lengths, uniform measure on graph neighbours with mass alpha on the
node, W1 by exact transport (POT `ot.emd2`) with graph shortest-path costs; node curvature = mean
over incident edges. alpha = 0 (their default) and 0.5 (Ollivier's lazy walk).

    uv run python experiments/exp-011-synthetic-geometry/orc_positive_control.py
"""

from __future__ import annotations

import time

import numpy as np
import ot
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors


def knn_union_graph(x: np.ndarray, k: int) -> csr_matrix:
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    n = len(x)
    rows = np.repeat(np.arange(n), k)
    g = csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return g.maximum(g.T).tocsr()


def orc(graph: csr_matrix, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Edge curvatures (u < v) and node curvature (mean over incident edges)."""
    d = dijkstra(graph, directed=False)
    n = graph.shape[0]
    coo = graph.tocoo()
    edges = [(u, v) for u, v in zip(coo.row, coo.col, strict=True) if u < v]
    node_sum, node_cnt = np.zeros(n), np.zeros(n)
    kap = np.empty(len(edges))

    def measure(i):
        nb = graph.getrow(i).indices
        return np.r_[i, nb], np.r_[alpha, np.full(len(nb), (1 - alpha) / len(nb))]

    for e, (u, v) in enumerate(edges):
        su, mu = measure(u)
        sv, mv = measure(v)
        kap[e] = 1 - ot.emd2(mu, mv, d[np.ix_(su, sv)]) / d[u, v]
        node_sum[[u, v]] += kap[e]
        node_cnt[[u, v]] += 1
    return kap, node_sum / np.maximum(node_cnt, 1)


def sphere(n, d, rng):
    g = rng.standard_normal((n, d + 1))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def clifford_torus(n, d, rng):
    t = rng.uniform(0, 2 * np.pi, size=(n, d))
    return np.concatenate([np.cos(t), np.sin(t)], axis=1)


def check_against_theirs():
    from personamanifold import PersonaManifold
    from personamanifold.curvature import edge_curvature

    rng = np.random.default_rng(0)
    pm = PersonaManifold(n_neighbors=5, variance=1.0, tangent_dim=3, ridge=1e-4)
    pm.fit(rng.standard_normal((40, 3)))
    kap, _ = orc(pm.graph_, alpha=0.0)
    coo = pm.graph_.tocoo()
    edges = [(u, v) for u, v in zip(coo.row, coo.col, strict=True) if u < v]
    worst = max(abs(kap[i] - edge_curvature(pm, u, v)) for i, (u, v) in enumerate(edges[:20]))
    print(f"check vs PersonaManifold edge_curvature: max |diff| over 20 edges = {worst:.2e}")


def main() -> None:
    check_against_theirs()
    rng = np.random.default_rng(1)
    cases = [(2, 3000, 10), (2, 3000, 30), (5, 3000, 10), (5, 3000, 25), (20, 2000, 10)]
    print("ID  n     k   alpha | sphere S^d mean node ORC | flat torus T^d | sphere - torus")
    for d, n, k in cases:
        xs, xt = sphere(n, d, rng), clifford_torus(n, d, rng)
        gs, gt = knn_union_graph(xs, k), knn_union_graph(xt, k)
        for alpha in (0.0, 0.5):
            t0 = time.time()
            ns, nt = orc(gs, alpha)[1], orc(gt, alpha)[1]
            print(
                f"{d:<3} {n:<5} {k:<3} {alpha:<5} | {ns.mean():+.4f} (sd {ns.std():.3f})"
                f"          | {nt.mean():+.4f} (sd {nt.std():.3f}) | {ns.mean() - nt.mean():+.4f}"
                f"   [{time.time() - t0:.0f}s]",
                flush=True,
            )


if __name__ == "__main__":
    main()
