"""Manifold-geometry estimators (experiments/proposal-geometry-aware-prh.md, WP0 onward).

- `knn_graph`, `geodesic_distances`: graph geodesics as in Isomap (Tenenbaum et al. 2000): the
  symmetrised kNN graph with Euclidean edge lengths (sklearn), shortest paths by Dijkstra (scipy).
- `delta_hyperbolicity`: Gromov's four-point delta on random quadruples. For x, y, z, w let
  S1 >= S2 >= S3 be the three sums d(x,y)+d(z,w), d(x,z)+d(y,w), d(x,w)+d(y,z); delta of the
  quadruple = (S1 - S2) / 2, and the space's delta is the max over quadruples (a tree has 0).
  Reported relative to the diameter, delta_rel = 2 delta / diam (Khrulkov et al. 2020), so it
  lies in [0, 1]. The max over sampled quadruples is a lower bound of the max over all of them.
- `ollivier_ricci`: Ollivier (2009) curvature of the kNN graph via GraphRicciCurvature,
  kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y), with the lazy walk m_x = alpha delta_x + (1 - alpha)
  uniform on x's neighbours (alpha = 0.5, `exp_power=0` makes the neighbour weights uniform), W1
  solved exactly ("OTD") with graph shortest-path costs. Node curvature = mean over its edges.
- `twonn_scales`: TwoNN at growing scale by decimation, as DADApy's `return_id_scaling_2NN`:
  the estimate on random subsets of n / 2^j points, the scale being the subset's mean distance to
  the second neighbour.
- `local_pca_profile`: the local linear dimension (PCs for 90% of the variance, skdim `lPCA`) of
  growing kNN neighbourhoods. On a curved manifold it equals the ID for small neighbourhoods and
  grows with their radius; on a flat one it stays at the ID.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import skdim
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from geoprh import intrinsic_dim as idim


def knn_graph(x: np.ndarray, k: int, n_jobs: int = 1):
    """Symmetrised kNN graph (scipy CSR), Euclidean edge lengths, self excluded."""
    g = kneighbors_graph(x, k, mode="distance", include_self=False, n_jobs=n_jobs)
    return g.maximum(g.T).tocsr()


def components(graph) -> np.ndarray:
    """Connected-component label per node."""
    return connected_components(graph, directed=False)[1]


def geodesic_distances(graph, sources=None) -> np.ndarray:
    """Shortest-path lengths from `sources` (all nodes if None) to every node; inf if unreachable."""
    return shortest_path(graph, method="D", directed=False, indices=sources)


def delta_hyperbolicity(d: np.ndarray, n_quads: int = 200_000, seed: int = 0) -> dict:
    """Four-point delta on `n_quads` random quadruples of the finite distance matrix `d`."""
    rng = np.random.default_rng(seed)
    q = rng.integers(0, d.shape[0], size=(n_quads, 4))
    x, y, z, w = q.T
    s = np.stack([d[x, y] + d[z, w], d[x, z] + d[y, w], d[x, w] + d[y, z]], axis=1)
    s.sort(axis=1)
    delta = (s[:, 2] - s[:, 1]) / 2
    diam = float(d.max())
    return {
        "delta": float(delta.max()),
        "diam": diam,
        "delta_rel": 2 * float(delta.max()) / diam,
        "delta_rel_mean": 2 * float(delta.mean()) / diam,
    }


def ollivier_ricci(graph, alpha: float = 0.5, proc: int = 1) -> dict:
    """Edge and node Ollivier-Ricci curvature of a weighted graph (scipy sparse)."""
    g = nx.from_scipy_sparse_array(graph, edge_attribute="weight")
    orc = OllivierRicci(g, alpha=alpha, method="OTD", exp_power=0, proc=proc, verbose="ERROR")
    orc.compute_ricci_curvature()
    edges = np.array([c for _, _, c in orc.G.edges(data="ricciCurvature")])
    nodes = np.array([orc.G.nodes[i].get("ricciCurvature", np.nan) for i in range(graph.shape[0])])
    return {"edge": edges, "node": nodes}


def twonn_scales(x: np.ndarray, max_halvings: int = 5, min_n: int = 64, seed: int = 0) -> list:
    """[{"n", "scale", "id"}] for the full set and random subsets of n / 2^j points."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(max_halvings + 1):
        m = x.shape[0] // 2**j
        if m < min_n:
            break
        sub = x[rng.choice(x.shape[0], m, replace=False)] if j else x
        r2 = NearestNeighbors(n_neighbors=3).fit(sub).kneighbors(sub)[0][:, 2]
        out.append({"n": m, "scale": float(r2.mean()), "id": idim.two_nn(sub)["id"]})
    return out


def local_pca_profile(
    x: np.ndarray, ks: list[int], n_centres: int = 200, variance: float = 0.9, seed: int = 0
) -> list:
    """[{"k", "radius", "dim"}]: mean local PCA dimension of the k-NN neighbourhood of random
    centres, and the mean distance to the k-th neighbour."""
    rng = np.random.default_rng(seed)
    centres = rng.choice(x.shape[0], min(n_centres, x.shape[0]), replace=False)
    ks = [k for k in ks if k < x.shape[0]]
    dist, idx = NearestNeighbors(n_neighbors=max(ks) + 1).fit(x).kneighbors(x[centres])
    rule = skdim.id.lPCA(ver="ratio", alphaRatio=variance, fit_explained_variance=True)
    out = []
    for k in ks:
        dims = []
        for row in idx:
            pts = x[row[: k + 1]]
            pts = pts - pts.mean(axis=0)
            # local spectrum from the (k+1) x (k+1) Gram matrix (same nonzero eigenvalues as the
            # covariance, much faster than skdim's own PCA fit); skdim applies the rule
            ev = np.linalg.eigvalsh(pts @ pts.T)[::-1] / k
            dims.append(rule.fit(ev[ev > 1e-12 * ev[0]]).dimension_)
        out.append({"k": k, "radius": float(dist[:, k].mean()), "dim": float(np.mean(dims))})
    return out
