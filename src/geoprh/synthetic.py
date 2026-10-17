"""Synthetic manifolds with known geometry, for validating the estimators in `geoprh.geometry`
(exp-011, WP0 of experiments/proposal-geometry-aware-prh.md).

Every generator returns (z, truth): intrinsic-space points z [n, m] and a dict with the known
intrinsic dimension, the sign of the curvature, and a function giving true geodesic distances
from a set of rows to all rows (None where there is no closed form). `embed` places z in a
D-dimensional ambient space by a random isometry (orthonormal columns) and adds isotropic noise.

- flat:     uniform in [-1, 1]^d. Curvature 0; geodesic = Euclidean.
- sphere:   uniform on the unit sphere S^d in R^(d+1). Curvature +1; geodesic = arccos <u, v>.
- swiss:    a d-dimensional Swiss roll: the first intrinsic coordinate is rolled into the spiral
            (t cos t, t sin t), t in [1.5 pi, 4.5 pi]; the other d - 1 are uniform in [0, 20].
            Intrinsically flat (curvature 0), extrinsically curved; the geodesic is Euclidean in
            the unrolled coordinates (arc length of the spiral, the rest).
- tree:     a "thick tree": points on the edges of a random binary tree of depth 6 in R^d (edge
            length 0.7^level), plus Gaussian noise of sd 0.02 in R^d. Locally d-dimensional, tree-like
            (negatively curved) at large scale; no closed-form geodesic.
- clusters: 10 Gaussian blobs N(c, I_d) with centres c ~ N(0, 1.5^2 I_d): flat pieces with
            sparse bridges, the case that local curvature estimators can mistake for negative
            curvature; no closed-form geodesic.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _euclid_geodesic(u: np.ndarray):
    def geo(rows):
        return cdist(u[rows], u)

    return geo


def flat(n: int, d: int, rng) -> tuple[np.ndarray, dict]:
    z = rng.uniform(-1, 1, size=(n, d))
    return z, {"id": d, "curvature": 0, "geodesic": _euclid_geodesic(z)}


def sphere(n: int, d: int, rng) -> tuple[np.ndarray, dict]:
    g = rng.standard_normal((n, d + 1))
    z = g / np.linalg.norm(g, axis=1, keepdims=True)

    def geo(rows):
        return np.arccos(np.clip(z[rows] @ z.T, -1.0, 1.0))

    return z, {"id": d, "curvature": 1, "geodesic": geo}


def _spiral_arc_length(t: np.ndarray) -> np.ndarray:
    # arc length of (t cos t, t sin t) from 0 to t
    return 0.5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t))


def swiss(n: int, d: int, rng) -> tuple[np.ndarray, dict]:
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    rest = rng.uniform(0, 20, size=(n, d - 1))
    z = np.column_stack([t * np.cos(t), t * np.sin(t), rest])
    unrolled = np.column_stack([_spiral_arc_length(t), rest])
    return z, {"id": d, "curvature": 0, "geodesic": _euclid_geodesic(unrolled)}


def tree(n: int, d: int, rng, depth: int = 6, noise: float = 0.02) -> tuple[np.ndarray, dict]:
    starts, ends = [], []
    nodes = [(np.zeros(d), 0)]
    while nodes:
        p, level = nodes.pop()
        if level == depth:
            continue
        for _ in range(2):
            v = rng.standard_normal(d)
            q = p + 0.7**level * v / np.linalg.norm(v)
            starts.append(p)
            ends.append(q)
            nodes.append((q, level + 1))
    starts, ends = np.array(starts), np.array(ends)
    lengths = np.linalg.norm(ends - starts, axis=1)
    e = rng.choice(len(starts), n, p=lengths / lengths.sum())  # uniform along total length
    s = rng.uniform(0, 1, (n, 1))
    z = starts[e] + s * (ends[e] - starts[e]) + noise * rng.standard_normal((n, d))
    return z, {"id": d, "curvature": -1, "geodesic": None}


def clusters(n: int, d: int, rng, n_clusters: int = 10, spread: float = 1.5):
    centres = spread * rng.standard_normal((n_clusters, d))
    z = centres[rng.integers(0, n_clusters, n)] + rng.standard_normal((n, d))
    return z, {"id": d, "curvature": None, "geodesic": None}


MANIFOLDS = {"flat": flat, "sphere": sphere, "swiss": swiss, "tree": tree, "clusters": clusters}


def embed(z: np.ndarray, ambient: int, rng, noise_sd: float = 0.0) -> np.ndarray:
    """Random isometric embedding of z [n, m] into R^ambient, plus N(0, noise_sd^2) per coordinate."""
    q, _ = np.linalg.qr(rng.standard_normal((ambient, z.shape[1])))
    x = z @ q.T
    if noise_sd > 0:
        x = x + noise_sd * rng.standard_normal(x.shape)
    return x
