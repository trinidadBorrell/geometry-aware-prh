"""exp-028: caption graphs, their regularised spectral embedding, Nyström extension, PPMI word vectors.

Spectral embedding (Balestriero and LeCun 2610.39302, Prop. 1; Laplacian eigenmaps): for a symmetric
non-negative affinity A with degrees d, the top eigenvectors v_i of M = D^{-1/2} A D^{-1/2} give
psi_i = v_i / sqrt(d), the eigenvectors of the random walk P = D^{-1} A (P psi_i = mu_i psi_i). The
first (mu = 1, psi constant) is dropped. Regularisation (Amini et al. 2013) adds tau / n to every
entry of A, so A_tau = W + (tau / n) 1 1^T and d_tau = d + tau; M is applied as a linear operator, so
A_tau is never formed. With tau = 0 this is `sklearn.manifold.spectral_embedding` (tested).

Nyström extension of a new point x with affinities a(x, j) to the n graph nodes:
psi_i(x) = sum_j a_tau(x, j) psi_i(j) / (mu_i d_tau(x)), which reproduces psi on the nodes themselves.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh


def knn_affinity(x: sp.csr_matrix, k: int, y: sp.csr_matrix | None = None, chunk: int = 1024):
    """Top-k cosine similarities of each row of `y` (default: `x`, self excluded) among the rows of
    `x`. Rows are l2-normalised sparse vectors; zero similarities are dropped. [m, n] csr."""
    self_graph = y is None
    y = x if self_graph else y
    xt = x.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, y.shape[0], chunk):
        sim = (y[s : s + chunk] @ xt).toarray()
        if self_graph:
            idx = np.arange(s, min(s + chunk, y.shape[0]))
            sim[idx - s, idx] = -np.inf
        kk = min(k, sim.shape[1] - 1)
        top = np.argpartition(-sim, kk, axis=1)[:, :kk]
        v = np.take_along_axis(sim, top, 1)
        keep = v > 0
        r = np.broadcast_to(np.arange(s, s + sim.shape[0])[:, None], top.shape)
        rows.append(r[keep])
        cols.append(top[keep])
        vals.append(v[keep])
    shape = (y.shape[0], x.shape[0])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape
    )


def symmetrise(w: sp.csr_matrix) -> sp.csr_matrix:
    return w.maximum(w.T).tocsr()


def graph_stats(w: sp.csr_matrix) -> dict:
    from scipy.sparse.csgraph import connected_components

    n_comp, labels = connected_components(w, directed=False)
    sizes = np.bincount(labels)
    deg = np.asarray(w.sum(1)).ravel()
    return {
        "n": int(w.shape[0]),
        "edges": int(w.nnz // 2),
        "components": int(n_comp),
        "largest_component": int(sizes.max()),
        "isolated": int((deg == 0).sum()),
        "mean_degree": float(deg.mean()),
    }


def spectral(w: sp.csr_matrix, k: int, tau: float | str = "mean", seed: int = 0):
    """(psi [n, k], mu [k], tau): the top k non-trivial random-walk eigenvectors of A_tau."""
    n = w.shape[0]
    deg = np.asarray(w.sum(1)).ravel().astype(np.float64)
    tau = float(deg.mean()) if tau == "mean" else float(tau)
    dt = deg + tau
    if (dt <= 0).any():
        raise ValueError("zero degree: isolated nodes need tau > 0")
    s = 1.0 / np.sqrt(dt)
    w = w.astype(np.float64)

    def mv(v):
        v = np.asarray(v, dtype=np.float64)
        y = s[:, None] * (v if v.ndim == 2 else v[:, None])
        out = w @ y + (tau / n) * y.sum(0, keepdims=True)
        out = s[:, None] * out
        return out if v.ndim == 2 else out[:, 0]

    op = LinearOperator((n, n), matvec=mv, matmat=mv, dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(n)
    mu, v = eigsh(op, k=k + 1, which="LA", v0=v0)
    order = np.argsort(-mu)
    mu, v = mu[order], v[:, order]
    psi = v * s[:, None]
    psi = _sign_flip(psi)
    return psi[:, 1:], mu[1:], tau


def _sign_flip(u: np.ndarray) -> np.ndarray:
    """sklearn's deterministic sign: the largest-magnitude entry of each column is positive."""
    idx = np.argmax(np.abs(u), axis=0)
    return u * np.sign(u[idx, np.arange(u.shape[1])])


def nystrom(a_new: sp.csr_matrix, psi: np.ndarray, mu: np.ndarray, tau: float) -> np.ndarray:
    """Extend psi [n, k] to new points with affinities a_new [m, n] to the graph nodes."""
    n = psi.shape[0]
    num = a_new @ psi + (tau / n) * psi.sum(0, keepdims=True)
    d_new = np.asarray(a_new.sum(1)).ravel() + tau
    return num / (d_new[:, None] * mu[None, :])


def ppmi_matrix(binary_docs: sp.csr_matrix, alpha: float = 0.75) -> np.ndarray:
    """[V, V] PPMI with co-occurrence = both words in one caption, context smoothing alpha:
    PPMI(w, c) = max(0, log(n(w, c) * sum_c' n(c')^a / (n(w) * n(c)^a)))."""
    b = binary_docs.astype(np.float32)
    c = (b.T @ b).toarray()
    np.fill_diagonal(c, 0.0)
    row = c.sum(1)
    ctx = c.sum(0) ** alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(c * ctx.sum() / (row[:, None] * ctx[None, :]))
    del c
    pmi[~np.isfinite(pmi)] = 0.0
    np.maximum(pmi, 0.0, out=pmi)
    return pmi


def ppmi_vectors(
    binary_docs: sp.csr_matrix, k: int, alpha: float = 0.75, power: float = 0.5, seed: int = 0
) -> np.ndarray:
    """[V, k] word vectors: truncated SVD of word-word PPMI, U S^power.

    Levy, Goldberg and Dagan (2015) defaults: context-distribution smoothing alpha = 0.75 and
    symmetric weighting U S^0.5. `randomized_svd` from scikit-learn."""
    from sklearn.utils.extmath import randomized_svd

    pmi = ppmi_matrix(binary_docs, alpha)
    u, sv, _vt = randomized_svd(pmi, n_components=k, random_state=seed)
    return (u * sv[None, :] ** power).astype(np.float32)
