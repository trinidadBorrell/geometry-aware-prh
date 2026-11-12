"""exp-028 spectral.py: eigenmap vs sklearn and the formula, Nyström, kNN affinity, PPMI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PATH = Path(__file__).resolve().parents[1] / "experiments" / "exp-028-spectral-lexical-floor"
spec = importlib.util.spec_from_file_location("exp028_spectral", PATH / "spectral.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def toy_graph(n=60, seed=0):
    """Three noisy blocks, connected."""
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(3), n // 3)
    p = np.where(lab[:, None] == lab[None, :], 0.5, 0.05)
    a = (rng.random((n, n)) < p) * rng.random((n, n))
    a = np.triu(a, 1)
    a = a + a.T
    return sp.csr_matrix(a)


def subspace_gap(a, b):
    """Largest principal-angle sine between the column spaces of a and b."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.sqrt(max(0.0, 1 - s.min() ** 2)))


def test_matches_formula_and_sklearn():
    from sklearn.manifold import spectral_embedding

    w = toy_graph()
    k = 5
    psi, mu, _tau = S.spectral(w, k, tau=0.0)
    # formula written out: dense eigendecomposition of D^-1/2 A D^-1/2
    a = w.toarray()
    d = a.sum(1)
    m = a / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
    ev, vec = np.linalg.eigh(m)
    order = np.argsort(-ev)
    ref = (vec[:, order] / np.sqrt(d)[:, None])[:, 1 : k + 1]
    assert np.allclose(mu, ev[order][1 : k + 1], atol=1e-8)
    assert subspace_gap(psi, ref) < 1e-6
    # random-walk eigen-equation
    p = a / d[:, None]
    assert np.allclose(p @ psi, psi * mu[None, :], atol=1e-8)
    sk = spectral_embedding(w, n_components=k, norm_laplacian=True, drop_first=True, random_state=0)
    assert subspace_gap(psi, sk) < 1e-5


def test_regularised_eigen_equation():
    w = toy_graph(seed=1)
    psi, mu, tau = S.spectral(w, 4)
    n = w.shape[0]
    a = w.toarray() + tau / n
    p = a / a.sum(1)[:, None]
    assert np.allclose(p @ psi, psi * mu[None, :], atol=1e-8)


def test_nystrom_reproduces_nodes():
    w = toy_graph(seed=2)
    for tau in (0.0, "mean"):
        psi, mu, t = S.spectral(w, 6, tau=tau)
        ext = S.nystrom(w, psi, mu, t)
        assert np.abs(ext - psi).max() < 1e-6


def test_isolated_needs_tau():
    w = toy_graph().tolil()
    w[0, :] = 0
    w[:, 0] = 0
    w = w.tocsr()
    try:
        S.spectral(w, 3, tau=0.0)
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass
    psi, _mu, _tau = S.spectral(w, 3)
    assert np.isfinite(psi).all()


def test_knn_affinity():
    rng = np.random.default_rng(0)
    x = sp.csr_matrix(np.abs(rng.standard_normal((40, 15))) * (rng.random((40, 15)) < 0.3))
    norms = np.sqrt(np.asarray(x.multiply(x).sum(1))).ravel()
    norms[norms == 0] = 1
    x = sp.diags(1 / norms) @ x
    w = S.knn_affinity(sp.csr_matrix(x), 5, chunk=7)
    dense = (x @ x.T).toarray()
    np.fill_diagonal(dense, -np.inf)
    for i in range(40):
        row = w[i].toarray().ravel()
        assert row[i] == 0
        top = np.sort(dense[i])[::-1][:5]
        assert np.allclose(np.sort(row[row > 0])[::-1], top[top > 0])
    assert (S.symmetrise(w) != S.symmetrise(w).T).nnz == 0


def test_ppmi():
    rng = np.random.default_rng(0)
    b = sp.csr_matrix((rng.random((200, 30)) < 0.2).astype(np.float32))
    v = S.ppmi_vectors(b, 5)
    assert v.shape == (30, 5) and np.isfinite(v).all()
    m = S.ppmi_matrix(b)
    # direct formula, loop over pairs, alpha = 0.75
    dense = b.toarray()
    n_wc = dense.T @ dense
    np.fill_diagonal(n_wc, 0)
    n_w = n_wc.sum(1)
    n_c = n_wc.sum(0) ** 0.75
    for i in range(30):
        for j in range(30):
            want = (
                0.0
                if n_wc[i, j] == 0
                else max(0.0, np.log(n_wc[i, j] * n_c.sum() / (n_w[i] * n_c[j])))
            )
            assert abs(m[i, j] - want) < 1e-4
    # full-rank SVD reproduces the matrix: (U S^0.5)(U S^0.5)^T equals PPMI on its PSD part only,
    # so check the top singular value instead
    top = np.linalg.svd(m, compute_uv=False)[0]
    assert abs(np.linalg.norm(S.ppmi_vectors(b, 1, power=1.0)) - top) < 1e-3 * top
