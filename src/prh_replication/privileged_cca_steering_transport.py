"""Privileged ordinary-text CCA encoders/decoders for the transport pilot."""

from __future__ import annotations

from typing import Any

import numpy as np

from prh_replication.sae_crosscov import pearson_cols, project


def select_privileged_pairs(rho: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest positive Pearson correlations."""
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    pos = np.where(rho > 0)[0]
    if pos.size == 0:
        raise RuntimeError("no positive CCA correlations")
    order = pos[np.argsort(-rho[pos])]
    return np.asarray(order[: int(min(k, order.size))], dtype=np.int64)


def build_maps(
    U: np.ndarray,
    L: np.ndarray,
    sigma: np.ndarray,
    idx: np.ndarray,
) -> dict[str, Any]:
    """E: native Δh → standardised privileged coords; D = pinv(E)."""
    U = np.asarray(U, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    idx = np.asarray(idx, dtype=np.int64)
    Ls = L[:, idx]
    sig = sigma[idx]
    # z = ((U.T v) @ Ls) / sig  =>  E = diag(1/sig) @ Ls.T @ U.T
    E = (Ls.T @ U.T) / sig.reshape(-1, 1)
    D, diag = pinv_decode_matrix(E)
    return {"E": E, "D": D, "idx": idx, "sigma_sel": sig, "decoder": diag}


def pinv_decode_matrix(E: np.ndarray, *, tol: float = 1e-8) -> tuple[np.ndarray, dict[str, Any]]:
    E = np.asarray(E, dtype=np.float64)
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    smax = float(s[0]) if s.size else 0.0
    keep = s > float(tol) * smax if smax > 0 else np.zeros_like(s, dtype=bool)
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    D = (Vt.T * s_inv) @ U.T
    cond = float(s[0] / s[keep][-1]) if np.any(keep) else float("inf")
    return D, {"singular_values": s, "n_kept": int(np.sum(keep)), "cond": cond, "tol": tol}


def encode(v: np.ndarray, E: np.ndarray) -> np.ndarray:
    return np.asarray(E, dtype=np.float64) @ np.asarray(v, dtype=np.float64).reshape(-1)


def decode(z: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.asarray(D, dtype=np.float64) @ np.asarray(z, dtype=np.float64).reshape(-1)


def reproduce_err(E: np.ndarray, D: np.ndarray, z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(E @ (D @ z) - z) / (np.linalg.norm(z) + 1e-30))


def energy_in_columns(v: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of ‖v‖² in column span of basis."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    B = np.asarray(basis, dtype=np.float64)
    proj = B @ (B.T @ v)
    return float(np.dot(proj, proj) / (np.dot(v, v) + 1e-30))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def score_stds(X: np.ndarray, mu: np.ndarray, U: np.ndarray, L: np.ndarray) -> np.ndarray:
    Z = project(X, mu, U) @ L
    return np.std(Z, axis=0, ddof=1)


def ranking_rho(Xa: np.ndarray, Xb: np.ndarray, mu_a, U_a, L_a, mu_b, U_b, L_b) -> np.ndarray:
    Za = project(Xa, mu_a, U_a) @ L_a
    Zb = project(Xb, mu_b, U_b) @ L_b
    return pearson_cols(Za, Zb)


def match_budget(v: np.ndarray, alpha: float, r: float, *, near_zero_rel: float = 1e-8) -> dict[str, Any]:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm < float(near_zero_rel) * max(float(r), 1.0):
        return {"vector": np.zeros_like(v), "natural_norm": nrm, "degenerate": True, "applied_norm": 0.0}
    out = float(alpha) * float(r) * (v / nrm)
    return {"vector": out, "natural_norm": nrm, "degenerate": False, "applied_norm": float(np.linalg.norm(out))}


def residual_scale_median(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    xc = X - X.mean(0, keepdims=True)
    return float(np.median(np.linalg.norm(xc, axis=1)))
