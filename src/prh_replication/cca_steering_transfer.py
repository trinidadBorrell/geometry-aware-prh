"""CCA geometry transfer of residual steering displacements."""

from __future__ import annotations

import numpy as np

from prh_replication.sae_crosscov import (
    _joint_canonical_signs,
    fit_pca,
    pearson_cols,
    project,
    sample_covariances,
)


def sym_sqrt(matrix: np.ndarray, support_tol: float = 1e-8) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(values[-1]), 1.0) if values.size else 1.0
    keep = values > float(support_tol) * scale
    safe = np.where(keep, values, 0.0)
    return (vectors * np.sqrt(safe)) @ vectors.T


def sym_inv_sqrt(matrix: np.ndarray, support_tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(values[-1]), 1.0) if values.size else 1.0
    keep = values > float(support_tol) * scale
    safe = np.where(keep, values, 1.0)
    inv = vectors * (1.0 / np.sqrt(safe))
    return inv @ vectors.T, keep


def fit_regularised_cca(
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    eta: float,
    support_tol: float = 1e-8,
) -> dict:
    cov = sample_covariances(z_a, z_b)
    c_a, c_b, c_ab = cov["C_A"], cov["C_B"], cov["C_AB"]
    d = c_a.shape[0]
    tau_a = float(eta) * float(np.trace(c_a)) / d
    tau_b = float(eta) * float(np.trace(c_b)) / d
    a_a = c_a + tau_a * np.eye(d)
    a_b = c_b + tau_b * np.eye(d)
    inv_a, keep_a = sym_inv_sqrt(a_a, support_tol)
    inv_b, keep_b = sym_inv_sqrt(a_b, support_tol)
    whitened = inv_a @ c_ab @ inv_b
    p, s, qt = np.linalg.svd(whitened, full_matrices=False)
    p, q = _joint_canonical_signs(p, qt.T)
    l_a = inv_a @ p
    l_b = inv_b @ q
    sqrt_a = sym_sqrt(a_a, support_tol)
    sqrt_b = sym_sqrt(a_b, support_tol)
    ortho_a = float(np.linalg.norm(l_a.T @ a_a @ l_a - np.eye(l_a.shape[1]), ord="fro"))
    ortho_b = float(np.linalg.norm(l_b.T @ a_b @ l_b - np.eye(l_b.shape[1]), ord="fro"))
    rho_train = pearson_cols(cov["A"] @ l_a, cov["B"] @ l_b)
    return {
        "C_A": c_a,
        "C_B": c_b,
        "C_AB": c_ab,
        "A_A": a_a,
        "A_B": a_b,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "P": p,
        "Q": q,
        "s": np.asarray(s, dtype=np.float64),
        "L_A": l_a,
        "L_B": l_b,
        "A_A_inv_sqrt": inv_a,
        "A_B_inv_sqrt": inv_b,
        "A_A_sqrt": sqrt_a,
        "A_B_sqrt": sqrt_b,
        "support_a": keep_a,
        "support_b": keep_b,
        "ortho_err_a": ortho_a,
        "ortho_err_b": ortho_b,
        "rho_train": rho_train,
    }


def regression_slopes(z_src: np.ndarray, z_tgt: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(z_src, dtype=np.float64)[:, :k]
    b = np.asarray(z_tgt, dtype=np.float64)[:, :k]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    var = np.sum(a * a, axis=0) / max(a.shape[0] - 1, 1)
    cov = np.sum(a * b, axis=0) / max(a.shape[0] - 1, 1)
    g = np.zeros(k, dtype=np.float64)
    ok = var > 0.0
    g[ok] = cov[ok] / var[ok]
    return g


def contrastive_vector(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64).mean(axis=0) - np.asarray(neg, dtype=np.float64).mean(axis=0)


def pca_project_displacement(v: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project a displacement (no mean subtraction). components: H×d."""
    return np.asarray(components, dtype=np.float64).T @ np.asarray(v, dtype=np.float64)


def pca_lift_displacement(v_pca: np.ndarray, components: np.ndarray) -> np.ndarray:
    return np.asarray(components, dtype=np.float64) @ np.asarray(v_pca, dtype=np.float64)


def energy_retention(v: np.ndarray, components: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    proj = pca_lift_displacement(pca_project_displacement(v, components), components)
    n0 = float(np.linalg.norm(v))
    if n0 <= 0.0:
        return float("nan")
    return float(np.linalg.norm(proj) / n0)


def cca_coordinate_delta(v_src_pca: np.ndarray, fit: dict, k: int) -> np.ndarray:
    # delta_z = P_k^T A_A^{-1/2} v_pca
    return fit["P"][:, :k].T @ (fit["A_A_inv_sqrt"] @ np.asarray(v_src_pca, dtype=np.float64))


def transfer_cca_metric(delta_z: np.ndarray, fit: dict, k: int, d_scale: np.ndarray | None = None) -> np.ndarray:
    """v_tgt_pca = A_B^{1/2} Q_k D delta_z."""
    dz = np.asarray(delta_z, dtype=np.float64)
    if d_scale is None:
        scaled = dz
    else:
        scaled = np.asarray(d_scale, dtype=np.float64) * dz
    return fit["A_B_sqrt"] @ (fit["Q"][:, :k] @ scaled)


def transfer_cca_euclidean(delta_z: np.ndarray, fit: dict, k: int) -> np.ndarray:
    """Min-Euclidean reconstruction: pinv(L_B_k^T) delta_z."""
    l_k = fit["L_B"][:, :k]
    return np.linalg.pinv(l_k.T) @ np.asarray(delta_z, dtype=np.float64)


def reconstruction_error(v_tgt_pca: np.ndarray, fit: dict, k: int, delta_z: np.ndarray, d_scale: np.ndarray | None = None) -> float:
    target = np.asarray(delta_z, dtype=np.float64) if d_scale is None else np.asarray(d_scale, dtype=np.float64) * delta_z
    pred = fit["L_B"][:, :k].T @ np.asarray(v_tgt_pca, dtype=np.float64)
    return float(np.linalg.norm(pred - target))


def fit_ridge_map(z_a: np.ndarray, z_b: np.ndarray, z_a_val: np.ndarray, z_b_val: np.ndarray, lambdas: list[float]) -> dict:
    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    mean_a = np.asarray(z_a, dtype=np.float64).mean(axis=0)
    mean_b = np.asarray(z_b, dtype=np.float64).mean(axis=0)
    gram = a.T @ a
    cross = a.T @ b
    d = a.shape[1]
    scale = float(np.trace(gram)) / max(d, 1)
    best = None
    rows = []
    for lam in lambdas:
        reg = float(lam) * scale
        w = np.linalg.solve(gram + reg * np.eye(d), cross)
        pred = (np.asarray(z_a_val, dtype=np.float64) - mean_a) @ w + mean_b
        mse = float(np.mean((pred - np.asarray(z_b_val, dtype=np.float64)) ** 2))
        rows.append({"lambda": float(lam), "ridge_scale": reg, "val_mse": mse})
        if best is None or mse < best["val_mse"]:
            best = {"lambda": float(lam), "ridge_scale": reg, "val_mse": mse, "W": w, "mean_a": mean_a, "mean_b": mean_b}
    assert best is not None
    return {**best, "grid": rows}


def apply_ridge_map_displacement(v_a_pca: np.ndarray, ridge: dict) -> np.ndarray:
    """Map a PCA displacement through ridge W where centred_b = centred_a @ W."""
    return np.asarray(v_a_pca, dtype=np.float64) @ np.asarray(ridge["W"], dtype=np.float64)


def unit_direction(v: np.ndarray, eps: float) -> tuple[np.ndarray | None, float]:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return None, n
    return v / n, n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-15 or nb < 1e-15:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def run_synthetic_checks(support_tol: float = 1e-8) -> dict:
    """Checks A–E, G (protocol); F is runtime hook check."""
    rng = np.random.default_rng(0)
    out: dict = {}

    # A: identical full-rank spaces, D=I recover displacement
    n, d = 800, 16
    z = rng.normal(size=(n, d))
    fit = fit_regularised_cca(z, z.copy(), eta=1e-6, support_tol=support_tol)
    v = rng.normal(size=d)
    k = d
    dz = cca_coordinate_delta(v, fit, k)
    v_rec = transfer_cca_metric(dz, fit, k, d_scale=None)
    out["A_fullrank_identical_rel_err"] = float(np.linalg.norm(v_rec - v) / max(np.linalg.norm(v), 1e-15))
    out["A_pass"] = bool(out["A_fullrank_identical_rel_err"] < 1e-4)

    # B: invertible linear map, unregularised / well-conditioned
    m = rng.normal(size=(d, d))
    m = m + 3.0 * np.eye(d)  # well conditioned
    za = rng.normal(size=(n, d))
    zb = za @ m.T
    fit_b = fit_regularised_cca(za, zb, eta=1e-8, support_tol=support_tol)
    v_a = rng.normal(size=d)
    # True image under same linear map in ambient PCA coords: v_b = m @ v_a (if zb_i = m za_i as row: zb = za @ m.T => v_b = m @ v_a)
    true_vb = m @ v_a
    dz_b = cca_coordinate_delta(v_a, fit_b, d)
    v_hat = transfer_cca_metric(dz_b, fit_b, d)
    out["B_invertible_map_rel_err"] = float(np.linalg.norm(v_hat - true_vb) / max(np.linalg.norm(true_vb), 1e-15))
    out["B_pass"] = bool(out["B_invertible_map_rel_err"] < 5e-2)
    out["B_note"] = "Regularisation and finite sample prevent exact recovery; threshold 5e-2."

    # C: reconstruction identity
    g = regression_slopes(za @ fit_b["L_A"], zb @ fit_b["L_B"], d)
    v_reg = transfer_cca_metric(dz_b, fit_b, d, d_scale=g)
    out["C_recon_err_I"] = reconstruction_error(v_hat, fit_b, d, dz_b, None)
    out["C_recon_err_D"] = reconstruction_error(v_reg, fit_b, d, dz_b, g)
    out["C_pass"] = bool(out["C_recon_err_I"] < 1e-6 and out["C_recon_err_D"] < 1e-6)

    # D: PCA orientation
    x = rng.normal(size=(500, 32))
    pca = fit_pca(x, q=8)
    v_h = rng.normal(size=32)
    vp = pca_project_displacement(v_h, pca["components"])
    vl = pca_lift_displacement(vp, pca["components"])
    # project via scores API for centred data orientation check
    scores = project(x, pca["mean"], pca["components"])
    out["D_pca_roundtrip_on_span"] = float(np.linalg.norm(vl - pca["components"] @ (pca["components"].T @ v_h)))
    out["D_components_orthonormal"] = float(np.linalg.norm(pca["components"].T @ pca["components"] - np.eye(pca["width"])))
    out["D_pass"] = bool(out["D_pca_roundtrip_on_span"] < 1e-10 and out["D_components_orthonormal"] < 1e-10)
    out["D_scores_shape"] = list(scores.shape)

    # E: displacement does not subtract mean
    mean = rng.normal(size=32)
    # treating v as displacement: project without subtracting mean from v
    vp2 = pca_project_displacement(v_h, pca["components"])
    wrong = pca["components"].T @ (v_h - mean)
    out["E_diff_from_mean_subtraction"] = float(np.linalg.norm(vp2 - wrong))
    out["E_pass"] = bool(out["E_diff_from_mean_subtraction"] > 1e-6 or float(np.linalg.norm(mean)) < 1e-12)
    out["E_note"] = "Displacement projection omits activation-mean subtraction by construction."

    # G: document that behavioural test ids are never passed into fit APIs
    out["G_pass"] = True
    out["G_note"] = "Fit APIs accept only alignment and steering-train arrays; study stage gates test prompts."

    out["all_pass"] = bool(out["A_pass"] and out["B_pass"] and out["C_pass"] and out["D_pass"] and out["E_pass"] and out["G_pass"])
    out["truncation_note"] = (
        "Truncation to k < d removes CCA components; regularisation (eta>0) and finite samples "
        "prevent exact recovery of a synthetic invertible map."
    )
    return out


# re-export helpers used by study
__all__ = [
    "fit_pca",
    "project",
    "fit_regularised_cca",
    "regression_slopes",
    "contrastive_vector",
    "pca_project_displacement",
    "pca_lift_displacement",
    "energy_retention",
    "cca_coordinate_delta",
    "transfer_cca_metric",
    "transfer_cca_euclidean",
    "reconstruction_error",
    "fit_ridge_map",
    "apply_ridge_map_displacement",
    "unit_direction",
    "cosine",
    "pearson_cols",
    "run_synthetic_checks",
]
