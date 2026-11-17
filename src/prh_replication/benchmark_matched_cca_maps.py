"""Fit task-matched / ordinary-text CCA maps for benchmark transfer."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from prh_replication.cca_steering_transfer import fit_regularised_cca
from prh_replication.privileged_cca_steering_transport import build_maps, select_privileged_pairs
from prh_replication.sae_crosscov import fit_pca, pearson_cols, project


def sha16_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def choose_pca_dim(n_rows: int, requested: int, min_dim: int = 64) -> int:
    """Prespecified fallback when the dataset is too small for stable PCA-256."""
    # need comfortably more rows than components
    cap = max(int(min_dim), min(int(requested), max(8, n_rows // 3)))
    return int(cap)


def fit_pca_cca_maps(
    Xa_fit: np.ndarray,
    Xb_fit: np.ndarray,
    Xa_rank: np.ndarray,
    Xb_rank: np.ndarray,
    *,
    pca_dim: int = 256,
    n_pairs: int = 16,
    eta: float = 0.01,
    support_tol: float = 1e-8,
    min_pca_dim: int = 64,
    family_ids_fit: list[str] | None = None,
    family_ids_rank: list[str] | None = None,
) -> dict[str, Any]:
    """Fit PCA + regularised CCA; select top-n_pairs by signed Pearson on ranking split."""
    Xa_fit = np.asarray(Xa_fit, dtype=np.float64)
    Xb_fit = np.asarray(Xb_fit, dtype=np.float64)
    Xa_rank = np.asarray(Xa_rank, dtype=np.float64)
    Xb_rank = np.asarray(Xb_rank, dtype=np.float64)
    assert Xa_fit.shape[0] == Xb_fit.shape[0]
    assert Xa_rank.shape[0] == Xb_rank.shape[0]

    n_rows = int(Xa_fit.shape[0])
    pca_used = choose_pca_dim(n_rows, int(pca_dim), int(min_pca_dim))
    pca_a = fit_pca(Xa_fit, q=pca_used)
    pca_b = fit_pca(Xb_fit, q=pca_used)
    Za = project(Xa_fit, pca_a["mean"], pca_a["components"])
    Zb = project(Xb_fit, pca_b["mean"], pca_b["components"])
    cca = fit_regularised_cca(Za, Zb, eta=float(eta), support_tol=float(support_tol))

    Za_r = project(Xa_rank, pca_a["mean"], pca_a["components"])
    Zb_r = project(Xb_rank, pca_b["mean"], pca_b["components"])
    rho_rank = pearson_cols(Za_r @ cca["L_A"], Zb_r @ cca["L_B"])
    # preserve paired signs: flip B so rho >= 0 before selection, then keep sign in encoders via σ
    signs = np.sign(rho_rank)
    signs[signs == 0] = 1.0
    L_B_signed = cca["L_B"] * signs.reshape(1, -1)
    rho_signed = pearson_cols(Za_r @ cca["L_A"], Zb_r @ L_B_signed)
    idx = select_privileged_pairs(rho_signed, int(n_pairs))

    # fit-split score stds over all CCA modes (build_maps indexes with idx)
    Sa_all = Za @ cca["L_A"]
    Sb_all = Zb @ L_B_signed
    sigma_a = np.maximum(np.std(Sa_all, axis=0, ddof=1), 1e-12)
    sigma_b = np.maximum(np.std(Sb_all, axis=0, ddof=1), 1e-12)

    map_a = build_maps(pca_a["components"], cca["L_A"], sigma_a, idx)
    map_b = build_maps(pca_b["components"], L_B_signed, sigma_b, idx)

    E_A, E_B = map_a["E"], map_b["E"]
    D_A, D_B = map_a["D"], map_b["D"]
    M_A = E_A.T @ E_A
    M_B = E_B.T @ E_B

    n_fam_fit = len(set(family_ids_fit)) if family_ids_fit else None
    n_fam_rank = len(set(family_ids_rank)) if family_ids_rank else None

    return {
        "E_A": E_A,
        "E_B": E_B,
        "D_A": D_A,
        "D_B": D_B,
        "M_A": M_A,
        "M_B": M_B,
        "U_A": pca_a["components"],
        "U_B": pca_b["components"],
        "mu_A": pca_a["mean"],
        "mu_B": pca_b["mean"],
        "L_A": cca["L_A"],
        "L_B": L_B_signed,
        "idx": idx,
        "sigma_a": sigma_a,
        "sigma_b": sigma_b,
        "rho_rank": rho_signed,
        "rho_rank_selected": rho_signed[idx],
        "pca_dim_requested": int(pca_dim),
        "pca_dim_used": int(pca_used),
        "pca_dim_reduced": bool(pca_used < int(pca_dim)),
        "n_pairs": int(len(idx)),
        "n_activation_rows_fit": n_rows,
        "n_activation_rows_rank": int(Xa_rank.shape[0]),
        "n_families_fit": n_fam_fit,
        "n_families_rank": n_fam_rank,
        "eta": float(eta),
        "sha": {
            "E_A": sha16_arr(E_A),
            "E_B": sha16_arr(E_B),
            "D_A": sha16_arr(D_A),
            "D_B": sha16_arr(D_B),
            "M_A": sha16_arr(M_A),
            "M_B": sha16_arr(M_B),
        },
    }


def load_archived_maps(parent_dir, geo_path) -> dict[str, Any]:
    import numpy as np
    from pathlib import Path

    parent_dir = Path(parent_dir)
    maps = np.load(parent_dir / "artifacts" / "maps.npz")
    geo = np.load(geo_path)
    E_A = np.asarray(maps["E_A"], dtype=np.float64)
    E_B = np.asarray(maps["E_B"], dtype=np.float64)
    D_A = np.asarray(maps["D_A"], dtype=np.float64)
    D_B = np.asarray(maps["D_B"], dtype=np.float64)
    return {
        "E_A": E_A,
        "E_B": E_B,
        "D_A": D_A,
        "D_B": D_B,
        "M_A": E_A.T @ E_A,
        "M_B": E_B.T @ E_B,
        "U_A": np.asarray(geo["U_A"], dtype=np.float64),
        "U_B": np.asarray(geo["U_B"], dtype=np.float64),
        "idx": np.asarray(maps["idx"], dtype=np.int64),
        "source": "archived_presentation_map",
        "sha": {
            "E_A": sha16_arr(E_A),
            "E_B": sha16_arr(E_B),
            "D_A": sha16_arr(D_A),
            "D_B": sha16_arr(D_B),
        },
    }


def decoder_right_inverse_ok(E: np.ndarray, D: np.ndarray, atol: float = 1e-5) -> bool:
    k = E.shape[0]
    rng = np.random.default_rng(0)
    z = rng.standard_normal((32, k))
    z_hat = (z @ D.T) @ E.T
    return bool(np.allclose(z_hat, z, atol=atol))


def family_equal_weights(family_ids: list[str], n_rows: int) -> np.ndarray:
    """Weight so each family contributes equally (rows within family share mass)."""
    fams = np.asarray(family_ids)
    w = np.zeros(n_rows, dtype=np.float64)
    for f in np.unique(fams):
        m = fams == f
        w[m] = 1.0 / (m.sum() * len(np.unique(fams)))
    return w * n_rows  # mean weight ≈ 1
