"""Nonorthogonal factor metrics on a frozen PCA subspace.

B = A^{-T} D A^{-1} is the inner product in coordinates z = A^{-1} u.
It is not A D A^{-1}. The full metric is M = I + U (B - I) U^T, with the
residual subspace left at identity.

Column normalisation absorbs column scales into D and does not by itself
change B. The singular-value clip that follows can change B. Both steps sit
outside the autograd graph. Gradients are taken at the accepted factors.
exp(S) keeps torch.matrix_exp and the audited spectral projection.

When a factor step leaves the eigenvalue box or the distortion budget, the
induced B is clipped and scaled with that same projection, then refactored
into the declared (A, D). The stored factors are the ones that produce the
accepted B. Line search remains only as a fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from prh_replication.anisotropic_kernels import (
    LOG4,
    _scores_from_deltas,
    knn_from_sq,
    make_pack,
    metric_sq_distances,
    mnn_from_knn,
    n_params,
    pack_to_torch,
    scores_numpy,
    to_numpy64,
)
from prh_replication.metric_stability import (
    _excess_torch,
    _grad_to_vech,
    _selection_better,
    b_matrix_exp,
    draw_quarter_indices,
    dump_rng,
    initial_vech,
    load_rng,
    project_matrices,
)
from prh_replication.release_anisotropy import vech_from_s

SV_LO = 0.5
SV_HI = 2.0
EIG_LO = 0.25
EIG_HI = 4.0
BOUND_TOL = 1e-6
DIST_TOL = 1e-7
ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 1e-3, 0.0)
MODES = ("a_only", "shared_pc", "separate")
FAMILIES = ("exp_s", "fixed_diag", "oblique_diag", "block2", "block4")


def family_block(family: str) -> int:
    if family == "exp_s":
        return 0
    if family in ("fixed_diag", "oblique_diag"):
        return 1
    if family == "block2":
        return 2
    if family == "block4":
        return 4
    raise ValueError(family)


def learns_basis(family: str) -> bool:
    return family in ("oblique_diag", "block2", "block4")


def block_mask(q: int, block: int) -> np.ndarray:
    if block < 1 or q % block != 0:
        raise ValueError("block size must divide q")
    mask = np.zeros((q, q), dtype=np.float64)
    for start in range(0, q, block):
        mask[start : start + block, start : start + block] = 1.0
    return mask


def mask_symmetric(h: np.ndarray, block: int) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    q = h.shape[0] if h.ndim == 2 else h.shape[0]
    if h.ndim == 1:
        h = np.diag(h)
    h = 0.5 * (h + h.T)
    if block <= 1:
        return np.diag(np.diag(h))
    return h * block_mask(q, block)


def sym_expm(h: np.ndarray) -> np.ndarray:
    t = torch.tensor(np.asarray(h, dtype=np.float64), dtype=torch.float64)
    with torch.no_grad():
        t = 0.5 * (t + t.T)
        out = torch.matrix_exp(t)
        out = 0.5 * (out + out.T)
    return out.cpu().numpy()


def sym_logm_blocks(d: np.ndarray, block: int) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    q = d.shape[0]
    block = 1 if block <= 1 else block
    out = np.zeros((q, q), dtype=np.float64)
    for start in range(0, q, block):
        sl = slice(start, start + block)
        block_d = 0.5 * (d[sl, sl] + d[sl, sl].T)
        ev, vec = np.linalg.eigh(block_d)
        if float(ev.min()) <= 0.0:
            raise ValueError("block weight is not positive definite")
        out[sl, sl] = (vec * np.log(ev)) @ vec.T
    return out


def induced_b(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """B = A^{-T} D A^{-1} via a linear solve. Not A D A^{-1}."""
    a = np.asarray(a, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    eye = np.eye(a.shape[0], dtype=np.float64)
    try:
        y = np.linalg.solve(a, eye)
    except np.linalg.LinAlgError as exc:
        raise ValueError("A is singular") from exc
    b = y.T @ d @ y
    return 0.5 * (b + b.T)


def induced_b_torch(a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(a.shape[0], dtype=torch.float64)
    y = torch.linalg.solve(a, eye)
    b = y.T @ d @ y
    return 0.5 * (b + b.T)


def distortion_of_b(b: np.ndarray) -> tuple[float, np.ndarray]:
    b = 0.5 * (np.asarray(b, dtype=np.float64) + np.asarray(b, dtype=np.float64).T)
    ev = np.linalg.eigvalsh(b)
    if float(ev.min()) <= 0.0 or not np.isfinite(ev).all():
        return float("inf"), ev
    log_ev = np.log(ev)
    return float(np.sum(log_ev * log_ev) / ev.shape[0]), ev


def r_basis(a: np.ndarray) -> float:
    q = a.shape[0]
    gram = a.T @ a - np.eye(q, dtype=np.float64)
    return float(np.sum(gram * gram) / q)


def r_off(h: np.ndarray) -> float:
    h = np.asarray(h, dtype=np.float64)
    if h.ndim == 1:
        return 0.0
    off = h - np.diag(np.diag(h))
    return float(np.sum(off * off) / h.shape[0])


def off_norm(h: np.ndarray) -> float:
    h = np.asarray(h, dtype=np.float64)
    if h.ndim == 1:
        return 0.0
    off = h - np.diag(np.diag(h))
    return float(np.linalg.norm(off))


def singular_values(a: np.ndarray) -> np.ndarray:
    return np.linalg.svd(np.asarray(a, dtype=np.float64), compute_uv=False)


def condition_number(sv: np.ndarray) -> float:
    sv = np.asarray(sv, dtype=np.float64)
    if float(sv.min()) <= 0.0:
        return float("inf")
    return float(sv.max() / sv.min())


def weighting_complexity(q: int, family: str) -> int:
    """Off-diagonal weighting parameters in one D, or in one S for exp(S)."""
    block = family_block(family)
    if family == "exp_s":
        return int(q * (q - 1) // 2)
    if block <= 1:
        return 0
    n_blocks = q // block
    return int(n_blocks * block * (block - 1) // 2)


def complexity_total(q: int, family: str, mode: str) -> int:
    one = weighting_complexity(q, family)
    if mode == "separate":
        return int(2 * one)
    if mode in ("a_only", "shared_pc"):
        return int(one)
    if mode == "identity":
        return 0
    raise ValueError(mode)


def penalised_selection(val_excess: float, c_total: int, q: int, coef: float = 0.01) -> float:
    denom = q * (q - 1) / 2.0
    return float(val_excess) - float(coef) * float(c_total) / denom


def sorted_log_spectrum(b: np.ndarray) -> np.ndarray:
    _, ev = distortion_of_b(b)
    if not np.isfinite(ev).all() or float(ev.min()) <= 0.0:
        raise ValueError("spectrum requires a positive definite matrix")
    return np.sort(np.log(ev))


def spectrum_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mean_subtract(log_ev: np.ndarray) -> np.ndarray:
    v = np.asarray(log_ev, dtype=np.float64)
    return v - float(v.mean())


def _column_norms(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(a, dtype=np.float64, copy=True)
    q = a.shape[0]
    norms = np.linalg.norm(a, axis=0)
    for j in range(q):
        if norms[j] < 1e-8 or not np.isfinite(norms[j]):
            a[:, j] = 0.0
            a[j, j] = 1.0
            norms[j] = 1.0
    return a, norms


def project_factors(a: np.ndarray, h: np.ndarray, family: str) -> dict[str, Any]:
    """Gauge-fix columns, clip singular values, rebuild D and B."""
    block = family_block(family)
    q = int(np.asarray(h).shape[-1] if np.asarray(h).ndim == 1 else np.asarray(h).shape[0])
    if family == "fixed_diag":
        hh = mask_symmetric(h, 1)
        d = sym_expm(hh)
        b = 0.5 * (d + d.T)
        dist, ev = distortion_of_b(b)
        eye = np.eye(q, dtype=np.float64)
        return _pack_factors(eye, hh, d, b, np.ones(q), dist, ev)
    if not learns_basis(family):
        raise ValueError(family)
    raw, norms = _column_norms(a)
    a_unit = raw / norms
    hh = mask_symmetric(h, block)
    d = sym_expm(hh)
    # A = A_unit diag(norms). Preserving B requires D_new = diag(norms)^{-1} D diag(norms)^{-1}.
    inv_scale = np.diag(1.0 / norms)
    d_abs = inv_scale @ d @ inv_scale
    d_abs = mask_symmetric(d_abs, block) if block > 1 else np.diag(np.diag(d_abs))
    d_abs = 0.5 * (d_abs + d_abs.T)
    try:
        h_abs = sym_logm_blocks(d_abs, block)
    except ValueError:
        h_abs = hh
        d_abs = d
    u, sv, vt = np.linalg.svd(a_unit, full_matrices=False)
    sv_clip = np.clip(sv, SV_LO, SV_HI)
    a_clip = (u * sv_clip) @ vt
    d_final = sym_expm(h_abs)
    if block > 1:
        d_final = d_final * block_mask(q, block)
        d_final = 0.5 * (d_final + d_final.T)
    b = induced_b(a_clip, d_final)
    dist, ev = distortion_of_b(b)
    return _pack_factors(a_clip, h_abs, d_final, b, sv_clip, dist, ev)


def _pack_factors(a, h, d, b, sv, dist, ev) -> dict[str, Any]:
    return {
        "A": np.asarray(a, dtype=np.float64),
        "H": np.asarray(h, dtype=np.float64),
        "D": np.asarray(d, dtype=np.float64),
        "B": np.asarray(b, dtype=np.float64),
        "singular_values": np.asarray(sv, dtype=np.float64),
        "distortion": float(dist),
        "eig": np.asarray(ev, dtype=np.float64),
        "r_basis": r_basis(a),
        "r_off": r_off(h),
        "cond": condition_number(sv),
    }


def eig_residual(ev: np.ndarray) -> float:
    if ev.size == 0 or not np.isfinite(ev).all():
        return float("inf")
    return float(max(0.0, EIG_LO - float(ev.min()), float(ev.max()) - EIG_HI))


def total_distortion(parts: list[dict[str, Any]], mode: str) -> float:
    if mode == "a_only":
        return float(parts[0]["distortion"])
    if mode == "shared_pc":
        return float(2.0 * parts[0]["distortion"])
    if mode == "separate":
        return float(parts[0]["distortion"] + parts[1]["distortion"])
    raise ValueError(mode)


def factors_feasible(parts: list[dict[str, Any]], mode: str, rho: float) -> bool:
    if not np.isfinite(total_distortion(parts, mode)):
        return False
    if total_distortion(parts, mode) > float(rho) + DIST_TOL:
        return False
    for part in parts:
        if eig_residual(part["eig"]) > BOUND_TOL:
            return False
        sv = part["singular_values"]
        if float(sv.min()) < SV_LO - 1e-8 or float(sv.max()) > SV_HI + 1e-8:
            return False
        if not np.isfinite(part["B"]).all():
            return False
    return True


def identity_factors(q: int, family: str) -> dict[str, Any]:
    eye = np.eye(q, dtype=np.float64)
    h = np.zeros((q, q), dtype=np.float64)
    if family == "fixed_diag":
        return project_factors(eye, np.zeros(q), family)
    return project_factors(eye, h, family)


def init_raw_factors(q: int, seed: int, pert: float, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Full Gaussian draw, masked later. Seed 0 is identity. Side changes the stream."""
    if int(seed) == 0:
        return np.eye(q, dtype=np.float64), np.zeros((q, q), dtype=np.float64)
    rng = np.random.default_rng(10_000 * int(side) + int(seed))
    drift = float(pert) * rng.normal(size=(q, q))
    sym = float(pert) * rng.normal(size=(q, q))
    sym = 0.5 * (sym + sym.T)
    return np.eye(q, dtype=np.float64) + drift, sym


def _lerp(prev: np.ndarray, proposed: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * prev + alpha * proposed


def _active_count(mode: str) -> int:
    if mode == "separate":
        return 2
    if mode in ("a_only", "shared_pc"):
        return 1
    raise ValueError(mode)


def realize_state(raw: list[tuple[np.ndarray, np.ndarray]], family: str, mode: str, rho: float) -> tuple[list[dict[str, Any]], float] | None:
    try:
        built = [project_factors(a, h, family) for a, h in raw]
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None
    if mode == "shared_pc":
        stored = [built[0], built[0]]
        check = [built[0]]
    else:
        stored = built
        check = built
    if not factors_feasible(check, mode, rho):
        return None
    return stored, 1.0


def backtrack(prev_raw: list[tuple[np.ndarray, np.ndarray]], proposed_raw: list[tuple[np.ndarray, np.ndarray]], prev_parts: list[dict[str, Any]], family: str, mode: str, rho: float) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]], float]:
    """Largest feasible step from the previous factors toward the proposal.

    Alpha 0 returns the previous factors unchanged. Other alphas lerp the raw
    matrices and then apply the factor projection, so the stored A and D match B.
    """
    for alpha in ALPHAS:
        if alpha == 0.0:
            return prev_parts, prev_raw, 0.0
        trial = [(_lerp(a0, a1, alpha), _lerp(h0, h1, alpha)) for (a0, h0), (a1, h1) in zip(prev_raw, proposed_raw)]
        got = realize_state(trial, family, mode, rho)
        if got is not None:
            parts, _ = got
            accepted_raw = [(p["A"].copy(), p["H"].copy()) for p in parts[: _active_count(mode)]]
            return parts, accepted_raw, float(alpha)
    return prev_parts, prev_raw, 0.0


def _sym_logm(b: np.ndarray) -> np.ndarray:
    b = 0.5 * (np.asarray(b, dtype=np.float64) + np.asarray(b, dtype=np.float64).T)
    ev, vec = np.linalg.eigh(b)
    ev = np.clip(ev, 1e-30, None)
    return (vec * np.log(ev)) @ vec.T


def _align_basis(u: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Permute and sign columns so the basis stays near I when the spectrum is simple."""
    q = int(u.shape[0])
    score = np.abs(np.asarray(u, dtype=np.float64)).copy()
    perm = np.empty(q, dtype=int)
    for _ in range(q):
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        perm[i] = j
        score[i, :] = -1.0
        score[:, j] = -1.0
    signs = np.sign(u[np.arange(q), perm])
    signs[signs == 0.0] = 1.0
    return u[:, perm] * signs, np.asarray(lam, dtype=np.float64)[perm]


def project_induced_matrices(bs: list[np.ndarray], mode: str, rho: float) -> list[np.ndarray]:
    """Eigenvalue clip and equal-total distortion scale, matching exp(S)."""
    q = int(bs[0].shape[0])
    s_a = torch.tensor(_sym_logm(bs[0]), dtype=torch.float64)
    if mode == "separate":
        if len(bs) != 2:
            raise ValueError("separate retraction needs both sides")
        s_b = torch.tensor(_sym_logm(bs[1]), dtype=torch.float64)
    elif mode == "shared_pc":
        s_b = s_a.clone()
    elif mode == "a_only":
        s_b = torch.zeros((q, q), dtype=torch.float64)
    else:
        raise ValueError(mode)
    s_a, s_b = project_matrices(s_a, s_b, rho=float(rho), accounting="equal_total", mode=mode)
    out = [b_matrix_exp(s_a.detach().cpu().numpy())]
    if mode == "separate":
        out.append(b_matrix_exp(s_b.detach().cpu().numpy()))
    return out


def orthogonal_factors(b_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = Q, D = Λ for B* = Q Λ Q^T. Singular values are 1."""
    b_star = 0.5 * (np.asarray(b_star, dtype=np.float64) + np.asarray(b_star, dtype=np.float64).T)
    lam, vec = np.linalg.eigh(b_star)
    vec, lam = _align_basis(vec, lam)
    lam = np.clip(lam, EIG_LO, EIG_HI)
    return vec, np.diag(np.log(lam))


def _block_rotation(lam_block: np.ndarray, h_want: np.ndarray) -> np.ndarray:
    """Orthogonal R such that R^T diag(lam) R uses the eigenvectors of exp(H_want).

    Eigenvalues stay equal to lam, so a later congruence can keep B fixed while
    restoring within-block orientation from the proposed weight.
    """
    h_want = 0.5 * (np.asarray(h_want, dtype=np.float64) + np.asarray(h_want, dtype=np.float64).T)
    _, vec = np.linalg.eigh(sym_expm(h_want))
    perm = np.argsort(np.asarray(lam_block, dtype=np.float64))
    permute = np.eye(lam_block.shape[0], dtype=np.float64)[:, perm]
    return permute @ vec.T


def refactor_projected(
    a: np.ndarray,
    b_star: np.ndarray,
    family: str,
    h_want: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild (A, H) so the induced metric matches the spectrally projected B.

    Diagonal and block families diagonalise A^T B* A, with eigenvectors aligned
    toward I. That congruence reproduces B* before the singular-value clip.
    Block families then rotate inside each block so D shares eigenvectors with
    the proposed block weight. Off-block entries stay zero, and the rotation
    does not change B. If the singular-value clip would leave this
    factorization, the caller stores the orthogonal eigenbasis of B* instead.
    Fixed coordinates keep A = I and the diagonal of log B*.
    """
    b_star = 0.5 * (np.asarray(b_star, dtype=np.float64) + np.asarray(b_star, dtype=np.float64).T)
    q = int(b_star.shape[0])
    if family == "fixed_diag":
        return np.eye(q, dtype=np.float64), np.diag(np.diag(_sym_logm(b_star)))
    block = family_block(family)
    a = np.asarray(a, dtype=np.float64)
    gram = a.T @ b_star @ a
    gram = 0.5 * (gram + gram.T)
    lam, vec = np.linalg.eigh(gram)
    vec, lam = _align_basis(vec, lam)
    lam = np.clip(lam, 1e-30, None)
    a_new = a @ vec
    if block <= 1 or h_want is None:
        return a_new, np.diag(np.log(lam))
    h_want = mask_symmetric(h_want, block)
    rotation = np.eye(q, dtype=np.float64)
    for start in range(0, q, block):
        sl = slice(start, start + block)
        rotation[sl, sl] = _block_rotation(lam[sl], h_want[sl, sl])
    a_new = a_new @ rotation
    d = rotation.T @ np.diag(lam) @ rotation
    d = mask_symmetric(d, block)
    return a_new, sym_logm_blocks(0.5 * (d + d.T), block)


def _raw_from_stored(parts: list[dict[str, Any]], mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(part["A"].copy(), part["H"].copy()) for part in parts[: _active_count(mode)]]


def accept_step(
    prev_raw: list[tuple[np.ndarray, np.ndarray]],
    proposed_raw: list[tuple[np.ndarray, np.ndarray]],
    prev_parts: list[dict[str, Any]],
    family: str,
    mode: str,
    rho: float,
) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]], float, str]:
    """Feasible Adam step, or a factor-consistent retraction onto the same constraints as exp(S).

    Line search toward the previous factors is used only when refactoring cannot
    land inside the eigenvalue box and the distortion budget. Alpha 0 keeps
    the previous factors. The returned factors reproduce the returned B.
    """
    got = realize_state(proposed_raw, family, mode, rho)
    if got is not None:
        parts, _ = got
        return parts, _raw_from_stored(parts, mode), 1.0, "accepted"
    try:
        gauged = [project_factors(a, h, family) for a, h in proposed_raw]
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        parts, raw, alpha = backtrack(prev_raw, proposed_raw, prev_parts, family, mode, rho)
        return parts, raw, alpha, "backtrack"
    if any((not np.isfinite(g["B"]).all()) or (not math.isfinite(float(g["distortion"]))) for g in gauged):
        parts, raw, alpha = backtrack(prev_raw, proposed_raw, prev_parts, family, mode, rho)
        return parts, raw, alpha, "backtrack"
    bstars = project_induced_matrices([g["B"] for g in gauged], mode, rho)
    n_side = _active_count(mode)
    try:
        refactored = [refactor_projected(gauged[i]["A"], bstars[i], family, gauged[i]["H"]) for i in range(n_side)]
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        parts, raw, alpha = backtrack(prev_raw, proposed_raw, prev_parts, family, mode, rho)
        return parts, raw, alpha, "backtrack"
    got = realize_state(refactored, family, mode, rho)
    if got is None and learns_basis(family):
        refactored = [orthogonal_factors(bstars[i]) for i in range(n_side)]
        got = realize_state(refactored, family, mode, rho)
    if got is not None:
        parts, _ = got
        return parts, _raw_from_stored(parts, mode), 1.0, "retracted"
    parts, raw, alpha = backtrack(prev_raw, refactored, prev_parts, family, mode, rho)
    return parts, raw, alpha, "backtrack"


def _family_specs(design: dict) -> list[dict[str, Any]]:
    strong = design["lambda_profiles"]["strong"]
    stronger = design["lambda_profiles"]["stronger"]
    rows = [
        {"family": "exp_s", "profile": "na", "lambda_basis": 0.0, "lambda_off": 0.0},
        {"family": "fixed_diag", "profile": "na", "lambda_basis": 0.0, "lambda_off": 0.0},
        {"family": "oblique_diag", "profile": "basis", "lambda_basis": float(strong["lambda_basis"]), "lambda_off": 0.0},
        {"family": "block2", "profile": "strong", "lambda_basis": float(strong["lambda_basis"]), "lambda_off": float(strong["lambda_off"])},
        {"family": "block2", "profile": "stronger", "lambda_basis": float(stronger["lambda_basis"]), "lambda_off": float(stronger["lambda_off"])},
        {"family": "block4", "profile": "strong", "lambda_basis": float(strong["lambda_basis"]), "lambda_off": float(strong["lambda_off"])},
        {"family": "block4", "profile": "stronger", "lambda_basis": float(stronger["lambda_basis"]), "lambda_off": float(stronger["lambda_off"])},
    ]
    if abs(rows[2]["lambda_basis"] - 0.1) > 1e-12:
        raise ValueError("oblique diagonal uses the configured basis penalty")
    return rows


def job_name(job: dict[str, Any]) -> str:
    rho = f"{float(job['rho']):.1f}".replace(".", "p")
    corr = "true" if job["correspondence"] == "true" else f"shuffle{int(job['perm_seed'])}"
    return (
        f"{job['side_a']}__{job['side_b']}__{job['mode']}__{job['family']}__{job['profile']}"
        f"__rho{rho}__seed{int(job['seed'])}__{corr}"
    )


def build_jobs(design: dict, vl_pairs: list[tuple[str, str]], ll_pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    specs = _family_specs(design)
    jobs: list[dict[str, Any]] = []
    for kind, pairs in (("vl", vl_pairs), ("ll", ll_pairs)):
        for side_a, side_b in pairs:
            for spec in specs:
                for mode in design["modes"]:
                    for rho in design["budgets_rho"]:
                        for seed in design["seeds"]:
                            jobs.append(
                                {
                                    "kind": kind,
                                    "side_a": side_a,
                                    "side_b": side_b,
                                    "mode": mode,
                                    "family": spec["family"],
                                    "profile": spec["profile"],
                                    "lambda_basis": spec["lambda_basis"],
                                    "lambda_off": spec["lambda_off"],
                                    "rho": float(rho),
                                    "seed": int(seed),
                                    "correspondence": "true",
                                    "perm_seed": None,
                                }
                            )
    shuffle_families = {(row["family"], row["profile"]) for row in design["shuffle"]["families"]}
    for side_a, side_b in design["control_pairs"]:
        kind = "ll" if side_b not in {b for _, b in vl_pairs} else "vl"
        # control pairs are the predetermined list; kind follows the partner.
        if any(side_b == b and side_a == a for a, b in ll_pairs):
            kind = "ll"
        elif any(side_b == b and side_a == a for a, b in vl_pairs):
            kind = "vl"
        for spec in specs:
            if (spec["family"], spec["profile"]) not in shuffle_families:
                continue
            for seed in design["seeds"]:
                jobs.append(
                    {
                        "kind": kind,
                        "side_a": side_a,
                        "side_b": side_b,
                        "mode": design["shuffle"]["mode"],
                        "family": spec["family"],
                        "profile": spec["profile"],
                        "lambda_basis": spec["lambda_basis"],
                        "lambda_off": spec["lambda_off"],
                        "rho": float(design["shuffle"]["rho"]),
                        "seed": int(seed),
                        "correspondence": "shuffled",
                        "perm_seed": int(seed),
                    }
                )
    names = [job_name(job) for job in jobs]
    if len(names) != len(set(names)):
        raise RuntimeError("job names are not unique")
    return jobs


def excess_on_pack(pack: dict[str, Any], b_a: np.ndarray, b_b: np.ndarray) -> float:
    b_a = np.asarray(b_a, dtype=np.float64)
    b_b = np.asarray(b_b, dtype=np.float64)
    return float(scores_numpy(b_a - np.eye(b_a.shape[0]), b_b - np.eye(b_b.shape[0]), pack)["excess"])


def score_metrics(features_a, features_b, basis_a, basis_b, b_a, b_b, knn_k: int = 10) -> dict[str, Any]:
    za = to_numpy64(features_a) - to_numpy64(basis_a["mu"])
    zb = to_numpy64(features_b) - to_numpy64(basis_b["mu"])
    b_a = np.asarray(b_a, dtype=np.float64)
    b_b = np.asarray(b_b, dtype=np.float64)
    pack = make_pack(za, zb, basis_a["U"], basis_b["U"])
    sc = scores_numpy(b_a - np.eye(b_a.shape[0]), b_b - np.eye(b_b.shape[0]), pack)
    mnn = mnn_from_knn(
        knn_from_sq(metric_sq_distances(za, basis_a["U"], b_a), knn_k),
        knn_from_sq(metric_sq_distances(zb, basis_b["U"], b_b), knn_k),
    )
    return {
        "a": sc["a"],
        "b": sc["b"],
        "ratio": sc["ratio"] if math.isfinite(sc["ratio"]) else None,
        "excess": sc["excess"],
        "valid_a": sc["valid_a"],
        "degenerate": sc["degenerate"],
        "r_eff_k": sc["r_eff_k"],
        "r_eff_l": sc["r_eff_l"],
        "mnn_k10": mnn,
        "n": int(za.shape[0]),
    }


def generalized_eigenvalues(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Eigenvalues of D v = λ AᵀA v, which match those of B = A^{-T} D A^{-1}."""
    ata = np.asarray(a, dtype=np.float64).T @ np.asarray(a, dtype=np.float64)
    chol = np.linalg.cholesky(0.5 * (ata + ata.T))
    mid = np.linalg.solve(chol, np.asarray(d, dtype=np.float64))
    reduced = np.linalg.solve(chol, mid.T).T
    reduced = 0.5 * (reduced + reduced.T)
    return np.sort(np.linalg.eigvalsh(reduced))


def _penalty_torch(a: torch.Tensor, h: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor]:
    gram = a.T @ a - torch.eye(q, dtype=torch.float64)
    rb = (gram * gram).sum() / q
    if h.ndim == 1:
        ro = torch.zeros((), dtype=torch.float64)
    else:
        off = h - torch.diag(torch.diag(h))
        ro = (off * off).sum() / q
    return rb, ro


def _mask_torch(h: torch.Tensor, block: int) -> torch.Tensor:
    if h.ndim == 1:
        return h
    h = 0.5 * (h + h.T)
    if block <= 1:
        return torch.diag(torch.diag(h))
    q = h.shape[0]
    mask = torch.zeros(q, q, dtype=torch.float64)
    for start in range(0, q, block):
        mask[start : start + block, start : start + block] = 1.0
    return h * mask


def factor_objective(a_parts: list[torch.Tensor], h_parts: list[torch.Tensor], pack_t: dict[str, Any], *, family: str, mode: str, lambda_basis: float, lambda_off: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (penalised loss to minimise, unpenalised excess). H is q by q."""
    block = max(family_block(family), 1)
    built = []
    for a, h in zip(a_parts, h_parts):
        hh = _mask_torch(h, block)
        d = torch.matrix_exp(hh)
        d = 0.5 * (d + d.T)
        b = induced_b_torch(a, d)
        built.append((a, hh, b))
    if mode == "shared_pc":
        b_a = built[0][2]
        b_b = b_a
        penalty_sides = [built[0]]
        mult = 2.0
    elif mode == "a_only":
        b_a = built[0][2]
        q_b = int(pack_t["q_b"])
        b_b = torch.eye(q_b, dtype=torch.float64)
        penalty_sides = [built[0]]
        mult = 1.0
    else:
        b_a, b_b = built[0][2], built[1][2]
        penalty_sides = built
        mult = 1.0
    excess = _scores_from_deltas(b_a - torch.eye(b_a.shape[0], dtype=torch.float64), b_b - torch.eye(b_b.shape[0], dtype=torch.float64), pack_t)["excess"]
    rb = torch.zeros((), dtype=torch.float64)
    ro = torch.zeros((), dtype=torch.float64)
    q = int(a_parts[0].shape[0])
    for a, h, _b in penalty_sides:
        rbi, roi = _penalty_torch(a, h, q)
        rb = rb + rbi
        ro = ro + roi
    if mode == "shared_pc":
        rb = rb * mult
        ro = ro * mult
    loss = -excess + float(lambda_basis) * rb + float(lambda_off) * ro
    return loss, excess


def _as_params(raw: list[tuple[np.ndarray, np.ndarray]], family: str) -> list[dict[str, torch.Tensor]]:
    block = max(family_block(family), 1)
    params = []
    for a, h in raw:
        row: dict[str, torch.Tensor] = {}
        if learns_basis(family):
            row["A"] = torch.tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, requires_grad=True)
        row["H"] = torch.tensor(mask_symmetric(h, block), dtype=torch.float64, requires_grad=True)
        params.append(row)
    return params


def _raw_from_params(params: list[dict[str, torch.Tensor]], family: str, q: int) -> list[tuple[np.ndarray, np.ndarray]]:
    raw = []
    for row in params:
        if learns_basis(family):
            a = row["A"].detach().cpu().numpy().astype(np.float64, copy=True)
        else:
            a = np.eye(q, dtype=np.float64)
        h = row["H"].detach().cpu().numpy().astype(np.float64, copy=True)
        raw.append((a, h))
    return raw


def _exp_project(vechs: list[torch.Tensor], mode: str, rho: float, q: int) -> tuple[np.ndarray, np.ndarray]:
    from prh_replication.anisotropic_kernels import _unvech

    s_a = _unvech(vechs[0].detach(), q)
    if mode == "separate":
        s_b = _unvech(vechs[1].detach(), q)
    else:
        s_b = torch.zeros(q, q, dtype=torch.float64)
    s_a, s_b = project_matrices(s_a, s_b, rho=float(rho), accounting="equal_total", mode=mode)
    return s_a.detach().cpu().numpy(), s_b.detach().cpu().numpy()


def _init_exp(q: int, seed: int, pert: float, mode: str, rho: float) -> list[torch.Tensor]:
    va = initial_vech(q, 0 if int(seed) == 0 else int(seed), pert)
    params = [va.clone()]
    if mode == "separate":
        vb = initial_vech(q, 0 if int(seed) == 0 else int(seed) + 1009, pert)
        params.append(vb.clone())
    tensors = [p.detach().clone().requires_grad_(True) for p in params]
    s_a, s_b = _exp_project(tensors, mode, rho, q)
    with torch.no_grad():
        tensors[0].copy_(vech_from_s(s_a))
        if mode == "separate":
            tensors[1].copy_(vech_from_s(s_b))
    return tensors


def _shrink_init(family: str, mode: str, rho: float, q: int, seed: int, pert: float) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    n_side = _active_count(mode)
    proposed = []
    identity = []
    for side in range(n_side):
        a, h = init_raw_factors(q, seed, pert, side)
        proposed.append((a, h))
        identity.append((np.eye(q), np.zeros((q, q))))
    id_parts = [identity_factors(q, family) for _ in range(n_side)]
    if mode == "shared_pc":
        id_parts = [id_parts[0], id_parts[0]]
    parts, raw, alpha = backtrack(identity, proposed, id_parts if mode != "a_only" else id_parts, family, mode, rho)
    if alpha < 0:
        raise RuntimeError("identity factors were infeasible")
    return parts, raw


def _regulariser_totals(parts: list[dict[str, Any]], mode: str) -> tuple[float, float]:
    if mode == "a_only":
        use = parts[:1]
        mult = 1.0
    elif mode == "shared_pc":
        use = parts[:1]
        mult = 2.0
    else:
        use = parts
        mult = 1.0
    rb = mult * sum(float(p["r_basis"]) for p in use)
    ro = mult * sum(float(p["r_off"]) for p in use)
    return float(rb), float(ro)


def _bs(parts: list[dict[str, Any]], mode: str, q_b: int) -> tuple[np.ndarray, np.ndarray]:
    if mode == "a_only":
        return parts[0]["B"], np.eye(q_b, dtype=np.float64)
    if mode == "shared_pc":
        return parts[0]["B"], parts[0]["B"]
    return parts[0]["B"], parts[1]["B"]


def run_factor_stream(
    features_a: np.ndarray,
    features_b: np.ndarray,
    ids: list[str],
    basis_a: dict[str, Any],
    basis_b: dict[str, Any],
    *,
    family: str,
    mode: str,
    rho: float,
    lambda_basis: float,
    lambda_off: float,
    init_seed: int,
    sampling_seed: int,
    n_updates: int = 120,
    lr: float = 0.03,
    grad_clip: float = 5.0,
    pert_scale: float = 0.05,
    quarter_fraction: float = 0.25,
    val_features_a: np.ndarray | None = None,
    val_features_b: np.ndarray | None = None,
    resume: dict[str, Any] | None = None,
    record_indices: bool = False,
    on_checkpoint: Any = None,
) -> dict[str, Any]:
    """Quarter-gallery fit. Validation excess selects checkpoints. Test rows are not accepted."""
    if family not in FAMILIES or mode not in MODES:
        raise ValueError((family, mode))
    if family_block(family) not in (0, 1) and int(basis_a["q"]) % family_block(family) != 0:
        raise ValueError("block size does not divide q")
    n = int(features_a.shape[0])
    if features_b.shape[0] != n or len(ids) != n or len(set(ids)) != n:
        raise ValueError("training rows and ids must describe one gallery")
    q = int(basis_a["q"])
    q_b = int(basis_b["q"])
    if mode == "shared_pc" and q != q_b:
        raise ValueError("shared_pc requires equal PCA dimension")
    train_pack = make_pack(to_numpy64(features_a) - to_numpy64(basis_a["mu"]), to_numpy64(features_b) - to_numpy64(basis_b["mu"]), basis_a["U"], basis_b["U"])
    val_pack = None
    if val_features_a is not None:
        val_pack = make_pack(
            to_numpy64(val_features_a) - to_numpy64(basis_a["mu"]),
            to_numpy64(val_features_b) - to_numpy64(basis_b["mu"]),
            basis_a["U"],
            basis_b["U"],
        )
    status = {
        "stopped": "completed",
        "nonfinite_objective": 0,
        "nonfinite_grad": 0,
        "backtracks": 0,
        "full_rejects": 0,
        "retractions": 0,
        "max_eig_residual": 0.0,
        "max_dist_residual": 0.0,
    }
    subset_indices: list[list[int]] = []
    optim_seconds = 0.0
    validation_seconds = 0.0
    train_eval_seconds = 0.0
    selected: dict[str, Any] | None = None
    selected_parts: list[dict[str, Any]] | None = None

    if family == "exp_s":
        tensors = _init_exp(q, init_seed, pert_scale, mode, rho) if resume is None else None
        factor_params = None
        parts: list[dict[str, Any]] | None = None
        raw: list[tuple[np.ndarray, np.ndarray]] | None = None
    else:
        tensors = None
        if resume is None:
            parts, raw = _shrink_init(family, mode, rho, q, init_seed, pert_scale)
            factor_params = _as_params(raw, family)
        else:
            factor_params = None
            parts = None
            raw = None

    curves: list[dict[str, Any]] = []
    updates_done = 0
    rng = np.random.Generator(np.random.PCG64(int(sampling_seed)))
    opt_params: list[torch.Tensor] = []
    if resume is None:
        if family == "exp_s":
            opt_params = tensors
        else:
            opt_params = [tensor for row in factor_params for tensor in row.values()]
        opt = torch.optim.Adam(opt_params, lr=float(lr))
    else:
        rng = load_rng(resume["rng"])
        curves = [dict(row) for row in resume["curves"]]
        updates_done = int(resume["updates_done"])
        optim_seconds = float(resume.get("optim_seconds", 0.0))
        validation_seconds = float(resume.get("validation_seconds", 0.0))
        train_eval_seconds = float(resume.get("train_eval_seconds", 0.0))
        status.update(resume.get("status") or {})
        selected = resume.get("selected")
        selected_parts = resume.get("selected_parts")
        if family == "exp_s":
            tensors = [torch.tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, requires_grad=True) for v in resume["vechs"]]
            opt_params = tensors
        else:
            raw = [(np.asarray(a, dtype=np.float64), np.asarray(h, dtype=np.float64)) for a, h in resume["raw"]]
            factor_params = _as_params(raw, family)
            opt_params = [tensor for row in factor_params for tensor in row.values()]
            parts = resume["parts"]
        opt = torch.optim.Adam(opt_params, lr=float(lr))
        from prh_replication.metric_stability import _load_adam

        for tensor, blob in zip(opt_params, resume.get("adam") or []):
            _load_adam(opt, tensor, blob)
        if record_indices and resume.get("subset_indices"):
            subset_indices = [list(map(int, row)) for row in resume["subset_indices"]]

    def current_matrices() -> tuple[np.ndarray, np.ndarray, float, float, float]:
        if family == "exp_s":
            s_a, s_b = _exp_project(tensors, mode, rho, q)
            b_a = b_matrix_exp(s_a)
            b_b = b_matrix_exp(s_b) if mode != "a_only" else np.eye(q_b)
            if mode == "shared_pc":
                b_b = b_a
                s_b = s_a
            da, _ev_a = distortion_of_b(b_a)
            db = 0.0 if mode == "a_only" else distortion_of_b(b_b)[0]
            if mode == "shared_pc":
                dist = 2.0 * da
            elif mode == "separate":
                dist = da + db
            else:
                dist = da
            return b_a, b_b, dist, 0.0, 0.0
        b_a, b_b = _bs(parts, mode, q_b)
        dist = total_distortion([parts[0]] if mode != "separate" else parts, mode)
        rb, ro = _regulariser_totals(parts, mode)
        return b_a, b_b, dist, rb, ro

    def snapshot_parts() -> list[dict[str, Any]]:
        if family == "exp_s":
            s_a, s_b = _exp_project(tensors, mode, rho, q)
            out = [{"S": s_a, "B": b_matrix_exp(s_a)}]
            if mode == "separate":
                out.append({"S": s_b, "B": b_matrix_exp(s_b)})
            elif mode == "shared_pc":
                out.append({"S": s_a.copy(), "B": out[0]["B"].copy()})
            return out
        return parts

    def consider(step: int, val: float | None, dist: float) -> None:
        nonlocal selected, selected_parts
        if val is None or not math.isfinite(float(val)):
            return
        cand = {"step": int(step), "val_excess": float(val), "d_total": float(dist)}
        if _selection_better(cand, selected):
            selected = cand
            selected_parts = _copy_parts(snapshot_parts())

    def checkpoint(update_excess: float | None, update_objective: float | None, alpha: float | None) -> None:
        nonlocal train_eval_seconds, validation_seconds
        import time

        b_a, b_b, dist, rb, ro = current_matrices()
        t0 = time.perf_counter()
        train_excess = excess_on_pack(train_pack, b_a, b_b)
        train_eval_seconds += time.perf_counter() - t0
        val_excess = None
        if val_pack is not None:
            t1 = time.perf_counter()
            val_excess = excess_on_pack(val_pack, b_a, b_b)
            validation_seconds += time.perf_counter() - t1
        ev_res, dist_res = _residuals(parts, mode, rho) if family != "exp_s" else _exp_residuals(b_a, b_b, mode, rho)
        status["max_eig_residual"] = float(max(status["max_eig_residual"], ev_res))
        status["max_dist_residual"] = float(max(status["max_dist_residual"], dist_res))
        curves.append(
            {
                "step": int(updates_done),
                "full_train_excess": float(train_excess),
                "full_val_excess": val_excess,
                "d_total": float(dist),
                "update_excess": update_excess,
                "update_objective": update_objective,
                "alpha": alpha,
                "r_basis_total": float(rb),
                "r_off_total": float(ro),
            }
        )
        consider(updates_done, val_excess, dist)
        if on_checkpoint is not None and (updates_done == 0 or updates_done % 20 == 0 or updates_done == int(n_updates)):
            on_checkpoint(resume_state())

    def resume_state() -> dict[str, Any]:
        from prh_replication.metric_stability import _adam_blob

        state: dict[str, Any] = {
            "updates_done": int(updates_done),
            "rng": dump_rng(rng),
            "curves": curves,
            "selected": selected,
            "selected_parts": _copy_parts(selected_parts) if selected_parts is not None else None,
            "subset_indices": subset_indices if record_indices else None,
            "optim_seconds": optim_seconds,
            "validation_seconds": validation_seconds,
            "train_eval_seconds": train_eval_seconds,
            "status": status,
            "adam": [_adam_blob(opt, tensor) for tensor in opt_params],
        }
        if family == "exp_s":
            state["vechs"] = [t.detach().cpu().numpy().copy() for t in tensors]
        else:
            state["raw"] = raw
            state["parts"] = _copy_parts(parts)
        return state

    if resume is None:
        checkpoint(None, None, None)
    import time

    while updates_done < int(n_updates) and status["stopped"] == "completed":
        t0 = time.perf_counter()
        idx = draw_quarter_indices(n, rng, quarter_fraction)
        if record_indices:
            subset_indices.append(idx.astype(int).tolist())
        pack = make_pack(
            to_numpy64(features_a[idx]) - to_numpy64(basis_a["mu"]),
            to_numpy64(features_b[idx]) - to_numpy64(basis_b["mu"]),
            basis_a["U"],
            basis_b["U"],
        )
        pack_t = pack_to_torch(pack)
        opt.zero_grad(set_to_none=True)
        alpha = 1.0
        try:
            if family == "exp_s":
                update_excess, ok = _exp_backward(tensors, pack_t, mode, rho, q, q_b)
                if not ok:
                    status["nonfinite_objective"] += 1
                    status["stopped"] = "nonfinite_objective"
                    optim_seconds += time.perf_counter() - t0
                    break
            else:
                update_excess, ok = _factor_backward(factor_params, pack_t, family, mode, lambda_basis, lambda_off)
                if not ok:
                    status["nonfinite_grad"] += 1
                    status["stopped"] = "nonfinite_grad"
                    optim_seconds += time.perf_counter() - t0
                    break
        except RuntimeError:
            status["nonfinite_objective"] += 1
            status["stopped"] = "runtime"
            optim_seconds += time.perf_counter() - t0
            break
        torch.nn.utils.clip_grad_norm_(opt_params, float(grad_clip))
        if family == "exp_s":
            opt.step()
            with torch.no_grad():
                s_a, s_b = _exp_project(tensors, mode, rho, q)
                tensors[0].copy_(vech_from_s(s_a))
                if mode == "separate":
                    tensors[1].copy_(vech_from_s(s_b))
            b_a = b_matrix_exp(s_a)
            b_b = np.eye(q_b) if mode == "a_only" else (b_a if mode == "shared_pc" else b_matrix_exp(s_b))
            ev_res, dist_res = _exp_residuals(b_a, b_b, mode, rho)
            if ev_res > 1e-5 or dist_res > 1e-5:
                status["stopped"] = "constraint_violation"
                optim_seconds += time.perf_counter() - t0
                break
            rb, ro = 0.0, 0.0
        else:
            prev_raw = [(a.copy(), h.copy()) for a, h in raw]
            prev_parts = parts
            opt.step()
            proposed = _raw_from_params(factor_params, family, q)
            parts, raw, alpha, how = accept_step(prev_raw, proposed, prev_parts, family, mode, rho)
            if how == "backtrack" and alpha < 1.0:
                status["backtracks"] += 1
            if how == "backtrack" and alpha == 0.0:
                status["full_rejects"] += 1
            if how == "retracted":
                status["retractions"] += 1
            _copy_raw_into_params(factor_params, raw, family)
            b_a, b_b = _bs(parts, mode, q_b)
            rb, ro = _regulariser_totals(parts, mode)
        update_excess = excess_on_pack(pack, b_a, b_b)
        objective = float(update_excess) - float(lambda_basis) * float(rb) - float(lambda_off) * float(ro)
        optim_seconds += time.perf_counter() - t0
        updates_done += 1
        checkpoint(float(update_excess), float(objective), float(alpha))

    b_a, b_b, dist, rb, ro = current_matrices()
    return {
        "ok": status["stopped"] == "completed",
        "family": family,
        "mode": mode,
        "rho": float(rho),
        "lambda_basis": float(lambda_basis),
        "lambda_off": float(lambda_off),
        "updates_done": int(updates_done),
        "status": status,
        "curves": curves,
        "selected": selected,
        "selected_parts": _copy_parts(selected_parts) if selected_parts is not None else None,
        "final_parts": _copy_parts(snapshot_parts()),
        "optim_seconds": optim_seconds,
        "validation_seconds": validation_seconds,
        "train_eval_seconds": train_eval_seconds,
        "examples_processed": int(updates_done * int(round(n * float(quarter_fraction)))),
        "n_val_checks": int(len(curves)),
        "objective_note": (
            "Each update differentiates excess after centring inside a fresh training quarter, "
            "minus the configured basis and off-diagonal penalties. "
            "The subset gradient is not an unbiased full-gallery gradient. "
            "Checkpoint selection uses unpenalised full-validation excess."
        ),
        "final_distortion": float(dist),
        "final_r_basis": float(rb),
        "final_r_off": float(ro),
        "subset_indices": subset_indices if record_indices else None,
        "train_pack_n": int(train_pack["n"]),
    }


def _copy_parts(parts: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if parts is None:
        return None
    out = []
    for part in parts:
        out.append({key: (np.array(val, copy=True) if isinstance(val, np.ndarray) else val) for key, val in part.items()})
    return out


def _copy_raw_into_params(params: list[dict[str, torch.Tensor]], raw: list[tuple[np.ndarray, np.ndarray]], family: str) -> None:
    block = max(family_block(family), 1)
    with torch.no_grad():
        for row, (a, h) in zip(params, raw):
            if learns_basis(family):
                row["A"].copy_(torch.tensor(a, dtype=torch.float64))
            row["H"].copy_(torch.tensor(mask_symmetric(h, block), dtype=torch.float64))


def _residuals(parts: list[dict[str, Any]] | None, mode: str, rho: float) -> tuple[float, float]:
    if not parts:
        return 0.0, 0.0
    check = [parts[0]] if mode != "separate" else parts
    ev = max(eig_residual(p["eig"]) for p in check)
    dist = max(0.0, total_distortion(check, mode) - float(rho))
    return float(ev), float(dist)


def _exp_residuals(b_a, b_b, mode: str, rho: float) -> tuple[float, float]:
    da, ev_a = distortion_of_b(b_a)
    if mode == "a_only":
        dist = da
        ev = eig_residual(ev_a)
    else:
        db, ev_b = distortion_of_b(b_b)
        dist = 2.0 * da if mode == "shared_pc" else da + db
        ev = max(eig_residual(ev_a), eig_residual(ev_b))
    return float(ev), float(max(0.0, dist - float(rho)))


def _factor_backward(params, pack_t, family, mode, lambda_basis, lambda_off) -> tuple[float, bool]:
    a_list = []
    h_list = []
    leaves_a = []
    leaves_h = []
    q = int(params[0]["H"].shape[0])
    for row in params:
        if learns_basis(family):
            leaf_a = row["A"].detach().requires_grad_(True)
        else:
            leaf_a = torch.eye(q, dtype=torch.float64)
        leaf_h = row["H"].detach().requires_grad_(True)
        a_list.append(leaf_a)
        h_list.append(leaf_h)
        leaves_a.append(leaf_a if learns_basis(family) else None)
        leaves_h.append(leaf_h)
    loss, excess = factor_objective(a_list, h_list, pack_t, family=family, mode=mode, lambda_basis=lambda_basis, lambda_off=lambda_off)
    if not torch.isfinite(loss) or not loss.requires_grad:
        return float("nan"), False
    loss.backward()
    for row, leaf_a, leaf_h in zip(params, leaves_a, leaves_h):
        if leaf_a is not None:
            if leaf_a.grad is None or not torch.isfinite(leaf_a.grad).all():
                return float("nan"), False
            row["A"].grad = leaf_a.grad.detach()
        if leaf_h.grad is None or not torch.isfinite(leaf_h.grad).all():
            return float("nan"), False
        row["H"].grad = leaf_h.grad.detach()
    return float(excess.detach()), True


def _exp_backward(tensors, pack_t, mode, rho, q, q_b) -> tuple[float, bool]:
    from prh_replication.anisotropic_kernels import _unvech

    s_a = _unvech(tensors[0].detach(), q)
    s_b = _unvech(tensors[1].detach(), q) if mode == "separate" else torch.zeros(q_b, q_b, dtype=torch.float64)
    s_a, s_b = project_matrices(s_a, s_b, rho=float(rho), accounting="equal_total", mode=mode)
    var_a = s_a.detach().requires_grad_(True)
    if mode == "separate":
        var_b = s_b.detach().requires_grad_(True)
    elif mode == "shared_pc":
        var_b = var_a
    else:
        var_b = s_b.detach()
    excess = _excess_torch(var_a, var_b, [pack_t], mode)
    if not torch.isfinite(excess) or not excess.requires_grad:
        return float("nan"), False
    excess.backward()
    if var_a.grad is None or not torch.isfinite(var_a.grad).all():
        return float("nan"), False
    tensors[0].grad = -_grad_to_vech(var_a.grad)
    if mode == "separate":
        if var_b.grad is None or not torch.isfinite(var_b.grad).all():
            return float("nan"), False
        tensors[1].grad = -_grad_to_vech(var_b.grad)
    return float(excess.detach()), True
