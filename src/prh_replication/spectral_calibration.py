"""Spectral calibration of feature maps.

Mathematical arrays use examples as columns: Z has shape (q, n), and a map
acts as Y = T @ Z. Repository caches store examples as rows; ``rows_to_columns``
and ``apply_rows`` are the only orientation adapters.

A kernel metric M satisfies k(x, x') = x.T @ M @ x' and T.T @ T = M.
The canonical SPD factor is T = M^{1/2}. Applying M itself is not a feature map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import fractional_matrix_power, schur

from prh_replication.kernels import REL_CENTER


def alpha_grid(count: int = 21) -> tuple[float, ...]:
    """i/20 for i = 0..20 when count is 21. The grid is not data-dependent."""
    if int(count) < 2:
        raise ValueError("alpha grid needs at least two points")
    return tuple(i / (int(count) - 1) for i in range(int(count)))


def rows_to_columns(z_rows: np.ndarray) -> np.ndarray:
    z = np.asarray(z_rows, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("expected a 2d array of row examples")
    return np.ascontiguousarray(z.T)


def columns_to_rows(z_columns: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(z_columns, dtype=np.float64).T)


def apply_columns(transform: np.ndarray, z_columns: np.ndarray) -> np.ndarray:
    return np.asarray(transform, dtype=np.float64) @ np.asarray(z_columns, dtype=np.float64)


def apply_rows(transform: np.ndarray, z_rows: np.ndarray) -> np.ndarray:
    """Row-cache form of Y = T @ Z. Equals ``columns_to_rows(T @ rows_to_columns(Z))``."""
    transform = np.asarray(transform, dtype=np.float64)
    z_rows = np.asarray(z_rows, dtype=np.float64)
    return z_rows @ transform.T


def conjugate_signs(transform: np.ndarray, source_signs: np.ndarray, target_signs: np.ndarray) -> np.ndarray:
    """T' = S_y T S_x for diagonal sign matrices. S_x^{-1} = S_x."""
    source = np.asarray(source_signs, dtype=np.float64).reshape(-1)
    target = np.asarray(target_signs, dtype=np.float64).reshape(-1)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (target.size, source.size):
        raise ValueError("sign vectors do not match the transform")
    if not np.all(np.isin(source, (-1.0, 1.0))) or not np.all(np.isin(target, (-1.0, 1.0))):
        raise ValueError("sign conventions are ±1")
    return target[:, None] * transform * source[None, :]


def flip_columns(z_columns: np.ndarray, signs: np.ndarray) -> np.ndarray:
    signs = np.asarray(signs, dtype=np.float64).reshape(-1)
    z_columns = np.asarray(z_columns, dtype=np.float64)
    if signs.size != z_columns.shape[0]:
        raise ValueError("sign vector does not match the feature dimension")
    return signs[:, None] * z_columns


def training_covariance(z_columns: np.ndarray) -> np.ndarray:
    """C = Z Zᵀ / n for column examples. Does not centre again."""
    z = np.asarray(z_columns, dtype=np.float64)
    n = z.shape[1]
    if n < 2:
        raise ValueError("covariance needs at least two examples")
    return (z @ z.T) / float(n)


def covariance_diagnostics(z_columns: np.ndarray) -> dict[str, Any]:
    cov = training_covariance(z_columns)
    diagonal = np.diag(cov).copy()
    off = cov - np.diag(diagonal)
    scale = float(np.linalg.norm(cov))
    gaps = diagonal[:-1] - diagonal[1:] if diagonal.size > 1 else np.zeros(0)
    return {
        "variances": diagonal,
        "offdiag_rel": float(np.linalg.norm(off) / scale) if scale > 0 else 0.0,
        "decreasing": bool(diagonal.size <= 1 or np.all(gaps >= -1e-8 * max(float(diagonal[0]), 1.0))),
        "column_mean_norm": float(np.linalg.norm(np.asarray(z_columns, dtype=np.float64).mean(axis=1))),
    }


def covariance_map(var_x: np.ndarray, var_y: np.ndarray, alpha: float, *, var_tol: float = 1e-12) -> np.ndarray:
    """Diagonal T_α = diag((λ_Y / λ_X)^{α/2}). Marginal variances only."""
    vx = np.asarray(var_x, dtype=np.float64).reshape(-1)
    vy = np.asarray(var_y, dtype=np.float64).reshape(-1)
    if vx.shape != vy.shape:
        raise ValueError("covariance map needs equal dimensions")
    if np.any(vx <= var_tol) or np.any(vy <= var_tol):
        raise ValueError("covariance map requires positive training variances")
    return np.diag((vy / vx) ** (float(alpha) / 2.0))


def ridge_map(z_x: np.ndarray, z_y: np.ndarray, eta_coef: float = 1e-3) -> dict[str, Any]:
    """Minimise (1/n) ||T Z_X - Z_Y||_F^2 + η ||T||_F^2 with η = eta_coef * tr(C_X) / q.

    Closed form T = (Z_Y Z_Xᵀ / n) (C_X + η I)^{-1}. Training columns only.
    """
    z_x = np.asarray(z_x, dtype=np.float64)
    z_y = np.asarray(z_y, dtype=np.float64)
    if z_x.ndim != 2 or z_y.ndim != 2 or z_x.shape[1] != z_y.shape[1]:
        raise ValueError("ridge expects paired column examples")
    q = int(z_x.shape[0])
    if int(z_y.shape[0]) != q:
        raise ValueError("ridge map is square in the matched PCA dimension")
    n = float(z_x.shape[1])
    cov_x = (z_x @ z_x.T) / n
    eta = float(eta_coef) * float(np.trace(cov_x)) / float(q)
    cross = (z_y @ z_x.T) / n
    gram = cov_x + eta * np.eye(q, dtype=np.float64)
    transform = np.linalg.solve(gram, cross.T).T
    residual = transform @ z_x - z_y
    objective = float(np.sum(residual * residual) / n + eta * np.sum(transform * transform))
    return {
        "T": transform,
        "eta": eta,
        "eta_coef": float(eta_coef),
        "objective": objective,
        "objective_convention": "(1/n) ||T Z_X - Z_Y||_F^2 + eta ||T||_F^2",
        "orientation": "examples_as_columns",
        "q": q,
        "n": int(n),
    }


def ridge_map_rows(z_x_rows: np.ndarray, z_y_rows: np.ndarray, eta_coef: float = 1e-3) -> dict[str, Any]:
    fitted = ridge_map(rows_to_columns(z_x_rows), rows_to_columns(z_y_rows), eta_coef=eta_coef)
    return fitted


def select_exponent(alphas: np.ndarray, scores: np.ndarray, tol: float) -> int | None:
    """Smallest index whose score is within ``tol`` of the best finite score."""
    alphas = np.asarray(alphas, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if alphas.shape != scores.shape:
        raise ValueError("alphas and scores must match")
    finite = np.isfinite(scores)
    if not np.any(finite):
        return None
    best = float(np.max(scores[finite]))
    candidates = np.flatnonzero(finite & (scores >= best - float(tol)))
    return int(candidates[0])


def _sym(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def numerical_support(singular_values: np.ndarray, rank_tol: float) -> np.ndarray:
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size == 0:
        return np.zeros(0, dtype=bool)
    scale = float(singular_values[0])
    thresh = float(rank_tol) * scale if scale > 0 else float(rank_tol)
    return singular_values > thresh


def svd_flatten(transform: np.ndarray, alpha: float, rank_tol: float = 1e-8) -> dict[str, Any]:
    """T = L Σ Rᵀ becomes L Σ^α Rᵀ. Zero singular values stay zero."""
    transform = np.asarray(transform, dtype=np.float64)
    left, singular, vh = np.linalg.svd(transform, full_matrices=False)
    keep = numerical_support(singular, rank_tol)
    powered = np.zeros_like(singular)
    if np.any(keep):
        powered[keep] = singular[keep] ** float(alpha)
    flattened = (left * powered) @ vh
    rank = int(np.count_nonzero(keep))
    return {
        "T": flattened,
        "singular_values": powered,
        "source_singular_values": singular,
        "numerical_rank": rank,
        "partial_isometry": bool(rank < singular.size),
        "sv_min": float(powered[keep].min()) if rank else 0.0,
        "sv_max": float(powered.max()) if powered.size else 0.0,
    }


def induced_metric(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return _sym(transform.T @ transform)


def spd_power(matrix: np.ndarray, alpha: float, *, eig_tol: float = 1e-12) -> np.ndarray:
    """Real power of a symmetric positive semidefinite matrix.

    Eigenvalues below ``-eig_tol`` raise. They are not replaced by absolute values.
    Exact numerical zeros stay zero, including at α = 0.
    """
    sym = _sym(matrix)
    values, vectors = np.linalg.eigh(sym)
    if float(values.min()) < -float(eig_tol):
        raise ValueError("spd_power refuses negative eigenvalues")
    powered = np.zeros_like(values)
    positive = values > float(eig_tol)
    powered[positive] = values[positive] ** float(alpha)
    return _sym((vectors * powered) @ vectors.T)


def spd_sqrt(matrix: np.ndarray) -> np.ndarray:
    return spd_power(matrix, 0.5)


def explicit_factor(basis: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """F = D^{1/2} A^{-1}. Then Fᵀ F = A^{-⊤} D A^{-1}, not A D A^{-1}."""
    weight_sqrt = spd_sqrt(weight)
    solved = np.linalg.solve(np.asarray(basis, dtype=np.float64), np.eye(basis.shape[0], dtype=np.float64))
    return weight_sqrt @ solved


def distortion(transform: np.ndarray, *, eig_tol: float = 1e-12) -> float:
    """‖log(Tᵀ T)‖_F² / q for an invertible square feature transform. Otherwise NaN."""
    metric = induced_metric(transform)
    values = np.linalg.eigvalsh(metric)
    q = int(values.size)
    if q == 0 or float(values.min()) <= float(eig_tol):
        return float("nan")
    logs = np.log(values)
    return float(np.sum(logs * logs) / q)


def residual_distortion(metric_b: np.ndarray) -> float:
    """‖log B‖_F² / q. Residual identity eigenvalues add zero and are not in this q."""
    values = np.linalg.eigvalsh(_sym(metric_b))
    if float(values.min()) <= 0.0:
        return float("nan")
    logs = np.log(values)
    return float(np.sum(logs * logs) / values.size)


def condition_number(sv_min: float, sv_max: float, rank: int, dimension: int) -> float | None:
    if rank != dimension or sv_min <= 0.0 or not math.isfinite(sv_min):
        return None
    return float(sv_max / sv_min)


def _branch_reasons(eigenvalues: np.ndarray, zero_tol: float, branch_tol: float) -> list[str]:
    reasons = []
    for lam in np.asarray(eigenvalues):
        magnitude = abs(complex(lam))
        if magnitude <= float(zero_tol):
            reasons.append("near_zero_eigenvalue")
        elif float(np.real(lam)) < 0.0 and abs(float(np.imag(lam))) <= float(branch_tol) * max(1.0, magnitude):
            reasons.append("negative_real_eigenvalue")
    return reasons


def spectrum_report(transform: np.ndarray) -> dict[str, Any]:
    """Schur eigenvalues for the branch cut, plus an explicit eigenvector diagnostic."""
    transform = np.asarray(transform, dtype=np.float64)
    triangular, _vectors = schur(transform, output="complex")
    eigenvalues = np.diag(triangular)
    report: dict[str, Any] = {
        "eigenvalues_real": np.real(eigenvalues).astype(np.float64),
        "eigenvalues_imag": np.imag(eigenvalues).astype(np.float64),
        "reconstruction_error": None,
        "eigenvector_cond": None,
        "diagonalizable_residual": None,
    }
    try:
        values, vectors = np.linalg.eig(transform)
        reconstructed = vectors @ np.diag(values) @ np.linalg.inv(vectors)
        scale = max(float(np.linalg.norm(transform)), 1.0)
        report["reconstruction_error"] = float(np.linalg.norm(reconstructed - transform) / scale)
        report["eigenvector_cond"] = float(np.linalg.cond(vectors))
        report["diagonalizable_residual"] = report["reconstruction_error"]
    except np.linalg.LinAlgError:
        report["reconstruction_error"] = None
        report["eigenvector_cond"] = None
    return report


@dataclass
class PowerResult:
    matrix: np.ndarray | None
    eligible: bool
    reason: str
    path: str
    partial_isometry: bool
    numerical_rank: int
    sv_min: float
    sv_max: float
    cond: float | None
    distortion: float | None
    imag_residual: float | None
    reconstruction_error: float | None
    eigenvector_cond: float | None

    def induced(self) -> np.ndarray | None:
        if self.matrix is None:
            return None
        return induced_metric(self.matrix)

    def geometry(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "path": self.path,
            "partial_isometry": self.partial_isometry,
            "numerical_rank": self.numerical_rank,
            "sv_min": self.sv_min,
            "sv_max": self.sv_max,
            "cond": self.cond,
            "distortion": self.distortion,
            "imag_residual": self.imag_residual,
            "reconstruction_error": self.reconstruction_error,
            "eigenvector_cond": self.eigenvector_cond,
        }


def _geometry_from_matrix(matrix: np.ndarray, *, path: str, reason: str, rank_tol: float, partial: bool, rank: int, imag_residual: float | None, reconstruction_error: float | None, eigenvector_cond: float | None) -> PowerResult:
    _left, singular, _vh = np.linalg.svd(matrix, full_matrices=False)
    keep = numerical_support(singular, rank_tol)
    used_rank = int(np.count_nonzero(keep))
    sv_min = float(singular[keep].min()) if used_rank else 0.0
    sv_max = float(singular[0]) if singular.size else 0.0
    return PowerResult(
        matrix=matrix,
        eligible=True,
        reason=reason,
        path=path,
        partial_isometry=bool(partial or used_rank < matrix.shape[0]),
        numerical_rank=used_rank if path == "svd" else rank,
        sv_min=sv_min,
        sv_max=sv_max,
        cond=condition_number(sv_min, sv_max, used_rank, int(matrix.shape[0])),
        distortion=distortion(matrix),
        imag_residual=imag_residual,
        reconstruction_error=reconstruction_error,
        eigenvector_cond=eigenvector_cond,
    )


class FeatureFamily:
    """Singular-value flattening and literal eigenvalue powers of one frozen square map."""

    def __init__(self, transform: np.ndarray, *, rank_tol: float = 1e-8, imag_tol: float = 1e-8, recon_tol: float = 1e-6, eig_cond_max: float = 1e8, branch_tol: float = 1e-8, zero_tol: float = 1e-10, sym_tol: float = 1e-8):
        self.transform = np.asarray(transform, dtype=np.float64)
        if self.transform.ndim != 2 or self.transform.shape[0] != self.transform.shape[1]:
            raise ValueError("spectral families are square")
        self.rank_tol = float(rank_tol)
        self.imag_tol = float(imag_tol)
        self.recon_tol = float(recon_tol)
        self.eig_cond_max = float(eig_cond_max)
        self.branch_tol = float(branch_tol)
        self.zero_tol = float(zero_tol)
        self.sym_tol = float(sym_tol)
        self.q = int(self.transform.shape[0])
        scale = max(float(np.linalg.norm(self.transform)), 1.0)
        self.symmetric = float(np.linalg.norm(self.transform - self.transform.T)) <= self.sym_tol * scale
        self.spectrum = spectrum_report(self.transform)
        eigenvalues = self.spectrum["eigenvalues_real"] + 1j * self.spectrum["eigenvalues_imag"]
        self.branch_hits = _branch_reasons(eigenvalues, self.zero_tol, self.branch_tol)
        imag_part = np.max(np.abs(self.spectrum["eigenvalues_imag"])) if self.q else 0.0
        real_positive = (not self.branch_hits) and float(imag_part) <= self.imag_tol and bool(np.all(self.spectrum["eigenvalues_real"] > self.zero_tol))
        self.spd = bool(self.symmetric and real_positive)
        if self.branch_hits:
            self.eig_reason = self.branch_hits[0]
            self.eig_eligible = False
        elif self.spd:
            self.eig_reason = "spd_eigh"
            self.eig_eligible = True
        else:
            self.eig_reason = "principal_fractional"
            self.eig_eligible = True
        self._eigh: tuple[np.ndarray, np.ndarray] | None = None
        if self.spd:
            values, vectors = np.linalg.eigh(_sym(self.transform))
            self._eigh = (values, vectors)

    def svd_at(self, alpha: float) -> PowerResult:
        flattened = svd_flatten(self.transform, alpha, self.rank_tol)
        return _geometry_from_matrix(
            flattened["T"],
            path="svd",
            reason="partial_isometry" if flattened["partial_isometry"] and abs(float(alpha)) == 0.0 else "svd",
            rank_tol=self.rank_tol,
            partial=flattened["partial_isometry"],
            rank=int(flattened["numerical_rank"]),
            imag_residual=0.0,
            reconstruction_error=float(np.linalg.norm(flattened["T"] - self.transform) / max(np.linalg.norm(self.transform), 1.0)) if abs(float(alpha) - 1.0) == 0.0 else None,
            eigenvector_cond=None,
        )

    def eigenvalue_at(self, alpha: float) -> PowerResult:
        if abs(float(alpha) - 1.0) == 0.0:
            return _geometry_from_matrix(
                self.transform.copy(),
                path="eigenvalue",
                reason="alpha_1_endpoint",
                rank_tol=self.rank_tol,
                partial=False,
                rank=self.q,
                imag_residual=0.0,
                reconstruction_error=0.0,
                eigenvector_cond=self.spectrum["eigenvector_cond"],
            )
        if not self.eig_eligible:
            return PowerResult(
                None,
                False,
                self.eig_reason,
                "eigenvalue",
                False,
                0,
                float("nan"),
                float("nan"),
                None,
                None,
                None,
                self.spectrum["reconstruction_error"],
                self.spectrum["eigenvector_cond"],
            )
        if self.spd and self._eigh is not None:
            values, vectors = self._eigh
            powered = np.zeros_like(values)
            positive = values > self.zero_tol
            powered[positive] = values[positive] ** float(alpha)
            matrix = _sym((vectors * powered) @ vectors.T)
            return _geometry_from_matrix(
                matrix,
                path="eigenvalue",
                reason="spd_eigh",
                rank_tol=self.rank_tol,
                partial=bool(np.count_nonzero(positive) < self.q),
                rank=int(np.count_nonzero(positive)),
                imag_residual=0.0,
                reconstruction_error=self.spectrum["reconstruction_error"],
                eigenvector_cond=self.spectrum["eigenvector_cond"],
            )
        raw = fractional_matrix_power(self.transform, float(alpha))
        imag_residual = float(np.max(np.abs(np.imag(raw)))) if np.iscomplexobj(raw) else 0.0
        if imag_residual > self.imag_tol or not np.isfinite(raw).all():
            return PowerResult(
                None,
                False,
                "imaginary_residual",
                "eigenvalue",
                False,
                0,
                float("nan"),
                float("nan"),
                None,
                None,
                imag_residual,
                self.spectrum["reconstruction_error"],
                self.spectrum["eigenvector_cond"],
            )
        matrix = np.real(raw).astype(np.float64)
        return _geometry_from_matrix(
            matrix,
            path="eigenvalue",
            reason=self.eig_reason,
            rank_tol=self.rank_tol,
            partial=False,
            rank=self.q,
            imag_residual=imag_residual,
            reconstruction_error=self.spectrum["reconstruction_error"],
            eigenvector_cond=self.spectrum["eigenvector_cond"],
        )


def gram_scores(k: np.ndarray, l: np.ndarray) -> dict[str, float]:
    """Float64 centred-Gram cosine. Matches ``extension_stats`` without official epsilon."""
    k = np.asarray(k, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    if k.shape != l.shape or k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("grams must be paired and square")

    def _center(gram: np.ndarray) -> np.ndarray:
        return gram - gram.mean(axis=1, keepdims=True) - gram.mean(axis=0, keepdims=True) + gram.mean()

    kc, lc = _center(k), _center(l)
    n = k.shape[0]
    ip = float(np.sum(kc * lc))
    nk = float(np.linalg.norm(kc))
    nl = float(np.linalg.norm(lc))
    trk = float(np.trace(kc))
    trl = float(np.trace(lc))
    kn = float(np.linalg.norm(k))
    ln = float(np.linalg.norm(l))
    degenerate = (not math.isfinite(nk)) or (not math.isfinite(nl)) or nk <= REL_CENTER * (kn + 1e-12) or nl <= REL_CENTER * (ln + 1e-12)
    finite = all(math.isfinite(v) for v in (ip, nk, nl, trk, trl))
    valid = finite and (not degenerate) and nk > 0.0 and nl > 0.0
    a = ip / (nk * nl) if valid else float("nan")
    b = (trk * trl) / ((n - 1.0) * nk * nl) if valid else float("nan")
    den = trk * trl
    ratio_undefined = (not valid) or abs(den) <= REL_CENTER * n * nk * nl
    ratio = ((n - 1.0) * ip / den) if not ratio_undefined else float("nan")
    excess = (a - b) if valid and math.isfinite(b) else float("nan")
    return {
        "a": a,
        "b": b,
        "ratio": ratio,
        "excess": excess,
        "r_eff_k": (trk * trk / (nk * nk)) if nk > 0 else float("nan"),
        "r_eff_l": (trl * trl / (nl * nl)) if nl > 0 else float("nan"),
        "valid_a": float(valid),
    }


def feature_scores_columns(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, float]:
    """Linear CKA of column examples. Gram centring is the CKA centring."""
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    return gram_scores(y_a.T @ y_a, y_b.T @ y_b)


def feature_scores_rows(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, float]:
    return feature_scores_columns(rows_to_columns(y_a), rows_to_columns(y_b))


def scores_from_deltas_numpy(delta_a: np.ndarray, delta_b: np.ndarray, pack: dict[str, Any]) -> dict[str, float]:
    """Numpy form of the contracted full-space score. Residual identity is the zero delta."""
    delta_a = np.asarray(delta_a, dtype=np.float64)
    delta_b = np.asarray(delta_b, dtype=np.float64)
    ip = float(pack["ip_lin"]) + float(np.trace(delta_a @ pack["l_on_a"])) + float(np.trace(delta_b @ pack["k_on_b"])) + float(np.trace(delta_a @ pack["gab"] @ delta_b @ pack["gab"].T))
    nk2 = float(pack["nk2_lin"]) + 2.0 * float(np.trace(delta_a @ pack["ca"])) + float(np.trace(delta_a @ pack["gaa"] @ delta_a @ pack["gaa"]))
    nl2 = float(pack["nl2_lin"]) + 2.0 * float(np.trace(delta_b @ pack["cb"])) + float(np.trace(delta_b @ pack["gbb"] @ delta_b @ pack["gbb"]))
    trk = float(pack["trk_lin"]) + float(np.trace(delta_a @ pack["gaa"]))
    trl = float(pack["trl_lin"]) + float(np.trace(delta_b @ pack["gbb"]))
    n = float(pack["n"])
    nk = math.sqrt(max(nk2, 0.0))
    nl = math.sqrt(max(nl2, 0.0))
    finite = all(math.isfinite(v) for v in (ip, nk, nl, trk, trl))
    valid = finite and nk > 0.0 and nl > 0.0
    a = ip / (nk * nl) if valid else float("nan")
    b = (trk * trl) / ((n - 1.0) * nk * nl) if valid else float("nan")
    den = trk * trl
    ratio_undefined = (not valid) or abs(den) <= REL_CENTER * n * nk * nl
    ratio = ((n - 1.0) * ip / den) if not ratio_undefined else float("nan")
    excess = (a - b) if valid and math.isfinite(b) else float("nan")
    return {
        "a": a,
        "b": b,
        "ratio": ratio,
        "excess": excess,
        "r_eff_k": (trk * trk / nk2) if nk2 > 0 else float("nan"),
        "r_eff_l": (trl * trl / nl2) if nl2 > 0 else float("nan"),
        "valid_a": float(valid),
    }


def sq_euclidean_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    gram = features @ features.T
    squared = np.sum(features * features, axis=1)
    return squared[:, None] + squared[None, :] - 2.0 * gram


def knn_indices(distances: np.ndarray, k: int) -> np.ndarray:
    work = np.array(distances, copy=True, dtype=np.float64)
    np.fill_diagonal(work, np.inf)
    k = int(min(k, work.shape[0] - 1))
    part = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    row = np.arange(work.shape[0])[:, None]
    order = np.argsort(work[row, part], axis=1)
    return part[row, order]


def mutual_knn(knn_a: np.ndarray, knn_b: np.ndarray) -> float:
    left = np.sort(knn_a, axis=1)
    right = np.sort(knn_b, axis=1)
    hits = left[:, :, None] == right[:, None, :]
    k = left.shape[1]
    return float(hits.any(axis=2).sum() / (left.shape[0] * k))


def mnn_rows(y_a: np.ndarray, y_b: np.ndarray, k: int = 10) -> float:
    return mutual_knn(knn_indices(sq_euclidean_rows(y_a), k), knn_indices(sq_euclidean_rows(y_b), k))


def mnn_columns(y_a: np.ndarray, y_b: np.ndarray, k: int = 10) -> float:
    return mnn_rows(columns_to_rows(y_a), columns_to_rows(y_b), k)


def subspace_delta_sq(scores: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Extra squared distance from a subspace metric delta = B − I. ``scores`` is (n, q)."""
    mid = scores @ np.asarray(delta, dtype=np.float64) @ scores.T
    diag = np.diag(mid)
    return diag[:, None] + diag[None, :] - 2.0 * mid


def apply_subspace_factor_rows(features: np.ndarray, basis: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Row form of y = z + U (F − I) Uᵀ z. Residual components orthogonal to U are unchanged."""
    features = np.asarray(features, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    coefficients = features @ basis
    return features + coefficients @ (factor - np.eye(factor.shape[0], dtype=np.float64)).T @ basis.T


def run_synthetic_checks(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Float64 fixture checks. No gallery and no exploratory-test array."""
    cfg = cfg or {}
    rank_tol = float(cfg.get("rank_tol", 1e-8))
    imag_tol = float(cfg.get("imag_tol", 1e-8))
    tie_tol = float(cfg.get("tie_tol", 1e-8))
    rng = np.random.default_rng(20261006)
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: dict[str, Any] | None = None) -> None:
        checks.append({"name": name, "ok": bool(ok), **(detail or {})})
        if not ok:
            raise AssertionError(name)

    q, n = 6, 40
    z_x = rng.normal(size=(q, n))
    z_y = rng.normal(size=(q, n))
    gauss = rng.normal(size=(q, q))
    orthogonal, _r = np.linalg.qr(gauss)
    orthogonal *= np.sign(np.diag(_r))
    base = feature_scores_columns(z_x, z_y)
    rotated = feature_scores_columns(orthogonal @ z_x, z_y)
    record("orthogonal_cka", abs(base["a"] - rotated["a"]) < 1e-8 and abs(base["excess"] - rotated["excess"]) < 1e-8)
    mnn_base = mnn_columns(z_x, z_y, k=5)
    mnn_rot = mnn_columns(orthogonal @ z_x, z_y, k=5)
    dist = sq_euclidean_rows(columns_to_rows(z_x))
    dist_rot = sq_euclidean_rows(columns_to_rows(orthogonal @ z_x))
    record("orthogonal_mnn", abs(mnn_base - mnn_rot) < 1e-8 and np.allclose(dist, dist_rot, atol=1e-8))

    lam_x = np.array([4.0, 2.0, 1.0, 0.5])
    lam_y = np.array([1.0, 1.0, 3.0, 0.25])
    hx, _ = np.linalg.qr(rng.normal(size=(80, 4)))
    hy, _ = np.linalg.qr(rng.normal(size=(80, 4)))
    zx = (np.diag(np.sqrt(lam_x * 80.0)) @ hx.T)
    zy = (np.diag(np.sqrt(lam_y * 80.0)) @ hy.T)
    fitted = covariance_map(lam_x, lam_y, 1.0)
    matched = training_covariance(fitted @ zx)
    record("covariance_match_alpha1", np.allclose(matched, np.diag(lam_y), atol=1e-8))
    perm = rng.permutation(zy.shape[1])
    again = covariance_map(np.diag(training_covariance(zx)), np.diag(training_covariance(zy[:, perm])), 1.0)
    record("covariance_permutation", np.allclose(again, fitted, atol=1e-8))
    ridge_true = ridge_map(zx, zy, eta_coef=1e-3)
    ridge_perm = ridge_map(zx, zy[:, perm], eta_coef=1e-3)
    record("ridge_uses_correspondence", float(np.linalg.norm(ridge_true["T"] - ridge_perm["T"])) > 1e-3)

    eye = np.eye(q)
    family_eye = FeatureFamily(eye, rank_tol=rank_tol)
    record("alpha0_spd_is_identity", np.allclose(family_eye.svd_at(0.0).matrix, eye) and np.allclose(family_eye.eigenvalue_at(0.0).matrix, eye))
    dense = rng.normal(size=(q, q))
    dense = dense @ dense.T + 0.3 * eye
    family_spd = FeatureFamily(dense, rank_tol=rank_tol)
    record("alpha1_svd", np.allclose(family_spd.svd_at(1.0).matrix, dense, atol=1e-8))
    record("alpha1_eig", np.allclose(family_spd.eigenvalue_at(1.0).matrix, dense, atol=1e-8))
    alphas = alpha_grid()
    spd_gap = max(float(np.linalg.norm(family_spd.svd_at(a).matrix - family_spd.eigenvalue_at(a).matrix)) for a in alphas)
    record("spd_svd_eig_agree", spd_gap < 1e-7, {"gap": spd_gap})
    distort_gap = max(abs(family_spd.svd_at(a).distortion - (a ** 2) * family_spd.svd_at(1.0).distortion) for a in alphas)
    record("svd_distortion_quadratic", distort_gap < 1e-6, {"gap": distort_gap})

    raw = rng.normal(size=(q, q))
    raw_family = FeatureFamily(raw + 0.4 * eye, rank_tol=rank_tol)
    cka_0 = feature_scores_columns(raw_family.svd_at(0.0).matrix @ z_x, z_y)["a"]
    cka_id = feature_scores_columns(z_x, z_y)["a"]
    record("full_rank_svd_alpha0_cka", abs(cka_0 - cka_id) < 1e-7)
    polar = raw_family.svd_at(0.0).matrix
    record("full_rank_svd_alpha0_orthogonal", np.allclose(polar.T @ polar, np.eye(q), atol=1e-7))

    low = rng.normal(size=(q, 2))
    low_map = low @ low.T
    low_family = FeatureFamily(low_map, rank_tol=rank_tol)
    low_0 = low_family.svd_at(0.0)
    record("rank_deficient_partial_isometry", low_0.partial_isometry and low_0.numerical_rank == 2 and not np.allclose(low_0.matrix, np.eye(q)))

    signed = np.diag(np.array([1.5, -0.4, 0.8, 0.2, 1.1, 0.7]))
    signed_family = FeatureFamily(signed, rank_tol=rank_tol)
    refused = signed_family.eigenvalue_at(0.5)
    record("negative_eigenvalue_unsupported", (not refused.eligible) and refused.reason == "negative_real_eigenvalue" and refused.matrix is None)
    record("negative_not_relabelled_svd", refused.path == "eigenvalue")

    scale, theta = 1.4, 0.6
    block = scale * np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    complex_family = FeatureFamily(block, rank_tol=rank_tol, imag_tol=imag_tol)
    powered = complex_family.eigenvalue_at(0.5)
    record(
        "complex_pair_real_power",
        powered.eligible and powered.imag_residual is not None and powered.imag_residual <= imag_tol and np.allclose(complex_family.eigenvalue_at(1.0).matrix, block, atol=1e-8),
        {"imag": powered.imag_residual, "reason": powered.reason},
    )

    basis = rng.normal(size=(q, q))
    basis = basis / np.linalg.norm(basis, axis=0)
    weight = np.diag(np.exp(rng.normal(size=q)))
    factor = explicit_factor(basis, weight)
    solved = np.linalg.solve(basis, np.eye(q))
    metric_formula = solved.T @ weight @ solved
    metric = induced_metric(factor)
    canonical = spd_sqrt(metric)
    record("explicit_factor_metric", np.allclose(metric, metric_formula, atol=1e-8))
    record("canonical_sqrt_metric", np.allclose(canonical.T @ canonical, metric, atol=1e-8))
    factor_family = FeatureFamily(factor, rank_tol=rank_tol)
    canonical_family = FeatureFamily(canonical, rank_tol=rank_tol)
    svd_gap = max(float(np.linalg.norm(induced_metric(factor_family.svd_at(a).matrix) - induced_metric(canonical_family.svd_at(a).matrix))) for a in (0.0, 0.35, 0.7, 1.0))
    record("canonical_explicit_svd_agree", svd_gap < 1e-6, {"gap": svd_gap})
    upper = np.array([[2.0, 20.0], [0.0, 1.0]])
    upper_metric = induced_metric(upper)
    upper_canonical = spd_sqrt(upper_metric)
    upper_family = FeatureFamily(upper, rank_tol=rank_tol)
    upper_canonical_family = FeatureFamily(upper_canonical, rank_tol=rank_tol)
    eig_mats = []
    for exponent in (0.35, 0.7):
        left = upper_family.eigenvalue_at(exponent)
        right = upper_canonical_family.eigenvalue_at(exponent)
        record(f"triangular_eig_eligible_{exponent}", left.eligible and right.eligible and left.matrix is not None and right.matrix is not None)
        eig_mats.append(float(np.linalg.norm(induced_metric(left.matrix) - induced_metric(right.matrix))))
    eig_gap = max(eig_mats)
    record("explicit_eig_not_forced_equal", eig_gap > 1e-3, {"gap": eig_gap})

    triangle = np.array([[1.0, 100.0], [0.0, 1.0]])
    tri_family = FeatureFamily(triangle, rank_tol=rank_tol)
    tri_eig = tri_family.eigenvalue_at(0.5)
    tri_svd = tri_family.svd_at(0.5)
    record("nonnormal_eig_supported", tri_eig.eligible and tri_eig.matrix is not None and tri_eig.path == "eigenvalue")
    record("nonnormal_eig_not_svd", tri_eig.matrix is not None and not np.allclose(tri_eig.matrix, tri_svd.matrix, atol=1e-3))

    scores_tie = np.array([0.2, 0.5, 0.5 + 1e-12, 0.1])
    record("tie_smaller_exponent", select_exponent(np.arange(4), scores_tie, tie_tol) == 1)
    record("alpha_grid", alpha_grid()[0] == 0.0 and alpha_grid()[-1] == 1.0 and len(alpha_grid()) == 21)

    # Row adapter and direct-versus-metric CKA.
    z_rows = columns_to_rows(z_x)
    applied = apply_rows(dense, z_rows)
    record("row_adapter", np.allclose(applied, columns_to_rows(dense @ z_x)))
    metric_b = canonical.T @ canonical
    direct = feature_scores_rows(apply_rows(canonical, z_rows), columns_to_rows(z_y))
    gram = gram_scores(z_rows @ metric_b @ z_rows.T, columns_to_rows(z_y) @ columns_to_rows(z_y).T)
    record("feature_cka_matches_metric_gram", abs(direct["a"] - gram["a"]) < 1e-8)

    # Residual lift: ambient d > q, alpha identity residual.
    d = 10
    ambient = rng.normal(size=(n, d))
    u, _s, _vt = np.linalg.svd(ambient - ambient.mean(axis=0), full_matrices=False)
    basis_u = _vt[:3].T
    draw = rng.normal(size=(3, 3))
    factor_q = spd_sqrt(draw @ draw.T + np.eye(3))
    lifted = apply_subspace_factor_rows(ambient, basis_u, factor_q)
    identity_lift = apply_subspace_factor_rows(ambient, basis_u, np.eye(3))
    record("residual_identity_endpoint", np.allclose(identity_lift, ambient))
    # y^T y metric equals z^T M z on the subspace.
    coeff = (ambient - 0) @ basis_u
    metric_q = factor_q.T @ factor_q
    gram_lift = lifted @ lifted.T
    gram_formula = ambient @ ambient.T + coeff @ (metric_q - np.eye(3)) @ coeff.T
    record("lift_matches_residual_metric", np.allclose(gram_lift, gram_formula, atol=1e-8))

    from prh_replication.anisotropic_kernels import make_pack, scores_numpy

    pack = make_pack(ambient, ambient, basis_u, basis_u)
    contracted = scores_from_deltas_numpy(metric_q - np.eye(3), np.zeros((3, 3)), pack)
    reference = scores_numpy(metric_q - np.eye(3), np.zeros((3, 3)), pack)
    record("contracted_numpy_matches_torch", abs(contracted["a"] - reference["a"]) < 1e-8 and abs(contracted["excess"] - reference["excess"]) < 1e-8)
    zero = scores_from_deltas_numpy(np.zeros((3, 3)), np.zeros((3, 3)), pack)
    linear = feature_scores_rows(ambient, ambient)
    record("zero_delta_is_linear_cka", abs(zero["a"] - linear["a"]) < 1e-7)

    ridge_rows = ridge_map_rows(columns_to_rows(zx), columns_to_rows(zy), eta_coef=1e-3)
    record("ridge_row_translation", np.allclose(ridge_rows["T"], ridge_map(zx, zy, eta_coef=1e-3)["T"]))

    failed = [row["name"] for row in checks if not row["ok"]]
    return {"ok": not failed, "checks": checks, "n": len(checks)}
