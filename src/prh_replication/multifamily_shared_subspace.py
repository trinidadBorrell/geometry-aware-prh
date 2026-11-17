"""Pairwise CCA consensus and MAXVAR for a shared cross-family subspace.

PCA whitening is fit once per model. Pairwise CCA is the SVD of the
cross-covariance in those fixed whitened coordinates, so every edge that
touches a model uses the same transform.
"""

from __future__ import annotations

import numpy as np

from prh_replication.cca_steering_transfer import sym_inv_sqrt
from prh_replication.sae_crosscov import fit_pca


def fit_pca_whitener(features: np.ndarray, q: int, eta: float) -> dict:
    pca = fit_pca(features, q=int(q))
    components = np.asarray(pca["components"], dtype=np.float64)
    mean = np.asarray(pca["mean"], dtype=np.float64)
    scores = (np.asarray(features, dtype=np.float64) - mean) @ components
    cov = (scores.T @ scores) / max(scores.shape[0] - 1, 1)
    width = int(components.shape[1])
    tau = float(eta) * float(np.trace(cov)) / max(width, 1)
    whitener, keep = sym_inv_sqrt(cov + tau * np.eye(width))
    return {
        "mean": mean,
        "components": components,
        "whitener": np.asarray(whitener, dtype=np.float64),
        "tau": tau,
        "width": width,
        "variance_fraction": float(pca["variance_fraction"]),
        "support": int(np.sum(keep)),
    }


def whiten_activations(features: np.ndarray, pack: dict, *, displacement: bool = False) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if displacement:
        scores = values @ pack["components"]
    else:
        scores = (values - pack["mean"]) @ pack["components"]
    return scores @ pack["whitener"]


def _column_signs(matrix: np.ndarray) -> np.ndarray:
    signs = np.ones(matrix.shape[1], dtype=np.float64)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0.0:
            signs[j] = -1.0
    return signs


def cca_directions(left: np.ndarray, right: np.ndarray, k: int) -> dict:
    """CCA in already-whitened coordinates. Columns of U are orthonormal."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = max(a.shape[0] - 1, 1)
    cross = (a.T @ b) / denom
    p, s, qt = np.linalg.svd(cross, full_matrices=False)
    width = min(int(k), int(s.shape[0]))
    u_a = p[:, :width]
    u_b = qt.T[:, :width]
    signs = _column_signs(u_a)
    u_a = u_a * signs
    u_b = u_b * signs
    return {"U_left": u_a, "U_right": u_b, "singular": np.asarray(s[:width], dtype=np.float64)}


def mean_squared_cosine(left: np.ndarray, right: np.ndarray) -> float:
    cross = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    singular = np.linalg.svd(cross, compute_uv=False)
    if singular.size == 0:
        return float("nan")
    return float(np.mean(np.clip(singular, 0.0, 1.0) ** 2))


def principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cross = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    singular = np.linalg.svd(cross, compute_uv=False)
    return np.arccos(np.clip(singular, 0.0, 1.0))


def projector(basis: np.ndarray) -> np.ndarray:
    u = np.asarray(basis, dtype=np.float64)
    return u @ u.T


def consensus_basis(partner_bases: list[np.ndarray], k: int) -> dict:
    if not partner_bases:
        raise ValueError("consensus needs at least one partner basis")
    mean = np.mean([projector(u) for u in partner_bases], axis=0)
    mean = 0.5 * (mean + mean.T)
    values, vectors = np.linalg.eigh(mean)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    width = min(int(k), vectors.shape[1], partner_bases[0].shape[1])
    basis = vectors[:, :width]
    signs = _column_signs(basis)
    basis = basis * signs
    supports = []
    for col in range(width):
        direction = basis[:, col]
        per_partner = [float(np.sum((u.T @ direction) ** 2)) for u in partner_bases]
        supports.append({
            "average": float(values[col]),
            "weakest": float(min(per_partner)),
            "per_partner": per_partner,
        })
    return {"values": values, "basis": basis, "supports": supports}


def random_orthonormal(dim: int, k: int, n_frames: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    frames = []
    for _ in range(int(n_frames)):
        raw = rng.normal(size=(int(dim), int(k)))
        q, _ = np.linalg.qr(raw)
        q = q[:, :k]
        q = q * _column_signs(q)
        frames.append(q)
    return frames


def standardise_fit(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=1), 1e-8)
    return (values - mean) / scale, mean, scale


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """R orthogonal, reflections allowed, minimising ||source @ R - target||."""
    cross = np.asarray(source, dtype=np.float64).T @ np.asarray(target, dtype=np.float64)
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return u @ vt


def align_scores(scores: dict[str, np.ndarray], anchor: str, n_iter: int = 40) -> dict[str, np.ndarray]:
    names = list(scores)
    if anchor not in scores:
        raise ValueError(f"anchor {anchor} missing")
    rotations = {name: np.eye(scores[name].shape[1], dtype=np.float64) for name in names}
    for _ in range(int(n_iter)):
        changed = 0.0
        for name in names:
            if name == anchor:
                continue
            target = np.mean(
                [scores[other] @ rotations[other] for other in names if other != name],
                axis=0,
            )
            updated = orthogonal_procrustes(scores[name], target)
            changed = max(changed, float(np.max(np.abs(updated - rotations[name]))))
            rotations[name] = updated
        if changed < 1e-8:
            break
    return rotations


def maxvar_gcca(views: dict[str, np.ndarray], k: int, eta: float) -> dict:
    names = list(views)
    centered = {}
    projectors = None
    grams = {}
    for name in names:
        values = np.asarray(views[name], dtype=np.float64)
        values = values - values.mean(axis=0, keepdims=True)
        centered[name] = values
        gram = values.T @ values
        width = gram.shape[0]
        lam = float(eta) * float(np.trace(gram)) / max(width, 1)
        grams[name] = (gram, lam)
        block = values @ np.linalg.solve(gram + lam * np.eye(width), values.T)
        projectors = block if projectors is None else projectors + block
    projectors = 0.5 * (projectors + projectors.T)
    values, vectors = np.linalg.eigh(projectors)
    order = np.argsort(values)[::-1]
    width = min(int(k), vectors.shape[1])
    scores = vectors[:, order[:width]]
    scores = scores * _column_signs(scores)
    standardised, mean, scale = standardise_fit(scores)
    loadings = {}
    for name in names:
        gram, lam = grams[name]
        raw = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), centered[name].T @ scores)
        loadings[name] = raw
    return {
        "scores": standardised,
        "score_mean": mean,
        "score_scale": scale,
        "eigenvalues": values[order[:width]],
        "loadings": loadings,
        "view_mean": {name: np.asarray(views[name], dtype=np.float64).mean(axis=0) for name in names},
    }


def activation_consensus(whitened: np.ndarray, basis: np.ndarray, score_mean: np.ndarray, score_scale: np.ndarray, rotation: np.ndarray, view_mean: np.ndarray) -> np.ndarray:
    centered = np.asarray(whitened, dtype=np.float64) - np.asarray(view_mean, dtype=np.float64)
    scores = centered @ basis
    return ((scores - score_mean) / score_scale) @ rotation


def displacement_consensus(whitened_delta: np.ndarray, basis: np.ndarray, score_scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    scores = np.asarray(whitened_delta, dtype=np.float64) @ basis
    return (scores / score_scale) @ rotation


def fixed_axis_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    cors = []
    for col in range(min(a.shape[1], b.shape[1])):
        x = a[:, col] - a[:, col].mean()
        y = b[:, col] - b[:, col].mean()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        cors.append(float(np.dot(x, y) / denom) if denom > 0 else float("nan"))
    return float(np.nanmean(cors)) if cors else float("nan")


def document_means(rows: np.ndarray, doc_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    order = list(dict.fromkeys(doc_ids))
    grouped = {doc: [] for doc in order}
    for row, doc in zip(np.asarray(rows, dtype=np.float64), doc_ids):
        grouped[doc].append(row)
    means = np.stack([np.mean(grouped[doc], axis=0) for doc in order])
    return means, order


def top1_retrieval(query: np.ndarray, gallery: np.ndarray) -> float:
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    gn = np.linalg.norm(g, axis=1, keepdims=True)
    q = q / np.maximum(qn, 1e-12)
    g = g / np.maximum(gn, 1e-12)
    hits = np.argmax(q @ g.T, axis=1)
    return float(np.mean(hits == np.arange(q.shape[0])))


def ridge_maps(x_fit, y_fit, x_val, y_val, doc_fit, doc_val, alphas: list[float]) -> dict:
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    x_mean = x_fit.mean(axis=0)
    x_scale = np.maximum(x_fit.std(axis=0, ddof=1), 1e-8)
    y_mean = y_fit.mean(axis=0)
    xs = (x_fit - x_mean) / x_scale
    ys = y_fit - y_mean
    gram = xs.T @ xs
    rhs = xs.T @ ys
    best = None
    rows = []
    for alpha in alphas:
        slope = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
        pred = y_mean + ((x_val - x_mean) / x_scale) @ slope
        mse = _document_mse(pred, y_val, doc_val)
        rows.append({"alpha": float(alpha), "val_mse": mse})
        if best is None or mse < best["val_mse"]:
            best = {"alpha": float(alpha), "val_mse": mse, "slope": slope}
    slope = best["slope"]
    decode = slope.T / x_scale.reshape(1, -1)
    intercept = y_mean - (x_mean / x_scale) @ slope
    baseline = _document_mse(np.repeat(y_mean.reshape(1, -1), len(y_val), 0), y_val, doc_val)
    return {
        "decode": decode,
        "intercept": intercept,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "alpha": best["alpha"],
        "val_mse": best["val_mse"],
        "baseline_mse": baseline,
        "alphas": rows,
    }


def apply_linear(decode: np.ndarray, intercept: np.ndarray, coords: np.ndarray, *, displacement: bool) -> np.ndarray:
    """decode is (out, in). A displacement does not receive the intercept."""
    linear = np.asarray(decode, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    flat = coords.ndim == 1
    if flat:
        coords = coords.reshape(1, -1)
    out = coords @ linear.T
    if not displacement:
        out = out + np.asarray(intercept, dtype=np.float64)
    return out[0] if flat else out


def _document_mse(pred, target, doc_ids) -> float:
    err = np.mean((pred - target) ** 2, axis=1)
    buckets = {}
    for value, doc in zip(err, doc_ids):
        buckets.setdefault(doc, []).append(float(value))
    return float(np.mean([np.mean(v) for v in buckets.values()]))


def _self_check() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(400, 8))
    map_a = rng.normal(size=(8, 40))
    map_b = rng.normal(size=(8, 40))
    x = latent @ map_a + 0.01 * rng.normal(size=(400, 40))
    y = latent @ map_b + 0.01 * rng.normal(size=(400, 40))
    pack_a = fit_pca_whitener(x, 16, 0.01)
    pack_b = fit_pca_whitener(y, 16, 0.01)
    wa = whiten_activations(x, pack_a)
    wb = whiten_activations(y, pack_b)
    fit = cca_directions(wa, wb, 8)
    if not np.allclose(fit["U_left"].T @ fit["U_left"], np.eye(8), atol=1e-6):
        raise AssertionError("CCA basis is not orthonormal")
    if float(np.mean(fit["singular"][:4])) < 0.5:
        raise AssertionError("shared synthetic views did not produce a strong CCA")
    noise = cca_directions(rng.normal(size=wa.shape), rng.normal(size=wb.shape), 8)
    if mean_squared_cosine(fit["U_left"], noise["U_left"]) > 0.5:
        raise AssertionError("unrelated CCA bases overlapped")
    rotated = wa @ orthogonal_procrustes(wa, wb)
    if np.linalg.norm(rotated - wb) > np.linalg.norm(wa - wb):
        raise AssertionError("Procrustes increased the discrepancy")


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
