"""Second-order (moment) curvature estimator for point clouds (exp-013).

Near a point p, a d-dimensional manifold M in R^D is the graph of its second fundamental form II:

    x = p + u + 1/2 II(u, u) + O(|u|^3),     u in the tangent space T_p M, II(u, u) normal to it.

For the neighbours of p with tangent offsets u spread isotropically (a uniform ball of radius r
has E[u_i u_j] = r^2/(d+2) delta_ij and E[u_i u_j u_k u_l] = r^4/((d+2)(d+4)) (dd + dd + dd)),
two moments of the neighbourhood, measured in the normal space N_p M, carry the curvature:

- the normal offset of the neighbourhood's centroid from p (first moment):
      m = 1/2 E[II(u, u)] = 1/2 (T/d) H,            H = sum_i II(e_i, e_i), the mean-curvature vector
- the variance of the neighbours in the normal space (second moment):
      V_N = 1/4 (E|II(u,u)|^2 - |E II(u,u)|^2)
          = 1/4 (T/d)^2 [ (d+2)/(d+4) (|H|^2 + 2 |II|^2) - |H|^2 ]
  with T = E|u|^2 the variance in the tangent space (so T/d = r^2/(d+2)) and
  |II|^2 = sum_ij |II(e_i, e_j)|^2.

Both grow with the scale: |m|^2 = b T^2 and V_N = e T^2, plus constants from noise (noise in p
itself enters m, noise in the neighbours enters V_N, neither grows with T). Regressing |m|^2 and
V_N on T^2 over several neighbourhood sizes k gives the slopes b, e, hence

    |H|^2  = 4 d^2 b
    |II|^2 = [ (4 d^2 e + |H|^2) (d+4)/(d+2) - |H|^2 ] / 2
    S      = |H|^2 - |II|^2                                   (Gauss equation)

S is the scalar curvature, an intrinsic quantity (unchanged by bending without stretching);
|H| and |II| are extrinsic (how M sits in R^D). For the unit sphere S^d: |H|^2 = d^2,
|II|^2 = d, S = d(d-1); for any intrinsically flat manifold (plane, Swiss roll, flat torus) S = 0
however much it is rolled. We report S / (d(d-1)), which is 1/R^2 for a sphere of radius R.

Tangent space: the top-d principal directions of each neighbourhood (PCA centred at its
centroid). Spectra come from the k x k Gram matrix (same nonzero eigenvalues as the covariance).
Assumptions: roughly uniform density and isotropic neighbourhoods, r small enough for the
second-order expansion; how far these hold at our n and ID is what exp-013 tests.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def neighbourhood_moments(
    x: np.ndarray,
    ds: int | list[int],
    ks: list[int],
    n_centres: int = 300,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    """Per centre, k and tangent dimension d: T (tangent variance), V_N (normal variance per
    point) and m2 (squared normal offset of p from the neighbourhood centroid).

    Variances use the unbiased 1/(k-1) normalisation, and m2 is corrected for the sampling noise
    of the centroid: with k neighbours, |normal part of p - c|^2 has expectation
    |m|^2 + V_N / k, a term that shrinks with k and would otherwise bias |H| low.
    Several d share one eigendecomposition per neighbourhood.

    Returns {"k": [K], "ds": [D], "centres": [C], "T" / "VN" / "m2": [C, K, D]}.
    """
    ds = [ds] if isinstance(ds, int) else list(ds)
    rng = np.random.default_rng(seed)
    centres = rng.choice(x.shape[0], min(n_centres, x.shape[0]), replace=False)
    ks = [k for k in ks if max(ds) + 2 <= k < x.shape[0]]
    idx = NearestNeighbors(n_neighbors=max(ks) + 1, n_jobs=n_jobs).fit(x)
    idx = idx.kneighbors(x[centres], return_distance=False)
    shape = (len(centres), len(ks), len(ds))
    out = {key: np.empty(shape) for key in ("T", "VN", "m2")}
    for ci, row in enumerate(idx):
        p = x[row[0]]
        for ki, k in enumerate(ks):
            nb = x[row[1 : k + 1]]  # the k neighbours, p excluded
            c = nb.mean(axis=0)
            y = nb - c
            lam, u = np.linalg.eigh(y @ y.T)
            lam, u = lam[::-1].clip(min=0), u[:, ::-1]
            var = lam / (k - 1)
            off = p - c
            # projections of the offset on every principal direction (orthonormal basis y^T u / sqrt(lam))
            proj = (u.T @ (y @ off)) / np.sqrt(np.maximum(lam, 1e-300))
            off2 = off @ off
            cum_proj = np.cumsum(proj**2)
            cum_var = np.cumsum(var)
            for di, d in enumerate(ds):
                vn = cum_var[-1] - cum_var[d - 1]
                out["T"][ci, ki, di] = cum_var[d - 1]
                out["VN"][ci, ki, di] = vn
                out["m2"][ci, ki, di] = off2 - cum_proj[d - 1] - vn / k
    return {"k": ks, "ds": ds, "centres": centres, **out}


def _slope(t2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y = a + b t2; returns (b, a)."""
    b, a = np.polyfit(t2, y, 1)
    return float(b), float(a)


def _from_slopes(b: float, e: float, d: int) -> tuple[float, float, float]:
    h2 = 4 * d**2 * b
    ii2 = ((4 * d**2 * e + h2) * (d + 4) / (d + 2) - h2) / 2
    return h2, ii2, h2 - ii2


def _select(mom: dict, d: int) -> dict:
    di = mom["ds"].index(d)
    return {key: mom[key][:, :, di] for key in ("T", "VN", "m2")}


def select_tangent_dim(mom: dict, tol: float = 0.5) -> dict:
    """Choose d from the scaling of the normal variance (the idea of multiscale SVD, Little,
    Maggioni & Rosasco): tangent variance grows like r^2 (like T), curvature-driven normal
    variance like r^4 (like T^2), noise not at all. If d is too small by m, m true tangent
    directions sit in the "normal" part, each carrying about T/d of variance, so V_N gains a term
    a T with a ~ m/d. For each candidate d, fit V_N = c + a T + e T^2 over the scales; a * d
    estimates the number of leaked tangent directions. Return the smallest d where it is < tol.
    """
    leaked = {}
    for d in sorted(mom["ds"]):
        sel = _select(mom, d)
        t, vn = sel["T"].mean(axis=0), sel["VN"].mean(axis=0)
        _, a, _ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t, t**2]), vn, rcond=None)[0]
        leaked[d] = float(a * d)
    chosen = next((d for d in sorted(leaked) if leaked[d] < tol), max(leaked))
    return {"d": chosen, "leaked": leaked}


def curvature_profile(mom: dict, d: int) -> list[dict]:
    """Scale profile: the estimate from each pair of consecutive k (slopes between two scales).
    Where the second-order expansion holds, it is flat in k."""
    sel = _select(mom, d)
    t2 = sel["T"].mean(axis=0) ** 2
    m2, vn = sel["m2"].mean(axis=0), sel["VN"].mean(axis=0)
    out = []
    for i in range(len(t2) - 1):
        dt = t2[i + 1] - t2[i]
        h2, ii2, s = _from_slopes((m2[i + 1] - m2[i]) / dt, (vn[i + 1] - vn[i]) / dt, d)
        out.append(
            {
                "k": [mom["k"][i], mom["k"][i + 1]],
                "radius": float(np.sqrt(sel["T"].mean(axis=0)[i + 1] * (d + 2) / d)),
                "S_norm": s / (d * (d - 1)) if d > 1 else float("nan"),
                "H_norm": float(np.sqrt(max(h2, 0)) / d),
                "II2": ii2,
            }
        )
    return out


def curvature_from_moments(mom: dict, d: int, n_scales: int | None = None) -> dict:
    """Pooled estimate: centre-averaged moments per k, regressed on T^2 across the first
    `n_scales` values of k (all if None)."""
    sl = slice(0, n_scales)
    mom = {**_select(mom, d), "k": mom["k"]}
    t2 = mom["T"].mean(axis=0)[sl] ** 2
    b, m2_0 = _slope(t2, mom["m2"].mean(axis=0)[sl])
    e, vn_0 = _slope(t2, mom["VN"].mean(axis=0)[sl])
    h2, ii2, s = _from_slopes(b, e, d)
    return {
        "H2": h2,
        "II2": ii2,
        "S": s,
        "S_norm": s / (d * (d - 1)) if d > 1 else float("nan"),
        "H_norm": float(np.sqrt(max(h2, 0)) / d),
        "m2_intercept": m2_0,
        "VN_intercept": vn_0,
        "T": mom["T"].mean(axis=0).tolist(),
        "VN": mom["VN"].mean(axis=0).tolist(),
        "m2": mom["m2"].mean(axis=0).tolist(),
        "k": list(mom["k"]),
    }
