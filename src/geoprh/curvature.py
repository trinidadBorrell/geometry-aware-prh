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
    x: np.ndarray, d: int, ks: list[int], n_centres: int = 300, seed: int = 0, n_jobs: int = 1
) -> dict:
    """Per centre and k: T (tangent variance), V_N (normal variance), m2 = |normal part of p - c|^2.

    Returns {"k": [K], "T": [C, K], "VN": [C, K], "m2": [C, K], "centres": [C]}.
    """
    rng = np.random.default_rng(seed)
    centres = rng.choice(x.shape[0], min(n_centres, x.shape[0]), replace=False)
    ks = [k for k in ks if d + 2 <= k < x.shape[0]]
    idx = NearestNeighbors(n_neighbors=max(ks) + 1, n_jobs=n_jobs).fit(x)
    idx = idx.kneighbors(x[centres], return_distance=False)
    out = {key: np.empty((len(centres), len(ks))) for key in ("T", "VN", "m2")}
    for ci, row in enumerate(idx):
        p = x[row[0]]
        for ki, k in enumerate(ks):
            nb = x[row[1 : k + 1]]  # the k neighbours, p excluded
            c = nb.mean(axis=0)
            y = nb - c
            lam, u = np.linalg.eigh(y @ y.T)
            lam, u = lam[::-1].clip(min=0), u[:, ::-1]
            var = lam / k
            basis = y.T @ (u[:, :d] / np.sqrt(np.maximum(lam[:d], 1e-300)))  # D x d, orthonormal
            off = p - c
            out["T"][ci, ki] = var[:d].sum()
            out["VN"][ci, ki] = var[d:].sum()
            out["m2"][ci, ki] = off @ off - np.sum((basis.T @ off) ** 2)
    return {"k": ks, "centres": centres, **out}


def _slope(t2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y = a + b t2; returns (b, a)."""
    b, a = np.polyfit(t2, y, 1)
    return float(b), float(a)


def _from_slopes(b: float, e: float, d: int) -> tuple[float, float, float]:
    h2 = 4 * d**2 * b
    ii2 = ((4 * d**2 * e + h2) * (d + 4) / (d + 2) - h2) / 2
    return h2, ii2, h2 - ii2


def curvature_profile(mom: dict, d: int) -> list[dict]:
    """Scale profile: the estimate from each pair of consecutive k (slopes between two scales).
    Where the second-order expansion holds, it is flat in k."""
    t2 = mom["T"].mean(axis=0) ** 2
    m2, vn = mom["m2"].mean(axis=0), mom["VN"].mean(axis=0)
    out = []
    for i in range(len(t2) - 1):
        dt = t2[i + 1] - t2[i]
        h2, ii2, s = _from_slopes((m2[i + 1] - m2[i]) / dt, (vn[i + 1] - vn[i]) / dt, d)
        out.append(
            {
                "k": [mom["k"][i], mom["k"][i + 1]],
                "radius": float(np.sqrt(mom["T"].mean(axis=0)[i + 1] * (d + 2) / d)),
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
