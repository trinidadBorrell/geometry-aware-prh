"""Polynomial, Gegenbauer, and isotropic spectral-shell kernels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from prh_replication.kernels import REL_CENTER, center_gram_o2, extension_stats


P_MAX = 12
NU_GRID = (
    0.1,
    0.14893891324883998,
    0.2218279987974549,
    0.3303882106905792,
    0.4920766105048365,
    0.7328935560376306,
    1.0915636976332257,
    1.6257631086737803,
    2.4213939060592855,
    3.6063977691583373,
    5.371329644814837,
    8.0,
)
NU_MAX = 8.0
HYP0F1_SERIES_LIM = 80.0
LIN_J_EPS = 1e-8

FAMILIES = ("monomial", "spherical", "lin_fourier", "sph_fourier")
REGIMES = {
    "unreg": (0.0, 0.0, 0.0),
    "nl": (1.0, 0.0, 0.0),
    "moderate": (0.5, 0.5, 0.25),
    "strong": (2.0, 2.0, 1.0),
}
OBJECTIVES = ("ratio", "excess")


def clamp_inner(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-1.0, 1.0)


def gegenbauer_normalised(t: torch.Tensor, d: int, p_max: int) -> list[torch.Tensor]:
    """Z_p,d(t) for p=1..p_max. Requires d>2. Z_0 omitted (centring)."""
    if d <= 2:
        raise ValueError(f"Gegenbauer construction requires d>2, got {d}")
    t = clamp_inner(t)
    zs = [t.clone()]
    if p_max == 1:
        return zs
    z_prev2 = torch.ones_like(t)
    z_prev1 = t
    for p in range(2, p_max + 1):
        a = (2 * p + d - 4) / (p + d - 3)
        b = (p - 1) / (p + d - 3)
        z = a * t * z_prev1 - b * z_prev2
        zs.append(z)
        z_prev2, z_prev1 = z_prev1, z
    return zs


def gegenbauer_scalar(t: float, d: int, p: int) -> float:
    zs = gegenbauer_normalised(torch.tensor([[float(t)]]), d, p)
    return float(zs[p - 1][0, 0])


def hyp0f1_series_tensor(b: float, z: torch.Tensor, max_k: int = 160) -> torch.Tensor:
    """0F1(;b;z) power series. z may be negative."""
    term = torch.ones_like(z)
    out = torch.ones_like(z)
    for k in range(1, max_k + 1):
        term = term * z / ((b + k - 1) * k)
        out = out + term
        if float(term.abs().max()) < 1e-12:
            break
    return out


def fourier_profile(r: torch.Tensor, nu: float, d: int) -> tuple[torch.Tensor, float]:
    """κ_{ν,d}(r)=0F1(;d/2; -d ν² r² / 4). Returns (values, gaussian_fallback_frac)."""
    if d <= 2:
        raise ValueError(f"spectral shells require d>2, got {d}")
    r = r.clamp(min=0)
    u = (d * (nu * nu) * r * r) / 4.0
    b = 0.5 * d
    series_mask = u <= HYP0F1_SERIES_LIM
    gauss = torch.exp(-0.5 * (nu * nu) * r * r)
    out = gauss.clone()
    if bool(series_mask.any()):
        z = -u[series_mask]
        out = out.clone()
        out[series_mask] = hyp0f1_series_tensor(b, z)
    frac = float((~series_mask).float().mean()) if r.numel() else 0.0
    return out, frac


def fourier_gram(dsq: torch.Tensor, s: float, nu: float, d: int) -> tuple[torch.Tensor, float]:
    if not math.isfinite(s) or s <= 0:
        raise ValueError("nonpositive distance scale")
    r = torch.sqrt(dsq.clamp(min=0)) / s
    k, frac = fourier_profile(r, nu, d)
    k = k.clone()
    k.fill_diagonal_(1.0)
    return k, frac


def monomial_grams(t: torch.Tensor, p_max: int) -> list[torch.Tensor]:
    t = clamp_inner(t)
    out = []
    pwr = torch.ones_like(t)
    for _p in range(1, p_max + 1):
        pwr = pwr * t
        k = pwr.clone()
        k.fill_diagonal_(1.0)
        out.append(k)
    return out


def family_basis_names(family: str, p_max: int = P_MAX, nu_grid=NU_GRID) -> list[dict]:
    names = []
    if family == "monomial":
        for p in range(1, p_max + 1):
            names.append({"name": f"t^{p}", "kind": "monomial", "p": p, "nu": None})
    elif family == "spherical":
        for p in range(1, p_max + 1):
            names.append({"name": f"Z{p}", "kind": "spherical", "p": p, "nu": None})
    elif family == "lin_fourier":
        names.append({"name": "linear", "kind": "monomial", "p": 1, "nu": None})
        for nu in nu_grid:
            names.append({"name": f"fourier_{nu:.6g}", "kind": "fourier", "p": None, "nu": float(nu)})
    elif family == "sph_fourier":
        for p in range(1, p_max + 1):
            names.append({"name": f"Z{p}", "kind": "spherical", "p": p, "nu": None})
        for nu in nu_grid:
            names.append({"name": f"fourier_{nu:.6g}", "kind": "fourier", "p": None, "nu": float(nu)})
    else:
        raise ValueError(family)
    return names


def build_basis_grams(
    t: torch.Tensor,
    dsq: torch.Tensor,
    s: float,
    d: int,
    family: str,
    p_max: int = P_MAX,
    nu_grid=NU_GRID,
) -> tuple[list[torch.Tensor], list[dict], float]:
    specs = family_basis_names(family, p_max, nu_grid)
    grams: list[torch.Tensor] = []
    fallback = 0.0
    n_f = 0
    mono = monomial_grams(t, p_max) if any(s0["kind"] == "monomial" for s0 in specs) else None
    sph = gegenbauer_normalised(t, d, p_max) if any(s0["kind"] == "spherical" for s0 in specs) else None
    if sph is not None:
        for g in sph:
            g.fill_diagonal_(1.0)
    for spec in specs:
        if spec["kind"] == "monomial":
            grams.append(mono[spec["p"] - 1])
        elif spec["kind"] == "spherical":
            grams.append(sph[spec["p"] - 1])
        else:
            k, frac = fourier_gram(dsq, s, spec["nu"], d)
            grams.append(k)
            fallback += frac
            n_f += 1
    fb = fallback / n_f if n_f else 0.0
    return grams, specs, fb


def center_stack(grams: list[torch.Tensor]) -> list[torch.Tensor]:
    return [center_gram_o2(g) for g in grams]


def contract_pair(kc_list: list[torch.Tensor], lc_list: list[torch.Tensor]) -> dict[str, np.ndarray]:
    if len(kc_list) != len(lc_list):
        raise ValueError("basis counts must match for shared coefficients")
    mats_a = [g.detach().cpu().contiguous().numpy().astype(np.float64, copy=False) for g in kc_list]
    mats_b = [g.detach().cpu().contiguous().numpy().astype(np.float64, copy=False) for g in lc_list]
    m = len(mats_a)
    M = np.empty((m, m), dtype=np.float64)
    A = np.empty((m, m), dtype=np.float64)
    B = np.empty((m, m), dtype=np.float64)
    tr_a = np.array([float(np.trace(x)) for x in mats_a])
    tr_b = np.array([float(np.trace(x)) for x in mats_b])
    for i in range(m):
        for j in range(i, m):
            A[i, j] = A[j, i] = float(np.sum(mats_a[i] * mats_a[j]))
            B[i, j] = B[j, i] = float(np.sum(mats_b[i] * mats_b[j]))
        for j in range(m):
            M[i, j] = float(np.sum(mats_a[i] * mats_b[j]))
    return {"M": M, "A": A, "B": B, "tr_a": tr_a, "tr_b": tr_b, "n": int(kc_list[0].shape[0])}


def cka_from_stats(c: np.ndarray, stats: dict[str, np.ndarray]) -> dict[str, float]:
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    ip = float(c @ stats["M"] @ c)
    nk2 = float(c @ stats["A"] @ c)
    nl2 = float(c @ stats["B"] @ c)
    trk = float(stats["tr_a"] @ c)
    trl = float(stats["tr_b"] @ c)
    n = int(stats["n"])
    nk = math.sqrt(max(nk2, 0.0))
    nl = math.sqrt(max(nl2, 0.0))
    finite = all(math.isfinite(v) for v in (ip, nk, nl, trk, trl))
    valid = finite and nk > 0 and nl > 0
    a = ip / (nk * nl) if valid else float("nan")
    b = (trk * trl) / ((n - 1) * nk * nl) if valid else float("nan")
    den = trk * trl
    ratio_undef = (not valid) or abs(den) <= REL_CENTER * n * nk * nl
    ratio = ((n - 1) * ip / den) if not ratio_undef else float("nan")
    excess = (a - b) if valid and math.isfinite(b) else float("nan")
    return {"a": a, "b": b, "ratio": ratio, "excess": excess, "valid_a": valid, "ip": ip, "nk": nk, "nl": nl}


def penalty_and_parts(c: np.ndarray, specs: list[dict], lam_nl, lam_c, lam_s) -> tuple[float, dict]:
    c = np.asarray(c, dtype=np.float64)
    lin_idx = 0
    for i, sp in enumerate(specs):
        if sp.get("p") == 1 or sp["name"] == "linear":
            lin_idx = i
            break
    c_lin = float(c[lin_idx])
    nl = 1.0 - c_lin
    poly = 0.0
    four = 0.0
    smooth = 0.0
    p_max = max((sp["p"] or 1) for sp in specs)
    fou_idx = [i for i, sp in enumerate(specs) if sp["kind"] == "fourier"]
    for i, sp in enumerate(specs):
        if sp["kind"] in ("monomial", "spherical") and sp.get("p") and sp["p"] >= 2:
            poly += float(c[i]) * ((sp["p"] - 1) / max(p_max - 1, 1)) ** 2
        if sp["kind"] == "fourier":
            four += float(c[i]) * (float(sp["nu"]) / NU_MAX) ** 2
    if len(fou_idx) >= 2:
        cf = c[fou_idx]
        dlt = np.diff(cf)
        smooth = float(np.sum(dlt * dlt))
    R = lam_nl * nl + lam_c * (poly + four) + lam_s * smooth
    return float(R), {"nl_mass": nl, "c_linear": c_lin, "poly_cost": poly, "fourier_cost": four, "smooth": smooth, "lin_idx": lin_idx}


def project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def objective_value(st: dict[str, float], name: str) -> float:
    return float(st[name])


def scaled_objective(st: dict[str, float], name: str, j_lin: float) -> float:
    j = objective_value(st, name)
    if not math.isfinite(j) or not math.isfinite(j_lin) or j_lin <= LIN_J_EPS:
        return float("nan")
    return j / j_lin


def fit_simplex(
    stats: dict[str, np.ndarray],
    specs: list[dict],
    obj: str,
    j_lin: float,
    lams: tuple[float, float, float],
    n_steps: int = 250,
) -> dict[str, Any]:
    m = stats["M"].shape[0]
    starts = [np.zeros(m), np.ones(m) / m]
    starts[0][0] = 1.0
    verts = []
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        verts.append((objective_value(cka_from_stats(e, stats), obj), e))
    verts.sort(key=lambda z: -z[0] if math.isfinite(z[0]) else -1e9)
    starts.append(verts[0][1].copy())
    rng = np.random.default_rng(0)
    for s in range(3):
        starts.append(rng.dirichlet(np.ones(m)))

    def loss_grad(c):
        st = cka_from_stats(c, stats)
        jsc = scaled_objective(st, obj, j_lin)
        R, parts = penalty_and_parts(c, specs, *lams)
        if not math.isfinite(jsc):
            return float("nan"), np.zeros_like(c), st, parts
        # finite difference on simplex via ambient then project
        eps = 1e-5
        g = np.zeros_like(c)
        for i in range(m):
            cp = c.copy()
            cp[i] += eps
            cp = project_simplex(cp)
            stp = cka_from_stats(cp, stats)
            j2 = scaled_objective(stp, obj, j_lin)
            R2, _ = penalty_and_parts(cp, specs, *lams)
            g[i] = ((j2 - R2) - (jsc - R)) / eps
        return jsc - R, g, st, parts

    best = None
    traces = []
    for si, c0 in enumerate(starts):
        c = project_simplex(c0 + 1e-8)
        last = None
        for step in range(n_steps):
            val, g, st, parts = loss_grad(c)
            if not math.isfinite(val):
                break
            lr = 0.15 / math.sqrt(1 + step / 40)
            c = project_simplex(c + lr * g)
            last = (val, c.copy(), st, parts, si)
        if last is not None:
            traces.append({"start": si, "penalised": last[0], "c": last[1].tolist(), "a": last[2]["a"]})
            if best is None or last[0] > best[0]:
                best = last
    if best is None:
        c = np.zeros(m)
        c[0] = 1.0
        st = cka_from_stats(c, stats)
        R, parts = penalty_and_parts(c, specs, *lams)
        return {"c": c.tolist(), "penalised": float("nan"), "unpenalised": st, "penalty": R, "parts": parts, "starts": traces, "ok": False}
    val, c, st, parts, si = best
    R, parts = penalty_and_parts(c, specs, *lams)
    st_final = cka_from_stats(c, stats)
    return {
        "c": c.tolist(),
        "penalised": val,
        "unpenalised": st_final,
        "penalty": R,
        "parts": parts,
        "best_start": si,
        "n_starts": len(starts),
        "start_spread": float(np.std([t["penalised"] for t in traces])) if traces else 0.0,
        "ok": True,
    }


def mix_grams(grams: list[torch.Tensor], c: np.ndarray) -> torch.Tensor:
    k = c[0] * grams[0]
    for ci, g in zip(c[1:], grams[1:]):
        if ci == 0:
            continue
        k = k + float(ci) * g
    return k


def gaussian_limit_error(d: int, nu_grid=NU_GRID, n_r: int = 64) -> dict:
    r = torch.linspace(0, 2.5, n_r)
    rows = []
    for nu in nu_grid:
        k, frac = fourier_profile(r, float(nu), d)
        g = torch.exp(-0.5 * (nu * nu) * r * r)
        rows.append({"nu": float(nu), "max_abs": float((k - g).abs().max()), "fallback_frac": frac})
    return {"d": d, "max_over_grid": max(r["max_abs"] for r in rows), "per_nu": rows}
