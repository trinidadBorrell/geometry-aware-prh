"""Bounded anisotropic metrics in a frozen training-PCA subspace.

M = I + U(B − I)Uᵀ acts as B inside span(U) and identity on the residual.
k_M(x, x′) = zᵀ M z′ with z = x − μ_train. CKA uses sample-centred Grams.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from prh_replication.kernels import REL_CENTER, center_gram_o2, extension_stats


Q_DEFAULT = 32
LOG4 = math.log(4.0)
EIG_LO = 0.25
EIG_HI = 4.0
TAU_GRID = (0.0, 0.01, 0.1, 1.0)
OBJECTIVES = ("ratio", "excess")
FAMILIES = ("diag", "full")
BOUND_ATOL = 1e-5
LIN_EPS = 1e-8


def to_numpy64(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def sample_center(z: np.ndarray) -> np.ndarray:
    return z - z.mean(axis=0, keepdims=True)


def fit_pca(z: np.ndarray, q: int = Q_DEFAULT, rank_tol: float = 1e-8) -> dict[str, Any]:
    """PCA of train-mean-centred features. Columns of U are orthonormal."""
    z = to_numpy64(z)
    n, d = z.shape
    if n < 2 or d < 1:
        raise ValueError("PCA requires n>=2 and d>=1")
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    s0 = float(s[0]) if s.size else 0.0
    thresh = rank_tol * s0 if s0 > 0 else rank_tol
    numerical_rank = int(np.sum(s > thresh))
    q_use = int(min(q, numerical_rank, d, n - 1))
    if q_use < 1:
        raise ValueError("PCA subspace is empty")
    u = vt[:q_use].T.copy()
    total = float(np.sum(s * s))
    covered = float(np.sum(s[:q_use] ** 2) / total) if total > 0 else float("nan")
    return {
        "U": u,
        "q": q_use,
        "q_requested": int(q),
        "numerical_rank": numerical_rank,
        "singular_values": s.astype(np.float64),
        "variance_fraction": covered,
        "n": int(n),
        "d": int(d),
        "reduced": q_use < q,
    }


def fro_sq_gram(z: np.ndarray) -> float:
    """||Z Zᵀ||_F² = ||Zᵀ Z||_F², using the smaller factor."""
    n, d = z.shape
    if d <= n:
        g = z.T @ z
    else:
        g = z @ z.T
    return float(np.sum(g * g))


def make_pack(za: np.ndarray, zb: np.ndarray, ua: np.ndarray, ub: np.ndarray) -> dict[str, Any]:
    """Contracted statistics for Kc = K_lin,c + Pc ΔA Pcᵀ (and side B)."""
    za, zb = to_numpy64(za), to_numpy64(zb)
    ua, ub = to_numpy64(ua), to_numpy64(ub)
    if za.shape[0] != zb.shape[0]:
        raise ValueError("paired sample counts must match")
    zca, zcb = sample_center(za), sample_center(zb)
    pc, qc = zca @ ua, zcb @ ub
    cross = zca.T @ zcb
    ip_lin = float(np.sum(cross * cross))
    nk2 = fro_sq_gram(zca)
    nl2 = fro_sq_gram(zcb)
    trk = float(np.sum(zca * zca))
    trl = float(np.sum(zcb * zcb))
    xa = zca.T @ pc
    xb = zcb.T @ qc
    xab = zca.T @ qc
    xba = zcb.T @ pc
    pack = {
        "n": int(za.shape[0]),
        "q_a": int(ua.shape[1]),
        "q_b": int(ub.shape[1]),
        "ip_lin": ip_lin,
        "nk2_lin": nk2,
        "nl2_lin": nl2,
        "trk_lin": trk,
        "trl_lin": trl,
        "gaa": pc.T @ pc,
        "gbb": qc.T @ qc,
        "gab": pc.T @ qc,
        "ca": xa.T @ xa,
        "cb": xb.T @ xb,
        "k_on_b": xab.T @ xab,
        "l_on_a": xba.T @ xba,
    }
    return pack


def pack_to_torch(pack: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in pack.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.tensor(v, dtype=torch.float64)
        else:
            out[k] = v
    return out


def _scores_from_deltas(
    da: torch.Tensor,
    db: torch.Tensor,
    p: dict[str, Any],
) -> dict[str, torch.Tensor]:
    ip = p["ip_lin"] + torch.trace(da @ p["l_on_a"]) + torch.trace(db @ p["k_on_b"]) + torch.trace(da @ p["gab"] @ db @ p["gab"].T)
    nk2 = p["nk2_lin"] + 2.0 * torch.trace(da @ p["ca"]) + torch.trace(da @ p["gaa"] @ da @ p["gaa"])
    nl2 = p["nl2_lin"] + 2.0 * torch.trace(db @ p["cb"]) + torch.trace(db @ p["gbb"] @ db @ p["gbb"])
    trk = p["trk_lin"] + torch.trace(da @ p["gaa"])
    trl = p["trl_lin"] + torch.trace(db @ p["gbb"])
    n = float(p["n"])
    nk = torch.sqrt(torch.clamp(nk2, min=0.0))
    nl = torch.sqrt(torch.clamp(nl2, min=0.0))
    finite = torch.isfinite(ip) & torch.isfinite(nk) & torch.isfinite(nl) & torch.isfinite(trk) & torch.isfinite(trl)
    valid = finite & (nk > 0) & (nl > 0)
    a = torch.where(valid, ip / (nk * nl), torch.tensor(float("nan"), dtype=torch.float64))
    b = torch.where(valid, (trk * trl) / ((n - 1.0) * nk * nl), torch.tensor(float("nan"), dtype=torch.float64))
    den = trk * trl
    ratio_undef = (~valid) | (den.abs() <= REL_CENTER * n * nk * nl)
    ratio = torch.where(~ratio_undef, (n - 1.0) * ip / den, torch.tensor(float("nan"), dtype=torch.float64))
    excess = torch.where(valid & torch.isfinite(b), a - b, torch.tensor(float("nan"), dtype=torch.float64))
    r_eff_k = torch.where(nk2 > 0, (trk * trk) / nk2, torch.tensor(float("nan"), dtype=torch.float64))
    r_eff_l = torch.where(nl2 > 0, (trl * trl) / nl2, torch.tensor(float("nan"), dtype=torch.float64))
    return {
        "a": a,
        "b": b,
        "ratio": ratio,
        "excess": excess,
        "ip": ip,
        "nk": nk,
        "nl": nl,
        "trk": trk,
        "trl": trl,
        "valid_a": valid,
        "ratio_undefined": ratio_undef,
        "r_eff_k": r_eff_k,
        "r_eff_l": r_eff_l,
        "degenerate": (~valid) & finite,
    }


def scores_numpy(da: np.ndarray, db: np.ndarray, pack: dict[str, Any]) -> dict[str, float]:
    p = pack_to_torch(pack)
    out = _scores_from_deltas(torch.tensor(da, dtype=torch.float64), torch.tensor(db, dtype=torch.float64), p)
    rec = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            rec[k] = bool(v.item()) if v.dtype == torch.bool else float(v.item())
        else:
            rec[k] = v
    return rec


def identity_deltas(qa: int, qb: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((qa, qa)), np.zeros((qb, qb))


def metric_gram(z: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    z, u, b = to_numpy64(z), to_numpy64(u), to_numpy64(b)
    p = z @ u
    delta = b - np.eye(b.shape[0])
    return z @ z.T + p @ delta @ p.T


def truncation_gram(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    p = to_numpy64(z) @ to_numpy64(u)
    return p @ p.T


def reconstruct_b(family: str, params: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return B, log B, eigenvalues of B from unconstrained parameters."""
    bound = LOG4
    if family == "diag":
        h = params.clamp(-bound, bound)
        eig = torch.exp(h)
        b = torch.diag(eig)
        logb = torch.diag(h)
        return b, logb, eig
    if family != "full":
        raise ValueError(family)
    s = _unvech(params, q)
    s_sym = 0.5 * (s + s.T)
    eye = torch.eye(q, dtype=s_sym.dtype, device=s_sym.device)
    if not torch.isfinite(s_sym).all():
        s_sym = torch.zeros_like(s_sym)
    try:
        evals, vecs = torch.linalg.eigh(s_sym)
    except RuntimeError:
        evals, vecs = torch.linalg.eigh(s_sym + 1e-6 * eye)
    ev = evals.clamp(-bound, bound)
    logb = vecs @ torch.diag(ev) @ vecs.T
    eig = torch.exp(ev)
    b = vecs @ torch.diag(eig) @ vecs.T
    b = 0.5 * (b + b.T)
    return b, logb, eig


def n_params(family: str, q: int) -> int:
    if family == "diag":
        return q
    return q * (q + 1) // 2


def _vech(s: torch.Tensor) -> torch.Tensor:
    q = s.shape[0]
    i, j = torch.triu_indices(q, q, device=s.device)
    return s[i, j]


def _unvech(v: torch.Tensor, q: int) -> torch.Tensor:
    s = torch.zeros(q, q, dtype=v.dtype, device=v.device)
    i, j = torch.triu_indices(q, q, device=v.device)
    s[i, j] = v
    return s + s.T - torch.diag(torch.diag(s))


def distortion_r(log_a: torch.Tensor, log_b: torch.Tensor, qa: int, qb: int) -> torch.Tensor:
    return 0.5 * ((log_a * log_a).sum() / qa + (log_b * log_b).sum() / qb)


def train_objective(
    scores: dict[str, torch.Tensor],
    objective: str,
    tau: float,
    r: torch.Tensor,
    lin: dict[str, float],
) -> torch.Tensor:
    nan = torch.tensor(float("nan"), dtype=torch.float64)
    if objective == "ratio":
        jlin = lin["ratio"]
        if not math.isfinite(jlin) or jlin <= LIN_EPS:
            return nan
        return scores["ratio"] / jlin - 1.0 - float(tau) * r
    if objective == "excess":
        alin = lin["a"]
        if not math.isfinite(alin) or alin <= LIN_EPS:
            return nan
        dlin = lin["excess"]
        return (scores["excess"] - dlin) / alin - float(tau) * r
    raise ValueError(objective)


def unpenalised(scores: dict[str, Any], objective: str) -> float:
    key = "ratio" if objective == "ratio" else "excess"
    v = scores[key]
    return float(v.item() if isinstance(v, torch.Tensor) else v)


def metric_diagnostics(b: np.ndarray, logb: np.ndarray, eig: np.ndarray) -> dict[str, float]:
    eig = np.asarray(eig, dtype=np.float64)
    cond = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    logabs = np.abs(np.log(np.clip(eig, 1e-30, None)))
    frac = float(np.mean(logabs >= LOG4 - BOUND_ATOL))
    return {
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "cond": cond,
        "frac_at_bounds": frac,
        "r_side": float(0.5 * np.sum(np.asarray(logb) ** 2) / len(eig) * 2) if False else float(np.sum(np.asarray(logb) ** 2) / len(eig)),
        "log_fro2": float(np.sum(np.asarray(logb) ** 2)),
    }


def params_to_metrics(family: str, pa: torch.Tensor, pb: torch.Tensor, qa: int, qb: int):
    ba, la, ea = reconstruct_b(family, pa, qa)
    bb, lb, eb = reconstruct_b(family, pb, qb)
    da = ba - torch.eye(qa, dtype=ba.dtype, device=ba.device)
    db = bb - torch.eye(qb, dtype=bb.dtype, device=bb.device)
    return ba, bb, la, lb, ea, eb, da, db


def _as_torch_pack(pack: dict[str, Any]) -> dict[str, Any]:
    if any(isinstance(v, np.ndarray) for v in pack.values()):
        return pack_to_torch(pack)
    return pack


def evaluate_params(family: str, pa: torch.Tensor, pb: torch.Tensor, pack: dict[str, Any], tau: float, objective: str, lin: dict[str, float]):
    p = _as_torch_pack(pack)
    qa, qb = int(p["q_a"]), int(p["q_b"])
    ba, bb, la, lb, ea, eb, da, db = params_to_metrics(family, pa, pb, qa, qb)
    sc = _scores_from_deltas(da, db, p)
    r = distortion_r(la, lb, qa, qb)
    pen = train_objective(sc, objective, tau, r, lin)
    return sc, r, pen, ba, bb, la, lb, ea, eb


def _init_params(family: str, qa: int, qb: int, kind: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    na, nb = n_params(family, qa), n_params(family, qb)
    if kind == "identity":
        return torch.zeros(na, dtype=torch.float64), torch.zeros(nb, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    scale = 0.05
    return scale * torch.randn(na, dtype=torch.float64, generator=g), scale * torch.randn(nb, dtype=torch.float64, generator=g)


def fit_metrics(
    pack: dict[str, Any],
    family: str,
    objective: str,
    tau: float,
    lin: dict[str, float],
    n_steps: int = 120,
    lr: float | None = None,
) -> dict[str, Any]:
    """Three starts (identity + two perturbations). Select by penalised train objective."""
    if lr is None:
        lr = 0.08 if family == "diag" else 0.03
    p = pack_to_torch(pack)
    qa, qb = int(p["q_a"]), int(p["q_b"])
    starts = [("identity", 0), ("pert0", 11), ("pert1", 23)]
    best = None
    records = []
    for name, seed in starts:
        pa, pb = _init_params(family, qa, qb, "identity" if name == "identity" else "pert", seed)
        pa = pa.clone().requires_grad_(True)
        pb = pb.clone().requires_grad_(True)
        opt = torch.optim.Adam([pa, pb], lr=lr)
        hist = []
        ok = True
        for step in range(n_steps):
            opt.zero_grad()
            try:
                sc, r, pen, *_ = evaluate_params(family, pa, pb, p, tau, objective, lin)
            except RuntimeError:
                ok = False
                break
            if (not torch.isfinite(pen)) or (not pen.requires_grad):
                ok = False
                break
            loss = -pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pa, pb], 5.0)
            opt.step()
            with torch.no_grad():
                pa.clamp_(-4.0, 4.0)
                pb.clamp_(-4.0, 4.0)
            if (step + 1) % max(n_steps // 4, 1) == 0 or step == n_steps - 1:
                hist.append(float(pen.detach()))
        try:
            with torch.no_grad():
                sc, r, pen, ba, bb, la, lb, ea, eb = evaluate_params(family, pa, pb, p, tau, objective, lin)
        except RuntimeError:
            continue
        rec = {
            "start": name,
            "ok": ok and bool(torch.isfinite(pen)),
            "penalised": float(pen.detach()) if torch.isfinite(pen) else float("-inf"),
            "r": float(r.detach()),
            "train": {k: float(v.detach()) if v.dtype != torch.bool else bool(v.detach()) for k, v in sc.items()},
            "pa": pa.detach().clone(),
            "pb": pb.detach().clone(),
            "eig_a": ea.detach().cpu().numpy(),
            "eig_b": eb.detach().cpu().numpy(),
            "b_a": ba.detach().cpu().numpy(),
            "b_b": bb.detach().cpu().numpy(),
            "log_a": la.detach().cpu().numpy(),
            "log_b": lb.detach().cpu().numpy(),
            "hist": hist,
        }
        records.append(rec)
        if best is None or rec["penalised"] > best["penalised"] + 1e-15:
            best = rec
        elif abs(rec["penalised"] - best["penalised"]) <= 1e-15 and rec["r"] < best["r"]:
            best = rec
    # explicit identity (already a start, but re-score zeros in case Adam moved them)
    iz = scores_numpy(*identity_deltas(qa, qb), pack)
    r0 = 0.0
    pen0 = train_objective(
        {k: torch.tensor(v, dtype=torch.float64) if not isinstance(v, bool) else torch.tensor(v) for k, v in iz.items() if k in ("a", "b", "ratio", "excess")},
        objective,
        tau,
        torch.tensor(0.0, dtype=torch.float64),
        lin,
    )
    # simpler identity penalised
    if objective == "ratio":
        pen0f = (iz["ratio"] / lin["ratio"] - 1.0) if lin["ratio"] > LIN_EPS else float("nan")
    else:
        pen0f = ((iz["excess"] - lin["excess"]) / lin["a"]) if lin["a"] > LIN_EPS else float("nan")
    if math.isfinite(pen0f) and (best is None or pen0f > best["penalised"] + 1e-15 or (abs(pen0f - best["penalised"]) <= 1e-15 and 0.0 <= best["r"])):
        if best is None or pen0f > best["penalised"] + 1e-15 or (abs(pen0f - best["penalised"]) <= 1e-12 and best["r"] > 0):
            best = {
                "start": "identity_explicit",
                "ok": True,
                "penalised": float(pen0f),
                "r": 0.0,
                "train": iz,
                "pa": torch.zeros(n_params(family, qa), dtype=torch.float64),
                "pb": torch.zeros(n_params(family, qb), dtype=torch.float64),
                "eig_a": np.ones(qa),
                "eig_b": np.ones(qb),
                "b_a": np.eye(qa),
                "b_b": np.eye(qb),
                "log_a": np.zeros((qa, qa)),
                "log_b": np.zeros((qb, qb)),
                "hist": [],
            }
    spread = float(np.std([r["penalised"] for r in records if math.isfinite(r["penalised"])])) if records else 0.0
    ba, bb = best["b_a"], best["b_b"]
    da, db = ba - np.eye(qa), bb - np.eye(qb)
    out_scores = scores_numpy(da, db, pack)
    diag_a = metric_diagnostics(ba, best["log_a"], best["eig_a"])
    diag_b = metric_diagnostics(bb, best["log_b"], best["eig_b"])
    return {
        "family": family,
        "objective": objective,
        "tau": float(tau),
        "selected_start": best["start"],
        "ok": best["ok"],
        "penalised": best["penalised"],
        "r": float(0.5 * (diag_a["log_fro2"] / qa + diag_b["log_fro2"] / qb)),
        "train": out_scores,
        "b_a": ba,
        "b_b": bb,
        "eig_a": np.asarray(best["eig_a"], dtype=np.float64),
        "eig_b": np.asarray(best["eig_b"], dtype=np.float64),
        "diag_a": diag_a,
        "diag_b": diag_b,
        "frac_at_bounds": 0.5 * (diag_a["frac_at_bounds"] + diag_b["frac_at_bounds"]),
        "cond_a": diag_a["cond"],
        "cond_b": diag_b["cond"],
        "start_spread": spread,
        "n_starts": len(records),
        "identity_penalised": float(pen0f) if math.isfinite(pen0f) else None,
    }


def select_tau(candidates: dict[float, dict], pack_val: dict, objective: str) -> tuple[float, dict]:
    """Max unpenalised validation objective; ties prefer lower R then lower τ."""
    ranked = []
    for tau, fit in candidates.items():
        sc = scores_numpy(fit["b_a"] - np.eye(fit["b_a"].shape[0]), fit["b_b"] - np.eye(fit["b_b"].shape[0]), pack_val)
        val = unpenalised(sc, objective)
        r = fit["r"]
        ranked.append((val if math.isfinite(val) else -1e18, -r, -float(tau), tau, fit, sc))
    ranked.sort(reverse=True)
    val, _, _, tau, fit, sc = ranked[0]
    out = dict(fit)
    out["selected_tau"] = tau
    out["val"] = sc
    out["val_unpenalised"] = val
    return tau, out


def metric_sq_distances(z: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    z, u, b = to_numpy64(z), to_numpy64(u), to_numpy64(b)
    p = z @ u
    g = z @ z.T
    dsq = np.diag(g)[:, None] + np.diag(g)[None, :] - 2.0 * g
    delta = b - np.eye(b.shape[0])
    mid = p @ delta @ p.T
    extra = np.diag(mid)[:, None] + np.diag(mid)[None, :] - 2.0 * mid
    return dsq + extra


def knn_from_sq(dsq: np.ndarray, k: int = 10) -> np.ndarray:
    d = np.array(dsq, copy=True, dtype=np.float64)
    np.fill_diagonal(d, np.inf)
    k = int(min(k, d.shape[0] - 1))
    part = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    row = np.arange(d.shape[0])[:, None]
    order = np.argsort(d[row, part], axis=1)
    return part[row, order]


def knn_overlap(knn_a: np.ndarray, knn_b: np.ndarray) -> float:
    k = knn_a.shape[1]
    hits = []
    for i in range(knn_a.shape[0]):
        hits.append(len(set(knn_a[i].tolist()) & set(knn_b[i].tolist())) / k)
    return float(np.mean(hits))


def mnn_from_knn(knn_a: np.ndarray, knn_b: np.ndarray) -> float:
    k = knn_a.shape[1]
    hits = []
    for i in range(knn_a.shape[0]):
        hits.append(len(set(knn_a[i].tolist()) & set(knn_b[i].tolist())) / k)
    return float(np.mean(hits))


def principal_angles(ua: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Principal angles between column spaces."""
    qa = np.linalg.qr(to_numpy64(ua), mode="reduced")[0]
    qb = np.linalg.qr(to_numpy64(ub), mode="reduced")[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def fro_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = to_numpy64(a).ravel(), to_numpy64(b).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def centred_diag_energy(kc: np.ndarray) -> float:
    fro2 = float(np.sum(kc * kc))
    diag2 = float(np.sum(np.diag(kc) ** 2))
    return float(diag2 / fro2) if fro2 > 0 else float("nan")
