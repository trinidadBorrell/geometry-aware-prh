"""One-sided budget-constrained anisotropic metrics for the release-consistency experiment.

Primary objective is CKA excess a−b. Distortion D(M)=||S||_F²/q is a hard
budget, not a Lagrange penalty. Eigenvalues of B=exp(S) stay in [1/4, 4]
by clipping eigenvalues of S, then scaling S toward zero if needed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from prh_replication.anisotropic_kernels import (
    EIG_HI,
    EIG_LO,
    LOG4,
    LIN_EPS,
    Q_DEFAULT,
    _unvech,
    fit_metrics,
    identity_deltas,
    make_pack,
    metric_diagnostics,
    n_params,
    pack_to_torch,
    scores_numpy,
    to_numpy64,
)

NEAR_ZERO_S = 1e-6
PSD_EPS = 1e-12


def distortion_d(s: np.ndarray | torch.Tensor, q: int) -> float:
    s = to_numpy64(s) if not isinstance(s, torch.Tensor) else s
    if isinstance(s, torch.Tensor):
        return float((s * s).sum() / q)
    return float(np.sum(s * s) / q)


def project_S(s: torch.Tensor, rho: float, q: int) -> torch.Tensor:
    """Symmetrise, clip eigenvalues to ±log 4, then scale toward 0 if D>ρ."""
    s_sym = 0.5 * (s + s.T)
    eye = torch.eye(q, dtype=s_sym.dtype, device=s_sym.device)
    if not torch.isfinite(s_sym).all():
        return torch.zeros_like(s_sym)
    try:
        evals, vecs = torch.linalg.eigh(s_sym)
    except RuntimeError:
        evals, vecs = torch.linalg.eigh(s_sym + 1e-6 * eye)
    ev = evals.clamp(-LOG4, LOG4)
    if float(rho) <= 0.0:
        return torch.zeros_like(s_sym)
    fro2 = (ev * ev).sum()
    budget = float(rho) * q
    if fro2 > budget and fro2 > 0:
        ev = ev * torch.sqrt(budget / fro2)
    return vecs @ torch.diag(ev) @ vecs.T


def reconstruct_budgeted(params: torch.Tensor, q: int, rho: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = project_S(_unvech(params, q), rho, q)
    evals, vecs = torch.linalg.eigh(s)
    eig = torch.exp(evals)
    b = vecs @ torch.diag(eig) @ vecs.T
    b = 0.5 * (b + b.T)
    return b, s, eig


def split_delta_psd_factors(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q×q PSD factors of (B−I). G± = H(ZU) C± (ZU)ᵀ H equals the ambient split."""
    c = to_numpy64(b) - np.eye(b.shape[0])
    return split_delta_psd(c)


def split_delta_psd_subspace(u: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSD split of ΔM=U(B−I)Uᵀ via the q×q factor (B−I). Equivalent, cheap."""
    cp, cm = split_delta_psd_factors(b)
    uu = to_numpy64(u)
    return uu @ cp @ uu.T, uu @ cm @ uu.T


def split_delta_psd(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ΔM = ΔM_+ − ΔM_- with both PSD (eigh of a symmetric matrix)."""
    d = 0.5 * (to_numpy64(delta) + to_numpy64(delta).T)
    evals, vecs = np.linalg.eigh(d)
    pos = np.clip(evals, 0.0, None)
    neg = np.clip(-evals, 0.0, None)
    d_plus = (vecs * pos) @ vecs.T
    d_minus = (vecs * neg) @ vecs.T
    return 0.5 * (d_plus + d_plus.T), 0.5 * (d_minus + d_minus.T)


def subspace_metric(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    u, b = to_numpy64(u), to_numpy64(b)
    q = b.shape[0]
    return np.eye(u.shape[0]) + u @ (b - np.eye(q)) @ u.T


def signed_correction(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    return to_numpy64(u) @ (to_numpy64(b) - np.eye(b.shape[0])) @ to_numpy64(u).T


def response_grams(z: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    """H Z ΔM Zᵀ H with sample-centering H."""
    z = to_numpy64(z)
    zc = z - z.mean(axis=0, keepdims=True)
    mid = zc @ to_numpy64(delta_m) @ zc.T
    return mid


def response_grams_factor(z: np.ndarray, u: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Same G as response_grams(z, U C Uᵀ) without forming the ambient matrix."""
    z = to_numpy64(z)
    zc = z - z.mean(axis=0, keepdims=True)
    zu = zc @ to_numpy64(u)
    return zu @ to_numpy64(c) @ zu.T


def fro_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = to_numpy64(a).ravel(), to_numpy64(b).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= NEAR_ZERO_S or nb <= NEAR_ZERO_S:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def fro_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(to_numpy64(a) - to_numpy64(b)))


def participation_ratio(evals: np.ndarray) -> float:
    e = np.clip(to_numpy64(evals), 0.0, None)
    s1 = float(np.sum(e))
    s2 = float(np.sum(e * e))
    if s2 <= 0:
        return float("nan")
    return s1 * s1 / s2


def native_anisotropy(z: np.ndarray, n_lead: int = 8, n_cum: int = 32) -> dict[str, Any]:
    z = to_numpy64(z)
    zc = z - z.mean(0, keepdims=True)
    n, d = zc.shape
    # covariance eigenvalues via SVD of centred features
    _, s, _ = np.linalg.svd(zc, full_matrices=False)
    # sample covariance eig = s^2 / (n-1)
    eig = (s * s) / max(n - 1, 1)
    total = float(np.sum(eig))
    lead = eig[:n_lead].tolist()
    cum32 = float(np.sum(eig[: min(n_cum, eig.size)]) / total) if total > 0 else float("nan")
    return {
        "n": int(n),
        "d": int(d),
        "total_variance": total,
        "participation_effective_rank": participation_ratio(eig),
        "leading_eigs": lead,
        "var_first_32_pcs": cum32,
        "eig_max": float(eig[0]) if eig.size else float("nan"),
        "eig_min_pos": float(eig[eig > 1e-12].min()) if np.any(eig > 1e-12) else 0.0,
    }


def up_down_projectors(s: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    s = 0.5 * (to_numpy64(s) + to_numpy64(s).T)
    mag = float(np.linalg.norm(s, ord="fro"))
    info = {"s_fro": mag, "direction_defined": mag > NEAR_ZERO_S, "n_up": 0, "n_down": 0}
    if mag <= NEAR_ZERO_S:
        return None, None, info
    evals, vecs = np.linalg.eigh(s)
    up_idx = np.where(evals > eps)[0]
    dn_idx = np.where(evals < -eps)[0]
    info["n_up"] = int(up_idx.size)
    info["n_down"] = int(dn_idx.size)
    p_up = vecs[:, up_idx] @ vecs[:, up_idx].T if up_idx.size else None
    p_dn = vecs[:, dn_idx] @ vecs[:, dn_idx].T if dn_idx.size else None
    return p_up, p_dn, info


def projector_agreement(p: np.ndarray | None, q: np.ndarray | None) -> float:
    if p is None or q is None:
        return float("nan")
    return fro_cosine(p, q)


def _init_vech(q: int, kind: str, seed: int) -> torch.Tensor:
    n = n_params("full", q)
    if kind == "identity":
        return torch.zeros(n, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    return 0.05 * torch.randn(n, dtype=torch.float64, generator=g)


def _vech_local(s: torch.Tensor) -> torch.Tensor:
    q = s.shape[0]
    i, j = torch.triu_indices(q, q, device=s.device)
    return s[i, j]


def vech_from_s(s: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Vech of an already-projected symmetric S (idempotent under project_S if feasible)."""
    t = torch.as_tensor(np.asarray(s, dtype=np.float64), dtype=torch.float64)
    t = 0.5 * (t + t.T)
    return _vech_local(t).clone()


def decompose_s(s: np.ndarray) -> dict[str, Any]:
    """S = μ I + S̃ with tr(S̃)=0. D = μ² + ||S̃||_F²/q."""
    s = 0.5 * (to_numpy64(s) + to_numpy64(s).T)
    q = s.shape[0]
    mu = float(np.trace(s) / q)
    st = s - mu * np.eye(q)
    d_uni = float(mu * mu)
    d_dir = float(np.sum(st * st) / q)
    d = float(np.sum(s * s) / q)
    return {
        "mu": mu,
        "s_tilde": st,
        "d_uniform": d_uni,
        "d_directional": d_dir,
        "d": d,
        "direction_defined": float(np.linalg.norm(st)) > NEAR_ZERO_S,
    }


def b_from_s(s: np.ndarray) -> np.ndarray:
    s = 0.5 * (to_numpy64(s) + to_numpy64(s).T)
    evals, vecs = np.linalg.eigh(s)
    b = (vecs * np.exp(evals)) @ vecs.T
    return 0.5 * (b + b.T)


def uniform_only_b(s: np.ndarray) -> np.ndarray:
    dec = decompose_s(s)
    q = s.shape[0]
    return math.exp(dec["mu"]) * np.eye(q)


def direction_only_b(s: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """B=exp(S̃). Eigenvalues may leave [1/4,4]; do not clip."""
    dec = decompose_s(s)
    b = b_from_s(dec["s_tilde"])
    eig = np.linalg.eigvalsh(b)
    info = {
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "within_spectral_bounds": bool(eig.min() >= EIG_LO - 1e-10 and eig.max() <= EIG_HI + 1e-10),
        "d_directional": dec["d_directional"],
    }
    return b, info


def haar_rotate_s(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep eigenvalues of S; replace eigenvectors with a Haar-random basis."""
    s = 0.5 * (to_numpy64(s) + to_numpy64(s).T)
    q = s.shape[0]
    g = rng.normal(size=(q, q))
    q_mat, r = np.linalg.qr(g)
    q_mat = q_mat * np.sign(np.diag(r))
    evals = np.linalg.eigvalsh(s)
    out = (q_mat * evals) @ q_mat.T
    return 0.5 * (out + out.T)


def _record_from_tensors(start: str, b, s, eig, sc, rho: float, q: int) -> dict[str, Any]:
    train = {k: float(v.detach()) if v.dtype != torch.bool else bool(v.detach()) for k, v in sc.items()}
    s_np = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    e_np = eig.detach().cpu().numpy() if isinstance(eig, torch.Tensor) else np.asarray(eig)
    return {
        "start": start,
        "ok": True,
        "train_excess": train["excess"],
        "d": float(np.sum(s_np ** 2) / q),
        "rho": float(rho),
        "b": b_np,
        "s": s_np,
        "eig": e_np,
        "train": train,
        "params_vech": vech_from_s(s_np).numpy().tolist(),
    }


def evaluate_one_sided_vech(pack: dict[str, Any], vech: torch.Tensor, rho: float) -> dict[str, Any]:
    """Score a frozen vech under budget ρ without updating. Same pack/objective as fitting."""
    from prh_replication.anisotropic_kernels import _scores_from_deltas

    p = pack_to_torch(pack)
    qa, qb = int(p["q_a"]), int(p["q_b"])
    db0 = torch.zeros(qb, qb, dtype=torch.float64)
    with torch.no_grad():
        b, s, eig = reconstruct_budgeted(vech.to(dtype=torch.float64), qa, rho)
        da = b - torch.eye(qa, dtype=b.dtype)
        sc = _scores_from_deltas(da, db0, p)
    rec = _record_from_tensors("eval", b, s, eig, sc, rho, qa)
    rec["feasible"] = rec["d"] <= float(rho) + 1e-8
    rec["eig_ok"] = float(np.min(rec["eig"])) >= EIG_LO - 1e-8 and float(np.max(rec["eig"])) <= EIG_HI + 1e-8
    return rec


def _consider(best: dict | None, rec: dict) -> dict:
    if rec is None or not math.isfinite(rec.get("train_excess", float("nan"))):
        return best
    if best is None or rec["train_excess"] > best["train_excess"] + 1e-15:
        return rec
    if abs(rec["train_excess"] - best["train_excess"]) <= 1e-15 and rec["d"] < best["d"]:
        return rec
    return best


def fit_one_sided(
    pack: dict[str, Any],
    rho: float,
    n_steps: int = 120,
    lr: float = 0.03,
    incumbents: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """One-sided fit with nested feasible incumbents and best-iterate retention.

    ρ is an upper bound on D(M). Smaller-budget S matrices remain feasible
    and are scored without update. Adam may not improve them; they are kept.
    """
    from prh_replication.anisotropic_kernels import _scores_from_deltas

    p = pack_to_torch(pack)
    qa, qb = int(p["q_a"]), int(p["q_b"])
    db0 = torch.zeros(qb, qb, dtype=torch.float64)
    id_scores = scores_numpy(*identity_deltas(qa, qb), pack)
    id_rec = {
        "start": "identity_explicit",
        "ok": True,
        "train_excess": id_scores["excess"],
        "d": 0.0,
        "rho": float(rho),
        "b": np.eye(qa),
        "s": np.zeros((qa, qa)),
        "eig": np.ones(qa),
        "train": id_scores,
        "params_vech": [0.0] * n_params("full", qa),
    }
    if float(rho) <= 0.0:
        diag = metric_diagnostics(np.eye(qa), np.zeros((qa, qa)), np.ones(qa))
        return {
            "rho": 0.0,
            "ok": True,
            "selected_start": "identity_exact",
            "train": id_scores,
            "b_a": np.eye(qa),
            "s_a": np.zeros((qa, qa)),
            "eig_a": np.ones(qa),
            "d_attained": 0.0,
            "diag_a": diag,
            "identity_excess": id_scores["excess"],
            "params_vech": id_rec["params_vech"],
            "n_starts": 0,
        }

    best: dict | None = id_rec
    # Frozen incumbents (e.g. smaller-budget solutions): evaluate only.
    for i, s_inc in enumerate(incumbents or []):
        vech = vech_from_s(s_inc)
        rec = evaluate_one_sided_vech(pack, vech, rho)
        rec["start"] = f"incumbent_{i}"
        best = _consider(best, rec)

    continuation = vech_from_s(incumbents[-1]) if incumbents else None
    starts = []
    if continuation is not None:
        starts.append(("continuation", continuation))
    starts.append(("identity", torch.zeros(n_params("full", qa), dtype=torch.float64)))
    starts.append(("pert0", _init_vech(qa, "pert", 11)))
    n_opt = 0
    for name, init in starts:
        params = init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=lr)
        n_opt += 1
        # iteration zero
        rec0 = evaluate_one_sided_vech(pack, params.detach(), rho)
        rec0["start"] = f"{name}_iter0"
        best = _consider(best, rec0)
        for step in range(n_steps):
            opt.zero_grad()
            try:
                b, s, eig = reconstruct_budgeted(params, qa, rho)
                da = b - torch.eye(qa, dtype=b.dtype)
                sc = _scores_from_deltas(da, db0, p)
                obj = sc["excess"]
            except RuntimeError:
                break
            if not torch.isfinite(obj):
                break
            if obj.requires_grad:
                (-obj).backward()
                torch.nn.utils.clip_grad_norm_([params], 5.0)
                opt.step()
            else:
                # Repeated-eigenvalue / disconnected graph at identity: do not discard the point.
                break
            rec_s = evaluate_one_sided_vech(pack, params.detach(), rho)
            rec_s["start"] = f"{name}_iter{step + 1}"
            best = _consider(best, rec_s)

    assert best is not None
    diag = metric_diagnostics(best["b"], best["s"], best["eig"])
    return {
        "rho": float(rho),
        "ok": True,
        "selected_start": best["start"],
        "train": best["train"],
        "b_a": np.asarray(best["b"], dtype=np.float64),
        "s_a": np.asarray(best["s"], dtype=np.float64),
        "eig_a": np.asarray(best["eig"], dtype=np.float64),
        "d_attained": float(best["d"]),
        "diag_a": diag,
        "identity_excess": id_scores["excess"],
        "params_vech": best.get("params_vech"),
        "n_starts": n_opt,
    }


def fit_one_sided_shared(
    packs: list[dict[str, Any]],
    rho: float,
    n_steps: int = 120,
    lr: float = 0.03,
    incumbents: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Equal-weight mean train excess; nested incumbents and best-iterate retention."""
    from prh_replication.anisotropic_kernels import _scores_from_deltas

    if not packs:
        raise ValueError("no packs")
    ts = [pack_to_torch(p) for p in packs]
    qa = int(ts[0]["q_a"])
    id_mean = float(np.mean([scores_numpy(*identity_deltas(int(p["q_a"]), int(p["q_b"])), p)["excess"] for p in packs]))
    if float(rho) <= 0.0:
        return {
            "rho": 0.0,
            "ok": True,
            "selected_start": "identity_exact",
            "train_mean_excess": id_mean,
            "b_a": np.eye(qa),
            "s_a": np.zeros((qa, qa)),
            "eig_a": np.ones(qa),
            "d_attained": 0.0,
        }

    def score_vech(vech: torch.Tensor, start: str) -> dict[str, Any]:
        with torch.no_grad():
            b, s, eig = reconstruct_budgeted(vech.to(dtype=torch.float64), qa, rho)
            da = b - torch.eye(qa, dtype=b.dtype)
            parts = []
            for p in ts:
                db0 = torch.zeros(int(p["q_b"]), int(p["q_b"]), dtype=torch.float64)
                parts.append(float(_scores_from_deltas(da, db0, p)["excess"].detach()))
            mean_ex = float(np.mean(parts))
            s_np = s.cpu().numpy()
        return {
            "start": start,
            "ok": True,
            "train_mean_excess": mean_ex,
            "train_excess": mean_ex,
            "d": float(np.sum(s_np ** 2) / qa),
            "b": b.cpu().numpy(),
            "s": s_np,
            "eig": eig.cpu().numpy(),
        }

    best = {
        "start": "identity_explicit",
        "ok": True,
        "train_mean_excess": id_mean,
        "train_excess": id_mean,
        "d": 0.0,
        "b": np.eye(qa),
        "s": np.zeros((qa, qa)),
        "eig": np.ones(qa),
    }
    for i, s_inc in enumerate(incumbents or []):
        best = _consider(best, score_vech(vech_from_s(s_inc), f"incumbent_{i}"))
    continuation = vech_from_s(incumbents[-1]) if incumbents else None
    starts = []
    if continuation is not None:
        starts.append(("continuation", continuation))
    starts.append(("identity", torch.zeros(n_params("full", qa), dtype=torch.float64)))
    starts.append(("pert0", _init_vech(qa, "pert", 11)))
    for name, init in starts:
        params = init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=lr)
        best = _consider(best, score_vech(params.detach(), f"{name}_iter0"))
        for step in range(n_steps):
            opt.zero_grad()
            try:
                b, s, eig = reconstruct_budgeted(params, qa, rho)
                da = b - torch.eye(qa, dtype=b.dtype)
                parts = []
                for p in ts:
                    db0 = torch.zeros(int(p["q_b"]), int(p["q_b"]), dtype=torch.float64)
                    parts.append(_scores_from_deltas(da, db0, p)["excess"])
                obj = torch.stack(parts).mean()
            except RuntimeError:
                break
            if not torch.isfinite(obj):
                break
            if not obj.requires_grad:
                break
            (-obj).backward()
            torch.nn.utils.clip_grad_norm_([params], 5.0)
            opt.step()
            best = _consider(best, score_vech(params.detach(), f"{name}_iter{step + 1}"))
    return {
        "rho": float(rho),
        "ok": True,
        "selected_start": best["start"],
        "train_mean_excess": best["train_mean_excess"],
        "b_a": np.asarray(best["b"], dtype=np.float64),
        "s_a": np.asarray(best["s"], dtype=np.float64),
        "eig_a": np.asarray(best["eig"], dtype=np.float64),
        "d_attained": float(best["d"]),
    }


def select_rho_by_val(candidates: dict[float, dict], pack_val: dict, key: str = "excess") -> tuple[float, dict]:
    """Headline operating point: max unpenalised val excess; ties prefer lower rho then lower D."""
    ranked = []
    for rho, fit in candidates.items():
        ba = fit["b_a"]
        sc = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_val["q_b"], pack_val["q_b"])), pack_val)
        val = sc[key]
        d = fit["d_attained"]
        ranked.append((val if math.isfinite(val) else -1e18, -float(rho), -d, rho, fit, sc))
    ranked.sort(reverse=True)
    val, _, _, rho, fit, sc = ranked[0]
    out = dict(fit)
    out["selected_rho"] = rho
    out["val"] = sc
    out["val_unpenalised"] = val
    return rho, out


def select_rho_shared(candidates: dict[float, dict], packs_val: list[dict]) -> tuple[float, dict]:
    ranked = []
    for rho, fit in candidates.items():
        ba = fit["b_a"]
        vals = []
        for pv in packs_val:
            sc = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pv["q_b"], pv["q_b"])), pv)
            vals.append(sc["excess"])
        val = float(np.mean(vals)) if vals else float("nan")
        ranked.append((val if math.isfinite(val) else -1e18, -float(rho), -fit["d_attained"], rho, fit))
    ranked.sort(reverse=True)
    val, _, _, rho, fit = ranked[0]
    out = dict(fit)
    out["selected_rho"] = rho
    out["val_mean_excess"] = val
    return rho, out


def match_response_directions(
    z_train_a: np.ndarray,
    z_train_b: np.ndarray,
    z_test_a: np.ndarray,
    z_test_b: np.ndarray,
    u_a: np.ndarray,
    u_b: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    k: int = 4,
) -> dict[str, Any]:
    """Match upweighted U-coordinates using train responses; score on test. Sign-invariant."""
    def top_vecs(s, k):
        evals, vecs = np.linalg.eigh(0.5 * (to_numpy64(s) + to_numpy64(s).T))
        order = np.argsort(evals)[::-1]
        pos = [i for i in order if evals[i] > 1e-8]
        if not pos:
            return None, evals
        take = pos[:k]
        return vecs[:, take], evals[take]

    va, ea = top_vecs(s_a, k)
    vb, eb = top_vecs(s_b, k)
    if va is None or vb is None:
        return {"defined": False, "reason": "no upweighted directions"}
    k_use = min(va.shape[1], vb.shape[1])
    va, vb = va[:, :k_use], vb[:, :k_use]
    ra = to_numpy64(z_train_a) @ to_numpy64(u_a) @ va
    rb = to_numpy64(z_train_b) @ to_numpy64(u_b) @ vb
    # Hungarian via greedy |cosine| matching (k is tiny)
    used = set()
    pairs = []
    for i in range(k_use):
        best_j, best_c = None, -1.0
        ni = np.linalg.norm(ra[:, i])
        if ni <= NEAR_ZERO_S:
            continue
        for j in range(k_use):
            if j in used:
                continue
            nj = np.linalg.norm(rb[:, j])
            if nj <= NEAR_ZERO_S:
                continue
            c = abs(float(np.dot(ra[:, i], rb[:, j]) / (ni * nj)))
            if c > best_c:
                best_c, best_j = c, j
        if best_j is None:
            continue
        used.add(best_j)
        sign = np.sign(np.dot(ra[:, i], rb[:, best_j])) or 1.0
        pairs.append((i, best_j, sign, best_c))
    test_cos = []
    ta = to_numpy64(z_test_a) @ to_numpy64(u_a) @ va
    tb = to_numpy64(z_test_b) @ to_numpy64(u_b) @ vb
    for i, j, sign, _ in pairs:
        ni = np.linalg.norm(ta[:, i])
        nj = np.linalg.norm(tb[:, j])
        if ni <= NEAR_ZERO_S or nj <= NEAR_ZERO_S:
            test_cos.append(float("nan"))
        else:
            test_cos.append(float(sign * np.dot(ta[:, i], tb[:, j]) / (ni * nj)))
    return {
        "defined": True,
        "k": k_use,
        "train_abs_cos": [p[3] for p in pairs],
        "test_signed_cos": test_cos,
        "mean_test_abs_cos": float(np.nanmean(np.abs(test_cos))) if test_cos else float("nan"),
        "note": "Train matching only; test is held-out examples. Sign ambiguity allowed. Not a concept identifier.",
    }


def two_sided_max_budget(pack: dict[str, Any], n_steps: int = 120) -> dict[str, Any]:
    """Bridge to the previous experiment: two-sided full metric, spectral cap only.

    At ρ=(log 4)² the Frobenius budget is implied by the spectral bounds.
    """
    id_sc = scores_numpy(*identity_deltas(pack["q_a"], pack["q_b"]), pack)
    lin = {"a": id_sc["a"], "b": id_sc["b"], "ratio": id_sc["ratio"], "excess": id_sc["excess"]}
    return fit_metrics(pack, family="full", objective="excess", tau=0.0, lin=lin, n_steps=n_steps)

