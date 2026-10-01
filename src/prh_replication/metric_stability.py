"""Budgeted anisotropic metrics: modes, projected gradients, stability.

``fit_anisotropic`` never sees validation or exploratory-test arrays.
``run_a_only_stream`` may score a validation gallery for checkpoint
selection only. It does not take exploratory-test rows, and it does not
keep or reject an update because a sampled subset scored lower.
Historical ``fit_one_sided`` is not used: its identity-start gradient dies
inside ``eigh``. Here ``B = exp(S)`` is ``torch.matrix_exp``, and spectral
projection runs under ``no_grad``.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np
import torch

from prh_replication.anisotropic_kernels import (
    LOG4,
    Q_DEFAULT,
    _unvech,
    fit_pca,
    knn_from_sq,
    make_pack,
    metric_diagnostics,
    metric_gram,
    metric_sq_distances,
    mnn_from_knn,
    n_params,
    pack_to_torch,
    principal_angles,
    sample_center,
    scores_numpy,
    to_numpy64,
)
from prh_replication.anisotropic_kernels import _scores_from_deltas
from prh_replication.kernels import REL_CENTER, extension_stats
from prh_replication.release_anisotropy import (
    NEAR_ZERO_S,
    decompose_s,
    project_S,
    vech_from_s,
)

MODES = ("identity", "a_only", "b_only", "separate", "shared_pc")
ACCOUNTING = ("equal_total", "per_side")
MONOTONE_ATOL = 1e-8
SPECTRAL_RHO = 10.0  # above (log 4)^2, so project_S only clips eigenvalues
SIGN_CONVENTION = "largest_abs_entry_nonnegative"


def coco_image_id(sample_id: str) -> str:
    prefix = "coco-val2017-"
    body = sample_id[len(prefix) :] if sample_id.startswith(prefix) else ""
    if not body.isdigit():
        raise ValueError(f"not a COCO image sample id: {sample_id}")
    return body


def assert_split_image_disjoint(splits: dict[str, list[str]]) -> dict[str, Any]:
    images: dict[str, list[str]] = {}
    for name, ids in splits.items():
        im = [coco_image_id(s) for s in ids]
        if len(im) != len(set(im)):
            raise ValueError(f"duplicate image id in split {name}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample id in split {name}")
        images[name] = im
    names = list(images)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            n_inter = len(set(images[left]) & set(images[right]))
            overlaps[f"{left}&{right}"] = n_inter
            if n_inter:
                raise ValueError(f"image overlap {left} vs {right}: {n_inter}")
    return {"n": {k: len(v) for k, v in splits.items()}, "overlaps": overlaps}


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def canonical_signs(u: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-magnitude entry is nonnegative."""
    out = np.array(to_numpy64(u), copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0:
            out[:, j] *= -1.0
    return out


def fit_basis(x: np.ndarray, q: int = Q_DEFAULT) -> dict[str, Any]:
    """Mean and signed PCA from these rows only. ``x`` is already clip-then-L2."""
    x = to_numpy64(x)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("basis requires a 2d sample with n>=2")
    mu = x.mean(axis=0)
    rec = fit_pca(x - mu, q=q)
    u = canonical_signs(rec["U"])
    basis = {
        "mu": mu,
        "U": u,
        "q": int(rec["q"]),
        "q_requested": int(q),
        "numerical_rank": int(rec["numerical_rank"]),
        "variance_fraction": float(rec["variance_fraction"]),
        "reduced": bool(rec["reduced"] or rec["q"] != q),
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "sign_convention": SIGN_CONVENTION,
        "singular_values_head": to_numpy64(rec["singular_values"][: min(8, rec["singular_values"].size)]),
    }
    basis["id"] = _basis_id(basis)
    return basis


def _basis_id(basis: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(basis["mu"], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(basis["U"], dtype=np.float64).tobytes())
    h.update(basis["sign_convention"].encode())
    return h.hexdigest()[:16]


def haar_orthogonal(q: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=(q, q))
    q_mat, r = np.linalg.qr(g)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q_mat * signs


def rotation_matrices(q: int, seed: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [haar_orthogonal(q, rng) for _ in range(n)]


def with_rotated_basis(basis: dict[str, Any], rotation: np.ndarray) -> dict[str, Any]:
    """Same subspace and mean; columns are U R. Signs are not re-canonicalised."""
    out = dict(basis)
    out["U"] = to_numpy64(basis["U"]) @ to_numpy64(rotation)
    out["id"] = _basis_id(out)
    out["rotation_applied"] = True
    return out


def conjugate_s(s: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """U' = U R implies S' = Rᵀ S R so that U' exp(S') U'ᵀ = U exp(S) Uᵀ."""
    r = to_numpy64(rotation)
    out = r.T @ to_numpy64(s) @ r
    return 0.5 * (out + out.T)


def nested_sequences(ids: list[str], sizes: list[int], n_seq: int, seed: int) -> list[dict[int, list[str]]]:
    """Nested subset IDs. At ``len(ids)`` every sequence is the full list, original order."""
    n = len(ids)
    if len(set(ids)) != n:
        raise ValueError("subset ids are not unique")
    out = []
    for seq in range(n_seq):
        rng = np.random.default_rng(int(seed) + int(seq))
        perm = rng.permutation(n)
        one: dict[int, list[str]] = {}
        prev: set[str] | None = None
        for sz in sorted(int(s) for s in sizes):
            if sz > n or sz < 2:
                raise ValueError(f"bad subset size {sz}")
            chosen = list(ids) if sz == n else [ids[int(i)] for i in perm[:sz]]
            if prev is not None and not prev <= set(chosen):
                raise RuntimeError("nested subset construction failed")
            one[sz] = chosen
            prev = set(chosen)
        out.append(one)
    return out


def _spectral_clip(s: torch.Tensor) -> torch.Tensor:
    return project_S(s, SPECTRAL_RHO, int(s.shape[0]))


def _scale_to(s: torch.Tensor, rho: float, q: int) -> torch.Tensor:
    fro2 = (s * s).sum()
    budget = float(rho) * float(q)
    if float(fro2) > budget and float(fro2) > 0.0:
        s = s * torch.sqrt(s.new_tensor(budget) / fro2)
    return s


def project_matrices(
    s_a: torch.Tensor,
    s_b: torch.Tensor,
    *,
    rho: float,
    accounting: str,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spectral clip, then distortion scale. No autograd."""
    if mode not in MODES:
        raise ValueError(mode)
    if accounting not in ACCOUNTING:
        raise ValueError(accounting)
    with torch.no_grad():
        tied = mode == "shared_pc"
        active_a = mode in ("a_only", "separate", "shared_pc")
        active_b = mode in ("b_only", "separate", "shared_pc")
        if active_a:
            s_a = _spectral_clip(0.5 * (s_a + s_a.T))
        else:
            s_a = torch.zeros_like(s_a)
        if tied:
            s_b = s_a.clone()
        elif active_b:
            s_b = _spectral_clip(0.5 * (s_b + s_b.T))
        else:
            s_b = torch.zeros_like(s_b)
        if float(rho) <= 0.0 or mode == "identity":
            return torch.zeros_like(s_a), torch.zeros_like(s_b)
        qa, qb = int(s_a.shape[0]), int(s_b.shape[0])
        if accounting == "per_side":
            if active_a:
                s_a = _scale_to(s_a, rho, qa)
            if tied:
                s_b = s_a.clone()
            elif active_b:
                s_b = _scale_to(s_b, rho, qb)
        else:
            da = (s_a * s_a).sum() / qa
            db = (s_b * s_b).sum() / qb
            total = da + db
            if float(total) > float(rho) and float(total) > 0.0:
                alpha = torch.sqrt(total.new_tensor(float(rho)) / total)
                s_a = s_a * alpha
                s_b = s_a.clone() if tied else s_b * alpha
        s_a = 0.5 * (s_a + s_a.T)
        s_b = s_a.clone() if tied else 0.5 * (s_b + s_b.T)
        return s_a, s_b


def b_matrix_exp(s: np.ndarray | torch.Tensor) -> np.ndarray:
    t = torch.as_tensor(np.asarray(to_numpy64(s)), dtype=torch.float64)
    with torch.no_grad():
        b = torch.matrix_exp(0.5 * (t + t.T))
        b = 0.5 * (b + b.T)
    return b.detach().cpu().numpy()


def distortions(s_a: np.ndarray, s_b: np.ndarray) -> tuple[float, float, float]:
    da = float(np.sum(s_a * s_a) / s_a.shape[0])
    db = float(np.sum(s_b * s_b) / s_b.shape[0])
    return da, db, da + db


def constraint_ok(s_a: np.ndarray, s_b: np.ndarray, rho: float, accounting: str, mode: str) -> bool:
    da, db, total = distortions(s_a, s_b)
    tol = 1e-7
    if mode == "identity" or float(rho) <= 0.0:
        budget_ok = da <= tol and db <= tol
    elif accounting == "per_side":
        budget_ok = da <= float(rho) + tol and db <= float(rho) + tol
    else:
        budget_ok = total <= float(rho) + tol
    for s in (s_a, s_b):
        ev = np.linalg.eigvalsh(0.5 * (s + s.T))
        if float(ev.min()) < -LOG4 - 1e-6 or float(ev.max()) > LOG4 + 1e-6:
            return False
    return budget_ok


def centred_features_degenerate(x: np.ndarray, mu: np.ndarray) -> bool:
    """True when sample-centring leaves only numerical dust.

    Contracted CKA on that dust is an indeterminate 0/0 that can print as 1.
    The check matches the relative centred-norm idea in ``extension_stats``
    and does not change that historical function.
    """
    z = to_numpy64(x) - to_numpy64(mu)
    zc = sample_center(z)
    nz = float(np.linalg.norm(z))
    nc = float(np.linalg.norm(zc))
    if not np.isfinite(nz) or not np.isfinite(nc):
        return True
    return nc <= max(REL_CENTER * (nz + 1e-12), 1e-10)


def _packs(features_a, features_b, basis_a, basis_b) -> list[dict[str, Any]]:
    xs = features_a
    if isinstance(features_b, np.ndarray):
        features_b = [features_b]
        basis_b = [basis_b]
    if len(features_b) != len(basis_b):
        raise ValueError("partner features and bases differ in length")
    packs = []
    for y, by in zip(features_b, basis_b):
        if xs.shape[0] != y.shape[0]:
            raise ValueError("paired sample counts must match")
        za = to_numpy64(xs) - to_numpy64(basis_a["mu"])
        zb = to_numpy64(y) - to_numpy64(by["mu"])
        packs.append(make_pack(za, zb, basis_a["U"], by["U"]))
    return packs


def _score_sides(s_a: np.ndarray, s_b: np.ndarray, packs: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ba = b_matrix_exp(s_a)
    bb = b_matrix_exp(s_b)
    if mode in ("identity", "a_only"):
        bb = np.eye(s_b.shape[0])
        s_b = np.zeros_like(s_b)
    elif mode == "b_only":
        ba = np.eye(s_a.shape[0])
        s_a = np.zeros_like(s_a)
    parts = []
    for p in packs:
        da = ba - np.eye(ba.shape[0])
        db = bb - np.eye(bb.shape[0])
        if da.shape[0] != int(p["q_a"]) or db.shape[0] != int(p["q_b"]):
            raise ValueError("S dimension does not match the pack")
        parts.append(scores_numpy(da, db, p))
    excesses = [float(r["excess"]) for r in parts]
    mean_ex = float(np.mean(excesses)) if excesses else float("nan")
    return {
        "partners": parts,
        "excess": mean_ex,
        "a": float(np.mean([r["a"] for r in parts])),
        "b": float(np.mean([r["b"] for r in parts])),
        "ratio": float(np.nanmean([r["ratio"] for r in parts])),
        "degenerate": any(bool(r["degenerate"]) for r in parts),
        "valid_a": all(bool(r["valid_a"]) for r in parts),
        "b_a": ba,
        "b_b": bb,
        "s_a": s_a,
        "s_b": s_b,
    }


def _excess_torch(s_a: torch.Tensor, s_b: torch.Tensor, packs_t: list[dict[str, Any]], mode: str) -> torch.Tensor:
    qa, qb = int(s_a.shape[0]), int(s_b.shape[0])
    eye_a = torch.eye(qa, dtype=torch.float64)
    eye_b = torch.eye(qb, dtype=torch.float64)
    if mode == "shared_pc":
        b = torch.matrix_exp(s_a)
        da = b - eye_a
        db = b - eye_b
    elif mode == "a_only":
        da = torch.matrix_exp(s_a) - eye_a
        db = torch.zeros(qb, qb, dtype=torch.float64)
    elif mode == "b_only":
        da = torch.zeros(qa, qa, dtype=torch.float64)
        db = torch.matrix_exp(s_b) - eye_b
    elif mode == "separate":
        da = torch.matrix_exp(s_a) - eye_a
        db = torch.matrix_exp(s_b) - eye_b
    else:
        raise ValueError(mode)
    vals = []
    for p in packs_t:
        vals.append(_scores_from_deltas(da, db, p)["excess"])
    return torch.stack(vals).mean()


def _grad_to_vech(g: torch.Tensor) -> torch.Tensor:
    g = 0.5 * (g + g.T)
    i, j = torch.triu_indices(g.shape[0], g.shape[0])
    scale = torch.where(i == j, torch.ones((), dtype=torch.float64), torch.full((), 2.0, dtype=torch.float64))
    return scale * g[i, j]


def gradient_at_vech(
    features_a: np.ndarray,
    features_b: np.ndarray | list[np.ndarray],
    basis_a: dict[str, Any],
    basis_b: dict[str, Any] | list[dict[str, Any]],
    vech_a: np.ndarray,
    vech_b: np.ndarray | None,
    *,
    mode: str,
    rho: float,
    accounting: str = "equal_total",
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Euclidean gradient of mean excess at the projected point, pulled back to vech."""
    packs = _packs(features_a, features_b, basis_a, basis_b)
    packs_t = [pack_to_torch(p) for p in packs]
    qa = int(packs[0]["q_a"])
    qb = int(packs[0]["q_b"])
    raw_a = _unvech(torch.tensor(vech_a, dtype=torch.float64), qa)
    raw_b = _unvech(torch.tensor(vech_b if vech_b is not None else np.zeros(n_params("full", qb)), dtype=torch.float64), qb)
    s_a, s_b = project_matrices(raw_a, raw_b, rho=rho, accounting=accounting, mode=mode)
    active_a = mode in ("a_only", "separate", "shared_pc")
    active_b = mode in ("b_only", "separate")
    s_a_v = s_a.detach().requires_grad_(True) if active_a else s_a.detach()
    s_b_v = s_a_v if mode == "shared_pc" else (s_b.detach().requires_grad_(True) if active_b else s_b.detach())
    obj = _excess_torch(s_a_v, s_b_v, packs_t, mode)
    if not torch.isfinite(obj):
        return np.full(vech_a.shape, np.nan), None, float("nan")
    if not obj.requires_grad:
        z = np.zeros(n_params("full", qa))
        return z, (np.zeros(n_params("full", qb)) if active_b else None), float(obj.detach())
    obj.backward()
    g_a = _grad_to_vech(s_a_v.grad).detach().cpu().numpy() if s_a_v.grad is not None else np.zeros(n_params("full", qa))
    g_b = None
    if active_b:
        g_b = _grad_to_vech(s_b_v.grad).detach().cpu().numpy() if s_b_v.grad is not None else np.zeros(n_params("full", qb))
    return g_a, g_b, float(obj.detach())


def _init_vech(q: int, seed: int, scale: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return float(scale) * torch.randn(n_params("full", q), dtype=torch.float64, generator=g)


def _as_np(s: torch.Tensor) -> np.ndarray:
    return s.detach().cpu().numpy().astype(np.float64, copy=False)


def fit_anisotropic(
    features_a: np.ndarray,
    features_b: np.ndarray | list[np.ndarray],
    ids: list[str],
    basis_a: dict[str, Any],
    basis_b: dict[str, Any] | list[dict[str, Any]],
    *,
    mode: str,
    objective: str = "excess",
    rho: float,
    accounting: str = "equal_total",
    init_seed: int = 11,
    incumbents: list[dict[str, np.ndarray]] | None = None,
    n_steps: int = 120,
    lr: float = 0.03,
    grad_clip: float = 5.0,
    pert_scale: float = 0.05,
    pack_cache: dict | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Fit one mode. ``features_*`` are the training rows only."""
    if objective != "excess":
        raise ValueError("this sweep maximises excess only")
    if mode not in MODES:
        raise ValueError(mode)
    if accounting not in ACCOUNTING:
        raise ValueError(accounting)
    if isinstance(features_b, list) and mode != "a_only":
        raise ValueError("multiple partners are supported for A-only mean excess")
    if pack_cache is not None and cache_key is not None and cache_key in pack_cache:
        packs = pack_cache[cache_key]
    else:
        packs = _packs(features_a, features_b, basis_a, basis_b)
        if pack_cache is not None and cache_key is not None:
            pack_cache[cache_key] = packs
    if len(ids) != int(packs[0]["n"]):
        raise ValueError("sample ids do not match the fitting rows")
    qa, qb = int(packs[0]["q_a"]), int(packs[0]["q_b"])
    if mode == "shared_pc" and qa != qb:
        return {
            "ok": False,
            "skipped": True,
            "reason": "shared_pc requires equal PCA dimension",
            "q_a": qa,
            "q_b": qb,
            "mode": mode,
            "rho": float(rho),
        }
    bases_b = basis_b if isinstance(basis_b, list) else [basis_b]
    if any(int(b["q"]) != qb for b in bases_b) or int(basis_a["q"]) != qa:
        raise ValueError("basis q does not match the pack")

    def zeros():
        return np.zeros((qa, qa)), np.zeros((qb, qb))

    id_s = zeros()
    rows_b = features_b if isinstance(features_b, list) else [features_b]
    bases_chk = basis_b if isinstance(basis_b, list) else [basis_b]
    rows_bad = centred_features_degenerate(features_a, basis_a["mu"]) or any(
        centred_features_degenerate(y, b["mu"]) for y, b in zip(rows_b, bases_chk)
    )
    ident = _score_sides(*id_s, packs, "identity")
    if rows_bad:
        ident["excess"] = float("nan")
        ident["a"] = float("nan")
        ident["b"] = float("nan")
        ident["ratio"] = float("nan")
        ident["degenerate"] = True
        ident["valid_a"] = False
    status = {
        "nonfinite_objective": 0,
        "nonfinite_grad": 0,
        "no_grad": 0,
        "zero_grad_at_identity": False,
        "constraint_violation": False,
        "degenerate_train": bool(rows_bad or ident["degenerate"] or not ident["valid_a"]),
        "starts": [],
    }
    if mode == "identity" or float(rho) <= 0.0 or status["degenerate_train"]:
        rec = _pack_result(
            mode=mode,
            rho=float(rho),
            accounting=accounting,
            s_a=id_s[0],
            s_b=id_s[1],
            train=ident,
            selected_start="identity_exact" if not status["degenerate_train"] else "identity_degenerate",
            status=status,
            ids=ids,
            basis_a=basis_a,
            basis_b=bases_b,
            n_steps=0,
            lr=lr,
            init_seed=init_seed,
        )
        return rec

    packs_t = [pack_to_torch(p) for p in packs]
    best: dict[str, Any] | None = None

    def consider(s_a_np, s_b_np, start: str, iteration: int):
        nonlocal best
        if not constraint_ok(s_a_np, s_b_np, rho, accounting, mode):
            status["constraint_violation"] = True
        scored = _score_sides(s_a_np, s_b_np, packs, mode)
        ex = scored["excess"]
        if not math.isfinite(ex):
            status["nonfinite_objective"] += 1
            return
        da, db, total = distortions(scored["s_a"], scored["s_b"])
        cand = {
            "start": start,
            "iteration": iteration,
            "excess": ex,
            "d_total": total,
            "scored": scored,
        }
        if best is None or ex > best["excess"] + 1e-15 or (abs(ex - best["excess"]) <= 1e-15 and total < best["d_total"]):
            best = cand

    consider(*id_s, "identity_explicit", 0)
    for i, inc in enumerate(incumbents or []):
        sa = np.asarray(inc["s_a"], dtype=np.float64)
        sb = np.asarray(inc.get("s_b", np.zeros((qb, qb))), dtype=np.float64)
        if sa.shape != (qa, qa) or sb.shape != (qb, qb):
            raise ValueError("incumbent shape does not match q")
        psa, psb = project_matrices(
            torch.tensor(sa, dtype=torch.float64),
            torch.tensor(sb, dtype=torch.float64),
            rho=rho,
            accounting=accounting,
            mode=mode,
        )
        consider(_as_np(psa), _as_np(psb), f"incumbent_{i}", 0)

    def vech_pair(sa: np.ndarray, sb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return vech_from_s(sa), vech_from_s(sb)

    starts: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    if incumbents:
        last = incumbents[-1]
        sa = np.asarray(last["s_a"], dtype=np.float64)
        sb = np.asarray(last.get("s_b", np.zeros((qb, qb))), dtype=np.float64)
        va, vb = vech_pair(sa, sb)
        starts.append(("continuation", va, vb))
    starts.append(("identity", torch.zeros(n_params("full", qa), dtype=torch.float64), torch.zeros(n_params("full", qb), dtype=torch.float64)))
    starts.append(("pert", _init_vech(qa, init_seed, pert_scale), _init_vech(qb, init_seed + 1, pert_scale)))
    if not incumbents:
        starts.append(("pert2", _init_vech(qa, init_seed + 1000, pert_scale), _init_vech(qb, init_seed + 1001, pert_scale)))

    active_a = mode in ("a_only", "separate", "shared_pc")
    active_b = mode in ("b_only", "separate")
    for name, va0, vb0 in starts:
        params = []
        if active_a:
            pa = va0.clone().detach().requires_grad_(True)
            params.append(pa)
        else:
            pa = va0.clone().detach()
        if active_b:
            pb = vb0.clone().detach().requires_grad_(True)
            params.append(pb)
        else:
            pb = vb0.clone().detach()
        opt = torch.optim.Adam(params, lr=lr) if params else None
        trace = []
        grad_norm0 = None
        stopped = "completed"
        for step in range(n_steps + 1):
            raw_a = _unvech(pa.detach(), qa)
            raw_b = _unvech(pb.detach(), qb)
            s_a, s_b = project_matrices(raw_a, raw_b, rho=rho, accounting=accounting, mode=mode)
            s_a_np, s_b_np = _as_np(s_a), _as_np(s_b)
            consider(s_a_np, s_b_np, name, step)
            scored_now = _score_sides(s_a_np, s_b_np, packs, mode)
            trace.append(scored_now["excess"])
            if step == n_steps:
                break
            if opt is None:
                stopped = "no_parameters"
                break
            opt.zero_grad(set_to_none=True)
            s_a_v = s_a.detach().requires_grad_(True) if active_a else s_a.detach()
            s_b_v = s_a_v if mode == "shared_pc" else (s_b.detach().requires_grad_(True) if active_b else s_b.detach())
            try:
                obj = _excess_torch(s_a_v, s_b_v, packs_t, mode)
            except RuntimeError:
                status["nonfinite_objective"] += 1
                stopped = "runtime"
                break
            if not torch.isfinite(obj):
                status["nonfinite_objective"] += 1
                stopped = "nonfinite_objective"
                break
            if not obj.requires_grad:
                status["no_grad"] += 1
                stopped = "no_grad"
                if name == "identity":
                    status["zero_grad_at_identity"] = True
                break
            obj.backward()
            if active_a:
                if s_a_v.grad is None or not torch.isfinite(s_a_v.grad).all():
                    status["nonfinite_grad"] += 1
                    stopped = "nonfinite_grad"
                    break
                # Adam descends. Negate so the step ascends excess, matching (-excess).backward().
                pa.grad = -_grad_to_vech(s_a_v.grad)
            if active_b:
                if s_b_v.grad is None or not torch.isfinite(s_b_v.grad).all():
                    status["nonfinite_grad"] += 1
                    stopped = "nonfinite_grad"
                    break
                pb.grad = -_grad_to_vech(s_b_v.grad)
            if name == "identity" and step == 0:
                g0 = pa.grad if active_a else pb.grad
                grad_norm0 = float(torch.linalg.norm(g0))
                if grad_norm0 <= 1e-12:
                    status["zero_grad_at_identity"] = True
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            with torch.no_grad():
                s_a2, s_b2 = project_matrices(
                    _unvech(pa.detach(), qa),
                    _unvech(pb.detach(), qb),
                    rho=rho,
                    accounting=accounting,
                    mode=mode,
                )
                if active_a:
                    pa.copy_(vech_from_s(_as_np(s_a2)))
                if active_b:
                    pb.copy_(vech_from_s(_as_np(s_b2)))
                if mode == "shared_pc" and active_a:
                    pb = pa
        finite_trace = [v for v in trace if isinstance(v, float) and math.isfinite(v)]
        improved = bool(finite_trace) and max(finite_trace) > finite_trace[0] + 1e-8
        status["starts"].append(
            {
                "name": name,
                "stopped": stopped,
                "n_scored": len(trace),
                "iter0_excess": trace[0] if trace else None,
                "best_excess": max(finite_trace) if finite_trace else None,
                "final_excess": finite_trace[-1] if finite_trace else None,
                "improved": improved,
                "identity_grad_norm": grad_norm0,
                "trace": [None if not isinstance(v, float) or not math.isfinite(v) else v for v in trace],
            }
        )

    if best is None:
        raise RuntimeError("no finite feasible iterate")
    status["stalled"] = not any(s["improved"] for s in status["starts"])
    return _pack_result(
        mode=mode,
        rho=float(rho),
        accounting=accounting,
        s_a=best["scored"]["s_a"],
        s_b=best["scored"]["s_b"],
        train=best["scored"],
        selected_start=f"{best['start']}_iter{best['iteration']}",
        status=status,
        ids=ids,
        basis_a=basis_a,
        basis_b=bases_b,
        n_steps=n_steps,
        lr=lr,
        init_seed=init_seed,
    )


def _pack_result(*, mode, rho, accounting, s_a, s_b, train, selected_start, status, ids, basis_a, basis_b, n_steps, lr, init_seed) -> dict[str, Any]:
    s_a = np.asarray(train["s_a"], dtype=np.float64)
    s_b = np.asarray(train["s_b"], dtype=np.float64)
    da, db, total = distortions(s_a, s_b)
    diag_a = metric_diagnostics(train["b_a"], s_a, np.linalg.eigvalsh(train["b_a"]))
    diag_b = metric_diagnostics(train["b_b"], s_b, np.linalg.eigvalsh(train["b_b"]))
    dec_a = decompose_s(s_a)
    dec_b = decompose_s(s_b)
    dec_a.pop("s_tilde", None)
    dec_b.pop("s_tilde", None)
    return {
        "ok": True,
        "skipped": False,
        "mode": mode,
        "objective": "excess",
        "rho": float(rho),
        "accounting": accounting,
        "selected_start": selected_start,
        "train_excess": float(train["excess"]),
        "train_a": float(train["a"]),
        "train_b": float(train["b"]),
        "train_ratio": float(train["ratio"]) if math.isfinite(train["ratio"]) else None,
        "train_degenerate": bool(train["degenerate"]),
        "train_partner_excess": [float(r["excess"]) for r in train["partners"]],
        "d_a": da,
        "d_b": db,
        "d_total": total,
        "feasible": constraint_ok(s_a, s_b, rho, accounting, mode),
        "diag_a": diag_a,
        "diag_b": diag_b,
        "decompose_a": dec_a,
        "decompose_b": dec_b,
        "s_a": s_a,
        "s_b": s_b,
        "eig_b_a": np.linalg.eigvalsh(train["b_a"]),
        "eig_b_b": np.linalg.eigvalsh(train["b_b"]),
        "sample_ids_hash": ids_hash(list(ids)),
        "n": len(ids),
        "basis_a_id": basis_a["id"],
        "basis_b_ids": [b["id"] for b in basis_b],
        "q_a": int(s_a.shape[0]),
        "q_b": int(s_b.shape[0]),
        "n_steps": int(n_steps),
        "lr": float(lr),
        "init_seed": int(init_seed),
        "status": status,
        "numerical_stabilisation": [
            "centred gram with nonpositive norm returns nan excess",
            "nk2 clamped at 0 inside the contracted score",
            "no diagonal jitter",
            "matrix_exp on the gradient path",
            "spectral projection outside the graph",
        ],
    }


def select_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Max validation excess. Ties: lower rho, then lower total distortion.

    Keys beginning with ``test`` are ignored, so exploratory-test numbers
    cannot decide the budget.
    """

    def value(row: dict[str, Any]) -> float:
        val = row.get("val_excess")
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            return -1e300
        return float(val)

    ranked = sorted(rows, key=lambda r: (value(r), -float(r["rho"]), -float(r["d_total"])), reverse=True)
    chosen = dict(ranked[0])
    chosen["selection"] = "max_val_excess_then_lower_rho_then_lower_d"
    return chosen


def evaluate_metric(
    features_a: np.ndarray,
    features_b: np.ndarray,
    basis_a: dict[str, Any],
    basis_b: dict[str, Any],
    s_a: np.ndarray,
    s_b: np.ndarray,
    *,
    knn_k: int = 10,
) -> dict[str, Any]:
    """Score a frozen metric. Sample-centring for CKA matches ``make_pack``."""
    if centred_features_degenerate(features_a, basis_a["mu"]) or centred_features_degenerate(features_b, basis_b["mu"]):
        return {
            "a": float("nan"),
            "b": float("nan"),
            "ratio": None,
            "excess": float("nan"),
            "valid_a": False,
            "degenerate": True,
            "ratio_undefined": True,
            "r_eff_k": float("nan"),
            "r_eff_l": float("nan"),
            "mnn_k10": float("nan"),
            "n": int(np.asarray(features_a).shape[0]),
        }
    za = to_numpy64(features_a) - to_numpy64(basis_a["mu"])
    zb = to_numpy64(features_b) - to_numpy64(basis_b["mu"])
    ba = b_matrix_exp(s_a)
    bb = b_matrix_exp(s_b)
    pack = make_pack(za, zb, basis_a["U"], basis_b["U"])
    sc = scores_numpy(ba - np.eye(ba.shape[0]), bb - np.eye(bb.shape[0]), pack)
    dsq_a = metric_sq_distances(za, basis_a["U"], ba)
    dsq_b = metric_sq_distances(zb, basis_b["U"], bb)
    mnn = mnn_from_knn(knn_from_sq(dsq_a, knn_k), knn_from_sq(dsq_b, knn_k))
    return {
        "a": sc["a"],
        "b": sc["b"],
        "ratio": sc["ratio"] if math.isfinite(sc["ratio"]) else None,
        "excess": sc["excess"],
        "valid_a": sc["valid_a"],
        "degenerate": sc["degenerate"],
        "ratio_undefined": sc["ratio_undefined"],
        "r_eff_k": sc["r_eff_k"],
        "r_eff_l": sc["r_eff_l"],
        "mnn_k10": mnn,
        "n": int(features_a.shape[0]),
    }


def scores_direct_f64(za: np.ndarray, zb: np.ndarray, ua: np.ndarray, ub: np.ndarray, ba: np.ndarray, bb: np.ndarray) -> dict[str, Any]:
    ka = torch.tensor(metric_gram(za, ua, ba), dtype=torch.float64)
    kb = torch.tensor(metric_gram(zb, ub, bb), dtype=torch.float64)
    return extension_stats(ka, kb)


def ambient_apply(u: np.ndarray, b: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = to_numpy64(b) - np.eye(b.shape[0])
    vv = to_numpy64(v)
    return vv + to_numpy64(u) @ (c @ (to_numpy64(u).T @ vv))


def delta_c(b: np.ndarray) -> np.ndarray:
    c = to_numpy64(b) - np.eye(b.shape[0])
    return 0.5 * (c + c.T)


def delta_inner(u1: np.ndarray, c1: np.ndarray, u2: np.ndarray, c2: np.ndarray) -> float:
    w = to_numpy64(u1).T @ to_numpy64(u2)
    return float(np.trace(to_numpy64(c1) @ w @ to_numpy64(c2) @ w.T))


def delta_m_cosine(u1: np.ndarray, b1: np.ndarray, u2: np.ndarray, b2: np.ndarray) -> dict[str, Any]:
    c1, c2 = delta_c(b1), delta_c(b2)
    n1, n2 = float(np.linalg.norm(c1)), float(np.linalg.norm(c2))
    if n1 <= NEAR_ZERO_S or n2 <= NEAR_ZERO_S:
        return {"cosine": None, "distance": None, "nearly_zero": True, "norm_a": n1, "norm_b": n2}
    ip = delta_inner(u1, c1, u2, c2)
    dist = math.sqrt(max(n1 * n1 + n2 * n2 - 2.0 * ip, 0.0))
    return {"cosine": ip / (n1 * n2), "distance": dist, "nearly_zero": False, "norm_a": n1, "norm_b": n2}


def signed_frame(u: np.ndarray, s: np.ndarray, which: str, k: int, eps: float = 1e-8) -> dict[str, Any]:
    sm = 0.5 * (to_numpy64(s) + to_numpy64(s).T)
    if float(np.linalg.norm(sm)) <= NEAR_ZERO_S:
        return {"defined": False, "reason": "nearly_zero", "available": 0}
    evals, vecs = np.linalg.eigh(sm)
    if which == "amplified":
        idx = np.where(evals > eps)[0]
        idx = idx[np.argsort(evals[idx])[::-1]] if idx.size else idx
    elif which == "suppressed":
        idx = np.where(evals < -eps)[0]
        idx = idx[np.argsort(evals[idx])] if idx.size else idx
    else:
        raise ValueError(which)
    if int(idx.size) < int(k):
        return {"defined": False, "reason": "insufficient_signed_rank", "available": int(idx.size)}
    frame = to_numpy64(u) @ vecs[:, idx[:k]]
    return {"defined": True, "frame": frame, "available": int(idx.size), "k": int(k)}


def subspace_compare(u1, s1, u2, s2, which: str, k: int) -> dict[str, Any]:
    f1 = signed_frame(u1, s1, which, k)
    f2 = signed_frame(u2, s2, which, k)
    if not f1["defined"] or not f2["defined"]:
        return {
            "defined": False,
            "reason": f1.get("reason") if not f1["defined"] else f2.get("reason"),
            "available_a": f1["available"],
            "available_b": f2["available"],
            "overlap": None,
            "principal_angles": None,
        }
    angles = principal_angles(f1["frame"], f2["frame"])
    overlap = float(np.mean(np.cos(angles) ** 2))
    return {
        "defined": True,
        "overlap": overlap,
        "principal_angles": angles.tolist(),
        "available_a": f1["available"],
        "available_b": f2["available"],
    }


def centred_kernel(features: np.ndarray, mu: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = to_numpy64(features) - to_numpy64(mu)
    return metric_gram(sample_center(z), u, b)


def gram_cosine(k1: np.ndarray, k2: np.ndarray) -> float | None:
    a = to_numpy64(k1).ravel()
    b = to_numpy64(k2).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= NEAR_ZERO_S or nb <= NEAR_ZERO_S:
        return None
    return float(np.dot(a, b) / (na * nb))


def correction_gram(features: np.ndarray, mu: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    zc = sample_center(to_numpy64(features) - to_numpy64(mu))
    c = delta_c(b)
    p = zc @ to_numpy64(u)
    return p @ c @ p.T


def draw_quarter_indices(n: int, rng: np.random.Generator, fraction: float = 0.25) -> np.ndarray:
    """Uniform subset without replacement. Sorted so row order is the pool order."""
    if int(n) < 4:
        raise ValueError("quarter sampling needs at least 4 rows")
    size = float(n) * float(fraction)
    if abs(size - round(size)) > 1e-8:
        raise ValueError("subset size is not an integer")
    k = int(round(size))
    if k < 2 or k > int(n):
        raise ValueError("subset size must be between 2 and n")
    idx = np.asarray(rng.choice(int(n), size=k, replace=False), dtype=np.int64)
    if len(set(idx.tolist())) != k:
        raise RuntimeError("draw produced duplicate indices")
    return np.sort(idx)


def initial_vech(q: int, init_seed: int, pert_scale: float = 0.05) -> torch.Tensor:
    """Seed 0 is exact identity. Other seeds use the existing perturbation generator."""
    if int(init_seed) == 0:
        return torch.zeros(n_params("full", int(q)), dtype=torch.float64)
    return _init_vech(int(q), int(init_seed), float(pert_scale))


def _selection_better(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
    if old is None:
        return True
    val = float(new["val_excess"])
    old_val = float(old["val_excess"])
    dist = float(new["d_total"])
    old_dist = float(old["d_total"])
    if val > old_val + 1e-15:
        return True
    if abs(val - old_val) <= 1e-15 and dist < old_dist - 1e-12:
        return True
    if abs(val - old_val) <= 1e-15 and abs(dist - old_dist) <= 1e-12 and int(new["step"]) < int(old["step"]):
        return True
    return False


def choose_val_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest full-validation excess, then lower distortion, then earlier step.

    Keys whose names start with ``test`` are never read.
    """
    best: dict[str, Any] | None = None
    for rec in records:
        if "val_excess" not in rec or "d_total" not in rec or "step" not in rec:
            continue
        val = rec["val_excess"]
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        cand = {"step": int(rec["step"]), "val_excess": float(val), "d_total": float(rec["d_total"])}
        if _selection_better(cand, best):
            best = cand
    return best


def dump_rng(rng: np.random.Generator) -> dict[str, int]:
    state = rng.bit_generator.state
    inner = state["state"]
    return {"state": int(inner["state"]), "inc": int(inner["inc"])}


def load_rng(blob: dict[str, int]) -> np.random.Generator:
    bg = np.random.PCG64()
    bg.state = {
        "bit_generator": "PCG64",
        "state": {"state": int(blob["state"]), "inc": int(blob["inc"])},
        "has_uint32": 0,
        "uinteger": 0,
    }
    return np.random.Generator(bg)


def _adam_blob(opt: torch.optim.Optimizer, param: torch.Tensor) -> dict[str, Any] | None:
    state = opt.state.get(param)
    if not state:
        return None
    step = state["step"]
    step_i = int(step.item() if torch.is_tensor(step) else step)
    return {
        "exp_avg": state["exp_avg"].detach().cpu().numpy().copy(),
        "exp_avg_sq": state["exp_avg_sq"].detach().cpu().numpy().copy(),
        "step": step_i,
    }


def _load_adam(opt: torch.optim.Optimizer, param: torch.Tensor, blob: dict[str, Any] | None) -> None:
    if not blob:
        return
    opt.state[param] = {
        "exp_avg": torch.tensor(np.asarray(blob["exp_avg"], dtype=np.float64), dtype=torch.float64),
        "exp_avg_sq": torch.tensor(np.asarray(blob["exp_avg_sq"], dtype=np.float64), dtype=torch.float64),
        "step": torch.tensor(float(blob["step"]), dtype=torch.float64),
    }


def _project_a_only(vech: torch.Tensor, qa: int, qb: int, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
    raw = _unvech(vech.detach(), qa)
    zeros = torch.zeros(qb, qb, dtype=torch.float64)
    return project_matrices(raw, zeros, rho=float(rho), accounting="equal_total", mode="a_only")


def run_a_only_stream(
    features_a: np.ndarray,
    features_b: np.ndarray,
    ids: list[str],
    basis_a: dict[str, Any],
    basis_b: dict[str, Any],
    *,
    rho: float,
    init_seed: int,
    n_updates: int,
    procedure: str,
    sampling_seed: int | None = None,
    lr: float = 0.03,
    grad_clip: float = 5.0,
    pert_scale: float = 0.05,
    val_features_a: np.ndarray | None = None,
    val_features_b: np.ndarray | None = None,
    resume: dict[str, Any] | None = None,
    primary_updates: int | None = None,
    quarter_fraction: float = 0.25,
    record_indices: bool = False,
    on_checkpoint: Any = None,
) -> dict[str, Any]:
    """One persistent language-side S. Partner metric stays identity.

    ``procedure='full'`` differentiates excess on every training row.
    ``procedure='quarter'`` differentiates excess on a fresh uniform subset.
    That subset gradient estimates the derivative of expected subset excess.
    It is not an unbiased estimator of the full-gallery gradient: centring
    and the normalising norms depend on the subset.

    Checkpoint choice uses full-validation excess only. The Adam state always
    continues from the latest feasible matrix.
    """
    if procedure not in ("full", "quarter"):
        raise ValueError(procedure)
    if procedure == "quarter" and sampling_seed is None and resume is None:
        raise ValueError("quarter sampling requires a sampling seed")
    if procedure == "full" and sampling_seed is not None:
        raise ValueError("full-gallery runs do not take a sampling seed")
    n = int(features_a.shape[0])
    if features_b.shape[0] != n or len(ids) != n:
        raise ValueError("training rows and ids must describe the same gallery")
    if len(set(ids)) != n:
        raise ValueError("training ids are not unique")
    primary = int(n_updates if primary_updates is None else primary_updates)
    if primary < 0 or primary > int(n_updates):
        raise ValueError("primary_updates must lie between 0 and n_updates")
    qa, qb = int(basis_a["q"]), int(basis_b["q"])
    train_pack = _packs(features_a, features_b, basis_a, basis_b)[0]
    val_pack = None
    if val_features_a is not None or val_features_b is not None:
        if val_features_a is None or val_features_b is None:
            raise ValueError("validation features must be supplied on both sides")
        val_pack = _packs(val_features_a, val_features_b, basis_a, basis_b)[0]
    zeros_b = np.zeros((qb, qb), dtype=np.float64)

    def score_pack(s_np: np.ndarray, pack: dict[str, Any]) -> dict[str, Any]:
        return _score_sides(s_np, zeros_b, [pack], "a_only")

    snapshots = {
        "s_at_primary": None,
        "s_selected_primary": None,
        "s_at_final": None,
        "s_selected_final": None,
    }
    selected_primary: dict[str, Any] | None = None
    selected_final: dict[str, Any] | None = None
    status = {
        "stopped": "completed",
        "nonfinite_objective": 0,
        "nonfinite_grad": 0,
        "no_grad": 0,
        "zero_grad_at_identity": False,
        "constraint_violation": False,
        "skipped_updates": 0,
    }
    subset_indices: list[list[int]] = []
    optim_seconds = 0.0
    validation_seconds = 0.0
    train_eval_seconds = 0.0

    if resume is None:
        pa = initial_vech(qa, init_seed, pert_scale).requires_grad_(True)
        opt = torch.optim.Adam([pa], lr=float(lr))
        rng = np.random.Generator(np.random.PCG64(int(sampling_seed))) if procedure == "quarter" else None
        curves: list[dict[str, Any]] = []
        updates_done = 0
        with torch.no_grad():
            s_a, _ = _project_a_only(pa, qa, qb, rho)
            pa.copy_(vech_from_s(_as_np(s_a)))
    else:
        pa = torch.tensor(np.asarray(resume["vech"], dtype=np.float64), dtype=torch.float64).requires_grad_(True)
        opt = torch.optim.Adam([pa], lr=float(lr))
        _load_adam(opt, pa, resume.get("adam"))
        rng = load_rng(resume["rng"]) if resume.get("rng") else None
        curves = [dict(row) for row in resume["curves"]]
        updates_done = int(resume["updates_done"])
        optim_seconds = float(resume.get("optim_seconds", 0.0))
        validation_seconds = float(resume.get("validation_seconds", 0.0))
        train_eval_seconds = float(resume.get("train_eval_seconds", 0.0))
        status.update(resume.get("status") or {})
        selected_primary = resume.get("selected_primary")
        selected_final = resume.get("selected_final")
        for key in snapshots:
            if resume.get(key) is not None:
                snapshots[key] = np.asarray(resume[key], dtype=np.float64)
        if record_indices and resume.get("subset_indices"):
            subset_indices = [list(map(int, row)) for row in resume["subset_indices"]]

    def current_s() -> np.ndarray:
        s_a, _ = _project_a_only(pa, qa, qb, rho)
        return _as_np(s_a)

    def consider(step: int, val: float | None, dist: float, s_np: np.ndarray) -> None:
        nonlocal selected_primary, selected_final
        if val is None or not math.isfinite(float(val)):
            return
        cand = {"step": int(step), "val_excess": float(val), "d_total": float(dist)}
        if step <= primary and _selection_better(cand, selected_primary):
            selected_primary = cand
            snapshots["s_selected_primary"] = s_np.copy()
        if _selection_better(cand, selected_final):
            selected_final = cand
            snapshots["s_selected_final"] = s_np.copy()

    def checkpoint(update_objective: float | None) -> None:
        nonlocal train_eval_seconds, validation_seconds
        s_np = current_s()
        if not constraint_ok(s_np, zeros_b, rho, "equal_total", "a_only"):
            status["constraint_violation"] = True
            status["stopped"] = "constraint_violation"
        t0 = time.perf_counter()
        train = score_pack(s_np, train_pack)
        train_eval_seconds += time.perf_counter() - t0
        val_excess = None
        if val_pack is not None:
            t1 = time.perf_counter()
            val_excess = float(score_pack(s_np, val_pack)["excess"])
            validation_seconds += time.perf_counter() - t1
        _, _, dist = distortions(s_np, zeros_b)
        row = {
            "step": int(updates_done),
            "full_train_excess": float(train["excess"]),
            "full_val_excess": val_excess,
            "d_total": float(dist),
            "update_objective_excess": update_objective,
        }
        curves.append(row)
        consider(updates_done, val_excess, dist, s_np)
        if updates_done == primary:
            snapshots["s_at_primary"] = s_np.copy()
        if updates_done == int(n_updates):
            snapshots["s_at_final"] = s_np.copy()
        if on_checkpoint is not None:
            on_checkpoint(resume_state())

    def resume_state() -> dict[str, Any]:
        return {
            "updates_done": int(updates_done),
            "vech": _as_np(pa.detach()).copy(),
            "adam": _adam_blob(opt, pa),
            "rng": dump_rng(rng) if rng is not None else None,
            "curves": curves,
            "selected_primary": selected_primary,
            "selected_final": selected_final,
            "s_at_primary": snapshots["s_at_primary"],
            "s_selected_primary": snapshots["s_selected_primary"],
            "s_at_final": snapshots["s_at_final"],
            "s_selected_final": snapshots["s_selected_final"],
            "subset_indices": subset_indices if record_indices else None,
            "optim_seconds": optim_seconds,
            "validation_seconds": validation_seconds,
            "train_eval_seconds": train_eval_seconds,
            "status": status,
        }

    if resume is None:
        checkpoint(None)
    while updates_done < int(n_updates) and status["stopped"] == "completed":
        t0 = time.perf_counter()
        if procedure == "quarter":
            idx = draw_quarter_indices(n, rng, quarter_fraction)
            if record_indices:
                subset_indices.append(idx.astype(int).tolist())
            chosen = [ids[int(i)] for i in idx]
            if len(set(chosen)) != len(chosen) or not set(chosen) <= set(ids):
                raise RuntimeError("subset ids left the training pool")
            pack = _packs(features_a[idx], features_b[idx], basis_a, basis_b)[0]
        else:
            pack = train_pack
        pack_t = pack_to_torch(pack)
        opt.zero_grad(set_to_none=True)
        s_a, s_b = _project_a_only(pa, qa, qb, rho)
        s_var = s_a.detach().requires_grad_(True)
        try:
            obj = _excess_torch(s_var, s_b.detach(), [pack_t], "a_only")
        except RuntimeError:
            status["nonfinite_objective"] += 1
            status["stopped"] = "runtime"
            optim_seconds += time.perf_counter() - t0
            break
        if not torch.isfinite(obj):
            status["nonfinite_objective"] += 1
            status["stopped"] = "nonfinite_objective"
            optim_seconds += time.perf_counter() - t0
            break
        if not obj.requires_grad:
            status["no_grad"] += 1
            status["stopped"] = "no_grad"
            optim_seconds += time.perf_counter() - t0
            break
        obj.backward()
        if s_var.grad is None or not torch.isfinite(s_var.grad).all():
            status["nonfinite_grad"] += 1
            status["stopped"] = "nonfinite_grad"
            optim_seconds += time.perf_counter() - t0
            break
        pa.grad = -_grad_to_vech(s_var.grad)
        grad_norm = float(torch.linalg.norm(pa.grad))
        if updates_done == 0 and int(init_seed) == 0 and grad_norm <= 1e-12:
            status["zero_grad_at_identity"] = True
        torch.nn.utils.clip_grad_norm_([pa], float(grad_clip))
        opt.step()
        with torch.no_grad():
            s_next, _ = _project_a_only(pa, qa, qb, rho)
            pa.copy_(vech_from_s(_as_np(s_next)))
        if not constraint_ok(_as_np(s_next), zeros_b, rho, "equal_total", "a_only"):
            status["constraint_violation"] = True
            status["stopped"] = "constraint_violation"
            optim_seconds += time.perf_counter() - t0
            break
        objective_value = float(obj.detach())
        optim_seconds += time.perf_counter() - t0
        updates_done += 1
        checkpoint(objective_value)

    s_final = current_s()
    if snapshots["s_at_final"] is None and updates_done == int(n_updates):
        snapshots["s_at_final"] = s_final.copy()
    quarter_n = int(round(n * float(quarter_fraction))) if procedure == "quarter" else n
    return {
        "ok": status["stopped"] == "completed",
        "procedure": procedure,
        "objective": "full_gallery_excess" if procedure == "full" else "expected_subset_excess",
        "objective_note": (
            "Each quarter update differentiates excess after centring and normalising inside that subset. "
            "The expected subset excess is not full-gallery excess, and this gradient is not an unbiased "
            "estimator of the full-gallery gradient."
            if procedure == "quarter"
            else "Each update differentiates excess on the complete training gallery."
        ),
        "rho": float(rho),
        "init_seed": int(init_seed),
        "sampling_seed": None if sampling_seed is None else int(sampling_seed),
        "n_updates_requested": int(n_updates),
        "updates_done": int(updates_done),
        "primary_updates": int(primary),
        "n_train": n,
        "examples_per_update": int(quarter_n),
        "examples_processed": int(updates_done) * int(quarter_n),
        "n_val_checks": int(sum(1 for row in curves if row["full_val_excess"] is not None)),
        "n_val_checks_primary": int(
            sum(1 for row in curves if row["full_val_excess"] is not None and int(row["step"]) <= primary)
        ),
        "examples_processed_primary": int(min(updates_done, primary)) * int(quarter_n),
        "curves": curves,
        "selected_primary": selected_primary,
        "selected_final": selected_final,
        "s_at_primary": snapshots["s_at_primary"],
        "s_selected_primary": snapshots["s_selected_primary"],
        "s_at_final": snapshots["s_at_final"],
        "s_selected_final": snapshots["s_selected_final"],
        "feasible": bool(constraint_ok(s_final, zeros_b, rho, "equal_total", "a_only") and not status["constraint_violation"]),
        "optim_seconds": float(optim_seconds),
        "validation_seconds": float(validation_seconds),
        "train_eval_seconds": float(train_eval_seconds),
        "basis_a_id": basis_a["id"],
        "basis_b_id": basis_b["id"],
        "lr": float(lr),
        "grad_clip": float(grad_clip),
        "status": status,
        "subset_indices": subset_indices if record_indices else None,
        "resume_state": resume_state(),
    }
