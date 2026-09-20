"""Learned-kernel CKA: extension estimator, radial families, diagnostics.

Official platonic-rep CKA uses biased HSIC with +1e-6 in the denominator
(`metrics.cka_from_grams`). The extension estimator below is the ordinary
centred-Gram cosine without that epsilon, so a/b matches the analytic
unrestricted-permutation formula when the grams are centred PSD and nonzero.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from prh_replication.metrics import hsic_unbiased


REL_CENTER = 1e-12
OFFICIAL_EPS = 1e-6

# Frozen in configs/learned_kernels.json (numpy 10**linspace).
LAMBDA_GRID = (
    0.1,
    0.12115276586285885,
    0.14677992676220694,
    0.1778279410038923,
    0.21544346900318834,
    0.26101572156825364,
    0.31622776601683794,
    0.3831186849557287,
    0.46415888336127786,
    0.5623413251903491,
    0.6812920690579611,
    0.8254041852680184,
    1.0,
    1.2115276586285881,
    1.467799267622069,
    1.7782794100389228,
    2.1544346900318834,
    2.610157215682536,
    3.1622776601683795,
    3.831186849557287,
    4.6415888336127775,
    5.623413251903491,
    6.812920690579611,
    8.254041852680182,
    10.0,
)
ALPHA_GRID = (
    0.1,
    0.23713737056616552,
    0.5623413251903491,
    1.333521432163324,
    3.1622776601683795,
    7.498942093324558,
    17.78279410038923,
    42.169650342858226,
    100.0,
)
PROFILE_R_MAX = 3.0
PROFILE_N_NODES = 513


def center_gram_o2(k: torch.Tensor) -> torch.Tensor:
    """HKH via row/col means. Algebraically the same as H @ K @ H, O(n^2)."""
    row = k.mean(dim=1, keepdim=True)
    col = k.mean(dim=0, keepdim=True)
    return k - row - col + k.mean()


def _is_bad_center(k: torch.Tensor, kc: torch.Tensor) -> bool:
    kn = float(torch.linalg.norm(k, ord="fro"))
    nkc = float(torch.linalg.norm(kc, ord="fro"))
    if not math.isfinite(nkc):
        return True
    return nkc <= REL_CENTER * (kn + 1e-12)


def extension_stats(k: torch.Tensor, l: torch.Tensor) -> dict[str, Any]:
    """Return a, b, ratio, excess plus validity flags. No official eps."""
    kc = center_gram_o2(k)
    lc = center_gram_o2(l)
    n = k.shape[0]
    ip = float((kc * lc).sum())
    nk = float(torch.linalg.norm(kc, ord="fro"))
    nl = float(torch.linalg.norm(lc, ord="fro"))
    trk = float(kc.trace())
    trl = float(lc.trace())
    degenerate = _is_bad_center(k, kc) or _is_bad_center(l, lc)
    finite = all(math.isfinite(v) for v in (ip, nk, nl, trk, trl))
    valid_a = finite and (not degenerate) and nk > 0.0 and nl > 0.0
    a = ip / (nk * nl) if valid_a else float("nan")
    b = (trk * trl) / ((n - 1) * nk * nl) if valid_a else float("nan")
    den_tr = trk * trl
    ratio_undef = (not valid_a) or abs(den_tr) <= REL_CENTER * n * nk * nl
    ratio = ((n - 1) * ip / den_tr) if not ratio_undef else float("nan")
    excess = (a - b) if valid_a and math.isfinite(b) else float("nan")
    official = ip / (nk * nl + OFFICIAL_EPS) if finite and nk > 0 and nl > 0 else float("nan")
    return {
        "a": a,
        "b": b,
        "ratio": ratio,
        "excess": excess,
        "official_cka": official,
        "ip": ip,
        "nk": nk,
        "nl": nl,
        "trk": trk,
        "trl": trl,
        "n": n,
        "degenerate": degenerate,
        "ratio_undefined": ratio_undef,
        "valid_a": valid_a,
    }


def u_centred_cka(k: torch.Tensor, l: torch.Tensor) -> dict[str, Any]:
    """Song et al. unbiased-HSIC CKA. May be negative. Not a fitting objective."""
    num = hsic_unbiased(k, l)
    d1 = hsic_unbiased(k, k)
    d2 = hsic_unbiased(l, l)
    den = torch.sqrt(d1 * d2)
    zero_den = (not torch.isfinite(den)) or float(den) <= 0
    val = float("nan") if zero_den else float((num / den).item())
    return {
        "u_centred_cka": val,
        "u_centred_zero_denominator": bool(zero_den),
        "u_hsic_kl": float(num),
        "u_hsic_kk": float(d1),
        "u_hsic_ll": float(d2),
    }


def rbf_gram_from_dsq(dsq: torch.Tensor, sigma: float) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"nonpositive RBF sigma {sigma}")
    return torch.exp(dsq * (-0.5 / (sigma * sigma)))


def rq_gram_from_dsq(dsq: torch.Tensor, sigma: float, alpha: float) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"invalid RQ params sigma={sigma} alpha={alpha}")
    return (1.0 + dsq / (2.0 * alpha * sigma * sigma)).pow(-alpha)


def linear_gram(feats: torch.Tensor) -> torch.Tensor:
    x = feats.float()
    return x @ x.T


def pairwise_sq_distances(feats: torch.Tensor) -> torch.Tensor:
    return torch.cdist(feats.float(), feats.float(), p=2).pow(2)


def median_offdiag_from_dsq(dsq: torch.Tensor) -> float:
    n = dsq.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=dsq.device)
    d = torch.sqrt(dsq[mask].clamp(min=0))
    if d.numel() == 0:
        return float("nan")
    return float(d.median())


def kernel_from_spec(
    dsq: torch.Tensor,
    feats: torch.Tensor,
    family: str,
    sigma: float | None,
    alpha: float | None,
) -> torch.Tensor:
    if family == "linear":
        return linear_gram(feats)
    if family == "rbf":
        return rbf_gram_from_dsq(dsq, float(sigma))
    if family == "rq":
        return rq_gram_from_dsq(dsq, float(sigma), float(alpha))
    raise ValueError(family)


def gram_diagnostics(k: torch.Tensor) -> dict[str, Any]:
    n = k.shape[0]
    kc = center_gram_o2(k)
    fro2 = float((kc * kc).sum())
    diag2 = float((kc.diag() ** 2).sum())
    off_mask = ~torch.eye(n, dtype=torch.bool, device=k.device)
    k_off = k[off_mask]
    qs = torch.quantile(k_off, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=k.device))
    evals = torch.linalg.eigvalsh(kc)
    pos = evals.clamp(min=0)
    psum = float(pos.sum())
    p2 = float(pos.pow(2).sum())
    erank = (psum * psum / p2) if p2 > 0 else float("nan")
    return {
        "centred_fro_norm": math.sqrt(fro2) if fro2 >= 0 else float("nan"),
        "centred_diag_energy_frac": (diag2 / fro2) if fro2 > 0 else float("nan"),
        "centred_offdiag_energy_frac": ((fro2 - diag2) / fro2) if fro2 > 0 else float("nan"),
        "k_off_min": float(qs[0]),
        "k_off_q25": float(qs[1]),
        "k_off_median": float(qs[2]),
        "k_off_q75": float(qs[3]),
        "k_off_max": float(qs[4]),
        "mean_kernel": float(k.mean()),
        "mean_offdiag_kernel": float(k_off.mean()),
        "effective_rank_nonneg_eigs": erank,
        "effective_rank_definition": "(sum max(lambda,0))^2 / sum max(lambda,0)^2 of centred Gram",
        "n_neg_eigs": int((evals < -1e-8).sum()),
    }


def rbf_profile(r: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.exp(-r.pow(2) / (2.0 * lam * lam))


def rq_profile(r: torch.Tensor, lam: float, alpha: float) -> torch.Tensor:
    return (1.0 + r.pow(2) / (2.0 * alpha * lam * lam)).pow(-alpha)


def profile_nodes() -> torch.Tensor:
    return torch.linspace(0.0, PROFILE_R_MAX, PROFILE_N_NODES)


def _trapz(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum() * 0.5


def profile_inner_products(f: torch.Tensor, g: torch.Tensor, r: torch.Tensor) -> dict[str, float]:
    """Uniform probability ν on [0, R]; trapezoid quadrature.

    Mean-centring is of radial profiles under ν, not Gram double-centring.
    """
    length = PROFILE_R_MAX
    # ∫ h dν = (1/R) ∫_0^R h(r) dr
    def mean(h):
        return _trapz(h, r) / length

    def inner(u, v):
        return float(mean(u * v))

    raw_ip = inner(f, g)
    nf = math.sqrt(inner(f, f))
    ng = math.sqrt(inner(g, g))
    raw_cos = raw_ip / (nf * ng) if nf > REL_CENTER and ng > REL_CENTER else float("nan")
    fm = f - mean(f)
    gm = g - mean(g)
    cip = inner(fm, gm)
    nfm = math.sqrt(inner(fm, fm))
    ngm = math.sqrt(inner(gm, gm))
    near_zero = nfm <= REL_CENTER or ngm <= REL_CENTER
    centred_cos = cip / (nfm * ngm) if not near_zero else float("nan")
    return {
        "raw_normalised_l2": raw_cos,
        "centred_normalised_l2": centred_cos,
        "profile_norm_f": nf,
        "profile_norm_g": ng,
        "centred_profile_norm_f": nfm,
        "centred_profile_norm_g": ngm,
        "near_zero_centred_profile": near_zero,
    }


def mass_outside_interval(dsq: torch.Tensor, scale: float, rmax: float = PROFILE_R_MAX) -> float:
    if not math.isfinite(scale) or scale <= 0:
        return float("nan")
    n = dsq.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=dsq.device)
    r = torch.sqrt(dsq[mask].clamp(min=0)) / scale
    return float((r > rmax).float().mean())


def permute_gram(k: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return k[perm][:, perm]


def mc_cka_mean(k: torch.Tensor, l: torch.Tensor, n_perm: int, seed: int) -> dict[str, float]:
    if n_perm <= 0:
        return {"mc_mean": float("nan"), "mc_std": float("nan"), "mc_n_valid": 0, "n_perm": 0}
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = l.shape[0]
    scores = []
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=g)
        st = extension_stats(k, permute_gram(l, perm))
        if st["valid_a"]:
            scores.append(st["a"])
    t = torch.tensor(scores) if scores else torch.tensor([float("nan")])
    return {
        "mc_mean": float(t.nanmean()) if scores else float("nan"),
        "mc_std": float(t.std(unbiased=True)) if len(scores) > 1 else float("nan"),
        "mc_n_valid": len(scores),
        "n_perm": n_perm,
    }


def objective_value(stats: dict[str, Any], name: str) -> float:
    if name == "cka":
        return stats["a"]
    if name == "ratio":
        return stats["ratio"]
    if name == "excess":
        return stats["excess"]
    raise ValueError(name)


def is_boundary(family: str, lam: float | None, alpha: float | None) -> bool:
    if lam is None:
        return False
    on_l = abs(lam - 0.1) < 1e-12 or abs(lam - 10.0) < 1e-12
    if family == "rq" and alpha is not None:
        on_a = abs(alpha - 0.1) < 1e-12 or abs(alpha - 100.0) < 1e-12
        return on_l or on_a
    return on_l
