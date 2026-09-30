"""Local CKA over each point's neighbourhood (exp-009), vectorised over points and layers.

``mutual`` is Emily's `cknna_local` (branch `emily`, geometry_aware_convergence/metrics.py): for
every point i, take its mutual neighbours S_i = N_K(i) & N_L(i), compute CKA with unbiased HSIC
(platonic-rep `hsic_unbiased`) on the Gram blocks of {i} + S_i, skip points with fewer than 4
mutual neighbours (unbiased HSIC needs a block of >= 5), and average the kept points.

``union`` is the same local CKA on {i} + (N_K(i) | N_L(i)): every point is kept, and neighbours only
one model has count against the score.

For a block indicator w (1 on the block's points) of size m, and Kt, Lt the Grams with a zero
diagonal, unbiased HSIC of the block is

    [ wᵀ(Kt∘Lt)w + (wᵀKt w)(wᵀLt w)/((m-1)(m-2)) - 2 Σ_b w_b (Kt w)_b (Lt w)_b/(m-2) ] / (m(m-3))

so with all blocks stacked as rows of B, every term is a product B @ (n x n matrix) followed by a
row-wise dot with B: five n^3 products per layer pair give all n local scores.

Everything here runs in float64. The blocks are small and tight (a point and its nearest
neighbours, cosines ~0.9), so the three HSIC terms nearly cancel; in float32 a per-point score can
be off by 30 at k = 50 and exceed 1, which it cannot in exact arithmetic (unbiased HSIC is an inner
product of U-centred matrices, so each local CKA is bounded by 1).
"""

from __future__ import annotations

import torch

MIN_MUTUAL = 4  # Emily's default for unbiased HSIC


def _local_cka(B: torch.Tensor, Kt, Lt, KK, LL, KL) -> torch.Tensor:
    """Per-point CKA (unbiased HSIC) on the blocks given by the rows of B.

    B: [..., n, n] 0/1 block indicators (self included); Kt, Lt: zero-diagonal Grams;
    KK = Kt*Kt, LL = Lt*Lt, KL = Kt*Lt. All broadcast against B's leading dims.
    """
    m = B.sum(-1)
    U, V = B @ Kt, B @ Lt  # row i: (Kt w_i)ᵀ, (Lt w_i)ᵀ
    sK, sL = (U * B).sum(-1), (V * B).sum(-1)

    def hsic(t1, s1, s2, t3):
        return (t1 + s1 * s2 / ((m - 1) * (m - 2)) - 2 * t3 / (m - 2)) / (m * (m - 3))

    kl = hsic(((B @ KL) * B).sum(-1), sK, sL, (B * U * V).sum(-1))
    kk = hsic(((B @ KK) * B).sum(-1), sK, sK, (B * U * U).sum(-1))
    ll = hsic(((B @ LL) * B).sum(-1), sL, sL, (B * V * V).sum(-1))
    return kl / (torch.sqrt(kk * ll) + 1e-6)


class LocalGram:
    """Zero-diagonal Gram of one layer and its square, for the local metrics."""

    def __init__(self, K: torch.Tensor):
        self.Kt = K.to(torch.float64, copy=True)
        self.Kt.diagonal(dim1=-2, dim2=-1).fill_(0)
        self.KK = self.Kt * self.Kt


def local_scores(
    mK: torch.Tensor, mL: torch.Tensor, gk: LocalGram, gl: LocalGram
) -> dict[str, torch.Tensor]:
    """Per-point scores [..., n] of `mutual` (NaN where skipped) and `union`.

    mK, mL: k-NN masks (self excluded) broadcastable to each other; gk, gl: the two layers.
    """
    n = mK.shape[-1]
    mK, mL = mK.to(gk.Kt.dtype), mL.to(gk.Kt.dtype)
    eye = torch.eye(n, device=mK.device, dtype=mK.dtype)
    KL = gk.Kt * gl.Kt
    W = mK * mL
    mutual = _local_cka(W + eye, gk.Kt, gl.Kt, gk.KK, gl.KK, KL)
    mutual = torch.where(W.sum(-1) >= MIN_MUTUAL, mutual, torch.nan)
    union = _local_cka(torch.maximum(mK, mL) + eye, gk.Kt, gl.Kt, gk.KK, gl.KK, KL)
    return {"mutual": mutual, "union": union}


def reduce(scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mean over points (kept points for `mutual`, as Emily's), plus mutual coverage."""
    kept = ~torch.isnan(scores["mutual"])
    mutual = torch.nansum(scores["mutual"], -1) / kept.sum(-1).clamp(min=1)
    return {
        "local_mutual": torch.where(kept.any(-1), mutual, torch.zeros_like(mutual)),
        "local_union": scores["union"].mean(-1),
        "coverage": kept.float().mean(-1),
    }


def rowwise(mK, mL, Kc, Lc, k: int) -> dict[str, torch.Tensor]:
    """Per-point versions of mutual kNN and the centred CKNNA (rows of the global sums)."""
    a = mK * mL
    num = (a * Kc * Lc).sum(-1)
    den = torch.sqrt((mK * Kc * Kc).sum(-1) * (mL * Lc * Lc).sum(-1))
    return {"mknn": a.sum(-1) / k, "centred": num / den}
