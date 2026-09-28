"""CKNNA variants, mutual kNN and linear CKA over whole layer stacks, with a permutation null.

Four CKNNA definitions (exp-007, exp-008). With K = X X^T of one layer (l2-normalised rows),
m the k-NN indicator (m_ij = 1 if j is among the k nearest neighbours of i, self excluded),
a = m_K * m_L the mutual-neighbour mask, and H = I - 11^T / n:

- ``paper``    Huh et al. 2024, App. A, Eqs. 12, 16-18, as printed: centre each row
               (Kr = K - row mean), then sum over mutual neighbours:
               sum(a Kr Lr) / sqrt(sum(m_K Kr^2) sum(m_L Lr^2)). Bounded by 1.
- ``centred``  the same with the CKA centring of Eq. 14 (Kc = H K H) in place of the row
               centring: sum(a Kc Lc) / sqrt(sum(m_K Kc^2) sum(m_L Lc^2)). Bounded by 1; at
               k = n - 1 it is linear CKA without the diagonal terms.
- ``code``     platonic-rep `AlignmentMetrics.cknna` (unbiased=True), copied unchanged into
               Aristotelian: mask the raw kernel, then unbiased HSIC,
               hsic_u(a K, a L) / (sqrt(hsic_u(m_K K, m_K K) hsic_u(m_L L, m_L L)) + 1e-6).
               Not bounded by 1.
- ``eq36``     Groger et al. 2026, App. B.2.3, Eq. 36: mask the raw kernel, then double-centre,
               <H(a K)H, H(a L)H>_F / (||H(m_K K)H||_F ||H(m_L L)H||_F). Not bounded by 1.

Every layer x layer term is a matrix product: a Frobenius inner product of flattened n x n
matrices, or, for the row and column sums of the masked products, a batched product over the
sample index. A permutation of the image-caption pairing acts on the second model's matrices,
so the null reuses everything precomputed for the observed score.
"""

from __future__ import annotations

import torch

CKNNA_VARIANTS = ("paper", "centred", "code", "eq36")
PER_K = ("mknn", *CKNNA_VARIANTS)


def _flat_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[La, n, n] x [Lb, n, n] -> [La, Lb] Frobenius inner products."""
    return x.reshape(x.shape[0], -1) @ y.reshape(y.shape[0], -1).T


def _col(x: torch.Tensor, left: bool) -> torch.Tensor:
    """Layout for column sums (batch over column j, contract over row i)."""
    return (x.permute(2, 0, 1) if left else x.permute(2, 1, 0)).contiguous()


def _row(x: torch.Tensor, left: bool) -> torch.Tensor:
    """Layout for row sums (batch over row i, contract over column j)."""
    return (x.permute(1, 0, 2) if left else x.permute(1, 2, 0)).contiguous()


def _permute(x: torch.Tensor, perm: torch.Tensor, dims: tuple[int, int]) -> torch.Tensor:
    """Relabel the samples along the two sample axes of x."""
    idx = [slice(None)] * 3
    idx[dims[0]], idx[dims[1]] = perm[:, None], perm[None, :]
    return x[tuple(idx)]


class Stack:
    """One model's layers: Grams, their centrings and neighbour order, on one device."""

    def __init__(self, layers: list[torch.Tensor], device: str | torch.device = "cpu"):
        grams = []
        for layer in layers:
            x = layer.float().to(device)
            grams.append(x @ x.T)
        self.K = torch.stack(grams)
        self.n = self.K.shape[1]
        del grams
        self.Kr = self.K - self.K.mean(2, keepdim=True)
        self.Kc = self.Kr - self.K.mean(1, keepdim=True) + self.K.mean((1, 2), keepdim=True)
        self.cka_self = (self.Kc * self.Kc).sum((1, 2))
        khat = self.K.clone()
        khat.diagonal(dim1=1, dim2=2).fill_(float("-inf"))
        # neighbours by decreasing similarity, self excluded
        self.order = torch.argsort(khat, dim=2, descending=True)[:, :, : self.n - 1]
        self.order = self.order.to(torch.int16 if self.n < 2**15 else torch.int32)
        del khat

    def __len__(self) -> int:
        return self.K.shape[0]

    def at_k(self, k: int) -> AtK:
        return AtK(self, k)


class AtK:
    """The masked matrices of a Stack at one k, plus each variant's self term."""

    def __init__(self, s: Stack, k: int):
        n = s.n
        self.n, self.k = n, k
        m = torch.zeros_like(s.K).scatter_(2, s.order[:, :, :k].long(), 1.0)
        mK = m * s.K
        self.m, self.mK = m, mK
        self.mKr, self.mKc = m * s.Kr, m * s.Kc
        self.mmK = m * m.transpose(1, 2) * s.K  # hsic_u's K * L^T term keeps reciprocal edges
        self.self = {
            "paper": (self.mKr * s.Kr).sum((1, 2)),
            "centred": (self.mKc * s.Kc).sum((1, 2)),
            "code": _hsic_u_self(mK),
            "eq36": _hcentred_norm(mK),
        }
        self._layouts: dict[bool, dict[str, torch.Tensor]] = {}

    def layouts(self, left: bool) -> dict[str, torch.Tensor]:
        """Row/column-sum layouts of m and m*K, built on first use."""
        if left not in self._layouts:
            self._layouts[left] = {
                "m_col": _col(self.m, left),
                "mK_col": _col(self.mK, left),
                "m_row": _row(self.m, left),
                "mK_row": _row(self.mK, left),
            }
        return self._layouts[left]

    def permuted(self, perm: torch.Tensor) -> dict[str, torch.Tensor]:
        """The right-hand tensors after relabelling the samples by perm."""
        lay = self.layouts(left=False)
        out = {
            name: _permute(getattr(self, name), perm, (1, 2))
            for name in ("m", "mK", "mKr", "mKc", "mmK")
        }
        out["m_col"], out["mK_col"] = (_permute(lay[f], perm, (0, 1)) for f in ("m_col", "mK_col"))
        out["m_row"], out["mK_row"] = (_permute(lay[f], perm, (0, 1)) for f in ("m_row", "mK_row"))
        return out

    def right(self) -> dict[str, torch.Tensor]:
        lay = self.layouts(left=False)
        return {
            **{name: getattr(self, name) for name in ("m", "mK", "mKr", "mKc", "mmK")},
            **lay,
        }


def _hsic_u_self(A: torch.Tensor) -> torch.Tensor:
    """platonic-rep hsic_unbiased(A, A) per layer; A has a zero diagonal."""
    n = A.shape[1]
    t1 = (A * A.transpose(1, 2)).sum((1, 2))
    total = A.sum((1, 2))
    t3 = (A.sum(1) * A.sum(2)).sum(1)  # sum(A @ A) = colsum(A) . rowsum(A)
    return (t1 + total * total / ((n - 1) * (n - 2)) - 2 * t3 / (n - 2)) / (n * (n - 3))


def _hcentred_norm(A: torch.Tensor) -> torch.Tensor:
    """||H A H||_F per layer."""
    n = A.shape[1]
    sq = (A * A).sum((1, 2)) - (A.sum(1) ** 2).sum(1) / n - (A.sum(2) ** 2).sum(1) / n
    sq = sq + A.sum((1, 2)) ** 2 / n**2
    return sq.sqrt()


def per_k_scores(a: AtK, b: AtK, right: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """[La, Lb] mutual kNN and the four CKNNA variants at one k.

    ``right`` is ``b.right()`` or ``b.permuted(perm)``; the self terms of b do not change under
    a permutation.
    """
    n, k = a.n, a.k
    left = a.layouts(left=True)
    out = {"mknn": _flat_dot(a.m, right["m"]) / (n * k)}
    for name, key in (("paper", "mKr"), ("centred", "mKc")):
        num = _flat_dot(getattr(a, key), right[key])
        out[name] = num / torch.sqrt(a.self[name][:, None] * b.self[name][None, :])

    # sums of the masked products aK = m_A m_B K and aL = m_A m_B L
    sum_aK = _flat_dot(a.mK, right["m"])
    sum_aL = _flat_dot(a.m, right["mK"])
    col_aK = torch.bmm(left["mK_col"], right["m_col"])  # [n, La, Lb], column j sums
    col_aL = torch.bmm(left["m_col"], right["mK_col"])
    row_aK = torch.bmm(left["mK_row"], right["m_row"])  # [n, La, Lb], row i sums
    row_aL = torch.bmm(left["m_row"], right["mK_row"])

    t1 = _flat_dot(a.mmK, right["mmK"])
    t3 = (col_aK * row_aL).sum(0)
    hsic = (t1 + sum_aK * sum_aL / ((n - 1) * (n - 2)) - 2 * t3 / (n - 2)) / (n * (n - 3))
    out["code"] = hsic / (torch.sqrt(a.self["code"][:, None] * b.self["code"][None, :]) + 1e-6)

    fro = _flat_dot(a.mK, right["mK"])
    num = fro - (col_aK * col_aL).sum(0) / n - (row_aK * row_aL).sum(0) / n
    num = num + sum_aK * sum_aL / n**2
    out["eq36"] = num / (a.self["eq36"][:, None] * b.self["eq36"][None, :])
    return out


def cka(a: Stack, b: Stack, perm: torch.Tensor | None = None) -> torch.Tensor:
    """[La, Lb] linear (biased) CKA, platonic-rep `cka`, optionally with b relabelled."""
    Lc = b.Kc if perm is None else _permute(b.Kc, perm, (1, 2))
    return _flat_dot(a.Kc, Lc) / (torch.sqrt(a.cka_self[:, None] * b.cka_self[None, :]) + 1e-6)
