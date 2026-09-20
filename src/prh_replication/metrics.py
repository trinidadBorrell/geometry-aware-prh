"""Fixed-kernel alignment metrics.

Mutual kNN follows the official platonic-rep implementation and Appendix A
equation (11): mean |S_i ∩ T_i| / k, not Jaccard. Self-neighbours are
excluded by filling the Gram diagonal before argsort.

Linear CKA uses biased HSIC (Kornblith et al. 2019) with an inner-product
kernel. RBF CKA uses K_ij = exp(-||x_i-x_j||^2 / (2 sigma^2)).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=eps)


def remove_outliers(feats: torch.Tensor, q: float = 0.95, exact: bool = True) -> torch.Tensor:
    """Match platonic-rep metrics.remove_outliers."""
    if q >= 1:
        return feats
    flat = feats.abs()
    if exact:
        q_val = flat.reshape(-1).sort().values[int(q * flat.numel())]
    else:
        q_val = torch.quantile(flat.flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-q_val, q_val)


def prepare_original(feats: torch.Tensor, q: float = 0.95, normalize: bool = True) -> torch.Tensor:
    x = remove_outliers(feats.float(), q=q, exact=True)
    if normalize:
        x = l2_normalize(x)
    return x


def nearest_neighbors(feats: torch.Tensor, topk: int = 10) -> torch.Tensor:
    """Inner-product kNN with self excluded. Ties follow torch.argsort (stable)."""
    assert feats.ndim == 2
    gram = feats @ feats.T
    gram = gram.fill_diagonal_(-1e8)
    return gram.argsort(dim=1, descending=True)[:, :topk]


def mutual_knn(feats_a: torch.Tensor, feats_b: torch.Tensor, topk: int = 10) -> torch.Tensor:
    knn_a = nearest_neighbors(feats_a, topk)
    knn_b = nearest_neighbors(feats_b, topk)
    n = knn_a.shape[0]
    device = knn_a.device
    mask_a = torch.zeros(n, n, device=device, dtype=torch.float32)
    mask_b = torch.zeros(n, n, device=device, dtype=torch.float32)
    rows = torch.arange(n, device=device).unsqueeze(1)
    mask_a[rows, knn_a] = 1.0
    mask_b[rows, knn_b] = 1.0
    return (mask_a * mask_b).sum(dim=1) / topk


def mutual_knn_score(feats_a: torch.Tensor, feats_b: torch.Tensor, topk: int = 10) -> float:
    return float(mutual_knn(feats_a, feats_b, topk).mean().item())


def _center_gram(k: torch.Tensor) -> torch.Tensor:
    n = k.shape[0]
    h = torch.eye(n, dtype=k.dtype, device=k.device) - 1.0 / n
    return h @ k @ h


def linear_gram(feats: torch.Tensor) -> torch.Tensor:
    return feats @ feats.T


def rbf_gram(feats: torch.Tensor, sigma: float) -> torch.Tensor:
    dist_sq = torch.cdist(feats, feats, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma * sigma))


def hsic_biased(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    h = torch.eye(k.shape[0], dtype=k.dtype, device=k.device) - 1.0 / k.shape[0]
    return torch.trace(k @ h @ l @ h)


def hsic_unbiased(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Song et al. 2012 unbiased HSIC (platonic-rep)."""
    m = k.shape[0]
    k_t = k.clone().fill_diagonal_(0)
    l_t = l.clone().fill_diagonal_(0)
    value = (
        torch.sum(k_t * l_t.T)
        + (torch.sum(k_t) * torch.sum(l_t) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(k_t @ l_t) / (m - 2))
    )
    return value / (m * (m - 3))


def cka_from_grams(k: torch.Tensor, l: torch.Tensor, unbiased: bool = False, eps: float = 1e-6) -> float:
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    num = hsic_fn(k, l)
    den = torch.sqrt(hsic_fn(k, k) * hsic_fn(l, l)) + eps
    if not torch.isfinite(den) or float(den.item()) <= 0:
        return float("nan")
    return float((num / den).item())


def linear_cka(feats_a: torch.Tensor, feats_b: torch.Tensor, unbiased: bool = False) -> float:
    return cka_from_grams(linear_gram(feats_a), linear_gram(feats_b), unbiased=unbiased)


def rbf_cka(feats_a: torch.Tensor, feats_b: torch.Tensor, sigma_a: float, sigma_b: float, unbiased: bool = False) -> float:
    return cka_from_grams(rbf_gram(feats_a, sigma_a), rbf_gram(feats_b, sigma_b), unbiased=unbiased)


def median_offdiag_distance(feats: torch.Tensor) -> float:
    dist = torch.cdist(feats.float(), feats.float(), p=2)
    n = dist.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
    return float(dist[mask].median().item())


def permutation_cka_mean_formula(kc: torch.Tensor, lc: torch.Tensor) -> float:
    """E[CKA] under uniform unrestricted permutations of one side.

    b = tr(Kc) tr(Lc) / ((n-1) ||Kc||_F ||Lc||_F)
    for nonzero centred Grams. Applies to ordinary (biased) CKA only.
    """
    n = kc.shape[0]
    num = torch.trace(kc) * torch.trace(lc)
    den = (n - 1) * torch.linalg.norm(kc, ord="fro") * torch.linalg.norm(lc, ord="fro")
    return float((num / den).item())


def center_gram(k: torch.Tensor) -> torch.Tensor:
    return _center_gram(k)


def relative_layer_indices(n_layers: int, fractions: Iterable[float]) -> list[int]:
    idxs = []
    last = max(n_layers - 1, 0)
    for f in fractions:
        i = int(round(float(f) * last))
        i = min(max(i, 0), last)
        if i not in idxs:
            idxs.append(i)
    return idxs


def mknn_from_indices(knn_a: torch.Tensor, knn_b: torch.Tensor) -> float:
    n, topk = knn_a.shape
    device = knn_a.device
    mask_a = torch.zeros(n, n, device=device, dtype=torch.float32)
    mask_b = torch.zeros(n, n, device=device, dtype=torch.float32)
    rows = torch.arange(n, device=device).unsqueeze(1)
    mask_a[rows, knn_a] = 1.0
    mask_b[rows, knn_b] = 1.0
    return float(((mask_a * mask_b).sum(dim=1) / topk).mean().item())


def max_mknn_over_layers(x_layers: torch.Tensor, y_layers: torch.Tensor, topk: int = 10) -> tuple[float, tuple[int, int]]:
    knn_x = [nearest_neighbors(x_layers[:, i], topk) for i in range(x_layers.shape[1])]
    knn_y = [nearest_neighbors(y_layers[:, j], topk) for j in range(y_layers.shape[1])]
    best, best_ij = float("-inf"), (0, 0)
    for i, kx in enumerate(knn_x):
        for j, ky in enumerate(knn_y):
            s = mknn_from_indices(kx, ky)
            if s > best:
                best, best_ij = s, (i, j)
    return best, best_ij


def max_cka_over_layers(
    x_layers: torch.Tensor,
    y_layers: torch.Tensor,
    kind: str = "linear",
    sigma_a: float | None = None,
    sigma_b: float | None = None,
) -> tuple[float, tuple[int, int]]:
    def gram(feats, sigma=None):
        if kind == "linear":
            return center_gram(linear_gram(feats))
        return center_gram(rbf_gram(feats, sigma))

    gx = [gram(x_layers[:, i], sigma_a) for i in range(x_layers.shape[1])]
    gy = [gram(y_layers[:, j], sigma_b) for j in range(y_layers.shape[1])]
    best, best_ij = float("-inf"), (0, 0)
    eps = 1e-6
    nx = [torch.linalg.norm(g, ord="fro") for g in gx]
    ny = [torch.linalg.norm(g, ord="fro") for g in gy]
    for i, ka in enumerate(gx):
        for j, kb in enumerate(gy):
            num = torch.sum(ka * kb)
            den = nx[i] * ny[j] + eps
            s = float((num / den).item())
            if s > best:
                best, best_ij = s, (i, j)
    return best, best_ij


def apply_layer_prep(layer: torch.Tensor, protocol: str) -> torch.Tensor:
    if protocol == "original":
        return prepare_original(layer)
    if protocol == "modern":
        return layer.float()
    raise ValueError(protocol)
