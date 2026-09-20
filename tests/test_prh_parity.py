"""Parity with platonic-rep@dcd76ba metrics and clip+L2 prep."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prh_replication.metrics import linear_cka, mutual_knn_score, prepare_original, rbf_cka, remove_outliers
from prh_replication.prh_ref import PRH_RBF_SIGMA, prh_prepare_layer


def official_remove_outliers(feats, q=0.95, exact=True, max_threshold=None):
    if q == 1:
        return feats
    if exact:
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    if max_threshold is not None:
        q_val = max(max_threshold, q_val)
    return feats.clamp(-q_val, q_val)


def official_hsic_biased(K, L):
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def official_cka(feats_A, feats_B, kernel_metric="ip", rbf_sigma=1.0):
    if kernel_metric == "ip":
        K = torch.mm(feats_A, feats_A.T)
        L = torch.mm(feats_B, feats_B.T)
    else:
        K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
        L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
    hsic_kk = official_hsic_biased(K, K)
    hsic_ll = official_hsic_biased(L, L)
    hsic_kl = official_hsic_biased(K, L)
    return (hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item()


def official_mknn(feats_A, feats_B, topk=10):
    knn_A = (feats_A @ feats_A.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    knn_B = (feats_B @ feats_B.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    n = knn_A.shape[0]
    range_tensor = torch.arange(n).unsqueeze(1)
    lvm_mask = torch.zeros(n, n)
    llm_mask = torch.zeros(n, n)
    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0
    return ((lvm_mask * llm_mask).sum(dim=1) / topk).mean().item()


def test_clip_matches_official():
    torch.manual_seed(0)
    x = torch.randn(32, 64) * 3
    a = official_remove_outliers(x.clone(), q=0.95, exact=True)
    b = remove_outliers(x.clone(), q=0.95, exact=True)
    assert torch.allclose(a, b)


def test_prep_is_clip_then_l2():
    torch.manual_seed(1)
    x = torch.randn(16, 8) * 5
    y = prh_prepare_layer(x)
    z = F.normalize(official_remove_outliers(x.float(), 0.95, True), dim=-1)
    assert torch.allclose(y, z, atol=1e-6)


def test_mknn_and_cka_match_official_on_normalized_features():
    torch.manual_seed(2)
    a = F.normalize(torch.randn(48, 16), dim=-1)
    b = F.normalize(torch.randn(48, 24), dim=-1)
    assert abs(mutual_knn_score(a, b, 10) - official_mknn(a, b, 10)) < 1e-6
    assert abs(linear_cka(a, b) - official_cka(a, b, "ip")) < 1e-5
    assert abs(rbf_cka(a, b, PRH_RBF_SIGMA, PRH_RBF_SIGMA) - official_cka(a, b, "rbf", 1.0)) < 1e-5


def test_prepare_original_alias():
    torch.manual_seed(3)
    x = torch.randn(12, 7)
    assert torch.allclose(prepare_original(x), prh_prepare_layer(x))


def test_mknn_overlap_fast_matches_official():
    from prh_replication.metrics import nearest_neighbors
    from prh_replication.prh_ref import mknn_overlap_fast

    torch.manual_seed(4)
    a = F.normalize(torch.randn(40, 8), dim=-1)
    b = F.normalize(torch.randn(40, 8), dim=-1)
    ka, kb = nearest_neighbors(a, 10), nearest_neighbors(b, 10)
    assert abs(mknn_overlap_fast(ka, kb) - official_mknn(a, b, 10)) < 1e-6


def test_selection_aware_null_reuses_one_perm_and_exceeds_chance():
    from prh_replication.prh_ref import selection_aware_mknn_null, stack_prepared

    torch.manual_seed(5)
    xa = stack_prepared(torch.randn(64, 3, 16))
    xb = stack_prepared(torch.randn(64, 4, 12))
    out = selection_aware_mknn_null(xa, xb, n_perm=8, seed=0, topk=10)
    chance = 10 / 63
    assert out["n_perm"] == 8
    assert out["shuffle_mean"] > chance
    assert out["shuffle_mean"] < 0.4
