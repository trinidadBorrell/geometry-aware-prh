"""geoprh.cknna_variants against literal implementations and the upstream code."""

import pytest
import torch

from geoprh import cknna_variants as cv
from geoprh import prh_alignment as pa

M = pa.M  # platonic-rep metrics
N = 50


def _feats(layers, d, shared, seed):
    g = torch.Generator().manual_seed(seed)
    out = [
        shared @ torch.randn(shared.shape[1], d, generator=g) + torch.randn(N, d, generator=g)
        for _ in range(layers)
    ]
    return torch.stack(out, dim=1) + 2.0  # offset -> anisotropic, like real features


def _knn(G, k):
    g = G.clone().fill_diagonal_(float("-inf"))
    return torch.zeros(N, N).scatter_(1, torch.topk(g, k, dim=1).indices, 1.0)


def _literal(x, y, k, variant):
    """Each definition written out directly, in float64."""
    x, y = x.double(), y.double()
    K, L = x @ x.T, y @ y.T
    mK, mL = _knn(K, k).double(), _knn(L, k).double()
    a = mK * mL
    H = torch.eye(N, dtype=torch.float64) - 1.0 / N
    if variant == "paper":
        Kb, Lb = K - K.mean(1, keepdim=True), L - L.mean(1, keepdim=True)
    elif variant == "centred":
        Kb, Lb = H @ K @ H, H @ L @ H
    if variant in ("paper", "centred"):
        return (a * Kb * Lb).sum() / ((mK * Kb * Kb).sum() * (mL * Lb * Lb).sum()).sqrt()
    if variant == "eq36":

        def c(A):
            return H @ A @ H

        return (c(a * K) * c(a * L)).sum() / (c(mK * K).norm() * c(mL * L).norm())
    if variant == "mknn":
        return a.sum() / (N * k)
    raise ValueError(variant)


@pytest.fixture(scope="module")
def pair():
    shared = torch.randn(N, 5, generator=torch.Generator().manual_seed(0))
    xa, xb = (
        pa.prepare_layers(_feats(3, 24, shared, 1)),
        pa.prepare_layers(_feats(2, 32, shared, 2)),
    )
    return cv.Stack(xa), cv.Stack(xb), xa, xb


@pytest.mark.parametrize("k", [2, 7, 25, 40, N - 1])
def test_variants_match_definitions(pair, k):
    sa, sb, xa, xb = pair
    a, b = sa.at_k(k), sb.at_k(k)
    s = cv.per_k_scores(a, b, b.right())
    for i in range(len(sa)):
        for j in range(len(sb)):
            for name in ("mknn", "paper", "centred", "eq36"):
                ref = float(_literal(xa[i], xb[j], k, name))
                assert float(s[name][i, j]) == pytest.approx(ref, rel=2e-4, abs=1e-6), name
            ref = M.AlignmentMetrics.cknna(xa[i], xb[j], topk=k)
            assert float(s["code"][i, j]) == pytest.approx(ref, rel=2e-3, abs=1e-5)
    assert float(s["paper"].max()) <= 1 + 1e-5 and float(s["centred"].max()) <= 1 + 1e-5


def test_cka_matches_upstream(pair):
    sa, sb, xa, xb = pair
    ref = M.AlignmentMetrics.measure("cka", xa[2], xb[1])
    assert float(cv.cka(sa, sb)[2, 1]) == pytest.approx(ref, rel=1e-4)


def test_centred_at_full_k_is_offdiagonal_cka(pair):
    """At k = n-1 'centred' is sum_{i!=j} Kc Lc normalised: CKA with the diagonal removed."""
    sa, sb, _, _ = pair
    a, b = sa.at_k(N - 1), sb.at_k(N - 1)
    s = cv.per_k_scores(a, b, b.right())["centred"]
    off = ~torch.eye(N, dtype=torch.bool)
    Kc, Lc = sa.Kc[0][off], sb.Kc[1][off]
    assert float(s[0, 1]) == pytest.approx(float((Kc @ Lc) / (Kc.norm() * Lc.norm())), rel=1e-5)


def test_permuted_equals_recomputed(pair):
    """The null relabels b's samples; that must equal scoring against permuted features."""
    sa, sb, _, xb = pair
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(3))
    sp = cv.Stack([x[perm] for x in xb])
    for k in (5, 30):
        a, b, bp = sa.at_k(k), sb.at_k(k), sp.at_k(k)
        null = cv.per_k_scores(a, b, b.permuted(perm))
        ref = cv.per_k_scores(a, bp, bp.right())
        for name in cv.PER_K:
            torch.testing.assert_close(null[name], ref[name], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(cv.cka(sa, sb, perm), cv.cka(sa, sp), rtol=1e-4, atol=1e-6)


def test_calibration_matches_aristotelian(pair):
    """exp-008's null for 'code' and CKA reproduces Aristotelian's own gated calibration."""
    import importlib.util
    from pathlib import Path

    from aristotelian.experiments.layerwise_engine import (
        compute_alignment_gated_cka_cached,
        compute_alignment_gated_cknna_cached,
    )

    path = Path(__file__).resolve().parents[1] / "experiments/exp-008-cknna-null/calibrate.py"
    spec = importlib.util.spec_from_file_location("exp008", path)
    exp008 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp008)

    sa, sb, xa, xb = pair
    perms = exp008.make_perms(N, 30, seed=0)
    res, _ = exp008.calibrate_pair(sa, sb, [8], perms, alpha=0.05)
    grams_a, grams_b = [x @ x.T for x in xa], [y @ y.T for y in xb]
    ref = compute_alignment_gated_cknna_cached(
        grams_a, grams_b, topk=8, num_permutations=30, seed=0
    )
    ref_cka = compute_alignment_gated_cka_cached(
        grams_a, grams_b, num_permutations=30, seed=0, unbiased=False
    )
    for ours, theirs in ((res["per_k"][0]["code"], ref), (res["cka"], ref_cka)):
        for key in ("raw_score", "tau_alpha", "mu0", "g_score"):
            assert ours[key] == pytest.approx(theirs[key], rel=1e-3, abs=1e-5), key
        assert ours["p_value"] == pytest.approx(theirs["p_value"])
