import numpy as np
import pytest
from scipy.stats import spearmanr

from geoprh import paired


def test_rank_corr_is_spearman():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((3, 200)), rng.standard_normal((4, 200))
    b[1] = a[2] + 0.5 * b[1]
    m = paired.rank_corr_matrix(paired.standardized_ranks(a), paired.standardized_ranks(b))
    for i in range(3):
        for j in range(4):
            assert m[i, j] == pytest.approx(spearmanr(a[i], b[j]).statistic, abs=1e-12)


def test_residualize_removes_covariate():
    rng = np.random.default_rng(1)
    c = rng.standard_normal((2, 500))
    q = c + 0.3 * rng.standard_normal((2, 500))
    res = paired.residualize(q, c)
    zc = paired.standardized_ranks(c)
    assert np.abs((res * zc).mean(axis=1)).max() < 1e-12  # orthogonal to the covariate ranks
    assert res.std(axis=1) == pytest.approx(1.0)


def test_calibration_detects_shared_signal_only():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal(300)
    a = np.stack([shared + 0.5 * rng.standard_normal(300), rng.standard_normal(300)])
    b = np.stack([shared + 0.5 * rng.standard_normal(300)])
    res = paired.calibrated_rank_corr(
        paired.standardized_ranks(a), paired.standardized_ranks(b), permutations=100
    )
    assert res["p"][0, 0] < 0.02 and res["g"][0, 0] > 0.5
    assert res["p"][1, 0] > 0.05
