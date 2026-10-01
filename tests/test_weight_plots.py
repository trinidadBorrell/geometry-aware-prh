"""diag(B) post-processing: order, mean/SD, not sorted eigs."""

from __future__ import annotations

import json

import numpy as np

from prh_replication.weight_plots import catalog_fits, collect_group, parse_fit_stem, sample_mean_sd


def test_parse_fit_stem_rho():
    a, b, rho = parse_fit_stem("qwen2-7b__dinov2-small__rho1.921812055672805")
    assert a == "qwen2-7b"
    assert b == "dinov2-small"
    assert rho == 1.921812055672805


def test_diag_b_is_not_sorted_eigenvalues(tmp_path):
    b = np.array(
        [
            [2.0, 0.5, 0.0, 0.0],
            [0.5, 1.2, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    s = np.zeros((4, 4))
    payload = {"b_a": b.tolist(), "s_a": s.tolist(), "rho": 0.1, "ok": True, "d_attained": 0.1}
    (tmp_path / "qwen2-7b__dinov2-small__rho0.1.json").write_text(json.dumps(payload))
    rows = catalog_fits(tmp_path)
    assert len(rows) == 1
    diag = rows[0]["diag_b"]
    eigs = np.sort(np.linalg.eigvalsh(b))[::-1]
    assert not np.allclose(diag, eigs)
    assert np.allclose(diag, np.diag(b))
    assert not np.allclose(diag, np.exp(np.diag(s)))


def test_sample_sd_unavailable_for_single_partner():
    mean, sd, ok = sample_mean_sd(np.ones((1, 3)), axis=0)
    assert ok is False
    assert np.allclose(mean, 1.0)
    assert np.isnan(sd).all()
    stack2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean2, sd2, ok2 = sample_mean_sd(stack2, axis=0)
    assert ok2 is True
    assert np.allclose(mean2, [2.0, 3.0])
    assert np.allclose(sd2, stack2.std(axis=0, ddof=1))


def test_collect_group_preserves_pc_columns(tmp_path):
    for partner, diag0 in (("dinov2-small", 1.5), ("clip-laion-base", 2.5), ("vit-in21k-small", 3.5)):
        b = np.eye(3)
        b[0, 0] = diag0
        payload = {"b_a": b.tolist(), "s_a": np.zeros((3, 3)).tolist(), "rho": 0.1}
        (tmp_path / f"qwen2-7b__{partner}__rho0.1.json").write_text(json.dumps(payload))
    rows = catalog_fits(tmp_path)
    names, diag, _s, notes = collect_group(rows, "qwen2-7b", 0.1, "vision")
    assert notes == []
    assert names == ["dinov2-small", "vit-in21k-small", "clip-laion-base"]
    assert diag.shape == (3, 3)
    assert diag[0, 0] == 1.5
    assert diag[1, 0] == 3.5
    assert diag[2, 0] == 2.5
