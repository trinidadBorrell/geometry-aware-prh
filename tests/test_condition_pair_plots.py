"""Pair order and seed medians for the four fitting conditions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prh_replication.condition_pair_plots import mode_values, ordered_pairs, plot_conditions_by_pair


def _row(kind, side_a, side_b, family, profile, mode, rho, excess, seed=0):
    return {
        "kind": kind,
        "side_a": side_a,
        "side_b": side_b,
        "family": family,
        "profile": profile,
        "mode": mode,
        "rho": rho,
        "seed": seed,
        "correspondence": "true",
        "exploratory_excess": excess,
    }


def test_vision_pairs_follow_release_then_vision_order():
    rows = [
        _row("vl", "olmo-7b-0724", "clip-laion-base", "identity", "na", "identity", 0.0, 0.2),
        _row("vl", "qwen2-7b", "vit-in21k-small", "identity", "na", "identity", 0.0, 0.3),
        _row("vl", "qwen2-7b", "dinov2-small", "identity", "na", "identity", 0.0, 0.4),
    ]
    assert ordered_pairs(rows, "vl") == [
        ("qwen2-7b", "dinov2-small"),
        ("qwen2-7b", "vit-in21k-small"),
        ("olmo-7b-0724", "clip-laion-base"),
    ]


def test_mode_values_use_identity_once_and_median_of_seeds():
    pair = ("qwen2-7b", "dinov2-small")
    rows = [
        _row("vl", *pair, "identity", "na", "identity", 0.0, 0.30),
        _row("vl", *pair, "exp_s", "na", "a_only", 0.1, 0.10, seed=0),
        _row("vl", *pair, "exp_s", "na", "a_only", 0.1, 0.50, seed=1),
        _row("vl", *pair, "exp_s", "na", "a_only", 0.1, 0.20, seed=2),
        _row("vl", *pair, "exp_s", "na", "a_only", 0.4, 0.90, seed=0),
        _row("vl", "qwen2-7b", "olmo-7b-0724", "exp_s", "na", "a_only", 0.1, 0.77, seed=0),
    ]
    assert mode_values(rows, kind="vl", family="exp_s", profile="na", mode="identity", rho=0.1, pair=pair) == [0.30]
    learned = mode_values(rows, kind="vl", family="exp_s", profile="na", mode="a_only", rho=0.1, pair=pair)
    assert learned == [0.10, 0.50, 0.20]
    assert mode_values(rows, kind="vl", family="exp_s", profile="na", mode="a_only", rho=0.4, pair=pair) == [0.90]


def test_plot_writes_png(tmp_path):
    pair = ("qwen2-7b", "dinov2-small")
    rows = [_row("vl", *pair, "identity", "na", "identity", 0.0, 0.30)]
    for mode, excess in (("a_only", 0.40), ("shared_pc", 0.45), ("separate", 0.55)):
        for seed in (0, 1, 2):
            rows.append(_row("vl", *pair, "exp_s", "na", mode, 0.1, excess + 0.01 * seed, seed=seed))
            rows.append(_row("vl", *pair, "exp_s", "na", mode, 0.4, excess + 0.02, seed=seed))
    path = tmp_path / "vl_exp_s.png"
    plot_conditions_by_pair(
        plt,
        rows,
        kind="vl",
        family="exp_s",
        profile="na",
        title="Vision–language, exp(S)",
        path=path,
    )
    assert path.exists()
    assert path.stat().st_size > 1000
