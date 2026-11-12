"""exp-026: masked captions, and the chunked clamp against PRH's own."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DOC = [
    ("The ", True, "DET"),
    ("brown ", True, "ADJ"),
    ("dog", True, "NOUN"),
    (", ", False, "PUNCT"),
    ("on ", True, "ADP"),
    ("the ", True, "DET"),
    ("grass", True, "NOUN"),
    (".", False, "PUNCT"),
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MK = _load("exp026_masks_test", ROOT / "experiments/exp-026-word-masking/masks.py")


def test_remove_class_keeps_punctuation_and_order():
    assert MK.remove_class(DOC, MK.FUNCTION) == "brown dog, grass."
    assert MK.remove_class(DOC, MK.NOUN) == "The brown , on the ."


def test_only_class_keeps_words_in_order():
    assert MK.only_class(DOC, MK.NOUN) == "dog grass"


def test_random_controls_match_counts():
    rng = np.random.default_rng(0)
    n = MK.count_class(DOC, MK.FUNCTION)
    assert n == 3
    kept = MK.remove_random(DOC, n, rng)
    assert sum(w.strip(",.").isalpha() for w in kept.split()) == 6 - n
    assert "," in kept and "." in kept
    assert len(MK.keep_random(DOC, 2, rng).split()) == 2


def test_build_is_deterministic_and_intact_is_raw():
    a = MK.build([DOC, DOC], ["raw 1", "raw 2"], seed=3)
    b = MK.build([DOC, DOC], ["raw 1", "raw 2"], seed=3)
    assert a == b
    assert a["intact"] == ["raw 1", "raw 2"]


def test_orig_skips_nouns_only():
    assert "nouns" not in MK.conditions_for("orig")
    assert "nouns" in MK.conditions_for("gemini")
    assert len(MK.conditions_for("gemini")) == 10


def test_clamp_matches_prepare_features():
    from aristotelian.prh.preprocess import prepare_features

    x = torch.randn(600, 5, 8) * torch.linspace(0.5, 3, 5)[None, :, None]
    q = torch.quantile(x.abs().flatten(1), 0.95, dim=1).mean()
    assert torch.allclose(prepare_features(x, q=0.95, exact=False), x.clamp(-q, q))
    vals = [
        torch.quantile(x[s : s + 256].abs().flatten(1), 0.95, dim=1) for s in range(0, 600, 256)
    ]
    assert torch.allclose(torch.cat(vals).mean(), q)
