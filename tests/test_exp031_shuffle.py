"""exp-031: word shuffles, transformed cache variants and the trend analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from geoprh import activation_cache as ac

ROOT = Path(__file__).resolve().parents[1]
X = ROOT / "experiments" / "exp-031-shuffled-capability"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ex = _load("exp031_extract", X / "extract.py")
an = _load("exp031_analyze", X / "analyze.py")

CAPTIONS = [
    "A red tram on a city street.",
    "Portrait of Johann Sebastian Bach (1746)",
    "Map",
    "",
    "Kölner Dom   at night, seen from the Rhine",
]


def test_word_shuffle_keeps_the_bag_of_words():
    for seed in (0, 1):
        new = ex.transform_texts(CAPTIONS, f"shufw{seed}")
        assert len(new) == len(CAPTIONS)
        for old, s in zip(CAPTIONS, new, strict=True):
            assert Counter(s.split()) == Counter(old.split())


def test_word_shuffle_is_deterministic_and_seeded():
    a = ex.transform_texts(CAPTIONS * 20, "shufw0")
    assert a == ex.transform_texts(CAPTIONS * 20, "shufw0")
    b = ex.transform_texts(CAPTIONS * 20, "shufw1")
    assert a != b
    # rows with the same caption get different permutations (rng is keyed by row)
    long_rows = [s for s, c in zip(a, CAPTIONS * 20, strict=True) if c == CAPTIONS[0]]
    assert len(set(long_rows)) > 1


def test_short_captions_and_reext():
    assert ex.transform_texts(["Map", ""], "shufw0") == ["Map", ""]
    assert ex.transform_texts(CAPTIONS, "reext") == CAPTIONS
    with pytest.raises(ValueError):
        ex.parse_transform("shuffle")


def test_variant_key_is_unchanged_without_transform():
    assert ac.Variant().key == "minhuh_prh_wit_1024_pool-avg_cid-0"
    assert ac.Variant(transform="shufw0").key == "minhuh_prh_wit_1024_pool-avg_cid-0_tf-shufw0"
    vis = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    assert vis.key == "minhuh_prh_wit_1024_pool-cls"


def test_transformed_variant_round_trip(tmp_path):
    feats = torch.randn(6, 3, 4, dtype=torch.bfloat16)
    src = tmp_path / "f.pt"
    torch.save({"feats": feats, "num_params": 10, "loss": 1.0, "bpb": 2.0}, src)
    var = ac.Variant(transform="shufw1")
    assert ac.push_model(tmp_path / "cache", "org/m", var, src, extractor="t", seed=1) == 3
    assert ac.load_model(tmp_path / "cache", "org/m", ac.Variant()) is None
    got = ac.load_model(tmp_path / "cache", "org/m", var)
    assert torch.equal(got["feats"], feats) and got["bpb"] == 2.0
    meta = json.loads(ac.layer_paths(tmp_path / "cache", "org/m", 0, var)[1].read_text())
    assert meta["transform"] == "shufw1" and meta["seed"] == 1


def test_token_shuffle_keeps_specials_and_padding():
    ids = torch.tensor([[0, 0, 1, 5, 6, 7, 8], [1, 9, 10, 11, 12, 13, 14]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    new = ex.token_shuffle(ids, mask, {0, 1}, np.random.default_rng(0))
    assert torch.equal(new[:, :3][mask[:, :3] == 0], ids[:, :3][mask[:, :3] == 0])
    assert new[0, 2] == 1 and new[1, 0] == 1  # BOS stays first
    for r in range(2):
        assert sorted(new[r].tolist()) == sorted(ids[r].tolist())
    assert not torch.equal(new, ids)


def _write_pairs(path: Path, scores: np.ndarray, llms, lvms, *, stored_format: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, lm in enumerate(llms):
        for j, v in enumerate(lvms):
            res = {}
            for m in ("mutual_knn_k10", "cka_lin"):
                r = {
                    "raw_score": float(scores[i, j] + 0.01),
                    "mu0": 0.01,
                    "best_indices": [3, 5],
                    "g_score": float(scores[i, j]),
                }
                res["mutual_knn" if stored_format and m.startswith("mutual") else m] = r
            rec = {"llm": lm, "lvm": v, **(res if stored_format else {"results": res})}
            lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")


def test_analysis_separates_a_trend_from_none(tmp_path, monkeypatch):
    llms = [f"bigscience/bloomz-{i}" for i in range(5)] + [f"other/m{i}" for i in range(5)]
    lvms = [f"vit_{j}" for j in range(4)]
    bpb = np.linspace(2.0, 1.0, len(llms))  # later LLMs are better
    rng = np.random.default_rng(0)
    intact = (1 - bpb)[:, None] * 0.1 + 0.3 + rng.normal(0, 1e-3, (len(llms), len(lvms)))
    flat = 0.1 + rng.normal(0, 1e-2, intact.shape)
    stored = tmp_path / "prh_val"
    stored.mkdir()
    info = {
        m: {"bpb": float(b), "num_layers": 6, "num_params": 1}
        for m, b in zip(llms, bpb, strict=True)
    }
    (stored / "cross_modal.json").write_text(
        json.dumps({"llms": llms, "lvms": lvms, "models": info})
    )
    _write_pairs(
        tmp_path / "cal" / "calibrated_pairs.jsonl", intact, llms, lvms, stored_format=True
    )
    out = tmp_path / "out"
    for s in (0, 1):
        _write_pairs(
            out / f"shufw{s}" / "calibrated_pairs.jsonl", flat, llms, lvms, stored_format=False
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--out",
            str(out),
            "--stored",
            str(stored),
            "--stored-cal",
            str(tmp_path / "cal"),
        ],
    )
    an.main()
    res = json.loads((out / "trends.json").read_text())
    assert res["trends"]["I/mutual_knn_k10"]["median"] > 0.9
    assert abs(res["trends"]["S0/mutual_knn_k10"]["median"]) < 0.9
    assert res["wins"]["S0/mutual_knn_k10"] == len(llms) * len(lvms)
    assert res["depth"][f"{llms[0]}/I/cka_lin"] == pytest.approx(1.0)
    verdict = {(v[0], v[1]): v[3] for v in res["verdicts"]}
    assert verdict[("P2", "S0")] == "holds"
    assert (out / "fig_trend.png").exists()
    assert "reproduction" not in res

    # With same-GPU intact rows, they become the primary intact condition and the stored rows a
    # reproduction check.
    _write_pairs(
        out / "reext" / "calibrated_pairs.jsonl", intact + 0.001, llms, lvms, stored_format=False
    )
    an.main()
    res = json.loads((out / "trends.json").read_text())
    assert res["reproduction"]["mutual_knn_k10"]["median_abs_diff"] == pytest.approx(0.001)
    assert res["trends"]["Ist/mutual_knn_k10"]["median"] > 0.9
    assert {v[1] for v in res["verdicts"] if v[0] == "P1"} == {"S0", "S1"}
