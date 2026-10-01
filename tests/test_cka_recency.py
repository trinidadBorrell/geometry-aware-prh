from prh_replication.cka_recency import native_lookup, pair_ordinal, rank_family


def test_pair_ordinal_mean_and_vl():
    assert pair_ordinal(1, 3) == 2.0
    assert pair_ordinal(2, None) == 2.0
    assert pair_ordinal(None, 3) == 3.0
    assert pair_ordinal(None, None) is None


def test_native_lookup_either_order():
    rows = [{"a": "qwen2-7b", "b": "dinov2-small", "cka_a": 0.5}]
    assert native_lookup(rows, "qwen2-7b", "dinov2-small")["cka_a"] == 0.5
    assert native_lookup(rows, "dinov2-small", "qwen2-7b")["cka_a"] == 0.5
    assert native_lookup(rows, "qwen2-7b", "missing") is None


def test_rank_family_orders_by_release_date():
    manifest = {
        "base_panel": [
            {"key": "qwen2.5-7b", "family": "Qwen", "public_release_date": "2024-09-19"},
            {"key": "qwen2-7b", "family": "Qwen", "public_release_date": "2024-06-06"},
            {"key": "qwen3-8b-base", "family": "Qwen", "public_release_date": "2025-04-29"},
        ],
        "supplementary_qwen3x": [],
    }
    ranks = rank_family(manifest, ["qwen2.5-7b", "qwen3-8b-base", "qwen2-7b"])
    assert ranks == {"qwen2-7b": 1, "qwen2.5-7b": 2, "qwen3-8b-base": 3}
