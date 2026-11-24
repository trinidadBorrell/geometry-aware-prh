"""A supplied path is never downloaded. An omitted path downloads only with download=True."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rho_cca import RhoCCAError, resolve_dataset
from rho_cca.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "experiments" / "fetch_datasets.py"


def _write_marker(spec, dest: Path) -> None:
    target = dest / spec.marker
    if "*" in spec.marker:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "validation-00000.parquet").write_bytes(b"parquet")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if spec.marker.endswith(".json") or spec.marker.endswith("README"):
        target.write_text("{}")
    else:
        target.mkdir(parents=True, exist_ok=True)


def test_a_supplied_path_is_not_downloaded(tmp_path: Path):
    ready = tmp_path / "local-flickr"
    ready.mkdir()
    calls = []
    found = resolve_dataset(
        "flickr-captions",
        ready,
        cache_root=tmp_path / "cache",
        download=True,
        pull_fn=lambda spec, dest: calls.append((spec.name, dest)),
    )
    assert found == ready
    assert calls == []


def test_an_omitted_path_does_not_download_unless_asked(tmp_path: Path):
    calls = []

    def pull(spec, dest: Path) -> None:
        calls.append(spec.name)
        _write_marker(spec, dest)

    with pytest.raises(RhoCCAError, match="Nothing was downloaded"):
        resolve_dataset("coco-captions", None, cache_root=tmp_path / "cache", pull_fn=pull)
    assert calls == []


def test_download_fills_an_empty_cache_once_and_refuses_a_partial_directory(tmp_path: Path):
    calls = []

    def pull(spec, dest: Path) -> None:
        calls.append(spec.name)
        _write_marker(spec, dest)

    cache = tmp_path / "cache"
    first = resolve_dataset("coco-captions", None, cache_root=cache, download=True, pull_fn=pull)
    second = resolve_dataset("coco-captions", None, cache_root=cache, download=True, pull_fn=pull)
    assert first == second == cache / "coco-captions"
    assert calls == ["coco-captions"]
    assert (first / "annotations" / "captions_val2017.json").is_file()
    partial = tmp_path / "partial" / "imdb"
    partial.mkdir(parents=True)
    (partial / "leftover.txt").write_text("keep")
    with pytest.raises(RhoCCAError, match="Refusing to overwrite"):
        resolve_dataset("imdb", None, cache_root=tmp_path / "partial", download=True, pull_fn=pull)
    assert (partial / "leftover.txt").read_text() == "keep"
    assert calls == ["coco-captions"]


def test_a_missing_supplied_path_is_an_error(tmp_path: Path):
    with pytest.raises(RhoCCAError, match="does not exist"):
        resolve_dataset("nocaps", tmp_path / "absent", cache_root=tmp_path, download=True, pull_fn=_write_marker)
    assert not (tmp_path / "nocaps").exists()


def test_the_fetch_script_uses_a_passed_path_and_refuses_to_pull(tmp_path: Path):
    captions = tmp_path / "dataset_flickr30k.json"
    captions.write_text(json.dumps({"images": [{"split": "test"}, {"split": "train"}]}))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    used = subprocess.run(
        [
            sys.executable,
            str(FETCH),
            "--only",
            "flickr-captions",
            "--flickr-captions",
            str(captions),
            "--cache",
            str(tmp_path / "cache"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert used.returncode == 0, used.stderr
    payload = json.loads(used.stdout)
    assert payload["path"] == str(captions)
    assert payload["inventory"].startswith("1 Karpathy")
    missing = subprocess.run(
        [sys.executable, str(FETCH), "--only", "imdb", "--cache", str(tmp_path / "empty")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["status"] == "fail"


def test_each_public_source_has_a_marker_and_a_fetch_method():
    for name, spec in DATASETS.items():
        assert spec.name == name
        assert spec.marker
        assert spec.url or spec.kaggle or spec.hf_repo
