"""Resolve a dataset path. Downloads happen only when asked for.

A caller-supplied path is used as given. A missing path is an error and does
not trigger a download. Omitting the path returns an already complete cache
directory, or raises. Pass ``download=True`` to fill an empty cache. An
existing incomplete cache is left untouched.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rho_cca.records import RhoCCAError

Pull = Callable[["DatasetSource", Path], None]


@dataclass(frozen=True)
class DatasetSource:
    """One public dataset. ``marker`` is a path relative to the cache directory."""

    name: str
    kind: str
    marker: str
    url: str | None = None
    kaggle: str | None = None
    hf_repo: str | None = None
    hf_patterns: tuple[str, ...] = ()
    note: str = ""


DATASETS: dict[str, DatasetSource] = {
    "coco-images": DatasetSource(
        "coco-images",
        "zip",
        "val2017",
        url="http://images.cocodataset.org/zips/val2017.zip",
        note="COCO val2017 images, about 1 GB.",
    ),
    "coco-captions": DatasetSource(
        "coco-captions",
        "zip",
        "annotations/captions_val2017.json",
        url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        note="COCO train/val 2017 annotations. The caption file is captions_val2017.json.",
    ),
    "flickr-captions": DatasetSource(
        "flickr-captions",
        "zip",
        "dataset_flickr30k.json",
        url="https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
        note="Karpathy caption json. Test images use sentences[0] in file order.",
    ),
    "flickr-images": DatasetSource(
        "flickr-images",
        "kaggle",
        "flickr30k_images",
        kaggle="eeshawn/flickr30k",
        note="Kaggle mirror eeshawn/flickr30k. Needs a configured kaggle CLI. The Illinois original is request-only.",
    ),
    "nocaps": DatasetSource(
        "nocaps",
        "huggingface",
        "validation-*.parquet",
        hf_repo="lmms-lab-encoder/NoCaps",
        hf_patterns=("validation-*.parquet",),
        note="nocaps validation parquet. The official S3 bucket used by older loaders returns 403.",
    ),
    "imdb": DatasetSource(
        "imdb",
        "tar.gz",
        "aclImdb/README",
        url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        note="Stanford aclImdb v1 reviews.",
    ),
}


def resolve_dataset(
    name: str,
    location: str | Path | None = None,
    *,
    cache_root: str | Path,
    download: bool = False,
    pull_fn: Pull | None = None,
) -> Path:
    """Return ``location`` when it exists. Download only if ``download`` is true."""
    spec = DATASETS.get(name)
    if spec is None:
        known = ", ".join(sorted(DATASETS))
        raise RhoCCAError(f"unknown dataset {name}. Known datasets: {known}")
    if location is not None and str(location) != "":
        path = Path(location)
        if not path.exists():
            raise RhoCCAError(f"{name} path does not exist: {path}. Nothing was downloaded.")
        return path
    dest = Path(cache_root) / name
    if _ready(dest, spec):
        return dest
    if not download:
        raise RhoCCAError(
            f"{name} path was omitted and {dest} is not a complete cache. "
            f"Pass a path, or --download to store it in {dest}. Nothing was downloaded."
        )
    if dest.exists():
        raise RhoCCAError(f"{dest} already exists and is incomplete. Refusing to overwrite it.")
    (pull_fn or pull_dataset)(spec, dest)
    if not _ready(dest, spec):
        raise RhoCCAError(f"{name} download did not produce {spec.marker} under {dest}")
    return dest


def pull_dataset(spec: DatasetSource, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if spec.kind == "zip":
        _download_archive(spec, dest, zipfile.ZipFile)
    elif spec.kind == "tar.gz":
        _download_archive(spec, dest, tarfile.open)
    elif spec.kind == "kaggle":
        _pull_kaggle(spec, dest)
    elif spec.kind == "huggingface":
        _pull_huggingface(spec, dest)
    else:
        raise RhoCCAError(f"unsupported dataset kind {spec.kind}")


def _ready(dest: Path, spec: DatasetSource) -> bool:
    if not dest.exists():
        return False
    if spec.kind == "huggingface":
        return any(dest.glob(spec.marker))
    return (dest / spec.marker).exists()


def _download_archive(spec: DatasetSource, dest: Path, opener) -> None:
    if not spec.url:
        raise RhoCCAError(f"{spec.name} has no download URL")
    archive = dest / f".{spec.name}.download"
    request = urllib.request.Request(spec.url, headers={"User-Agent": "rho-cca"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with opener(archive) as packed:
            if hasattr(packed, "extractall"):
                packed.extractall(dest)
            else:
                packed.extractall(dest)
    finally:
        archive.unlink(missing_ok=True)


def _pull_kaggle(spec: DatasetSource, dest: Path) -> None:
    if shutil.which("kaggle") is None:
        raise RhoCCAError(
            f"{spec.name} is the Kaggle set {spec.kaggle}. Install and authenticate the kaggle CLI, or pass a local path."
        )
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", str(spec.kaggle), "-p", str(dest), "--unzip"],
        check=True,
    )


def _pull_huggingface(spec: DatasetSource, dest: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RhoCCAError(f"{spec.name} needs huggingface_hub, or pass a local directory of validation parquet files.") from exc
    snapshot_download(
        repo_id=str(spec.hf_repo),
        repo_type="dataset",
        allow_patterns=list(spec.hf_patterns),
        local_dir=str(dest),
    )
