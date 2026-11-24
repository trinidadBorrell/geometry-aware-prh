"""Resolve COCO, Flickr30k, nocaps, and IMDb paths.

A path argument is used as given. Omitting a path does not download anything.
``--download`` stores an omitted dataset in ``--cache`` (default
``~/.cache/rho-cca/datasets``) when that cache directory is absent. An
incomplete directory is not overwritten. This script is not part of the
default pytest run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rho_cca.datasets import DATASETS, resolve_dataset
from rho_cca.records import RhoCCAError

FLAGS = {
    "coco-images": "coco_images",
    "coco-captions": "coco_captions",
    "flickr-captions": "flickr_captions",
    "flickr-images": "flickr_images",
    "nocaps": "nocaps",
    "imdb": "imdb",
}


def inventory(name: str, path: Path) -> str:
    if name == "coco-captions":
        file = path if path.is_file() else path / "annotations" / "captions_val2017.json"
        if file.is_file():
            payload = json.loads(file.read_text())
            return f"{len(payload.get('annotations', []))} captions in {file}"
    if name == "flickr-captions":
        file = path if path.is_file() else path / "dataset_flickr30k.json"
        if file.is_file():
            images = json.loads(file.read_text()).get("images", [])
            test = sum(1 for row in images if row.get("split") == "test")
            return f"{test} Karpathy test images in {file}"
    if name == "coco-images":
        root = path if path.name == "val2017" else path / "val2017"
        if root.is_dir():
            return f"{sum(1 for _ in root.glob('*.jpg'))} jpeg files in {root}"
    if name == "flickr-images":
        root = path if (path / "flickr30k_images").is_dir() is False and path.is_dir() else path / "flickr30k_images"
        if path.is_dir() and any(path.glob("*.jpg")):
            root = path
        if root.is_dir():
            return f"{sum(1 for _ in root.glob('*.jpg'))} jpeg files in {root}"
    if name == "nocaps":
        files = sorted(path.glob("validation-*.parquet")) if path.is_dir() else []
        return f"{len(files)} validation parquet files in {path}"
    if name == "imdb":
        root = path if (path / "train").is_dir() else path / "aclImdb"
        pos = root / "train" / "pos"
        neg = root / "train" / "neg"
        if pos.is_dir() and neg.is_dir():
            return f"{sum(1 for _ in pos.glob('*.txt'))} positive and {sum(1 for _ in neg.glob('*.txt'))} negative train reviews"
    return f"present at {path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "rho-cca" / "datasets")
    parser.add_argument("--only", nargs="+", choices=sorted(DATASETS), help="Datasets to resolve. Default: all six.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download omitted datasets into --cache. Existing directories are not overwritten.",
    )
    parser.add_argument("--coco-images", type=Path)
    parser.add_argument("--coco-captions", type=Path)
    parser.add_argument("--flickr-captions", type=Path)
    parser.add_argument("--flickr-images", type=Path)
    parser.add_argument("--nocaps", type=Path)
    parser.add_argument("--imdb", type=Path)
    args = parser.parse_args(argv)
    names = args.only or list(DATASETS)
    failed = False
    for name in names:
        given = getattr(args, FLAGS[name])
        try:
            if given is None and args.download:
                print(f"download {name} into {args.cache / name}: {DATASETS[name].note}", flush=True)
            path = resolve_dataset(name, given, cache_root=args.cache, download=args.download)
        except (RhoCCAError, OSError, subprocess.CalledProcessError) as exc:
            failed = True
            print(json.dumps({"dataset": name, "status": "fail", "detail": str(exc)}), flush=True)
            continue
        print(json.dumps({"dataset": name, "status": "ready", "path": str(path), "inventory": inventory(name, path)}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
