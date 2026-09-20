"""Deterministic COCO val2017 manifests and original-protocol notes."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image
from tqdm import tqdm

from prh_replication.io_utils import read_json, write_json
from prh_replication.registry import SEED, Paths

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _stable_bucket(image_id: int, seed: int = SEED) -> float:
    h = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def ensure_coco_val(paths: Paths) -> tuple[Path, Path]:
    raw = paths.data / "raw" / "coco"
    z_img = download_file(COCO_VAL_URL, raw / "val2017.zip")
    z_ann = download_file(COCO_ANN_URL, raw / "annotations_trainval2017.zip")
    img_dir = raw / "val2017"
    cap_json = raw / "annotations" / "captions_val2017.json"
    if not img_dir.exists():
        with zipfile.ZipFile(z_img) as z:
            z.extractall(raw)
    if not cap_json.exists():
        with zipfile.ZipFile(z_ann) as z:
            z.extractall(raw)
    return img_dir, cap_json


def build_coco_manifest(paths: Paths, n_train: int = 2048, n_val: int = 1024, n_test: int = 1024) -> Path:
    img_dir, cap_json = ensure_coco_val(paths)
    coco = json.loads(cap_json.read_text())
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    records = []
    failures = []
    for image_id, anns in sorted(by_image.items()):
        anns_sorted = sorted(anns, key=lambda a: a["id"])
        info = images[image_id]
        rel = f"val2017/{info['file_name']}"
        path = img_dir / info["file_name"]
        try:
            with Image.open(path) as im:
                im.verify()
            caption = anns_sorted[0]["caption"]
            n_caps = len(anns_sorted)
            cap_ids = [a["id"] for a in anns_sorted]
        except Exception as e:
            failures.append({"image_id": image_id, "error": str(e)})
            continue
        records.append(
            {
                "sample_id": f"coco-val2017-{image_id}",
                "image_id": image_id,
                "file_name": rel,
                "abs_path": str(path),
                "caption": caption,
                "caption_id": cap_ids[0],
                "all_caption_ids": cap_ids,
                "n_captions": n_caps,
                "caption_chars": len(caption),
                "caption_bytes": len(caption.encode("utf-8")),
                "width": info.get("width"),
                "height": info.get("height"),
            }
        )
    records.sort(key=lambda r: r["image_id"])
    scored = sorted(records, key=lambda r: (_stable_bucket(r["image_id"]), r["image_id"]))
    need = n_train + n_val + n_test
    if len(scored) < need:
        raise RuntimeError(f"Only {len(scored)} valid COCO val images; need {need}")
    train, val, test = scored[:n_train], scored[n_train : n_train + n_val], scored[n_train + n_val : need]
    leftover = scored[need:]
    payload = {
        "dataset": "coco_val2017",
        "source_images": COCO_VAL_URL,
        "source_captions": COCO_ANN_URL,
        "caption_rule": "first caption by increasing COCO annotation id; original captions only",
        "split_rule": f"sha256('{SEED}:'+image_id) then image_id; first {n_train}/{n_val}/{n_test}",
        "seed": SEED,
        "n_valid": len(records),
        "n_failed": len(failures),
        "splits": {
            "train": [r["sample_id"] for r in train],
            "val": [r["sample_id"] for r in val],
            "test": [r["sample_id"] for r in test],
            "heldout": [r["sample_id"] for r in leftover],
        },
        "records": {r["sample_id"]: r for r in records},
        "failures": failures,
        "protocol_labels": {
            "test_gallery_size": n_test,
            "val_size": n_val,
            "train_size": n_train,
            "not_original_wit_1024": True,
        },
    }
    out = paths.data / "manifests" / "coco_val2017.json"
    write_json(out, payload)
    # also copy a slim manifest into the repo if possible
    slim = {
        k: payload[k]
        for k in [
            "dataset",
            "source_images",
            "source_captions",
            "caption_rule",
            "split_rule",
            "seed",
            "n_valid",
            "n_failed",
            "splits",
            "protocol_labels",
        ]
    }
    slim["failures"] = failures
    write_json(paths.repo / "data" / "manifests" / "coco_val2017_splits.json", slim)
    return out


def load_manifest(path: Path) -> dict:
    return read_json(path)


def split_records(manifest: dict, split: str) -> list[dict]:
    ids = manifest["splits"][split]
    recs = manifest["records"]
    return [recs[i] for i in ids]
