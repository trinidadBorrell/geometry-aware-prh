#!/usr/bin/env python3
"""Extract missing features for cca_reliability_replication_v1 (resume-safe)."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from prh_replication.extract_final import (
    PROTOCOL,
    _decoder_layers,
    _pool_mask_mean,
    extract_vision_final,
    spec_from_manifest,
)
from prh_replication.io_utils import load_features, write_json
from prh_replication.registry import MODELS, Paths


def _language_specs(repo: Path) -> dict:
    release = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    return {row["key"]: spec_from_manifest(row) for row in release["base_panel"]}


FLICKR_DATASET = "flickr30k_replication_v1"


def _coco_image_records(paths: Paths, sample_ids: list[str]) -> list[dict]:
    """Build image records from sample_id → val2017 filename (manifest may be corrupted)."""
    records = []
    root = paths.data / "raw" / "coco" / "val2017"
    for sid in sample_ids:
        image_id = int(str(sid).rsplit("-", 1)[-1])
        file_name = f"{image_id:012d}.jpg"
        abs_path = root / file_name
        if not abs_path.exists():
            raise FileNotFoundError(abs_path)
        records.append(
            {
                "sample_id": sid,
                "file_name": file_name,
                "abs_path": str(abs_path),
                "path": str(abs_path),
            }
        )
    return records


def _flickr_vision_records(records: list[dict]) -> list[dict]:
    out = []
    for row in records:
        out.append(
            {
                "sample_id": row["sample_id"],
                "file_name": row["filename"],
                "filename": row["filename"],
                "abs_path": row["path"],
                "path": row["path"],
            }
        )
    return out


def _extract_flickr_captions(paths: Paths, spec, rows: list[dict], split: str) -> Path:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from prh_replication.extract import _device_for, _dtype

    key = spec.key
    dest = paths.features / f"{FLICKR_DATASET}_{PROTOCOL}" / split / key / "captions.pt"
    sample_ids = [row["sample_id"] for row in rows]
    caption_ids = [row["caption_ids"] for row in rows]
    if dest.exists():
        cached = load_features(dest)
        if list(cached["sample_ids"]) == sample_ids and cached.get("caption_ids") == caption_ids:
            print(f"  {key} {split} cached -> {dest}", flush=True)
            return dest
        raise RuntimeError(f"caption cache mismatch: {dest}")

    texts = [caption for row in rows for caption in row["captions"]]
    n_params_est = (spec.n_params or 8_000_000_000) * 2
    device = _device_for("language", n_params_est)
    if device == "cuda":
        free_bytes, _total = torch.cuda.mem_get_info()
        if free_bytes < 18 * (1 << 30):
            print(f"{key} {split} GPU free {free_bytes / (1 << 30):.1f} GiB; using CPU", flush=True)
            device = "cpu"
    dtype = torch.float16 if device == "cpu" else _dtype(device)
    batch_size = 8 if device == "cpu" else 16
    if device == "cpu":
        torch.set_num_threads(16)
    print(f"{key} {split} device={device} dtype={dtype} captions={len(texts)}", flush=True)
    revision = spec.revision if spec.revision and len(str(spec.revision)) >= 16 else None
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kw = dict(revision=revision, dtype=dtype, device_map=None, trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.checkpoint, **load_kw)
        loader = "causal"
    except Exception as exc:
        loader = f"automodel_after_causal:{type(exc).__name__}"
        model = AutoModel.from_pretrained(spec.checkpoint, **load_kw)
    try:
        model = model.to(device).eval()
    except Exception:
        device = "cpu"
        dtype = torch.float16
        model = model.to(dtype=dtype, device="cpu").eval()
    layers, layer_path = _decoder_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden

    handle = layers[-1].register_forward_hook(_hook)
    chunks = []
    try:
        with torch.no_grad():
            started = time.perf_counter()
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[start : start + batch_size], padding=True, truncation=False, return_tensors="pt")
                tokens = {name: value.to(device) for name, value in tokens.items()}
                captured.clear()
                _ = model(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))
                if "h" not in captured or captured["h"].dim() != 3:
                    raise RuntimeError(f"final residual hook failed for {key}")
                chunks.append(_pool_mask_mean(captured["h"], tokens["attention_mask"]).cpu())
                done = start + tokens["input_ids"].shape[0]
                if done <= batch_size * 2 or done % (batch_size * 20) == 0:
                    rate = done / max(time.perf_counter() - started, 1e-6)
                    print(f"{key} {split} {done}/{len(texts)} {rate:.2f} captions/s", flush=True)
    finally:
        handle.remove()
    feats = torch.cat(chunks, dim=0).reshape(len(rows), len(rows[0]["captions"]), -1)
    if not torch.isfinite(feats.float()).all():
        raise RuntimeError(f"non-finite captions for {key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feats": feats.half(),
        "sample_ids": sample_ids,
        "caption_ids": caption_ids,
        "protocol": PROTOCOL,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "layer_path": layer_path,
        "hidden_dim": int(feats.shape[-1]),
        "device": device,
        "dtype": str(dtype),
        "loader": loader,
        "model_class": type(model).__name__,
        "pooling": "mask_mean",
        "caption_rule": "first five Karpathy sentences",
    }
    tmp = dest.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(dest)
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  wrote {dest}", flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--stage",
        choices=["coco_siglip", "flickr_vision", "flickr_language", "all"],
        default="all",
    )
    args = parser.parse_args()
    work = args.work.resolve()
    repo = args.repo.resolve()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / "cca_reliability_replication_v1.json").read_text())
    out = paths.results / "cca_reliability_replication_v1"
    status: dict = {"stages": {}}

    if args.stage in ("coco_siglip", "all"):
        print("=== COCO siglip-base ===", flush=True)
        split = json.loads((out / "split_manifest_coco.json").read_text())
        spec = MODELS["siglip-base"]
        for split_name, ids in (("train", split["train_ids"]), ("test", split["test_ids"])):
            records = _coco_image_records(paths, ids)
            path = extract_vision_final(spec, records, paths, "coco_val2017", split_name)
            print(f"  {split_name} -> {path}", flush=True)
        status["stages"]["coco_siglip"] = "done"

    if args.stage in ("flickr_vision", "all"):
        print("=== Flickr vision ===", flush=True)
        train_recs = _flickr_vision_records(json.loads((out / "flickr_records_train.json").read_text()))
        test_recs = _flickr_vision_records(json.loads((out / "flickr_records_test.json").read_text()))
        visions = list(config["anchor_visions"]) + list(config["new_visions"])
        for key in visions:
            spec = MODELS[key]
            for split_name, records in (("train", train_recs), ("test", test_recs)):
                path = extract_vision_final(spec, records, paths, FLICKR_DATASET, split_name)
                print(f"  {key} {split_name} -> {path}", flush=True)
        status["stages"]["flickr_vision"] = "done"

    if args.stage in ("flickr_language", "all"):
        print("=== Flickr language 5-cap ===", flush=True)
        train_raw = json.loads((out / "flickr_records_train.json").read_text())
        test_raw = json.loads((out / "flickr_records_test.json").read_text())
        lang_specs = _language_specs(repo)
        languages = list(config["anchor_languages"]) + list(config["new_languages"])
        for key in languages:
            if key not in lang_specs:
                raise KeyError(f"language {key} missing from release_models.json base_panel")
            for split_name, raw in (("train", train_raw), ("test", test_raw)):
                rows = [
                    {
                        "sample_id": r["sample_id"],
                        "captions": r["captions"],
                        "caption_ids": [f"{r['sample_id']}:c{i}" for i in range(len(r["captions"]))],
                    }
                    for r in raw
                ]
                _extract_flickr_captions(paths, lang_specs[key], rows, split_name)
        status["stages"]["flickr_language"] = "done"

    write_json(out / "extraction_status.json", status)
    print(status, flush=True)


if __name__ == "__main__":
    main()
