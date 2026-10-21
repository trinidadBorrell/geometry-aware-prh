"""Final-block residual extraction (pre final RMS/LayerNorm)."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import torch
from tqdm import tqdm

from prh_replication.extract import _device_for, _dtype, feature_path
from prh_replication.io_utils import config_key, load_features, save_features
from prh_replication.registry import MODELS, ModelSpec, Paths


PROTOCOL = "final_block_pre_norm"


def spec_from_manifest(row: dict) -> ModelSpec:
    return ModelSpec(
        key=row["key"],
        kind="language",
        source="hf",
        checkpoint=row["checkpoint"],
        revision=row.get("revision"),
        family=row.get("family", ""),
        release_date=row.get("public_release_date", ""),
        release_source=row.get("public_release_source", ""),
        n_params=row.get("n_params"),
        hidden_size=row.get("hidden_size"),
        n_layers=row.get("n_layers"),
        training_stage=row.get("stage", ""),
        objective="causal LM / post-train as documented",
        modality="multimodal" if row.get("multimodal") else "text",
        language_supervised=True,
        native_dtype=row.get("inference_precision", "float16"),
        access="public",
        panels=(row.get("panel", "base"),),
        notes=row.get("ancestry", ""),
    )


def final_feature_path(paths: Paths, spec: ModelSpec, dataset: str, split: str, sample_ids: list[str]) -> Path:
    key = config_key(
        {
            "protocol": PROTOCOL,
            "model": spec.checkpoint,
            "revision": spec.revision,
            "dataset": dataset,
            "split": split,
            "kind": spec.kind,
            "pool": "mask_mean" if spec.kind == "language" else "cls_last_block",
            "n": len(sample_ids),
            "id0": sample_ids[0] if sample_ids else None,
            "idN": sample_ids[-1] if sample_ids else None,
        }
    )
    return paths.features / f"{dataset}_{PROTOCOL}" / split / spec.key / f"{key}.pt"


def _decoder_layers(model) -> tuple[torch.nn.ModuleList, str]:
    paths = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("model.transformer.blocks", lambda m: getattr(getattr(getattr(m, "model", None), "transformer", None), "blocks", None)),
        ("model.transformer.layers", lambda m: getattr(getattr(getattr(m, "model", None), "transformer", None), "layers", None)),
        ("model.model.layers", lambda m: getattr(getattr(getattr(m, "model", None), "model", None), "layers", None)),
        ("language_model.model.layers", lambda m: getattr(getattr(getattr(m, "language_model", None), "model", None), "layers", None)),
        ("model.language_model.layers", lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "layers", None)),
        ("model.language_model.model.layers", lambda m: getattr(getattr(getattr(getattr(m, "model", None), "language_model", None), "model", None), "layers", None)),
    ]
    # Qwen3.5 / VL wrappers
    inner = model
    for attr in ("model", "transformer", "language_model", "text_model"):
        if hasattr(inner, attr):
            cand = getattr(inner, attr)
            if hasattr(cand, "layers") and cand.layers is not None and len(list(cand.layers)) > 0:
                return cand.layers, f"{attr}.layers"
            inner = cand
    for name, fn in paths:
        layers = fn(model)
        if layers is not None and len(list(layers)) > 0:
            return layers, name
    raise RuntimeError(f"could not find decoder layers on {type(model)}")


def _pool_mask_mean(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(h.dtype).unsqueeze(-1)
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1)


def extract_language_final(
    spec: ModelSpec,
    records: list[dict],
    paths: Paths,
    dataset: str,
    split: str,
    batch_size: int = 8,
) -> Path:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    ids = [r["sample_id"] for r in records]
    out = final_feature_path(paths, spec, dataset, split, ids)
    if out.exists():
        cached = load_features(out)
        if list(cached["sample_ids"]) != ids:
            raise ValueError(f"Cache ID mismatch {out}")
        if cached.get("protocol") != PROTOCOL:
            raise ValueError(f"Wrong protocol in cache {out}")
        return out

    texts = [r["caption"] for r in records]
    n_params_est = (spec.n_params or 8_000_000_000) * 2
    device = _device_for("language", n_params_est)
    dtype = torch.float16 if device == "cpu" else _dtype(device)
    if device == "cuda" and batch_size < 16:
        batch_size = 32
    print(f"{spec.key} extract device={device} dtype={dtype} batch={batch_size}", flush=True)
    revision = spec.revision if spec.revision and len(spec.revision) >= 16 else None
    tok = AutoTokenizer.from_pretrained(spec.checkpoint, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    load_kw = dict(
        revision=revision,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = None
    loader = "causal"
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.checkpoint, **load_kw)
    except Exception as exc:
        loader = f"automodel_after_causal:{type(exc).__name__}"
        model = AutoModel.from_pretrained(spec.checkpoint, **load_kw)
    try:
        model = model.to(device).eval()
    except Exception:
        device = "cpu"
        model = model.to(dtype=torch.float16, device="cpu").eval()

    layers, layer_path = _decoder_layers(model)
    last = layers[-1]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h

    handle = last.register_forward_hook(_hook)
    feats = []
    token_lengths: list[int] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc=f"{spec.key}:{split}"):
            batch_texts = texts[start : start + batch_size]
            tokens = tok(
                batch_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            token_lengths.extend(int(value) for value in tokens["attention_mask"].sum(dim=1).tolist())
            tokens = {k: v.to(device) for k, v in tokens.items()}
            captured.clear()
            _ = model(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))
            if "h" not in captured:
                raise RuntimeError(f"hook did not fire for {spec.key}")
            h = captured["h"]
            if h.dim() != 3:
                raise RuntimeError(f"expected BTD residual, got {tuple(h.shape)} for {spec.key}")
            pooled = _pool_mask_mean(h, tokens["attention_mask"]).cpu()
            feats.append(pooled)
    handle.remove()
    all_feats = torch.cat(feats, dim=0)
    if all_feats.shape[0] != len(ids) or not torch.isfinite(all_feats.float()).all():
        raise ValueError("Language final-layer features invalid")
    payload = {
        "feats": all_feats.unsqueeze(1).half(),
        "sample_ids": ids,
        "layer_names": [f"{layer_path}[-1].residual"],
        "hidden_state_note": (
            "Last decoder-block residual via forward hook, before final RMS/LayerNorm. "
            "Attention-mask mean pool. Left pad. Raw captions. No chat template. No generation."
        ),
        "protocol": PROTOCOL,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "device": device,
        "loader": loader,
        "layer_path": layer_path,
        "hidden_dim": int(all_feats.shape[1]),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tok).__name__,
        "token_lengths": token_lengths,
        "truncation": False,
        "chat_template": False,
        "tokenizer_model_max_length": _tokenizer_limit(tok),
    }
    save_features(out, payload)
    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def slice_vision_final(paths: Paths, key: str, split: str, sample_ids: list[str]) -> Path:
    spec = MODELS[key]
    dest = final_feature_path(paths, spec, "coco_val2017", split, sample_ids)
    if dest.exists():
        cached = load_features(dest)
        if list(cached["sample_ids"]) != sample_ids:
            raise ValueError(f"Cache ID mismatch {dest}")
        return dest
    src = feature_path(paths, spec, "coco_val2017", split, "original", sample_ids)
    if not src.exists():
        raise FileNotFoundError(src)
    obj = load_features(src)
    if list(obj["sample_ids"]) != list(sample_ids):
        raise ValueError(f"id mismatch slicing vision {key} {split}")
    feats = obj["feats"]
    last = feats[:, -1, :]
    payload = {
        "feats": last.unsqueeze(1).contiguous(),
        "sample_ids": sample_ids,
        "layer_names": [str(obj.get("layer_names", ["blocks.last"])[-1])],
        "hidden_state_note": "Last ViT block CLS from existing original-protocol cache (already pre-final-LN).",
        "protocol": PROTOCOL,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "source_cache": str(src),
        "reused": True,
    }
    save_features(dest, payload)
    return dest


def load_final_prepared(paths: Paths, spec: ModelSpec, split: str, sample_ids: list[str]):
    from prh_replication.prh_ref import prh_prepare_layer

    p = final_feature_path(paths, spec, "coco_val2017", split, sample_ids)
    obj = load_features(p)
    if list(obj["sample_ids"]) != list(sample_ids):
        raise ValueError(f"id mismatch {spec.key} {split}")
    if obj.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol mismatch {p}")
    feats = obj["feats"]
    if feats.ndim == 3:
        x = feats[:, 0]
    else:
        x = feats
    return prh_prepare_layer(x), obj


def _tokenizer_limit(tokenizer) -> int | None:
    raw = getattr(tokenizer, "model_max_length", None)
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        return None
    if limit <= 0 or limit > 10**9:
        return None
    return limit


def _parameter_sha256(model) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        array = parameter.detach().float().cpu().contiguous().numpy()
        digest.update(array.shape.__repr__().encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        raw = getattr(value, "__dict__", None)
        return raw if isinstance(raw, dict) else {}


def _json_config(value):
    if isinstance(value, dict):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def vision_last_cls(spec: ModelSpec, records: list[dict], batch_size: int = 32) -> tuple[torch.Tensor, dict]:
    """Last ViT-block CLS, matching the archived all-layer hook without storing earlier blocks."""
    import timm
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    from prh_replication.extract import _device_for

    device = _device_for("vision", 300_000_000)
    try:
        model = timm.create_model(spec.checkpoint, pretrained=True).to(device).eval()
    except Exception:
        device = "cpu"
        model = timm.create_model(spec.checkpoint, pretrained=True).to(device).eval()
    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_config)
    last = model.blocks[-1]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["cls"] = output[:, 0, :].detach()

    handle = last.register_forward_hook(_hook)
    rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc=spec.key):
            batch = records[start : start + batch_size]
            images = []
            for record in batch:
                with Image.open(record["abs_path"]) as image:
                    images.append(transform(image.convert("RGB")))
            _ = model(torch.stack(images).to(device))
            if "cls" not in captured or captured["cls"].shape[0] != len(batch):
                raise RuntimeError(f"last-block hook failed for {spec.key}")
            rows.append(captured["cls"].float().cpu())
            captured.clear()
    handle.remove()
    features = torch.cat(rows, dim=0)
    if features.shape[0] != len(records) or not torch.isfinite(features).all():
        raise ValueError(f"vision features invalid for {spec.key}")
    pretrained = _mapping(getattr(model, "pretrained_cfg", {}) or {})
    meta = {
        "checkpoint": spec.checkpoint,
        "parameter_sha256": _parameter_sha256(model),
        "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": device,
        "model_precision": "float32",
        "feature_storage_precision": "float16",
        "n_blocks": len(model.blocks),
        "layer_name": f"blocks.{len(model.blocks) - 1}.output_cls",
        "pretrained_cfg": _json_config(
            {key: pretrained.get(key) for key in ("architecture", "tag", "file", "url", "hf_hub_id", "mean", "std", "input_size", "interpolation", "crop_pct")}
        ),
        "data_config": _json_config(
            {key: data_config.get(key) for key in ("input_size", "interpolation", "mean", "std", "crop_pct")}
        ),
    }
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return features, meta


def extract_vision_final(
    spec: ModelSpec,
    records: list[dict],
    paths: Paths,
    dataset: str,
    split: str,
    batch_size: int = 32,
) -> Path:
    ids = [record["sample_id"] for record in records]
    out = final_feature_path(paths, spec, dataset, split, ids)
    if out.exists():
        cached = load_features(out)
        if list(cached["sample_ids"]) != ids:
            raise ValueError(f"Cache ID mismatch {out}")
        if cached.get("protocol") != PROTOCOL:
            raise ValueError(f"Wrong protocol in cache {out}")
        return out
    features, meta = vision_last_cls(spec, records, batch_size=batch_size)
    payload = {
        "feats": features.unsqueeze(1).half().contiguous(),
        "sample_ids": ids,
        "layer_names": [meta["layer_name"]],
        "hidden_state_note": "CLS token of the last ViT block output (post-residual). Final LN not applied. Earlier blocks are not stored.",
        "protocol": PROTOCOL,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        **meta,
    }
    save_features(out, payload)
    return out
