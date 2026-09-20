"""Final-block residual extraction (pre final RMS/LayerNorm)."""

from __future__ import annotations

import gc
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
    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc=f"{spec.key}:{split}"):
            batch_texts = texts[start : start + batch_size]
            tokens = tok(
                batch_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
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
