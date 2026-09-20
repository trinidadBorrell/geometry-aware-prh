"""Sequential feature extraction (one model at a time)."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from prh_replication.io_utils import config_key, load_features, save_features
from prh_replication.registry import ModelSpec, Paths


def _device_for(kind: str, param_bytes: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    # Leave the resident vLLM process alone: only use a small remainder.
    headroom = 1.5 * (1 << 30)
    if kind == "vision" and free > 2 * (1 << 30):
        return "cuda"
    if kind == "language" and free > param_bytes + headroom:
        return "cuda"
    return "cpu"


def _dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def feature_path(paths: Paths, spec: ModelSpec, dataset: str, split: str, protocol: str, sample_ids: list[str] | None = None) -> Path:
    del protocol  # extraction is shared; protocol only changes metric prep
    key = config_key(
        {
            "model": spec.checkpoint,
            "revision": spec.revision,
            "dataset": dataset,
            "split": split,
            "kind": spec.kind,
            "pool": "avg" if spec.kind == "language" else "cls",
            "max_length": None,
            "n": len(sample_ids) if sample_ids is not None else None,
            "id0": sample_ids[0] if sample_ids else None,
            "idN": sample_ids[-1] if sample_ids else None,
        }
    )
    return paths.features / dataset / split / spec.key / f"{key}.pt"


def extract_vision(
    spec: ModelSpec,
    records: list[dict],
    paths: Paths,
    dataset: str,
    split: str,
    protocol: str,
    batch_size: int = 8,
) -> Path:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    ids = [r["sample_id"] for r in records]
    out = feature_path(paths, spec, dataset, split, protocol, ids)
    if out.exists():
        cached = load_features(out)
        if list(cached["sample_ids"]) != ids:
            raise ValueError(f"Cache ID mismatch {out}")
        return out

    device = _device_for("vision", 300_000_000)
    dtype = torch.float32  # timm preprocess + hooks more stable in fp32
    try:
        model = timm.create_model(spec.checkpoint, pretrained=True).to(device).eval()
    except Exception:
        device = "cpu"
        model = timm.create_model(spec.checkpoint, pretrained=True).to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    n_blocks = len(model.blocks)
    captured: list[torch.Tensor] = []

    def make_hook(i):
        def _hook(_m, _inp, output):
            captured[i] = output[:, 0, :].detach()

        return _hook

    handles = []
    for i, blk in enumerate(model.blocks):
        captured.append(torch.empty(0))
        handles.append(blk.register_forward_hook(make_hook(i)))

    feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc=spec.key):
            batch = records[start : start + batch_size]
            ims = []
            for rec in batch:
                with Image.open(rec["abs_path"]) as im:
                    ims.append(transform(im.convert("RGB")))
            x = torch.stack(ims).to(device)
            _ = model(x)
            layer_stack = torch.stack(captured, dim=1).cpu()  # B, L, D
            feats.append(layer_stack)
    for h in handles:
        h.remove()
    all_feats = torch.cat(feats, dim=0)
    if all_feats.shape[0] != len(ids):
        raise ValueError("Vision feature count mismatch")
    payload = {
        "feats": all_feats.half(),
        "sample_ids": ids,
        "layer_names": [f"blocks.{i}.output_cls" for i in range(n_blocks)],
        "hidden_state_note": "CLS token of each ViT block output (post-residual). Final LN not applied. Matches original add_1 CLS intent.",
        "model_precision": str(dtype),
        "feature_storage_precision": "float16",
        "protocol": protocol,
        "checkpoint": spec.checkpoint,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "device": device,
    }
    save_features(out, payload)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _pool_hidden(hidden_states, attention_mask, pool: str) -> torch.Tensor:
    # hidden: tuple L+1 of B, T, D including embedding
    stacked = torch.stack(hidden_states, dim=1)  # B, L, T, D
    if pool == "avg":
        mask = attention_mask[:, None, :, None].to(stacked.dtype)
        return (stacked * mask).sum(2) / mask.sum(2).clamp(min=1)
    if pool == "last":
        # left padding: last position is the true last token
        return stacked[:, :, -1, :]
    raise ValueError(pool)


def extract_language(
    spec: ModelSpec,
    records: list[dict],
    paths: Paths,
    dataset: str,
    split: str,
    protocol: str,
    batch_size: int = 2,
    max_length: int | None = None,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ids = [r["sample_id"] for r in records]
    out = feature_path(paths, spec, dataset, split, protocol, ids)
    if out.exists():
        cached = load_features(out)
        if list(cached["sample_ids"]) != ids:
            raise ValueError(f"Cache ID mismatch {out}")
        return out

    texts = [r["caption"] for r in records]
    n_params_est = (spec.n_params or 1_000_000_000) * 2
    device = _device_for("language", n_params_est)
    dtype = _dtype(device)
    revision = spec.revision if spec.revision and len(spec.revision) >= 16 else None
    tok = AutoTokenizer.from_pretrained(spec.checkpoint, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        spec.checkpoint,
        revision=revision,
        torch_dtype=dtype,
        device_map=None,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    try:
        model = model.to(device).eval()
    except Exception:
        device = "cpu"
        dtype = torch.float32
        model = model.to(dtype=dtype, device=device).eval()
    pool = "avg"
    max_length = None  # COCO captions are short; never truncate in this phase

    feats = []
    losses = []
    bpbs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc=spec.key):
            batch_texts = texts[start : start + batch_size]
            tokens = tok(
                batch_texts,
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out_m = model(**tokens, output_hidden_states=True)
            pooled = _pool_hidden(out_m.hidden_states, tokens["attention_mask"], pool).cpu()
            feats.append(pooled)
            # token NLL on captions (reduced capability signal, not OpenWebText)
            logits = out_m.logits[:, :-1]
            labels = tokens["input_ids"][:, 1:]
            mask = tokens["attention_mask"][:, 1:].float()
            logp = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            nats = (logp * mask).sum(dim=1)
            losses.append((nats / mask.sum(dim=1).clamp(min=1)).cpu())
            nbytes = torch.tensor([len(s.encode("utf-8")) for s in batch_texts], dtype=torch.float32)
            bpbs.append((nats.cpu() * math.log2(math.e)) / nbytes)

    all_feats = torch.cat(feats, dim=0)
    if all_feats.shape[0] != len(ids) or not torch.isfinite(all_feats.float()).all():
        raise ValueError("Language features invalid")
    payload = {
        "feats": all_feats.half(),
        "sample_ids": ids,
        "layer_names": [f"hidden_states.{i}" for i in range(all_feats.shape[1])],
        "hidden_state_note": "All HF hidden_states including embeddings. Attention-mask mean pool. Left padding. No chat template.",
        "model_precision": str(dtype),
        "feature_storage_precision": "float16",
        "protocol": protocol,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "device": device,
        "caption_mean_nll": float(torch.cat(losses).mean()),
        "caption_mean_bpb": float(torch.cat(bpbs).mean()),
        "max_length": max_length,
        "pool": pool,
        "bos_eos": "tokenizer defaults; no extra BOS/EOS injected",
    }
    save_features(out, payload)
    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def extract_model(spec: ModelSpec, records, paths, dataset, split, protocol, **kwargs) -> Path:
    if spec.kind == "vision":
        return extract_vision(spec, records, paths, dataset, split, protocol, **kwargs)
    return extract_language(spec, records, paths, dataset, split, protocol, **kwargs)
