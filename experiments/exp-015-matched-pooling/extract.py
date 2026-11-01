"""exp-015: PRH wit_1024 activations with matched pooling across modalities.

The PRH cache has LLM features mean-pooled over the caption tokens (`pool-avg`) and ViT features
from the CLS token (`pool-cls`), so cross-modal comparisons mix two kinds of summary. This adds
the two missing ones, giving two matched pairs:
- one summary token:  LLM last token (`pool-last`)  x  ViT CLS token (`pool-cls`, cached)
- mean of tokens:     LLM mean (`pool-avg`, cached)  x  ViT mean of patch tokens (`pool-avg`)

Extraction mirrors platonic-rep `extract_features.py` (and exp-006): every hidden state of the
causal LM (0 = embeddings), platonic-rep `load_llm` / `load_tokenizer` (bf16, **left** padding,
so position -1 is each caption's last real token, as platonic-rep's own `last` option assumes);
timm ViT `blocks.{i}.add_1` outputs with the model's eval transform. ViT mean pooling averages
the patch tokens only, skipping the model's prefix tokens (CLS and any registers,
`num_prefix_tokens`). Data: `minhuh/prh` revision `wit_1024`, caption index 0, as the PRH cache.

`--pools` also accepts the cached poolings (LLM `avg`, ViT `cls`) to check this code reproduces
the cache (`--check`, writes nothing).

    uv run python experiments/exp-015-matched-pooling/extract.py --pools last,avg
    uv run python experiments/exp-015-matched-pooling/extract.py --models bigscience/bloomz-560m \\
        --pools avg --check --root results/activations
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import tempfile
from pathlib import Path

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import trange

from geoprh import activation_cache as ac

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "platonic-rep"))
from models import load_llm, load_tokenizer  # noqa: E402  (platonic-rep, not a package)

VIT_FX_CONCRETE_ARGS = {"attn_mask": None, "is_causal": False}  # as platonic-rep / exp-006
LLM_POOLS, VIT_POOLS = ("avg", "last"), ("cls", "avg")


def _exp012():
    path = REPO / "experiments" / "exp-012-id-mismatch" / "run.py"
    spec = importlib.util.spec_from_file_location("exp012_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def variant(model: str, pool: str, llms: list[str]) -> ac.Variant:
    language = model in llms
    return ac.Variant(
        dataset="minhuh/prh",
        subset="wit_1024",
        pool=pool,
        caption_idx=0 if language else None,
        modality="language" if language else "vision",
    )


@torch.no_grad()
def extract_llm(name: str, texts: list[str], pool: str, batch_size: int):
    model = load_llm(name)
    tokenizer = load_tokenizer(name)
    assert tokenizer.padding_side == "left"
    device = next(model.parameters()).device
    base = model.base_model  # hidden states without the LM head's vocab-sized logits
    feats = []
    for i in trange(0, len(texts), batch_size, desc=f"{name} {pool}"):
        tok = tokenizer(texts[i : i + batch_size], padding="longest", return_tensors="pt").to(
            device
        )
        out = base(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            output_hidden_states=True,
        )
        hs = torch.stack(out["hidden_states"]).permute(1, 0, 2, 3)  # [b, L, t, d]
        if pool == "avg":
            mask = tok["attention_mask"].unsqueeze(-1).unsqueeze(1)
            f = (hs * mask).sum(2) / mask.sum(2)
        else:  # last real token: left padding puts it at -1
            f = hs[:, :, -1, :]
        feats.append(f.float().cpu())
    extras = {"num_params": sum(p.numel() for p in model.parameters())}
    del model, base
    return torch.cat(feats), extras


@torch.no_grad()
def extract_vit(name: str, images: list, pool: str, batch_size: int, device: str = "cuda"):
    model = timm.create_model(name, pretrained=True).to(device).eval()
    extras = {"num_params": sum(p.numel() for p in model.parameters())}
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
    model = create_feature_extractor(model, return_nodes=nodes, concrete_args=VIT_FX_CONCRETE_ARGS)
    feats = []
    for i in trange(0, len(images), batch_size, desc=f"{name} {pool}"):
        ims = torch.stack([transform(im.convert("RGB")) for im in images[i : i + batch_size]])
        out = model(ims.to(device))
        if pool == "cls":
            f = torch.stack([v[:, 0, :] for v in out.values()], dim=1)
        else:  # mean of patch tokens, prefix tokens (CLS, registers) excluded
            f = torch.stack([v[:, n_prefix:, :].mean(1) for v in out.values()], dim=1)
        feats.append(f.float().cpu())
    extras["num_prefix_tokens"] = n_prefix
    del model
    return torch.cat(feats), extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--models", default="all", help="comma list or all")
    parser.add_argument(
        "--pools", default="last,avg", help="LLM pools in {avg,last}, ViT in {cls,avg}"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N samples (testing)")
    parser.add_argument(
        "--check", action="store_true", help="compare with the cache, write nothing"
    )
    args = parser.parse_args()
    from datasets import load_dataset

    e12 = _exp012()
    root = args.root or ac.default_root()
    data = load_dataset("minhuh/prh", revision="wit_1024", split="train")
    n = args.limit or len(data)
    texts = [str(data[i]["text"][0]) for i in range(n)]
    models = e12.LLMS + e12.VITS if args.models == "all" else args.models.split(",")
    pools = args.pools.split(",")
    for name in models:
        llm = name in e12.LLMS
        for pool in pools:
            if pool not in (LLM_POOLS if llm else VIT_POOLS):
                continue
            var = variant(name, pool, e12.LLMS)
            cached = ac.load_model(root, name, var)
            if cached is not None and not args.check:
                print(f"[skip] {name}: cached ({var.key})")
                continue
            if llm:
                feats, extras = extract_llm(name, texts, pool, args.batch_size)
            else:
                images = [data[i]["image"] for i in range(n)]
                feats, extras = extract_vit(name, images, pool, args.batch_size)
            if not torch.isfinite(feats).all():
                raise ValueError(f"{name} {pool}: non-finite features")
            if args.check:
                if cached is None:
                    print(f"[check] {name} {pool}: nothing cached to compare")
                    continue
                ref = cached["feats"][:n].float()
                rel = (feats - ref).norm(dim=-1) / ref.norm(dim=-1)
                cos = torch.nn.functional.cosine_similarity(feats, ref, dim=-1)
                print(
                    f"[check] {name} {pool}: shape {tuple(feats.shape)} vs {tuple(ref.shape)}; "
                    f"median rel. error {rel.median():.2e}, max {rel.max():.2e}; min cosine {cos.min():.6f}"
                )
                continue
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "feats.pt"
                torch.save({"feats": feats, **extras}, f)
                written = ac.push_model(
                    root, name, var, f, extractor="exp-015 extract.py (matched pooling)"
                )
            print(
                f"[cache] {name} {pool}: {tuple(feats.shape)}, {written} layer(s) written",
                flush=True,
            )
            del feats
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
