"""exp-006: extract PRH-style activations for up to 10k WIT samples into the shared cache.

Data: the first N rows of `askoepke/wit_1m_recaptioned` (config `wit_1m`, which excludes the
1024 PRH query images), with the original WIT caption. The subset is saved once as parquet so
every model sees exactly the same samples.

Extraction mirrors platonic-rep `extract_features.py`:
- language: every hidden state of the causal LM (0 = embeddings), mean-pooled over the attention
  mask; platonic-rep `load_llm` / `load_tokenizer` (bf16, left padding). Padding is per batch
  instead of over the whole dataset; masked mean pooling makes that immaterial.
- vision: timm ViT `blocks.{i}.add_1` outputs, CLS token, the model's own eval transform.

Activations go to the shared cache (`geoprh.activation_cache`, never overwritten) under the
dataset key `askoepke_wit_1m_recaptioned_first<N>-original_pool-<pool>`.

    uv run python experiments/exp-006-sample-size/extract.py --n 10240 \\
        --data /workspace/results/oddharak/exp-006-sample-size/wit_1m_first10240.parquet
"""

from __future__ import annotations

import argparse
import gc
import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import trange

from geoprh import activation_cache as ac

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "platonic-rep"))
from models import load_llm, load_tokenizer  # noqa: E402  (platonic-rep, not a package)

DATASET = "askoepke/wit_1m_recaptioned"
LLMS = ["bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7"]
LVMS = [
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
]
# as platonic-rep extract_features.py (timm>=1.0.26 ViTs take is_causal in forward)
VIT_FX_CONCRETE_ARGS = {"attn_mask": None, "is_causal": False}


def variant(n: int, modality: str) -> ac.Variant:
    pool = "avg" if modality == "language" else "cls"
    return ac.Variant(
        dataset=DATASET, subset=f"first{n}-original", pool=pool, caption_idx=None, modality=modality
    )


def load_subset(path: Path, n: int) -> pd.DataFrame:
    """First n rows of wit_1m (image bytes, original caption, url), cached as parquet."""
    if path.exists():
        df = pd.read_parquet(path)
        if len(df) >= n:
            return df.iloc[:n]
    from datasets import load_dataset

    stream = load_dataset(DATASET, "wit_1m", split="train", streaming=True)
    rows = []
    for row in stream:
        img = row["image"]
        data = img["bytes"] if isinstance(img, dict) else img
        if not isinstance(data, bytes):  # decoded PIL image
            buf = io.BytesIO()
            data.convert("RGB").save(buf, format="JPEG", quality=95)
            data = buf.getvalue()
        rows.append({"image": data, "caption": row["original_caption"], "url": row["url"]})
        if len(rows) == n:
            break
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"saved {len(df)} samples to {path}")
    return df


def _push(root: Path, model: str, var: ac.Variant, feats: torch.Tensor, extras: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "feats.pt"
        torch.save({"feats": feats, **extras}, f)
        n = ac.push_model(root, model, var, f, extractor="exp-006 extract.py (platonic-rep recipe)")
    print(f"[cache] {model}: {n} layer(s) written")


@torch.no_grad()
def extract_llm(name: str, texts: list[str], batch_size: int) -> tuple[torch.Tensor, dict]:
    model = load_llm(name)
    tokenizer = load_tokenizer(name)
    device = next(model.parameters()).device
    # the base transformer returns the same hidden_states without the LM head's
    # [batch, tokens, vocab] logits (250k vocab for bloom), which we do not need
    base = model.base_model
    feats = []
    for i in trange(0, len(texts), batch_size, desc=name):
        tok = tokenizer(texts[i : i + batch_size], padding="longest", return_tensors="pt").to(
            device
        )
        out = base(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            output_hidden_states=True,
        )
        hs = torch.stack(out["hidden_states"]).permute(1, 0, 2, 3)  # [b, L, t, d]
        mask = tok["attention_mask"].unsqueeze(-1).unsqueeze(1)
        feats.append(((hs * mask).sum(2) / mask.sum(2)).float().cpu())
    extras = {"num_params": sum(p.numel() for p in model.parameters())}
    del model
    return torch.cat(feats), extras


@torch.no_grad()
def extract_lvm(name: str, images: list[bytes], batch_size: int) -> tuple[torch.Tensor, dict]:
    model = timm.create_model(name, pretrained=True).cuda().eval()
    extras = {"num_params": sum(p.numel() for p in model.parameters())}
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
    model = create_feature_extractor(model, return_nodes=nodes, concrete_args=VIT_FX_CONCRETE_ARGS)
    feats = []
    for i in trange(0, len(images), batch_size, desc=name):
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in images[i : i + batch_size]]
        out = model(torch.stack([transform(im) for im in ims]).cuda())
        feats.append(torch.stack([v[:, 0, :] for v in out.values()], dim=1).float().cpu())
    del model
    return torch.cat(feats), extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10240)
    parser.add_argument("--data", type=Path, required=True, help="parquet cache of the subset")
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", default=None, help="comma-separated subset (default: all)")
    args = parser.parse_args()
    root = args.root or ac.default_root()
    df = load_subset(args.data, args.n)
    for modality, names in (("language", LLMS), ("vision", LVMS)):
        for name in names:
            if args.models and name not in args.models.split(","):
                continue
            var = variant(args.n, modality)
            if ac.load_model(root, name, var) is not None:
                print(f"[skip] {name}: cached ({var.key})")
                continue
            if modality == "language":
                feats, extras = extract_llm(name, df["caption"].tolist(), args.batch_size)
            else:
                feats, extras = extract_lvm(name, df["image"].tolist(), args.batch_size)
            print(f"{name}: feats {tuple(feats.shape)}")
            _push(root, name, var, feats, extras)
            del feats
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
