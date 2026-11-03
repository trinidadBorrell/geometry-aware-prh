"""exp-016 (WP8): features for the positive controls, ceilings and negative controls.

Datasets
- D1: PRH `minhuh/prh` wit_1024 (n = 1024, one original WIT caption per image)
- D2: the WIT-1M subset of exp-006 (first 10240 images with a non-empty original caption), with
  each image's original caption and its Gemini recaption (`prepare_d2.py`). Computed once at
  n = 10240 and saved twice: `d2s` (first 1024 rows, all layers) and `d2l` (all 10240 rows,
  7 evenly spaced layers, to limit storage).

Feature sets (`features/<key>__<d1|d2s|d2l>.pt`: {"layers": [n, L, d] float16 (float32 if float16
overflows), "layer_idx": [...], "final": [n, d'] (dual encoders only), "meta": {...}})
- Arm A (contrastive dual encoders): OpenAI CLIP ViT-L/14, OpenCLIP ViT-H/14 laion2b, SigLIP
  SO400M (open_clip). Text tower: every block's output at the pooled position (EOT token for CLIP,
  last position for SigLIP), plus the final projected embedding (`encode_text`; captions are
  truncated to the tower's context, 77 / 64 tokens). Image tower: every block's CLS token (CLIP)
  or token mean (SigLIP, no CLS), plus `encode_image`.
- LLMs (B1, B3): every hidden state mean-pooled over the caption tokens (platonic-rep recipe as
  exp-015, float32 accumulation), original and Gemini caption.
- ViTs (B2, B3, baseline): timm `blocks.{i}.add_1` CLS (exp-015 recipe); for B2 also on augmented
  images (random resized crop of 60-100% of the area at the image's own size, colour jitter,
  horizontal flip; 3 seeds, the same view of an image for a given seed).
- Negative control D: a randomly initialised ViT-B/16 (timm, seed 0), CLS.

    uv run python experiments/exp-016-positive-controls/extract.py --out <dir> --d2 <parquet>
    uv run python experiments/exp-016-positive-controls/extract.py --check   # tower hook check
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, trange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "platonic-rep"))
TOWERS = {
    "clip-L14": ("ViT-L-14", "openai"),
    "openclip-H14": ("ViT-H-14", "laion2b_s32b_b79k"),
    "siglip-so400m": ("ViT-SO400M-14-SigLIP", "webli"),
}
LLMS = [
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_7b",
    "huggyllama/llama-13b",
]
VITS = [
    "vit_large_patch16_224.augreg_in21k",
    "vit_huge_patch14_224.mae",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_huge_patch14_clip_224.laion2b",
]
RANDOM_VIT = "vit_base_patch16_224.augreg_in21k"
N_SMALL, N_LARGE, LARGE_DEPTHS = 1024, 10240, 7
MAX_LLM_TOKENS = 2048  # LLaMA context; Gemini captions are ~650 tokens (median)


def layer_subset(n_layers: int, n: int) -> list[int]:
    """All layers at n = 1024; LARGE_DEPTHS evenly spaced (first and last included) at 10240."""
    if n <= N_SMALL:
        return list(range(n_layers))
    return sorted(set(np.rint(np.linspace(0, n_layers - 1, LARGE_DEPTHS)).astype(int).tolist()))


def _half(x: torch.Tensor) -> torch.Tensor:
    """float16 unless that overflows (LLM outlier dimensions), then float32."""
    h = x.half()
    return h if torch.isfinite(h).all() else x.float()


def save(out: Path, key: str, tag: str, layers, final, meta: dict) -> None:
    """Write `<key>__d1`, or for D2 `<key>__d2s` (first 1024, all layers) and `<key>__d2l`."""
    for name, n in {"d1": [("d1", N_SMALL)], "d2": [("d2s", N_SMALL), ("d2l", N_LARGE)]}[tag]:
        idx = layer_subset(layers.shape[1], n)
        obj = {"layers": _half(layers[:n, idx]), "layer_idx": idx, "meta": {**meta, "n": n}}
        if final is not None:
            obj["final"] = _half(final[:n])
        f = out / "features" / f"{key}__{name}.pt"
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, f)
        shape = f", final {tuple(obj['final'].shape)}" if final is not None else ""
        print(f"[saved] {key}__{name}: layers {tuple(obj['layers'].shape)}{shape}", flush=True)


# ---------------------------------------------------------------- dual encoders (arm A)


def _text_blocks_and_pool(model):
    """(resblocks, pooled index function) of an open_clip text tower."""
    if hasattr(model, "transformer"):  # CLIP class: EOT token (argmax of token ids)
        return model.transformer.resblocks, lambda tok: tok.argmax(dim=-1)
    text = model.text
    if text.pool_type == "last":
        return text.transformer.resblocks, lambda tok: torch.full((tok.shape[0],), tok.shape[1] - 1)
    if text.pool_type == "argmax":
        return text.transformer.resblocks, lambda tok: tok.argmax(dim=-1)
    raise NotImplementedError(text.pool_type)


def _vision_blocks(model):
    """(blocks, pooling of a block output [b, t, d] to [b, d], name)."""
    v = model.visual
    if hasattr(v, "transformer"):  # open_clip VisionTransformer with a class token
        return v.transformer.resblocks, (lambda h: h[:, 0]), "cls"
    trunk = v.trunk  # timm trunk (SigLIP): no prefix tokens, attention-pooling head
    if int(getattr(trunk, "num_prefix_tokens", 0)):
        return trunk.blocks, (lambda h: h[:, 0]), "cls"
    return trunk.blocks, (lambda h: h.mean(1)), "token_mean"


class _Hooks:
    def __init__(self, blocks):
        self.out = []
        self.handles = [b.register_forward_hook(lambda m, i, o: self.out.append(o)) for b in blocks]

    def pop(self):
        out, self.out = self.out, []
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def tower_text(model, tokenizer, texts: list[str], batch: int, device: str):
    blocks, pool_idx = _text_blocks_and_pool(model)
    hooks = _Hooks(blocks)
    layers, finals = [], []
    for i in trange(0, len(texts), batch, desc="text tower"):
        tok = tokenizer(texts[i : i + batch]).to(device)
        final = model.encode_text(tok)
        outs = hooks.pop()
        idx = pool_idx(tok).to(device)
        rows = torch.arange(tok.shape[0], device=device)
        layers.append(torch.stack([o[rows, idx] for o in outs], 1).float().cpu())
        finals.append(final.float().cpu())
    hooks.remove()
    return torch.cat(layers), torch.cat(finals)


@torch.no_grad()
def tower_image(model, preprocess, images: list, batch: int, device: str):
    blocks, pool, _ = _vision_blocks(model)
    hooks = _Hooks(blocks)
    layers, finals = [], []
    for i in trange(0, len(images), batch, desc="image tower"):
        x = torch.stack([preprocess(im.convert("RGB")) for im in images[i : i + batch]]).to(device)
        final = model.encode_image(x)
        outs = hooks.pop()
        layers.append(torch.stack([pool(o) for o in outs], 1).float().cpu())
        finals.append(final.float().cpu())
    hooks.remove()
    return torch.cat(layers), torch.cat(finals)


def load_tower(key: str, device: str):
    import open_clip

    name, pretrained = TOWERS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    return model.to(device).eval(), preprocess, open_clip.get_tokenizer(name)


# ---------------------------------------------------------------- LLMs (B1, B3)


@torch.no_grad()
def llm_mean(model, tokenizer, texts: list[str], max_tokens: int = 16384) -> torch.Tensor:
    """Every hidden state, mean over the caption's tokens (platonic-rep recipe as exp-015
    `extract_llm` with pool avg, accumulated in float32). Captions are batched by length under a
    token budget (Gemini captions are ~20x longer than the originals); output in input order."""
    device = next(model.parameters()).device
    base = model.base_model  # hidden states without the LM head's vocab-sized logits
    lengths = [
        len(t) for t in tokenizer(texts, truncation=True, max_length=MAX_LLM_TOKENS)["input_ids"]
    ]
    order = np.argsort(lengths)[::-1]
    feats, i = [None] * len(texts), 0
    with tqdm(total=len(texts), desc="llm") as bar:
        while i < len(order):
            b = max(1, min(64, max_tokens // lengths[order[i]]))
            idx = order[i : i + b]
            tok = tokenizer(
                [texts[j] for j in idx],
                padding="longest",
                truncation=True,
                max_length=MAX_LLM_TOKENS,
                return_tensors="pt",
            )
            tok = tok.to(device)
            out = base(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                output_hidden_states=True,
            )
            mask = tok["attention_mask"].unsqueeze(-1).float()
            f = torch.stack(
                [(h.float() * mask).sum(1) / mask.sum(1) for h in out["hidden_states"]], 1
            )
            for j, row in zip(idx, f.cpu(), strict=True):
                feats[j] = row
            i += len(idx)
            bar.update(len(idx))
    return torch.stack(feats)


# ---------------------------------------------------------------- ViTs (B2, B3, baseline, D)


@torch.no_grad()
def vit_cls(name: str, images: list, batch: int, device: str, pretrained: bool = True):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision.models.feature_extraction import create_feature_extractor

    torch.manual_seed(0)
    model = timm.create_model(name, pretrained=pretrained).to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
    model = create_feature_extractor(
        model, return_nodes=nodes, concrete_args={"attn_mask": None, "is_causal": False}
    )
    feats = []
    desc = f"{name} ({'pretrained' if pretrained else 'random'})"
    for i in trange(0, len(images), batch, desc=desc):
        x = torch.stack([transform(im.convert("RGB")) for im in images[i : i + batch]]).to(device)
        out = model(x)
        feats.append(torch.stack([v[:, 0, :] for v in out.values()], 1).float().cpu())
    del model
    return torch.cat(feats)


def augment(images: list, seed: int) -> list:
    """The same random view of each image for a given seed (seeded per image), at the image's
    own size so the model's eval transform sees the same resolution as without augmentation."""
    from torchvision.transforms import v2

    jitter_flip = v2.Compose([v2.ColorJitter(0.4, 0.4, 0.4, 0.1), v2.RandomHorizontalFlip()])
    out = []
    for i, im in enumerate(images):
        im = im.convert("RGB")
        torch.manual_seed(seed * 1_000_003 + i)
        crop = v2.RandomResizedCrop(size=(im.height, im.width), scale=(0.6, 1.0), antialias=True)
        out.append(jitter_flip(crop(im)))
    return out


# ---------------------------------------------------------------- data


def load_d1():
    from datasets import load_dataset

    d = load_dataset("minhuh/prh", revision="wit_1024", split="train")
    return [str(r[0]) for r in d["text"]], [d[i]["image"] for i in range(len(d))]


def load_d2(path: Path, n: int):
    import io

    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(path).iloc[:n]
    images = [Image.open(io.BytesIO(b)) for b in df["image"]]
    return df["caption"].tolist(), df["gemini_caption"].tolist(), images


def hub_dir(name: str) -> str:
    """Hugging Face cache folder holding a model's weights ('' if it downloads none)."""
    if name in TOWERS:
        import open_clip

        repo = open_clip.get_pretrained_cfg(*TOWERS[name])["hf_hub"].strip("/")
    elif name in LLMS:
        repo = name
    elif name in VITS:
        repo = f"timm/{name}"
    else:  # the random ViT downloads nothing
        return ""
    return "models--" + repo.replace("/", "--")


# ---------------------------------------------------------------- checks


@torch.no_grad()
def check_towers(device: str) -> None:
    """The hooked last block, passed through the tower's own head, reproduces encode_text /
    encode_image (open_clip ViT-B-32, small download)."""
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    texts = ["a photo of a cat", "two dogs on a beach at sunset " * 30]  # second one truncated
    layers, final = tower_text(model, tokenizer, texts, 2, device)
    manual = model.ln_final(layers[:, -1].to(device)) @ model.text_projection
    print(
        "text: last block -> ln_final -> projection vs encode_text, max abs diff",
        float((manual.cpu() - final).abs().max()),
    )
    rng = np.random.default_rng(0)
    ims = [Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)) for _ in range(2)]
    layers, final = tower_image(model, preprocess, ims, 2, device)
    manual = model.visual.ln_post(layers[:, -1].to(device)) @ model.visual.proj
    print(
        "image: last block CLS -> ln_post -> proj vs encode_image, max abs diff",
        float((manual.cpu() - final).abs().max()),
    )
    aug = augment(ims, 1)
    assert aug[0].size == ims[0].size and augment(ims, 1)[0].tobytes() == aug[0].tobytes()
    assert augment(ims, 2)[0].tobytes() != aug[0].tobytes()
    print("augment: same size, deterministic per seed, differs across seeds")


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--d2", type=Path, help="prepare_d2.py parquet")
    parser.add_argument("--parts", default="towers,llms,vits,aug,random")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--llm-tokens", type=int, default=16384, help="tokens per LLM batch")
    parser.add_argument("--only", default=None, help="one tower key, LLM, ViT or 'random'")
    parser.add_argument("--hub-dir", default=None, help="print the HF cache folder of a model")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check:
        check_towers(device)
        return
    if args.hub_dir:
        print(hub_dir(args.hub_dir))
        return
    pick = lambda names: [m for m in names if args.only in (None, m)]  # noqa: E731
    parts = args.parts.split(",")
    feats = args.out / "features"
    feats.mkdir(parents=True, exist_ok=True)

    def done(key: str, tag: str) -> bool:
        names = {"d1": ["d1"], "d2": ["d2s", "d2l"]}[tag]
        return all((feats / f"{key}__{n}.pt").exists() for n in names)

    d1_text, d1_img = load_d1()
    d2_orig, d2_gem, d2_img = load_d2(args.d2, N_LARGE)
    sets = {  # tag: (captions by kind, images)
        "d1": ({"orig": d1_text}, d1_img),
        "d2": ({"orig": d2_orig, "gemini": d2_gem}, d2_img),
    }

    if "towers" in parts:
        for key in pick(TOWERS):
            model = None
            for tag, (caps, img) in sets.items():
                todo = [(f"{key}__text-{c}", t) for c, t in caps.items()]
                todo = [(k, t) for k, t in todo if not done(k, tag)]
                if not todo and done(f"{key}__image", tag):
                    continue
                if model is None:
                    model, preprocess, tokenizer = load_tower(key, device)
                for k, texts in todo:
                    lay, fin = tower_text(model, tokenizer, texts, args.batch, device)
                    save(args.out, k, tag, lay, fin, {"tower": "text", "model": TOWERS[key]})
                if not done(f"{key}__image", tag):
                    lay, fin = tower_image(model, preprocess, img, args.batch, device)
                    pool = _vision_blocks(model)[2]
                    meta = {"tower": "image", "model": TOWERS[key], "pool": pool}
                    save(args.out, f"{key}__image", tag, lay, fin, meta)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if "llms" in parts:
        from models import load_llm, load_tokenizer  # platonic-rep, not a package

        for name in pick(LLMS):
            stem = name.replace("/", "__")
            todo = [
                (tag, c, t)
                for tag, (caps, _) in sets.items()
                for c, t in caps.items()
                if not done(f"{stem}__{c}", tag)
            ]
            if not todo:
                continue
            model, tokenizer = load_llm(name), load_tokenizer(name)
            assert tokenizer.padding_side == "left"
            for tag, c, texts in todo:
                meta = {"model": name, "pool": "avg", "caption": c}
                save(
                    args.out,
                    f"{stem}__{c}",
                    tag,
                    llm_mean(model, tokenizer, texts, args.llm_tokens),
                    None,
                    meta,
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if "vits" in parts:
        for name in pick(VITS):
            for tag, (_, img) in sets.items():
                if not done(f"{name}__cls", tag):
                    lay = vit_cls(name, img, args.batch, device)
                    save(args.out, f"{name}__cls", tag, lay, None, {"model": name, "pool": "cls"})
            torch.cuda.empty_cache()

    if "aug" in parts:
        img = sets["d1"][1]
        for seed in (1, 2, 3):
            aug = None
            for name in pick(VITS):
                if not done(f"{name}__cls-aug{seed}", "d1"):
                    aug = aug or augment(img, seed)
                    lay = vit_cls(name, aug, args.batch, device)
                    meta = {"model": name, "pool": "cls", "augment_seed": seed}
                    save(args.out, f"{name}__cls-aug{seed}", "d1", lay, None, meta)

    if "random" in parts and args.only in (None, "random"):
        key = f"random-{RANDOM_VIT}__cls"
        meta = {"model": RANDOM_VIT, "pretrained": False, "seed": 0, "pool": "cls"}
        for tag, (_, img) in sets.items():
            if not done(key, tag):
                lay = vit_cls(RANDOM_VIT, img, args.batch, device, pretrained=False)
                save(args.out, key, tag, lay, None, meta)


if __name__ == "__main__":
    main()
