"""exp-020 / exp-018: caption features and bits per byte of language models (incl. checkpoints).

Features: every hidden state (embedding output first), mean-pooled over each caption's tokens,
accumulated in float32, for the D2 captions of exp-016 (10240 WIT-1M images, original and Gemini
caption). Saved as exp-016 does: `features/<stem>__<caption>__d2s.pt` (first 1024 rows, all
layers) and `__d2l.pt` (all rows, 7 evenly spaced layers), with stem = repo with '/' -> '__',
plus '@<revision>' for a checkpoint. Captions are right-padded (real tokens keep positions
0..L-1), without BOS, as the PRH recipe; batches are formed by length under a token budget.

Two kinds of model:
- `causal`: AutoModelForCausalLM, hidden states of `model.base_model`;
- `vlm`: the language model inside a vision-language model, run on text only (no image
  tokens): LLaVA-OneVision `model.model.language_model`, Idefics3 / SmolVLM
  `model.model.text_model`; the LM head is `model.lm_head`.

Bits per byte (`bpb/<stem>.json`): next-token NLL / (ln 2 * UTF-8 bytes), every text preceded by
one context token (BOS, else EOS) so that all its tokens are predicted, on (a) a fixed generic
text, the first 1 MB of WikiText-103 test cut into ~4000-character chunks, (b) the first 2048
original captions, (c) the first 256 Gemini captions.

    uv run python experiments/exp-020-lm-data-pilot/extract_lm.py --out <dir> --d2 <d2.parquet> \\
        --models Qwen/Qwen2.5-0.5B
    uv run python experiments/exp-020-lm-data-pilot/extract_lm.py --out <dir> --d2 <d2.parquet> \\
        --models EleutherAI/pythia-160m --revisions step0,step1000
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# exp-020 models: (repo, kind)
MODELS = {
    "Qwen/Qwen2.5-0.5B": "causal",
    "Qwen/Qwen2.5-Coder-0.5B": "causal",
    "Qwen/Qwen2.5-1.5B": "causal",
    "Qwen/Qwen2.5-Coder-1.5B": "causal",
    "Qwen/Qwen2.5-Math-1.5B": "causal",
    "Qwen/Qwen2-0.5B-Instruct": "causal",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": "vlm",
    "HuggingFaceTB/SmolLM2-135M-Instruct": "causal",
    "HuggingFaceTB/SmolVLM-256M-Instruct": "vlm",
}
N_SMALL, N_LARGE, LARGE_DEPTHS = 1024, 10240, 7
MAX_TOKENS = 2048  # per caption (Pythia's context); Gemini captions are ~650 tokens (median)
BPB_BYTES, BPB_CHUNK = 1_000_000, 4000
BPB_N = {"orig": 2048, "gemini": 256}


def stem(repo: str, revision: str | None) -> str:
    return repo.replace("/", "__") + (f"@{revision}" if revision else "")


def large_layers(n_layers: int) -> list[int]:
    """LARGE_DEPTHS evenly spaced layers, first and last included (as exp-016's d2l)."""
    return sorted(set(np.rint(np.linspace(0, n_layers - 1, LARGE_DEPTHS)).astype(int).tolist()))


def save(out: Path, key: str, layers: torch.Tensor, meta: dict, n_small: int, n_large: int):
    n_layers = layers.shape[1]
    for name, n, idx in (
        ("d2s", n_small, list(range(n_layers))),
        ("d2l", n_large, large_layers(n_layers)),
    ):
        x = layers[:n, idx]
        h = x.half()
        obj = {
            "layers": h if torch.isfinite(h).all() else x.float(),
            "layer_idx": idx,
            "meta": {**meta, "n": n, "n_layers": n_layers},
        }
        f = out / "features" / f"{key}__{name}.pt"
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, f)
        print(f"[saved] {key}__{name}: {tuple(obj['layers'].shape)}", flush=True)


def load(repo: str, revision: str | None, kind: str, device: str, cache_dir: str | None):
    """-> (decoder returning hidden states, LM head, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kw = {"revision": revision, "cache_dir": cache_dir}
    if kind == "causal":
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kw)
        dec, head = model.base_model, model.get_output_embeddings()
    else:
        model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=dtype, **kw)
        inner = model.model
        dec = getattr(inner, "language_model", None) or inner.text_model
        head = model.lm_head
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return model, dec, head, tok


def _batches(lengths: list[int], budget: int):
    order = np.argsort(lengths)[::-1]
    i = 0
    while i < len(order):
        b = max(1, min(64, budget // max(1, lengths[order[i]])))
        yield order[i : i + b]
        i += b


@torch.no_grad()
def mean_pooled(dec, tok, texts: list[str], budget: int) -> torch.Tensor:
    """[n, L + 1, d] float32: every hidden state, mean over the caption's tokens."""
    device = next(dec.parameters()).device
    ids = tok(texts, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)["input_ids"]
    feats = None
    with tqdm(total=len(texts), desc="features") as bar:
        for idx in _batches([len(x) for x in ids], budget):
            batch = tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(device)
            out = dec(**batch, output_hidden_states=True)
            mask = batch["attention_mask"].unsqueeze(-1).float()
            f = torch.stack([(h.float() * mask).sum(1) / mask.sum(1) for h in out.hidden_states], 1)
            if feats is None:
                feats = torch.empty(len(texts), *f.shape[1:])
            feats[torch.from_numpy(idx.copy())] = f.cpu()
            bar.update(len(idx))
    return feats


@torch.no_grad()
def bits_per_byte(dec, head, tok, texts: list[str], budget: int) -> dict:
    device = next(dec.parameters()).device
    ctx = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids, n_bytes = [], 0
    for t in texts:
        x = tok(t, add_special_tokens=False)["input_ids"]
        if len(x) > MAX_TOKENS - 1:  # count the bytes of what is scored
            x = x[: MAX_TOKENS - 1]
            t = tok.decode(x)
        ids.append([ctx, *x])
        n_bytes += len(t.encode("utf-8"))
    nats = 0.0
    for idx in _batches([len(x) for x in ids], budget):
        batch = tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(device)
        h = dec(**batch).last_hidden_state[:, :-1]
        target = batch["input_ids"][:, 1:]
        keep = batch["attention_mask"][:, 1:].bool()
        h, target = h[keep], target[keep]
        for s in range(0, len(target), 1024):  # logits over a large vocabulary in slices
            logits = head(h[s : s + 1024]).float()
            nats += float(
                torch.nn.functional.cross_entropy(logits, target[s : s + 1024], reduction="sum")
            )
    return {"bpb": nats / math.log(2) / n_bytes, "bytes": n_bytes, "texts": len(texts)}


def generic_text(out: Path) -> list[str]:
    path = out / "bpb_generic.txt"
    if not path.exists():  # only the 0.7 MB test file, not the whole dataset
        import pandas as pd
        from huggingface_hub import hf_hub_download

        f = hf_hub_download(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        path.write_text("".join(pd.read_parquet(f)["text"])[:BPB_BYTES], encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    return [text[i : i + BPB_CHUNK] for i in range(0, len(text), BPB_CHUNK)]


def captions(d2: Path, limit: int | None) -> dict[str, list[str]]:
    import pandas as pd

    df = pd.read_parquet(d2, columns=["caption", "gemini_caption"]).iloc[: limit or N_LARGE]
    return {"orig": df["caption"].tolist(), "gemini": df["gemini_caption"].tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--d2", type=Path, required=True, help="exp-016 d2.parquet")
    p.add_argument("--models", default=",".join(MODELS), help="comma list of repos")
    p.add_argument("--kind", default=None, help="override: causal or vlm")
    p.add_argument("--revisions", default=None, help="comma list of revisions (checkpoints)")
    p.add_argument("--captions", default="orig,gemini")
    p.add_argument("--tokens", type=int, default=16384, help="tokens per batch")
    p.add_argument("--cache-dir", default=None, help="private download dir, removed after use")
    p.add_argument("--limit", type=int, default=None, help="first n captions (smoke test)")
    p.add_argument("--no-bpb", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    caps = captions(args.d2, args.limit)
    n_large = min(N_LARGE, len(caps["orig"]))
    n_small = min(N_SMALL, n_large // 2 if args.limit else N_SMALL)
    wanted = args.captions.split(",")
    revisions = args.revisions.split(",") if args.revisions else [None]
    for repo in args.models.split(","):
        kind = args.kind or MODELS.get(repo, "causal")
        for rev in revisions:
            key = stem(repo, rev)
            bpb_path = args.out / "bpb" / f"{key}.json"
            todo = [
                c for c in wanted if not (args.out / "features" / f"{key}__{c}__d2l.pt").exists()
            ]
            if not todo and (args.no_bpb or bpb_path.exists()):
                print(f"{key}: done", flush=True)
                continue
            model, dec, head, tok = load(repo, rev, kind, device, args.cache_dir)
            meta = {"model": repo, "revision": rev, "kind": kind, "pool": "avg"}
            for c in todo:
                feats = mean_pooled(dec, tok, caps[c], args.tokens)
                save(args.out, f"{key}__{c}", feats, {**meta, "caption": c}, n_small, n_large)
            if not args.no_bpb and not bpb_path.exists():
                gen = generic_text(args.out)[: 25 if args.limit else None]
                res = {"generic": bits_per_byte(dec, head, tok, gen, args.tokens)}
                for c, n in BPB_N.items():
                    res[c] = bits_per_byte(dec, head, tok, caps[c][:n], args.tokens)
                bpb_path.parent.mkdir(parents=True, exist_ok=True)
                bpb_path.write_text(json.dumps({**meta, **res}, indent=1))
                print(f"[bpb] {key}: " + ", ".join(f"{k} {v['bpb']:.3f}" for k, v in res.items()))
            del model, dec, head
            gc.collect()
            torch.cuda.empty_cache()
            if args.cache_dir:
                shutil.rmtree(Path(args.cache_dir) / f"models--{repo.replace('/', '--')}", True)


if __name__ == "__main__":
    main()
