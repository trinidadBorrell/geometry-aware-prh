"""exp-023: lexical floor of the PRH setup: WIT-1024, the published 10 LLM x 17 ViT pool.

No forward passes: cached activations (`geoprh.activation_cache`) and the captions they were made
from. PRH's own procedure: q = 0.95 clamp and l2 norm of cached layers, raw mutual kNN (k = 10) and
CKA, max over layer pairs; no null calibration here. Unbiased CKA is kept as a check, because biased
CKA at n = 1024 understates high-rank word kernels (exp-022).

For each LLM x ViT:
- lm: at every LLM layer, the score at its best ViT layer. Layer 0 is the embedding layer;
- lexical: word kernels with no model, at their best ViT layer: TF, TF-IDF, binary and sublinear
  TF-IDF with this LLM's tokens (special tokens included, as the features were made), and word-level
  TF-IDF with and without English stopwords. No clamp: rows are sparse (exp-022).

The best-pair score must reproduce the stored `prh_val/cross_pairs.jsonl`, or the job stops.

    uv run python experiments/exp-023-prh-lexical-floor/run.py --out <dir> --stored <prh_val dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from geoprh import activation_cache as ac
from geoprh import lexical as lex
from geoprh import prh_alignment as pa

METRICS = {"mknn": pa.mutual_knn, "cka": pa.cka, "ucka": pa.unbiased_cka}
STORED = {"mknn": "mutual_knn", "cka": "cka", "ucka": "unbiased_cka"}
TOKEN_VARIANTS = ("tf", "tfidf", "binary", "tfidf_sublinear")
WORD_VARIANTS = ("word_tfidf", "word_tfidf_nostop")
LANG = ac.Variant(pool="avg", caption_idx=0)
K = 10


def load_captions(limit: int | None) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("minhuh/prh", revision="wit_1024", split="train").select_columns(["text"])
    texts = [str(t[0]) for t in ds["text"]]
    return texts[: limit or None]


def token_features(ids: list[list[int]]) -> dict[str, torch.Tensor]:
    counts = lex.count_matrix(ids)[0]
    return {
        "tf": lex.to_unit_rows(lex.tf(counts)),
        "tfidf": lex.to_unit_rows(lex.tfidf(counts)),
        "binary": lex.to_unit_rows(lex.binary(counts)),
        "tfidf_sublinear": lex.to_unit_rows(lex.tfidf(counts, sublinear=True)),
    }


def word_features(texts: list[str]) -> dict[str, torch.Tensor]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    out = {}
    for name, stop in (("word_tfidf", None), ("word_tfidf_nostop", "english")):
        vec = TfidfVectorizer(lowercase=True, stop_words=stop, norm="l2", dtype=np.float32)
        out[name] = lex.to_unit_rows(vec.fit_transform(texts))
    return out


def vision_layers(root: Path, vit: str) -> list[pa.Layer]:
    variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    return pa._layers_for(vit, variant, root, K)


def best_of(vision: list[pa.Layer], layer: pa.Layer) -> dict:
    """{metric: (score, ViT layer)} of one language layer at its best ViT layer."""
    out = {}
    for name, fn in METRICS.items():
        scores = [fn(v, layer) for v in vision]
        j = int(np.argmax(scores))
        out[name] = (float(scores[j]), j)
    return out


def pair_task(args) -> dict:
    llm, vit, root, lexical_path = args
    torch.set_num_threads(1)
    vision = vision_layers(root, vit)
    language = pa._layers_for(llm, LANG, root, K)
    by_layer = [best_of(vision, x) for x in language]
    lexical = {}
    feats = torch.load(lexical_path, weights_only=False)
    for name, x in feats.items():
        lexical[name] = best_of(vision, pa.Layer(x, K))
    return {
        "llm": llm,
        "vit": vit,
        "n_llm_layers": len(language),
        "n_vit_layers": len(vision),
        "lm": {m: [r[m][0] for r in by_layer] for m in METRICS},
        "lm_vit_layer": {m: [r[m][1] for r in by_layer] for m in METRICS},
        "lexical": {v: {m: r[m][0] for m in METRICS} for v, r in lexical.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stored", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None, help="activation cache")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scratch", type=Path, default=Path("/root/exp023"))
    args = p.parse_args()
    from transformers import AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    root = args.root or ac.default_root()
    meta = json.loads((args.stored / "cross_modal.json").read_text())
    llms = [m for m in meta["llms"] if ac.cached_layers(root, m, LANG)]
    vision_variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    vits = [m for m in meta["lvms"] if ac.cached_layers(root, m, vision_variant)]
    print(f"{len(llms)} LLMs x {len(vits)} ViTs", flush=True)
    texts = load_captions(args.limit)
    assert len(texts) == 1024
    words = word_features(texts)
    paths = {}
    stats = {}
    for llm in llms:
        tok = AutoTokenizer.from_pretrained(llm)
        ids = tok(texts)["input_ids"]  # special tokens included, as the cached features
        feats = {**token_features(ids), **words}
        path = args.scratch / f"{llm.replace('/', '__')}.pt"
        torch.save(feats, path)
        paths[llm] = path
        stats[llm] = {"tokens_median": float(np.median([len(x) for x in ids]))}
    (args.out / "captions_sample.json").write_text(json.dumps(texts[:20], indent=1))

    stored = {}
    for line in (args.stored / "cross_pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        stored[(r["llm"], r["vit" if "vit" in r else "lvm"])] = r
    path = args.out / "pairs.jsonl"
    done = set()
    if path.exists():
        done = {(r["llm"], r["vit"]) for r in map(json.loads, path.read_text().splitlines())}
    tasks = [(m, v, root, paths[m]) for m in llms for v in vits if (m, v) not in done]
    t0 = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(pair_task, t) for t in tasks)):
            r = fut.result()
            s = stored[(r["llm"], r["vit"])]
            for m, name in STORED.items():
                got, want = max(r["lm"][m]), s[name]["score"]
                if m == "cka" or m == "mknn":
                    assert abs(got - want) < 2e-3, f"{r['llm']} {r['vit']} {m}: {got} vs {want}"
            with path.open("a") as fh:
                fh.write(json.dumps({**r, **stats[r["llm"]]}) + "\n")
            print(
                f"[{i + 1}/{len(tasks)}] {r['llm']} x {r['vit']}: mKNN L0 {r['lm']['mknn'][0]:.3f} "
                f"best {max(r['lm']['mknn']):.3f} TF-IDF {r['lexical']['tfidf']['mknn']:.3f}; CKA "
                f"L0 {r['lm']['cka'][0]:.3f} best {max(r['lm']['cka']):.3f} "
                f"TF-IDF {r['lexical']['tfidf']['cka']:.3f}  {time.time() - t0:.0f}s",
                flush=True,
            )
    print("done", flush=True)


if __name__ == "__main__":
    main()
