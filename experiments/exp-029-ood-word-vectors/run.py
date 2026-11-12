"""exp-029: word vectors fitted on captions vs general text, scored like exp-028 on D2.

Stages (outputs on the volume, each resumable):
- `corpora`: stream C4 English and English Wikipedia (shuffled streams, seed 0), keep each document's
  analysed tokens, one line per document, until 10x the Gemini anchors' content-token count. The
  first 1x of each is the token-matched corpus.
- `vectors`: one PPMI-SVD fit (top 50000 words with count >= 5, U S^0.5, 1024 dimensions) per corpus:
  captions of each type (exp-028's 20000 anchors) with whole-caption and +-5 windows; C4 and
  Wikipedia at 1x and 10x with +-5 windows. Plus exp-028's exact W20000 refit (`repro`, the
  pipeline check) at 512 and at 1024 dimensions, and GloVe 6B 300d.
- `kernels`: per evaluation caption set, exp-028's three floors, and for every fit the caption vector
  = TF-IDF (idf from the evaluation captions, the fit's vocabulary) x word vectors, at k_s in DIMS.
  `same` / `cross` name the caption corpus of the same / other type. Control: the `same` whole-caption
  vectors with words permuted (seed 0).
- `score`: exp-028's `stage_score` (D2, DINOv2 and CLIP, all images and two halves). Then the `repro`
  rows must match exp-028's stored W20000 rows within 0.003, or the job fails.

    uv run python experiments/exp-029-ood-word-vectors/run.py --out <dir> --exp016 <dir> \\
        --anchors <exp028>/anchors.parquet --exp028-rows <exp028>/rows.jsonl --stored <exp022>/rows.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R28 = _module("exp028_run", ROOT / "exp-028-spectral-lexical-floor" / "run.py")
WD = _module("exp029_words", HERE / "words.py")
SP = R28.SP
CAPTIONS = R28.CAPTIONS  # {"orig": "caption", "gemini": "gemini_caption"}
DIMS = (128, 256, 512, 1024)
VOCAB, MIN_COUNT, WINDOW = 50000, 5, 5
STREAMS = {
    "web": ("allenai/c4", "en"),
    "wiki": ("wikimedia/wikipedia", "20231101.en"),
}
GLOVE = "sentence-transformers/average_word_embeddings_glove.6B.300d"
log = R28.log


def read_docs(path: Path, max_tokens: int | None = None):
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = line.split()
            if max_tokens is not None and n >= max_tokens:
                return
            n += len(d)
            yield d


def stage_corpora(args) -> None:
    import pandas as pd
    from datasets import load_dataset

    an = WD.analyzer()
    anchors = pd.read_parquet(args.anchors)
    budget = sum(len(an(t)) for t in anchors["gemini_caption"])
    meta_path = args.out / "corpora.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["gemini_anchor_tokens"] = budget
    (args.out / "corpora").mkdir(parents=True, exist_ok=True)
    for name, (repo, config) in STREAMS.items():
        path = args.out / "corpora" / f"{name}.txt.gz"
        if name in meta and path.exists():
            log(f"corpus {name}: already saved")
            continue
        t0 = time.time()
        target = args.scale * budget
        stream = load_dataset(repo, config, split="train", streaming=True).select_columns(["text"])
        stream = stream.shuffle(seed=0, buffer_size=10000)
        n_tok = n_doc = 0
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for row in stream:
                toks = an(row["text"] or "")
                if not toks:
                    continue
                fh.write(" ".join(toks) + "\n")
                n_tok += len(toks)
                n_doc += 1
                if n_tok >= target:
                    break
        tmp.replace(path)
        meta[name] = {"repo": repo, "config": config, "tokens": n_tok, "docs": n_doc}
        meta_path.write_text(json.dumps(meta, indent=1))
        log(
            f"corpus {name}: {n_doc} docs, {n_tok} tokens (target {target}), {time.time() - t0:.0f}s"
        )


def fit_plan(args) -> dict[str, tuple]:
    """name -> (kind, source, window, max_tokens)."""
    meta = json.loads((args.out / "corpora.json").read_text())
    b = meta["gemini_anchor_tokens"]
    plan = {}
    for cap in CAPTIONS:
        plan[f"cap-{cap}-doc"] = ("captions", cap, None, None)
        plan[f"cap-{cap}-win"] = ("captions", cap, WINDOW, None)
    for name in STREAMS:
        plan[f"{name}-1x"] = ("stream", name, WINDOW, b)
        plan[f"{name}-{args.scale}x"] = ("stream", name, WINDOW, None)
    return plan


def stage_vectors(args) -> None:
    import pandas as pd

    an = WD.analyzer()
    anchors = pd.read_parquet(args.anchors)
    vdir = args.out / "vectors"
    vdir.mkdir(parents=True, exist_ok=True)
    for name, (kind, src, window, max_tok) in fit_plan(args).items():
        path = vdir / f"{name}.npz"
        if path.exists():
            log(f"vectors {name}: already saved")
            continue
        t0 = time.time()

        def docs(kind=kind, src=src, max_tok=max_tok):
            if kind == "captions":
                return (an(t) for t in anchors[CAPTIONS[src]])
            return read_docs(args.out / "corpora" / f"{src}.txt.gz", max_tok)

        vocab = WD.build_vocab(docs(), VOCAB, MIN_COUNT)
        index = {w: i for i, w in enumerate(vocab)}
        ids = WD.to_ids(docs(), index)
        c = (
            WD.cooc_doc(ids, len(vocab))
            if window is None
            else WD.cooc_window(ids, len(vocab), window)
        )
        p = WD.ppmi_sparse(c)
        vecs = WD.vectors(p, max(DIMS))
        n_tok = sum(len(d) for d in docs())
        np.savez(
            path,
            vocab=np.array(vocab),
            vecs=vecs.astype(np.float16),  # half the volume space; caption vectors are float32
            meta=json.dumps({"tokens": n_tok, "cooc_nnz": int(c.nnz), "ppmi_nnz": int(p.nnz)}),
        )
        log(
            f"vectors {name}: V {len(vocab)}, {n_tok} tokens, PPMI nnz {p.nnz}, {time.time() - t0:.0f}s"
        )


def load_glove():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg = json.loads(
        Path(hf_hub_download(GLOVE, "0_WordEmbeddings/whitespacetokenizer_config.json")).read_text(
            encoding="utf-8"
        )
    )
    emb = load_file(hf_hub_download(GLOVE, "0_WordEmbeddings/model.safetensors"))[
        "emb_layer.weight"
    ]
    return cfg["vocab"], emb.numpy()


def caption_vectors(ev: list[str], vocab, vecs: np.ndarray) -> torch.Tensor:
    """TF-IDF over the fit's vocabulary (idf from the evaluation captions, l2 rows) x word vectors."""
    keep = {}
    for i, w in enumerate(vocab):
        keep.setdefault(str(w), i)  # first occurrence wins (GloVe has no duplicates; safe)
    words = list(keep)
    vec = R28.word_vectorizer(vocabulary=words)
    x = vec.fit_transform(ev)
    rows = vecs[np.array([keep[w] for w in words])]
    return torch.from_numpy(np.ascontiguousarray(x @ rows, dtype=np.float32))


def stage_kernels(args) -> None:
    import pandas as pd
    from sklearn.preprocessing import normalize
    from transformers import AutoTokenizer

    from geoprh import lexical as lex

    d2 = pd.read_parquet(args.exp016 / "d2.parquet", columns=list(CAPTIONS.values()))
    d2 = d2.iloc[: args.limit or R28.N_D2]
    anchors = pd.read_parquet(args.anchors)
    an = WD.analyzer()
    tok = AutoTokenizer.from_pretrained(R28.PY_TOKENIZER)
    kdir = args.out / "kernels"
    kdir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "kernels_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fits = {n: np.load(args.out / "vectors" / f"{n}.npz") for n in fit_plan(args)}
    glove = None if args.no_glove else load_glove()
    for cap, col in CAPTIONS.items():
        if (kdir / f"kernels_{cap}.pt").exists() and cap in meta:
            log(f"kernels {cap}: already saved")
            continue
        t0 = time.time()
        ev = d2[col].tolist()
        ev_tok = [an(t) for t in ev]
        other = "gemini" if cap == "orig" else "orig"
        feats, info = {}, {}
        ids = tok(ev, truncation=True, max_length=R28.MAX_TOKENS, add_special_tokens=False)[
            "input_ids"
        ]
        counts = lex.count_matrix(ids)[0]
        feats["F__tfidf"] = normalize(lex.tfidf(counts).tocsr()).astype(np.float32)
        feats["F__tfidf_sublinear"] = normalize(lex.tfidf(counts, sublinear=True).tocsr()).astype(
            np.float32
        )
        feats["F__word_nostop"] = R28.word_vectorizer().fit_transform(ev).tocsr().astype(np.float32)

        def add(label, vocab, vecs, dims, ev=ev, ev_tok=ev_tok, feats=feats, info=info):
            full = caption_vectors(ev, vocab, vecs)
            for k in dims:
                if k <= full.shape[1]:
                    feats[f"W{label}__k{k}__plain"] = full[:, :k]  # views share one storage
            info[label] = {
                "coverage": WD.coverage(ev_tok, set(map(str, vocab))),
                "dims": int(full.shape[1]),
            }

        rename = {
            f"cap-{cap}-doc": "same-doc",
            f"cap-{cap}-win": "same-win",
            f"cap-{other}-doc": "cross-doc",
            f"cap-{other}-win": "cross-win",
        }
        for name, z in fits.items():
            add(rename.get(name, name), z["vocab"], z["vecs"].astype(np.float32), DIMS)
        z = fits[f"cap-{cap}-doc"]
        perm = np.random.default_rng(0).permutation(len(z["vocab"]))
        add("shuffled", z["vocab"], z["vecs"][perm].astype(np.float32), DIMS)
        # exp-028's exact W20000 (vocabulary and idf from the anchors), for the pipeline check
        fit = anchors[col].tolist()[:20000]
        for top, dims in ((512, (128, 256, 512)), (1024, (1024,))):
            vw = R28.word_vectorizer(max_features=R28.W_VOCAB, min_df=2)
            xw = vw.fit_transform(fit)
            vecs = SP.ppmi_vectors((xw > 0).astype(np.float32), min(top, xw.shape[1] - 1))
            full = vw.transform(ev) @ vecs
            for k in [k for k in dims if k <= full.shape[1]]:  # all of them at 20000 words
                feats[f"Wrepro__k{k}__plain"] = torch.from_numpy(
                    np.ascontiguousarray(full[:, :k], np.float32)
                )
        if glove is not None:
            gv, ge = glove
            full = caption_vectors(ev, gv[1:], ge[1:])  # row 0 is PADDING_TOKEN
            feats["G__k300__plain"] = full
            info["G"] = {"coverage": WD.coverage(ev_tok, set(gv[1:])), "dims": 300}
        tmp = kdir / f"kernels_{cap}.pt.tmp"
        torch.save(feats, tmp)
        tmp.replace(kdir / f"kernels_{cap}.pt")
        meta[cap] = info
        meta_path.write_text(json.dumps(meta, indent=1))
        log(
            f"kernels {cap}: {len(feats)} features, {time.time() - t0:.0f}s; coverage "
            + ", ".join(f"{k} {v['coverage']:.3f}" for k, v in info.items())
        )


def stage_check(args) -> None:
    """Wrepro must reproduce exp-028's stored W20000 rows (full split)."""

    def best(r, m, k):
        if m == "cka":
            return (np.array(r["cka"]) - np.array(r["cka_mu0"])).max()
        return (np.array(r["knn"][k]) - r["knn_mu0"][k]).max()

    ours = {
        (r["target"], r["caption"], r["set"]): r
        for r in map(json.loads, (args.out / "rows.jsonl").read_text().splitlines())
        if r["split"] == "full"
    }
    theirs = {
        (r["target"], r["caption"], r["set"]): r
        for r in map(json.loads, args.exp028_rows.read_text().splitlines())
        if r["split"] == "full"
    }
    worst = 0.0
    for (t, c, s), r in ours.items():
        if not s.startswith("Wrepro__") or s.endswith("k1024__plain"):
            continue
        w = theirs[(t, c, s.replace("Wrepro", "W20000"))]
        for m, k in (("knn", "10"), ("knn", "1000"), ("cka", None)):
            worst = max(worst, abs(best(r, m, k) - best(w, m, k)))
    (args.out / "repro_check.json").write_text(json.dumps({"max_abs_diff": worst}))
    log(f"repro check vs exp-028 W20000: max abs diff {worst:.5f}")
    assert worst < 3e-3, "exp-029 does not reproduce exp-028's W20000"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, default=Path("/nonexistent"))
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--exp028-rows", type=Path, default=None)
    p.add_argument("--stored", type=Path, default=None, help="exp-022 rows.jsonl")
    p.add_argument("--stages", default="corpora,vectors,kernels,score,check")
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first n D2 rows")
    p.add_argument("--no-glove", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch = args.out / "kernels"
    for stage in args.stages.split(","):
        log(f"=== stage {stage}")
        {
            "corpora": stage_corpora,
            "vectors": stage_vectors,
            "kernels": stage_kernels,
            "score": R28.stage_score,
            "check": stage_check,
        }[stage](args)
        log(f"=== stage {stage} done")


if __name__ == "__main__":
    main()
