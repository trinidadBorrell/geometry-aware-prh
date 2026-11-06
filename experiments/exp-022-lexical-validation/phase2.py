"""exp-022 phase 2: how much alignment comes from words, order and depth? GPU forward passes.

Stages:
- shuffle: token ids shuffled within each caption, through trained and step-0 LMs. Layer 0 must not
  change. Any change at depth measures word order and context;
- factorial: random or trained input table crossed with random or trained blocks (Pythia);
- docci_prepare / docci_features / docci_lexical: DOCCI, a second dataset with long human captions:
  ViT and LM features, lexical baselines (different images from WIT, so no pairing with Gemini);
- pairs: calibrated alignment of the new feature sets against the ViTs (exp-020's engine);
- ridge: held-out ridge prediction of a ViT layer from lexical, deep and joint inputs (exp-017),
  with a synthetic calibration first.

Pythia on original captions needs no new code: exp-020 `extract_lm.py` and exp-018 `align.py`
(`infra/runpod/run_exp022_phase2.sh`).

    uv run python experiments/exp-022-lexical-validation/phase2.py --out <dir> --exp016 <dir> \\
        --exp018 <dir> --stages shuffle,factorial,pairs,ridge
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


X20 = _module("extract_lm", ROOT / "exp-020-lm-data-pilot" / "extract_lm.py")
A20 = _module("align020", ROOT / "exp-020-lm-data-pilot" / "align.py")
P1 = _module("phase1", HERE / "run.py")
MAPS = _module("maps017", ROOT / "exp-017-map-complexity" / "maps.py")
M16 = A20.M16
EX = M16.EX

from geoprh import lexical as lex  # noqa: E402

DINO, CLIP, MAE, AUGREG = P1.DINO, P1.CLIP, P1.MAE, P1.AUGREG
VITS_2, VITS_4 = [DINO, CLIP], [DINO, CLIP, MAE, AUGREG]
FINAL = P1.FINAL
PYTHIA_FULL = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
BIG_LMS = ["openlm-research/open_llama_7b", "bigscience/bloomz-7b1"]
BIG_CACHE = "/workspace/.cache/huggingface/hub"  # complete weights on the volume (exp-016)
DOCCI_TEXT = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
DOCCI_IMAGES = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"
KS = "10,30,100,300,1000"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def free(*_) -> None:
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- tokens


def shuffled_ids(ids: list[list[int]], special: set[int], seed: int) -> list[list[int]]:
    """Shuffle each caption's token ids; special tokens (BOS) keep their position."""
    rng = np.random.default_rng(seed)
    out = []
    for x in ids:
        arr = np.asarray(x)
        free_pos = np.flatnonzero(~np.isin(arr, list(special))) if special else np.arange(len(arr))
        arr = arr.copy()
        arr[free_pos] = arr[rng.permutation(free_pos)]
        out.append(arr.tolist())
    return out


def tokenise(tok, texts: list[str], special: bool) -> list[list[int]]:
    return tok(texts, truncation=True, max_length=X20.MAX_TOKENS, add_special_tokens=special)[
        "input_ids"
    ]


def save_layers(out: Path, key: str, feats: torch.Tensor, meta: dict, n: int) -> None:
    """d2l-style file: 7 evenly spaced layers, first and last included (extract_lm `save`)."""
    idx = X20.large_layers(feats.shape[1])
    h = feats[:, idx].half()
    (out / "features").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layers": h if torch.isfinite(h).all() else feats[:, idx].float(),
            "layer_idx": idx,
            "meta": {**meta, "n": n},
        },
        out / "features" / f"{key}__d2l.pt",
    )
    log(f"saved {key}: {tuple(h.shape)}")


# ---------------------------------------------------------------- shuffle


def stage_shuffle(args, feat_dir: Path) -> None:
    caps = X20.captions(args.exp016 / "d2.parquet", args.limit)
    n = len(caps["orig"])
    sets = [
        (r, rev, 0, False, args.cache_dir) for r in args.pythia for rev in ("step0", args.final)
    ]
    if args.pythia:
        sets += [(args.pythia[-1], args.final, 1, False, args.cache_dir)]  # second seed: noise
    sets += [(r, None, 0, True, args.big_cache) for r in args.big_lms]
    for repo, rev, seed, special, cache in sets:
        stem = X20.stem(repo, rev)
        todo = [
            c
            for c in ("gemini", "orig")
            if not (feat_dir / "features" / f"{stem}__{c}-shuf{seed}__d2l.pt").exists()
        ]
        if not todo:
            continue
        model, dec, _, tok = X20.load(repo, rev, "causal", device(), cache)
        for c in todo:
            ids = tokenise(tok, caps[c], special)
            shuf = shuffled_ids(ids, set(tok.all_special_ids), seed)
            feats = X20.mean_pooled_ids(dec, tok, shuf, args.tokens)
            check_layer0(dec, tok, ids, shuf, feats, args.tokens)
            meta = {"model": repo, "revision": rev, "caption": c, "shuffle_seed": seed}
            save_layers(feat_dir, f"{stem}__{c}-shuf{seed}", feats, meta, n)
            if special and seed == 0 and c == "orig":
                check_cached(args, repo, dec, tok, ids)
            del feats
        del model, dec
        free()
        if cache == args.cache_dir:
            shutil.rmtree(Path(cache) / f"models--{repo.replace('/', '--')}", True)


@torch.no_grad()
def check_layer0(dec, tok, ids, shuf, feats, budget) -> None:
    """A shuffle cannot change layer 0, a mean over tokens (BLOOM: of LayerNorm'd embeddings)."""
    rows = np.linspace(0, len(ids) - 1, 200).astype(int)
    want = X20.mean_pooled_ids(dec, tok, [ids[i] for i in rows], budget)[:, 0]
    cos = torch.nn.functional.cosine_similarity(feats[rows, 0], want, dim=1)
    log(f"layer 0 after shuffle vs unshuffled: min row cosine {float(cos.min()):.6f}")
    assert float(cos.min()) > 0.999, "shuffle changed layer 0"


@torch.no_grad()
def check_cached(args, repo, dec, tok, ids) -> None:
    """Our batching vs exp-016's cached features of the same captions (first 256, all layers)."""
    key = f"{repo.replace('/', '__')}__orig"
    cached = torch.load(args.exp016 / "features" / f"{key}__d2l.pt", weights_only=False)
    ours = X20.mean_pooled_ids(dec, tok, ids[:256], args.tokens)[:, cached["layer_idx"]]
    theirs = cached["layers"][:256].float()
    cos = torch.nn.functional.cosine_similarity(ours, theirs, dim=2)
    log(f"unshuffled vs exp-016 {repo}: min row cosine by layer {cos.min(0).values.tolist()}")


# ---------------------------------------------------------------- factorial


def stage_factorial(args, feat_dir: Path) -> None:
    caps = X20.captions(args.exp016 / "d2.parquet", args.limit)
    n = len(caps["orig"])
    for repo in args.pythia:
        tables = {}
        for rev in ("step0", args.final):
            model, dec, _, _ = X20.load(repo, rev, "causal", "cpu", args.cache_dir)
            tables[rev] = dec.get_input_embeddings().weight.detach().clone()
            del model, dec
        for blocks, table in (("step0", args.final), (args.final, "step0")):
            stem = f"{repo.replace('/', '__')}@blocks-{blocks}-table-{table}"
            todo = [
                c
                for c in ("gemini", "orig")
                if not (feat_dir / "features" / f"{stem}__{c}__d2l.pt").exists()
            ]
            if not todo:
                continue
            model, dec, _, tok = X20.load(repo, blocks, "causal", device(), args.cache_dir)
            w = dec.get_input_embeddings().weight
            w.data.copy_(tables[table].to(w.device, w.dtype))
            for c in todo:
                feats = X20.mean_pooled_ids(dec, tok, tokenise(tok, caps[c], False), args.tokens)
                save_layers(feat_dir, f"{stem}__{c}", feats, {"model": repo, "caption": c}, n)
                del feats
            del model, dec
            free()
        shutil.rmtree(Path(args.cache_dir) / f"models--{repo.replace('/', '--')}", True)


# ---------------------------------------------------------------- DOCCI


def _resize(item: tuple[str, bytes]) -> bytes | None:
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(item[1])).convert("RGB")
    except OSError:
        return None
    im.thumbnail((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def stage_docci_prepare(args, docci: Path) -> None:
    """DOCCI descriptions + images, downscaled once, in `d2.parquet`-style columns."""
    target = docci / "docci.parquet"
    if target.exists():
        return
    docci.mkdir(parents=True, exist_ok=True)
    scratch = args.docci_scratch
    scratch.mkdir(parents=True, exist_ok=True)
    txt, tgz = scratch / "descriptions.jsonl", scratch / "images.tar.gz"
    if args.docci_text and not txt.exists():
        shutil.copy(args.docci_text, txt)
    elif not txt.exists():
        urllib.request.urlretrieve(DOCCI_TEXT, txt)
    if args.docci_images and not tgz.exists():
        shutil.copy(args.docci_images, tgz)
    elif not tgz.exists():
        log("downloading DOCCI images (7.1 GB)")
        urllib.request.urlretrieve(DOCCI_IMAGES, tgz)
    rows = [json.loads(line) for line in txt.read_text(encoding="utf8").splitlines()]
    # train and test only (the qual_ splits are annotator qualification), a fixed random subset
    # of the same size as WIT's, since mutual kNN depends on n
    rows = [r for r in rows if r.get("split", "train") in ("train", "test")]
    rng = np.random.default_rng(0)
    chosen = rng.permutation(len(rows))[: args.docci_n]
    desc = {rows[i]["example_id"]: rows[i]["description"] for i in chosen}
    log(f"DOCCI: {len(desc)} descriptions chosen of {len(rows)}")
    ids, blobs = [], []
    with tarfile.open(tgz) as tar:
        for m in tar:
            stem = Path(m.name).stem
            if m.isfile() and stem in desc:
                ids.append(stem)
                blobs.append(tar.extractfile(m).read())
                if len(ids) % args.docci_batch == 0:
                    log(f"read {len(ids)} images")
                if len(ids) == len(desc):
                    break
    log(f"resizing {len(ids)} images")
    assert len(ids) > 0.99 * len(desc), "most chosen DOCCI images are missing from the archive"
    with ProcessPoolExecutor(args.workers) as pool:
        small = list(pool.map(_resize, zip(ids, blobs, strict=True), chunksize=32))
    del blobs
    keep = [i for i, b in enumerate(small) if b is not None]
    df = pd.DataFrame(
        {
            "example_id": [ids[i] for i in keep],
            "caption": [desc[ids[i]] for i in keep],
            "image": [small[i] for i in keep],
        }
    )
    df.to_parquet(target)
    tgz.unlink(missing_ok=True)
    log(f"DOCCI: {len(df)} rows, median words {df['caption'].str.split().str.len().median()}")


def stage_docci_features(args, docci: Path) -> None:
    from PIL import Image

    df = pd.read_parquet(docci / "docci.parquet")
    n = len(df)
    feat = docci / "features"
    feat.mkdir(exist_ok=True)
    blobs = df["image"].tolist()
    images = EX.LazyImages(n, lambda i: Image.open(io.BytesIO(blobs[i])))
    for v in args.vits:
        name = v.removesuffix("__cls")
        if (feat / f"{v}__d2l.pt").exists():
            continue
        x = EX.vit_cls(name, images, args.vit_batch, device())
        save_layers(docci, v, x, {"model": name}, n)
        free()
    rnd = f"random-{EX.RANDOM_VIT}__cls"
    if not (feat / f"{rnd}__d2l.pt").exists():
        x = EX.vit_cls(EX.RANDOM_VIT, images, args.vit_batch, device(), pretrained=False)
        save_layers(docci, rnd, x, {"model": EX.RANDOM_VIT, "random": True}, n)
        free()
    texts = df["caption"].tolist()
    sets = [(r, rev, False, args.cache_dir) for r in args.pythia for rev in ("step0", args.final)]
    sets += [(r, None, True, args.big_cache) for r in args.big_lms]
    for repo, rev, special, cache in sets:
        stem = X20.stem(repo, rev)
        if (feat / f"{stem}__docci__d2l.pt").exists():
            continue
        model, dec, _, tok = X20.load(repo, rev, "causal", device(), cache)
        x = X20.mean_pooled_ids(dec, tok, tokenise(tok, texts, special), args.tokens)
        save_layers(docci, f"{stem}__docci", x, {"model": repo, "revision": rev}, n)
        del model, dec, x
        free()
        if cache == args.cache_dir:
            shutil.rmtree(Path(cache) / f"models--{repo.replace('/', '--')}", True)


class DocciContext:
    """What `run.py` variant_features needs, for one caption set."""

    def __init__(self, args, texts: list[str]):
        from transformers import AutoTokenizer

        self.args = args
        self.tok = AutoTokenizer.from_pretrained(P1.PY_TOKENIZER, cache_dir=args.cache_dir)
        self.sets = {"orig": P1.Captions(texts, self.tok)}
        self.raw = {"orig": texts}
        self.n = len(texts)


def stage_docci_lexical(args, docci: Path, out: Path) -> None:
    df = pd.read_parquet(docci / "docci.parquet", columns=["caption"])
    ctx = DocciContext(args, df["caption"].tolist())
    args.llm_tokenizers = 0
    res = P1.Results(out / "docci_lexical.jsonl")
    n = ctx.n
    ks = P1.ks_for(n)
    variants = ["tf", "tfidf", "binary", "tfidf_sublinear", "word_tfidf", "word_tfidf_nostop"]
    variants += [f"first{m}_{w}" for m in (12, 25, 50, 100) for w in ("tf", "tfidf")]
    cs = ctx.sets["orig"]
    (out / "docci_stats.json").write_text(
        json.dumps(
            {
                "n": n,
                "tokens_median": float(np.median(cs.lengths)),
                "vocab": int(cs.counts.shape[1]),
                "r_tf": lex.participation_ratio(P1._gram(lex.to_unit_rows(lex.tf(cs.counts)))),
                "r_tfidf": lex.participation_ratio(
                    P1._gram(lex.to_unit_rows(lex.tfidf(cs.counts)))
                ),
            },
            indent=1,
        )
    )
    rnd = f"random-{EX.RANDOM_VIT}__cls"
    for t in [*args.vits, rnd]:
        layers, names = P1.load_layers(docci / "features" / f"{t}__d2l.pt")
        rep = P1.Rep(layers, names, k_max=max(ks), ranks=True)
        del layers
        for name in variants:
            if res.has(stage="docci", caption="docci", variant=name, target=t):
                continue
            v = P1.Rep([P1.variant_features(ctx, "orig", name)], [name], k_max=max(ks))
            row = {"stage": "docci", "caption": "docci", "variant": name, "target": t}
            res.add({**row, **P1.measure(rep, v, ks)})
            del v
        del rep
        gc.collect()


# ---------------------------------------------------------------- pairs


def stage_pairs(args, out: Path, feat_dir: Path, docci: Path) -> None:
    # never the DOCCI folder here: its ViT files have the same names as WIT's, on other images
    roots = [feat_dir, args.out_pythia, args.exp018, args.exp016]
    store = A20.Store(roots)
    avail = store.available()
    specs = []

    def add(group, a, b):
        if f"{a}__d2l" in avail and f"{b}__d2l" in avail:
            pid = A20.pair_id(group, a, b, "d2l")
            specs.append({"id": pid, "group": group, "a": a, "b": b, "tag": "d2l"})

    for key in sorted(avail):
        if not key.endswith("__d2l"):
            continue
        a = key.removesuffix("__d2l")
        if "-shuf" in a:
            for v in VITS_2:
                add("SHUF", a, v)
        elif "@blocks-" in a:
            for v in VITS_2:
                add("FACT", a, v)
    n = args.limit or 10240
    ks = [int(k) for k in KS.split(",") if int(k) < n // 2]
    log(f"pairs: {len(specs)} on WIT")
    A20.stage_pairs(specs, store, out, device(), args.permutations, 0.05, ks, 2)
    if not (docci / "docci.parquet").exists():
        return
    # DOCCI has its own feature files with the same ViT names, so it gets its own store and folder
    dstore = A20.Store([docci])
    davail = dstore.available()
    dspecs = []
    rnd = f"random-{EX.RANDOM_VIT}__cls"
    for key in sorted(davail):
        if key.endswith("__docci__d2l"):
            a = key.removesuffix("__d2l")
            for v in [*args.vits, rnd]:
                pid = A20.pair_id("DOCCI", a, v, "d2l")
                dspecs.append({"id": pid, "group": "DOCCI", "a": a, "b": v, "tag": "d2l"})
    (out / "docci_pairs").mkdir(parents=True, exist_ok=True)
    dn = len(pd.read_parquet(docci / "docci.parquet", columns=["caption"]))
    dks = [int(k) for k in KS.split(",") if int(k) < dn // 2]
    A20.stage_pairs(dspecs, dstore, out / "docci_pairs", device(), args.permutations, 0.05, dks, 2)


# ---------------------------------------------------------------- ridge


@torch.no_grad()
def held_out(inputs: dict[str, torch.Tensor], y: torch.Tensor, seed: int, dev: str) -> dict:
    """5-fold ridge on PCA inputs (exp-017): one model per input set, and all of them jointly."""
    parts = MAPS.folds(y.shape[0], seed)
    out: dict[str, list[dict]] = {}
    for f, test in enumerate(parts):
        train = np.concatenate([p for i, p in enumerate(parts) if i != f])
        rng = np.random.default_rng(seed + 1000 + f)
        mask = np.zeros(len(train), bool)
        mask[rng.choice(len(train), int(MAPS.VAL_FRAC * len(train)), replace=False)] = True
        vm = torch.from_numpy(mask).to(dev)
        tr, te = torch.from_numpy(train).to(dev), torch.from_numpy(test).to(dev)
        py = MAPS.PCA(y[tr])
        t, tq = py(y[tr]), py(y[te])
        z_tr, z_te = {}, {}
        for name, x in inputs.items():
            pca = MAPS.PCA(x[tr].to_dense() if x.is_sparse else x[tr])
            z_tr[name], z_te[name] = pca(x[tr]), pca(x[te])
        for name in [*inputs, "+".join(inputs)]:
            keys = name.split("+")
            a, b = torch.cat([z_tr[k] for k in keys], 1), torch.cat([z_te[k] for k in keys], 1)
            c = MAPS.fit_ridge(a[~vm], t[~vm], a[vm], t[vm])
            p = MAPS.ridge_predict(a, t, b, c)
            out.setdefault(name, []).append({**MAPS.readouts(p, tq), "c": c})
    return {
        k: {m: float(np.mean([x[m] for x in v])) for m in ("r2", "top1", "top5", "mknn")}
        for k, v in out.items()
    }


def ridge_calibration(dev: str) -> dict:
    """Known answers: a target shared only with words, shared only with deep features, or both."""
    gen = torch.Generator().manual_seed(0)
    n, d = 4096, 512

    def embed(z):
        return z @ torch.randn(z.shape[1], d, generator=gen) + 0.3 * torch.randn(
            n, d, generator=gen
        )

    words, extra = torch.randn(n, 40, generator=gen), torch.randn(n, 40, generator=gen)
    lexical, deep_noinfo = embed(words), embed(words)
    deep_extra = embed(torch.cat([words, extra], 1))
    res = {}
    for name, target, deep in (
        ("target_words__deep_is_words", words, deep_noinfo),
        ("target_words__deep_has_more", words, deep_extra),
        ("target_extra__deep_has_more", extra, deep_extra),
    ):
        y = embed(target)
        res[name] = held_out({"lexical": lexical.to(dev), "deep": deep.to(dev)}, y.to(dev), 0, dev)
        log(f"ridge calibration {name}: {res[name]}")
    return res


def stage_ridge(args, out: Path) -> None:
    dev = device()
    res = {"calibration": ridge_calibration(dev), "pairs": []}
    (out / "ridge.json").write_text(json.dumps(res, indent=1))
    ctx = P1.Context(argparse.Namespace(**{**vars(args), "llm_tokenizers": 0}))
    n = ctx.n
    ldefs = {
        c: {w: P1.variant_features(ctx, c, w) for w in ("tfidf", "tfidf_sublinear")}
        for c in ("gemini", "orig")
    }
    lm_sets = [
        ("gemini", args.exp018 / "features" / f"EleutherAI__pythia-410m@{FINAL}__gemini__d2l.pt"),
        ("orig", args.out_pythia / "features" / f"EleutherAI__pythia-410m@{FINAL}__orig__d2l.pt"),
    ]
    for repo in BIG_LMS:
        k = repo.replace("/", "__")
        lm_sets += [(c, args.exp016 / "features" / f"{k}__{c}__d2l.pt") for c in ("gemini", "orig")]
    for t in VITS_2:
        tl, _ = P1.load_layers(P1.feature_path(args, t))
        y = tl[-1].to(dev)
        del tl
        for caption, path in lm_sets:
            if not path.exists():
                log(f"ridge: {path.name} missing, SKIPPED")
                continue
            layers, names = P1.load_layers(path)
            for depth, i in (("last", len(names) - 1), ("deep5", 5), ("layer0", 0)):
                for w, lexical in ldefs[caption].items():
                    r = held_out({"lexical": lexical.to(dev), "lm": layers[i].to(dev)}, y, 0, dev)
                    res["pairs"].append(
                        {
                            "target": t,
                            "caption": caption,
                            "lm_file": path.name,
                            "layer": names[i],
                            "depth": depth,
                            "lexical_def": w,
                            "n": n,
                            **r,
                        }
                    )
                    (out / "ridge.json").write_text(json.dumps(res, indent=1))
                    log(f"ridge {path.name[:40]} {depth} {w} x {t[:12]}: {r}")
            del layers
            free()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="results on the volume")
    p.add_argument("--features", type=Path, required=True, help="scratch for shuffle and factorial")
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, required=True)
    p.add_argument("--out-pythia", type=Path, required=True, help="Pythia on original captions")
    p.add_argument("--docci", type=Path, required=True)
    p.add_argument("--docci-scratch", type=Path, default=Path("/root/docci"))
    p.add_argument("--docci-text", default=None, help="local file instead of the download (tests)")
    p.add_argument(
        "--docci-images", default=None, help="local file instead of the download (tests)"
    )
    p.add_argument("--docci-n", type=int, default=10240)
    p.add_argument("--docci-batch", type=int, default=2000)
    p.add_argument(
        "--stages",
        default="shuffle,factorial,docci_prepare,docci_features,docci_lexical,pairs,ridge",
    )
    p.add_argument("--pythia", type=lambda s: [x for x in s.split(",") if x], default=PYTHIA_FULL)
    p.add_argument("--big-lms", type=lambda s: [x for x in s.split(",") if x], default=BIG_LMS)
    p.add_argument("--big-cache", default=BIG_CACHE)
    p.add_argument("--final", default=FINAL)
    p.add_argument("--vits", type=lambda s: s.split(","), default=VITS_4)
    p.add_argument("--cache-dir", type=Path, default=Path("/root/hf-scratch"))
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--vit-batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--permutations", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.features.mkdir(parents=True, exist_ok=True)
    stages = {
        "shuffle": lambda: stage_shuffle(args, args.features),
        "factorial": lambda: stage_factorial(args, args.features),
        "docci_prepare": lambda: stage_docci_prepare(args, args.docci),
        "docci_features": lambda: stage_docci_features(args, args.docci),
        "docci_lexical": lambda: stage_docci_lexical(args, args.docci, args.out),
        "pairs": lambda: stage_pairs(args, args.out, args.features, args.docci),
        "ridge": lambda: stage_ridge(args, args.out),
    }
    failed = []
    for name in args.stages.split(","):
        log(f"=== stage {name}")
        t0 = time.time()
        try:
            stages[name]()
            log(f"=== stage {name} done in {time.time() - t0:.0f}s")
        except Exception:  # a failed stage must not cost the others; the exit code reports it
            traceback.print_exc()
            log(f"STAGE FAILED: {name}")
            failed.append(name)
    if failed:
        raise SystemExit(f"failed stages: {failed}")


if __name__ == "__main__":
    main()
