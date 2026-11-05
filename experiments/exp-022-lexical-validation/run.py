"""exp-022 phase 1: is the lexical floor of exp-021 real? CPU only, cached data, no forward passes.

Stages (see README for the check each serves and its pass rule):
- captions: raw caption inspection, template wrapper, language (V6, V8);
- identity: rebuild layer 0 from the real tables; clamp variants; Gaussian seeds (V1);
- numerics: exact float64 CKA and mutual kNN against stored scores (V2);
- grids: full layer grids from the stored exp-016 / exp-018 arrays (V4, V10);
- lexical: lexical variants against DINOv2 and CLIP (V6, V7, V8), permutation check of the nulls;
- targets: TF and TF-IDF against the other encoders and LMs (V9, V11);
- trained: debiased CKA of trained LMs, large-k mutual kNN on exp-016's LMs (V3, V10);
- subsets: disjoint and nested subsets, idf from the other half (V3, V7);
- concentration: overlap distribution, near-duplicates, within-cluster scores (V5);
- split: best cell chosen on one half of the samples, scored on the other (V4).

Null means are exact, not sampled: for mutual kNN k / (n - 1), for CKA of centred unit kernels
tr(A) tr(B) / (n - 1) under a uniform permutation. Stage `lexical` checks both against permutations.

    uv run python experiments/exp-022-lexical-validation/run.py --out <dir> --exp016 <dir> --exp018 <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp-016-positive-controls"))
import measure as M16  # noqa: E402  PRH preprocessing, as every earlier experiment

from geoprh import knn_sweep as ksw  # noqa: E402
from geoprh import lexical as lex  # noqa: E402

DINO = "vit_giant_patch14_dinov2.lvd142m__cls"
CLIP = "vit_huge_patch14_clip_224.laion2b__cls"
MAE = "vit_huge_patch14_224.mae__cls"
AUGREG = "vit_large_patch16_224.augreg_in21k__cls"
TOWERS = [f"{t}__image" for t in ("clip-L14", "openclip-H14", "siglip-so400m")]
RANDOM_VIT = "random-vit_base_patch16_224.augreg_in21k__cls"
REF = "EleutherAI__pythia-1.4b__gemini"
PY_SERIES = ["pythia-160m", "pythia-160m-seed1", "pythia-410m", "pythia-410m-seed1"]
FINAL = "step143000"
LLMS = [
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_7b",
    "huggyllama/llama-13b",
]
PY_TOKENIZER = "EleutherAI/pythia-160m"
MAX_TOKENS = 2048  # as exp-020 extract_lm.py
KS = [10, 30, 100, 300, 1000]
LENGTHS = [12, 25, 50, 100, 200, 400]
FENCE = re.compile(r"^\s*```(?:json)?|```\s*$")
WRAPPER = re.compile(r'^\{\s*"500w"\s*:\s*"?(.*?)"?\s*\}$', re.S)


def load_extract_lm():
    spec = importlib.util.spec_from_file_location(
        "extract_lm", ROOT / "exp-020-lm-data-pilot" / "extract_lm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- representations


class Rep:
    """Per-layer centred Grams, unbiased-HSIC parts and neighbour data of one representation."""

    def __init__(self, layers, names, k_max=0, ranks=False, centred=True, hsic=True):
        self.names = list(names)
        self.n = layers[0].shape[0]
        self.g, self.tr, self.h = [], [], []
        idx = []
        for x in layers:
            k = x @ x.T
            if centred:
                c = lex.centred_unit_gram(k)
                self.tr.append(float(c.trace()))
                self.g.append(c.float())
                del c
            if hsic:
                self.h.append(lex.HsicU(k))
            if k_max and not ranks:
                k = k.clone()
                k.fill_diagonal_(float("-inf"))
                idx.append(k.topk(k_max, dim=1).indices)
            del k
        self.sweep = None
        if k_max:
            if ranks:
                self.sweep = ksw.build_sweep_cache(layers, k_max)
            else:
                self.sweep = ksw.SweepCache(ranks=None, idx=torch.stack(idx))

    @property
    def k_max(self) -> int:
        return self.sweep.k_max


def load_layers(path: Path, keep=None):
    layers, names = M16.prepared_layers(path)
    if keep is not None:
        pairs = [(x, m) for x, m in zip(layers, names, strict=True) if m in keep]
        layers, names = [x for x, _ in pairs], [m for _, m in pairs]
    return layers, names


def measure(a: Rep, b: Rep, ks: list[int], knn: bool = True, debiased: bool = True) -> dict:
    """A x B, arrays [La, Lb]. A needs ranks for kNN, B neighbour lists."""
    n = a.n
    out = {"names_a": a.names, "names_b": b.names, "n": n}
    if a.g and b.g:
        gb = [x.flatten() for x in b.g]
        out["cka"] = [[float(torch.dot(x.flatten(), y)) for y in gb] for x in a.g]
        out["cka_mu0"] = [[ta * tb / (n - 1) for tb in b.tr] for ta in a.tr]
    if debiased and a.h and b.h:
        out["cka_u"] = [[x.cka(y) for y in b.h] for x in a.h]
    if knn:
        s = ksw.scores(a.sweep, b.sweep, ks)
        out["knn"] = {str(k): s[i].tolist() for i, k in enumerate(ks)}
        out["knn_mu0"] = {str(k): k / (n - 1) for k in ks}
    return out


def trained_sets(args) -> dict[str, Path]:
    """Feature files of every trained or checkpoint LM set, by key."""
    out = {}
    for s in PY_SERIES:
        for rev in ("step0", FINAL):
            key = f"EleutherAI__{s}@{rev}__gemini"
            out[key] = args.exp018 / "features" / f"{key}__d2l.pt"
    for m in LLMS:
        for c in ("gemini", "orig"):
            key = f"{m.replace('/', '__')}__{c}"
            out[key] = args.exp016 / "features" / f"{key}__d2l.pt"
    return out


def feature_path(args, key: str) -> Path:
    for root in (args.exp016, args.exp018):
        p = root / "features" / f"{key}__d2l.pt"
        if p.exists():
            return p
    raise FileNotFoundError(key)


# ---------------------------------------------------------------- captions and variants


class Captions:
    def __init__(self, texts: list[str], tok, special: bool = False):
        self.texts = texts
        self.ids = tok(texts, truncation=True, max_length=MAX_TOKENS, add_special_tokens=special)[
            "input_ids"
        ]
        self.counts, self.lengths, self.vocab = lex.count_matrix(self.ids)


class Context:
    def __init__(self, args):
        from transformers import AutoTokenizer

        self.args = args
        x20 = load_extract_lm()
        caps = x20.captions(args.exp016 / "d2.parquet", args.limit)
        self.raw = {"gemini": caps["gemini"], "orig": caps["orig"]}
        self.clean_gemini, self.n_wrapped = strip_wrapper(caps["gemini"])
        self.tok = AutoTokenizer.from_pretrained(PY_TOKENIZER, cache_dir=args.cache_dir)
        self.sets = {
            "gemini": Captions(self.clean_gemini, self.tok),
            "orig": Captions(caps["orig"], self.tok),
        }
        self.n = len(caps["orig"])


def strip_wrapper(texts: list[str]) -> tuple[list[str], int]:
    """Remove the prompt's `{"500w": ...}` JSON wrapper and code fences, if the string has them."""
    out, n = [], 0
    for t in texts:
        body = FENCE.sub("", t).strip()
        if '"500w"' not in body[:40]:
            out.append(t)
            continue
        n += 1
        try:
            out.append(json.loads(body)["500w"])
        except (json.JSONDecodeError, KeyError, TypeError):
            m = WRAPPER.match(body)
            out.append(m[1] if m else t)
    return out, n


def variant_features(ctx: Context, caption: str, name: str) -> torch.Tensor:
    """The l2-normalised rows of one lexical variant; `name` carries its parameters."""
    s = Captions(ctx.raw["gemini"], ctx.tok) if caption == "gemini_raw" else ctx.sets[caption]
    if name == "tf":
        return lex.to_unit_rows(lex.tf(s.counts))
    if name == "tfidf":
        return lex.to_unit_rows(lex.tfidf(s.counts))
    if name == "binary":
        return lex.to_unit_rows(lex.binary(s.counts))
    if name == "tfidf_sublinear":
        return lex.to_unit_rows(lex.tfidf(s.counts, sublinear=True))
    if m := re.fullmatch(r"mask(\d+)_(tf|tfidf)", name):
        c = lex.mask_top(s.counts, int(m[1]))
        return lex.to_unit_rows(lex.tf(c) if m[2] == "tf" else lex.tfidf(c))
    if m := re.fullmatch(r"(first|random)(\d+|matched)_(tf|tfidf)", name):
        length = ctx.sets["orig"].lengths if m[2] == "matched" else int(m[2])
        c = lex.count_matrix(lex.truncate(s.ids, length, m[1]))[0]
        return lex.to_unit_rows(lex.tf(c) if m[3] == "tf" else lex.tfidf(c))
    if m := re.fullmatch(r"tfidf_table_d(\d+)", name):
        gen = torch.Generator().manual_seed(0)
        table = torch.randn(s.counts.shape[1], int(m[1]), generator=gen).numpy()
        x = torch.from_numpy(np.asarray(lex.tfidf(s.counts) @ table)).float()
        return prh(x)
    if m := re.fullmatch(r"word_tfidf(_nostop)?", name):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            lowercase=True, stop_words="english" if m[1] else None, norm="l2", dtype=np.float32
        )
        return lex.to_unit_rows(vec.fit_transform(s.texts))
    if m := re.fullmatch(r"tok-(.+)_(tf|tfidf)", name):
        c = llm_caption_counts(ctx, caption, m[1])
        return lex.to_unit_rows(lex.tf(c) if m[2] == "tf" else lex.tfidf(c))
    raise ValueError(name)


def llm_caption_counts(ctx: Context, caption: str, repo_key: str) -> sp.csr_matrix:
    """Exp-016 tokenised with the model's special tokens (the mean includes them)."""
    from transformers import AutoTokenizer

    repo = repo_key.replace("__", "/")
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=ctx.args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return Captions(ctx.sets[caption].texts, tok, special=True).counts


def prh(x: torch.Tensor) -> torch.Tensor:
    from aristotelian.prh.preprocess import prepare_features

    return torch.nn.functional.normalize(prepare_features(x[:, None], q=0.95, exact=False), dim=-1)[
        :, 0
    ]


class Results:
    def __init__(self, path: Path):
        self.path = path
        self.done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.done.add(self.key(r))

    @staticmethod
    def key(r: dict) -> tuple:
        return (r["stage"], r["caption"], r["variant"], r["target"], r.get("subset", ""))

    def has(self, **r) -> bool:
        return self.key(r) in self.done

    def add(self, row: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        self.done.add(self.key(row))
        best = ""
        if "cka" in row:
            c, mu = np.array(row["cka"]), np.array(row["cka_mu0"])
            best = f"CKA excess {(c - mu).max():.3f}"
        if "cka_u" in row:
            best += f" debiased {np.array(row['cka_u']).max():.3f}"
        if "knn" in row and "10" in row["knn"]:
            best += f" knn10 excess {(np.array(row['knn']['10']) - row['knn_mu0']['10']).max():.3f}"
        log(
            f"{row['stage']} {row['caption']} {row['variant']} x {row['target']} {row.get('subset', '')}: {best}"
        )


# ---------------------------------------------------------------- stages


def stage_captions(ctx: Context, out: Path) -> None:
    rep = {"n_wrapped_gemini": ctx.n_wrapped}
    for c, texts in ctx.raw.items():
        s = pd.Series(texts)
        rep[c] = {
            "chars_median": float(s.str.len().median()),
            "head": [t[:160] for t in texts[:6]],
            "tail": [t[-80:] for t in texts[:3]],
            "contains_500w": int(s.str.contains('"500w"', regex=False).sum()),
            "starts_brace": int(s.str.lstrip().str.startswith("{").sum()),
            "non_ascii_rows": int((s.str.contains(r"[^\x00-\x7F]")).sum()),
            "duplicate_rows": int(s.duplicated().sum()),
        }
    for c in ("gemini", "orig"):
        cs = ctx.sets[c]
        total = np.asarray(cs.counts.sum(0)).ravel()
        df = np.asarray((cs.counts > 0).sum(0)).ravel()
        top = np.argsort(total)[::-1][:40]
        rep[c].update(
            tokens_median=float(np.median(cs.lengths)),
            tokens_q=[float(q) for q in np.quantile(cs.lengths, [0.05, 0.25, 0.75, 0.95])],
            vocab=int(cs.counts.shape[1]),
            top_tokens=[
                {
                    "tok": ctx.tok.decode([int(cs.vocab[i])]),
                    "share": float(total[i] / total.sum()),
                    "df": float(df[i] / ctx.n),
                }
                for i in top
            ],
            r_tf=lex.participation_ratio(_gram(lex.to_unit_rows(lex.tf(cs.counts)))),
            r_tfidf=lex.participation_ratio(_gram(lex.to_unit_rows(lex.tfidf(cs.counts)))),
        )
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        langs = []
        for t in ctx.raw["orig"]:
            try:
                langs.append(detect(t))
            except Exception:
                langs.append("unknown")
        rep["orig_language"] = pd.Series(langs).value_counts().head(15).to_dict()
        np.save(out / "orig_is_english.npy", np.array([x == "en" for x in langs]))
    except ImportError:
        log("langdetect not installed: language check SKIPPED")
    (out / "captions.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    log(f"captions: {ctx.n_wrapped} wrapped; {rep.get('orig_language')}")


def _gram(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T


def table_stats(table: torch.Tensor, cs: Captions) -> dict:
    """Is the table isotropic? Row norms by token frequency; mean vector; top PC share."""
    e = table[torch.from_numpy(cs.vocab.astype(np.int64))]
    norms = e.norm(dim=1)
    total = np.asarray(cs.counts.sum(0)).ravel()
    df = lex.idf(cs.counts)
    weights = torch.from_numpy(total / total.sum()).float()
    sample = e[torch.randperm(e.shape[0], generator=torch.Generator().manual_seed(0))[:4096]]
    sv = torch.linalg.svdvals(sample - sample.mean(0))
    return {
        "shape": list(table.shape),
        "norm_mean": float(norms.mean()),
        "norm_cv": float(norms.std() / norms.mean()),
        "norm_freq_weighted": float((norms * weights).sum()),
        "mean_vector_norm_ratio": float(e.mean(0).norm() / norms.mean()),
        "top_pc_share": float(sv[0] ** 2 / (sv**2).sum()),
        "pr_rows": float((sv**2).sum() ** 2 / (sv**4).sum()),
        "spearman_norm_idf": float(pd.Series(norms.numpy()).corr(pd.Series(df), method="spearman")),
    }


def stage_identity(ctx: Context, out: Path) -> None:
    from transformers import AutoModelForCausalLM

    args = ctx.args
    res = {"sets": [], "gaussian": []}
    cs = ctx.sets["gemini"]
    tf = lex.tf(cs.counts)
    tfx = lex.to_unit_rows(tf)
    tf_rep = Rep([tfx], ["tf"], k_max=10, ranks=True, hsic=False)
    del tfx
    keys = {k: feature_path(args, k) for k in (DINO, CLIP, REF)}
    targets = {
        k: Rep(*load_layers(p), centred=True, hsic=False) for k, p in keys.items()
    }  # Grams only
    log("identity: targets loaded")

    def against(rep: Rep, row: dict) -> None:
        row["cka_tf"] = measure(tf_rep, rep, [10], debiased=False)
        for k, t in targets.items():
            row[f"cka_{k[:12]}"] = measure(t, rep, [10], knn=False, debiased=False)

    for repo, rev, key in [tuple(x.split(",")) for x in args.identity_sets.split(";") if x]:
        rev = rev or None
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=rev, cache_dir=args.cache_dir, torch_dtype=torch.float32
        )
        table = model.get_input_embeddings().weight.detach().clone()
        del model
        shutil.rmtree(args.cache_dir / f"models--{repo.replace('/', '--')}", True)
        row = {"repo": repo, "rev": rev, "table": table_stats(table, cs)}
        rebuilt = torch.from_numpy(
            np.asarray(tf @ table[torch.from_numpy(cs.vocab.astype(np.int64))].numpy())
        ).float()
        del table
        path = feature_path(args, key)
        stored = torch.load(path, weights_only=False)
        assert stored["layer_idx"][0] == 0
        s0 = stored["layers"][:, 0].float()
        del stored
        cos = torch.nn.functional.cosine_similarity(rebuilt, s0, dim=1)
        row["row_cosine"] = {
            "mean": float(cos.mean()),
            "min": float(cos.min()),
            "q01": float(cos.quantile(0.01)),
        }
        row["norm_ratio_median"] = float((s0.norm(dim=1) / rebuilt.norm(dim=1)).median())
        layers, _ = load_layers(path)
        versions = {
            "pipeline_clamp_all_layers": layers[0],
            "own_clamp": prh(s0),
            "no_clamp": torch.nn.functional.normalize(s0, dim=-1),
            "rebuilt_own_clamp": prh(rebuilt),
        }
        del layers
        row["clamp_fraction"] = float(
            (s0.abs() >= torch.quantile(s0.abs().flatten(1), 0.95, dim=1).mean()).float().mean()
        )
        row["versions"] = {}
        for name, x in versions.items():
            rep = Rep([x], [name], k_max=10, hsic=False)
            v = {}
            against(rep, v)
            row["versions"][name] = v
            del rep
        res["sets"].append(row)
        (out / "identity.json").write_text(json.dumps(res, indent=1, default=float))
        log(f"identity {repo} {rev}: row cosine {row['row_cosine']}")
    for d in args.gaussian_widths:
        for seed in range(5):
            gen = torch.Generator().manual_seed(seed)
            table = torch.randn(cs.counts.shape[1], d, generator=gen).numpy()
            x = prh(torch.from_numpy(np.asarray(tf @ table)).float())
            rep = Rep([x], ["table"], k_max=10, hsic=False)
            v = {"d": d, "seed": seed}
            against(rep, v)
            res["gaussian"].append(v)
            del rep
    (out / "identity.json").write_text(json.dumps(res, indent=1, default=float))
    log("identity: done")


def stage_numerics(ctx: Context, out: Path) -> None:
    args = ctx.args
    wanted = {
        args.exp018: [
            f"LV__EleutherAI__pythia-160m@step0__gemini__{DINO}__d2l",
            f"LV__EleutherAI__pythia-410m@{FINAL}__gemini__{DINO}__d2l",
            f"LV__EleutherAI__pythia-160m@{FINAL}__gemini__{CLIP}__d2l",
            f"LL__EleutherAI__pythia-410m@{FINAL}__gemini__{REF}__d2l",
        ],
        args.exp016: [
            f"B3_gemini__huggyllama__llama-13b__gemini__{DINO}__d2l",
            f"B3_orig__openlm-research__open_llama_7b__orig__{CLIP}__d2l",
            "A_orig__openclip-H14__text-orig__openclip-H14__image__d2l",
            f"D_random__{RANDOM_VIT}__bigscience__bloomz-1b7__gemini__d2l",
        ],
    }
    rows = []
    for root, ids in wanted.items():
        rec_path = root / "pairs.jsonl"
        if not rec_path.exists():
            continue
        recs = {r["id"]: r for r in map(json.loads, rec_path.read_text().splitlines())}
        for pid in ids:
            if pid not in recs:
                log(f"numerics: {pid} not found, SKIPPED")
                continue
            r = recs[pid]
            la, _ = load_layers(feature_path(args, f"{r['a']}"))
            lb, _ = load_layers(feature_path(args, f"{r['b']}"))
            i, j = r["cka_lin"]["best"]["cell"]
            exact = lex.cka_features(la[i], lb[j])
            ki, kj = r["mutual_knn"]["best"]["cell"]
            na = ksw.build_sweep_cache([la[ki]], 10)
            nb = ksw.build_sweep_cache([lb[kj]], 10)
            knn = float(ksw.scores(na, nb, [10])[0, 0, 0])
            rows.append(
                {
                    "id": pid,
                    "cka_stored": r["cka_lin"]["best"]["score"],
                    "cka_exact": exact,
                    "knn_stored": r["mutual_knn"]["best"]["score"],
                    "knn_exact": knn,
                }
            )
            log(
                f"numerics {pid[:60]}: CKA {exact:.4f} vs {rows[-1]['cka_stored']:.4f}, knn {knn:.4f} vs {rows[-1]['knn_stored']:.4f}"
            )
            del la, lb, na, nb
    (out / "numerics.json").write_text(json.dumps(rows, indent=1))


def stage_grids(ctx: Context, out: Path) -> None:
    rows = []
    for root in (ctx.args.exp016, ctx.args.exp018):
        rec = root / "pairs.jsonl"
        if not rec.exists():
            continue
        for line in rec.read_text().splitlines():
            r = json.loads(line)
            if r.get("tag") != "d2l":
                continue
            z = np.load(root / "pairs" / f"{r['id']}.npz")
            m = {"cka": "cka_lin", "knn10": "mutual_knn"}
            for k in r.get("knn_sweep", {}):
                if k != "10":
                    m[f"knn{k}"] = f"sweep_k{k}"
            row = {x: r.get(x) for x in ("id", "group", "a", "b", "n", "names_a", "names_b")}
            row["best_cka"] = r["cka_lin"]["best"]["cell"]
            for name, key in m.items():
                row[name] = np.round(z[f"{key}_score"], 5).tolist()
                row[f"{name}_mu0"] = np.round(z[f"{key}_mu0"], 5).tolist()
            rows.append(row)
    (out / "grids_full.json").write_text(json.dumps(rows))
    log(f"grids: {len(rows)} pairs")


def lexical_plan(ctx: Context) -> list[tuple[str, str]]:
    plan = [("gemini", v) for v in ("tf", "tfidf")] + [("orig", v) for v in ("tf", "tfidf")]
    for mode in ("first", "random"):
        for length in LENGTHS:
            plan += [("gemini", f"{mode}{length}_{w}") for w in ("tf", "tfidf")]
        plan += [("gemini", f"{mode}matched_{w}") for w in ("tf", "tfidf")]
    if ctx.n_wrapped:
        plan += [("gemini_raw", v) for v in ("tf", "tfidf")]
    for k in (10, 100, 500):
        plan += [("gemini", f"mask{k}_tf"), ("gemini", f"mask{k}_tfidf")]
    plan += [("orig", f"mask{k}_tf") for k in (10,)]
    for c in ("gemini", "orig"):
        plan += [(c, v) for v in ("binary", "tfidf_sublinear", "word_tfidf", "word_tfidf_nostop")]
    plan += [("gemini", "tfidf_table_d4096")]
    if ctx.args.llm_tokenizers:
        for repo in LLMS:
            key = repo.replace("/", "__")
            for c in ("gemini", "orig"):
                plan += [(c, f"tok-{key}_{w}") for w in ("tf", "tfidf")]
    return plan


def perm_check(a: Rep, b: Rep, ks: list[int], permutations: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    n = a.n
    cka, knn = [], []
    tflat = torch.stack([x.flatten() for x in a.g])
    gram = b.g[0]
    for _ in range(permutations):
        p = torch.randperm(n, generator=gen)
        cka.append(tflat @ gram[p][:, p].flatten())
        knn.append(ksw.scores(a.sweep, b.sweep, ks, p)[:, :, 0])
    return {
        "cka_perm_mu0": torch.stack(cka).mean(0).tolist(),
        "cka_analytic_mu0": [t * b.tr[0] / (n - 1) for t in a.tr],
        "knn_perm_mu0": {str(k): torch.stack(knn).mean(0)[i].tolist() for i, k in enumerate(ks)},
        "knn_analytic_mu0": {str(k): k / (n - 1) for k in ks},
    }


def ks_for(n: int) -> list[int]:
    return [k for k in KS if k < n - 1]


def stage_lexical(ctx: Context, out: Path, res: Results) -> None:
    args = ctx.args
    ks = ks_for(ctx.n)
    k_max = max(ks)
    targets = {
        k: Rep(*load_layers(feature_path(args, k)), k_max=k_max, ranks=True) for k in (DINO, CLIP)
    }
    log("lexical: targets loaded")
    checks = []
    for caption, name in lexical_plan(ctx):
        todo = [
            t
            for t in targets
            if not res.has(stage="lexical", caption=caption, variant=name, target=t)
        ]
        check = name in ("tf", "tfidf") and caption in ("gemini", "orig")
        if not todo and not check:
            continue
        try:
            x = variant_features(ctx, caption, name)
        except Exception as e:  # a missing tokenizer must not cost the remaining variants
            log(f"lexical {caption} {name}: SKIPPED ({e})")
            continue
        empty = int((x.norm(dim=1) == 0).sum())
        v = Rep([x], [name], k_max=k_max)
        del x
        for t in todo:
            row = {"stage": "lexical", "caption": caption, "variant": name, "target": t}
            res.add({**row, "empty_rows": empty, **measure(targets[t], v, ks)})
        if check:
            pc = perm_check(
                targets[DINO], v, ks[:1] + ks[-1:], args.perm_check, zlib.crc32(name.encode())
            )
            checks.append({"caption": caption, "variant": name, **pc})
            (out / "perm_check.json").write_text(json.dumps(checks, indent=1, default=float))
        del v


def stage_targets(ctx: Context, out: Path, res: Results) -> None:
    args = ctx.args
    ks = ks_for(ctx.n)
    k_max = max(ks)
    variants = {}
    for c in ("gemini", "orig"):
        for name in ("tf", "tfidf"):
            variants[(c, name)] = Rep([variant_features(ctx, c, name)], [name], k_max=k_max)
    gem_only = [REF] + [f"EleutherAI__{s}@{r}__gemini" for s in PY_SERIES for r in ("step0", FINAL)]
    for t in [MAE, AUGREG, RANDOM_VIT, *TOWERS, *gem_only]:
        pairs = [
            k
            for k in variants
            if not (t in gem_only and k[0] != "gemini")
            and not res.has(stage="targets", caption=k[0], variant=k[1], target=t)
        ]
        if not pairs:
            continue
        layers, names = load_layers(feature_path(args, t))
        rep = Rep(layers, names, k_max=k_max, ranks=True)
        del layers
        for k in pairs:
            row = {"stage": "targets", "caption": k[0], "variant": k[1], "target": t}
            res.add({**row, **measure(rep, variants[k], ks)})
        del rep


def stage_trained(ctx: Context, out: Path, res: Results) -> None:
    """Debiased CKA of every trained LM set; large-k mutual kNN for exp-016's LMs."""
    args = ctx.args
    ks = ks_for(ctx.n)
    k_max = max(ks)
    vits = {}
    for t in (DINO, CLIP, MAE, AUGREG):
        full = t in (DINO, CLIP)
        vits[t] = Rep(
            *load_layers(feature_path(args, t)), k_max=k_max if full else 0, centred=False
        )
    log("trained: ViTs loaded")
    for key, path in trained_sets(args).items():
        if not path.exists():
            log(f"trained: {key} missing, SKIPPED")
            continue
        wanted = [DINO, CLIP] if key.startswith("EleutherAI") else [DINO, CLIP, MAE, AUGREG]
        wanted = [
            t for t in wanted if not res.has(stage="trained", caption="", variant=key, target=t)
        ]
        if not wanted:
            continue
        sweep = not key.startswith("EleutherAI")  # exp-018 stored its k sweep already
        layers, names = load_layers(path)
        rep = Rep(layers, names, k_max=k_max if sweep else 0, ranks=sweep, centred=False)
        del layers
        for t in wanted:
            knn = sweep and t in (DINO, CLIP)
            row = {"stage": "trained", "caption": "", "variant": key, "target": t}
            res.add({**row, **measure(rep, vits[t], ks, knn=knn)})
        del rep


def sub_rep(layers, idx, names, k_max, ranks) -> Rep:
    return Rep([x[idx] for x in layers], names, k_max=k_max, ranks=ranks)


def stage_subsets(ctx: Context, out: Path, res: Results) -> None:
    """Disjoint blocks and nested sets; k = 10 and k = n / 10. Rows keep their full-data norm."""
    args = ctx.args
    n = ctx.n
    block = n // 10
    subsets = {f"block{i}": np.arange(i * block, (i + 1) * block) for i in range(10)}
    for m in (2048, 4096):  # n = 10240 is covered by the stages lexical and trained
        if m < n:
            subsets[f"first{m}"] = np.arange(m)
    tgt = {k: load_layers(feature_path(args, k)) for k in (DINO, CLIP)}
    reps = {}
    for c in ("gemini", "orig"):
        for name in ("tf", "tfidf"):
            reps[(c, name)] = ([variant_features(ctx, c, name)], [name])

    def add_trained(key, keep_last_only=False):
        layers, names = load_layers(feature_path(args, key))
        for i in (0, len(names) - 1):
            reps[(key.rsplit("__", 1)[1], f"{key}:{names[i]}")] = ([layers[i]], [names[i]])

    add_trained(f"EleutherAI__pythia-410m@{FINAL}__gemini")
    add_trained("huggyllama__llama-13b__gemini")
    add_trained("huggyllama__llama-13b__orig")
    log("subsets: features loaded")
    for label, idx in subsets.items():
        m = len(idx)
        ks = [10, m // 10] if m // 10 > 10 else [10]
        tidx = torch.from_numpy(idx)
        for t, (layers, names) in tgt.items():
            todo = [
                k
                for k in reps
                if not res.has(stage="subsets", caption=k[0], variant=k[1], target=t, subset=label)
            ]
            if not todo:
                continue
            trep = sub_rep(layers, tidx, names, max(ks), True)
            for k in todo:
                vl, vn = reps[k]
                v = sub_rep(vl, tidx, vn, max(ks), False)
                row = {
                    "stage": "subsets",
                    "caption": k[0],
                    "variant": k[1],
                    "target": t,
                    "subset": label,
                }
                res.add({**row, **measure(trep, v, ks)})
                del v
            del trep
    # idf from the other half, scored on half A
    half = np.random.default_rng(0).permutation(n)[: n // 2]
    other = np.setdiff1d(np.arange(n), half)
    tidx = torch.from_numpy(half)
    ks = [10, 100]
    for t, (layers, names) in tgt.items():
        trep = sub_rep(layers, tidx, names, max(ks), True)
        for c in ("gemini", "orig"):
            counts = ctx.sets[c].counts
            for label, w in (("own", lex.idf(counts[half])), ("cross", lex.idf(counts[other]))):
                x = lex.to_unit_rows(lex.tfidf(counts[half], weights=w))
                v = Rep([x], ["tfidf"], k_max=max(ks))
                row = {
                    "stage": "crossfit",
                    "caption": c,
                    "variant": f"tfidf_idf_{label}",
                    "target": t,
                    "subset": "half",
                }
                res.add({**row, **measure(trep, v, ks)})
                del v, x
        del trep


def stage_concentration(ctx: Context, out: Path, res: Results) -> None:
    from sklearn.cluster import KMeans

    args = ctx.args
    n = ctx.n
    ks_for(n)
    dino, dino_names = load_layers(feature_path(args, DINO))
    clip, clip_names = load_layers(feature_path(args, CLIP))
    reps = {}
    for c in ("gemini", "orig"):
        for name in ("tf", "tfidf"):
            reps[(c, name)] = variant_features(ctx, c, name)
    py, py_names = load_layers(feature_path(args, f"EleutherAI__pythia-410m@{FINAL}__gemini"))
    reps[("gemini", f"pythia-410m:{py_names[0]}")] = py[0]
    reps[("gemini", f"pythia-410m:{py_names[-1]}")] = py[-1]
    del py
    ll, ll_names = load_layers(feature_path(args, "huggyllama__llama-13b__orig"))
    reps[("orig", f"llama-13b:{ll_names[0]}")] = ll[0]
    reps[("orig", f"llama-13b:{ll_names[-1]}")] = ll[-1]
    del ll
    report = {}

    # Near-duplicates: same image features or the same caption
    dup = M16.near_duplicates(dino[-1]) | M16.near_duplicates(clip[-1])
    report["near_duplicate_images"] = int(dup.sum())
    for c, texts in ctx.raw.items():
        d = pd.Series(texts).duplicated(keep=False).to_numpy()
        report[f"duplicate_captions_{c}"] = int(d.sum())
        dup = dup | d
    report["removed_total"] = int(dup.sum())
    keep = torch.from_numpy(np.flatnonzero(~dup))
    ks = ks_for(len(keep))
    for t, layers, names in ((DINO, dino, dino_names), (CLIP, clip, clip_names)):
        trep = sub_rep(layers, keep, names, max(ks), True)
        for k, x in reps.items():
            if res.has(stage="dedup", caption=k[0], variant=k[1], target=t):
                continue
            v = Rep([x[keep]], [k[1]], k_max=max(ks))
            row = {"stage": "dedup", "caption": k[0], "variant": k[1], "target": t}
            res.add({**row, **measure(trep, v, ks)})
            del v
        del trep

    # Per-sample overlap distribution at the best DINOv2 layer, k = 10
    trep = Rep(dino, dino_names, k_max=10, ranks=True, centred=False, hsic=False)
    conc = {}
    for k, x in reps.items():
        v = Rep([x], [k[1]], k_max=10, centred=False, hsic=False)
        o = ksw.overlaps(trep.sweep, v.sweep, [10])[0, :, 0]  # [layers, n]
        j = int(o.mean(1).argmax())
        s = o[j].sort(descending=True).values
        conc[f"{k[0]}|{k[1]}"] = {
            "layer": dino_names[j],
            "mean": float(s.mean()),
            "zero_fraction": float((s == 0).float().mean()),
            "top5pct_share": float(s[: n // 20].sum() / s.sum()),
            "top25pct_share": float(s[: n // 4].sum() / s.sum()),
            "quantiles": [
                float(q) for q in torch.quantile(s, torch.tensor([0.5, 0.75, 0.9, 0.99]))
            ],
        }
        del v
    report["overlap"] = conc
    del trep

    # Within image-type clusters: k-means on DINOv2's last layer
    labels = KMeans(n_clusters=args.clusters, n_init=3, random_state=0).fit_predict(
        dino[-1].numpy()
    )
    np.save(out / "clusters.npy", labels)
    cs = ctx.sets["gemini"]
    tfidf = lex.tfidf(cs.counts)
    report["clusters"] = []
    for cid in range(args.clusters):
        idx = np.flatnonzero(labels == cid)
        mean = np.asarray(tfidf[idx].mean(0)).ravel() - np.asarray(tfidf.mean(0)).ravel()
        top = np.argsort(mean)[::-1][:12]
        report["clusters"].append(
            {
                "cluster": cid,
                "size": len(idx),
                "top_tokens": [ctx.tok.decode([int(cs.vocab[i])]) for i in top],
            }
        )
    for cid in range(args.clusters):
        idx = np.flatnonzero(labels == cid)
        if len(idx) < args.min_cluster:
            continue
        tidx = torch.from_numpy(idx)
        ks = [k for k in (10, 30) if k < len(idx) // 2]
        for t, layers, names in ((DINO, dino, dino_names), (CLIP, clip, clip_names)):
            trep = sub_rep(layers, tidx, names, max(ks), True)
            for k, x in reps.items():
                label = f"c{cid}"
                if res.has(stage="cluster", caption=k[0], variant=k[1], target=t, subset=label):
                    continue
                v = Rep([x[tidx]], [k[1]], k_max=max(ks))
                row = {
                    "stage": "cluster",
                    "caption": k[0],
                    "variant": k[1],
                    "target": t,
                    "subset": label,
                }
                res.add({**row, **measure(trep, v, ks)})
                del v
            del trep
    (out / "concentration.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log(f"concentration: removed {report['removed_total']} near-duplicate rows")


def half_cka(rep_layers, idx) -> list[torch.Tensor]:
    return [lex.centred_unit_gram(x[idx] @ x[idx].T).float().flatten() for x in rep_layers]


def stage_split(ctx: Context, out: Path) -> None:
    """Cell chosen on half the samples, scored on the other half, both ways."""
    args = ctx.args
    n = ctx.n
    perm = np.random.default_rng(0).permutation(n)
    halves = [torch.from_numpy(perm[: n // 2]), torch.from_numpy(perm[n // 2 :])]
    rows = []
    sets = {
        f"EleutherAI__{s}@{FINAL}__gemini": feature_path(args, f"EleutherAI__{s}@{FINAL}__gemini")
        for s in PY_SERIES
    }
    sets["gemini|tfidf"] = None
    sets["orig|tfidf"] = None
    sets["huggyllama__llama-13b__orig"] = feature_path(args, "huggyllama__llama-13b__orig")
    for t in (DINO, CLIP):
        tl, tn = load_layers(feature_path(args, t))
        tsweep = ksw.build_sweep_cache(tl, 10)
        tg = [half_cka(tl, h) for h in halves]
        for key, path in sets.items():
            if path is None:
                c, name = key.split("|")
                layers, names = [variant_features(ctx, c, name)], [name]
            else:
                layers, names = load_layers(path)
            lsweep = ksw.SweepCache(
                ranks=None, idx=torch.stack([ksw.neighbours(x, 10) for x in layers])
            )
            o = ksw.overlaps(tsweep, lsweep, [10])[0].permute(1, 0, 2)  # [La, Lt, n]
            kn_a, kn_b = o[:, :, halves[0]].mean(-1), o[:, :, halves[1]].mean(-1)
            mu = 10 / (n - 1)
            lg = [half_cka(layers, h) for h in halves]
            ck = [
                torch.stack([torch.stack([torch.dot(a, b) for b in tg[h]]) for a in lg[h]])
                for h in (0, 1)
            ]
            row = {"set": key, "target": t, "names_a": names, "names_b": tn}

            def crossed(ga, gb, mask=None):
                """Mean of (select on A, score on B) and (select on B, score on A)."""
                out_ = []
                for sel, ev in ((ga, gb), (gb, ga)):
                    s = sel.clone()
                    if mask is not None:
                        s[~mask] = -1e9
                    i = int(s.argmax())
                    out_.append((float(ev.flatten()[i]), float(sel.flatten()[i])))
                return {
                    "held_out": np.mean([x[0] for x in out_]),
                    "in_sample": np.mean([x[1] for x in out_]),
                }

            deep = torch.ones_like(kn_a, dtype=torch.bool)
            deep[0] = len(names) == 1
            l0 = torch.zeros_like(kn_a, dtype=torch.bool)
            l0[0] = True
            row["knn10_excess_all"] = {k: v - mu for k, v in crossed(kn_a, kn_b).items()}
            row["knn10_excess_layer0"] = {k: v - mu for k, v in crossed(kn_a, kn_b, l0).items()}
            row["knn10_excess_deep"] = {k: v - mu for k, v in crossed(kn_a, kn_b, deep).items()}
            row["cka_all"] = crossed(ck[0], ck[1])
            row["cka_layer0"] = crossed(ck[0], ck[1], l0)
            row["cka_deep"] = crossed(ck[0], ck[1], deep)
            rows.append(row)
            (out / "split.json").write_text(json.dumps(rows, indent=1, default=float))
            log(f"split {key[:40]} x {t[:12]}: knn10 deep {row['knn10_excess_deep']}")
            del lg, lsweep
        del tl, tsweep, tg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, required=True)
    p.add_argument(
        "--stages",
        default="captions,identity,numerics,grids,lexical,trained,targets,subsets,concentration,split",
    )
    p.add_argument(
        "--identity-sets",
        default=";".join(
            f"EleutherAI/{s},{r},EleutherAI__{s}@{r}__gemini"
            for s in ("pythia-160m", "pythia-410m")
            for r in ("step0", FINAL)
        )
        + ";EleutherAI/pythia-1.4b,,"
        + REF,
    )
    p.add_argument(
        "--gaussian-widths", type=lambda s: [int(x) for x in s.split(",")], default=[768, 1024]
    )
    p.add_argument("--perm-check", type=int, default=30)
    p.add_argument("--clusters", type=int, default=20)
    p.add_argument("--min-cluster", type=int, default=100)
    p.add_argument("--llm-tokenizers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("/root/hf-scratch"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = Context(args)
    res = Results(args.out / "rows.jsonl")
    stages = {
        "captions": lambda: stage_captions(ctx, args.out),
        "identity": lambda: stage_identity(ctx, args.out),
        "numerics": lambda: stage_numerics(ctx, args.out),
        "grids": lambda: stage_grids(ctx, args.out),
        "lexical": lambda: stage_lexical(ctx, args.out, res),
        "trained": lambda: stage_trained(ctx, args.out, res),
        "targets": lambda: stage_targets(ctx, args.out, res),
        "subsets": lambda: stage_subsets(ctx, args.out, res),
        "concentration": lambda: stage_concentration(ctx, args.out, res),
        "split": lambda: stage_split(ctx, args.out),
    }
    failed = []
    for name in args.stages.split(","):
        log(f"=== stage {name}")
        t0 = time.time()
        try:
            stages[name]()
            log(f"=== stage {name} done in {time.time() - t0:.0f}s")
        except Exception:  # a failed check must not cost the others; the exit code reports it
            traceback.print_exc()
            log(f"STAGE FAILED: {name}")
            failed.append(name)
    if failed:
        raise SystemExit(f"failed stages: {failed}")


if __name__ == "__main__":
    main()
