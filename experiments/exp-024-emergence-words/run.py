"""exp-024: what structure emerges during training, and which kinds of words carry alignment.

No forward passes: exp-016 / exp-018 / exp-022 features and D2's captions (10240 WIT images).

Stage `clusters`: image types are unsupervised. k-means (K = 5, 20, 100) on the last layer of DINOv2
and of CLIP separately. A target ViT is always scored with the types defined by the other ViT, so a
type is not defined by the representation it is compared with.

Stage `words`: spaCy tags every caption. TF-IDF (scikit-learn) is built from one word class only, or
from all words except that class, and compared with the ViTs and with LM layers. Each class has
frequency-matched random word sets as a control: a class matters only if it beats words of the same
frequency.

Stage `typed`: for every checkpoint, LM layer, K and target:
- between_share: the share of the representation's variance between types (does clustering arise?);
- purity: how many of a sample's 10 nearest neighbours share its type, against the base rate;
- within_knn: mutual kNN when neighbours may only come from the sample's own type, against chance;
- knn_same / knn_other: global mutual-kNN overlap split into same-type and other-type neighbours;
- cka_between / cka_within: linear CKA of the between-type and within-type parts of the kernel.
With labels S, X = X_b + X_w with x_b the mean of its type, so K = K_b + K_w + cross terms. TF-IDF
and final larger LMs give the reference levels.

    uv run python experiments/exp-024-emergence-words/run.py --out <dir> --exp016 <dir> --exp018 <dir> \\
        --pythia-orig <dir> --stages clusters,words,typed
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import multiprocessing
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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


P1 = _module("exp022_run", ROOT / "exp-022-lexical-validation" / "run.py")
from geoprh import lexical as lex  # noqa: E402

DINO, CLIP = P1.DINO, P1.CLIP
OTHER = {DINO: ("clip", CLIP), CLIP: ("dino", DINO)}
SHORT = {DINO: "dino", CLIP: "clip"}
TYPE_KS = (5, 20, 100)
K_NN = 10
PY_SERIES = ["pythia-160m", "pythia-410m", "pythia-160m-seed1", "pythia-410m-seed1"]
REVS = [0, 16, 64, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000]
ORIG_REVS = [0, 1000, 4000, 16000, 64000, 143000]
LLMS = P1.LLMS
LM_LAYERS = (0, 3, 5, 6)  # of the 7 stored layers: embedding, middle, late, last

POS_CLASSES = {
    "noun": {"NOUN"},
    "proper": {"PROPN"},
    "adj": {"ADJ"},
    "verb": {"VERB"},
    "adv": {"ADV"},
    "number": {"NUM"},
    "function": {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART"},
}
# Fixed before looking at any results. Single lower-case tokens.
LEXICONS = {
    "medium": set(
        [
            "photograph",
            "photo",
            "photography",
            "image",
            "picture",
            "map",
            "diagram",
            "chart",
            "graph",
            "painting",
            "portrait",
            "illustration",
            "drawing",
            "sketch",
            "engraving",
            "etching",
            "lithograph",
            "logo",
            "emblem",
            "flag",
            "poster",
            "scan",
            "screenshot",
            "sculpture",
            "statue",
            "cartoon",
            "infographic",
            "monochrome",
            "grayscale",
            "sepia",
            "watercolor",
            "painted",
            "printed",
            "document",
            "page",
            "manuscript",
        ]
    ),
    "color": set(
        [
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "purple",
            "violet",
            "pink",
            "brown",
            "black",
            "white",
            "gray",
            "grey",
            "beige",
            "tan",
            "gold",
            "golden",
            "silver",
            "cream",
            "teal",
            "turquoise",
            "maroon",
            "navy",
            "crimson",
        ]
    ),
    "spatial": set(
        [
            "left",
            "right",
            "top",
            "bottom",
            "center",
            "centre",
            "middle",
            "background",
            "foreground",
            "above",
            "below",
            "behind",
            "front",
            "side",
            "corner",
            "edge",
            "beneath",
            "upper",
            "lower",
            "near",
            "far",
            "distance",
        ]
    ),
    "lighting": set(
        [
            "light",
            "lighting",
            "shadow",
            "shadows",
            "bright",
            "dark",
            "sunlight",
            "overcast",
            "soft",
            "diffuse",
            "illuminated",
            "contrast",
            "sunny",
            "cloudy",
            "mood",
            "atmosphere",
        ]
    ),
}
CLASSES = [*POS_CLASSES, *LEXICONS]
CONTROL_SEEDS = (0, 1)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Context:
    """Captions and targets, shared by the stages."""

    def __init__(self, args):
        self.args = args
        self.p1 = P1.Context(argparse.Namespace(**{**vars(args), "llm_tokenizers": 0}))
        self.n = self.p1.n


# ---------------------------------------------------------------- clusters


def stage_clusters(ctx: Context, out: Path) -> None:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    args = ctx.args
    vec = TfidfVectorizer(stop_words="english", dtype=np.float32, min_df=5)
    tfidf = vec.fit_transform(ctx.p1.sets["gemini"].texts)
    words = np.array(vec.get_feature_names_out())
    overall = np.asarray(tfidf.mean(0)).ravel()
    labels, report = {}, {}
    for short, key in (("dino", DINO), ("clip", CLIP)):
        layers, _ = P1.load_layers(P1.feature_path(args, key))
        x = layers[-1].numpy()
        del layers
        for k in TYPE_KS:
            k_use = min(k, ctx.n // 5)
            lab = KMeans(n_clusters=k_use, n_init=3, random_state=0).fit_predict(x)
            labels[f"{short}_{k}"] = lab
            rows = []
            for c in range(k_use):
                idx = np.flatnonzero(lab == c)
                diff = np.asarray(tfidf[idx].mean(0)).ravel() - overall
                rows.append(
                    {
                        "cluster": c,
                        "size": len(idx),
                        "words": words[np.argsort(diff)[::-1][:10]].tolist(),
                        "examples": [ctx.p1.sets["gemini"].texts[i][:110] for i in idx[:2]],
                    }
                )
            report[f"{short}_{k}"] = sorted(rows, key=lambda r: -r["size"])
            log(f"clusters {short} K = {k}: largest {max(r['size'] for r in rows)}")
    np.savez(out / "clusters.npz", **labels)
    (out / "clusters.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    from sklearn.metrics import adjusted_rand_score

    agree = {k: adjusted_rand_score(labels[f"dino_{k}"], labels[f"clip_{k}"]) for k in TYPE_KS}
    (out / "clusters_agreement.json").write_text(json.dumps(agree))
    log(f"DINOv2 vs CLIP type agreement (adjusted Rand): {agree}")


# ---------------------------------------------------------------- image types


def decompose(k: torch.Tensor, labels: torch.Tensor, sizes: torch.Tensor):
    """Centred kernel -> (between, within) parts, float64, with x_b the mean of x's type."""
    n, c = k.shape[0], sizes.shape[0]
    row = torch.zeros(n, c, dtype=torch.float64).index_add_(1, labels, k)  # sum over j in type d
    block = torch.zeros(c, c, dtype=torch.float64).index_add_(0, labels, row)
    block = block / (sizes[None] * sizes[:, None])
    k_b = block[labels][:, labels]
    m = row[:, labels] / sizes[labels]  # M[i, j] = mean of K[i, j'] over j' in j's type
    k_w = k - m - m.T + k_b
    return k_b, k_w


def neighbours(k: torch.Tensor, labels=None) -> torch.Tensor:
    """Top-10 of a cosine Gram, self excluded; with labels, only within the sample's own type."""
    s = k.clone()
    s.fill_diagonal_(float("-inf"))
    if labels is not None:
        s = s.masked_fill(labels[None] != labels[:, None], float("-inf"))
    return s.topk(K_NN, dim=1).indices.to(torch.int32)


def overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, :, None] == b[:, None, :]).any(2).float().sum(1) / K_NN


def unit(k: torch.Tensor) -> torch.Tensor:
    return (k / k.norm()).float().flatten()


def label_set(raw: np.lib.npyio.NpzFile, name: str):
    lab = torch.from_numpy(raw[name].astype(np.int64))
    sizes = torch.bincount(lab).double()
    return lab, sizes


def prepare_targets(args, ctx, out: Path, raw) -> Path:
    """Target kernels' parts to disk: workers map them instead of each holding a copy."""
    cache = args.scratch / "targets"
    cache.mkdir(parents=True, exist_ok=True)
    for key in (DINO, CLIP):
        source, _ = OTHER[key]
        layers, _ = P1.load_layers(P1.feature_path(args, key))
        x = layers[-1]
        del layers
        k_raw = x @ x.T
        k_c = k_raw.double()
        k_c = k_c - k_c.mean(0, keepdim=True)
        k_c = k_c - k_c.mean(1, keepdim=True)
        idx_all = neighbours(k_raw)
        torch.save({"g": unit(k_c), "idx": idx_all}, cache / f"{SHORT[key]}_all.pt")
        for k in TYPE_KS:
            lab, sizes = label_set(raw, f"{source}_{k}")
            k_b, k_w = decompose(k_c, lab, sizes)
            torch.save(
                {"g_b": unit(k_b), "g_w": unit(k_w), "idx_w": neighbours(k_raw, lab)},
                cache / f"{SHORT[key]}_{k}.pt",
            )
            del k_b, k_w
        del k_c, k_raw, x
        gc.collect()
        log(f"typed: target {SHORT[key]} ready")
    return cache


def score_layer(x: torch.Tensor, cache: Path, raw_path: Path, perm=None) -> list[dict]:
    """One language layer against both targets at every K."""
    raw = np.load(raw_path)
    n = x.shape[0]
    if perm is not None:  # break the pairing: the sample in row i is now another image's caption
        x = x[perm]
    k_raw = x @ x.T
    k_c = k_raw.double()
    k_c = k_c - k_c.mean(0, keepdim=True)
    k_c = k_c - k_c.mean(1, keepdim=True)
    tr = k_c.trace()
    idx_all = neighbours(k_raw)
    g_all = unit(k_c)
    rows = []
    for key in (DINO, CLIP):
        source, _ = OTHER[key]
        t_all = torch.load(cache / f"{SHORT[key]}_all.pt", mmap=True, weights_only=False)
        for k in TYPE_KS:
            lab, sizes = label_set(raw, f"{source}_{k}")
            t = torch.load(cache / f"{SHORT[key]}_{k}.pt", mmap=True, weights_only=False)
            k_b, k_w = decompose(k_c, lab, sizes)
            size_i = sizes[lab]
            valid = size_i > K_NN
            base = float(((size_i - 1) / (n - 1)).mean())
            same = lab[idx_all.long()] == lab[:, None]
            o = (idx_all[:, :, None] == t_all["idx"][:, None, :]).any(
                2
            )  # [n, 10] shared neighbours
            idx_w = neighbours(k_raw, lab)
            rows.append(
                {
                    "target": SHORT[key],
                    "label_source": source,
                    "k_types": k,
                    "between_share": float(k_b.trace() / tr),
                    "purity": float(same.float().mean()),
                    "purity_base": base,
                    "knn": float(o.float().mean()),
                    "knn_chance": K_NN / (n - 1),
                    "knn_same": float((o & same).float().sum(1).mean() / K_NN),
                    "knn_other": float((o & ~same).float().sum(1).mean() / K_NN),
                    "within_knn": float(overlap(idx_w, t["idx_w"])[valid].mean()),
                    "within_chance": float((K_NN / (size_i[valid] - 1)).mean()),
                    "cka": float(torch.dot(g_all, t_all["g"])),
                    "cka_between": float(torch.dot(unit(k_b), t["g_b"])),
                    "cka_within": float(torch.dot(unit(k_w), t["g_w"])),
                }
            )
            del k_b, k_w
    return rows


def set_task(args_tuple) -> list[dict]:
    meta, path, tfidf_path, keep, cache, raw_path, threads = args_tuple
    torch.set_num_threads(threads)
    if tfidf_path is not None:
        layers, names = [torch.load(tfidf_path, weights_only=False)], ["tfidf"]
    else:
        layers, names = P1.load_layers(Path(path))
    rows = []
    for i in keep if tfidf_path is None else [0]:
        for r in score_layer(layers[i].float(), Path(cache), Path(raw_path)):
            rows.append({**meta, "layer_index": i, "layer": names[i], **r})
    return rows


def null_task(args_tuple) -> list[dict]:
    meta, path, tfidf_path, keep, cache, raw_path, threads, permutations = args_tuple
    torch.set_num_threads(threads)
    if tfidf_path is not None:
        layers, names = [torch.load(tfidf_path, weights_only=False)], ["tfidf"]
    else:
        layers, names = P1.load_layers(Path(path))
    rows = []
    for i in keep if tfidf_path is None else [0]:
        x = layers[i].float()
        gen = torch.Generator().manual_seed(0)
        for p in range(permutations):
            perm = torch.randperm(x.shape[0], generator=gen)
            for r in score_layer(x, Path(cache), Path(raw_path), perm):
                rows.append({**meta, "layer_index": i, "layer": names[i], "perm": p, **r})
    return rows


def typed_sets(args, ctx) -> tuple[list, dict]:
    sets = []
    for s in PY_SERIES:
        for rev in REVS:
            key = f"EleutherAI__{s}@step{rev}__gemini"
            sets.append(
                (
                    {"set": key, "series": s, "step": rev, "caption": "gemini"},
                    args.exp018 / "features" / f"{key}__d2l.pt",
                )
            )
        for rev in ORIG_REVS:
            key = f"EleutherAI__{s}@step{rev}__orig"
            sets.append(
                (
                    {"set": key, "series": s, "step": rev, "caption": "orig"},
                    args.pythia_orig / "features" / f"{key}__d2l.pt",
                )
            )
    for m in LLMS:
        for c in ("gemini", "orig"):
            key = f"{m.replace('/', '__')}__{c}"
            sets.append(
                (
                    {"set": key, "series": m.split("/")[1], "step": -1, "caption": c},
                    args.exp016 / "features" / f"{key}__d2l.pt",
                )
            )
    tfidf = {}
    for c in ("gemini", "orig"):
        path = args.scratch / f"tfidf_{c}.pt"
        torch.save(P1.variant_features(ctx.p1, c, "tfidf"), path)
        tfidf[c] = path
    return [s for s in sets if s[1].exists() or log(f"typed: {s[1].name} missing, SKIPPED")], tfidf


def stage_typed(ctx: Context, out: Path) -> None:
    args = ctx.args
    raw_path = out / "clusters.npz"
    raw = np.load(raw_path)
    cache = prepare_targets(args, ctx, out, raw)
    sets, tfidf = typed_sets(args, ctx)
    path = out / "typed.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(x)["set"] for x in path.read_text().splitlines()}
    threads = max(1, args.threads // args.workers)
    tasks = [
        (m, str(p), None, LM_LAYERS, str(cache), str(raw_path), threads)
        for m, p in sets
        if m["set"] not in done
    ]
    tasks += [
        (
            {"set": f"tfidf__{c}", "series": "tfidf", "step": -1, "caption": c},
            None,
            str(p),
            (0,),
            str(cache),
            str(raw_path),
            threads,
        )
        for c, p in tfidf.items()
        if f"tfidf__{c}" not in done
    ]
    log(f"typed: {len(tasks)} feature sets")
    t0 = time.time()
    if args.workers == 1:
        results = map(set_task, tasks)
    else:
        # Spawn: this process has used torch's thread pool, and forked workers deadlock on its locks
        pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
        results = (f.result() for f in as_completed(pool.submit(set_task, t) for t in tasks))
    for i, rows in enumerate(results):
        with path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
        log(f"typed [{i + 1}/{len(tasks)}] {rows[0]['set']}: {time.time() - t0:.0f}s")
    if args.workers != 1:
        pool.shutdown()
    # Nulls: the pairing broken, at both ends of training and for word kernels
    picks = [m for m in sets if m[0]["series"] == "pythia-410m" and m[0]["step"] in (0, 143000)]
    tasks = [
        (m, str(p), None, (0, 6), str(cache), str(raw_path), args.threads, args.null_permutations)
        for m, p in picks
    ]
    tasks += [
        (
            {"set": f"tfidf__{c}", "series": "tfidf", "step": -1, "caption": c},
            None,
            str(p),
            (0,),
            str(cache),
            str(raw_path),
            args.threads,
            args.null_permutations,
        )
        for c, p in tfidf.items()
    ]
    with (out / "typed_null.jsonl").open("w") as fh:
        for t in tasks:
            for r in null_task(t):
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            log(f"typed null {t[0]['set']}")


# ---------------------------------------------------------------- word classes


def tag(texts: list[str], processes: int) -> list[list[tuple[str, str]]]:
    import spacy

    # attribute_ruler stays on: it maps the tagger's labels to pos_
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.max_length = max(nlp.max_length, max(len(t) for t in texts) + 1)
    return [
        [(t.lower_, t.pos_) for t in doc if t.is_alpha]
        for doc in nlp.pipe(texts, batch_size=16, n_process=processes)
    ]


def tagged(ctx: Context, out: Path) -> dict[str, list[list[tuple[str, str]]]]:
    path = ctx.args.scratch / "tagged.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    res = {}
    for c in ("gemini", "orig"):
        t0 = time.time()
        res[c] = tag(ctx.p1.raw[c] if c == "orig" else ctx.p1.clean_gemini, ctx.args.tag_processes)
        log(f"tagged {c}: {sum(len(x) for x in res[c])} words, {time.time() - t0:.0f}s")
    path.write_bytes(pickle.dumps(res))
    return res


def in_class(word: str, pos: str, cls: str) -> bool:
    if cls in POS_CLASSES:
        return pos in POS_CLASSES[cls]
    return word in LEXICONS[cls]


def tfidf_rows(docs: list[list[str]]) -> torch.Tensor:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer=lambda d: d, norm="l2", dtype=np.float32)
    return lex.to_unit_rows(vec.fit_transform(docs))


def matched_words(counts: dict[str, int], chosen: set[str], seed: int) -> set[str]:
    """For each chosen word, a random other word of about the same corpus frequency."""
    rng = np.random.default_rng(seed)
    others = sorted((w for w in counts if w not in chosen), key=lambda w: counts[w])
    if not others:
        return set()
    freq = np.array([counts[w] for w in others], dtype=float)
    out = set()
    for w in chosen:
        i = int(np.searchsorted(freq, counts[w]))
        lo, hi = max(0, i - 25), min(len(others), i + 25)
        out.add(others[int(rng.integers(lo, hi))])
    return out


def stage_words(ctx: Context, out: Path) -> None:
    """Two passes so 16 GB is enough: ViTs with every variant, then LMs with all / only variants."""
    args = ctx.args
    data = tagged(ctx, out)
    for s in ctx.p1.sets.values():  # token lists are only needed for tagging's captions
        s.ids = None
    gc.collect()
    ks = [k for k in (10, 100, 1000) if k < ctx.n // 2]
    k_max = max(ks)
    path = out / "words.jsonl"
    stats = {}
    for caption, tags in data.items():
        counts: dict[str, int] = {}
        for doc in tags:
            for w, _ in doc:
                counts[w] = counts.get(w, 0) + 1
        total = sum(counts.values())
        stats[caption] = {"words": total, "vocab": len(counts)}
        for cls in CLASSES:
            stats[caption][cls] = {
                "token_share": sum(1 for d in tags for w, p in d if in_class(w, p, cls)) / total,
                "vocab": len({w for d in tags for w, p in d if in_class(w, p, cls)}),
            }
    (out / "words_stats.json").write_text(json.dumps(stats, indent=1))

    def score(caption, name, docs, mode, cls, seed, targets) -> None:
        if not any(docs):
            log(f"words {caption} {name}: no words, SKIPPED")
            return
        empty = sum(1 for d in docs if not d)  # stay zero rows: no similarity to anything
        x = tfidf_rows(docs)
        v = P1.Rep([x], [name], k_max=k_max, hsic=False)
        del x
        for t, rep in targets.items():
            row = {
                "caption": caption,
                "variant": name,
                "mode": mode,
                "class": cls,
                "seed": seed,
                "target": t,
                "empty_rows": empty,
            }
            with path.open("a") as fh:
                fh.write(
                    json.dumps({**row, **P1.measure(rep, v, ks, debiased=False)}, default=float)
                    + "\n"
                )
        del v
        gc.collect()

    # Pass 1: the ViTs, with controls. The last 3 stored layers hold every best cell seen so far.
    vits = {}
    for t in (DINO, CLIP):
        layers, names = P1.load_layers(P1.feature_path(args, t))
        vits[SHORT[t]] = P1.Rep(layers[-3:], names[-3:], k_max=k_max, ranks=True, hsic=False)
        del layers
    log("words: ViTs loaded")
    for caption, tags in data.items():
        counts = {}
        for doc in tags:
            for w, _ in doc:
                counts[w] = counts.get(w, 0) + 1
        score(caption, "all", [[w for w, _ in d] for d in tags], "all", "all", None, vits)
        for cls in CLASSES:
            only = [[w for w, p in d if in_class(w, p, cls)] for d in tags]
            score(caption, f"only_{cls}", only, "only", cls, None, vits)
            score(
                caption,
                f"without_{cls}",
                [[w for w, p in d if not in_class(w, p, cls)] for d in tags],
                "without",
                cls,
                None,
                vits,
            )
            chosen = {w for d in only for w in d}
            del only
            if not chosen:
                continue
            for seed in CONTROL_SEEDS:
                ctrl = matched_words(counts, chosen, seed)
                if ctrl:
                    score(
                        caption,
                        f"control_{cls}_{seed}",
                        [[w for w, _ in d if w in ctrl] for d in tags],
                        "control",
                        cls,
                        seed,
                        vits,
                    )
            log(f"words {caption} {cls}: done")
    del vits
    gc.collect()

    # Pass 2: what the LMs themselves track, at the embedding layer and the last layer
    for caption, tags in data.items():
        lms = {}
        for name, p in (
            (
                "pythia-410m",
                (args.exp018 / "features" if caption == "gemini" else args.pythia_orig / "features")
                / f"EleutherAI__pythia-410m@step143000__{caption}__d2l.pt",
            ),
            ("llama-13b", args.exp016 / "features" / f"huggyllama__llama-13b__{caption}__d2l.pt"),
        ):
            if p.exists():
                layers, names = P1.load_layers(p)
                lms[name] = P1.Rep(
                    [layers[0], layers[-1]],
                    [names[0], names[-1]],
                    k_max=k_max,
                    ranks=True,
                    hsic=False,
                )
                del layers
        score(caption, "all", [[w for w, _ in d] for d in tags], "all", "all", None, lms)
        for cls in CLASSES:
            score(
                caption,
                f"only_{cls}",
                [[w for w, p in d if in_class(w, p, cls)] for d in tags],
                "only",
                cls,
                None,
                lms,
            )
        log(f"words {caption}: LMs done")
        del lms
        gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, required=True)
    p.add_argument("--pythia-orig", type=Path, required=True)
    p.add_argument("--scratch", type=Path, default=Path("/root/exp024"))
    p.add_argument("--stages", default="clusters,words,typed")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tag-processes", type=int, default=6)
    p.add_argument("--null-permutations", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("/root/hf-scratch"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    ctx = Context(args)
    stages = {"clusters": stage_clusters, "words": stage_words, "typed": stage_typed}
    failed = []
    for name in args.stages.split(","):
        log(f"=== stage {name}")
        t0 = time.time()
        try:
            stages[name](ctx, args.out)
            log(f"=== stage {name} done in {time.time() - t0:.0f}s")
        except Exception:  # a failed stage must not cost the other; the exit code reports it
            traceback.print_exc()
            log(f"STAGE FAILED: {name}")
            failed.append(name)
    if failed:
        raise SystemExit(f"failed stages: {failed}")


if __name__ == "__main__":
    main()
