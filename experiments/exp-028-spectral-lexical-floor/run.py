"""exp-028: the spectral lexical floor on D2 (n = 10240), original and Gemini captions.

Stages
- `anchors`: stream `askoepke/wit_1m_recaptioned` (text only) past D2's 10240 rows and keep the next
  N_ANCHORS rows with a non-empty original and Gemini caption. URLs must be disjoint from D2's.
- `kernels`: no language model anywhere. Per caption set:
  - F  floors as exp-022: token TF-IDF and sublinear TF-IDF (Pythia tokenizer, no special tokens),
       word TF-IDF without English stopwords;
  - T  regularised eigenmap of the evaluation captions' own kNN graph (transductive);
  - N  eigenmap of the anchors' graph, Nyström-extended to the evaluation captions (inductive),
       with the first 10000 and all 20000 anchors;
  - W  PPMI-SVD word vectors fitted on the anchors, caption = TF-IDF-weighted sum (inductive).
  Graph: word TF-IDF (no stopwords) cosine, top K_GRAPH neighbours per caption, symmetrised.
  Spectral kernels at k_s in KS_DIM, unweighted (psi) and eigenvalue-weighted (psi * mu).
- `score`: every kernel and the 4 exp-016 LLMs (7 stored layers) against DINOv2-g and CLIP-H CLS
  (7 layers), exp-022's `Rep` / `measure`: mutual kNN and linear CKA with exact null means. On all
  10240 images and on two fixed halves (k halved there, so k / n is unchanged), for split-half
  selection. The word floor must first reproduce exp-022's stored row, or the job stops.

    uv run python experiments/exp-028-spectral-lexical-floor/run.py --out <dir> --exp016 <dir> \\
        --stored <exp022>/rows.jsonl --stages anchors,kernels,score
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SP = _module("exp028_spectral", HERE / "spectral.py")
N_D2 = 10240
N_ANCHORS = 20000
ANCHOR_SIZES = (10000, 20000)
K_GRAPH = 50
KS_DIM = (8, 16, 32, 64, 128, 256)  # 512 dropped: 513-vector eigsh took 10-30 min per graph
W_VOCAB = 20000
KS_FULL = (10, 30, 100, 300, 1000)
PY_TOKENIZER = "EleutherAI/pythia-160m"
MAX_TOKENS = 2048
CAPTIONS = {"orig": "caption", "gemini": "gemini_caption"}
LLMS = [
    "bigscience__bloomz-1b7",
    "bigscience__bloomz-7b1",
    "openlm-research__open_llama_7b",
    "huggyllama__llama-13b",
]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- anchors


def stage_anchors(args) -> None:
    import pandas as pd
    from datasets import load_dataset

    d2 = pd.read_parquet(args.exp016 / "d2.parquet", columns=["url", "caption", "gemini_caption"])
    stream = load_dataset("askoepke/wit_1m_recaptioned", "wit_1m", split="train", streaming=True)
    stream = stream.select_columns(["original_caption", "gemini_caption", "url"])
    seen, rows = 0, []
    for row in stream:
        orig = (row["original_caption"] or "").strip()
        if not orig:
            continue
        seen += 1
        if seen <= N_D2:  # D2 is the first 10240 rows with an original caption (prepare_d2.py)
            continue
        gem = (row["gemini_caption"] or "").strip()
        if gem:
            rows.append({"url": row["url"], "caption": orig, "gemini_caption": gem})
        if len(rows) == args.n_anchors:
            break
    df = pd.DataFrame(rows)
    shared = set(df["url"]) & set(d2["url"])
    assert not shared, f"{len(shared)} anchor URLs are in D2"
    report = {
        "n_anchors": len(df),
        "shared_urls": 0,
        "duplicate_orig_captions_with_d2": int(df["caption"].isin(set(d2["caption"])).sum()),
        "duplicate_gemini_captions_with_d2": int(
            df["gemini_caption"].isin(set(d2["gemini_caption"])).sum()
        ),
    }
    df.to_parquet(args.out / "anchors.parquet")
    (args.out / "anchors.json").write_text(json.dumps(report, indent=1))
    log(f"anchors: {report}")


# ---------------------------------------------------------------- kernels


def word_vectorizer(**kw):
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(lowercase=True, stop_words="english", norm="l2", dtype=np.float32, **kw)


def spectral_features(psi, mu, prefix: str, dims) -> dict[str, np.ndarray]:
    out = {}
    for k in dims:
        if k <= psi.shape[1]:
            out[f"{prefix}__k{k}__plain"] = psi[:, :k]
            out[f"{prefix}__k{k}__weighted"] = psi[:, :k] * mu[None, :k]
    return out


def stage_kernels(args) -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    from geoprh import lexical as lex

    d2 = pd.read_parquet(args.exp016 / "d2.parquet", columns=list(CAPTIONS.values()))
    d2 = d2.iloc[: args.limit or N_D2]
    anchors = pd.read_parquet(args.out / "anchors.parquet")
    tok = AutoTokenizer.from_pretrained(PY_TOKENIZER)
    meta_path = args.out / "kernels_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    args.scratch.mkdir(parents=True, exist_ok=True)
    for cap, col in CAPTIONS.items():
        if (args.scratch / f"kernels_{cap}.pt").exists() and cap in meta:
            log(f"kernels {cap}: already saved, skipped")
            continue
        t0 = time.time()
        ev = d2[col].tolist()
        feats: dict[str, np.ndarray | torch.Tensor] = {}
        info = {}
        # F: exp-022's floors
        ids = tok(ev, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)["input_ids"]
        counts = lex.count_matrix(ids)[0]
        # floors stay sparse (Gemini's dense TF-IDF would be GBs); rows l2-normalised, as to_unit_rows
        feats["F__tfidf"] = normalize(lex.tfidf(counts).tocsr())
        feats["F__tfidf_sublinear"] = normalize(lex.tfidf(counts, sublinear=True).tocsr())
        x_ev = word_vectorizer().fit_transform(ev)
        feats["F__word_nostop"] = x_ev.tocsr()
        # T: the evaluation captions' own graph
        w = SP.symmetrise(SP.knn_affinity(x_ev, K_GRAPH))
        psi, mu, tau = SP.spectral(w, max(args.dims))
        info["T"] = {**SP.graph_stats(w), "tau": tau, "mu": mu.round(5).tolist()}
        feats.update(spectral_features(psi, mu, "T", args.dims))
        log(f"{cap} T: {info['T']['components']} components, {info['T']['isolated']} isolated")
        for m in args.anchor_sizes:
            if m > len(anchors):
                continue
            fit = anchors[col].tolist()[:m]
            # N: anchors' graph, Nyström to the evaluation captions
            vec = word_vectorizer()
            x_a = vec.fit_transform(fit)
            w_a = SP.symmetrise(SP.knn_affinity(x_a, K_GRAPH))
            psi_a, mu_a, tau_a = SP.spectral(w_a, max(args.dims))
            a_new = SP.knn_affinity(x_a, K_GRAPH, y=vec.transform(ev))
            ext = SP.nystrom(a_new, psi_a, mu_a, tau_a)
            no_edge = int((np.asarray(a_new.sum(1)).ravel() == 0).sum())
            info[f"N{m}"] = {
                **SP.graph_stats(w_a),
                "tau": tau_a,
                "mu": mu_a.round(5).tolist(),
                "eval_without_anchor_edges": no_edge,
            }
            feats.update(spectral_features(ext, mu_a, f"N{m}", args.dims))
            # W: PPMI-SVD word vectors from the anchors
            vw = word_vectorizer(max_features=W_VOCAB, min_df=2)
            xw = vw.fit_transform(fit)
            vecs = SP.ppmi_vectors((xw > 0).astype(np.float32), max(args.dims))
            cap_vec = vw.transform(ev) @ vecs
            info[f"W{m}"] = {"vocab": len(vw.vocabulary_)}
            for k in args.dims:
                feats[f"W{m}__k{k}__plain"] = cap_vec[:, :k]
            log(
                f"{cap} N{m}/W{m}: {no_edge} eval captions without anchor edges, {time.time() - t0:.0f}s"
            )
            del x_a, w_a, psi_a, a_new, ext, xw, vecs, cap_vec
            gc.collect()
        out = {
            k: v.astype(np.float32)
            if sp.issparse(v)
            else torch.from_numpy(np.ascontiguousarray(v, np.float32))
            for k, v in feats.items()
        }
        tmp = args.scratch / f"kernels_{cap}.pt.tmp"
        torch.save(out, tmp)
        tmp.replace(args.scratch / f"kernels_{cap}.pt")
        meta[cap] = info
        meta_path.write_text(json.dumps(meta, indent=1))  # after each caption set: resumable
        log(f"kernels {cap}: {len(out)} features, {time.time() - t0:.0f}s")
        del feats, out
        gc.collect()


# ---------------------------------------------------------------- scoring


def parse(name: str) -> dict:
    parts = name.split("__")
    fam = parts[0]
    d = {"family": fam[0] if fam[0] in "FTNW" else fam, "fit": fam, "k_s": None, "weight": None}
    if fam == "F":
        d["fit"] = parts[1]
    elif len(parts) == 3:
        d["k_s"] = int(parts[1].removeprefix("k"))
        d["weight"] = parts[2]
    return d


def check_stored(row: dict, stored_path: Path) -> dict:
    want = next(
        json.loads(x)
        for x in stored_path.read_text().splitlines()
        if '"lexical"' in x and '"word_tfidf_nostop"' in x and '"gemini"' in x and "dinov2" in x
    )
    got_c = np.array(row["cka"]) - np.array(row["cka_mu0"])
    want_c = np.array(want["cka"]) - np.array(want["cka_mu0"])
    got_k = np.array(row["knn"]["10"]) - row["knn_mu0"]["10"]
    want_k = np.array(want["knn"]["10"]) - want["knn_mu0"]["10"]
    report = {
        "max_abs_cka": float(np.abs(got_c - want_c).max()),
        "max_abs_knn10": float(np.abs(got_k - want_k).max()),
    }
    log(f"stored-row check: {report}")
    assert report["max_abs_cka"] < 3e-3 and report["max_abs_knn10"] < 3e-3, "differs from exp-022"
    return report


def stage_score(args) -> None:
    P1 = _module("exp022_run", ROOT / "exp-022-lexical-validation" / "run.py")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.limit or N_D2
    perm = np.random.default_rng(0).permutation(n)
    splits = {
        "full": (np.arange(n), list(KS_FULL)),
        "A": (np.sort(perm[: n // 2]), [k // 2 for k in KS_FULL]),
        "B": (np.sort(perm[n // 2 :]), [k // 2 for k in KS_FULL]),
    }
    if args.limit:
        splits = {s: (i, [k for k in ks if k < len(i) // 4]) for s, (i, ks) in splits.items()}
    (args.out / "splits.json").write_text(json.dumps({s: v[0].tolist() for s, v in splits.items()}))
    path = args.out / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["split"], r["target"], r["caption"], r["set"]))
    kernels = {
        c: torch.load(args.scratch / f"kernels_{c}.pt", weights_only=False) for c in CAPTIONS
    }
    vit = {"dino": P1.DINO, "clip": P1.CLIP}
    checked = bool(args.limit) or args.stored is None
    t_start = time.time()
    for split, (idx, ks) in splits.items():
        it = torch.as_tensor(idx)
        targets = {}
        for short, key in vit.items():
            layers, names = P1.load_layers(P1.feature_path(args, key))
            layers = [x[:n][it].to(dev) for x in layers]
            targets[short] = P1.Rep(layers, names, k_max=max(ks), ranks=True, hsic=False)
            del layers
        log(f"split {split}: targets ready (n = {len(idx)}, ks = {ks})")
        sets = []
        for cap in CAPTIONS:
            sets += [(cap, name, "kernel") for name in kernels[cap]]
            sets += [(cap, f"{m}__{cap}", "llm") for m in LLMS]
        sets.sort(key=lambda s: (s[1] != "F__word_nostop", s[0] != "gemini"))  # the check first
        for cap, name, kind in sets:
            todo = [t for t in targets if (split, t, cap, name) not in done]
            if not todo:
                continue
            if kind == "kernel":
                x = kernels[cap][name]
                x = torch.from_numpy(x[idx].toarray()) if sp.issparse(x) else x[it].float()
                layers = [F.normalize(x, dim=-1).to(dev)]  # zero rows stay zero
                del x
                names = [name]
            else:
                layers, names = P1.load_layers(args.exp016 / "features" / f"{name}__d2l.pt")
                layers = [x[:n][it].to(dev) for x in layers]
            rep = P1.Rep(layers, names, k_max=max(ks), ranks=False, hsic=False)
            del layers
            for t in todo:
                r = P1.measure(targets[t], rep, ks, debiased=False)
                row = {"split": split, "target": t, "caption": cap, "set": name, "kind": kind}
                row.update(parse(name) if kind == "kernel" else {"family": "LLM", "fit": name})
                row.update(r)
                if not checked and split == "full" and t == "dino" and name == "F__word_nostop":
                    rep_check = check_stored(row, args.stored)
                    (args.out / "checks.json").write_text(json.dumps(rep_check, indent=1))
                    checked = True
                with path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            del rep
            if dev == "cuda":
                torch.cuda.empty_cache()
        log(f"split {split} done, {time.time() - t_start:.0f}s")
        del targets
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    log("score done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, default=Path("/nonexistent"))
    p.add_argument("--stored", type=Path, default=None, help="exp-022 rows.jsonl")
    p.add_argument("--stages", default="anchors,kernels,score")
    p.add_argument("--scratch", type=Path, default=None, help="kernels; default <out>/kernels")
    p.add_argument("--n-anchors", type=int, default=N_ANCHORS)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first n D2 rows")
    p.add_argument("--dims", type=lambda s: [int(x) for x in s.split(",")], default=list(KS_DIM))
    p.add_argument(
        "--anchor-sizes", type=lambda s: [int(x) for x in s.split(",")], default=list(ANCHOR_SIZES)
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # kernels go to the volume: a pod deleted mid-run (2026-11-12) took ~2 h of container-disk work
    args.scratch = args.scratch or args.out / "kernels"
    for stage in args.stages.split(","):
        log(f"=== stage {stage}")
        {"anchors": stage_anchors, "kernels": stage_kernels, "score": stage_score}[stage](args)
        log(f"=== stage {stage} done")


if __name__ == "__main__":
    main()
