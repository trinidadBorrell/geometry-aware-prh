"""exp-026: mask word classes in captions, run LMs, score every hidden state against the ViTs.

Stage `mask` (CPU): spaCy tags the captions and `masks.py` builds each condition. It runs in its own
process, because spaCy forks workers and a fork after torch has used threads can deadlock.

Stage `score` (GPU): for each LM, caption set and condition, mean-pooled features of all hidden
states, PRH's clamp over all layers, then mutual kNN and linear CKA of every LM layer against the 7
stored layers of DINOv2 and CLIP. Excess uses the exact nulls (exp-022). Each condition also gets
its own TF-IDF floor, on the same masked tokens. Rows go straight to `--out`; features are not kept.

Before anything else counts, intact original captions must match exp-016's features, and the shuffled
Gemini condition must reproduce exp-022's stored scores at its 7 layers, or the job stops.

    uv run python experiments/exp-026-word-masking/run.py --out <dir> --exp016 <dir> --exp018 <dir> \\
        --stages mask
    uv run python experiments/exp-026-word-masking/run.py --out <dir> --exp016 <dir> --exp018 <dir> \\
        --stages score --grids <big_shuffle_grids.json>
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MK = _module("exp026_masks", HERE / "masks.py")
BIG_LMS = ["openlm-research/open_llama_7b", "bigscience/bloomz-7b1"]
BIG_CACHE = "/workspace/.cache/huggingface/hub"  # complete weights on the volume (exp-016)
KS = (10, 100, 1000)
Q = 0.95


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stage_mask(args) -> None:
    """Tags and masked captions. Imports nothing that starts torch's threads before the fork."""
    import pandas as pd
    import spacy

    df = pd.read_parquet(args.exp016 / "d2.parquet", columns=["caption", "gemini_caption"])
    df = df.iloc[: args.limit or len(df)]
    raw = {"orig": df["caption"].tolist(), "gemini": df["gemini_caption"].tolist()}
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.max_length = max(nlp.max_length, max(len(t) for v in raw.values() for t in v) + 1)
    res, stats, samples = {}, {}, {}
    for c, texts in raw.items():
        t0 = time.time()
        docs = [
            [(t.text_with_ws, t.is_alpha, t.pos_) for t in d]
            for d in nlp.pipe(texts, batch_size=16, n_process=args.tag_processes)
        ]
        log(f"tagged {c}: {time.time() - t0:.0f}s")
        res[c] = MK.build(docs, texts)
        stats[c] = {
            k: {
                "words_mean": float(np.mean([len(x.split()) for x in v])),
                "empty": sum(1 for x in v if not x.strip()),
            }
            for k, v in res[c].items()
        }
        samples[c] = {k: [x[:300] for x in v[:4]] for k, v in res[c].items()}
    args.scratch.mkdir(parents=True, exist_ok=True)
    (args.scratch / "masked.pkl").write_bytes(pickle.dumps(res))
    (args.out / "mask_stats.json").write_text(json.dumps(stats, indent=1))
    (args.out / "mask_samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    log(f"mask stats: {json.dumps(stats)}")


# ---------------------------------------------------------------- scoring


def clamp_value(feats: torch.Tensor, dev: str, chunk: int = 256) -> torch.Tensor:
    """`prepare_features`' threshold (q = 0.95, exact=False), a row chunk at a time for memory."""
    vals = [
        torch.quantile(feats[s : s + chunk].to(dev).float().abs().flatten(1), Q, dim=1).cpu()
        for s in range(0, feats.shape[0], chunk)
    ]
    return torch.cat(vals).mean()


class Target:
    """One ViT's stored layers: centred unit Grams and neighbour data, kept in the CPU's memory."""

    def __init__(self, short, layers, names, k_max, dev, ksw, lex):
        self.short, self.names = short, list(names)
        self.n = layers[0].shape[0]
        self.g, self.tr = [], []
        on_dev = [x.to(dev) for x in layers]
        for x in on_dev:
            c = lex.centred_unit_gram(x @ x.T)
            self.tr.append(float(c.trace()))
            self.g.append(c.float().cpu())
            del c
        cache = ksw.build_sweep_cache(on_dev, k_max)
        self.ranks, self.idx = cache.ranks.cpu(), cache.idx.cpu()


class Scorer:
    def __init__(self, targets, ks, dev, ksw, lex):
        self.targets, self.ks, self.dev, self.ksw, self.lex = targets, ks, dev, ksw, lex
        self.k_max = max(ks)

    @torch.no_grad()
    def score(self, layers) -> dict:
        """layers: iterable of [n, d] l2-normalised features. {target: arrays [L_lm, L_vit]}."""
        placed = {
            t.short: (
                [g.to(self.dev) for g in t.g],
                self.ksw.SweepCache(t.ranks.to(self.dev), t.idx.to(self.dev)),
            )
            for t in self.targets
        }
        out = {t.short: {"cka": [], "cka_mu0": [], "knn": []} for t in self.targets}
        for x in layers:
            x = x.to(self.dev)
            n = x.shape[0]
            c = self.lex.centred_unit_gram(x @ x.T)
            tr = float(c.trace())
            b = self.ksw.SweepCache(None, self.ksw.neighbours(x, self.k_max)[None])
            for t in self.targets:
                grams, cache = placed[t.short]
                out[t.short]["cka"].append([float((c * g.double()).sum()) for g in grams])
                out[t.short]["cka_mu0"].append([tr * q / (n - 1) for q in t.tr])
                # [ks, L_vit, 1]
                out[t.short]["knn"].append(self.ksw.scores(cache, b, list(self.ks))[:, :, 0].cpu())
            del c, b, x
        for r in out.values():
            r["knn"] = torch.stack(r["knn"], 1).numpy()  # [ks, L_lm, L_vit]
        del placed
        torch.cuda.empty_cache()
        return out


def rows_from(result, scorer, base: dict) -> list[dict]:
    rows = []
    for t in scorer.targets:
        r = result[t.short]
        rows.append(
            {
                **base,
                "target": t.short,
                "names_vit": t.names,
                "n": t.n,
                "cka": np.round(r["cka"], 6).tolist(),
                "cka_mu0": np.round(r["cka_mu0"], 6).tolist(),
                "knn": {str(k): np.round(r["knn"][i], 6).tolist() for i, k in enumerate(scorer.ks)},
                "knn_mu0": {str(k): k / (t.n - 1) for k in scorer.ks},
            }
        )
    return rows


def real_tokens(x: list[int], special: set[int]) -> bool:
    return any(t not in special for t in x)


def check_stored(result, grids, key, layer_idx, tol=0.003) -> dict:
    """Shuffled Gemini scores with the clamp over the 7 stored layers vs exp-022's stored grid."""
    g = next(x for x in grids if x["a"] == f"{key}__gemini-shuf0" and "dinov2" in x["b"])
    got_c, got_k = np.array(result["dino"]["cka"]), result["dino"]["knn"][0]
    want_c, want_k = np.array(g["cka"]), np.array(g["knn10"])
    report = {
        "layers": layer_idx,
        "max_abs_cka": float(np.abs(got_c - want_c).max()),
        "max_abs_knn10": float(np.abs(got_k - want_k).max()),
    }
    log(f"stored-score check {key}: {report}")
    assert report["max_abs_cka"] < tol and report["max_abs_knn10"] < tol, "differs from exp-022"
    return report


def stage_score(args) -> None:
    from geoprh import knn_sweep as ksw
    from geoprh import lexical as lex

    P2 = _module("exp022_phase2", ROOT / "exp-022-lexical-validation" / "phase2.py")
    P1, X20 = P2.P1, P2.X20
    dev = P2.device()
    masked = pickle.loads((args.scratch / "masked.pkl").read_bytes())
    n = len(masked["orig"]["intact"])
    ks = [k for k in KS if k < n // 2]
    targets = []
    for short, key in (("dino", P1.DINO), ("clip", P1.CLIP)):
        layers, names = P1.load_layers(P1.feature_path(args, key))
        targets.append(Target(short, [x[:n] for x in layers], names, max(ks), dev, ksw, lex))
        del layers
        log(f"target {short} ready")
    scorer = Scorer(targets, ks, dev, ksw, lex)
    path = args.out / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["lm"], r["caption"], r["condition"], r["kind"]))
    grids = json.loads(args.grids.read_text()) if args.grids and args.grids.exists() else []
    checks = {}
    t_start = time.time()

    def write(rows) -> None:
        with path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def floor(ids, keep, caption, cond, repo, base) -> None:
        if (repo, caption, cond, "floor") in done or MK.CONDITIONS[cond][1]:
            return  # a shuffle leaves the word kernel unchanged, so it shares its base's floor
        x = lex.to_unit_rows(lex.tfidf(lex.count_matrix([ids[i] for i in keep])[0]))
        full = torch.zeros(n, x.shape[1])
        full[torch.as_tensor(keep)] = x
        write(rows_from(scorer.score([full]), scorer, {**base, "kind": "floor"}))
        del x, full

    for repo in args.repos:
        key = repo.replace("/", "__")
        model, dec, _, tok = X20.load(repo, None, "causal", dev, args.big_cache)
        special = set(tok.all_special_ids)
        for caption in args.captions:
            for cond in MK.conditions_for(caption):
                if (repo, caption, cond, "lm") in done:
                    continue
                t0 = time.time()
                base_set, shuffle = MK.CONDITIONS[cond]
                ids = P2.tokenise(tok, masked[caption][base_set], True)
                if shuffle:
                    ids = P2.shuffled_ids(ids, special, 0)
                keep = [i for i, x in enumerate(ids) if real_tokens(x, special)]
                feats = X20.mean_pooled_ids(dec, tok, [ids[i] for i in keep], args.tokens)
                t1 = time.time()
                full = torch.zeros(n, *feats.shape[1:])
                full[torch.as_tensor(keep)] = feats
                del feats
                base = {
                    "lm": repo,
                    "caption": caption,
                    "condition": cond,
                    "shuffled": shuffle,
                    "tokens_mean": float(np.mean([len(x) for x in ids])),
                    "empty_rows": n - len(keep),
                    "n_lm_layers": full.shape[1],
                }
                if cond == "intact" and caption == "orig" and key not in checks:
                    cached = torch.load(
                        args.exp016 / "features" / f"{key}__orig__d2l.pt", weights_only=False
                    )
                    m = min(256, n)
                    theirs = cached["layers"][:m].float()
                    ours = full[:m][:, cached["layer_idx"]]
                    cos = F.cosine_similarity(ours, theirs, dim=2).min()
                    log(f"intact original vs exp-016 {key}: min row cosine {float(cos):.6f}")
                    assert float(cos) > 0.999, "intact features differ from exp-016"
                    checks[key] = {"exp016_min_cosine": float(cos)}
                if cond == "shuffled" and caption == "gemini" and grids:
                    idx = X20.large_layers(full.shape[1])
                    sub = full[:, idx]
                    v = clamp_value(sub, dev)
                    res = scorer.score(
                        F.normalize(sub[:, i].clamp(-v, v), dim=-1) for i in range(len(idx))
                    )
                    checks.setdefault(key, {})["stored"] = check_stored(res, grids, key, idx)
                    (args.out / "checks.json").write_text(json.dumps(checks, indent=1))
                    del sub, res
                v = clamp_value(full, dev)
                L = full.shape[1]
                res = scorer.score(F.normalize(full[:, i].clamp(-v, v), dim=-1) for i in range(L))
                write(rows_from(res, scorer, {**base, "kind": "lm", "clamp": float(v)}))
                del full, res
                floor(ids, keep, caption, cond, repo, base)
                gc.collect()
                torch.cuda.empty_cache()
                log(
                    f"done {key} {caption} {cond}: forward {t1 - t0:.0f}s, score {time.time() - t1:.0f}s, "
                    f"total {time.time() - t_start:.0f}s"
                )
        del model, dec
        gc.collect()
        torch.cuda.empty_cache()
    (args.out / "checks.json").write_text(json.dumps(checks, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, required=True)
    p.add_argument("--stages", default="mask,score")
    p.add_argument("--scratch", type=Path, default=Path("/root/exp026"))
    p.add_argument("--repos", type=lambda s: s.split(","), default=BIG_LMS)
    p.add_argument("--captions", type=lambda s: s.split(","), default=["gemini", "orig"])
    p.add_argument("--grids", type=Path, default=None)
    p.add_argument("--tokens", type=int, default=3000)
    p.add_argument("--tag-processes", type=int, default=6)
    p.add_argument("--big-cache", default=BIG_CACHE)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for stage in args.stages.split(","):
        log(f"=== stage {stage}")
        {"mask": stage_mask, "score": stage_score}[stage](args)
        log(f"=== stage {stage} done")


if __name__ == "__main__":
    main()
