"""exp-027: score the mean-pooled I-JEPA / MAE / DINOv2 features on D2 (n = 10240) against LLMs.

Targets (ViTs, 7 stored layers each): I-JEPA 1k and 22k, MAE and DINOv2-g mean-pooled (this
experiment's `extract.py`), MAE and DINOv2-g CLS (exp-016). Scored against each target:
- the 4 exp-016 LLMs on original and Gemini captions (7 stored layers each);
- word floors of each caption set: token TF-IDF (Pythia tokenizer, no special tokens, as exp-022)
  and word TF-IDF without English stopwords;
- the other ViTs (ViT x ViT reference).

Same preprocessing and measures as exp-022 / exp-026: PRH clamp (q = 0.95) over the stored layers
and l2 norm, linear CKA and mutual kNN (k = 10, 100, 1000) with their exact null means, every layer
pair. exp-026's `Target` / `Scorer` do the work on the GPU. Before anything else, bloomz-7b1 Gemini x
DINOv2 CLS must reproduce exp-016's stored grid (`grids_full.json`), or the job stops.

    uv run python experiments/exp-027-ijepa-vs-llm/score_d2.py --out <dir> --exp016 <dir> \\
        --grids <exp022>/grids_full.json
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
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R26 = _module("exp026_run", ROOT / "exp-026-word-masking" / "run.py")
M16 = _module("exp016_measure", ROOT / "exp-016-positive-controls" / "measure.py")
LLMS = [
    "bigscience__bloomz-1b7",
    "bigscience__bloomz-7b1",
    "openlm-research__open_llama_7b",
    "huggyllama__llama-13b",
]
CAPTIONS = ("orig", "gemini")
TARGETS = {  # short: (feature key, store)
    "ijepa1k": ("facebook__ijepa_vith14_1k__avg", "out"),
    "ijepa22k": ("facebook__ijepa_vith14_22k__avg", "out"),
    "mae_avg": ("vit_huge_patch14_224.mae__avg", "out"),
    "dino_avg": ("vit_giant_patch14_dinov2.lvd142m__avg", "out"),
    "mae_cls": ("vit_huge_patch14_224.mae__cls", "exp016"),
    "dino_cls": ("vit_giant_patch14_dinov2.lvd142m__cls", "exp016"),
}
CHECK = ("bigscience__bloomz-7b1__gemini", "vit_giant_patch14_dinov2.lvd142m__cls")
PY_TOKENIZER = "EleutherAI/pythia-160m"
MAX_TOKENS = 2048
KS = (10, 100, 1000)
log = R26.log


def path_of(args, key: str, store: str) -> Path:
    root = args.out if store == "out" else args.exp016
    return root / "features" / f"{key}__d2l.pt"


def prepared(path: Path, n: int | None):
    layers, names = M16.prepared_layers(path)
    layers = [x for x, m in zip(layers, names, strict=True) if m != "final"]
    names = [m for m in names if m != "final"]
    if n:
        layers = [x[:n] for x in layers]
    return layers, names


def floors(args, n: int) -> dict[str, list[torch.Tensor]]:
    """{'<caption>__tfidf' / '<caption>__word': [one unit-row dense layer]}."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoTokenizer

    from geoprh import lexical as lex

    df = pd.read_parquet(args.exp016 / "d2.parquet", columns=["caption", "gemini_caption"])
    texts = {"orig": df["caption"].tolist()[:n], "gemini": df["gemini_caption"].tolist()[:n]}
    tok = AutoTokenizer.from_pretrained(PY_TOKENIZER)
    out = {}
    for c, t in texts.items():
        ids = tok(t, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)["input_ids"]
        out[f"{c}__tfidf"] = lex.to_unit_rows(lex.tfidf(lex.count_matrix(ids)[0]))
        vec = TfidfVectorizer(lowercase=True, stop_words="english", norm="l2", dtype=np.float32)
        out[f"{c}__word"] = lex.to_unit_rows(vec.fit_transform(t))
    return {k: [v] for k, v in out.items()}


def check(rows: list[dict], grids_path: Path) -> dict:
    g = next(
        x
        for x in json.loads(grids_path.read_text())
        if (x["a"], x["b"]) == CHECK and x["n"] == 10240
    )
    r = next(x for x in rows if x["set"] == CHECK[0])
    report = {
        "max_abs_cka": float(np.abs(np.array(r["cka"]) - np.array(g["cka"])).max()),
        "max_abs_knn10": float(np.abs(np.array(r["knn"]["10"]) - np.array(g["knn10"])).max()),
        "names_match": r["names_vit"] == g["names_b"],
    }
    log(f"stored-grid check: {report}")
    assert report["names_match"], "layer names differ"
    assert report["max_abs_cka"] < 3e-3 and report["max_abs_knn10"] < 3e-3, "differs from exp-016"
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--grids", type=Path, default=None, help="exp-022 grids_full.json")
    p.add_argument("--targets", type=lambda s: s.split(","), default=list(TARGETS))
    p.add_argument("--group", type=int, default=2, help="targets held at once")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first n rows")
    args = p.parse_args()
    from geoprh import knn_sweep as ksw
    from geoprh import lexical as lex

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.limit
    ks = [k for k in KS if not n or k < n // 2]
    path = args.out / "d2_rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["target"], r["set"]))
    lexical = floors(args, n or 10240)
    log(f"floors ready: {list(lexical)}")
    sets = [(f"{m}__{c}", "llm", c) for c in CAPTIONS for m in LLMS]
    sets.sort(key=lambda s: s[0] != CHECK[0])  # the stored-grid check runs first
    sets += [(k, "floor", k.split("__")[0]) for k in lexical]
    sets += [(TARGETS[t][0], "vit", None) for t in TARGETS]
    order = sorted(args.targets, key=lambda t: t != "dino_cls")  # the check first
    checked = False
    t_start = time.time()
    for g0 in range(0, len(order), args.group):
        shorts = order[g0 : g0 + args.group]
        targets = []
        for s in shorts:
            layers, names = prepared(path_of(args, *TARGETS[s]), n)
            targets.append(R26.Target(s, layers, names, max(ks), dev, ksw, lex))
            del layers
            log(f"target {s} ready")
        scorer = R26.Scorer(targets, ks, dev, ksw, lex)
        for name, kind, caption in sets:
            todo = [t for t in shorts if (t, name) not in done and TARGETS[t][0] != name]
            if not todo:
                continue
            t0 = time.time()
            if kind == "floor":
                layers = lexical[name]
                names = ["floor"]
            elif kind == "llm":
                layers, names = prepared(args.exp016 / "features" / f"{name}__d2l.pt", n)
            else:
                short = next(t for t, v in TARGETS.items() if v[0] == name)
                layers, names = prepared(path_of(args, *TARGETS[short]), n)
            res = scorer.score(layers)
            base = {"set": name, "kind": kind, "caption": caption, "names_set": names}
            rows = [r for r in R26.rows_from(res, scorer, base) if r["target"] in todo]
            if not checked and "dino_cls" in shorts and name == CHECK[0] and args.grids and not n:
                rep = check(rows, args.grids)
                (args.out / "d2_checks.json").write_text(json.dumps(rep, indent=1))
                checked = True
            with path.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            for r in rows:
                done.add((r["target"], name))
            del layers, res
            log(
                f"{'+'.join(shorts)} x {name}: {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)"
            )
        del scorer, targets
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    log("score_d2 done")


if __name__ == "__main__":
    main()
