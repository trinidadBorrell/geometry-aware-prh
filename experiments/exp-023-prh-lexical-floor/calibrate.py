"""exp-023: Gröger's null calibration of the TF-IDF floor and LLM layer 0 on the PRH pool.

Algorithm 2 of Gröger et al. through `geoprh.prh_calibration`: permute the pairing, recompute the
score matrix, compare its max to the null of the max. Each series is calibrated over the cells it can
choose from: one word kernel or layer 0 x all ViT layers. The best-layer values (all LLM x all ViT
layers) are the stored `prh_val_calibrated` results. A full grid is first recomputed for one pair and
must reproduce those stored values, or the job stops.

    uv run python experiments/exp-023-prh-lexical-floor/calibrate.py --out <dir> --stored <prh_val> \\
        --stored-cal <prh_val_calibrated>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from geoprh import activation_cache as ac
from geoprh import prh_calibration as pc

HERE = Path(__file__).resolve().parent
SPECS = [pc.Spec("mutual_knn", 10), pc.Spec("cka_lin", None)]
LANG = ac.Variant(pool="avg", caption_idx=0)
VISION = ac.Variant(pool="cls", caption_idx=None, modality="vision")
Q = 0.95


def _run_module():
    spec = importlib.util.spec_from_file_location("exp023_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp023_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def vit_task(args) -> list[dict]:
    """All series against one ViT."""
    vit, series, root, permutations, seed = args
    torch.set_num_threads(1)
    vis = pc._caches(pc._layers(vit, VISION, root, Q, "cpu"), SPECS)
    out = []
    for s in series:
        feats = torch.load(s["path"], weights_only=False)
        lang = pc._caches(feats, SPECS)
        res = {
            sp.key: pc._calibrate(sp, vis, lang, permutations=permutations, alpha=0.05, seed=seed)
            for sp in SPECS
        }
        out.append({"vit": vit, "kind": s["kind"], "name": s["name"], "llms": s["llms"], **res})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stored", type=Path, required=True)
    p.add_argument("--stored-cal", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scratch", type=Path, default=Path("/root/exp023cal"))
    args = p.parse_args()
    from transformers import AutoTokenizer

    run = _run_module()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    root = args.root or ac.default_root()
    meta = json.loads((args.stored / "cross_modal.json").read_text())
    llms = [m for m in meta["llms"] if ac.cached_layers(root, m, LANG)]
    vits = [m for m in meta["lvms"] if ac.cached_layers(root, m, VISION)]
    texts = run.load_captions(None)
    assert len(texts) == 1024

    def save(tag: str, x: torch.Tensor) -> Path:
        path = args.scratch / f"{tag}.pt"
        torch.save(x, path)
        return path

    series = []
    tfidf: dict[str, dict] = {}
    for llm in llms:
        ids = AutoTokenizer.from_pretrained(llm)(texts)["input_ids"]
        x = run.token_features(ids)["tfidf"]
        h = hashlib.sha1(x.numpy().tobytes()).hexdigest()[:10]  # models sharing a tokenizer
        if h not in tfidf:
            tfidf[h] = {
                "kind": "tfidf",
                "name": h,
                "llms": [],
                "path": save(f"tfidf-{h}", x[:, None]),
            }
        tfidf[h]["llms"].append(llm)
        l0 = pc._layers(llm, LANG, root, Q, "cpu")[:, :1]  # clamped with all layers, as stored
        path = save(f"l0-{llm.replace('/', '__')}", l0)
        series.append({"kind": "layer0", "name": llm, "llms": [llm], "path": path})
    series += list(tfidf.values())
    word = run.word_features(texts)["word_tfidf_nostop"]
    series.append(
        {
            "kind": "word_tfidf_nostop",
            "name": "word",
            "llms": llms,
            "path": save("word", word[:, None]),
        }
    )
    print(f"{len(series)} series x {len(vits)} ViTs: {len(tfidf)} TF-IDF tokenizers", flush=True)

    # Pipeline check against the stored full-grid calibration
    stored = {}
    for line in (args.stored_cal / "calibrated_pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        stored[(r["llm"], r["lvm"])] = r
    llm, vit = llms[0], vits[0]
    vis = pc._caches(pc._layers(vit, VISION, root, Q, "cpu"), SPECS)
    lang = pc._caches(pc._layers(llm, LANG, root, Q, "cpu"), SPECS)
    t0 = time.time()
    for sp, key in zip(SPECS, ("mutual_knn", "cka_lin"), strict=True):
        got = pc._calibrate(
            sp, vis, lang, permutations=args.permutations, alpha=0.05, seed=args.seed
        )
        want = stored[(llm, vit)].get(key) or stored[(llm, vit)]["results"][sp.key]
        print(
            f"check {key}: raw {got['raw_score']:.4f} vs {want['raw_score']:.4f}; null mean {got['mu0']:.4f} vs {want['mu0']:.4f}; {time.time() - t0:.0f}s",
            flush=True,
        )
        assert abs(got["raw_score"] - want["raw_score"]) < 2e-3, "raw score differs"
        assert abs(got["mu0"] - want["mu0"]) < 2e-3, "null mean differs"
    del vis, lang

    path = args.out / "calibrated.jsonl"
    path.unlink(missing_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = [(v, series, root, args.permutations, args.seed) for v in vits]
        for i, fut in enumerate(as_completed(pool.submit(vit_task, t) for t in tasks)):
            rows = fut.result()
            with path.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
            print(f"[{i + 1}/{len(tasks)}] {rows[0]['vit']}: {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
