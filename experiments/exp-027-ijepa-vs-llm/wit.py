"""exp-027: calibrated PRH alignment on WIT-1024 with every ViT mean-pooled, I-JEPA included.

PRH's procedure (q = 0.95 clamp, l2 norm, max over layer pairs) and Gröger's Algorithm 2 (200
permutations, null of the max) through `geoprh.prh_calibration`, for mutual kNN (k = 10) and linear
CKA. Rows:
- `llm`: each of the 10 PRH LLMs (mean-pooled, cached) x each ViT, mean-pooled (`pool-avg`: I-JEPA
  1k and 22k from this experiment, the 17 PRH ViTs from exp-015). The I-JEPA 1k features made with
  the HF repo's own image processor (`pool-avg-hfproc`) are a check.
- `floor`: word TF-IDF with and without English stopwords (exp-023's kernels, no clamp) x each ViT.

Before anything else, one cached CLS pair must reproduce `prh_val_calibrated`, or the job stops.

    uv run python experiments/exp-027-ijepa-vs-llm/wit.py --out <dir> --stored-cal <prh_val_calibrated>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from geoprh import activation_cache as ac
from geoprh import prh_calibration as pc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPECS = [pc.Spec("mutual_knn", 10), pc.Spec("cka_lin", None)]
LANG = ac.Variant(pool="avg", caption_idx=0)
POOLS = {
    p: ac.Variant(pool=p, caption_idx=None, modality="vision") for p in ("avg", "cls", "avg-hfproc")
}
IJEPA = ["facebook/ijepa_vith14_1k", "facebook/ijepa_vith14_22k"]
FIRST = ["vit_huge_patch14_224.mae", "vit_giant_patch14_dinov2.lvd142m"]
Q = 0.95


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def calibrate(vis, lang, permutations: int, seed: int) -> dict:
    return {
        sp.key: pc._calibrate(sp, vis, lang, permutations=permutations, alpha=0.05, seed=seed)
        for sp in SPECS
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stored-cal", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vits", default="all", help="'all' or a comma-separated list")
    args = p.parse_args()
    from aristotelian.prh.prh_models import get_models

    R23 = _module("exp023_run", ROOT / "exp-023-prh-lexical-floor" / "run.py")
    root = args.root or ac.default_root()
    args.out.mkdir(parents=True, exist_ok=True)
    llms, lvms = get_models("val")
    llms = [m for m in llms if ac.cached_layers(root, m, LANG)]
    vits = IJEPA + FIRST + [m for m in lvms if m not in FIRST]
    if args.vits != "all":
        vits = [v for v in vits if v in args.vits.split(",")]
    series = [(v, "avg") for v in vits if ac.load_model(root, v, POOLS["avg"]) is not None]
    missing = [v for v in vits if (v, "avg") not in series]
    if ac.load_model(root, IJEPA[0], POOLS["avg-hfproc"]) is not None:
        series.insert(1, (IJEPA[0], "avg-hfproc"))
    log(f"{len(llms)} LLMs x {len(series)} ViT series; not cached as avg: {missing}")

    # Pipeline check against the stored full-grid calibration (CLS)
    stored = {}
    for line in (args.stored_cal / "calibrated_pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        stored[(r["llm"], r["lvm"])] = r
    llm0 = llms[0]
    vit0 = next(v for v in lvms if (llm0, v) in stored and ac.cached_layers(root, v, POOLS["cls"]))
    vis = pc._caches(pc._layers(vit0, POOLS["cls"], root, Q, args.device), SPECS)
    lang = pc._caches(pc._layers(llm0, LANG, root, Q, args.device), SPECS)
    got = calibrate(vis, lang, args.permutations, args.seed)
    report = {}
    for sp, key in zip(SPECS, ("mutual_knn", "cka_lin"), strict=True):
        g, w = got[sp.key], stored[(llm0, vit0)][key]
        report[key] = {"raw": [g["raw_score"], w["raw_score"]], "mu0": [g["mu0"], w["mu0"]]}
        assert abs(g["raw_score"] - w["raw_score"]) < 2e-3, f"{key} raw differs: {report}"
        assert abs(g["mu0"] - w["mu0"]) < 2e-3, f"{key} null mean differs: {report}"
    (args.out / "wit_checks.json").write_text(
        json.dumps({"pair": [llm0, vit0], **report}, indent=1)
    )
    log(f"stored-calibration check passed: {report}")
    del vis, lang

    path = args.out / "wit_rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["kind"], r["name"], r["vit"], r["pool"]))

    def write(row: dict) -> None:
        with path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    texts = R23.load_captions(None)
    assert len(texts) == 1024
    words = R23.word_features(texts)  # {"word_tfidf", "word_tfidf_nostop"}: [n, V] unit rows
    floors = {k: pc._caches(v.to(args.device)[:, None], SPECS) for k, v in words.items()}
    t0 = time.time()
    for vit, pool in series:
        todo_floor = [k for k in floors if ("floor", k, vit, pool) not in done]
        todo_llm = [m for m in llms if ("llm", m, vit, pool) not in done]
        if not (todo_floor or todo_llm):
            continue
        vis = pc._caches(pc._layers(vit, POOLS[pool], root, Q, args.device), SPECS)
        for k in todo_floor:
            res = calibrate(vis, floors[k], args.permutations, args.seed)
            write({"kind": "floor", "name": k, "vit": vit, "pool": pool, **res})
        for m in todo_llm:
            t1 = time.time()
            lang = pc._caches(pc._layers(m, LANG, root, Q, args.device), SPECS)
            res = calibrate(vis, lang, args.permutations, args.seed)
            write(
                {
                    "kind": "llm",
                    "name": m,
                    "vit": vit,
                    "pool": pool,
                    **res,
                    "seconds": round(time.time() - t1, 1),
                }
            )
            mk, ck = res[SPECS[0].key], res[SPECS[1].key]
            log(
                f"{m} x {vit} ({pool}): mKNN {mk['raw_score']:.3f} (null {mk['mu0']:.3f}) "
                f"CKA {ck['raw_score']:.3f} (null {ck['mu0']:.3f})  {time.time() - t0:.0f}s"
            )
            del lang
        del vis
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    log("wit done")


if __name__ == "__main__":
    main()
