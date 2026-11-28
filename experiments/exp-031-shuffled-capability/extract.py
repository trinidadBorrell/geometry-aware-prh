"""exp-031: PRH LLM features of word-shuffled WIT-1024 captions, plus the pipeline checks.

Extraction runs platonic-rep's own `extract_llm_features` (same loading, dtype, left padding to the
longest caption of the set, masked mean pooling, loss and bpb) on transformed captions, then
splits the feature file into the shared per-layer cache under the variant key
`minhuh_prh_wit_1024_pool-avg_cid-0_tf-<transform>`.

Transforms:
    shufw<seed>   words (whitespace split) permuted within each caption; rng = (seed, row)
    reext         the intact captions again (cache reproduction check; not pushed)

Commands (one model per process, so a stalled download cannot block the rest):
    uv run python experiments/exp-031-shuffled-capability/extract.py extract \\
        --model bigscience/bloomz-560m --transform shufw0 --scratch <dir> --text-root <cache>
    uv run python experiments/exp-031-shuffled-capability/extract.py check-layer0 \\
        --model bigscience/bloomz-560m --out <dir>
    uv run python experiments/exp-031-shuffled-capability/extract.py check-cache \\
        --model bigscience/bloomz-560m --scratch <dir> --root <cache> --out <dir>
    uv run python experiments/exp-031-shuffled-capability/extract.py check-score \\
        --stored-cal <prh_val_calibrated> --root <cache> --out <dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platonic-rep"))

from geoprh import activation_cache as ac  # noqa: E402

DATASET, SUBSET = "minhuh/prh", "wit_1024"
EXTRACTOR = "exp-031 extract.py via platonic-rep extract_llm_features"
LAYER0_TOL = 0.999
CACHE_TOL = 0.999
SCORE_TOL = 0.002
SCORE_PAIR = ("bigscience/bloomz-560m", "vit_tiny_patch16_224.augreg_in21k")


# --------------------------------------------------------------------------- transforms


def shuffle_words(text: str, rng: np.random.Generator) -> str:
    """Permute the whitespace-separated words of a caption; multi-token words stay whole."""
    words = text.split()
    if len(words) < 2:
        return " ".join(words)
    return " ".join(words[i] for i in rng.permutation(len(words)))


def parse_transform(transform: str) -> tuple[str, int | None]:
    if transform == "reext":
        return "reext", None
    if transform.startswith("shufw") and transform[5:].isdigit():
        return "shufw", int(transform[5:])
    raise ValueError(f"unknown transform {transform!r}")


def transform_texts(texts: list[str], transform: str) -> list[str]:
    kind, seed = parse_transform(transform)
    if kind == "reext":
        return list(texts)
    return [shuffle_words(t, np.random.default_rng([seed, i])) for i, t in enumerate(texts)]


TEXTS_FILE: Path | None = None  # set from --texts; read instead of the dataset when present


def load_texts(caption_idx: int = 0) -> list[str]:
    """Caption `caption_idx` of every WIT-1024 row, as platonic-rep reads it.

    Reads `TEXTS_FILE` when it exists, so that offline runs (weights from a read-only HF cache)
    do not need the dataset."""
    if TEXTS_FILE is not None and TEXTS_FILE.exists():
        return json.loads(TEXTS_FILE.read_text())
    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=SUBSET, split="train")
    return [str(x["text"][caption_idx]) for x in dataset]


def write_texts(path: Path) -> None:
    global TEXTS_FILE
    TEXTS_FILE = None
    texts = load_texts()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(texts))
    print(f"{len(texts)} captions -> {path}")


# --------------------------------------------------------------------------- extraction


def feature_file(scratch: Path, model: str, transform: str) -> Path:
    import utils  # platonic-rep

    return Path(
        utils.to_feature_filename(
            str(scratch / transform),
            DATASET,
            SUBSET,
            model,
            pool="avg",
            prompt=False,
            caption_idx=0,
        )
    )


def extract(model: str, transform: str, scratch: Path, batch_size: int, out: Path) -> Path:
    """Run platonic-rep's extractor on transformed captions; returns the feature file."""
    from extract_features import extract_llm_features  # platonic-rep

    texts = load_texts()
    new = transform_texts(texts, transform)
    out_dir = scratch / transform
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "captions.json").write_text(json.dumps(new))
    args = SimpleNamespace(
        output_dir=str(out_dir),
        dataset=DATASET,
        subset=SUBSET,
        pool="avg",
        prompt=False,
        caption_idx=0,
        force_remake=False,
        qlora=False,
        force_download=False,
        batch_size=batch_size,
    )
    extract_llm_features([model], [{"text": [t]} for t in new], args)
    path = feature_file(scratch, model, transform)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feats = payload["feats"]
    if not torch.isfinite(feats.float()).all():
        raise ValueError(f"{model} {transform}: non-finite features")
    rec = {
        "model": model,
        "transform": transform,
        "shape": list(feats.shape),
        "dtype": str(feats.dtype),
        "loss": float(payload["loss"]),
        "bpb": float(payload["bpb"]),
        "captions_changed": float(np.mean([a != b for a, b in zip(texts, new, strict=True)])),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "extract.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return path


def push(model: str, transform: str, scratch: Path, text_root: Path) -> int:
    kind, seed = parse_transform(transform)
    if kind == "reext":
        return 0
    variant = ac.Variant(pool="avg", caption_idx=0, transform=transform)
    n = ac.push_model(
        text_root,
        model,
        variant,
        feature_file(scratch, model, transform),
        extractor=EXTRACTOR,
        seed=seed,
    )
    print(f"[push] {model} {variant.key}: {n} new layer(s) under {text_root}")
    return n


# --------------------------------------------------------------------------- checks


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)


def token_shuffle(input_ids, attention_mask, special_ids: set[int], rng) -> torch.Tensor:
    """Permute each row's non-special, non-pad tokens in place (positions kept)."""
    ids = input_ids.clone()
    for r in range(ids.shape[0]):
        pos = [
            j
            for j in range(ids.shape[1])
            if attention_mask[r, j] and int(ids[r, j]) not in special_ids
        ]
        if len(pos) > 1:
            ids[r, pos] = ids[r, [pos[i] for i in rng.permutation(len(pos))]]
    return ids


def check_layer0(model: str, out: Path, n: int = 50) -> dict:
    """Mean-pooled layer 0 must not change when tokens are permuted (no additive positions)."""
    from models import load_llm, load_tokenizer  # platonic-rep

    texts = load_texts()[:n]
    lm, tok = load_llm(model), load_tokenizer(model)
    enc = tok(texts, padding="longest", return_tensors="pt")
    shuf = token_shuffle(
        enc["input_ids"],
        enc["attention_mask"],
        set(tok.all_special_ids),
        np.random.default_rng(0),
    )
    device = next(lm.parameters()).device
    mask = enc["attention_mask"].to(device)

    def layer0(ids):
        with torch.no_grad():
            hs = lm(input_ids=ids.to(device), attention_mask=mask).hidden_states[0]
        m = mask.unsqueeze(-1).to(hs.dtype)
        return ((hs * m).sum(1) / m.sum(1)).float().cpu()

    cos = _row_cosine(layer0(enc["input_ids"]), layer0(shuf))
    rec = {
        "check": "layer0",
        "model": model,
        "n": n,
        "min_row_cosine": float(cos.min()),
        "changed_rows": int((shuf != enc["input_ids"]).any(1).sum()),
        "pass": bool(cos.min() >= LAYER0_TOL),
    }
    _record(out, rec)
    lm = None
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def check_cache(model: str, scratch: Path, root: Path, out: Path) -> dict:
    """Re-extracted intact features must reproduce the cached ones at every layer."""
    payload = torch.load(
        feature_file(scratch, model, "reext"), map_location="cpu", weights_only=False
    )
    cached = ac.load_model(root, model, ac.Variant(pool="avg", caption_idx=0))
    if cached is None:
        raise FileNotFoundError(f"{model} intact features not cached under {root}")
    new, old = payload["feats"], cached["feats"]
    if new.shape != old.shape:
        raise ValueError(f"shape {tuple(new.shape)} vs cached {tuple(old.shape)}")
    per_layer = [float(_row_cosine(new[:, i], old[:, i]).min()) for i in range(new.shape[1])]
    rec = {
        "check": "cache",
        "model": model,
        "min_row_cosine_per_layer": per_layer,
        "min_row_cosine": min(per_layer),
        "bpb_new": float(payload["bpb"]),
        "bpb_cached": cached.get("bpb"),
        "pass": bool(min(per_layer) >= CACHE_TOL),
    }
    _record(out, rec)
    return rec


def check_score(stored_cal: Path, root: Path, out: Path, device: str) -> dict:
    """The calibration path reproduces one stored prh_val_calibrated pair."""
    from geoprh import prh_calibration as pc

    llm, lvm = SCORE_PAIR
    stored = next(
        r
        for r in map(json.loads, (stored_cal / "calibrated_pairs.jsonl").read_text().splitlines())
        if r["llm"] == llm and r["lvm"] == lvm
    )
    specs = pc.parse_specs("mutual_knn,cka_lin", "10")
    lang = pc._caches(
        pc._layers(llm, ac.Variant(pool="avg", caption_idx=0), root, 0.95, device), specs
    )
    vis = pc._caches(
        pc._layers(
            lvm, ac.Variant(pool="cls", caption_idx=None, modality="vision"), root, 0.95, device
        ),
        specs,
    )
    diffs = {}
    for spec, old_key in zip(specs, ("mutual_knn", "cka_lin"), strict=True):
        r = pc._calibrate(spec, vis, lang, permutations=200, alpha=0.05, seed=0)
        diffs[spec.key] = {
            f: abs(r[f] - stored[old_key][f]) for f in ("raw_score", "mu0", "g_score")
        }
    worst = max(v for d in diffs.values() for v in d.values())
    rec = {
        "check": "score",
        "pair": SCORE_PAIR,
        "abs_diff": diffs,
        "max_abs_diff": worst,
        "pass": bool(worst <= SCORE_TOL),
    }
    _record(out, rec)
    return rec


def _record(out: Path, rec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "checks.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


# --------------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "command", choices=["texts", "extract", "check-layer0", "check-cache", "check-score"]
    )
    p.add_argument("--texts", type=Path, default=None, help="captions JSON (see `texts`)")
    p.add_argument("--model", default=SCORE_PAIR[0])
    p.add_argument("--transform", default="shufw0")
    p.add_argument("--scratch", type=Path, default=Path("/root/exp031"))
    p.add_argument("--root", type=Path, default=None, help="cache with intact features")
    p.add_argument("--text-root", type=Path, default=None, help="cache for transformed features")
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--stored-cal", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    root = a.root or ac.default_root()
    global TEXTS_FILE
    TEXTS_FILE = a.texts

    if a.command == "texts":
        if a.texts is None:
            p.error("texts needs --texts")
        write_texts(a.texts)
        return
    if a.command == "extract":
        extract(a.model, a.transform, a.scratch, a.batch_size, a.out)
        push(a.model, a.transform, a.scratch, a.text_root or root)
        return
    if a.command == "check-layer0":
        rec = check_layer0(a.model, a.out)
    elif a.command == "check-cache":
        rec = check_cache(a.model, a.scratch, root, a.out)
    else:
        rec = check_score(a.stored_cal, root, a.out, a.device)
    if not rec["pass"]:
        sys.exit(f"check {rec['check']} failed")


if __name__ == "__main__":
    main()
