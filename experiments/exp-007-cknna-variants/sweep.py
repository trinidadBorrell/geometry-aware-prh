"""exp-007: k sweep of the four CKNNA definitions, with mutual kNN and linear CKA.

Variants (see `geoprh.cknna_variants`): `paper` (Huh et al. Eqs. 16-18 as printed, row-centred,
= exp-005), `centred` (the same with the CKA centring H K H), `code` (platonic-rep /
Aristotelian `cknna`, mask then unbiased HSIC) and `eq36` (Groger et al. Eq. 36, mask then
double-centre). For each (LLM, ViT) pair every layer pair is scored at every k; linear CKA once
per pair. Metrics are metric(vision, language) as upstream; the max over layer pairs is
reported. Preprocessing is platonic-rep's (q=0.95 clamp, l2 norm).

    uv run python experiments/exp-007-cknna-variants/sweep.py --out <dir>
    uv run python experiments/exp-007-cknna-variants/sweep.py --out <dir> --vision-vision
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geoprh import activation_cache as ac
from geoprh import cknna_variants as cv
from geoprh.prh_alignment import _git_sha, available, prepare_layers

KS = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1024]
TEXT = ac.Variant(pool="avg", caption_idx=0)
VISION = ac.Variant(pool="cls", caption_idx=None, modality="vision")


def load_stack(root: Path, model: str, variant: ac.Variant, device: str) -> cv.Stack:
    payload = ac.load_model(root, model, variant)
    if payload is None:
        raise FileNotFoundError(f"{model} not fully cached under {root}")
    return cv.Stack(prepare_layers(payload["feats"]), device)


def best(mat: torch.Tensor) -> tuple[float, list[int]]:
    """Max over layer pairs and its (vision, language) layer indices."""
    flat = int(mat.argmax())
    return float(mat.reshape(-1)[flat]), [flat // mat.shape[1], flat % mat.shape[1]]


def sweep(a: cv.Stack, b: cv.Stack, ks: list[int]) -> dict[str, np.ndarray]:
    """{metric: [len(ks), La, Lb]} for mKNN and the CKNNA variants, plus 'cka': [La, Lb]."""
    mats = {name: np.zeros((len(ks), len(a), len(b)), np.float32) for name in cv.PER_K}
    for t, k in enumerate(ks):
        ak, bk = a.at_k(k), b.at_k(k)
        for name, mat in cv.per_k_scores(ak, bk, bk.right()).items():
            mats[name][t] = mat.cpu().numpy()
        del ak, bk
    mats["cka"] = cv.cka(a, b).cpu().numpy()
    return mats


def grid(root: Path, llms: list[str], lvms: list[str], ks: list[int], out: Path, device: str):
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    partial = out / "sweep_pairs.jsonl"
    done = set()
    if partial.exists():
        done = {(r["llm"], r["lvm"]) for r in map(json.loads, partial.read_text().splitlines())}
    todo = [(lm, v) for lm in llms for v in lvms if (lm, v) not in done]
    print(f"{len(todo)} pairs to go, k = {ks}", flush=True)
    with partial.open("a") as fh:
        for c, llm in enumerate(llms):
            if all((llm, v) in done for v in lvms):
                continue
            lang = load_stack(root, llm, TEXT, device)
            for lvm in lvms:
                if (llm, lvm) in done:
                    continue
                t0 = time.time()
                vis = load_stack(root, lvm, VISION, device)
                ks_n = [min(k, vis.n - 1) for k in ks]
                mats = sweep(vis, lang, ks_n)
                stem = f"{llm.replace('/', '__')}__x__{lvm}"
                np.savez_compressed(out / "pairs" / f"{stem}.npz", ks=np.array(ks_n), **mats)
                per_k = []
                for t, k in enumerate(ks_n):
                    rec = {"k": k}
                    for name in cv.PER_K:
                        rec[name], rec[f"{name}_layers"] = best(torch.from_numpy(mats[name][t]))
                    per_k.append(rec)
                cka, cka_layers = best(torch.from_numpy(mats["cka"]))
                rec = {
                    "llm": llm,
                    "lvm": lvm,
                    "n": vis.n,
                    "cka": cka,
                    "cka_layers": cka_layers,
                    "per_k": per_k,
                    "seconds": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                peaks = "  ".join(
                    f"{name} {max(r[name] for r in per_k):.2f}" for name in cv.CKNNA_VARIANTS
                )
                print(
                    f"[{c + 1}/{len(llms)}] {llm} x {lvm}: CKA {cka:.2f}, peak {peaks} "
                    f"({rec['seconds']}s)",
                    flush=True,
                )
                del vis
            del lang


def vision_vision(root: Path, lvms: list[str], ks: list[int], out: Path, device: str) -> None:
    """PRH Fig. 12 setting: last-block CLS of every ViT against every other."""
    layers = []
    for m in lvms:
        payload = ac.load_model(root, m, VISION)
        layers.append(prepare_layers(payload["feats"])[-1])
    vis = cv.Stack(layers, device)
    ks = [min(k, vis.n - 1) for k in ks]
    mats = sweep(vis, vis, ks)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "vision_vision.npz", ks=np.array(ks), lvms=np.array(lvms), **mats)
    print(f"saved {out / 'vision_vision.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--llms", default="all", help="'all' (cached modelset) or a list")
    parser.add_argument("--lvms", default="all", help="'all' (cached modelset) or a list")
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument("--vision-vision", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    cached_llms, cached_lvms, _ = available(args.modelset, root)
    llms = cached_llms if args.llms == "all" else args.llms.split(",")
    lvms = cached_lvms if args.lvms == "all" else args.lvms.split(",")
    if args.vision_vision:
        vision_vision(root, lvms, ks, args.out, args.device)
    else:
        grid(root, llms, lvms, ks, args.out, args.device)
    meta = {"ks": ks, "git_sha": _git_sha(), "root": str(root), "device": args.device}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
