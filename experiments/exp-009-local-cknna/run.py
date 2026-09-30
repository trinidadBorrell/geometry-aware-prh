"""exp-009: local CKA (Emily's `cknna_local`) against the CKNNA definitions, mKNN and CKA.

For each (LLM, ViT) pair, every layer pair and every k:
- the exp-007 metrics (`geoprh.cknna_variants`): mutual kNN, CKNNA paper / centred / code / eq36;
- `local_mutual` (Emily's metric), `local_union`, and the coverage of `local_mutual` (fraction
  of points with >= 4 mutual neighbours), from `geoprh.local_cknna`;
- linear CKA once per pair.
Each metric reports the max over layer pairs (metric(vision, language), as upstream).

At each metric's best layer pair it also saves
- per-point scores (mKNN, centred CKNNA row-wise, local_mutual, local_union) at DIST_KS, and
- a permutation null (the image-caption pairing permuted, Aristotelian `batched_perms`, seed 0):
  chance level at that fixed layer pair. Unlike exp-008 this is not a max-over-layers null, so
  it estimates the chance level of the score, not a calibrated p-value.

    uv run python experiments/exp-009-local-cknna/run.py --out <dir> --permutations 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aristotelian.metrics.utils import batched_perms
from geoprh import activation_cache as ac
from geoprh import cknna_variants as cv
from geoprh import local_cknna as lc
from geoprh.prh_alignment import _git_sha, available, prepare_layers

KS = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1024]
DIST_KS = [10, 50, 200, 500, 1000]
TEXT = ac.Variant(pool="avg", caption_idx=0)
VISION = ac.Variant(pool="cls", caption_idx=None, modality="vision")
GLOBAL = cv.PER_K  # mknn, paper, centred, code, eq36
LOCAL = ("local_mutual", "local_union")
METRICS = (*GLOBAL, *LOCAL)


def load(root: Path, model: str, variant: ac.Variant) -> list[torch.Tensor]:
    payload = ac.load_model(root, model, variant)
    if payload is None:
        raise FileNotFoundError(f"{model} not fully cached under {root}")
    return prepare_layers(payload["feats"])


class Model:
    """A layer stack with the local-metric Grams."""

    def __init__(self, layers: list[torch.Tensor], device: str):
        self.layers = layers
        self.stack = cv.Stack(layers, device)
        self.device = device
        self._local: tuple[int, lc.LocalGram] | None = None

    def local_at(self, k: int) -> lc.LocalGram:
        """float32 Grams shifted by half each point's k-NN similarity (exact for the local CKA,
        and ~1000x more accurate in float32 than unshifted; see geoprh.local_cknna)."""
        if self._local is None or self._local[0] != k:
            self._local = None
            shift = lc.knn_shift(self.stack.K, self.stack.order, k)
            self._local = (k, lc.LocalGram(self.stack.K, shift, torch.float32))
        return self._local[1]

    def single(self, i: int) -> Model:
        return Model([self.layers[i]], self.device)


def local_grid(a: Model, b: Model, ak: cv.AtK, bk: cv.AtK, k: int) -> dict[str, torch.Tensor]:
    """[La, Lb] local_mutual, local_union and coverage (one bmm batch per vision layer)."""
    rows = {name: [] for name in (*LOCAL, "coverage")}
    ga, gb = a.local_at(k), b.local_at(k)
    for i in range(len(a.stack)):
        gk = _gram(ga.Kt[i], ga.KK[i])
        red = lc.reduce(lc.local_scores(ak.m[i], bk.m, gk, gb))
        for name in rows:
            rows[name].append(red[name])
    return {name: torch.stack(v) for name, v in rows.items()}


def best(mat: torch.Tensor) -> tuple[float, list[int]]:
    flat = int(torch.nan_to_num(mat, nan=-torch.inf).argmax())
    return float(mat.reshape(-1)[flat]), [flat // mat.shape[1], flat % mat.shape[1]]


def _gram(Kt: torch.Tensor, KK: torch.Tensor) -> lc.LocalGram:
    g = lc.LocalGram.__new__(lc.LocalGram)
    g.Kt, g.KK = Kt, KK
    return g


def at_pair(a: Model, b: Model, ak: cv.AtK, bk: cv.AtK, k: int, perm) -> dict:
    """All metrics for single-layer models a, b (b relabelled by perm), plus per-point arrays."""
    right = bk.right() if perm is None else bk.permuted(perm)
    out = {name: float(v[0, 0]) for name, v in cv.per_k_scores(ak, bk, right).items()}
    ga, gb = a.local_at(k), b.local_at(k)
    mL, Kt, KK, Lc = bk.m[0], gb.Kt[0], gb.KK[0], b.stack.Kc[0]
    if perm is not None:
        mL, Kt, KK, Lc = (x[perm][:, perm] for x in (mL, Kt, KK, Lc))
    pts = lc.local_scores(ak.m[0], mL, _gram(ga.Kt[0], ga.KK[0]), _gram(Kt, KK))
    red = lc.reduce(pts)
    rows = lc.rowwise(ak.m[0], mL, a.stack.Kc[0], Lc, k)
    out.update({name: float(red[name]) for name in (*LOCAL, "coverage")})
    out["_points"] = {
        "mknn": rows["mknn"],
        "centred": rows["centred"],
        "local_mutual": pts["mutual"],
        "local_union": pts["union"],
    }
    return out


def pair_task(a: Model, b: Model, ks: list[int], perms: torch.Tensor) -> tuple[dict, dict]:
    per_k, arrays = [], {}
    singles: dict[tuple[str, int], Model] = {}

    def single(m: Model, tag: str, i: int) -> Model:
        if (tag, i) not in singles:
            singles[(tag, i)] = m.single(i)
        return singles[(tag, i)]

    for k in ks:
        ak, bk = a.stack.at_k(k), b.stack.at_k(k)
        mats = dict(cv.per_k_scores(ak, bk, bk.right()))
        mats.update(local_grid(a, b, ak, bk, k))
        del ak, bk
        rec = {"k": k}
        by_pair: dict[tuple[int, int], list[str]] = {}
        for name in METRICS:
            raw, layers = best(mats[name])
            rec[name] = {"raw": raw, "layers": layers}
            by_pair.setdefault(tuple(layers), []).append(name)
        for (ia, ib), names in by_pair.items():
            sa, sb = single(a, "a", ia), single(b, "b", ib)
            sak, sbk = sa.stack.at_k(k), sb.stack.at_k(k)
            obs = at_pair(sa, sb, sak, sbk, k, None)
            nulls = [at_pair(sa, sb, sak, sbk, k, p) for p in perms]
            for name in names:
                null = np.array([r[name] for r in nulls], np.float32)
                rec[name].update(
                    null_mean=float(np.nanmean(null)),
                    null_sd=float(np.nanstd(null)),
                    null_q95=float(np.nanquantile(null, 0.95)),
                )
                arrays[f"null_{name}_k{k}"] = null
                if k in DIST_KS and name in obs["_points"]:
                    arrays[f"points_{name}_k{k}"] = obs["_points"][name].cpu().numpy()
                if name == "local_mutual":
                    arrays[f"null_coverage_k{k}"] = np.array([r["coverage"] for r in nulls])
        ia, ib = rec["local_mutual"]["layers"]
        rec["coverage"] = float(mats["coverage"][ia, ib])
        rec["coverage_mean"] = float(mats["coverage"].mean())
        for name in ("local_mutual", "local_union", "coverage"):
            arrays[f"grid_{name}_k{k}"] = mats[name].cpu().numpy()
        per_k.append(rec)
    cka = cv.cka(a.stack, b.stack)
    raw, layers = best(cka)
    sa, sb = single(a, "a", layers[0]), single(b, "b", layers[1])
    null = np.array([float(cv.cka(sa.stack, sb.stack, p)[0, 0]) for p in perms], np.float32)
    arrays["null_cka"] = null
    cka_rec = {
        "raw": raw,
        "layers": layers,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "null_q95": float(np.quantile(null, 0.95)),
    }
    return {"per_k": per_k, "cka": cka_rec}, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--llms", default="all")
    parser.add_argument("--lvms", default="all")
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    cached_llms, cached_lvms, _ = available(args.modelset, root)
    llms = cached_llms if args.llms == "all" else args.llms.split(",")
    lvms = cached_lvms if args.lvms == "all" else args.lvms.split(",")

    (args.out / "pairs").mkdir(parents=True, exist_ok=True)
    partial = args.out / "pairs.jsonl"
    done = set()
    if partial.exists():
        done = {(r["llm"], r["lvm"]) for r in map(json.loads, partial.read_text().splitlines())}
    todo = [(lm, v) for lm in llms for v in lvms if (lm, v) not in done]
    print(f"{len(todo)} pairs to go, {args.permutations} permutations, k = {ks}", flush=True)
    perms = None
    with partial.open("a") as fh:
        for c, llm in enumerate(llms):
            if all((llm, v) in done for v in lvms):
                continue
            lang = Model(load(root, llm, TEXT), args.device)
            for lvm in lvms:
                if (llm, lvm) in done:
                    continue
                t0 = time.time()
                vis = Model(load(root, lvm, VISION), args.device)
                n = vis.stack.n
                if perms is None:
                    perms = torch.cat(
                        list(
                            batched_perms(
                                n, args.permutations, device="cpu", seed=args.seed, chunk_size=8
                            )
                        )
                    ).to(args.device)
                ks_n = [min(k, n - 1) for k in ks]
                res, arrays = pair_task(vis, lang, ks_n, perms)
                stem = f"{llm.replace('/', '__')}__x__{lvm}"
                np.savez_compressed(args.out / "pairs" / f"{stem}.npz", ks=np.array(ks_n), **arrays)
                rec = {"llm": llm, "lvm": lvm, "n": n, **res, "seconds": round(time.time() - t0, 1)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                r10 = res["per_k"][0]
                print(
                    f"[{c + 1}/{len(llms)}] {llm} x {lvm}: k={r10['k']} "
                    + " ".join(
                        f"{m} {r10[m]['raw']:.2f}/{r10[m]['null_mean']:.2f}" for m in METRICS
                    )
                    + f" coverage {r10['coverage']:.2f} ({rec['seconds']}s)",
                    flush=True,
                )
                del vis
            del lang
    meta = {
        "ks": ks,
        "dist_ks": DIST_KS,
        "permutations": args.permutations,
        "seed": args.seed,
        "null": "fixed at each metric's best layer pair (not max over layers)",
        "git_sha": _git_sha(),
        "root": str(root),
        "device": args.device,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
