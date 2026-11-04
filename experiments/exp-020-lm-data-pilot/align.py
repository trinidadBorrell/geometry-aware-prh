"""exp-020: calibrated LM x ViT alignment and matched base-vs-continuation contrasts.

Features come from several output folders (`--roots`, searched in order): this experiment's LM
features (`extract_lm.py`) and exp-016's cached ViT features and geometry (D2: `d2s` = first
1024 rows, all layers; `d2l` = 10240 rows, 7 layers).

Stage `geometry`: exp-016's per-layer ORC / density / TwoNN for the LM `d2s` features of the first
root (exp-016 `measure.stage_geometry`).

Stage `pairs` (one row per pair in `pairs.jsonl`, arrays in `pairs/<id>.npz`):
- mutual kNN (k = 10) and linear CKA for every layer pair, calibrated per cell with a
  200-permutation null and the max over cells against the null of the max (exp-016
  `align_pair`);
- at n = 10240 (`d2l`): mutual kNN at every k in `--ks` (default 10 and n / 100) with
  `geoprh.knn_sweep`, calibrated the same way;
- at n = 1024 (`d2s`): ORC given density and density agreement (Spearman across samples), as
  exp-016;
- two readouts per measure: the best cell (largest score - null mean) and a fixed cell at
  relative depth ~0.75 on both sides (selection-free); per-sample mutual-kNN overlaps at both
  cells are saved for the contrasts.

Stage `contrasts` (`contrasts.jsonl`): for a base LM and its continuation against the same ViT,
caption set and n, the difference of calibrated excesses (continuation - base) at each model's
own best cell and at the fixed cell, with a 95% bootstrap CI over samples (the same resampled
samples for both models; neighbourhoods fixed, as exp-016's CI): mutual kNN from the per-sample
overlaps, linear CKA recomputed on the resampled rows of the PRH features.

    uv run python experiments/exp-020-lm-data-pilot/align.py --roots <exp020 out>,<exp016 out> \\
        --stages geometry,pairs,contrasts
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp-016-positive-controls"))
import measure as M16  # noqa: E402  exp-016 measure.py (align_pair, rank_pair, geometry)

from geoprh import knn_sweep as ksw  # noqa: E402
from geoprh import layer_calibration as lc  # noqa: E402
from geoprh import paired  # noqa: E402

VITS = M16.EX.VITS
CAPTIONS = ("orig", "gemini")
TAGS = ("d2s", "d2l")
FIXED_DEPTH = 0.75
CONTRASTS = [  # (base, continuation, what the continuation adds)
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-Coder-0.5B", "code"),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Coder-1.5B", "code"),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-1.5B", "math"),
    ("Qwen/Qwen2-0.5B-Instruct", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "images"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolVLM-256M-Instruct", "images"),
]


def st(repo: str) -> str:
    return repo.replace("/", "__")


# ---------------------------------------------------------------- features


class Store:
    def __init__(self, roots: list[Path]):
        self.roots = roots

    def _find(self, sub: str, name: str) -> Path | None:
        for r in self.roots:
            p = r / sub / name
            if p.exists():
                return p
        return None

    def feat(self, key: str) -> Path | None:
        return self._find("features", f"{key}.pt")

    def geom(self, key: str) -> Path | None:
        return self._find("geometry", f"{key}.npz")

    def available(self) -> set[str]:
        return {p.stem for r in self.roots for p in (r / "features").glob("*.pt")}


@dataclass
class Model:
    cache: lc.ModelCache
    sweep: ksw.SweepCache | None
    names: list[str]

    @property
    def depths(self) -> np.ndarray:
        idx = np.array([int(x) for x in self.names], dtype=float)
        return idx / max(idx.max(), 1)


class Caches:
    """kNN masks / Grams (k = 10) and, for a k sweep, neighbour ranks; least recently used evicted."""

    def __init__(self, store: Store, device: str, keep: int, k_max: int):
        self.store, self.device, self.keep, self.k_max = store, device, keep, k_max
        self.data: OrderedDict[str, Model] = OrderedDict()

    def get(self, key: str, sweep: bool) -> Model:
        if key in self.data and (self.data[key].sweep is not None or not sweep):
            self.data.move_to_end(key)
            return self.data[key]
        self.data.pop(key, None)
        while len(self.data) >= self.keep:
            self.data.popitem(last=False)
            torch.cuda.empty_cache()
        layers, names = M16.prepared_layers(self.store.feat(key))
        assert "final" not in names
        layers = [x.to(self.device) for x in layers]
        cache = lc.build_cache(torch.stack(layers, 1), 10)
        sw = ksw.build_sweep_cache(layers, self.k_max) if sweep else None
        self.data[key] = Model(cache, sw, names)
        return self.data[key]


def fixed_cell(a: Model, b: Model) -> tuple[int, int]:
    return int(np.abs(a.depths - FIXED_DEPTH).argmin()), int(
        np.abs(b.depths - FIXED_DEPTH).argmin()
    )


# ---------------------------------------------------------------- pairs


def sweep_rows(a: Model, b: Model, ks: list[int], permutations, alpha, seed, fixed):
    res = ksw.calibrated_sweep(a.sweep, b.sweep, ks, permutations, alpha, seed)
    rows, arrays = {}, {}
    for k, r in res.items():
        best = np.unravel_index(int((r["score"] - r["mu0"]).argmax()), r["score"].shape)
        rows[k] = {
            "best": M16.cell_summary(r, best),
            "fixed": M16.cell_summary(r, fixed),
            **{f"max_{m}": r[f"max_{m}"] for m in ("g", "p", "tau")},
        }
        for which, (i, j) in (("best", best), ("fixed", fixed)):
            sa = ksw.SweepCache(a.sweep.ranks[i : i + 1], a.sweep.idx[i : i + 1])
            sb = ksw.SweepCache(b.sweep.ranks[j : j + 1], b.sweep.idx[j : j + 1])
            arrays[f"o_k{k}_{which}"] = ksw.overlaps(sa, sb, [k])[0, 0, 0].cpu().numpy()
        for m in ("score", "mu0", "p"):
            arrays[f"sweep_k{k}_{m}"] = r[m].numpy()
    return rows, arrays


def orc_rows(store: Store, a: str, b: str, permutations, alpha, seed):
    ga, gb = store.geom(a), store.geom(b)
    if ga is None or gb is None:
        return {}, {}
    za_, zb_ = np.load(ga), np.load(gb)
    keep = ~(za_["dup"] | zb_["dup"])
    names_a, names_b = list(za_["names"]), list(zb_["names"])
    rows, arrays = {"orc_dropped": int((~keep).sum())}, {}
    for k in M16.KS:
        oa, da = M16.fill_degenerate(za_[f"orc_k{k}"][:, keep], za_[f"dens_k{k}"][:, keep])
        ob, db = M16.fill_degenerate(zb_[f"orc_k{k}"][:, keep], zb_[f"dens_k{k}"][:, keep])
        for q, (xa, xb) in {
            "orc": (paired.residualize(oa, da), paired.residualize(ob, db)),
            "dens": (paired.standardized_ranks(da), paired.standardized_ranks(db)),
        }.items():
            arr, sm = M16.rank_pair(xa, xb, permutations, alpha, seed, names_a, names_b)
            rows[f"{q}_k{k}"] = sm
            arrays.update({f"{q}_k{k}_{x}": v for x, v in arr.items()})
    return rows, arrays


def stage_pairs(specs, store, out, device, permutations, alpha, ks, keep):
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    path = out / "pairs.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
    specs = [s for s in specs if s["id"] not in done]
    specs.sort(key=lambda s: (s["tag"], s["a"]))  # n = 1024 first; caches reused per A
    caches = Caches(store, device, keep, max(ks))
    print(f"pairs: {len(specs)} to do", flush=True)
    t0 = time.time()
    for i, s in enumerate(specs):
        a_key, b_key, big = f"{s['a']}__{s['tag']}", f"{s['b']}__{s['tag']}", s["tag"] == "d2l"
        a = b = None  # free evicted caches
        a, b = caches.get(a_key, big), caches.get(b_key, big)
        seed = zlib.crc32(s["id"].encode())
        arrays, summary = M16.align_pair(
            a.cache, b.cache, a.names, b.names, permutations, alpha, seed
        )
        fixed = fixed_cell(a, b)
        best10 = tuple(summary["mutual_knn"]["best"]["cell"])
        for m in lc.METRICS:
            r = {x: arrays[f"{m}_{x}"] for x in ("score", "mu0", "tau", "p", "g")}
            summary[m]["fixed"] = M16.cell_summary(r, fixed)
        for which, (ci, cj) in (("best", best10), ("fixed", fixed)):
            o = (a.cache.masks[ci] * b.cache.masks[cj]).sum(1) / a.cache.k
            arrays[f"o_k10_{which}"] = o.cpu().numpy()
        row = {**s, "n": a.cache.masks.shape[1], "names_a": a.names, "names_b": b.names}
        row.update(summary)
        if big:
            row["knn_sweep"], arr = sweep_rows(a, b, ks, permutations, alpha, seed, fixed)
            arrays.update(arr)
        else:
            rows, arr = orc_rows(store, a_key, b_key, permutations, alpha, seed)
            row.update(rows)
            arrays.update(arr)
        np.savez(out / "pairs" / f"{s['id']}.npz", **arrays)
        with path.open("a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        mk = summary["mutual_knn"]["best"]
        print(
            f"[{i + 1}/{len(specs)}] {s['id']}: mkNN {mk['score']:.3f} (null {mk['mu0']:.3f}), "
            f"CKA {summary['cka_lin']['best']['score']:.3f}  {time.time() - t0:.0f}s",
            flush=True,
        )


# ---------------------------------------------------------------- contrasts


def _cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x - x.mean(0), y - y.mean(0)
    return (x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())


def stage_contrasts(contrasts, store, out, device, reps=200, seed=0):
    rows = {}
    for line in (out / "pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = r
    path = out / "contrasts.jsonl"
    path.unlink(missing_ok=True)
    feats: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def layer(key: str, name: str) -> torch.Tensor:
        if key not in feats:
            while len(feats) >= 6:  # a contrast touches 3-4 feature sets
                feats.popitem(last=False)
            layers, names = M16.prepared_layers(store.feat(key))
            feats[key] = dict(zip(names, layers, strict=True))
        feats.move_to_end(key)
        return feats[key][name].to(device)

    for c in contrasts:
        pb, pc = rows.get(c["base_id"]), rows.get(c["cont_id"])
        if pb is None or pc is None:
            continue
        zb, zc = (
            np.load(out / "pairs" / f"{c['base_id']}.npz"),
            np.load(out / "pairs" / f"{c['cont_id']}.npz"),
        )
        n = pb["n"]
        gen = torch.Generator().manual_seed(seed)
        boot = torch.randint(0, n, (reps, n), generator=gen)
        res = {**c, "n": n}
        ks = [10] + [int(k) for k in pb.get("knn_sweep", {}) if int(k) != 10]
        for which in ("best", "fixed"):
            for k in ks:
                sb = pb["mutual_knn"][which] if k == 10 else pb["knn_sweep"][str(k)][which]
                sc = pc["mutual_knn"][which] if k == 10 else pc["knn_sweep"][str(k)][which]
                ob, oc = (torch.from_numpy(z[f"o_k{k}_{which}"]) for z in (zb, zc))
                diff = (oc - ob)[boot].mean(1) - (sc["mu0"] - sb["mu0"])
                res[f"mknn_k{k}_{which}"] = {
                    "base": sb["score"] - sb["mu0"],
                    "cont": sc["score"] - sc["mu0"],
                    "diff": (sc["score"] - sc["mu0"]) - (sb["score"] - sb["mu0"]),
                    "ci": torch.quantile(diff, torch.tensor([0.025, 0.975])).tolist(),
                }
            cb, cc = pb["cka_lin"][which], pc["cka_lin"][which]
            xb = layer(pb["a"] + f"__{pb['tag']}", pb["names_a"][cb["cell"][0]])
            vb = layer(pb["b"] + f"__{pb['tag']}", pb["names_b"][cb["cell"][1]])
            xc = layer(pc["a"] + f"__{pc['tag']}", pc["names_a"][cc["cell"][0]])
            vc = layer(pc["b"] + f"__{pc['tag']}", pc["names_b"][cc["cell"][1]])
            assert abs(float(_cka(xb, vb)) - cb["score"]) < 1e-3, "CKA does not match align_pair"
            d = torch.stack(
                [_cka(xc[i], vc[i]) - _cka(xb[i], vb[i]) for i in boot.to(device)]
            ).cpu() - (cc["mu0"] - cb["mu0"])
            res[f"cka_{which}"] = {
                "base": cb["score"] - cb["mu0"],
                "cont": cc["score"] - cc["mu0"],
                "diff": (cc["score"] - cc["mu0"]) - (cb["score"] - cb["mu0"]),
                "ci": torch.quantile(d, torch.tensor([0.025, 0.975])).tolist(),
            }
        with path.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(
            f"{c['base']} -> {c['cont']} x {c['vit']} {c['caption']} {c['tag']}: "
            f"mkNN diff {res['mknn_k10_best']['diff']:+.3f} {np.round(res['mknn_k10_best']['ci'], 3)}, "
            f"CKA diff {res['cka_best']['diff']:+.3f} {np.round(res['cka_best']['ci'], 3)}",
            flush=True,
        )


# ---------------------------------------------------------------- exp-020 specs


def pair_id(group, a, b, tag) -> str:
    return f"{group}__{a}__{b}__{tag}"


def specs_020(avail: set[str]) -> tuple[list[dict], list[dict]]:
    lms = sorted({st(x) for c in CONTRASTS for x in c[:2]})
    specs, contrasts = [], []

    def add(group, a, b, tag, **extra):
        if f"{a}__{tag}" in avail and f"{b}__{tag}" in avail:
            specs.append(
                {
                    "id": pair_id(group, a, b, tag),
                    "group": group,
                    "a": a,
                    "b": b,
                    "tag": tag,
                    **extra,
                }
            )

    for m, v, c, tag in itertools.product(lms, VITS, CAPTIONS, TAGS):
        add("LV", f"{m}__{c}", f"{v}__cls", tag, caption=c)
    for base, cont, what in CONTRASTS:
        for c, tag in itertools.product(CAPTIONS, TAGS):
            add("LL", f"{st(base)}__{c}", f"{st(cont)}__{c}", tag, caption=c, adds=what)
            for v in VITS:
                contrasts.append(
                    {
                        "base": base,
                        "cont": cont,
                        "adds": what,
                        "vit": v,
                        "caption": c,
                        "tag": tag,
                        "base_id": pair_id("LV", f"{st(base)}__{c}", f"{v}__cls", tag),
                        "cont_id": pair_id("LV", f"{st(cont)}__{c}", f"{v}__cls", tag),
                    }
                )
    return specs, contrasts


def parse(description: str, default_ks: str, extra=None):
    """Shared CLI; `extra(parser)` adds experiment-specific options."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--roots", required=True, help="comma list of output dirs (first = this experiment)"
    )
    p.add_argument("--stages", default="geometry,pairs,contrasts")
    p.add_argument("--ks", default=default_ks, help="mutual-kNN sizes at n = 10240")
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--keep-caches", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tmp", type=Path, default=Path("/tmp/exp020_layers"))
    p.add_argument("--only", default=None, help="substring filter on pair ids (debugging)")
    if extra:
        extra(p)
    return p.parse_args()


def main() -> None:
    args = parse(__doc__.splitlines()[0], "10,102")
    roots = [Path(r) for r in args.roots.split(",")]
    store, out = Store(roots), roots[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stages = args.stages.split(",")
    if "geometry" in stages:
        M16.stage_geometry(out, args.workers, args.tmp)
    specs, contrasts = specs_020(store.available())
    if args.only:
        specs = [s for s in specs if args.only in s["id"]]
    if "pairs" in stages:
        ks = [int(k) for k in args.ks.split(",")]
        stage_pairs(specs, store, out, device, args.permutations, args.alpha, ks, args.keep_caches)
    if "contrasts" in stages:
        stage_contrasts(contrasts, store, out, device)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
