"""exp-016: alignment and per-sample local-structure agreement for every control pair.

Stage `geometry` (n = 1024 feature sets: d1, d2s): per layer (and the final projected embedding of
the dual encoders, as an extra last row) the node ORC and local density at k = 10, 30 and the
TwoNN ID, with exp-014's `layer_task` (PRH preprocessing, union kNN graph, exact ORC, alpha = 0).
Near-duplicate samples (distance to the nearest other sample < 1% of the median such distance, in
the last layer) are flagged and left out of the ORC statistics, as the duplicate caption pair
493 / 531 of wit_1024 in exp-014.

Stage `pairs`: for each pair in `pair_specs` (groups below), on the PRH features:
- mutual kNN (k = 10) and linear CKA for every layer pair, each cell calibrated against a
  200-permutation null of the sample pairing (`layer_calibration`, Algorithm 1 per cell), plus
  the max over cells calibrated against the null of the max (Algorithm 2 of Groger et al.);
- at n = 1024: Spearman agreement of ORC given density and of density (`paired`), calibrated
  the same way;
- for the best cell (largest score - null mean) and, for dual encoders, the final-embedding
  cell: the per-sample mutual-kNN overlap and a bootstrap 95% CI of its mean (samples resampled,
  neighbourhoods fixed) minus the null mean.

Groups
- A_orig / A_gemini: a dual encoder's text tower x its own image tower (positive control)
- A_cross: text tower of one dual encoder x image tower of another (d1)
- B1: one LLM, original caption x Gemini caption (same model, different text: a text ceiling)
- B2: one ViT, image x augmented image (same model, different view: an image ceiling)
- B3_orig / B3_gemini: LLM x ViT with original / Gemini captions (does richer text raise it?)
- X: LLM x ViT on d1 (the PRH setting, exp-012 / exp-014 reference)
- W_llm / W_vit: within-modality pairs on d1
- D_shuffle: A_orig, X and B2 pairs with B's samples shuffled once (must be null)
- D_random: a randomly initialised ViT-B/16 x LLMs, ViTs and text towers (should be ~null)

    uv run python experiments/exp-016-positive-controls/measure.py --root <dir> --stages geometry,pairs
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geoprh import layer_calibration as lc
from geoprh import paired

HERE = Path(__file__).resolve().parent
KS = (10, 30)
SHUFFLE_SEED = 12345
DUP_RATIO = 0.01


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EX = _load_module("exp016_extract", HERE / "extract.py")


def layer_task(args):
    """exp-014's per-layer ORC / density / TwoNN (wrapped so worker processes can import it)."""
    e14 = _load_module("exp014_compute", HERE.parent / "exp-014-orc-paired" / "compute.py")
    return e14.layer_task(args)


def _single_thread():
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def prepared_layers(path: Path) -> tuple[list[torch.Tensor], list[str]]:
    """PRH preprocessing (q = 0.95 clamp, l2 norm) of every layer and of the final embedding."""
    from aristotelian.prh.preprocess import prepare_features

    obj = torch.load(path, weights_only=False)
    groups = [(obj["layers"].float(), [str(i) for i in obj["layer_idx"]])]
    if "final" in obj:
        groups.append((obj["final"].float()[:, None], ["final"]))
    layers, names = [], []
    for x, nm in groups:
        x = F.normalize(prepare_features(x, q=0.95, exact=False), dim=-1)
        layers += list(x.unbind(1))
        names += nm
    return layers, names


# ---------------------------------------------------------------- geometry


def near_duplicates(x: torch.Tensor) -> np.ndarray:
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return (nn < DUP_RATIO * nn.median()).numpy()


def stage_geometry(root: Path, workers: int, tmp: Path) -> None:
    out = root / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    layout, tasks = {}, []
    for f in sorted((root / "features").glob("*.pt")):
        if f.stem.endswith("__d2l") or (out / f"{f.stem}.npz").exists():
            continue
        layers, names = prepared_layers(f)
        files = []
        for i, x in enumerate(layers):
            p = tmp / f"{f.stem}_{i:02d}.npy"
            np.save(p, x.numpy())
            files.append(str(p))
            tasks += [(str(p), k, k == KS[0]) for k in KS]
        layout[f.stem] = (files, names, near_duplicates(layers[-1]))
    print(f"geometry: {len(layout)} feature sets, {len(tasks)} layer tasks", flush=True)
    t0, res = time.time(), {}
    with ProcessPoolExecutor(workers, initializer=_single_thread) as pool:
        for i, (p, k, node, dens, twonn) in enumerate(pool.map(layer_task, tasks, chunksize=1)):
            res[(p, k)] = (node, dens, twonn)
            if i % 100 == 0:
                print(f"{i + 1}/{len(tasks)} layer tasks, {time.time() - t0:.0f}s", flush=True)
    for stem, (files, names, dup) in layout.items():
        arrays = {"names": np.array(names), "dup": dup}
        arrays["twonn"] = np.array([res[(p, KS[0])][2] for p in files])
        for k in KS:
            arrays[f"orc_k{k}"] = np.stack([res[(p, k)][0] for p in files])
            arrays[f"dens_k{k}"] = np.stack([res[(p, k)][1] for p in files])
        np.savez(out / f"{stem}.npz", **arrays)
        if dup.any():
            print(f"{stem}: near-duplicate samples {np.flatnonzero(dup).tolist()}", flush=True)
    print(f"geometry done, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- pairs


def pair_specs(avail: set[str]) -> list[dict]:
    specs = []

    def add(group, a, b, tag, shuffle=False):
        if f"{a}__{tag}" in avail and f"{b}__{tag}" in avail:
            specs.append({"group": group, "a": a, "b": b, "tag": tag, "shuffle": shuffle})

    llms = [m.replace("/", "__") for m in EX.LLMS]
    rnd = f"random-{EX.RANDOM_VIT}__cls"
    for t in EX.TOWERS:
        for tag in ("d1", "d2s", "d2l"):
            add("A_orig", f"{t}__text-orig", f"{t}__image", tag)
            add("A_gemini", f"{t}__text-gemini", f"{t}__image", tag)
        add("D_shuffle", f"{t}__text-orig", f"{t}__image", "d1", shuffle=True)
        add("D_random", rnd, f"{t}__text-orig", "d1")
    for t1, t2 in itertools.permutations(EX.TOWERS, 2):
        add("A_cross", f"{t1}__text-orig", f"{t2}__image", "d1")
    for m in llms:
        for tag in ("d2s", "d2l"):
            add("B1", f"{m}__orig", f"{m}__gemini", tag)
            for v in EX.VITS:
                add("B3_orig", f"{m}__orig", f"{v}__cls", tag)
                add("B3_gemini", f"{m}__gemini", f"{v}__cls", tag)
        for v in EX.VITS:
            add("X", f"{m}__orig", f"{v}__cls", "d1")
            add("D_shuffle", f"{m}__orig", f"{v}__cls", "d1", shuffle=True)
        for tag in ("d1", "d2s", "d2l"):
            add("D_random", rnd, f"{m}__orig", tag)
        for tag in ("d2s", "d2l"):
            add("D_random", rnd, f"{m}__gemini", tag)
    for v in EX.VITS:
        for s in (1, 2, 3):
            add("B2", f"{v}__cls", f"{v}__cls-aug{s}", "d1")
        add("D_shuffle", f"{v}__cls", f"{v}__cls-aug1", "d1", shuffle=True)
        add("D_random", rnd, f"{v}__cls", "d1")
    for a, b in itertools.combinations(llms, 2):
        add("W_llm", f"{a}__orig", f"{b}__orig", "d1")
    for a, b in itertools.combinations(EX.VITS, 2):
        add("W_vit", f"{a}__cls", f"{b}__cls", "d1")
    return specs


def pair_id(s: dict) -> str:
    return f"{s['group']}__{s['a']}__{s['b']}__{s['tag']}"


class Caches:
    """Alignment caches (kNN masks, centred Grams) on the device, least recently used evicted."""

    def __init__(self, root: Path, device: str, keep: int, k: int = 10):
        self.root, self.device, self.keep, self.k = root, device, keep, k
        self.data = OrderedDict()

    def get(self, key: str) -> tuple[lc.ModelCache, list[str]]:
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        while len(self.data) >= self.keep:  # evict before building: n = 10240 caches are ~4 GB
            self.data.popitem(last=False)
            torch.cuda.empty_cache()
        layers, names = prepared_layers(self.root / "features" / f"{key}.pt")
        runs = []  # consecutive layers of equal width (blocks, then the final embedding)
        for x in layers:
            if runs and runs[-1][-1].shape[1] == x.shape[1]:
                runs[-1].append(x)
            else:
                runs.append([x])
        parts = [lc.build_cache(torch.stack(r, 1).to(self.device), self.k) for r in runs]
        cache = lc.ModelCache(
            masks=torch.cat([p.masks for p in parts]),
            grams=torch.cat([p.grams for p in parts]),
            k=self.k,
        )
        self.data[key] = (cache, names)
        return cache, names


def permuted(c: lc.ModelCache, perm: torch.Tensor) -> lc.ModelCache:
    return lc.ModelCache(c.masks[:, perm][:, :, perm], c.grams[:, perm][:, :, perm], c.k)


def calibrate(observed: torch.Tensor, null: torch.Tensor, alpha: float) -> dict:
    """Per-cell calibration and the max over cells against the null of the max."""
    cells = lc.calibrate_cells(observed, null, alpha)
    mx = lc.calibrate_cells(
        observed.max().view(1, 1), null.flatten(1).max(1).values.view(-1, 1, 1), alpha
    )
    return {
        **{k: v.cpu().numpy() for k, v in cells.items()},
        **{f"max_{k}": float(v) for k, v in mx.items()},
    }


def bootstrap_excess(o: torch.Tensor, mu0: float, reps: int = 2000, seed: int = 0) -> list[float]:
    gen = torch.Generator(device=o.device).manual_seed(seed)
    idx = torch.randint(0, o.numel(), (reps, o.numel()), generator=gen, device=o.device)
    means = o[idx].mean(1) - mu0
    return [
        float(means.mean()),
        *torch.quantile(means, torch.tensor([0.025, 0.975], device=o.device)).tolist(),
    ]


def cell_summary(res: dict, cell: tuple[int, int]) -> dict:
    return {
        "cell": [int(c) for c in cell],
        **{k: float(res[k][cell]) for k in ("score", "mu0", "tau", "p", "g")},
    }


def align_pair(ca, cb, names_a, names_b, permutations, alpha, seed) -> tuple[dict, dict]:
    n = ca.masks.shape[1]
    gen = torch.Generator(device=ca.masks.device).manual_seed(seed)
    observed = lc.scores(ca, cb)
    nulls = {m: [] for m in lc.METRICS}
    for _ in range(permutations):
        perm = torch.randperm(n, generator=gen, device=ca.masks.device)
        for m, s in lc.scores(ca, cb, perm).items():
            nulls[m].append(s)
    arrays, summary = {}, {}
    for m in lc.METRICS:
        r = calibrate(observed[m], torch.stack(nulls[m]), alpha)
        for k in ("score", "mu0", "sd0", "tau", "p", "g"):
            arrays[f"{m}_{k}"] = r[k]
        best = np.unravel_index(int((r["score"] - r["mu0"]).argmax()), r["score"].shape)
        s = {
            "best": cell_summary(r, best),
            "max_g": r["max_g"],
            "max_p": r["max_p"],
            "max_tau": r["max_tau"],
            "frac_sig": float((r["p"] < 0.05).mean()),
        }
        cells = {"best": best}
        if names_a[-1] == "final" and names_b[-1] == "final":
            s["final"] = cell_summary(r, (len(names_a) - 1, len(names_b) - 1))
            cells["final"] = (len(names_a) - 1, len(names_b) - 1)
        if m == "mutual_knn":
            for which, (i, j) in cells.items():
                o = (ca.masks[i] * cb.masks[j]).sum(1) / ca.k
                s[which]["excess_ci"] = bootstrap_excess(o, float(r["mu0"][i, j]), seed=seed)
        summary[m] = s
    return arrays, summary


def rank_pair(za, zb, permutations, alpha, seed, names_a, names_b) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n = za.shape[1]
    obs = paired.rank_corr_matrix(za, zb)
    null = np.stack(
        [paired.rank_corr_matrix(za, zb, rng.permutation(n)) for _ in range(permutations)]
    )
    r = calibrate(torch.from_numpy(obs), torch.from_numpy(null), alpha)
    best = np.unravel_index(int(r["score"].argmax()), r["score"].shape)
    s = {
        "best": cell_summary(r, best),
        "max_g": r["max_g"],
        "max_p": r["max_p"],
        "max_tau": r["max_tau"],
        "frac_sig": float((r["p"] < 0.05).mean()),
    }
    if names_a[-1] == "final" and names_b[-1] == "final":
        s["final"] = cell_summary(r, (len(names_a) - 1, len(names_b) - 1))
    return {k: r[k] for k in ("score", "mu0", "tau", "p", "g")}, s


def fill_degenerate(orc: np.ndarray, dens: np.ndarray) -> tuple:
    """As exp-015: undefined ORC -> layer median, infinite density -> layer maximum."""
    orc, dens = orc.copy(), dens.copy()
    for i in range(orc.shape[0]):
        bad = ~np.isfinite(orc[i])
        if bad.any():
            orc[i, bad] = np.nanmedian(np.where(bad, np.nan, orc[i]))
        inf = ~np.isfinite(dens[i])
        if inf.any():
            dens[i, inf] = dens[i, ~inf].max() if (~inf).any() else 1.0
    return orc, dens


def stage_pairs(
    root: Path, device: str, permutations: int, alpha: float, keep_caches: int, only: str | None
) -> None:
    out = root / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    avail = {p.stem for p in (root / "features").glob("*.pt")}
    specs = pair_specs(avail)
    if only:
        specs = [s for s in specs if s["group"] in only.split(",")]
    # n = 10240 pairs last, grouped by A so caches are reused
    specs.sort(key=lambda s: (s["tag"] == "d2l", s["tag"], s["a"]))
    print(f"pairs: {len(specs)}", flush=True)
    caches = Caches(root, device, keep_caches)
    summary_path = root / "pairs.jsonl"
    done = (
        {json.loads(line)["id"] for line in summary_path.read_text().splitlines()}
        if summary_path.exists()
        else set()
    )
    t0 = time.time()
    for i, s in enumerate(specs):
        pid = pair_id(s) + ("__shuffled" if s["shuffle"] else "")
        if pid in done:
            continue
        tag = s["tag"]
        ca = cb = None  # so an evicted cache of the previous pair is freed
        ca, names_a = caches.get(f"{s['a']}__{tag}")
        cb, names_b = caches.get(f"{s['b']}__{tag}")
        n = ca.masks.shape[1]
        shuffle = (
            torch.from_numpy(np.random.default_rng(SHUFFLE_SEED).permutation(n))
            if s["shuffle"]
            else None
        )
        if shuffle is not None:
            cb = permuted(cb, shuffle.to(device))
        arrays, summary = align_pair(ca, cb, names_a, names_b, permutations, alpha, seed=i)
        row = {"id": pid, **s, "n": n, "names_a": names_a, "names_b": names_b, **summary}
        ga, gb = (
            root / "geometry" / f"{s['a']}__{tag}.npz",
            root / "geometry" / f"{s['b']}__{tag}.npz",
        )
        if tag != "d2l" and ga.exists() and gb.exists():
            za_, zb_ = np.load(ga), np.load(gb)
            perm = shuffle.numpy() if shuffle is not None else np.arange(n)
            keep = ~(za_["dup"] | zb_["dup"][perm])  # A's sample i is paired with B's perm[i]
            keep_b = perm[keep]
            names_ga, names_gb = list(za_["names"]), list(zb_["names"])
            row["orc_dropped"] = int((~keep).sum())
            for k in KS:
                oa, da = fill_degenerate(za_[f"orc_k{k}"][:, keep], za_[f"dens_k{k}"][:, keep])
                ob, db = fill_degenerate(zb_[f"orc_k{k}"][:, keep_b], zb_[f"dens_k{k}"][:, keep_b])
                ra, rb = paired.residualize(oa, da), paired.residualize(ob, db)
                arr, sm = rank_pair(ra, rb, permutations, alpha, i, names_ga, names_gb)
                arrays.update({f"orc_k{k}_{x}": v for x, v in arr.items()})
                row[f"orc_k{k}"] = sm
                za, zb = paired.standardized_ranks(da), paired.standardized_ranks(db)
                arr, sm = rank_pair(za, zb, permutations, alpha, i, names_ga, names_gb)
                arrays.update({f"dens_k{k}_{x}": v for x, v in arr.items()})
                row[f"dens_k{k}"] = sm
        np.savez(out / f"{pid}.npz", **arrays)
        with summary_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"[{i + 1}/{len(specs)}] {pid}: mkNN best {summary['mutual_knn']['best']['score']:.3f} "
            f"(null {summary['mutual_knn']['best']['mu0']:.3f}), CKA best {summary['cka_lin']['best']['score']:.3f}"
            + (
                f", ORC|dens best {row['orc_k10']['best']['score']:+.3f}"
                if "orc_k10" in row
                else ""
            )
            + f"  {time.time() - t0:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, required=True, help="exp-016 output dir (features/ inside)"
    )
    parser.add_argument("--stages", default="geometry,pairs")
    parser.add_argument("--groups", default=None, help="comma list of pair groups (default all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--keep-caches", type=int, default=6)
    parser.add_argument("--tmp", type=Path, default=Path("/tmp/exp016_layers"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stages = args.stages.split(",")
    if "geometry" in stages:
        stage_geometry(args.root, args.workers, args.tmp)
    if "pairs" in stages:
        stage_pairs(args.root, device, args.permutations, args.alpha, args.keep_caches, args.groups)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
