"""exp-013: intrinsic vs extrinsic curvature from neighbourhood moments (`geoprh.curvature`).

--mode synthetic: every manifold of `geoprh.synthetic` x ID x n x noise, embedded by a random
isometry into 1024 dimensions (noise as in exp-011: isotropic, norm = --noise x the median
nearest-neighbour distance). Per dataset: TwoNN ID, global linear dimension (PCA90, participation
ratio), and the curvature estimate (pooled over the three smallest usable neighbourhood sizes,
plus the scale profile) at the tangent dimension chosen from the data (`select_tangent_dim`), at
the true ID and at the TwoNN ID; and S / d(d-1) for every candidate d (0.7x to 2x TwoNN).

--mode models: the 8 models with 10240 cached WIT-1M samples (as exp-010: LLM mean-pooled
caption, every hidden state; ViT CLS, every block), PRH features (q = 0.95 clamp, l2 norm); per
layer the TwoNN ID, PCA90, and the curvature estimate at the chosen d and at the TwoNN d.

    uv run python experiments/exp-013-curvature-moments/run.py --mode synthetic --out <dir> --workers 8
    uv run python experiments/exp-013-curvature-moments/run.py --mode models --out <dir> --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KS = [10, 20, 40, 80, 160, 320, 640, 1000]
N_CENTRES = 300
N_POOLED = 3  # scales in the pooled estimate
AMBIENT = 1024
MODELS = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
]


def _single_thread():
    """Workers run single-threaded; the BLAS variables only take effect if set before numpy is
    imported, so the launcher (infra/runpod/run_exp013.sh) exports them too."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def candidate_ds(twonn: float, extra: tuple[int, ...] = ()) -> list[int]:
    """Candidate tangent dimensions: 0.7x to 2x the TwoNN ID (at most 120), about 15 values."""
    lo, hi = max(2, round(0.7 * twonn)), min(120, max(3, round(2.0 * twonn)))
    step = max(1, round((hi - lo) / 15))
    return sorted(set(range(lo, hi + 1, step)) | {min(120, max(2, round(twonn)))} | set(extra))


def _curvature(x: np.ndarray, ds: list[int], seed: int, keep: dict[str, int]) -> dict:
    """Moments for every candidate d; the data-driven d (`select_tangent_dim`); the pooled
    estimate and scale profile at that d and at each d named in `keep`; S/d(d-1) for every d."""
    from geoprh import curvature as cv

    mom = cv.neighbourhood_moments(x, ds, KS, n_centres=N_CENTRES, seed=seed)
    sel = cv.select_tangent_dim(mom)
    out = {"ds": ds, "k": mom["k"], "leaked": sel["leaked"]}
    out["S_by_d"] = {d: cv.curvature_from_moments(mom, d, N_POOLED)["S_norm"] for d in ds}
    for name, d in {"chosen": sel["d"], **keep}.items():
        out[name] = {
            "d": d,
            "pooled": cv.curvature_from_moments(mom, d, n_scales=N_POOLED),
            "profile": cv.curvature_profile(mom, d),
        }
    return out


def synthetic_task(args) -> dict:
    manifold, d, n, noise, seed = args
    from sklearn.neighbors import NearestNeighbors

    from geoprh import intrinsic_dim as idim
    from geoprh import synthetic as syn

    t0 = time.time()
    rng = np.random.default_rng(seed)
    z, truth = syn.MANIFOLDS[manifold](n, d, rng)
    x = syn.embed(z, max(AMBIENT, z.shape[1]), rng)
    del z
    if noise > 0:
        nn = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x[: min(n, 2000)])[0][:, 1]
        x += noise * np.median(nn) / np.sqrt(x.shape[1]) * rng.standard_normal(x.shape)
    sub = x[: min(n, 10000)]
    rec = {"manifold": manifold, "d": d, "n": n, "noise": noise, "seed": seed}
    rec["scalar_true"] = truth["scalar"]
    rec["id_twonn"] = idim.two_nn(sub)["id"]
    rec["pca"] = idim.pca_dim(x[: min(n, 5000)])
    ds = candidate_ds(rec["id_twonn"], extra=(d,) if d <= 120 else ())
    rec["curv"] = _curvature(x, ds, seed, {"true": d, "twonn": min(120, round(rec["id_twonn"]))})
    rec["seconds"] = time.time() - t0
    return rec


def prepare_model_layers(model: str, root, tmp: Path) -> list[tuple]:
    """Load one model, apply the PRH preprocessing (the q = 0.95 clamp is computed over all of a
    sample's layers, so it needs the whole model), and write one float32 .npy per layer, so that
    the layers can be processed in parallel. Returns the per-layer tasks."""
    import torch.nn.functional as F

    from aristotelian.prh.preprocess import prepare_features
    from geoprh import activation_cache as ac

    language = "/" in model
    var = ac.Variant(
        dataset="askoepke/wit_1m_recaptioned",
        subset="first10240-original",
        pool="avg" if language else "cls",
        caption_idx=None,
        modality="language" if language else "vision",
    )
    payload = ac.load_model(Path(root) if root else ac.default_root(), model, var)
    if payload is None:
        print(f"[skip] {model}: not cached", flush=True)
        return []
    feats = F.normalize(prepare_features(payload["feats"], q=0.95, exact=False), dim=-1)
    tasks = []
    for layer in range(feats.shape[1]):
        f = tmp / f"{model.replace('/', '__')}_{layer:02d}.npy"
        np.save(f, feats[:, layer].numpy())
        tasks.append((model, var.modality, layer, feats.shape[1], str(f)))
    return tasks


def layer_task(args) -> dict:
    model, modality, layer, num_layers, path = args
    from geoprh import intrinsic_dim as idim

    t0 = time.time()
    x = np.load(path).astype(np.float64)
    twonn = idim.two_nn(x)["id"]
    rec = {"model": model, "modality": modality, "layer": layer, "num_layers": num_layers}
    rec.update(dim=x.shape[1], n=x.shape[0], id_twonn=twonn, pca=idim.pca_dim(x))
    rec["curv"] = _curvature(x, candidate_ds(twonn), 0, {"twonn": round(twonn)})
    rec["seconds"] = time.time() - t0
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["synthetic", "models"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifolds", default="flat,sphere,swiss,torus_rff,veronese,tree,clusters")
    parser.add_argument("--dims", default="5,20,40")
    parser.add_argument("--sizes", default="10000,100000")
    parser.add_argument("--noise", default="0,0.1")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--root", default=None, help="activation cache root")
    parser.add_argument("--tmp", default="/tmp/exp013_layers", help="per-layer feature files")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _single_thread()

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    meta = {**{k: str(v) for k, v in vars(args).items()}, "git_sha": sha.strip()}
    meta.update(ks=KS, n_centres=N_CENTRES, n_pooled=N_POOLED, ambient=AMBIENT)
    (args.out / f"meta_{args.mode}.json").write_text(json.dumps(meta, indent=2))
    path = args.out / f"{args.mode}.jsonl"

    if args.mode == "synthetic":
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done.add((r["manifold"], r["d"], r["n"], r["noise"]))
        tasks = [
            (m, int(d), int(n), float(nz), args.seed + i)
            for i, (n, d, m, nz) in enumerate(
                (n, d, m, nz)
                for n in args.sizes.split(",")
                for d in args.dims.split(",")
                for m in args.manifolds.split(",")
                for nz in args.noise.split(",")
            )
        ]
        tasks = [t for t in tasks if t[:4] not in done]
        run, fmt = (
            synthetic_task,
            lambda r: (
                f"{r['manifold']:<9} d={r['d']:<3} n={r['n']:<6} noise={r['noise']}: "
                f"{r['seconds']:.0f}s chosen d {r['curv']['chosen']['d']} S_norm "
                f"{r['curv']['chosen']['pooled']['S_norm']:+.3f} (at true d "
                f"{r['curv']['true']['pooled']['S_norm']:+.3f}; true {r['scalar_true']})"
            ),
        )
    else:
        tmp = Path(args.tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        tasks = [t for m in args.models.split(",") for t in prepare_model_layers(m, args.root, tmp)]
        run, fmt = (
            layer_task,
            lambda r: (
                f"{r['model']} layer {r['layer']}: {r['seconds']:.0f}s 2NN {r['id_twonn']:.1f} "
                f"chosen d {r['curv']['chosen']['d']} S_norm "
                f"{r['curv']['chosen']['pooled']['S_norm']:+.3f}"
            ),
        )

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_single_thread)
    with path.open("a") as fout, pool:
        for res in pool.map(run, tasks):
            for r in res if isinstance(res, list) else [res]:
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            print(fmt(res), flush=True)


if __name__ == "__main__":
    main()
