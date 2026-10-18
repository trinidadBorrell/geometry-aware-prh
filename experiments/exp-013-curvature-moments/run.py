"""exp-013: intrinsic vs extrinsic curvature from neighbourhood moments (`geoprh.curvature`).

--mode synthetic: every manifold of `geoprh.synthetic` x ID x n x noise, embedded by a random
isometry into 1024 dimensions (noise as in exp-011: isotropic, norm = --noise x the median
nearest-neighbour distance). Per dataset: TwoNN ID, global linear dimension (PCA90, participation
ratio), and the curvature estimate with the true ID and with the TwoNN ID (rounded): the pooled
estimate over the three smallest neighbourhood sizes and the scale profile.

--mode models: the 8 models with 10240 cached WIT-1M samples (as exp-010: LLM mean-pooled
caption, every hidden state; ViT CLS, every block), PRH features (q = 0.95 clamp, l2 norm); per
layer the TwoNN ID (rounded, used as d), PCA90, and the curvature estimate.

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

KS = [10, 20, 40, 80, 160, 320, 640]
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


def _curvature(x: np.ndarray, d: int, seed: int) -> dict:
    from geoprh import curvature as cv

    d = max(int(d), 1)
    mom = cv.neighbourhood_moments(x, d, KS, n_centres=N_CENTRES, seed=seed)
    return {
        "d": d,
        "pooled": cv.curvature_from_moments(mom, d, n_scales=N_POOLED),
        "profile": cv.curvature_profile(mom, d),
    }


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
    rec["true_d"] = _curvature(x, d, seed)
    rec["est_d"] = _curvature(x, round(rec["id_twonn"]), seed)
    rec["seconds"] = time.time() - t0
    return rec


def model_task(args) -> list[dict]:
    model, root, seed = args
    import torch
    import torch.nn.functional as F

    from aristotelian.prh.preprocess import prepare_features
    from geoprh import activation_cache as ac
    from geoprh import intrinsic_dim as idim

    torch.set_num_threads(1)
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
        return [{"model": model, "error": "not cached"}]
    feats = F.normalize(prepare_features(payload["feats"], q=0.95, exact=False), dim=-1)
    out = []
    for layer in range(feats.shape[1]):
        t0 = time.time()
        x = feats[:, layer].double().numpy()
        twonn = idim.two_nn(x)["id"]
        rec = {"model": model, "modality": var.modality, "layer": layer}
        rec.update(num_layers=feats.shape[1], dim=feats.shape[2], n=feats.shape[0])
        rec["id_twonn"] = twonn
        rec["pca"] = idim.pca_dim(x)
        rec["est_d"] = _curvature(x, round(twonn), seed)
        rec["seconds"] = time.time() - t0
        out.append(rec)
        print(
            f"{model} layer {layer}: d={round(twonn)} S_norm "
            f"{rec['est_d']['pooled']['S_norm']:+.3f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["synthetic", "models"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifolds", default="flat,sphere,swiss,torus_rff,veronese,tree,clusters")
    parser.add_argument("--dims", default="2,5,20,40")
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--noise", default="0,0.1")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--root", default=None, help="activation cache root")
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
                f"{r['seconds']:.0f}s S_norm {r['true_d']['pooled']['S_norm']:+.3f} "
                f"(true {r['scalar_true']}) PCA90/ID {r['pca']['n90'] / r['d']:.1f}"
            ),
        )
    else:
        tasks = [(m, args.root, args.seed) for m in args.models.split(",")]
        run, fmt = model_task, lambda r: f"{r[0]['model']}: {len(r)} layers"

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_single_thread)
    with path.open("a") as fout, pool:
        for res in pool.map(run, tasks):
            for r in res if isinstance(res, list) else [res]:
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            print(fmt(res), flush=True)


if __name__ == "__main__":
    main()
