"""exp-014 step 2: per-layer node Ollivier-Ricci curvature and density for every PRH model.

Data: the PRH `wit_1024` activations of the 10 LLMs and 17 ViTs (as exp-012), PRH preprocessing
(q = 0.95 clamp, l2 norm). Per model, layer and k in --ks: the union kNN graph with Euclidean
edges, node ORC with PersonaManifold's convention (`geoprh.geometry.ollivier_ricci_exact`,
alpha = 0, node = mean over incident edges), the local density 1 / (distance to the k-th
neighbour), and the TwoNN ID. Reference: the mean node ORC of a uniformly sampled flat torus
T^d (Clifford, no boundary) with the same n, k and d = the layer's TwoNN ID (rounded), i.e. the
ORC level expected with no curvature at that ID.

Layers run in parallel (one process per layer and k); outputs one .npz per model under
`models/` (orc_k{k}, dens_k{k}: [L, n]; twonn: [L]) and `torus_ref.json`.

    uv run python experiments/exp-014-orc-paired/compute.py --out <dir> --workers 28
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

N_TORUS_SEED = 0


def _exp012():
    path = Path(__file__).resolve().parents[1] / "exp-012-id-mismatch" / "run.py"
    spec = importlib.util.spec_from_file_location("exp012_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _single_thread():
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def layer_task(args) -> tuple:
    path, k, want_id = args
    from geoprh import geometry as geo
    from geoprh import intrinsic_dim as idim

    x = np.load(path).astype(np.float64)
    out = geo.ollivier_ricci_exact(geo.knn_graph(x, k))
    twonn = idim.two_nn(x)["id"] if want_id else None
    return path, k, out["node"], 1.0 / geo.knn_distance(x, k), twonn


def torus_task(args) -> tuple:
    d, k, n = args
    from geoprh import geometry as geo

    rng = np.random.default_rng(N_TORUS_SEED + d)
    t = rng.uniform(0, 2 * np.pi, size=(n, d))
    x = np.concatenate([np.cos(t), np.sin(t)], axis=1)
    return d, k, float(geo.ollivier_ricci_exact(geo.knn_graph(x, k))["node"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--tmp", type=Path, default=Path("/tmp/exp014_layers"))
    parser.add_argument("--ks", default="10,30")
    parser.add_argument("--models", default="all", help="comma list or all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--llm-pool", default="avg", help="avg (cached PRH) or last (exp-015)")
    parser.add_argument("--vit-pool", default="cls", help="cls (cached PRH) or avg (exp-015)")
    args = parser.parse_args()
    import torch.nn.functional as F

    from aristotelian.prh.preprocess import prepare_features
    from geoprh import activation_cache as ac

    e12 = _exp012()
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    meta = {**{k: str(v) for k, v in vars(args).items()}, "git_sha": sha.strip(), "alpha": 0.0}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    models = e12.LLMS + e12.VITS if args.models == "all" else args.models.split(",")
    layout, tasks = {}, []
    for model in models:
        var = e12.variant(model)
        pool = args.llm_pool if model in e12.LLMS else args.vit_pool
        var = ac.Variant(var.dataset, var.subset, pool, var.caption_idx, var.modality)
        payload = ac.load_model(root, model, var)
        if payload is None:
            print(f"[skip] {model}: not cached", flush=True)
            continue
        feats = F.normalize(prepare_features(payload["feats"], q=0.95, exact=False), dim=-1)
        files = []
        for layer in range(feats.shape[1]):
            f = args.tmp / f"{model.replace('/', '__')}_{layer:02d}.npy"
            np.save(f, feats[:, layer].numpy())
            files.append(str(f))
            tasks += [(str(f), k, k == ks[0]) for k in ks]
        layout[model] = files
        print(f"{model}: {tuple(feats.shape)} prepared", flush=True)
        del payload, feats

    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(args.workers, initializer=_single_thread) as pool:
        for i, (path, k, node, dens, twonn) in enumerate(pool.map(layer_task, tasks, chunksize=1)):
            res[(path, k)] = (node, dens, twonn)
            if i % 50 == 0:
                print(f"{i + 1}/{len(tasks)} layer tasks, {time.time() - t0:.0f}s", flush=True)

    refs_needed = set()
    for model, files in layout.items():
        twonn = np.array([res[(f, ks[0])][2] for f in files])
        arrays = {"twonn": twonn}
        for k in ks:
            arrays[f"orc_k{k}"] = np.stack([res[(f, k)][0] for f in files])
            arrays[f"dens_k{k}"] = np.stack([res[(f, k)][1] for f in files])
            refs_needed |= {(int(d), k) for d in np.rint(twonn).astype(int)}
        np.savez(args.out / "models" / f"{model.replace('/', '__')}.npz", **arrays)
    n = len(next(iter(res.values()))[0])
    with ProcessPoolExecutor(args.workers, initializer=_single_thread) as pool:
        refs = list(pool.map(torus_task, [(d, k, n) for d, k in sorted(refs_needed)]))
    (args.out / "torus_ref.json").write_text(
        json.dumps({f"d{d}_k{k}": v for d, k, v in refs}, indent=1)
    )
    print(f"done: {len(layout)} models, {len(refs)} torus references, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
