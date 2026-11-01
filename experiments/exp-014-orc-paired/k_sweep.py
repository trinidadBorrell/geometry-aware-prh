"""exp-014 k-sweep: how node ORC and its cross-model agreement change with neighbourhood size.

For 2 LLMs and 2 ViTs (PRH wit_1024, PRH features) at relative depths 0.1, 0.5 and 0.9, and
k in --ks (5 to 200 of n = 1024): node ORC (exact, PersonaManifold convention, alpha = 0) and
local density; the flat-torus reference (uniform Clifford torus, same n, k and rounded TwoNN ID);
and, for the within-modality pairs (LLM x LLM, ViT x ViT) at matching depths, the Spearman
across samples of ORC given density, with a 200-permutation null.

Raw ORC rises with k for any data (neighbourhoods overlap more; in the limit of a complete graph
kappa -> 1), so the readouts are the gap to the flat reference, ORC(k) - ORC_flat(k), which
measures how much more locally clustered the data is than uniform sampling at that scale, and the
paired agreement, which says at which scale two models agree most on per-sample ORC.

    uv run python experiments/exp-014-orc-paired/k_sweep.py --out <dir> --workers 28
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MODELS = {
    "language": ["bigscience/bloomz-7b1", "huggyllama/llama-13b"],
    "vision": ["vit_giant_patch14_dinov2.lvd142m", "vit_huge_patch14_clip_224.laion2b"],
}
DEPTHS = (0.1, 0.5, 0.9)
DROP = [493, 531]  # exact-duplicate caption pair in wit_1024


def _exp012():
    path = Path(__file__).resolve().parents[1] / "exp-012-id-mismatch" / "run.py"
    spec = importlib.util.spec_from_file_location("exp012_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _single_thread():
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def layer_task(args):
    path, k = args
    from geoprh import geometry as geo

    x = np.load(path).astype(np.float64)
    out = geo.ollivier_ricci_exact(geo.knn_graph(x, k))
    return path, k, out["node"], 1.0 / geo.knn_distance(x, k)


def torus_task(args):
    d, k, n = args
    from geoprh import geometry as geo

    rng = np.random.default_rng(d)
    t = rng.uniform(0, 2 * np.pi, size=(n, d))
    x = np.concatenate([np.cos(t), np.sin(t)], axis=1)
    return d, k, float(np.median(geo.ollivier_ricci_exact(geo.knn_graph(x, k))["node"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--tmp", type=Path, default=Path("/tmp/exp014_ksweep"))
    parser.add_argument("--ks", default="5,10,20,50,100,200")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--llms", default=",".join(MODELS["language"]), help="two LLMs")
    parser.add_argument("--vits", default=",".join(MODELS["vision"]), help="two ViTs")
    args = parser.parse_args()
    import torch.nn.functional as F

    from aristotelian.prh.preprocess import prepare_features
    from geoprh import activation_cache as ac
    from geoprh import intrinsic_dim as idim
    from geoprh import paired

    e12 = _exp012()
    models = {"language": args.llms.split(","), "vision": args.vits.split(",")}
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    (args.out / "meta_ksweep.json").write_text(json.dumps({"git_sha": sha.strip(), "ks": ks}))

    layers, tasks = {}, []
    for group in models.values():
        for m in group:
            payload = ac.load_model(root, m, e12.variant(m))
            feats = F.normalize(prepare_features(payload["feats"], q=0.95, exact=False), dim=-1)
            n_layers = feats.shape[1]
            for depth in DEPTHS:
                la = round(depth * (n_layers - 1))
                f = args.tmp / f"{m.replace('/', '__')}_{la:02d}.npy"
                x = feats[:, la].numpy()
                np.save(f, x)
                layers[(m, depth)] = {"layer": la, "path": str(f), "twonn": idim.two_nn(x)["id"]}
                tasks += [(str(f), k) for k in ks]
            print(f"{m}: prepared", flush=True)
    # largest k first, so the slowest tasks start early
    tasks.sort(key=lambda t: -t[1])
    n = 1024
    ref_tasks = sorted(
        {(int(np.rint(v["twonn"])), k, n) for v in layers.values() for k in ks}, key=lambda t: -t[1]
    )
    res, refs = {}, {}
    with ProcessPoolExecutor(args.workers, initializer=_single_thread) as pool:
        for path, k, node, dens in pool.map(layer_task, tasks):
            res[(path, k)] = (node, dens)
            print(f"done {Path(path).stem} k={k}", flush=True)
        for d, k, v in pool.map(torus_task, ref_tasks):
            refs[(d, k)] = v

    keep = np.setdiff1d(np.arange(n), DROP)
    out = {"levels": [], "pairs": []}
    for (m, depth), v in layers.items():
        d = int(np.rint(v["twonn"]))
        for k in ks:
            node = res[(v["path"], k)][0][keep]
            out["levels"].append(
                {
                    "model": m,
                    "depth": depth,
                    "layer": v["layer"],
                    "twonn": v["twonn"],
                    "k": k,
                    "median_orc": float(np.median(node)),
                    "flat_ref": refs[(d, k)],
                    "gap": float(np.median(node) - refs[(d, k)]),
                }
            )
    for modality, (a, b) in models.items():
        for depth in DEPTHS:
            va, vb = layers[(a, depth)], layers[(b, depth)]
            for k in ks:
                oa, da = (z[keep][None] for z in res[(va["path"], k)])
                ob, db = (z[keep][None] for z in res[(vb["path"], k)])
                r = paired.calibrated_rank_corr(
                    paired.residualize(oa, da), paired.residualize(ob, db)
                )
                r2 = paired.calibrated_rank_corr(
                    paired.standardized_ranks(da), paired.standardized_ranks(db)
                )
                out["pairs"].append(
                    {
                        "modality": modality,
                        "a": a,
                        "b": b,
                        "depth": depth,
                        "k": k,
                        "orc_given_density": float(r["score"][0, 0]),
                        "tau": float(r["tau"][0, 0]),
                        "density": float(r2["score"][0, 0]),
                    }
                )
    (args.out / "k_sweep.json").write_text(json.dumps(out, indent=1))
    for p in out["pairs"]:
        print(
            f"{p['modality']} depth {p['depth']} k={p['k']}: ORC|dens {p['orc_given_density']:+.3f} "
            f"(tau {p['tau']:.3f}), density {p['density']:+.3f}"
        )


if __name__ == "__main__":
    main()
