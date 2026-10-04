"""exp-010: Levina-Bickel intrinsic dimension per layer, vs number of samples.

Data (--data), LLM: every hidden state mean-pooled over the caption; ViT: every block's CLS token:
- wit1m10k: exp-006's cached activations of the first 10240 WIT-1M samples
  (`askoepke_wit_1m_recaptioned_first10240-original_pool-{avg,cls}`), 8 models
- prh1024:  the PRH `wit_1024` set (`minhuh_prh_wit_1024_pool-{avg_cid-0,cls}`), every model of
  the val grid (10 LLMs, 17 ViTs)

Sample sets: for each n in --sizes, the disjoint row blocks [n s, n (s+1)) that fit in the data,
e.g. 10 x 1000, 5 x 2000, 2 x 5000, 1 x 10000 of 10240. Every set gives one estimate per layer, so
the spread at each n is the variability across sample sets.

Geometries (--preps):
- raw: the activations as cached
- prh: the PRH preprocessing the alignment metrics see (platonic-rep / exp-006): q = 0.95
  outlier clamp over the sample's layers (Aristotelian `prepare_features`), then l2 norm

Estimators (--estimator), `geoprh.intrinsic_dim` via scikit-dimension; exact duplicate rows
dropped first (`n_unique` in the output):
- lb:    Levina & Bickel 2004 Eqs. 8-9 (`MLE`), averaged over k = 10..20
- twonn: TwoNN (Facco et al. 2017) as DADApy's `compute_id_2NN`, used by Valeriani et al. 2023
- pca:   linear dimension, number of PCs for 90% variance ("id") and participation ratio ("pr")

    uv run python experiments/exp-010-intrinsic-dim/run.py --out <dir> \\
        --models bigscience/bloomz-560m
    uv run python experiments/exp-010-intrinsic-dim/run.py --out <dir> --data prh1024 \\
        --models all --sizes 1024
    uv run python experiments/exp-010-intrinsic-dim/run.py --out <dir> --inventory
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from aristotelian.prh.preprocess import prepare_features
from geoprh import activation_cache as ac
from geoprh import intrinsic_dim as idim

DATA = {  # name: (dataset, subset, caption_idx of the text features)
    "wit1m10k": ("askoepke/wit_1m_recaptioned", "first10240-original", None),
    "prh1024": ("minhuh/prh", "wit_1024", 0),
}


def variant(model: str, data: str = "wit1m10k") -> ac.Variant:
    # HF language models are "org/name", timm vision models have no "/"
    language = "/" in model
    dataset, subset, cid = DATA[data]
    return ac.Variant(
        dataset=dataset,
        subset=subset,
        pool="avg" if language else "cls",
        caption_idx=cid if language else None,
        modality="language" if language else "vision",
    )


def layers(feats: torch.Tensor, prep: str) -> torch.Tensor:
    """[n, L, d] features in the requested geometry."""
    if prep == "raw":
        return feats.float()
    if prep == "prh":
        return F.normalize(prepare_features(feats, q=0.95, exact=False), dim=-1)
    raise ValueError(prep)


def inventory(root: Path) -> list[dict]:
    """Every cached (model, dataset key): samples, layers, dim."""
    rows = []
    for meta in sorted(root.glob("*/layer_00/*.meta.json")):
        m = json.loads(meta.read_text())
        n_layers = sum(1 for d in meta.parent.parent.glob("layer_*") if (d / meta.name).exists())
        rows.append(
            {
                "model": m["model"],
                "modality": m["modality"],
                "key": meta.name.removesuffix(".meta.json"),
                "n": m["shape"][0],
                "dim": m["shape"][1],
                "num_layers": m["num_layers"],
                "layers_cached": n_layers,
                "num_params": m.get("num_params"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--data", choices=list(DATA), default="wit1m10k")
    parser.add_argument("--models", default="bigscience/bloomz-560m", help="comma list or all")
    parser.add_argument("--sizes", default="1000,2000,5000,10000")
    parser.add_argument("--preps", default="raw,prh")
    parser.add_argument("--estimator", choices=list(idim.ESTIMATORS), default="lb")
    parser.add_argument("--k1", type=int, default=10)
    parser.add_argument("--k2", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--inventory", action="store_true", help="only list the cache")
    args = parser.parse_args()
    root = args.root or ac.default_root()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    n_jobs = args.threads or 1  # sklearn kNN search

    if args.inventory:
        rows = inventory(root)
        (args.out / "inventory.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(f"{r['model']:<50} {r['key']:<60} n={r['n']:<6} L={r['layers_cached']}")
        return

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    meta = {**{k: str(v) for k, v in vars(args).items()}, "git_sha": sha.strip()}
    meta["threads"] = torch.get_num_threads()
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    sizes = [int(s) for s in args.sizes.split(",")]
    if args.models == "all":  # every model with this data cached
        rows = inventory(root)
        models = [r["model"] for r in rows if r["key"] == variant(r["model"], args.data).key]
    else:
        models = args.models.split(",")
    with (args.out / "intrinsic_dim.jsonl").open("a") as fout:
        for model in models:
            var = variant(model, args.data)
            payload = ac.load_model(root, model, var)
            if payload is None:
                print(f"[skip] {model}: not fully cached ({var.key})")
                continue
            feats = payload["feats"]
            print(f"{model}: feats {tuple(feats.shape)}", flush=True)
            for n in sizes:
                for s in range(feats.shape[0] // n):
                    for prep in args.preps.split(","):
                        t0 = time.time()
                        x = layers(feats[n * s : n * (s + 1)], prep)
                        for layer in range(x.shape[1]):
                            xl = x[:, layer].numpy()
                            if args.estimator == "lb":
                                est = idim.levina_bickel(xl, args.k1, args.k2, n_jobs=n_jobs)
                            elif args.estimator == "twonn":
                                est = idim.two_nn(xl, n_jobs=n_jobs)
                            else:
                                est = idim.pca_dim(xl)
                            rec = {
                                "model": model,
                                "data": args.data,
                                "estimator": args.estimator,
                                "modality": var.modality,
                                "num_layers": x.shape[1],
                                "layer": layer,
                                "dim": x.shape[2],
                                "n": n,
                                "set": s,
                                "prep": prep,
                                **est,
                            }
                            fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        print(
                            f"  n={n} set={s} {prep}: {time.time() - t0:.1f}s "
                            f"(last layer id {est['id']:.1f})",
                            flush=True,
                        )
            del feats, payload


if __name__ == "__main__":
    main()
