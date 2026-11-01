"""exp-014 per-sample check: does ORC agreement carry information beyond shared neighbours?

For a model pair at the layer pair where density-corrected ORC agrees most (exp-014 `pairs/`),
per sample s:
- o(s) = |N_A(s) & N_B(s)| / k, the share of s's k nearest neighbours the two models have in
  common (the per-sample form of mutual kNN; same Euclidean kNN on PRH features as the ORC graph)
- node ORC and local density in each model (exp-014 `models/`)

Statistics (Spearman across samples, rank scale):
- orc|dens:     ORC agreement with each model's density removed (as in analyze.py)
- orc|dens,o:   the same with o(s) also removed from both sides (partial)
- by stratum:   orc|dens agreement within samples with o = 0 (no shared neighbour), 0 < o <= 0.3,
                and o > 0.3, each with a within-stratum permutation null (200 shuffles)
If ORC agrees even where the models share none of a sample's neighbours, it carries alignment
information that mutual kNN cannot see.

Runs on the models cached locally (results/activations): 3 BLOOMZ sizes and 4 ViTs.

    uv run python experiments/exp-014-orc-paired/per_sample.py --root results/exp-014-orc-paired \\
        --cache results/activations
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from aristotelian.prh.preprocess import prepare_features
from geoprh import activation_cache as ac

DROP = [493, 531]  # exact-duplicate caption pair in wit_1024 (see README)
STRATA = {"o = 0": (-1.0, 0.0), "0 < o <= 0.3": (0.0, 0.3), "o > 0.3": (0.3, 1.0)}


def _exp012():
    path = Path(__file__).resolve().parents[1] / "exp-012-id-mismatch" / "run.py"
    spec = importlib.util.spec_from_file_location("exp012_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def zrank(v: np.ndarray) -> np.ndarray:
    r = rankdata(v)
    return (r - r.mean()) / r.std()


def resid(y: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Standardized residual of rank(y) after least squares on the ranks of covs."""
    x = np.column_stack([np.ones(len(y))] + [zrank(c) for c in covs])
    zy = zrank(y)
    r = zy - x @ np.linalg.lstsq(x, zy, rcond=None)[0]
    return r / r.std()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def perm_p(a: np.ndarray, b: np.ndarray, rng, n_perm: int = 200) -> float:
    obs = corr(a, b)
    null = np.array([corr(a, rng.permutation(b)) for _ in range(n_perm)])
    return float((1 + (null >= obs).sum()) / (n_perm + 1))


def knn_sets(x: np.ndarray, k: int) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ks", default="10,30")
    args = parser.parse_args()
    e12 = _exp012()
    rng = np.random.default_rng(0)
    slug = lambda m: m.replace("/", "__")  # noqa: E731
    local = [m for m in e12.LLMS + e12.VITS if (args.cache / slug(m)).exists()]
    feats = {}
    for m in local:
        payload = ac.load_model(args.cache, m, e12.variant(m))
        feats[m] = F.normalize(prepare_features(payload["feats"], q=0.95, exact=False), dim=-1)
    pairs = [
        ("T1", a, b) for a, b in itertools.combinations([m for m in local if m in e12.LLMS], 2)
    ]
    pairs += [
        ("T2", a, b) for a, b in itertools.combinations([m for m in local if m in e12.VITS], 2)
    ]
    keep = np.setdiff1d(np.arange(1024), DROP)

    recs, lines = [], []
    for k in (int(v) for v in args.ks.split(",")):
        for track, a, b in pairs:
            best = json.loads(
                next(
                    line
                    for line in (args.root / "pairs.jsonl").read_text().splitlines()
                    if f'"a": "{a}", "b": "{b}"' in line
                )
            )[f"orc_dens_k{k}"]["best"]
            la, lb = best
            za, zb = (
                np.load(args.root / "models" / f"{slug(a)}.npz"),
                np.load(args.root / "models" / f"{slug(b)}.npz"),
            )
            na = knn_sets(feats[a][:, la].numpy(), k)
            nb = knn_sets(feats[b][:, lb].numpy(), k)
            o = np.array([len(np.intersect1d(na[s], nb[s])) / k for s in range(len(na))])[keep]
            oa, ob = za[f"orc_k{k}"][la][keep], zb[f"orc_k{k}"][lb][keep]
            da, db = za[f"dens_k{k}"][la][keep], zb[f"dens_k{k}"][lb][keep]
            ra, rb = resid(oa, [da]), resid(ob, [db])
            rec = {"k": k, "track": track, "a": a, "b": b, "layers": [la, lb]}
            rec["mean_overlap"] = float(o.mean())
            rec["orc|dens"] = corr(ra, rb)
            rec["orc|dens,o"] = corr(resid(oa, [da, o]), resid(ob, [db, o]))
            rec["strata"] = {}
            for name, (lo, hi) in STRATA.items():
                m = (o > lo) & (o <= hi)
                if m.sum() >= 30:
                    rec["strata"][name] = {
                        "n": int(m.sum()),
                        "r": corr(ra[m], rb[m]),
                        "p": perm_p(ra[m], rb[m], rng),
                    }
            recs.append(rec)
            st = "  ".join(
                f"{n}: {v['r']:+.2f} (n {v['n']}, p {v['p']:.3f})" for n, v in rec["strata"].items()
            )
            lines.append(
                f"k={k} {track} {a.split('/')[-1][:24]:<24} x {b.split('/')[-1][:24]:<24} "
                f"L{la}/{lb} overlap {o.mean():.2f} | orc|dens {rec['orc|dens']:+.2f} "
                f"orc|dens,o {rec['orc|dens,o']:+.2f} | {st}"
            )
            print(lines[-1], flush=True)
    (args.root / "per_sample.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (args.root / "per_sample.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
