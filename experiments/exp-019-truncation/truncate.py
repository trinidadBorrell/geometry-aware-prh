"""exp-019: cross-model alignment as a function of dimension truncation (random and PCA).

For every model pair and every truncation setting (operator, size m, random draw), both models are
truncated the same way (`geoprh.truncation`), and mutual kNN (k = 10) and linear CKA are computed
for every layer pair, calibrated per cell against a permutation null of the sample pairing, with
the max over cells against the null of the max. The same permutations are used at every setting.
Per model and setting, self-retention (truncated layer vs the same full layer) is recorded too, so
that cross-model curves can be read against what truncation does to each model on its own.

    uv run python experiments/exp-019-truncation/truncate.py synthetic --out results/exp-019-truncation/synthetic
    uv run python experiments/exp-019-truncation/truncate.py real --root results/activations --out results/exp-019-truncation/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from geoprh import activation_cache as ac
from geoprh import truncation as tr

REPO = Path(__file__).resolve().parents[2]
PCA_M = (
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
)  # m = 1 excluded: l2 on one coordinate is a sign, ties broken by index fake agreement
DROP_M = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RAND_M = (2, 4, 8, 16, 32, 64, 128, 256, 512)
PILOT_LLMS = ("bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7")
PILOT_VITS = (
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
)
SHUFFLE_SEED = 12345


def settings(draws: int) -> list[tuple[str, int, int]]:
    out = [("prh", 0, 0), ("none", 0, 0)]
    out += [("top", m, 0) for m in PCA_M]
    out += [("drop_top", m, 0) for m in DROP_M]
    out += [(op, m, d) for op in ("rand_coord", "rand_proj") for m in RAND_M for d in range(draws)]
    return out


def _gen(name: str, op: str, m: int, draw: int) -> torch.Generator:
    h = int(hashlib.sha256(f"{name}|{op}|{m}|{draw}".encode()).hexdigest()[:12], 16)
    return torch.Generator().manual_seed(h)


def _layers(model: tr.Model, name: str, op: str, m: int, draw: int) -> list[torch.Tensor]:
    if op == "prh":
        return model.prh
    return model.truncate(op, m, _gen(name, op, m, draw))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _r(t: torch.Tensor) -> list:
    return np.round(t.cpu().numpy().astype(float), 4).tolist()


def run(models: dict, pairs: list, out: Path, draws: int, permutations: int, info: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n = next(iter(models.values())).n
    gen = torch.Generator().manual_seed(0)
    perms = torch.stack([torch.randperm(n, generator=gen) for _ in range(permutations)])
    rows, selfrows, fixed = [], [], {}
    full = {name: tr.build_cache(mdl.truncate("none", 0)) for name, mdl in models.items()}
    todo = settings(draws)
    t0 = time.time()
    for s, (op, m, draw) in enumerate(todo):
        caches = {}
        for name, mdl in models.items():
            caches[name] = tr.build_cache(_layers(mdl, name, op, m, draw))
            ret = tr.self_retention(caches[name], full[name])
            selfrows.append(
                {
                    "model": name,
                    "op": op,
                    "m": m,
                    "draw": draw,
                    "size": mdl.size(op, m) if op != "prh" else mdl.d,
                    "d": mdl.d,
                    **{k: _r(v) for k, v in ret.items()},
                }
            )
        allres = tr.calibrated_pairs(caches, [(a, b) for _, a, b in pairs], perms)
        for group, a, b in pairs:
            for metric, r in allres[(a, b)].items():
                key = (group, a, b, metric)
                if op == "none":
                    exc = r["score"] - r["mu0"]
                    fixed[key] = [int(i) for i in divmod(int(exc.argmax()), exc.shape[1])]
                fi, fj = fixed.get(key, r["max_cell"])
                rows.append(
                    {
                        "group": group,
                        "a": a,
                        "b": b,
                        "metric": metric,
                        "op": op,
                        "m": m,
                        "draw": draw,
                        "size_a": models[a].size(op, m) if op != "prh" else models[a].d,
                        "size_b": models[b].size(op, m) if op != "prh" else models[b].d,
                        **{
                            k: r[k]
                            for k in (
                                "max_score",
                                "max_mu0",
                                "max_tau",
                                "max_g",
                                "max_p",
                                "max_cell",
                            )
                        },
                        "fixed_cell": [fi, fj],
                        "fixed_score": float(r["score"][fi, fj]),
                        "fixed_mu0": float(r["mu0"][fi, fj]),
                        "fixed_g": float(r["g"][fi, fj]),
                        "score": _r(r["score"]),
                        "mu0": _r(r["mu0"]),
                    }
                )
        print(f"{s + 1}/{len(todo)} {op} m={m} draw={draw}  {time.time() - t0:.0f}s", flush=True)
    with open(out / "pairs.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "self.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in selfrows)
    meta = {
        **info,
        "git_sha": _git_sha(),
        "permutations": permutations,
        "draws": draws,
        "n": n,
        "seconds": round(time.time() - t0),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def even_layers(num: int, keep: int) -> list[int]:
    return sorted(set(np.linspace(0, num - 1, min(keep, num)).round().astype(int).tolist()))


def cmd_real(args) -> None:
    root = Path(args.root)
    llms = args.llms.split(",") if args.llms else list(PILOT_LLMS)
    vits = args.vits.split(",") if args.vits else list(PILOT_VITS)
    models, layer_idx = {}, {}
    for name, variant in [(m, ac.Variant(pool="avg", caption_idx=0)) for m in llms] + [
        (v, ac.Variant(pool="cls", caption_idx=None, modality="vision")) for v in vits
    ]:
        payload = ac.load_model(root, name, variant)
        if payload is None:
            raise FileNotFoundError(f"{name} not fully cached under {root}")
        feats = payload["feats"]
        feats = torch.stack(feats, 1) if isinstance(feats, list) else feats
        mdl = tr.Model(feats)
        layer_idx[name] = even_layers(feats.shape[1], args.layers)
        mdl.select(layer_idx[name])
        models[name] = mdl
        print(f"{name}: n={mdl.n} d={mdl.d} rank={mdl.rank} layers={layer_idx[name]}", flush=True)
    pairs = [("X", a, b) for a in llms for b in vits]
    pairs += [("W_llm", a, b) for a, b in itertools.combinations(llms, 2)]
    pairs += [("W_vit", a, b) for a, b in itertools.combinations(vits, 2)]
    # negative control: one LLM x ViT pair with the ViT's samples shuffled once
    sa, sb = llms[-1], vits[-1]
    g = torch.Generator().manual_seed(SHUFFLE_SEED)
    shuf = tr.Model(
        ac.load_model(root, sb, ac.Variant(pool="cls", caption_idx=None, modality="vision"))[
            "feats"
        ]
    )
    p = torch.randperm(shuf.n, generator=g)
    shuf.layers = [x[p] for x in shuf.layers]
    shuf.prh = [x[p] for x in shuf.prh]
    shuf.select(layer_idx[sb])
    models[f"{sb}__shuffled"] = shuf
    pairs.append(("D_shuffle", sa, f"{sb}__shuffled"))
    info = {
        "root": str(root),
        "llms": llms,
        "vits": vits,
        "layers": layer_idx,
        "dataset": "minhuh/prh wit_1024",
        "pipeline": "clamp q=0.95 -> centre -> truncate -> l2",
    }
    run(models, pairs, Path(args.out), args.draws, args.permutations, info)


def cmd_synthetic(args) -> None:
    """Shared content in the high- vs the low-variance band; one layer, n = 1024, d = 512."""
    n, s, p, d = 1024, 8, 16, 512
    models, pairs = {}, []
    for scen, (ssd, psd) in {
        "shared_top": (3.0, 1.0),
        "shared_mid": (1.5, 1.5),
        "shared_tail": (1.0, 3.0),
    }.items():
        shared = torch.randn(n, s, generator=torch.Generator().manual_seed(len(scen)))
        for side, seed in (("A", 1), ("B", 2)):
            feats = tr.synthetic_model(shared, p, ssd, psd, d, layers=1, seed=seed + 10 * len(scen))
            models[f"{scen}_{side}"] = tr.Model(feats)
        pairs.append((scen, f"{scen}_A", f"{scen}_B"))
    info = {"synthetic": {"n": n, "shared": s, "private": p, "d": d}}
    run(models, pairs, Path(args.out), args.draws, args.permutations, info)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("--root", default=str(ac.default_root()))
    r.add_argument("--llms", default="")
    r.add_argument("--vits", default="")
    r.add_argument("--layers", type=int, default=7, help="evenly spaced layers per model")
    s = sub.add_parser("synthetic")
    for x in (r, s):
        x.add_argument("--out", required=True)
        x.add_argument("--draws", type=int, default=3)
        x.add_argument("--permutations", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    {"real": cmd_real, "synthetic": cmd_synthetic}[args.cmd](args)


if __name__ == "__main__":
    main()
