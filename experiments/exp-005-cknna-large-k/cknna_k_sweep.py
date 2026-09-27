"""exp-005: sweep the neighbourhood size k for mutual kNN and CKNNA, with CKA as reference.

For each (LLM, ViT) pair, scores every layer pair at every k with

* `mutual_knn`  platonic-rep mutual kNN at k (chance level k / (n - 1));
* `cknna`       platonic-rep CKNNA at k (unbiased HSIC): hsic(M*K, M*L) /
                sqrt(hsic(Mk*K, Mk*K) hsic(Ml*L, Ml*L)), with M = Mk * Ml the joint mask and
                Mk, Ml each model's own top-k mask. At k = n - 1 it is unbiased CKA exactly;

and once per pair with linear CKA (biased, platonic-rep `cka`). Each metric reports the max
over layer pairs, as upstream `compute_score` (metric(vision, language), best starts at 0).
Preprocessing is platonic-rep's (q=0.95 clamp, l2 norm), via geoprh.prh_alignment.

With --diagnostics it also scores three variants of CKNNA that share its masks, to explain
why CKNNA exceeds 1 at large k:

* `same_mask`   same numerator, both denominators on the joint mask M;
* `centered`    CKNNA on the centred Grams HKH, HLH;
* `mask_only`   CKNNA with K = L = 1 off the diagonal (pure mask structure);

plus a permutation null of CKNNA at the best layer pair and per-layer anisotropy.

Every layer x layer term is a matrix product over flattened n x n matrices except the
column-sum/row-sum term of the unbiased HSIC, which is a loop over one side's layers.

    uv run python experiments/exp-005-cknna-large-k/cknna_k_sweep.py --out <dir> --workers 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from geoprh import activation_cache as ac
from geoprh.prh_alignment import _git_sha, available, prepare_layers

KS = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1024]
DIAG_LLMS = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_13b",
    "huggyllama/llama-13b",
]
DIAG_LVMS = [
    "vit_base_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_large_patch16_224.augreg_in21k",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_huge_patch14_clip_224.laion2b",
]
NULL_PERMS = 10


def hsic_u(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """platonic-rep hsic_unbiased for inputs whose diagonal is already zero."""
    m = a.shape[0]
    value = (
        (a * b.T).sum()
        + a.sum() * b.sum() / ((m - 1) * (m - 2))
        - 2 * (a.sum(0) * b.sum(1)).sum() / (m - 2)
    )
    return value / (m * (m - 3))


def hsic_pairs(pa, pb, qa, qb) -> torch.Tensor:
    """[La, Lb] matrix of hsic_u(pa[i] * pb[j], qa[i] * qb[j]); inputs [L, n, n], zero diagonal.

    sum(P * Q.T) factorises into (pa * qa^T)[i] . (pb * qb^T)[j], and sum(P), sum(Q) into
    pa[i] . pb[j] and qa[i] . qb[j], so those are matrix products; only
    sum_c colsum(P)_c rowsum(Q)_c needs a loop.
    """
    la, n = pa.shape[0], pa.shape[1]
    flat = lambda x: x.reshape(x.shape[0], -1)  # noqa: E731
    term1 = flat(pa * qa.transpose(1, 2)) @ flat(pb * qb.transpose(1, 2)).T
    sum_p = flat(pa) @ flat(pb).T
    sum_q = flat(qa) @ flat(qb).T
    term3 = torch.empty_like(term1)
    for i in range(la):
        col_p = (pa[i] * pb).sum(1)  # [Lb, n]
        row_q = (qa[i] * qb).sum(2)  # [Lb, n]
        term3[i] = (col_p * row_q).sum(1)
    value = term1 + sum_p * sum_q / ((n - 1) * (n - 2)) - 2 * term3 / (n - 2)
    return value / (n * (n - 3))


def self_hsic(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """[L] vector of hsic_u(m[l] * x[l], m[l] * x[l])."""
    return torch.stack([hsic_u(a, a) for a in m * x])


class Model:
    """All layers of one model: Grams, centred Grams, neighbour order, anisotropy."""

    def __init__(self, feats: torch.Tensor):
        layers = prepare_layers(feats)
        n = layers[0].shape[0]
        self.n = n
        self.K = torch.stack([x @ x.T for x in layers])
        self.K.diagonal(dim1=1, dim2=2).zero_()  # every consumer ignores the diagonal
        full = torch.stack([x @ x.T for x in layers])
        mean0, mean1 = full.mean(1, keepdim=True), full.mean(2, keepdim=True)
        self.Kc = full - mean0 - mean1 + full.mean((1, 2), keepdim=True)
        off = ~torch.eye(n, dtype=torch.bool)
        self.mean_cos = [k[off].mean().item() for k in full]
        self.std_cos = [k[off].std().item() for k in full]
        khat = full.clone()
        khat.diagonal(dim1=1, dim2=2).fill_(float("-inf"))
        # neighbours by decreasing similarity, self excluded (upstream topk on K_hat)
        self.order = torch.argsort(khat, dim=2, descending=True)[:, :, : n - 1].to(torch.int16)
        self.hsic_b = (self.Kc * self.Kc).sum((1, 2))
        self.Kc_diag0 = self.Kc.clone()
        self.Kc_diag0.diagonal(dim1=1, dim2=2).zero_()
        del full, khat

    def __len__(self) -> int:
        return self.K.shape[0]

    def masks(self, k: int) -> torch.Tensor:
        idx = self.order[:, :, :k].long()
        return torch.zeros(len(self), self.n, self.n).scatter_(2, idx, 1.0)


def _ratio(num, den_a, den_b):
    return num / (torch.sqrt(den_a[:, None] * den_b[None, :]) + 1e-6)


def _best(mat: torch.Tensor) -> tuple[float, list[int]]:
    """Upstream compute_score: max over layer pairs, starting from 0."""
    flat = int(mat.argmax())
    best = float(mat.reshape(-1)[flat])
    if best <= 0:
        return 0.0, [-1, -1]
    return best, [flat // mat.shape[1], flat % mat.shape[1]]


def _null(v: Model, lang: Model, i: int, j: int, k: int, seed: int) -> list[float]:
    """CKNNA at layer pair (i, j) with the language samples permuted."""
    n = v.n
    ma = v.masks(k)[i]
    mb = lang.masks(k)[j]
    den = torch.sqrt(hsic_u(ma * v.K[i], ma * v.K[i]) * hsic_u(mb * lang.K[j], mb * lang.K[j]))
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(NULL_PERMS):
        p = torch.randperm(n, generator=gen)
        m = ma * mb[p][:, p]
        out.append((hsic_u(m * v.K[i], m * lang.K[j][p][:, p]) / (den + 1e-6)).item())
    return out


def pair_task(args) -> dict:
    llm, lvm, root, ks, out_dir, diagnostics, seed = args
    torch.set_num_threads(1)
    t0 = time.time()
    lang = Model(ac.load_model(root, llm, ac.Variant(pool="avg", caption_idx=0))["feats"])
    vis = Model(
        ac.load_model(root, lvm, ac.Variant(pool="cls", caption_idx=None, modality="vision"))[
            "feats"
        ]
    )
    n = vis.n
    ks = [min(k, n - 1) for k in ks]
    names = ["mutual_knn", "cknna"]
    if diagnostics:
        names += ["same_mask", "centered", "mask_only", "joint_density"]
    mats = {name: np.zeros((len(ks), len(vis), len(lang)), dtype=np.float32) for name in names}

    flat = lambda x: x.reshape(x.shape[0], -1)  # noqa: E731
    cka = (flat(vis.Kc) @ flat(lang.Kc).T) / (torch.sqrt(vis.hsic_b[:, None] * lang.hsic_b) + 1e-6)
    per_k = []
    for t, k in enumerate(ks):
        ma, mb = vis.masks(k), lang.masks(k)
        mak, mbl = ma * vis.K, mb * lang.K
        overlap = flat(ma) @ flat(mb).T  # |joint mask|
        num = hsic_pairs(mak, mb, ma, mbl)
        self_a, self_b = self_hsic(ma, vis.K), self_hsic(mb, lang.K)
        cknna = _ratio(num, self_a, self_b)
        mats["mutual_knn"][t] = (overlap / (n * k)).numpy()
        mats["cknna"][t] = cknna.numpy()
        rec = {"k": k, "p": k / (n - 1)}
        for name in ("mutual_knn", "cknna"):
            rec[name], rec[f"{name}_layers"] = _best(torch.from_numpy(mats[name][t]))
        rec["cknna_frac_pairs_above_1"] = float((cknna > 1).float().mean())
        if diagnostics:
            same_a = hsic_pairs(mak, mb, mak, mb)
            same_b = hsic_pairs(ma, mbl, ma, mbl)
            mats["same_mask"][t] = (num / (torch.sqrt(same_a * same_b) + 1e-6)).numpy()
            num_c = hsic_pairs(ma * vis.Kc_diag0, mb, ma, mb * lang.Kc_diag0)
            mats["centered"][t] = _ratio(
                num_c, self_hsic(ma, vis.Kc_diag0), self_hsic(mb, lang.Kc_diag0)
            ).numpy()
            mats["mask_only"][t] = _ratio(
                hsic_pairs(ma, mb, ma, mb), self_hsic(ma, ma), self_hsic(mb, mb)
            ).numpy()
            mats["joint_density"][t] = (overlap / (n * (n - 1))).numpy()
            i, j = rec["cknna_layers"]
            null = _null(vis, lang, i, j, k, seed)
            rec["cknna_null_mean"], rec["cknna_null_max"] = float(np.mean(null)), max(null)
            rec["at_best"] = {
                name: float(mats[name][t, i, j])
                for name in ("same_mask", "centered", "mask_only", "joint_density")
            }
            rec["at_best"].update(
                num=float(num[i, j]),
                self_a=float(self_a[i]),
                self_b=float(self_b[j]),
                same_a=float(same_a[i, j]),
                same_b=float(same_b[i, j]),
            )
            for name in ("same_mask", "centered", "mask_only"):
                rec[name], rec[f"{name}_layers"] = _best(torch.from_numpy(mats[name][t]))
            del same_a, same_b, num_c
        per_k.append(rec)
        del ma, mb, mak, mbl
    cka_best, cka_layers = _best(cka)
    stem = f"{llm.replace('/', '__')}__x__{lvm}"
    np.savez_compressed(out_dir / "pairs" / f"{stem}.npz", ks=np.array(ks), cka=cka.numpy(), **mats)
    return {
        "llm": llm,
        "lvm": lvm,
        "n": n,
        "cka": cka_best,
        "cka_layers": cka_layers,
        "per_k": per_k,
        "anisotropy": {
            "vision_mean_cos": vis.mean_cos,
            "vision_std_cos": vis.std_cos,
            "language_mean_cos": lang.mean_cos,
            "language_std_cos": lang.std_cos,
        },
        "diagnostics": diagnostics,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--llms", default="all", help="'all' (cached modelset), 'diag' or a list")
    parser.add_argument("--lvms", default="all", help="'all' (cached modelset), 'diag' or a list")
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    cached_llms, cached_lvms, _ = available(args.modelset, root)

    def pick(arg, cached, diag):
        if arg == "all":
            return cached
        return diag if arg == "diag" else arg.split(",")

    llms = pick(args.llms, cached_llms, DIAG_LLMS)
    lvms = pick(args.lvms, cached_lvms, DIAG_LVMS)
    (args.out / "pairs").mkdir(parents=True, exist_ok=True)
    partial = args.out / "sweep_pairs.jsonl"
    done = set()
    if partial.exists():
        done = {(r["llm"], r["lvm"]) for r in map(json.loads, partial.read_text().splitlines())}
    tasks = [
        (lm, v, root, ks, args.out, args.diagnostics, args.seed)
        for lm in llms
        for v in lvms
        if (lm, v) not in done
    ]
    print(f"{len(tasks)} pairs to go, k = {ks}, diagnostics = {args.diagnostics}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, partial.open("a") as fh:
        futures = [pool.submit(pair_task, t) for t in tasks]
        for c, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            peak = max(rec["per_k"], key=lambda r: r["cknna"])
            print(
                f"[{c}/{len(tasks)}] {rec['llm']} x {rec['lvm']}: CKA {rec['cka']:.2f}, "
                f"peak CKNNA {peak['cknna']:.2f} at k={peak['k']} ({rec['seconds']}s)",
                flush=True,
            )
    meta = {"ks": ks, "null_perms": NULL_PERMS, "git_sha": _git_sha(), "root": str(root)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
