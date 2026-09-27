"""exp-005: sweep the neighbourhood size k for mutual kNN and CKNNA, with CKA as reference.

CKNNA is computed as published in Huh et al. 2024 (Appendix A, Eqs. 16-18):

    Kbar_ij = K_ij - E_l[K_il]                      (row-centred kernel, Eq. 12)
    Align(K, L) = sum_ij alpha_ij Kbar_ij Lbar_ij    (Eq. 16)
    alpha_ij = 1[j in knn_K(i) and j in knn_L(i) and i != j]   (Eq. 17)
    CKNNA = Align(K, L) / sqrt(Align(K, K) Align(L, L))         (Eq. 18)

This is not what platonic-rep's `AlignmentMetrics.cknna` computes: the code masks the raw kernel
first and centres the masked matrix afterwards (unbiased HSIC), which lets the score exceed 1 at
large k. The published form is bounded by 1 (Cauchy-Schwarz: the numerator sums over a subset of
each model's own neighbours). At k = n - 1 it is the row-centred CKA of Eq. 15.

For each (LLM, ViT) pair every layer pair is scored with mutual kNN(k) and CKNNA(k) at every k,
and once with linear CKA (platonic-rep `cka`); each metric reports the max over layer pairs, as
upstream `compute_score` (metric(vision, language), best starts at 0). Preprocessing is
platonic-rep's (q=0.95 clamp, l2 norm). All layer x layer terms are matrix products over
flattened n x n matrices.

    uv run python experiments/exp-005-cknna-large-k/cknna_k_sweep.py --out <dir> --workers 12
    uv run python experiments/exp-005-cknna-large-k/cknna_k_sweep.py --out <dir> --vision-vision
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


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


class Model:
    """A stack of layers: row-centred and double-centred Grams, neighbour order.

    Takes upstream features [n, L, d] (preprocessed here) or a list of preprocessed layers.
    """

    def __init__(self, feats: torch.Tensor | list[torch.Tensor]):
        layers = feats if isinstance(feats, list) else prepare_layers(feats)
        n = layers[0].shape[0]
        self.n = n
        K = torch.stack([x @ x.T for x in layers])
        # CKNNA (Eq. 12): subtract each row's mean over the dataset
        self.Kbar = K - K.mean(2, keepdim=True)
        # linear CKA (platonic-rep): double-centred Gram
        self.Kc = self.Kbar - K.mean(1, keepdim=True) + K.mean((1, 2), keepdim=True)
        self.hsic_b = (self.Kc * self.Kc).sum((1, 2))
        khat = K.clone()
        khat.diagonal(dim1=1, dim2=2).fill_(float("-inf"))
        # neighbours by decreasing similarity, self excluded
        self.order = torch.argsort(khat, dim=2, descending=True)[:, :, : n - 1].to(torch.int16)
        off = ~torch.eye(n, dtype=torch.bool)
        self.mean_cos = [k[off].mean().item() for k in K]
        del K, khat

    def __len__(self) -> int:
        return self.Kbar.shape[0]

    def masks(self, k: int) -> torch.Tensor:
        idx = self.order[:, :, :k].long()
        return torch.zeros(len(self), self.n, self.n).scatter_(2, idx, 1.0)


def scores(a: Model, b: Model, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[La, Lb] mutual kNN and CKNNA (Eqs. 16-18) at neighbourhood size k."""
    ma, mb = a.masks(k), b.masks(k)
    mknn = (_flat(ma) @ _flat(mb).T) / (a.n * k)
    # alpha = ma * mb, so sum(alpha Kbar Lbar) = (ma Kbar) . (mb Lbar)
    wa, wb = ma * a.Kbar, mb * b.Kbar
    num = _flat(wa) @ _flat(wb).T
    self_a, self_b = (wa * a.Kbar).sum((1, 2)), (wb * b.Kbar).sum((1, 2))
    cknna = num / torch.sqrt(self_a[:, None] * self_b[None, :])
    return mknn, cknna


def linear_cka(a: Model, b: Model) -> torch.Tensor:
    return (_flat(a.Kc) @ _flat(b.Kc).T) / (torch.sqrt(a.hsic_b[:, None] * b.hsic_b) + 1e-6)


def _best(mat: torch.Tensor) -> tuple[float, list[int]]:
    """Upstream compute_score: max over layer pairs, starting from 0."""
    flat = int(mat.argmax())
    best = float(mat.reshape(-1)[flat])
    if best <= 0:
        return 0.0, [-1, -1]
    return best, [flat // mat.shape[1], flat % mat.shape[1]]


def pair_task(args) -> dict:
    llm, lvm, root, ks, out_dir = args
    torch.set_num_threads(1)
    t0 = time.time()
    lang = Model(ac.load_model(root, llm, ac.Variant(pool="avg", caption_idx=0))["feats"])
    vis_variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    vis = Model(ac.load_model(root, lvm, vis_variant)["feats"])
    n = vis.n
    ks = [min(k, n - 1) for k in ks]
    mats = {
        name: np.zeros((len(ks), len(vis), len(lang)), np.float32) for name in ("mknn", "cknna")
    }
    per_k = []
    for t, k in enumerate(ks):
        mknn, cknna = scores(vis, lang, k)
        mats["mknn"][t], mats["cknna"][t] = mknn.numpy(), cknna.numpy()
        rec = {"k": k}
        for name, mat in (("mutual_knn", mknn), ("cknna", cknna)):
            rec[name], rec[f"{name}_layers"] = _best(mat)
        per_k.append(rec)
    cka = linear_cka(vis, lang)
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
        "seconds": round(time.time() - t0, 1),
    }


def vision_vision(root: Path, lvms: list[str], ks: list[int], out: Path) -> None:
    """PRH Fig. 12 setting: last-block CLS of every ViT against every other."""
    variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    vis = Model([prepare_layers(ac.load_model(root, m, variant)["feats"])[-1] for m in lvms])
    ks = [min(k, vis.n - 1) for k in ks]
    mknn = np.zeros((len(ks), len(vis), len(vis)), np.float32)
    cknna = np.zeros_like(mknn)
    for t, k in enumerate(ks):
        a, b = scores(vis, vis, k)
        mknn[t], cknna[t] = a.numpy(), b.numpy()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "vision_vision.npz",
        ks=np.array(ks),
        lvms=np.array(lvms),
        mean_cos=np.array(vis.mean_cos),
        mknn=mknn,
        cknna=cknna,
        cka=linear_cka(vis, vis).numpy(),
    )
    print(f"saved {out / 'vision_vision.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--llms", default="all", help="'all' (cached modelset) or a list")
    parser.add_argument("--lvms", default="all", help="'all' (cached modelset) or a list")
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument(
        "--vision-vision", action="store_true", help="last-block ViT x ViT (PRH Fig. 12) instead"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    root = args.root or ac.default_root()
    ks = [int(k) for k in args.ks.split(",")]
    cached_llms, cached_lvms, _ = available(args.modelset, root)
    llms = cached_llms if args.llms == "all" else args.llms.split(",")
    lvms = cached_lvms if args.lvms == "all" else args.lvms.split(",")
    if args.vision_vision:
        vision_vision(root, lvms, ks, args.out)
        return
    (args.out / "pairs").mkdir(parents=True, exist_ok=True)
    partial = args.out / "sweep_pairs.jsonl"
    done = set()
    if partial.exists():
        done = {(r["llm"], r["lvm"]) for r in map(json.loads, partial.read_text().splitlines())}
    tasks = [(lm, v, root, ks, args.out) for lm in llms for v in lvms if (lm, v) not in done]
    print(f"{len(tasks)} pairs to go, k = {ks}", flush=True)
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
    meta = {"ks": ks, "git_sha": _git_sha(), "root": str(root), "cknna": "Huh et al. Eqs. 16-18"}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
