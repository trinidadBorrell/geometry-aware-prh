"""PRH alignment on CPU with per-layer caching (platonic-rep metrics, same semantics).

Upstream `platonic-rep/measure_alignment.py` needs CUDA and recomputes kernels and
nearest neighbours for every (vision layer, language layer) pair. This module keeps its
procedure -- outlier clamping (q=0.95), l2 normalisation, score every layer pair, report
the max -- but computes per-layer quantities once:

* Gram matrix, centred Gram (CKA), diagonal-free Gram sums (unbiased CKA), kNN indices
  (mutual/cycle/edit/LCS kNN) and top-k masks (CKNNA) are built once per layer.
* CKA / unbiased CKA / CKNNA then reduce to O(n^2) elementwise sums per layer pair.
* SVCCA, cycle/edit/LCS kNN call the upstream functions directly.

`tests/test_prh_alignment.py` checks every metric against `metrics.AlignmentMetrics`.

As in upstream `compute_alignment`, metrics are called as metric(vision, language) and the
best score starts at 0 (non-positive scores never win).

Example:
    uv run python -m geoprh.prh_alignment cross  --modelset val --out results/prh_val
    uv run python -m geoprh.prh_alignment vision --modelset val --out results/prh_val
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geoprh import activation_cache as ac

REPO = Path(__file__).resolve().parents[2]
KNN_K = 10
CKNNA_KS = (10, 20, 50, 100, 200, 500, 800, 1000)


def _load_upstream_metrics():
    """Import platonic-rep/metrics.py under a unique name (it is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "platonic_metrics", REPO / "platonic-rep" / "metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_upstream_metrics()


# --------------------------------------------------------------------------------------
# preprocessing (upstream measure_alignment.prepare_features + compute_score normalisation)
# --------------------------------------------------------------------------------------
def prepare_layers(feats: torch.Tensor, q: float = 0.95) -> list[torch.Tensor]:
    """[n, L, d] -> list of L l2-normalised [n, d] float32 layers (upstream semantics)."""
    feats = M.remove_outliers(feats.float(), q=q, exact=False)
    return [F.normalize(feats[:, i, :], p=2, dim=-1) for i in range(feats.shape[1])]


# --------------------------------------------------------------------------------------
# per-layer cache
# --------------------------------------------------------------------------------------
class Layer:
    """Everything a layer contributes to the pairwise metrics, computed once."""

    def __init__(self, x: torch.Tensor, cknna_kmax: int):
        n = x.shape[0]
        self.x = x
        self.n = n
        self.K = x @ x.T
        # mutual/cycle/edit/LCS kNN: exactly upstream compute_nearest_neighbors
        self.knn = M.compute_nearest_neighbors(x, KNN_K)
        # biased CKA: trace(K H L H) = <HKH, HLH>_F
        self.Kc = self.K - self.K.mean(0, keepdim=True) - self.K.mean(1, keepdim=True)
        self.Kc += self.K.mean()
        self.hsic_b = (self.Kc * self.Kc).sum()
        # unbiased CKA pieces of the diagonal-free Gram Kt (not stored: <Kt, Lt> is
        # <K, L> minus the diagonal product)
        self.diag = self.K.diagonal().clone()
        self.Kt_sum = self.K.sum() - self.diag.sum()
        self.Kt_col = self.K.sum(0) - self.diag
        self.Kt_row = self.K.sum(1) - self.diag
        self.hsic_u = _hsic_unbiased_parts(self, self)
        # CKNNA: top-k (diagonal excluded) of K, sorted, for the largest k needed
        kmax = min(cknna_kmax, n - 1)
        khat = self.K.clone().fill_diagonal_(float("-inf"))
        self.topk = torch.topk(khat, kmax, dim=1).indices.to(torch.int16)
        self._cknna_self: dict[int, torch.Tensor] = {}

    def mask(self, k: int) -> torch.Tensor:
        return torch.zeros(self.n, self.n).scatter_(1, self.topk[:, :k].long(), 1)

    def cknna_self(self, k: int) -> torch.Tensor:
        if k not in self._cknna_self:
            mk = self.mask(k)
            a = mk * self.K
            self._cknna_self[k] = _hsic_unbiased(a, a)
        return self._cknna_self[k]


def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Upstream hsic_unbiased with the O(n^3) term rewritten as a vector product."""
    m = K.shape[0]
    Kt = K.clone().fill_diagonal_(0)
    Lt = L.clone().fill_diagonal_(0)
    value = (
        (Kt * Lt.T).sum()
        + Kt.sum() * Lt.sum() / ((m - 1) * (m - 2))
        - 2 * (Kt.sum(0) * Lt.sum(1)).sum() / (m - 2)
    )
    return value / (m * (m - 3))


def _hsic_unbiased_parts(a: Layer, b: Layer) -> torch.Tensor:
    """_hsic_unbiased for two cached layers (symmetric Grams, diagonals removed)."""
    m = a.n
    value = (
        (a.K * b.K).sum()
        - (a.diag * b.diag).sum()
        + a.Kt_sum * b.Kt_sum / ((m - 1) * (m - 2))
        - 2 * (a.Kt_col * b.Kt_row).sum() / (m - 2)
    )
    return value / (m * (m - 3))


# --------------------------------------------------------------------------------------
# metrics on cached layers; argument order is (vision, language) as upstream
# --------------------------------------------------------------------------------------
def mutual_knn(a: Layer, b: Layer) -> float:
    n, topk = a.knn.shape
    rows = torch.arange(n).unsqueeze(1)
    ma, mb = torch.zeros(n, n), torch.zeros(n, n)
    ma[rows, a.knn] = 1.0
    mb[rows, b.knn] = 1.0
    return ((ma * mb).sum(dim=1) / topk).mean().item()


def cycle_knn(a: Layer, b: Layer) -> float:
    return M.compute_knn_accuracy(a.knn[b.knn]).item()


def lcs_knn(a: Layer, b: Layer) -> float:
    return M.longest_ordinal_sequence(a.knn, b.knn).float().mean().item()


def edit_distance_knn(a: Layer, b: Layer) -> float:
    dist = M.compute_distance(a.knn, b.knn, M.TAF.edit_distance)
    return (1 - torch.mean(dist) / a.knn.shape[1]).item()


def cka(a: Layer, b: Layer) -> float:
    hsic_kl = (a.Kc * b.Kc).sum()
    return (hsic_kl / (torch.sqrt(a.hsic_b * b.hsic_b) + 1e-6)).item()


def unbiased_cka(a: Layer, b: Layer) -> float:
    hsic_kl = _hsic_unbiased_parts(a, b)
    return (hsic_kl / (torch.sqrt(a.hsic_u * b.hsic_u) + 1e-6)).item()


def cknna(a: Layer, b: Layer, k: int) -> float:
    mask = a.mask(k) * b.mask(k)
    sim_kl = _hsic_unbiased(mask * a.K, mask * b.K)
    return sim_kl.item() / (torch.sqrt(a.cknna_self(k) * b.cknna_self(k)) + 1e-6).item()


def svcca(a: Layer, b: Layer) -> float:
    torch.manual_seed(0)  # svd_lowrank is randomised; fix it per pair
    np.random.seed(0)  # noqa: NPY002 -- upstream svcca draws from the global numpy RNG
    return float(M.AlignmentMetrics.svcca(a.x, b.x, cca_dim=10))


CROSS_METRICS = {
    "mutual_knn": mutual_knn,
    "cycle_knn": cycle_knn,
    "lcs_knn": lcs_knn,
    "edit_distance_knn": edit_distance_knn,
    "cka": cka,
    "unbiased_cka": unbiased_cka,
    "svcca": svcca,
    **{f"cknna_k{k}": (lambda a, b, k=k: cknna(a, b, k)) for k in CKNNA_KS},
}


def best_over_layers(vision: list[Layer], language: list[Layer], fn) -> tuple[float, tuple]:
    """Upstream compute_score: max over layer pairs, starting from 0."""
    best, best_idx = 0.0, (-1, -1)
    for i, a in enumerate(vision):
        for j, b in enumerate(language):
            score = fn(a, b)
            if score > best:
                best, best_idx = score, (i, j)
    return best, best_idx


# --------------------------------------------------------------------------------------
# cross-modal job: every (language model, vision model) pair
# --------------------------------------------------------------------------------------
def _layers_for(model: str, variant: ac.Variant, root: Path, kmax: int) -> list[Layer]:
    payload = ac.load_model(root, model, variant)
    if payload is None:
        raise FileNotFoundError(f"{model} not fully cached under {root}")
    return [Layer(x, kmax) for x in prepare_layers(payload["feats"])]


def _pair_task(args) -> dict:
    llm, lvm, root, metrics = args
    torch.set_num_threads(1)
    t0 = time.time()
    lang = _layers_for(llm, ac.Variant(pool="avg", caption_idx=0), root, max(CKNNA_KS))
    vis = _layers_for(
        lvm, ac.Variant(pool="cls", caption_idx=None, modality="vision"), root, max(CKNNA_KS)
    )
    out = {"llm": llm, "lvm": lvm}
    for name in metrics:
        score, idx = best_over_layers(vis, lang, CROSS_METRICS[name])
        out[name] = {"score": score, "vision_layer": idx[0], "language_layer": idx[1]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def available(modelset: str, root: Path) -> tuple[list[str], list[str], dict]:
    """Models of the modelset that are fully cached, plus their per-model extras."""
    llms, lvms, info = [], [], {}
    for model, variant in ac.models(modelset):
        layers = ac.cached_layers(root, model, variant)
        if not layers:
            print(f"[skip] {model}: not in cache")
            continue
        meta = json.loads(ac.layer_paths(root, model, layers[0], variant)[1].read_text())
        info[model] = {
            "num_params": meta["num_params"],
            "num_layers": meta["num_layers"],
            **meta.get("extras", {}),
        }
        (llms if variant.modality == "language" else lvms).append(model)
    return llms, lvms, info


def run_cross(modelset: str, root: Path, out: Path, workers: int, metrics: list[str]) -> None:
    llms, lvms, info = available(modelset, root)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "cross_pairs.jsonl"
    done = set()
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            done.add((rec["llm"], rec["lvm"]))
    # biggest pairs first for load balancing
    tasks = sorted(
        [(lm, v, root, metrics) for lm in llms for v in lvms if (lm, v) not in done],
        key=lambda t: -info[t[0]]["num_layers"] * info[t[1]]["num_layers"],
    )
    print(f"{len(llms)} LLMs x {len(lvms)} ViTs, {len(done)} pairs done, {len(tasks)} to go")
    with ProcessPoolExecutor(max_workers=workers) as pool, partial.open("a") as fh:
        futures = [pool.submit(_pair_task, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{n}/{len(tasks)}] {rec['llm']} x {rec['lvm']} ({rec['seconds']}s)", flush=True)

    records = {(r["llm"], r["lvm"]): r for r in map(json.loads, partial.read_text().splitlines())}
    arrays = {}
    for name in metrics:
        scores = np.full((len(llms), len(lvms)), np.nan)
        idx = np.full((len(llms), len(lvms), 2), -1)
        for i, llm in enumerate(llms):
            for j, lvm in enumerate(lvms):
                rec = records[(llm, lvm)][name]
                scores[i, j] = rec["score"]
                idx[i, j] = (rec["vision_layer"], rec["language_layer"])
        arrays[f"{name}__scores"] = scores
        arrays[f"{name}__indices"] = idx
    np.savez(out / "cross_modal.npz", **arrays)
    meta = {
        "llms": llms,
        "lvms": lvms,
        "models": info,
        "metrics": metrics,
        "procedure": "platonic-rep measure_alignment (q=0.95 clamp, l2 norm, max over layer "
        "pairs, metric(vision, language)), cached CPU implementation",
        "git_sha": _git_sha(),
        "activations_root": str(root),
    }
    (out / "cross_modal.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / 'cross_modal.npz'}")


# --------------------------------------------------------------------------------------
# vision-vision job (PRH Fig. 12): last block, several metrics and batch sizes
# --------------------------------------------------------------------------------------
VISION_METRICS = (
    [("mutual_knn", {"topk": 10}, bsz) for bsz in (1000, 512, 256, 128)]
    + [("lcs_knn", {"topk": 10}, 1000), ("edit_distance_knn", {"topk": 10}, 1000)]
    + [
        ("cknna", {"topk": k}, 1000)
        for k in (5, 10, 50, 100, 250, 500, 600, 700, 800, 900, 950, 975, 1000)
    ]
    + [
        ("cka", {}, 1000),
        ("unbiased_cka", {}, 1000),
        ("svcca", {"cca_dim": 10}, 1000),
        ("cycle_knn", {"topk": 10}, 1000),
    ]
)


def _vision_label(metric: str, kwargs: dict, bsz: int) -> str:
    k = f"k={kwargs['topk']}, " if "topk" in kwargs else ""
    return f"{metric} ({k}bsz={bsz})"


def run_vision(modelset: str, root: Path, out: Path, seed: int = 0) -> None:
    _, lvms, _ = available(modelset, root)
    variant = ac.Variant(pool="cls", caption_idx=None, modality="vision")
    last = {}
    for m in lvms:
        feats = ac.load_model(root, m, variant)["feats"]
        last[m] = prepare_layers(feats)[-1]  # final block CLS token
    n = next(iter(last.values())).shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    pairs = [(a, b) for i, a in enumerate(lvms) for b in lvms[i + 1 :]]
    labels = [_vision_label(*spec) for spec in VISION_METRICS]
    scores = np.zeros((len(VISION_METRICS), len(pairs)))
    for p, (a, b) in enumerate(pairs):
        for s, (metric, kwargs, bsz) in enumerate(VISION_METRICS):
            sub = perm[:bsz]
            torch.manual_seed(0)
            np.random.seed(0)  # noqa: NPY002 -- upstream svcca uses the global numpy RNG
            val = M.AlignmentMetrics.measure(metric, last[a][sub], last[b][sub], **kwargs)
            scores[s, p] = float(val)
        print(f"[{p + 1}/{len(pairs)}] {a} vs {b}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vision_vision.npz", scores=scores)
    meta = {
        "lvms": lvms,
        "pairs": pairs,
        "metric_labels": labels,
        "layer": "final block CLS",
        "subsample_seed": seed,
        "git_sha": _git_sha(),
        "activations_root": str(root),
    }
    (out / "vision_vision.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / 'vision_vision.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=["cross", "vision"])
    parser.add_argument("--modelset", default="val")
    parser.add_argument("--root", type=Path, default=None, help="activation cache root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--metrics", default=",".join(CROSS_METRICS))
    args = parser.parse_args()
    root = args.root or ac.default_root()
    if args.job == "cross":
        run_cross(args.modelset, root, args.out, args.workers, args.metrics.split(","))
    else:
        run_vision(args.modelset, root, args.out)


if __name__ == "__main__":
    main()
