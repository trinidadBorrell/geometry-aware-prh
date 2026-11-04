"""Dimension truncation of representations and calibrated alignment under it (exp-019).

Pipeline for every layer of a model, so that every operator at full size gives the same geometry:

    raw [n, L, d] -> PRH outlier clamp (q = 0.95, upstream `remove_outliers`, one threshold over all
    layers) -> centre each layer -> truncate to m coordinates -> l2 normalise -> metrics.

The clamp comes first because it is part of how the PRH defines the representation, and clamping
after truncation would depend on the basis (and on zeroed coordinates). Centring makes the PCA
operators well defined; it changes the untruncated score slightly from the uncentred PRH score,
which is reported separately as a reference. All operators return orthonormal coordinates of a
projection, so inner products (and hence cosine kNN and linear CKA) equal those of the projection
in the original space.

Operators (m = coordinates kept, except `drop_top`, where m = leading PCs removed):
- `none`: the centred layer;
- `top`: the top-m principal components (Tsukagoshi & Sasano 2025, PCA);
- `drop_top`: everything except the top-m principal components;
- `rand_coord`: m coordinates of the model's own basis, uniformly without replacement
  (Takeshita et al. 2025, random K% truncation);
- `rand_proj`: m directions of a Haar-random orthonormal basis (a random projection: the
  Johnson-Lindenstrauss control for `rand_coord`; the two differ only if the basis is privileged).

Scores (`calibrated_grid`): mutual kNN (k = 10) and linear CKA for every layer pair, with
`layer_calibration`'s caches and per-cell Algorithm 1, plus the max over cells against the null of
the max (Algorithm 2); permutations are batched into one matrix product per layer of B.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from geoprh import knn_sweep as ks
from geoprh import layer_calibration as lc

OPS = ("none", "top", "drop_top", "rand_coord", "rand_proj")
REPO = Path(__file__).resolve().parents[2]


def _upstream_remove_outliers():
    spec = importlib.util.spec_from_file_location(
        "prh_metrics", REPO / "platonic-rep" / "metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.remove_outliers


remove_outliers = _upstream_remove_outliers()


class Model:
    """Clamped, centred layers of one model and their principal axes."""

    def __init__(self, feats: torch.Tensor, q: float = 0.95):
        x = remove_outliers(feats.float(), q=q, exact=False)  # [n, L, d], upstream semantics
        self.prh = [F.normalize(v, dim=-1) for v in x.unbind(1)]  # uncentred PRH reference
        x = x - x.mean(0, keepdim=True)
        self.layers = list(x.unbind(1))
        self.n, self.d = x.shape[0], x.shape[2]
        self._v: dict[int, torch.Tensor] = {}

    def select(self, layers: list[int]) -> None:
        """Keep only these layers (after the clamp, whose threshold uses all layers as upstream)."""
        self.layers = [self.layers[i] for i in layers]
        self.prh = [self.prh[i] for i in layers]
        self._v = {}

    def axes(self, layer: int) -> torch.Tensor:
        """[d, r] right singular vectors (descending), r = numerical rank."""
        if layer not in self._v:
            _, s, vh = torch.linalg.svd(self.layers[layer].double(), full_matrices=False)
            r = int((s > s[0] * 1e-6).sum())
            self._v[layer] = vh[:r].T.float()
        return self._v[layer]

    @property
    def rank(self) -> int:
        return min(self.axes(i).shape[1] for i in range(len(self.layers)))

    def size(self, op: str, m: int) -> int:
        """Coordinates actually kept by (op, m) (m is capped at what the operator allows)."""
        if op == "none":
            return self.d
        if op == "top":
            return min(m, self.rank)
        if op == "drop_top":
            return self.rank - min(m, self.rank - 1)
        return min(m, self.d)

    def truncate(self, op: str, m: int, gen: torch.Generator | None = None) -> list[torch.Tensor]:
        """l2-normalised [n, m'] layers. One random draw (coordinates / basis) for all layers."""
        if op not in OPS:
            raise ValueError(op)
        if op == "rand_coord":
            idx = torch.randperm(self.d, generator=gen)[: min(m, self.d)]
        if op == "rand_proj":
            g = torch.randn(self.d, min(m, self.d), generator=gen, dtype=torch.float64)
            basis = torch.linalg.qr(g).Q.float()
        out = []
        for i, x in enumerate(self.layers):
            if op == "none":
                y = x
            elif op == "top":
                y = x @ self.axes(i)[:, : min(m, self.rank)]
            elif op == "drop_top":
                # up to the model's common rank, so all layers keep the same number of axes
                # (layer ranks differ by a few near-null directions)
                y = x @ self.axes(i)[:, min(m, self.rank - 1) : self.rank]
            elif op == "rand_coord":
                y = x[:, idx]
            else:
                y = x @ basis
            out.append(F.normalize(y, dim=-1))
        return out


@dataclass
class Cache:
    """One model's operands at one setting: dense caches (CKA, self-retention) and kNN lists."""

    dense: lc.ModelCache
    sweep: ks.SweepCache


def build_cache(layers: list[torch.Tensor], k: int = 10) -> Cache:
    return Cache(lc.build_cache(torch.stack(layers, 1), k), ks.build_sweep_cache(layers, k))


def batched_scores(a: lc.ModelCache, b: lc.ModelCache, perms: torch.Tensor | None) -> dict:
    """{metric: [P, La, Lb]} (P = 1 and no permutation if perms is None); same as `lc.scores`."""
    n = a.masks.shape[1]
    out = {}
    for metric, xa, xb in (("mutual_knn", a.masks, b.masks), ("cka_lin", a.grams, b.grams)):
        fa = xa.flatten(1)
        if perms is None:
            stack = xb.flatten(1)[None]  # [1, Lb, n^2]
        else:
            stack = torch.stack([xb[:, p][:, :, p].flatten(1) for p in perms])  # [P, Lb, n^2]
        s = (stack.flatten(0, 1) @ fa.T).view(stack.shape[0], stack.shape[1], -1).transpose(1, 2)
        out[metric] = s / (n * a.k) if metric == "mutual_knn" else s
    return out


def _calibrate(observed: dict, nulls: dict, alpha: float) -> dict:
    out = {}
    for m, obs in observed.items():
        null = torch.cat(nulls[m]) if isinstance(nulls[m], list) else nulls[m]
        cells = lc.calibrate_cells(obs, null, alpha)
        mx = lc.calibrate_cells(
            obs.max().view(1, 1), null.flatten(1).max(1).values.view(-1, 1, 1), alpha
        )
        out[m] = {**cells, **{f"max_{k}": float(v) for k, v in mx.items()}}
        out[m]["max_cell"] = [int(i) for i in divmod(int(obs.argmax()), obs.shape[1])]
    return out


def calibrated_pairs(
    caches: dict,
    pairs: list[tuple[str, str]],
    perms: torch.Tensor,
    alpha: float = 0.05,
    chunk: int = 10,
) -> dict:
    """{(a, b): {metric: per-cell calibrate_cells(...) + max_* (null of the max)}}.

    perms: [P, n] row permutations of B, the same for every pair and setting. Each B model's
    permuted centred Grams are built once per chunk and shared by all pairs with that B; mutual
    kNN nulls use `knn_sweep` (a gather of neighbour lists instead of n x n masks)."""
    k = next(iter(caches.values())).dense.k
    observed, nulls = {}, {}
    for a, b in pairs:
        A, B = caches[a], caches[b]
        observed[(a, b)] = {
            "mutual_knn": ks.scores(A.sweep, B.sweep, [k])[0],
            "cka_lin": A.dense.grams.flatten(1) @ B.dense.grams.flatten(1).T,
        }
        nulls[(a, b)] = {"mutual_knn": [], "cka_lin": []}
    by_b: dict[str, list[str]] = {}
    for a, b in pairs:
        by_b.setdefault(b, []).append(a)
    for s in range(0, perms.shape[0], chunk):
        pc = perms[s : s + chunk]
        for b, alist in by_b.items():
            g = caches[b].dense.grams
            stack = torch.stack([g[:, p][:, :, p].flatten(1) for p in pc])  # [Pc, Lb, n^2]
            flat = stack.flatten(0, 1)
            for a in alist:
                A = caches[a]
                cka = (
                    (flat @ A.dense.grams.flatten(1).T)
                    .view(len(pc), g.shape[0], -1)
                    .transpose(1, 2)
                )
                knn = torch.stack([ks.scores(A.sweep, caches[b].sweep, [k], p)[0] for p in pc])
                nulls[(a, b)]["cka_lin"].append(cka)
                nulls[(a, b)]["mutual_knn"].append(knn)
            del stack, flat
    return {pair: _calibrate(observed[pair], nulls[pair], alpha) for pair in pairs}


def calibrated_grid(a: Cache, b: Cache, perms: torch.Tensor, alpha: float = 0.05) -> dict:
    """`calibrated_pairs` for one pair."""
    return calibrated_pairs({"a": a, "b": b}, [("a", "b")], perms, alpha)[("a", "b")]


def self_retention(t: Cache, full: Cache) -> dict:
    """Per layer: mutual kNN and linear CKA between a truncated layer and the same full layer."""
    t, full = t.dense, full.dense
    n = t.masks.shape[1]
    knn = (t.masks * full.masks).flatten(1).sum(1) / (n * t.k)
    cka = (t.grams * full.grams).flatten(1).sum(1)
    return {"mutual_knn": knn, "cka_lin": cka}


def synthetic_model(
    shared: torch.Tensor,
    private_dim: int,
    shared_sd: float,
    private_sd: float,
    d: int,
    layers: int = 1,
    noise: float = 0.05,
    seed: int = 0,
) -> torch.Tensor:
    """[n, layers, d]: shared latents (sd shared_sd) and own private latents (sd private_sd),
    rotated into d dimensions by a random orthonormal map per layer, plus isotropic noise.
    Placing the shared content in the high- or the low-variance band tests whether the
    truncation curves locate it."""
    g = torch.Generator().manual_seed(seed)
    n, s = shared.shape
    z = torch.cat([shared_sd * shared, private_sd * torch.randn(n, private_dim, generator=g)], 1)
    out = []
    for _ in range(layers):
        q = torch.linalg.qr(torch.randn(d, s + private_dim, generator=g, dtype=torch.float64)).Q
        out.append(z @ q.float().T + noise * torch.randn(n, d, generator=g))
    return torch.stack(out, 1)
