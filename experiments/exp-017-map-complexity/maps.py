"""exp-017: is the map between two representations nonlinear? Linear vs nonlinear held-out prediction.

H1 ("wrong ruler") in operational form: two models share structure that is related by a smooth
nonlinear map, which linear / Euclidean rulers miss. Testing that needs a map, not curvature: if a
flexible regressor predicts model B from model A (on held-out samples) clearly better than a linear
one, there is a learnable warp that some other ruler could undo; if it does not, no ruler built on
a smooth correspondence can find more agreement at this n.

Per pair (A, B), direction A -> B, 5-fold CV over samples:
- inputs: PRH features (q = 0.95 clamp, l2 norm), PCA to R = 256 dimensions fitted on the training
  fold for A (input) and for B (target), each scaled to unit mean variance per dimension;
- methods, hyperparameters chosen on an inner 10% validation split of the training fold, then
  refitted on the whole training fold:
  ridge (linear), kNN regression, Nystroem RBF kernel ridge (2048 landmarks), and a residual MLP
  (the ridge solution plus a 2-hidden-layer GELU MLP, width 1024, last layer zero-initialised, so
  it starts at the linear map; AdamW, early stopping on the validation R^2);
- held-out readouts on the test fold, in the target's PCA space: R^2 (variance-weighted), retrieval
  of the true target among the test fold (top-1 / top-5, cosine), and mutual kNN (k = 10) between
  predicted and true targets.

    uv run python experiments/exp-017-map-complexity/maps.py synthetic --out results/exp-017-map-complexity
    uv run python experiments/exp-017-map-complexity/maps.py real --features <exp-016 features dir> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
R = 256
FOLDS = 5
VAL_FRAC = 0.1
K_MKNN = 10
RIDGE_C = torch.logspace(-6, 2, 17).tolist()  # lambda = c * mean eigenvalue of the Gram matrix
KNN_KS = (5, 10, 20, 50, 100, 200)
KRR_G = (0.25, 0.5, 1.0, 2.0, 4.0)  # gamma = g / median squared distance
KRR_M = 2048
MLP_WIDTH, MLP_DROP, MLP_WD, MLP_LR = 1024, 0.1, 1e-2, 1e-3
MLP_BATCH, MLP_MAX_EPOCHS, MLP_PATIENCE = 512, 400, 30
METHODS = ("ridge", "knn", "krr", "mlp")


# ---------------------------------------------------------------- preprocessing


def prh(x: torch.Tensor) -> torch.Tensor:
    """PRH preprocessing of [n, L, d] or [n, d] features (as exp-016 `prepared_layers`)."""
    from aristotelian.prh.preprocess import prepare_features

    squeeze = x.ndim == 2
    x = x[:, None] if squeeze else x
    x = F.normalize(prepare_features(x.float(), q=0.95, exact=False), dim=-1)
    return x[:, 0] if squeeze else x


class PCA:
    """Top-R principal components of the training rows, scaled to unit mean variance per dim."""

    def __init__(self, x: torch.Tensor, r: int = R):
        self.mu = x.mean(0)
        xc = (x - self.mu).double()
        cov = xc.T @ xc if xc.shape[1] <= xc.shape[0] else None
        if cov is not None:
            e, v = torch.linalg.eigh(cov)
            e, v = e.flip(0), v.flip(1)
        else:  # d > n: through the Gram matrix
            e, u = torch.linalg.eigh(xc @ xc.T)
            e, u = e.flip(0), u.flip(1)
            v = xc.T @ u[:, :r] / e[:r].clamp_min(1e-12).sqrt()
        r = min(r, v.shape[1])
        self.v = v[:, :r].float()
        self.kept = float(e[:r].sum() / e.clamp_min(0).sum())
        z = (x - self.mu) @ self.v
        self.s = float(z.pow(2).sum(1).mean().div(r).sqrt())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) @ self.v / self.s


# ---------------------------------------------------------------- metrics


def r2(p: torch.Tensor, t: torch.Tensor) -> float:
    return float(1 - (p - t).pow(2).sum() / (t - t.mean(0)).pow(2).sum())


def knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    s = x @ x.T
    s.fill_diagonal_(-float("inf"))
    return s.topk(k, dim=1).indices


def mutual_knn(a: torch.Tensor, b: torch.Tensor, k: int = K_MKNN) -> float:
    ia, ib = knn_idx(a, k), knn_idx(b, k)
    n = a.shape[0]
    ma = torch.zeros(n, n, dtype=torch.bool, device=a.device).scatter_(1, ia, True)
    mb = torch.zeros(n, n, dtype=torch.bool, device=a.device).scatter_(1, ib, True)
    return float((ma & mb).sum(1).float().mean() / k)


def readouts(p: torch.Tensor, t: torch.Tensor) -> dict:
    s = F.normalize(p - t.mean(0), dim=-1) @ F.normalize(t - t.mean(0), dim=-1).T
    rank = (s > s.diagonal()[:, None]).sum(1)
    return {
        "r2": r2(p, t),
        "top1": float((rank < 1).float().mean()),
        "top5": float((rank < 5).float().mean()),
        "mknn": mutual_knn(p, t),
    }


# ---------------------------------------------------------------- regressors


class Ridge:
    """Ridge on centred inputs and targets via one eigendecomposition; lambda over a grid."""

    def __init__(self, x: torch.Tensor, t: torch.Tensor):
        self.mx, self.mt = x.mean(0), t.mean(0)
        xc = (x - self.mx).double()
        e, self.q = torch.linalg.eigh(xc.T @ xc)
        self.e = e.clamp_min(0)
        self.b = self.q.T @ (xc.T @ (t - self.mt).double())
        self.scale = float(self.e.mean())

    def w(self, c: float) -> torch.Tensor:
        return (self.q @ (self.b / (self.e + c * self.scale)[:, None])).float()

    def predict(self, x: torch.Tensor, c: float) -> torch.Tensor:
        return (x - self.mx) @ self.w(c) + self.mt


def fit_ridge(x, t, xv, tv):
    m = Ridge(x, t)
    scores = [r2(m.predict(xv, c), tv) for c in RIDGE_C]
    return RIDGE_C[int(np.argmax(scores))]


def ridge_predict(x, t, xq, c):
    return Ridge(x, t).predict(xq, c)


def sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.pow(2).sum(1)[:, None] + b.pow(2).sum(1)[None] - 2 * a @ b.T).clamp_min(0)


def knn_predict_all(x, t, xq, ks=KNN_KS):
    idx = (-sqdist(xq, x)).topk(max(ks), dim=1).indices
    tn = t[idx]  # [q, kmax, r]
    cs = tn.cumsum(1)
    return {k: cs[:, k - 1] / k for k in ks}


def fit_knn(x, t, xv, tv):
    preds = knn_predict_all(x, t, xv)
    return max(preds, key=lambda k: r2(preds[k], tv))


class Nystroem:
    def __init__(self, x: torch.Tensor, g: float, m: int, gen: torch.Generator):
        sel = torch.randperm(x.shape[0], generator=gen)[: min(m, x.shape[0])].to(x.device)
        self.land = x[sel]
        sub = x[torch.randperm(x.shape[0], generator=gen)[:2000].to(x.device)]
        med = float(sqdist(sub, sub).median())
        self.gamma = g / med
        e, u = torch.linalg.eigh(torch.exp(-self.gamma * sqdist(self.land, self.land)).double())
        keep = e > e.max() * 1e-8
        self.proj = (u[:, keep] / e[keep].sqrt()).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * sqdist(x, self.land)) @ self.proj


def fit_krr(x, t, xv, tv, seed):
    best = (-math.inf, None)
    for g in KRR_G:
        ny = Nystroem(x, g, KRR_M, torch.Generator().manual_seed(seed))
        m = Ridge(ny(x), t)
        fv = ny(xv)
        for c in RIDGE_C:
            s = r2(m.predict(fv, c), tv)
            if s > best[0]:
                best = (s, (g, c))
    return best[1]


def krr_predict(x, t, xq, params, seed):
    g, c = params
    ny = Nystroem(x, g, KRR_M, torch.Generator().manual_seed(seed))
    return Ridge(ny(x), t).predict(ny(xq), c)


class ResMLP(nn.Module):
    """Linear map (initialised at the ridge solution) plus a residual MLP starting at zero."""

    def __init__(self, din: int, dout: int, w0: torch.Tensor, b0: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        with torch.no_grad():
            self.lin.weight.copy_(w0.T)
            self.lin.bias.copy_(b0)
        self.net = nn.Sequential(
            nn.Linear(din, MLP_WIDTH), nn.GELU(), nn.Dropout(MLP_DROP),
            nn.Linear(MLP_WIDTH, MLP_WIDTH), nn.GELU(), nn.Dropout(MLP_DROP),
            nn.Linear(MLP_WIDTH, dout),
        )  # fmt: skip
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.lin(x) + self.net(x)


def train_mlp(x, t, c_ridge, seed, epochs=None, xv=None, tv=None):
    """Train on (x, t); with a validation set, early-stop and return the best epoch count."""
    torch.manual_seed(seed)
    rm = Ridge(x, t)
    w = rm.w(c_ridge)
    model = ResMLP(x.shape[1], t.shape[1], w, rm.mt - rm.mx @ w).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    best, best_ep, best_state, bad = -math.inf, 0, None, 0
    for ep in range(1, (epochs or MLP_MAX_EPOCHS) + 1):
        model.train()
        for idx in torch.randperm(n, generator=gen).to(x.device).split(MLP_BATCH):
            loss = (model(x[idx]) - t[idx]).pow(2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if xv is None:
            continue
        model.eval()
        with torch.no_grad():
            s = r2(model(xv), tv)
        if s > best:
            best, best_ep, bad = s, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    if xv is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep


# ---------------------------------------------------------------- one pair, one direction


def folds(n: int, seed: int, k: int = FOLDS) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, k)


@torch.no_grad()
def _nograd(f, *a, **kw):
    return f(*a, **kw)


def run_direction(a: torch.Tensor, b: torch.Tensor, seed: int, fold_ids, device: str) -> list[dict]:
    """a, b: [n, d] preprocessed features (rows paired). Returns one record per fold."""
    n = a.shape[0]
    parts = folds(n, seed)
    out = []
    for f in fold_ids:
        t0 = time.time()
        test = parts[f]
        train = np.concatenate([p for i, p in enumerate(parts) if i != f])
        rng = np.random.default_rng(seed + 1000 + f)
        val_mask = np.zeros(len(train), bool)
        val_mask[rng.choice(len(train), int(VAL_FRAC * len(train)), replace=False)] = True
        pa, pb = PCA(a[train].to(device)), PCA(b[train].to(device))
        x, t = pa(a[train].to(device)), pb(b[train].to(device))
        xq, tq = pa(a[test].to(device)), pb(b[test].to(device))
        vm = torch.from_numpy(val_mask).to(device)
        xi, ti, xv, tv = x[~vm], t[~vm], x[vm], t[vm]
        rec = {"fold": int(f), "n_train": len(train), "n_test": len(test),
               "kept_a": pa.kept, "kept_b": pb.kept, "methods": {}}  # fmt: skip

        c = _nograd(fit_ridge, xi, ti, xv, tv)
        p = _nograd(ridge_predict, x, t, xq, c)
        rec["methods"]["ridge"] = {**readouts(p, tq), "c": c}

        k = _nograd(fit_knn, xi, ti, xv, tv)
        p = _nograd(knn_predict_all, x, t, xq, (k,))[k]
        rec["methods"]["knn"] = {**readouts(p, tq), "k": k}

        kp = _nograd(fit_krr, xi, ti, xv, tv, seed + f)
        p = _nograd(krr_predict, x, t, xq, kp, seed + f)
        rec["methods"]["krr"] = {**readouts(p, tq), "g": kp[0], "c": kp[1]}

        _, ep = train_mlp(xi, ti, c, seed + f, xv=xv, tv=tv)
        model, _ = train_mlp(x, t, c, seed + f, epochs=max(ep, 1))
        with torch.no_grad():
            p = model(xq)
        rec["methods"]["mlp"] = {**readouts(p, tq), "epochs": ep}
        rec["sec"] = round(time.time() - t0, 1)
        out.append(rec)
    return out


def run_pair(a, b, meta: dict, seed: int, fold_ids, device: str, sink) -> None:
    for direction, (x, y) in {"a2b": (a, b), "b2a": (b, a)}.items():
        for rec in run_direction(x, y, seed, fold_ids, device):
            row = {**meta, "dir": direction, **rec}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            m = row["methods"]
            print(f"{meta['id']} {direction} f{rec['fold']}: "
                  + " ".join(f"{k} {m[k]['r2']:.3f}/{m[k]['top1']:.3f}" for k in METHODS)
                  + f" ({rec['sec']} s)", flush=True)  # fmt: skip


# ---------------------------------------------------------------- synthetic power check


def _std(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0)) / x.std(0).clamp_min(1e-6)


def _noisy(x: torch.Tensor, gen: torch.Generator, sd: float) -> torch.Tensor:
    return _std(x) + sd * torch.randn(x.shape, generator=gen)


def lin_embed(u, gen, d=1024):
    return u @ (torch.randn(u.shape[1], d, generator=gen) / u.shape[1] ** 0.5)


def rff_embed(u, gen, s, d=1024):
    w = torch.randn(u.shape[1], d // 2, generator=gen) * (s / u.shape[1] ** 0.5)
    ph = u @ w
    return torch.cat([ph.cos(), ph.sin()], 1)


def tanh_embed(u, gen, depth=3, width=1024):
    x = u
    for _ in range(depth):
        x = _std(
            torch.tanh(
                x @ (torch.randn(x.shape[1], width, generator=gen) * (2.0 / x.shape[1] ** 0.5))
            )
        )
    return x


SYNTH = {
    # name: (description, builder(z, za_private, zb_private, gen) -> (A, B) clean)
    "null": "A = linear(z), B = RFF(z') with independent z': nothing shared",
    "linear": "A = linear(z), B = random linear map of A: a linear relation",
    "rff_mild": "A = linear(z), B = RFF(z), phase sd 1 rad: same latent, mildly curved embedding",
    "rff_strong": "A = linear(z), B = RFF(z), phase sd 3 rad: same latent, strongly wound embedding",
    "rff_private": "rff_mild with as many private latent coordinates as shared ones (private sd 1)",
    "tanh": "A, B = two random 3-layer tanh networks of z (exp-016 arm C, rho = 1)",
    "tanh_private": "tanh with as many private coordinates as shared ones (arm C scale, sigma = 1)",
}


def synth_pair(cond: str, m: int, n: int, seed: int, noise: float = 0.3):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, m, generator=gen)
    pa, pb = torch.randn(n, m, generator=gen), torch.randn(n, m, generator=gen)
    if cond == "null":
        a, b = lin_embed(z, gen), rff_embed(torch.randn(n, m, generator=gen), gen, 1.0)
    elif cond == "linear":
        a = lin_embed(z, gen)
        b = _std(a) @ (torch.randn(a.shape[1], 1024, generator=gen) / a.shape[1] ** 0.5)
    elif cond == "rff_mild":
        a, b = lin_embed(z, gen), rff_embed(z, gen, 1.0)
    elif cond == "rff_strong":
        a, b = lin_embed(z, gen), rff_embed(z, gen, 3.0)
    elif cond == "rff_private":
        a, b = lin_embed(torch.cat([z, pa], 1), gen), rff_embed(torch.cat([z, pb], 1), gen, 1.0)
    elif cond == "tanh":
        a, b = tanh_embed(z, gen), tanh_embed(z, gen)
    elif cond == "tanh_private":
        a, b = tanh_embed(torch.cat([z, pa], 1), gen), tanh_embed(torch.cat([z, pb], 1), gen)
    else:
        raise ValueError(cond)
    return prh(_noisy(a, gen, noise)), prh(_noisy(b, gen, noise))


def cmd_synthetic(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "synthetic.jsonl"
    done = set()
    if path.exists():
        done = {(r["id"], r["dir"], r["fold"]) for r in map(json.loads, path.open())}
    with path.open("a") as sink:
        for cond, m in itertools.product(args.conds, args.ms):
            pid = f"{cond}_m{m}"
            if all((pid, d, f) in done for d in ("a2b", "b2a") for f in range(args.folds)):
                continue
            a, b = synth_pair(cond, m, args.n, args.seed)
            meta = {"id": pid, "group": "synthetic", "cond": cond, "m": m, "n": args.n}
            run_pair(a, b, meta, args.seed, range(args.folds), args.device, sink)


# ---------------------------------------------------------------- real pairs (exp-016 features)

LLMS = [
    "bigscience__bloomz-1b7",
    "bigscience__bloomz-7b1",
    "openlm-research__open_llama_7b",
    "huggyllama__llama-13b",
]
VITS = [
    "vit_large_patch16_224.augreg_in21k__cls",
    "vit_huge_patch14_224.mae__cls",
    "vit_giant_patch14_dinov2.lvd142m__cls",
    "vit_huge_patch14_clip_224.laion2b__cls",
]
TOWERS = ["clip-L14", "openclip-H14", "siglip-so400m"]
RANDOM = "random-vit_base_patch16_224.augreg_in21k__cls"


def real_specs() -> list[dict]:
    s = []
    for cap in ("gemini", "orig"):
        s += [{"group": f"B3_{cap}", "a": f"{lm}__{cap}", "b": v} for lm in LLMS for v in VITS]
        s += [{"group": f"A_{cap}", "a": f"{t}__text-{cap}", "b": f"{t}__image", "cell": ("final", "final")}
              for t in TOWERS]  # fmt: skip
    s += [{"group": "B1", "a": f"{lm}__orig", "b": f"{lm}__gemini"} for lm in LLMS]
    s += [
        {"group": "W_llm", "a": f"{x}__gemini", "b": f"{y}__gemini"}
        for x, y in itertools.combinations(LLMS, 2)
    ]
    s += [{"group": "W_vit", "a": x, "b": y} for x, y in itertools.combinations(VITS, 2)]
    s += [{"group": "D_random", "a": RANDOM, "b": f"{lm}__gemini"} for lm in LLMS]
    s += [{**x, "group": "D_shuffle", "shuffle": True} for x in s if x["group"] == "B3_gemini"]
    # within-modality references at the last layer (the mutual-kNN best cell is often an early,
    # input-level layer: LLM embeddings of the same text, early ViT blocks)
    s += [{**x, "group": f"{x['group']}_last", "cell": ("last", "last")}
          for x in s if x["group"] in ("W_llm", "W_vit")]  # fmt: skip
    for x in s:
        x["id"] = f"{x['group']}__{x['a']}__{x['b']}"
    return s


class FeatureCache:
    def __init__(self, root: Path, keep: int = 4):
        self.root, self.keep, self.data = root, keep, {}
        spec = importlib.util.spec_from_file_location(
            "e16m", HERE.parent / "exp-016-positive-controls" / "measure.py"
        )
        self.e16 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.e16)

    def get(self, key: str) -> tuple[list[torch.Tensor], list[str]]:
        if key not in self.data:
            if len(self.data) >= self.keep:
                self.data.pop(next(iter(self.data)))
            self.data[key] = self.e16.prepared_layers(self.root / f"{key}__d2l.pt")
        self.data[key] = self.data.pop(key)  # LRU order
        return self.data[key]


def best_cell(la, na, lb, nb, device) -> tuple[int, int, float]:
    """Layer pair with the largest (uncalibrated) mutual kNN, k = 10, on all samples."""

    def knn(x):
        x = F.normalize(x.to(device), dim=-1)
        res = []
        for i0 in range(0, x.shape[0], 2048):
            s = x[i0 : i0 + 2048] @ x.T
            s[torch.arange(s.shape[0]), torch.arange(i0, i0 + s.shape[0])] = -float("inf")
            res.append(s.topk(K_MKNN, dim=1).indices)
        return torch.cat(res)

    ka = [knn(x) for x in la]
    kb = [knn(x) for x in lb]
    best = (-1.0, 0, 0)
    for i, j in itertools.product(range(len(ka)), range(len(kb))):
        ov = (ka[i][:, :, None] == kb[j][:, None, :]).any(2).float().mean()
        if float(ov) > best[0]:
            best = (float(ov), i, j)
    return best[1], best[2], best[0]


def cmd_real(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "real.jsonl"
    done = set()
    if path.exists():
        done = {(r["id"], r["dir"], r["fold"]) for r in map(json.loads, path.open())}
    cache = FeatureCache(args.features)
    specs = [s for s in real_specs() if not args.groups or s["group"] in args.groups]
    with path.open("a") as sink:
        for s in specs:
            fold_ids = [0] if s.get("shuffle") else list(range(FOLDS))
            if all((s["id"], d, f) in done for d in ("a2b", "b2a") for f in fold_ids):
                continue
            la, na = cache.get(s["a"])
            lb, nb = cache.get(s["b"])
            if "cell" in s:

                def pick(names, c):
                    return (
                        max(i for i, nm in enumerate(names) if nm != "final")
                        if c == "last"
                        else names.index(c)
                    )

                i, j, ov = pick(na, s["cell"][0]), pick(nb, s["cell"][1]), None
            else:
                ln = [x for x, nm in zip(la, na, strict=True) if nm != "final"]
                lm = [x for x, nm in zip(lb, nb, strict=True) if nm != "final"]
                i, j, ov = best_cell(ln, na, lm, nb, args.device)
            a, b = la[i], lb[j]
            if s.get("shuffle"):
                b = b[torch.from_numpy(np.random.default_rng(12345).permutation(b.shape[0]))]
            meta = {**{k: v for k, v in s.items() if k != "cell"}, "layer_a": na[i], "layer_b": nb[j],
                    "mknn_cell": ov, "n": a.shape[0]}  # fmt: skip
            run_pair(a, b, meta, args.seed, fold_ids, args.device, sink)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conds", nargs="+", default=list(SYNTH))
    p.add_argument("--ms", nargs="+", type=int, default=[12, 32])
    p.add_argument("--n", type=int, default=10240)
    p.add_argument("--folds", type=int, default=FOLDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p = sub.add_parser("real")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    {"synthetic": cmd_synthetic, "real": cmd_real}[args.cmd](args)


if __name__ == "__main__":
    main()
