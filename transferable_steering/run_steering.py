import argparse
import gc
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks.months import MONTHS, get_months_dataset


MODEL_A = "meta-llama/Meta-Llama-3.1-8B"  # Model A
MODEL_B = "Qwen/Qwen3-8B-Base"            # Model B
LAYER_FRAC = 0.875
DATASET = "months"

# Ground-truth neighbours per node for each dataset (months: a ring => 2). Used as the default K for mKNN.
DATASET_K = {"months": 2}


# --------------------------------------------------------------------------------------
# Model / data plumbing (unchanged)
# --------------------------------------------------------------------------------------
def load(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device).eval()
    return tok, model


def layer_index(model, frac=LAYER_FRAC):
    """Index of the decoder layer at depth `frac` (1.0 -> last layer; its output is pre final-norm)."""
    n = model.config.num_hidden_layers
    return min(int(frac * n), n - 1)


def month_token_ids(tok):
    ids = []
    for m in MONTHS:
        t = tok.encode(" " + m, add_special_tokens=False)
        if len(t) != 1:
            print(f"[warn] ' {m}' is {len(t)} tokens in this tokenizer; using the first")
        ids.append(t[0])
    assert len(set(ids)) == 12, "month first-tokens collide"
    return torch.tensor(ids)


def build_dataset():
    texts, ans = get_months_dataset()
    return Dataset.from_dict({"text": texts, "answer": [MONTHS.index(a) for a in ans]})


@torch.no_grad()
def run_model(tok, model, texts, layer, month_ids, batch_size=32):
    """
    Returns
      acts:  (N, d)  residual stream after decoder layer `layer`, at the last token
      probs: (N, 12) P(first token = month m), from the full-vocab softmax (rows sum to <= 1)
    """
    store = {}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store["h"] = h[:, -1, :].float().cpu()

    handle = model.model.layers[layer].register_forward_hook(hook)
    acts, probs = [], []
    try:
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            acts.append(store["h"])
            probs.append(torch.softmax(logits, -1)[:, month_ids.to(logits.device)].cpu())
    finally:
        handle.remove()
    return torch.cat(acts), torch.cat(probs)


def gate(probs, answers):
    """Accuracy over the 12 month tokens, and mean probability mass on them ('other' = 1 - mass)."""
    acc = (probs.argmax(-1) == torch.as_tensor(answers)).float().mean().item()
    mass = probs.sum(-1).mean().item()
    return acc, mass


def centroids(acts, answers):
    """(12, d): mean activation over prompts whose correct ANSWER is month m."""
    answers = torch.as_tensor(answers)
    return torch.stack([acts[answers == m].mean(0) for m in range(12)])


def circ_dist(a, b, n=12):
    d = abs(a - b) % n
    return min(d, n - d)


# --------------------------------------------------------------------------------------
# Concept graph: hard-coded ring, or mutual-kNN on centroid similarity   (numpy only)
# --------------------------------------------------------------------------------------
def default_k(dataset, n_nodes):
    """
    K for the mKNN graph as a function of the dataset: the ground-truth number of neighbours per
    concept. months -> 2 (ring). Unknown datasets fall back to ceil(log2(n)), clamped to [2, n-1].
    """
    if dataset in DATASET_K:
        return min(DATASET_K[dataset], n_nodes - 1)
    return int(min(max(2, np.ceil(np.log2(n_nodes))), n_nodes - 1))


def ring_graph(n=12):
    A = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def centroid_similarity(cents):
    """Cosine similarity between mean-centred centroids (the shared mean would otherwise dominate)."""
    C = (cents.cpu().numpy() if torch.is_tensor(cents) else np.asarray(cents)).astype(np.float64)
    C = C - C.mean(0, keepdims=True)
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
    return C @ C.T


def _connect_components(A, S):
    """Add the most-similar cross-component edge until the graph is connected. Returns (A_union, aug_mask)."""
    A, aug = A.copy(), np.zeros_like(A)
    while True:
        n_comp, lab = connected_components(csr_matrix(A), directed=False)
        if n_comp == 1:
            return A, aug
        cross = lab[:, None] != lab[None, :]
        i, j = np.unravel_index(np.argmax(np.where(cross, S, -np.inf)), S.shape)
        A[i, j] = A[j, i] = aug[i, j] = aug[j, i] = 1


def mutual_knn_graph(cents, k, connect=True):
    """
    Undirected graph with edge i-j iff j is in kNN(i) AND i is in kNN(j) under centroid cosine similarity.
    Mutual kNN can leave nodes isolated or the graph disconnected; with connect=True the most similar
    cross-component edges are added until every pair has a route.
    Returns (A_union, similarity, aug): A_union = mutual edges + artificial edges (used for routing);
    aug = 0/1 mask of ONLY the artificial edges, so the pure mutual-kNN graph is A_union - aug.
    """
    S = centroid_similarity(cents)
    n = S.shape[0]
    S_off = S.copy()
    np.fill_diagonal(S_off, -np.inf)
    nn = np.argsort(-S_off, axis=1)[:, :k]
    knn = np.zeros((n, n), dtype=bool)
    knn[np.arange(n)[:, None], nn] = True
    A = (knn & knn.T).astype(np.uint8)
    aug = np.zeros_like(A)
    if connect:
        A, aug = _connect_components(A, S_off)
    return A, S, aug


def joint_graph(A_a, A_b):
    """Edges present in BOTH models' graphs (mKNN-A+B). May be disconnected; saved for analysis only."""
    return (A_a.astype(bool) & A_b.astype(bool)).astype(np.uint8)


def graph_routes(adj, pairs):
    """Fewest-hop route (list of node ids, source first) for each (x, y) in pairs."""
    _, pred = shortest_path(csr_matrix(adj), method="D", directed=False, unweighted=True, return_predecessors=True)
    routes = []
    for x, y in pairs:
        r, cur = [y], y
        while cur != x:
            cur = pred[x, cur]
            assert cur >= 0, f"no route {x}->{y}; graph is disconnected"
            r.append(int(cur))
        routes.append(r[::-1])
    return routes


def save_graph(adj, path, sim=None, aug=None):
    """adj saved as given. With aug (artificial-edge mask): `path` holds the PURE mutual-kNN edges, plus
    `_augmented.npy` (artificial edges only) and `_connected.npy` (union, what routing uses)."""
    if aug is None:
        np.save(path, adj)
    else:
        np.save(path, adj - aug)
        np.save(path.replace(".npy", "_augmented.npy"), aug)
        np.save(path.replace(".npy", "_connected.npy"), adj)
    if sim is not None:
        np.save(path.replace(".npy", "_similarity.npy"), sim)


# --------------------------------------------------------------------------------------
# Manifold: PCA subspace + spline along a route of centroids   (numpy only)
# --------------------------------------------------------------------------------------
class Manifold:
    """
    PCA is fit on the prompt activations (as in the paper; centroids alone give at most 11 components),
    centroids are projected into it, and a cubic spline through the centroids on a route gives the path.
    z_at(route, f) are PCA coordinates at fraction f in [0, 1] along the route (f=0 source, f=1 target).
    """

    def __init__(self, cents, acts=None, n_pca=8):
        C = (cents.cpu().numpy() if torch.is_tensor(cents) else np.asarray(cents)).astype(np.float64)
        fit_on = C if acts is None else (acts.cpu().numpy() if torch.is_tensor(acts) else np.asarray(acts)).astype(np.float64)
        self.pca = PCA(n_components=n_pca).fit(fit_on)
        self.W = self.pca.components_.astype(np.float32)   # (k, d)
        self.mu = self.pca.mean_.astype(np.float32)        # (d,)
        self.Z = self.pca.transform(C)                     # (12, k) centroid coordinates
        Cc = C - C.mean(0)
        recon = (C - self.pca.mean_) @ self.pca.components_.T @ self.pca.components_ + self.pca.mean_
        Rc = recon - recon.mean(0)
        self.captured = 1 - ((Cc - Rc) ** 2).sum() / (Cc ** 2).sum()   # centroid variance inside the subspace
        self._splines = {}

    def z_at(self, route, f):
        key = tuple(route)
        if key not in self._splines:
            self._splines[key] = CubicSpline(np.arange(len(route)), self.Z[list(route)])
        return self._splines[key](f * (len(route) - 1))


# --------------------------------------------------------------------------------------
# Steering. One hook helper, four public methods: {linear, manifold} x {additive, replace}
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _steer(tok, model, texts, layer, edit, month_ids, batch_size=32):
    """Apply edit(h_last) -> h_last_new at the last token of decoder layer `layer`; return month-token probs."""
    month_ids = month_ids.to(model.device)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] = edit(h[:, -1, :])
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    probs = []
    try:
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            probs.append(torch.softmax(logits, -1)[:, month_ids].cpu())
    finally:
        handle.remove()
    return torch.cat(probs)


def steer_linear_additive(tok, model, texts, layer, cent_src, cent_tgt, f, month_ids, batch_size=32):
    """h <- h + f * (c_tgt - c_src). Keeps everything the prompt carries."""
    vec = torch.as_tensor(cent_tgt - cent_src, dtype=torch.float32)
    return _steer(tok, model, texts, layer, lambda hl: hl + (f * vec).to(device=hl.device, dtype=hl.dtype),
                  month_ids, batch_size)


def steer_linear_replace(tok, model, texts, layer, cent_src, cent_tgt, f, month_ids, batch_size=32):
    """h <- (1 - f) c_src + f c_tgt. The WHOLE last-token residual is overwritten (paper's linear baseline)."""
    point = torch.as_tensor((1 - f) * cent_src + f * cent_tgt, dtype=torch.float32)
    return _steer(tok, model, texts, layer, lambda hl: point.to(device=hl.device, dtype=hl.dtype).expand_as(hl),
                  month_ids, batch_size)


def steer_manifold_additive(tok, model, texts, layer, manifold, route, f, month_ids, batch_size=32):
    """h <- h + (path(f) - path(0)), path = spline through the centroids on `route`; lives in the PCA subspace."""
    vec = torch.as_tensor((manifold.z_at(route, f) - manifold.z_at(route, 0.0)) @ manifold.W, dtype=torch.float32)
    return _steer(tok, model, texts, layer, lambda hl: hl + vec.to(device=hl.device, dtype=hl.dtype),
                  month_ids, batch_size)


def steer_manifold_replace(tok, model, texts, layer, manifold, route, f, month_ids, batch_size=32):
    """
    Replace only the PCA-subspace coordinates with path(f); keep the prompt's remainder r = h - mu - W^T W (h - mu):
        h <- mu + W^T z_path(f) + r  =  h + (z_path(f) - z_prompt) W,   z_prompt = W (h - mu)
    """
    z_t, cache = None, {}

    def edit(hl):
        nonlocal z_t
        if "W" not in cache or cache["W"].device != hl.device:
            cache["W"] = torch.as_tensor(manifold.W, device=hl.device)
            cache["mu"] = torch.as_tensor(manifold.mu, device=hl.device)
            z_t = torch.as_tensor(manifold.z_at(route, f), dtype=torch.float32, device=hl.device)
        x = hl.float()
        z_p = (x - cache["mu"]) @ cache["W"].T
        return (x + (z_t - z_p) @ cache["W"]).to(hl.dtype)

    return _steer(tok, model, texts, layer, edit, month_ids, batch_size)


def steer_manifold_route_transfer(tok, model, texts, layer, manifold_tgt, route_src, f, month_ids,
                                  mode="replace", batch_size=32):
    """
    Route transfer A -> B. `route_src` is the shortest route between the two concepts in MODEL A's graph (node ids =
    concept names, which are shared across models). Model B then builds its OWN spline through ITS OWN centroids
    along that route (manifold_tgt = B's Manifold) and steers along it with the chosen mode. Nothing from A's
    activation space is used, only the sequence of concept names, so d_model need not match.
    """
    steer = steer_manifold_replace if mode == "replace" else steer_manifold_additive
    return steer(tok, model, texts, layer, manifold_tgt, route_src, f, month_ids, batch_size)


METHODS = {
    "additive": (steer_linear_additive, steer_manifold_additive),
    "replace": (steer_linear_replace, steer_manifold_replace),
}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
CONDITIONS = {
    "linear_steering_A":       ("linear A→A",           "#1f77b4", "-"),
    "linear_steering_B":       ("linear B→B",           "#1f77b4", "--"),
    "linear_transfer_naive":   ("linear A→B (naive)",   "#1f77b4", ":"),
    "manifold_steering_A":     ("manifold A→A",         "#d95f02", "-"),
    "manifold_steering_B":     ("manifold B→B",         "#d95f02", "--"),
    "manifold_transfer_naive": ("manifold A→B (naive)", "#d95f02", ":"),
    "manifold_route_transfer": ("manifold A-route→B",   "#2ca02c", "-."),
}


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _circ(a, b, n=12):
    d = np.mod(a - b, n)
    return np.minimum(d, n - d)


def evaluate_steering_ensemble(results, out_png="months_steering.png", title=None):
    """
    results[cond]: (P pairs, S steps, n_src prompts, 12) month-token probabilities.
    results["pairs"]: list of (source, target); results["fracs"]: the S step fractions.
    Per condition and step, averaged over prompts (then pairs):
      p_target   P(target month), renormalized over the 12 month tokens
      d_intended expected circular distance to the intended arc position x + f*delta, always measured against
                 the TRUE month ring (so a graph route that leaves the ring is penalised)
      off_mass   1 - sum of month-token probabilities (fluency check)
    """
    pairs, fracs = results["pairs"], np.asarray(results["fracs"])
    xs = np.array([x for x, _ in pairs])
    ys = np.array([y for _, y in pairs])
    delta = np.mod(ys - xs + 6, 12) - 6
    u = np.mod(xs[:, None] + fracs[None, :] * delta[:, None], 12)
    months = np.arange(12)

    summary = {}
    for key in CONDITIONS:
        if key not in results:
            continue
        T = _np(results[key]).astype(np.float64)
        P, S, n, _ = T.shape
        mass = T.sum(-1)
        Pn = T / np.clip(mass[..., None], 1e-12, None)

        p_target = np.take_along_axis(Pn, ys[:, None, None, None].repeat(S, 1).repeat(n, 2), axis=-1)[..., 0].mean(-1)
        p_source0 = np.take_along_axis(Pn[:, 0], xs[:, None, None].repeat(n, 1), axis=-1)[..., 0].mean(-1)
        D = _circ(months[None, None, :], u[:, :, None])
        d_intended = (Pn * D[:, :, None, :]).sum(-1).mean(-1)
        off_mass = (1 - mass).mean(-1)
        summary[key] = dict(p_target=p_target, d_intended=d_intended, off_mass=off_mass, p_source0=p_source0)

    def ms(a):
        return f"{a.mean():.3f}±{a.std(ddof=1) / np.sqrt(len(a)):.3f}"

    print(f"{'condition':24s} {'P(src)@f=0':>13s} {'P(tgt)@f=1':>13s} {'d_intended(mean f)':>20s} {'off_mass@f=.5':>15s} {'off_mass@f=1':>14s}")
    mid = int(np.argmin(np.abs(fracs - 0.5)))
    for key, s in summary.items():
        print(f"{CONDITIONS[key][0]:24s} {ms(s['p_source0']):>13s} {ms(s['p_target'][:, -1]):>13s} "
              f"{ms(s['d_intended'].mean(1)):>20s} {ms(s['off_mass'][:, mid]):>15s} {ms(s['off_mass'][:, -1]):>14s}")
    print("mean ± s.e.m. over pairs. Under replacement, f=0 overwrites the prompt with the SOURCE centroid (linear: whole "
          "residual; manifold: subspace only), so P(src)@f=0 is not the unsteered value and need not match between "
          "linear and manifold.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("p_target", "P(target month)"), ("d_intended", "expected circ. dist to intended position"),
              ("off_mass", "mass off month tokens")]
    for ax, (m, ttl) in zip(axes, panels):
        for key, s in summary.items():
            label, c, ls = CONDITIONS[key]
            ax.plot(fracs, s[m].mean(0), color=c, ls=ls, label=label)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("f (fraction of the way to target)")
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95 if title else 1])
    fig.savefig(out_png, dpi=140)
    return summary


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["additive", "replace"], default="replace",
                   help="additive: h += delta.  replace: overwrite the last-token residual (linear) or the PCA-subspace "
                        "coordinates (manifold) with the interpolated point.")
    p.add_argument("--graph", choices=["ring", "mknn"], default="ring",
                   help="ring: hard-coded month ring.  mknn: mutual-kNN graph on mean-centred centroid cosine similarity, "
                        "computed per model; manifold routes follow its shortest paths.")
    p.add_argument("--k", type=int, default=None, help="K for mKNN. Default: default_k(dataset, n_nodes) (months: 2).")
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--n-pca", type=int, default=8)
    p.add_argument("--layer-sweep", action="store_true",
                   help="repeat everything at depths 1/8 ... 8/8 of num_hidden_layers (8x the compute). "
                        "Default: single layer at LAYER_FRAC.")
    p.add_argument("--out-dir", default=".")
    return p.parse_args()


def plot_concept_graphs(adjs, cents=None, out_png="graphs.png", title=None, augs=None):
    """
    adjs: {"A": adj, "B": adj, "joint": adj} (any subset); cents: {"A": (12,d), "B": (12,d)} optional.
    Top row: nodes on the true month ring; gray = ring edge, red = off-ring edge (the graph disagrees with the ring).
    Bottom row (needs cents): nodes at their top-2 PCA coordinates of the centroids, so edges are drawn where the
    model actually places the months.
    """
    n = 12
    ring = ring_graph(n).astype(bool)
    ang = 2 * np.pi * np.arange(n) / n
    circ = np.stack([np.sin(ang), np.cos(ang)], 1)      # Jan at top, clockwise
    names = list(adjs)
    rows = 2 if cents else 1
    fig, axes = plt.subplots(rows, len(names), figsize=(4.2 * len(names), 4.2 * rows), squeeze=False)
    for j, nm in enumerate(names):
        A = np.asarray(adjs[nm]).astype(bool)
        for r in range(rows):
            ax = axes[r, j]
            if r == 0:
                xy = circ
            elif nm in cents:
                C = _np(cents[nm]).astype(np.float64)
                C = C - C.mean(0)
                xy = C @ np.linalg.svd(C, full_matrices=False)[2][:2].T
            else:
                ax.axis("off")
                continue
            G = np.asarray((augs or {}).get(nm, np.zeros_like(A))).astype(bool)   # artificial edges
            for i, k in zip(*np.nonzero(np.triu(A, 1))):
                on = ring[i, k]
                if G[i, k]:   # added only to connect components: blue dashed, whether or not it is on the ring
                    ax.plot(*xy[[i, k]].T, color="#1f77b4", lw=2.0, ls="--", zorder=1)
                else:
                    ax.plot(*xy[[i, k]].T, color="#999999" if on else "#d62728", lw=1.6 if on else 2.0, zorder=1)
            ax.scatter(*xy.T, s=260, color="white", edgecolor="black", zorder=2)
            for i in range(n):
                ax.text(*xy[i], MONTHS[i][:3], ha="center", va="center", fontsize=7, zorder=3)
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                M = A & ~G   # mutual-kNN edges only
                n_e, n_off, n_aug = int(M.sum() // 2), int((np.triu(M, 1) & ~ring).sum()), int(G.sum() // 2)
                ax.set_title(f"{nm}: {n_e} mutual edges, {n_off} off-ring" +
                             (f", +{n_aug} artificial (blue)" if n_aug else "") +
                             (f", {int(((~A) & ring).sum() // 2)} ring edges missing" if n_off or n_e < n else ""),
                             fontsize=9)
            else:
                ax.set_title(f"{nm}: top-2 PCs of centroids", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_layer_sweep(sweep_summary, depths, out_png):
    """sweep_summary[depth][cond] = summary dict from evaluate_steering_ensemble."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("p_target", "P(target) at f=1", lambda s: s["p_target"][:, -1].mean()),
              ("d_intended", "d_intended (mean over f)", lambda s: s["d_intended"].mean()),
              ("off_mass", "off-month mass at f=1", lambda s: s["off_mass"][:, -1].mean())]
    for ax, (_, ttl, fn) in zip(axes, panels):
        for key, (label, c, ls) in CONDITIONS.items():
            ys = [fn(sweep_summary[d][key]) for d in depths if key in sweep_summary[d]]
            if ys:
                ax.plot(depths, ys, color=c, ls=ls, marker="o", ms=3, label=label)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("layer depth (fraction of num_hidden_layers)")
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_png, dpi=140)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    steer_linear, steer_manifold = METHODS[args.mode]
    depths = [i / 8 for i in range(1, 9)] if args.layer_sweep else [LAYER_FRAC]

    ds = build_dataset()
    texts, answers = ds["text"], ds["answer"]

    rng = np.random.default_rng(0)
    all_pairs = list(itertools.permutations(range(12), 2))
    idx = rng.choice(len(all_pairs), size=args.n_pairs, replace=False)
    steer_pairs = [all_pairs[i] for i in idx]
    fracs = [k / args.n_steps for k in range(args.n_steps + 1)]

    src_texts = {x: [t for t, a in zip(texts, answers) if a == x] for x in range(12)}
    n_src = len(src_texts[0])
    assert all(len(v) == n_src for v in src_texts.values())

    k = args.k if args.k is not None else default_k(DATASET, 12)

    def sweep(tok, model, layer, month_ids, cent, manifold, routes):
        lin_all, man_all = [], []
        for (x, y), route in zip(steer_pairs, routes):
            for f in fracs:
                lin_all.append(steer_linear(tok, model, src_texts[x], layer, cent[x], cent[y], f, month_ids))
                man_all.append(steer_manifold(tok, model, src_texts[x], layer, manifold, route, f, month_ids))
        shape = (args.n_pairs, len(fracs), n_src, 12)
        return torch.stack(lin_all).reshape(shape), torch.stack(man_all).reshape(shape)

    def route_transfer_sweep(tok, model, layer, month_ids, manifold_B, routes_A):
        out = []
        for (x, y), route in zip(steer_pairs, routes_A):
            for f in fracs:
                out.append(steer_manifold_route_transfer(tok, model, src_texts[x], layer, manifold_B, route, f,
                                                         month_ids, mode=args.mode))
        return torch.stack(out).reshape(args.n_pairs, len(fracs), n_src, 12)

    # keyed by depth: state[d][name] = (cents, manifold, routes);  results[d] = {cond: tensor}
    state = {d: {} for d in depths}
    results = {d: {} for d in depths}
    adjs = {d: {} for d in depths}
    augs = {d: {} for d in depths}

    for name, path in (("A", MODEL_A), ("B", MODEL_B)):
        tok, model = load(path)
        month_ids = month_token_ids(tok)

        for d in depths:
            layer = layer_index(model, d)
            acts, p = run_model(tok, model, texts, layer, month_ids)
            acc, mass = gate(p, answers)
            print(f"{name} depth={d:.3f} layer={layer} accuracy={acc:.3f} month-token mass={mass:.3f}")
            if acc < 0.5:
                print(f"[warn] {name} layer {layer} is below 50% on the task; steering results will be hard to interpret")

            cent = centroids(acts, answers)
            manifold = Manifold(cent, acts=acts, n_pca=args.n_pca)
            print(f"{name} depth={d:.3f}: {args.n_pca} PCA dims capture {manifold.captured:.3f} of centroid variance")
            del acts

            stem = f"d{d:.3f}_{name}"
            if args.graph == "ring":
                adj = ring_graph()
                aug = np.zeros_like(adj)
                save_graph(adj, os.path.join(args.out_dir, f"adjacency_ring_{stem}.npy"))
            else:
                adj, sim, aug = mutual_knn_graph(cent, k)
                save_graph(adj, os.path.join(args.out_dir, f"adjacency_mknn_k{k}_{stem}.npy"), sim, aug)
                print(f"{name} depth={d:.3f}: mKNN k={k}: {int((adj - aug).sum() // 2)} mutual edges"
                      + (f" + {int(aug.sum() // 2)} artificial edges added to connect components" if aug.any() else ""))
            routes = graph_routes(adj, steer_pairs)
            adjs[d][name] = adj
            augs[d][name] = aug
            state[d][name] = (cent, manifold, routes)

            results[d][f"linear_steering_{name}"], results[d][f"manifold_steering_{name}"] = sweep(
                tok, model, layer, month_ids, cent, manifold, routes)

            if name == "B":
                cA, mA, rA = state[d]["A"]
                assert cA.shape[1] == model.config.hidden_size, "d_model must be the same for naive transfer"
                results[d]["linear_transfer_naive"], results[d]["manifold_transfer_naive"] = sweep(
                    tok, model, layer, month_ids, cA, mA, rA)

                rB = state[d]["B"][2]
                n_diff = sum(a != b for a, b in zip(rA, rB))
                print(f"depth={d:.3f}: A's route differs from B's own route for {n_diff}/{len(rA)} pairs")
                if n_diff == 0:   # identical routes -> route transfer == B->B manifold; skip the compute
                    results[d]["manifold_route_transfer"] = results[d]["manifold_steering_B"]
                else:
                    results[d]["manifold_route_transfer"] = route_transfer_sweep(
                        tok, model, layer, month_ids, state[d]["B"][1], rA)

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    sweep_summary = {}
    for d in depths:
        tag = f"{args.mode}_{args.graph}_d{d:.3f}"
        if args.graph == "mknn":
            save_graph(joint_graph(adjs[d]["A"] - augs[d]["A"], adjs[d]["B"] - augs[d]["B"]),
                       os.path.join(args.out_dir, f"adjacency_mknn_k{k}_d{d:.3f}_joint.npy"))
        # how often does each route equal the true shorter ring arc (ties at distance 6 excluded)?
        for name in ("A", "B"):
            ok = [r == [(x + s * i) % 12 for i in range(len(r))]
                  for (x, y), r in zip(steer_pairs, state[d][name][2])
                  for s in [1 if (y - x) % 12 <= 6 else -1] if (y - x) % 12 != 6]
            print(f"depth={d:.3f} {name}: {np.mean(ok):.2f} of routes equal the ring's shorter arc")

        gdict = dict(adjs[d])
        if args.graph == "mknn":
            gdict["joint"] = joint_graph(adjs[d]["A"] - augs[d]["A"], adjs[d]["B"] - augs[d]["B"])
        plot_concept_graphs(gdict, {n: state[d][n][0] for n in ("A", "B")},
                            os.path.join(args.out_dir, f"graph_{args.graph}_d{d:.3f}.png"),
                            title=f"{args.graph} graph" + (f" (k={k})" if args.graph == "mknn" else "") + f", depth={d:.3f}", augs=augs[d])

        res = results[d]
        res["pairs"], res["fracs"] = steer_pairs, fracs
        res["routes"] = {n: state[d][n][2] for n in ("A", "B")}
        res["mode"], res["graph"], res["depth"], res["adjacency"] = args.mode, args.graph, d, adjs[d]
        res["artificial_edges"] = augs[d]
        torch.save(res, os.path.join(args.out_dir, f"months_results_{tag}.pt"))
        print(f"\n=== depth {d:.3f} ===")
        sweep_summary[d] = evaluate_steering_ensemble(
            res, os.path.join(args.out_dir, f"months_steering_{tag}.png"),
            title=f"mode={args.mode}, graph={args.graph}, depth={d:.3f}")

    if args.layer_sweep:
        plot_layer_sweep(sweep_summary, depths,
                         os.path.join(args.out_dir, f"months_layer_sweep_{args.mode}_{args.graph}.png"))


if __name__ == "__main__":
    run(parse_args())