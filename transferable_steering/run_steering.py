import itertools
 
from tasks.months import MONTHS, get_months_dataset

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.interpolate import CubicSpline

import torch
from datasets import Dataset 

from transformers import AutoModelForCausalLM, AutoTokenizer
 

MODEL_A = "meta-llama/Meta-Llama-3.1-8B"  # Model A
MODEL_B = "Qwen/Qwen3-8B-Base" # Model B
LAYER_FRAC = 0.875

def load(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device).eval()
    return tok, model


def layer_index(model, frac=LAYER_FRAC):
    return int(frac * model.config.num_hidden_layers)
 
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
 
@torch.no_grad()
def steer_linear(tok, model, texts, layer, vec, alpha, month_ids, batch_size=32):
    """
    Linear steering.
    tok, model: tokenizer, model
    texts: input texts
    layer: layer to add steering vector
    vec: directional steering vector, target centroid - source centroid
    alpha: steering coefficient
    month_ids: map probabilities
    """
    month_ids = month_ids.to(model.device)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] += (alpha * vec).to(device=h.device, dtype=h.dtype)
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
 
def compute_manifold(cents, n_pca=8):
    C = (cents.cpu().numpy() if torch.is_tensor(cents) else np.asarray(cents)).astype(np.float64)
    pca = PCA(n_components=n_pca).fit(C)
    Z = pca.transform(C) # (12, k)
    cs = CubicSpline(np.arange(13), np.vstack([Z, Z[:1]]), bc_type="periodic")

    def point(u): # u: month coordinate, period 12
        z = cs(np.mod(u, 12)) # (k,)
        return torch.tensor(pca.inverse_transform(z[None])[0], dtype=torch.float32)

    return point
 
def steer_manifold(tok, model, texts, layer, path_point, t0, t1, month_ids, batch_size=32):
    vec = path_point(t1) - path_point(t0) # vector lies on the manifold.
    return steer_linear(tok, model, texts, layer, vec, alpha=1.0, month_ids=month_ids, batch_size=batch_size)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = {
    "linear_steering_A":       ("linear A→A",           "#1f77b4", "-"),
    "linear_steering_B":       ("linear B→B",           "#1f77b4", "--"),
    "linear_transfer_naive":   ("linear A→B (naive)",   "#1f77b4", ":"),
    "manifold_steering_A":     ("manifold A→A",         "#d95f02", "-"),
    "manifold_steering_B":     ("manifold B→B",         "#d95f02", "--"),
    "manifold_transfer_naive": ("manifold A→B (naive)", "#d95f02", ":"),
}


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _circ(a, b, n=12):
    d = np.mod(a - b, n)
    return np.minimum(d, n - d)


def evaluate_steering_ensemble(results, out_png="months_steering.png"):
    """
    results[cond]: (P pairs, S steps, n_src prompts, 12) month-token probabilities.
    results["pairs"]: list of (source, target); results["fracs"]: the S step fractions.
    Per condition and step, averaged over prompts (then pairs):
      p_target   P(target month), renormalized over the 12 month tokens
      d_intended expected circular distance to the intended arc position x + f*delta
                 (target only at f=1; tracks intermediate months along the way)
      off_mass   1 - sum of month-token probabilities (fluency check)
    """
    pairs, fracs = results["pairs"], np.asarray(results["fracs"])
    xs = np.array([x for x, _ in pairs])
    ys = np.array([y for _, y in pairs])
    delta = np.mod(ys - xs + 6, 12) - 6                                   # signed shorter arc
    u = np.mod(xs[:, None] + fracs[None, :] * delta[:, None], 12)         # (P, S) intended position
    months = np.arange(12)

    summary = {}
    for key, (label, _, _) in CONDITIONS.items():
        if key not in results:
            continue
        T = _np(results[key]).astype(np.float64)                          # (P, S, n, 12)
        P, S, n, _ = T.shape
        mass = T.sum(-1)                                                  # (P, S, n)
        Pn = T / np.clip(mass[..., None], 1e-12, None)                    # renormalized over months

        p_target = np.take_along_axis(Pn, ys[:, None, None, None].repeat(S, 1).repeat(n, 2), axis=-1)[..., 0].mean(-1)
        p_source0 = np.take_along_axis(Pn[:, 0], xs[:, None, None].repeat(n, 1), axis=-1)[..., 0].mean(-1)  # f=0
        D = _circ(months[None, None, :], u[:, :, None])                   # (P, S, 12)
        d_intended = (Pn * D[:, :, None, :]).sum(-1).mean(-1)             # (P, S)
        off_mass = (1 - mass).mean(-1)                                    # (P, S)
        summary[key] = dict(p_target=p_target, d_intended=d_intended, off_mass=off_mass, p_source0=p_source0)

    def ms(a):
        return f"{a.mean():.3f}±{a.std(ddof=1) / np.sqrt(len(a)):.3f}"

    print(f"{'condition':24s} {'P(src)@f=0':>13s} {'P(tgt)@f=1':>13s} {'d_intended(mean f)':>20s} {'off_mass@f=.5':>15s} {'off_mass@f=1':>14s}")
    mid = int(np.argmin(np.abs(fracs - 0.5)))
    for key, s in summary.items():
        print(f"{CONDITIONS[key][0]:24s} {ms(s['p_source0']):>13s} {ms(s['p_target'][:, -1]):>13s} "
              f"{ms(s['d_intended'].mean(1)):>20s} {ms(s['off_mass'][:, mid]):>15s} {ms(s['off_mass'][:, -1]):>14s}")
    print("mean ± s.e.m. over pairs. Sanity: P(src)@f=0 should be high and match across linear/manifold of the same model.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("p_target", "P(target month)"), ("d_intended", "expected circ. dist to intended position"),
              ("off_mass", "mass off month tokens")]
    for ax, (m, title) in zip(axes, panels):
        for key, s in summary.items():
            label, c, ls = CONDITIONS[key]
            ax.plot(fracs, s[m].mean(0), color=c, ls=ls, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("f (fraction of the way to target)")
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_png, dpi=140)
    return summary

import gc
import itertools

N_PAIRS = 50
N_STEPS = 50


def run():
    ds = build_dataset()
    texts, answers = ds["text"], ds["answer"]

    rng = np.random.default_rng(0)
    all_pairs = list(itertools.permutations(range(12), 2))
    idx = rng.choice(len(all_pairs), size=N_PAIRS, replace=False)
    steer_pairs = [all_pairs[i] for i in idx]
    fracs = [k / N_STEPS for k in range(N_STEPS + 1)] 

    src_texts = {x: [t for t, a in zip(texts, answers) if a == x] for x in range(12)}
    n_src = len(src_texts[0])
    assert all(len(v) == n_src for v in src_texts.values())

    def sweep(tok, model, layer, month_ids, cent, manifold):
        lin_all, man_all = [], []
        for x, y in steer_pairs:
            delta = (y - x + 6) % 12 - 6
            vec_lin = cent[y] - cent[x]
            for f in fracs:
                lin_all.append(steer_linear(tok, model, src_texts[x], layer, vec_lin, f, month_ids))
                man_all.append(steer_manifold(tok, model, src_texts[x], layer, manifold, x, x + f * delta, month_ids))
        shape = (N_PAIRS, len(fracs), n_src, 12)
        return torch.stack(lin_all).reshape(shape), torch.stack(man_all).reshape(shape)

    cents, manifolds, results = {}, {}, {}

    for name, path in (("A", MODEL_A), ("B", MODEL_B)):
        tok, model = load(path)
        layer = layer_index(model)
        month_ids = month_token_ids(tok)

        acts, p = run_model(tok, model, texts, layer, month_ids)
        acc, mass = gate(p, answers)
        print(f"{name} layer={layer} accuracy={acc:.3f} month-token mass={mass:.3f}")
        if acc < 0.5:
            print(f"[warn] model {name} is below 50% on the task; steering results will be hard to interpret")

        cents[name] = centroids(acts, answers)
        manifolds[name] = compute_manifold(cents[name])

        results[f"linear_steering_{name}"], results[f"manifold_steering_{name}"] = sweep(
            tok, model, layer, month_ids, cents[name], manifolds[name])

        if name == "B":
            assert cents["A"].shape[1] == model.config.hidden_size, "d_model must be the same for naive transfer"
            results["linear_transfer_naive"], results["manifold_transfer_naive"] = sweep(
                tok, model, layer, month_ids, cents["A"], manifolds["A"])

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    results["pairs"] = steer_pairs
    results["fracs"] = fracs
    torch.save(results, "months_results.pt")
    evaluate_steering_ensemble(results)

if __name__ == "__main__":
    run()