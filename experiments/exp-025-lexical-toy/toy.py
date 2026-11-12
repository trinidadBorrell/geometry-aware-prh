"""exp-025: toy examples for the lexical floor. No real data, no language model.

Part 1, six captions: the exact bag-of-words cosines, next to a simulated vision encoder that never
sees a word. Both are functions of the same scene.

Part 2, simulation. Each scene has 8 attributes a caption can mention and 8 it never mentions, each with
6 values. A "vision model" is a random tanh network on all 16. A caption names m of the 8 mentionable
attributes, one word each, plus 3 function words per content word, drawn from a Zipf law. Nothing is
trained, and the two sides share only the scene.

    uv run python experiments/exp-025-lexical-toy/toy.py --out results/exp-025-lexical-toy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MENTION, N_PRIVATE, N_VALUES = 8, 8, 6
N_FUNCTION = 50
WIDTH = 1024
K = 10
TABLE_D = 768
DETAILS = (1, 2, 4, 8)
SEEDS = range(5)
VOCABS = (10, 100, 1000, 10000)
SERIES = {
    "content": ("Content words only", "#2a78d6", "-"),
    "tfidf": ("TF-IDF", "#1baf7a", "-"),
    "table": ("TF x random table (untrained layer 0)", "#eb6834", "-"),
    "tf": ("TF", "#8a63d2", "-"),
    "shuffled": ("TF-IDF, pairing shuffled", "#52514e", "--"),
}

CAPTIONS = [
    "a brown dog on the grass",
    "a black dog on the grass",
    "a brown cat on the sofa",
    "a black cat on the sofa",
    "a red car on the road",
    "a blue car on the road",
]
SCENES = [
    {"object": "dog", "colour": "brown", "place": "grass"},
    {"object": "dog", "colour": "black", "place": "grass"},
    {"object": "cat", "colour": "brown", "place": "sofa"},
    {"object": "cat", "colour": "black", "place": "sofa"},
    {"object": "car", "colour": "red", "place": "road"},
    {"object": "car", "colour": "blue", "place": "road"},
]


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def encoder(x: np.ndarray, seed: int, noise: float = 0.0) -> np.ndarray:
    """A random 2-layer tanh network. Its weights are never fitted to anything."""
    rng = np.random.default_rng(seed)
    h = x
    for _ in range(2):
        w = rng.normal(size=(h.shape[1], WIDTH)) / np.sqrt(h.shape[1])
        b = rng.normal(size=WIDTH) * 0.1
        h = np.tanh(h @ w + b)
    if noise:
        h = h + noise * rng.normal(size=h.shape)
    return unit(h)


def centred_unit(x: np.ndarray) -> np.ndarray:
    k = x @ x.T
    k = k - k.mean(0, keepdims=True)
    k = k - k.mean(1, keepdims=True)
    return k / np.linalg.norm(k)


def cka_and_null(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Linear CKA of l2-normalised rows, and its exact mean under a random pairing."""
    ca, cb = centred_unit(a), centred_unit(b)
    return float((ca * cb).sum()), float(np.trace(ca) * np.trace(cb) / (len(a) - 1))


def neighbours(x: np.ndarray, k: int) -> np.ndarray:
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def mutual_knn(a: np.ndarray, b: np.ndarray, k: int) -> float:
    na, nb = neighbours(a, k), neighbours(b, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(na, nb, strict=True)]))


def part1(out: Path) -> dict:
    names = sorted({v for s in SCENES for v in s.values()})
    x = np.array([[float(n in s.values()) for n in names] for s in SCENES])
    vision = encoder(x, seed=0)
    full = unit(_count(CAPTIONS)[0])
    content = unit(_count(CAPTIONS, drop={"a", "on", "the"})[0])
    res = {
        "word_cosine": np.round(full @ full.T, 3).tolist(),
        "content_cosine": np.round(content @ content.T, 3).tolist(),
        "vision_cosine": np.round(vision @ vision.T, 3).tolist(),
    }
    for name, w in (("all words", full), ("content words", content)):
        c, mu = cka_and_null(w, vision)
        res[name] = {"cka": c, "cka_null": mu, "knn1": mutual_knn(w, vision, 1)}

    labels = [" ".join(c.split()[1:3]) for c in CAPTIONS]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    mats = (
        (full @ full.T, "Word kernel: cosine of word counts"),
        (content @ content.T, "Same, without a, on, the"),
        (vision @ vision.T, "Random vision encoder, no words"),
    )
    for ax, (m, title) in zip(axes, mats, strict=True):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(6), labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticks(range(6), labels, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        for i in range(6):
            for j in range(6):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if m[i, j] > 0.6 else "#26251f",
                )
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_toy_kernels.{ext}", dpi=200)
    plt.close(fig)
    return res


def _count(texts: list[str], drop=frozenset()) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({w for t in texts for w in t.split() if w not in drop})
    pos = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        for w in t.split():
            if w in pos:
                x[i, pos[w]] += 1
    return x, vocab


def simulate(n: int, m: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    attrs = rng.integers(N_VALUES, size=(n, N_MENTION + N_PRIVATE))
    onehot = np.eye(N_VALUES)[attrs].reshape(n, -1)
    vision = encoder(onehot, seed=1000 + seed, noise=0.02)
    zipf = 1 / np.arange(1, N_FUNCTION + 1)
    zipf = zipf / zipf.sum()
    docs, content = [], []
    for row in attrs:
        chosen = rng.choice(N_MENTION, size=m, replace=False)
        words = [f"attr{i}value{row[i]}" for i in chosen]
        function = [f"fn{j}" for j in rng.choice(N_FUNCTION, size=3 * m, p=zipf)]
        docs.append(" ".join(words + function))
        content.append(" ".join(words))
    tf = _count(docs)[0]
    tf_rows = tf / tf.sum(1, keepdims=True)
    feats = {
        "tf": unit(tf_rows),
        "tfidf": unit(TfidfVectorizer(token_pattern=r"\S+").fit_transform(docs).toarray()),
        "content": unit(TfidfVectorizer(token_pattern=r"\S+").fit_transform(content).toarray()),
        "table": unit(tf_rows @ rng.normal(size=(tf.shape[1], TABLE_D))),
    }
    feats["shuffled"] = feats["tfidf"][rng.permutation(n)]
    out = {}
    for name, x in feats.items():
        c, mu = cka_and_null(x, vision)
        out[name] = {
            "cka": c,
            "cka_null": mu,
            "cka_excess": c - mu,
            "knn": mutual_knn(x, vision, K),
            "knn_excess": mutual_knn(x, vision, K) - K / (n - 1),
        }
    return out


def random_words(n: int, vocab: int, seed: int) -> tuple[float, float]:
    """Captions of 8 words drawn at random, whatever the scene: (raw CKA, its exact null)."""
    rng = np.random.default_rng(seed)
    attrs = rng.integers(N_VALUES, size=(n, N_MENTION + N_PRIVATE))
    vision = encoder(np.eye(N_VALUES)[attrs].reshape(n, -1), seed=1000 + seed, noise=0.02)
    x = np.zeros((n, vocab))
    np.add.at(x, (np.repeat(np.arange(n), 8), rng.integers(vocab, size=8 * n)), 1.0)
    return cka_and_null(unit(x), vision)


def part2(out: Path, n: int) -> list[dict]:
    rows = []
    for m in DETAILS:
        for seed in SEEDS:
            res = simulate(n, m, seed)
            for name, r in res.items():
                rows.append({"words": m * 4, "named": m, "seed": seed, "series": name, **r})
            print(f"m = {m}, seed {seed}", flush=True)
    with (out / "toy_sim.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True}
    )
    plt.rcParams.update({"grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    panels = (
        ("knn_excess", "Mutual kNN, k = 10, excess over chance"),
        ("cka_excess", "Linear CKA, excess over null"),
    )

    def pick(series, field):
        return np.array(
            [[r[field] for r in rows if r["series"] == series and r["named"] == m] for m in DETAILS]
        )

    x = np.array(DETAILS)
    for ax, (field, title) in zip(axes, panels, strict=False):
        for name, (label, color, ls) in SERIES.items():
            y = pick(name, field)
            ax.fill_between(x, y.min(1), y.max(1), color=color, alpha=0.16, lw=0)
            ax.plot(x, y.mean(1), color=color, ls=ls, lw=2, marker="o", ms=4, label=label)
        ax.set_title(title, fontsize=9.5, loc="left")
    ax = axes[2]
    vocabs = np.array(VOCABS)
    noise = np.array([[random_words(n, v, seed) for seed in SEEDS] for v in VOCABS])
    raw, null = noise[:, :, 0], noise[:, :, 1]
    ax.fill_between(vocabs, raw.min(1), raw.max(1), color="#2a78d6", alpha=0.16, lw=0)
    ax.plot(vocabs, raw.mean(1), color="#2a78d6", lw=2, marker="o", ms=4, label="Raw CKA, measured")
    ax.plot(vocabs, null.mean(1), color="#52514e", lw=1.2, ls="--", label="Exact null")
    ax.set_xscale("log")
    ax.set_xlabel("vocabulary of random words")
    ax.set_title("Raw CKA of captions unrelated to the image", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors="#52514e", length=0)
    with (out / "toy_random_words.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vocab", "raw_mean", "raw_min", "raw_max", "null_mean"])
        for v, a, lo, hi, b in zip(
            VOCABS, raw.mean(1), raw.min(1), raw.max(1), null.mean(1), strict=True
        ):
            w.writerow([v, a, lo, hi, b])
    print("random words", np.round(raw.mean(1), 3).tolist(), np.round(null.mean(1), 3).tolist())
    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel("attributes a caption names (of 8)")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.tick_params(colors="#52514e", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=5, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_toy_sim.{ext}", dpi=200)
    plt.close(fig)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = part1(a.out)
    rows = part2(a.out, a.n)
    summary = {}
    for m in DETAILS:
        summary[m] = {}
        for name in SERIES:
            pick = [r for r in rows if r["series"] == name and r["named"] == m]
            summary[m][name] = {
                f: {
                    "mean": float(np.mean([r[f] for r in pick])),
                    "min": float(np.min([r[f] for r in pick])),
                    "max": float(np.max([r[f] for r in pick])),
                }
                for f in ("knn_excess", "cka_excess", "cka", "cka_null")
            }
    (a.out / "toy.json").write_text(json.dumps({"six": res, "sim": summary}, indent=1))
    print(json.dumps(res["word_cosine"][0]), json.dumps(res["vision_cosine"][0]))
    print({k: res[k] for k in ("all words", "content words")})
    for m in DETAILS:
        print(
            m,
            {
                s: (
                    round(summary[m][s]["knn_excess"]["mean"], 3),
                    round(summary[m][s]["cka_excess"]["mean"], 3),
                )
                for s in SERIES
            },
        )
        print(
            "  raw/null",
            {
                s: (
                    round(summary[m][s]["cka"]["mean"], 3),
                    round(summary[m][s]["cka_null"]["mean"], 3),
                )
                for s in ("tf", "tfidf", "content")
            },
        )


if __name__ == "__main__":
    main()
