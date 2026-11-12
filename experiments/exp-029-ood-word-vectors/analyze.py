"""exp-029: gap closure per word-vector corpus, coverage, predictions, figure.

Same split-half rule as exp-028: each family's configuration (dimension; for LLMs the cell) is chosen
on half A and scored on half B; closure = (fit - best floor) / (LLM - best floor), LLM = mean of 4.

    uv run python experiments/exp-029-ood-word-vectors/analyze.py --out results/exp-029-ood-word-vectors
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "exp028_analyze", HERE.parent / "exp-028-spectral-lexical-floor" / "analyze.py"
)
A28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A28)

MEAS = [("knn", "10"), ("knn", "100"), ("knn", "1000"), ("cka", None)]
ORDER = [
    "Wsame-doc",
    "Wsame-win",
    "Wcross-doc",
    "Wcross-win",
    "Wweb-1x",
    "Wweb-10x",
    "Wwiki-1x",
    "Wwiki-10x",
    "G",
    "Wshuffled",
    "Wrepro",
]
LABEL = {
    "Wsame-doc": "captions, same type, whole caption",
    "Wsame-win": "captions, same type, +-5",
    "Wcross-doc": "captions, other type, whole caption",
    "Wcross-win": "captions, other type, +-5",
    "Wweb-1x": "C4 web, 1x",
    "Wweb-10x": "C4 web, 10x",
    "Wwiki-1x": "Wikipedia, 1x",
    "Wwiki-10x": "Wikipedia, 10x",
    "G": "GloVe 6B",
    "Wshuffled": "control: words shuffled",
    "Wrepro": "exp-028 W20000 refit",
}
GENERAL = ["Wweb-1x", "Wweb-10x", "Wwiki-1x", "Wwiki-10x"]


def fam(name: str, r: dict) -> str:
    return "LLM" if r["kind"] == "llm" else name.split("__")[0]


def closures(by: dict) -> dict:
    """{(cap, target): {family: [half-B value per measure]}, plus floor, LLM, closure}."""
    out = {}
    for (split, target, cap), _ in by.items():
        if split != "A":
            continue
        rows = {s: by.get((s, target, cap), {}) for s in ("A", "B")}
        groups = defaultdict(list)
        for n, r in rows["A"].items():
            groups[fam(n, r)].append(n)
        vals = {
            f: [A28.pick(rows, ns, m, k)[0] for m, k in MEAS]
            for f, ns in groups.items()
            if f != "LLM"
        }
        vals["LLM"] = [
            float(np.mean([A28.pick(rows, [n], m, k)[0] for n in groups["LLM"]])) for m, k in MEAS
        ]
        gen = [n for f in GENERAL for n in groups.get(f, [])]
        vals["best general text (web / wiki)"] = [A28.pick(rows, gen, m, k)[0] for m, k in MEAS]
        f0, l0 = vals["F"], vals["LLM"]
        clo = {
            f: [
                (v - f_) / (l_ - f_) if l_ - f_ > 0.02 * max(abs(l_), 1e-9) else np.nan
                for v, f_, l_ in zip(vs, f0, l0, strict=True)
            ]
            for f, vs in vals.items()
            if f not in ("F", "LLM")
        }
        out[(cap, target)] = {"vals": vals, "closure": clo}
    return out


def tables(res: dict, meta: dict) -> list[str]:
    lines = ["# exp-029 summary"]
    if meta:
        lines += [
            "\n### Vocabulary coverage of the evaluation captions (share of content-word tokens)\n",
            "| Fit | original | Gemini |",
            "|---|---:|---:|",
        ]
        keys = [
            k
            for k in [
                "same-doc",
                "same-win",
                "cross-doc",
                "cross-win",
                "web-1x",
                "web-10x",
                "wiki-1x",
                "wiki-10x",
                "G",
            ]
            if k in meta.get("orig", {})
        ]
        for k in keys:
            lines.append(
                f"| {k} | {meta['orig'][k]['coverage']:.3f} | {meta['gemini'][k]['coverage']:.3f} |"
            )
    for (cap, target), d in sorted(res.items()):
        lines += [
            f"\n### {cap} captions, {target}: half B (chosen on half A), closure in brackets\n",
            "| Kernel | k = 10 | k = 100 | k = 1000 | CKA |",
            "|---|---:|---:|---:|---:|",
        ]
        for f in ["F", *ORDER, "best general text (web / wiki)", "LLM"]:
            if f not in d["vals"]:
                continue
            v = d["vals"][f]
            c = d["closure"].get(f)
            cells = [
                f"{x:.3f}" + (f" ({cc:.0%})" if c is not None and np.isfinite(cc) else "")
                for x, cc in zip(v, c or [np.nan] * 4, strict=True)
            ]
            lines.append(f"| {LABEL.get(f, f)} | " + " | ".join(cells) + " |")
    return lines


def predictions(res: dict) -> list[str]:
    def clo(cap, t, f, i):
        return res[(cap, t)]["closure"].get(f, [np.nan] * 4)[i]

    out = [
        "\n## Predictions (half B, k = 1000 and CKA, DINOv2 and CLIP)\n",
        "| | Test | Result | Verdict |",
        "|---|---|---|---|",
    ]
    ts = ["dino", "clip"]
    o1 = [
        (clo("gemini", t, "Wweb-1x", i), clo("gemini", t, "Wsame-doc", i))
        for t in ts
        for i in (2, 3)
    ]
    out.append(
        "| O1 | Gemini: web 1x closes less than same-type captions | "
        + "; ".join(f"{a:.0%} vs {b:.0%}" for a, b in o1)
        + f" | {'holds' if all(a < b for a, b in o1) else '**fails**'} |"
    )
    o2 = [
        (
            max(clo("orig", t, "Wwiki-1x", i), clo("orig", t, "Wwiki-10x", i)),
            clo("orig", t, "Wsame-doc", i),
        )
        for t in ts
        for i in (2, 3)
    ]
    out.append(
        "| O2 | Original: Wikipedia >= same-type captions | "
        + "; ".join(f"{a:.0%} vs {b:.0%}" for a, b in o2)
        + f" | {'holds' if all(a >= b for a, b in o2) else '**fails**'} |"
    )
    o3 = [
        (
            clo("gemini", t, f"W{s}-10x", i),
            clo("gemini", t, f"W{s}-1x", i),
            clo("gemini", t, "Wsame-doc", i),
        )
        for t in ts
        for s in ("web", "wiki")
        for i in (2, 3)
    ]
    ok = all(a > b and a < c for a, b, c in o3)
    out.append(
        "| O3 | Gemini: 10x > 1x, still below same-type | "
        + "; ".join(f"{a:.0%}/{b:.0%}/{c:.0%}" for a, b, c in o3)
        + f" | {'holds' if ok else '**fails**'} |"
    )
    o4 = [clo(c, t, "Wshuffled", i) for c in ("orig", "gemini") for t in ts for i in (2, 3)]
    out.append(
        "| O4 | shuffled control at or below the floor | "
        + ", ".join(f"{x:.0%}" for x in o4)
        + f" | {'holds' if all(x <= 0.05 for x in o4 if np.isfinite(x)) else '**fails**'} |"
    )
    main = [
        (c, t, res[(c, t)]["closure"]["best general text (web / wiki)"][i])
        for c in ("orig", "gemini")
        for t in ts
        for i in (2, 3)
    ]
    out += [
        "\n**Main readout (best general text, closure):** "
        + "; ".join(
            f"{c}/{t}/{'k1000' if i == 2 else 'CKA'} {x:.0%}"
            for (c, t, x), i in zip(main, [2, 3] * 4, strict=True)
        )
    ]
    return out


def figure(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fams = [f for f in ORDER if f not in ("Wrepro",)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    colors = {
        ("orig", 2): "#2a78d6",
        ("orig", 3): "#86b6ef",
        ("gemini", 2): "#eb6834",
        ("gemini", 3): "#f2a07c",
    }
    for ax, t in zip(axes, ("dino", "clip"), strict=True):
        for j, (cap, i) in enumerate(colors):
            if (cap, t) not in res:
                continue
            ys = np.array([res[(cap, t)]["closure"].get(f, [np.nan] * 4)[i] for f in fams], float)
            low = ys < -2
            ys = np.where(low, -1.95, ys)
            x = np.arange(len(fams)) + (j - 1.5) * 0.18
            ax.plot(
                x[~low],
                ys[~low],
                "o",
                ms=7,
                color=colors[(cap, i)],
                mec="#fcfcfb",
                mew=1.5,
                label=f"{cap}, {'k = 1000' if i == 2 else 'CKA'}",
            )
            ax.plot(x[low], ys[low], "v", ms=8, color=colors[(cap, i)])  # below -2, clipped
        ax.set_ylim(-2.05, 1.35)
        ax.axhline(0, color="#52514e", lw=1)
        ax.axhline(0.5, color="#9a9890", lw=1, ls="--")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([LABEL[f] for f in fams], rotation=60, ha="right", fontsize=8)
        ax.set_title(
            f"{'DINOv2' if t == 'dino' else 'CLIP'}: gap closure, half B", loc="left", fontsize=10
        )
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("(kernel - word floor) / (LLMs - word floor)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    by = A28.load(args.out)
    meta_path = args.out / "kernels_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    res = closures(by)
    lines = tables(res, meta) + predictions(res)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure(res, args.out / "fig_closure.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
