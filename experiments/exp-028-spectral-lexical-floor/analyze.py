"""exp-028: tables and figure from `rows.jsonl` (run.py `score`).

Every number is excess over the exact null. For each row (one kernel or LLM against one ViT), "best"
is the maximum over its cells (ViT layer x kernel / LLM layer). Split-half: a configuration (floor
variant; spectral dimension and weighting; LLM cell) is chosen on half A and scored on half B.

    uv run python experiments/exp-028-spectral-lexical-floor/analyze.py --out results/exp-028-spectral-lexical-floor
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MEAS_FULL = [("knn", "10"), ("knn", "100"), ("knn", "1000"), ("cka", None)]
HALF_K = {"10": "5", "30": "15", "100": "50", "300": "150", "1000": "500"}
MNAME = {
    "10": "k = 10",
    "30": "k = 30",
    "100": "k = 100",
    "300": "k = 300",
    "1000": "k = 1000",
    None: "CKA",
}
COLORS = {
    "T": "#2a78d6",
    "N": "#eb6834",
    "W": "#1baf7a",
    "F": "#52514e",
    "L0": "#eda100",
    "LLM": "#6250d6",
}


def grid(r: dict, m: str, k: str | None) -> np.ndarray:
    """[L_vit, L_set] excess."""
    if m == "cka":
        return np.array(r["cka"]) - np.array(r["cka_mu0"])
    return np.array(r["knn"][k]) - r["knn_mu0"][k]


def kk(split: str, k: str | None) -> str | None:
    return k if (k is None or split == "full") else HALF_K[k]


BIG, SMALL = 20000, 10000


def load(out: Path) -> dict:
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    by = defaultdict(dict)  # (split, target, caption) -> set -> row
    for r in rows:
        by[(r["split"], r["target"], r["caption"])][r["set"]] = r
    return by


def family_of(name: str, r: dict) -> str:
    if r["kind"] == "llm":
        return "LLM"
    return name.split("__")[0] if not name.startswith("F__") else "F"


def best(r: dict, m: str, k: str | None) -> float:
    return float(grid(r, m, k).max())


def full_table(by: dict, target: str) -> list[str]:
    out = []
    for cap in ("orig", "gemini"):
        sets = by.get(("full", target, cap), {})
        if not sets:
            continue
        fams = defaultdict(list)
        for name, r in sets.items():
            fams[family_of(name, r)].append((name, r))
        out += [
            f"\n### {cap} captions, {target}, all 10240 images: best cell (best configuration in the family)\n",
            "| Kernel | " + " | ".join(MNAME[k] for _, k in MEAS_FULL) + " |",
            "|" + "---|" * (len(MEAS_FULL) + 1),
        ]
        order = ["F", "T", f"N{SMALL}", f"N{BIG}", f"W{SMALL}", f"W{BIG}"]
        for fam in order:
            if fam not in fams:
                continue
            cells = []
            for m, k in MEAS_FULL:
                vals = [(best(r, m, k), n) for n, r in fams[fam]]
                v, n = max(vals)
                tag = n.split("__", 1)[1] if "__" in n else n
                cells.append(f"{v:.3f} ({tag})")
            out.append(f"| {fam} | " + " | ".join(cells) + " |")
        llm = fams.get("LLM", [])
        if llm:
            l0 = [np.mean([float(grid(r, m, k)[:, 0].max()) for _, r in llm]) for m, k in MEAS_FULL]
            deep = [np.mean([best(r, m, k) for _, r in llm]) for m, k in MEAS_FULL]
            out.append("| LLM layer 0 (mean of 4) | " + " | ".join(f"{v:.3f}" for v in l0) + " |")
            out.append(
                "| LLM best layer (mean of 4) | " + " | ".join(f"{v:.3f}" for v in deep) + " |"
            )
    return out


def pick(rows: dict, names: list[str], m: str, k: str | None, split_sel="A", split_eval="B"):
    """Choose (set, cell) on half A, return its value on half B and the choice."""
    best_v, choice = -np.inf, None
    for n in names:
        g = grid(rows[split_sel][n], m, kk(split_sel, k))
        i = np.unravel_index(np.argmax(g), g.shape)
        if g[i] > best_v:
            best_v, choice = g[i], (n, i)
    n, i = choice
    return float(grid(rows[split_eval][n], m, kk(split_eval, k))[i]), n


def split_table(by: dict, target: str) -> tuple[list[str], dict]:
    out, res = [], {}
    meas = [("knn", "10"), ("knn", "100"), ("knn", "1000"), ("cka", None)]
    for cap in ("orig", "gemini"):
        rows = {s: by.get((s, target, cap), {}) for s in ("A", "B")}
        if not rows["A"] or not rows["B"]:
            continue
        names = list(rows["A"])
        fam = defaultdict(list)
        for n in names:
            fam[family_of(n, rows["A"][n])].append(n)
        out += [
            f"\n### {cap} captions, {target}, split-half (chosen on half A, scored on half B; k halved)\n",
            "| | " + " | ".join(MNAME[k] for _, k in meas) + " |",
            "|" + "---|" * (len(meas) + 1),
        ]
        vals = {}
        for f in ["F", "T", f"N{SMALL}", f"N{BIG}", f"W{SMALL}", f"W{BIG}"]:
            if f in fam:
                vals[f] = [pick(rows, fam[f], m, k)[0] for m, k in meas]
        inductive = [n for f in (f"N{BIG}", f"W{BIG}") for n in fam.get(f, [])]
        vals[f"best inductive (N{BIG} or W{BIG})"] = [
            pick(rows, inductive, m, k)[0] for m, k in meas
        ]
        llm = fam.get("LLM", [])
        vals["LLM (mean of 4, own best cell)"] = [
            float(np.mean([pick(rows, [n], m, k)[0] for n in llm])) for m, k in meas
        ]
        for name, v in vals.items():
            out.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
        f_, ind, llm_v, t_ = (
            vals["F"],
            vals[f"best inductive (N{BIG} or W{BIG})"],
            vals["LLM (mean of 4, own best cell)"],
            vals.get("T"),
        )
        closure = [
            (i - f) / (lv - f) if lv - f > 1e-9 else np.nan
            for f, i, lv in zip(f_, ind, llm_v, strict=True)
        ]
        out.append(
            "| **gap closure, inductive** | " + " | ".join(f"**{c:.0%}**" for c in closure) + " |"
        )
        if t_:
            ct = [
                (t - f) / (lv - f) if lv - f > 1e-9 else np.nan
                for f, t, lv in zip(f_, t_, llm_v, strict=True)
            ]
            out.append(
                "| gap closure, T (transductive) | " + " | ".join(f"{c:.0%}" for c in ct) + " |"
            )
        res[cap] = {"vals": vals, "closure": closure, "meas": [MNAME[k] for _, k in meas]}
    return out, res


def stability(by: dict) -> list[str]:
    out = [
        "\n### Anchor stability: larger minus smaller anchor set, best cell, all images\n",
        "| Caption | Target | Family | " + " | ".join(MNAME[k] for _, k in MEAS_FULL) + " |",
        "|---|---|---|" + "---|" * len(MEAS_FULL),
    ]
    for (split, target, cap), sets in sorted(by.items()):
        if split != "full":
            continue
        for fam in ("N", "W"):
            a = [r for n, r in sets.items() if n.startswith(f"{fam}{SMALL}__")]
            b = [r for n, r in sets.items() if n.startswith(f"{fam}{BIG}__")]
            if not (a and b):
                continue
            d = [
                max(best(r, m, k) for r in b) - max(best(r, m, k) for r in a) for m, k in MEAS_FULL
            ]
            out.append(f"| {cap} | {target} | {fam} | " + " | ".join(f"{x:+.3f}" for x in d) + " |")
    return out


def predictions(by: dict) -> list[str]:
    out = [
        "\n## Predictions (DINOv2 and CLIP, all images, best cell)\n",
        "| | Test | Result | Verdict |",
        "|---|---|---|---|",
    ]

    def fam_best(sets, f, m, k):
        v = [best(r, m, k) for n, r in sets.items() if family_of(n, r) == f]
        return max(v) if v else np.nan

    s1, s2, s3, s4 = [], [], [], []
    for t in ("dino", "clip"):
        g = by.get(("full", t, "gemini"), {})
        o = by.get(("full", t, "orig"), {})
        if g:
            s1.append(fam_best(g, "T", "knn", "1000") - fam_best(g, "F", "knn", "1000"))
            for f in ("T", f"N{BIG}", f"W{BIG}"):
                s2.append(fam_best(g, f, "knn", "10") - fam_best(g, "F", "knn", "10"))
        if o:
            for f in ("T", f"N{BIG}", f"W{BIG}"):
                s2.append(fam_best(o, f, "knn", "10") - fam_best(o, "F", "knn", "10"))
            s3.append(
                (
                    fam_best(o, f"W{BIG}", "cka", None) - fam_best(o, "F", "cka", None),
                    max(fam_best(o, "T", "cka", None), fam_best(o, f"N{BIG}", "cka", None))
                    - fam_best(o, "F", "cka", None),
                )
            )
        for cap_sets in (g, o):
            for fam in ("T", f"N{BIG}"):
                pl = [
                    best(r, "cka", None)
                    for n, r in cap_sets.items()
                    if family_of(n, r) == fam and n.endswith("plain")
                ]
                we = [
                    best(r, "cka", None)
                    for n, r in cap_sets.items()
                    if family_of(n, r) == fam and n.endswith("weighted")
                ]
                if pl and we:
                    s4.append(max(we) - max(pl))
    if s1:
        out.append(
            f"| S1 | T > best F at k = 1000, Gemini | differences {', '.join(f'{x:+.3f}' for x in s1)} | {'holds' if min(s1) > 0 else '**fails**'} |"
        )
    if s2:
        out.append(
            f"| S2 | no spectral / word-vector kernel beats best F by > 0.02 at k = 10 | largest difference {max(s2):+.3f} | {'holds' if max(s2) <= 0.02 else '**fails**'} |"
        )
    if s3:
        ok = all(w > tn for w, tn in s3)
        out.append(
            f"| S3 | original captions: W gains most in CKA, T / N little | W {', '.join(f'{w:+.3f}' for w, _ in s3)}; best of T / N {', '.join(f'{tn:+.3f}' for _, tn in s3)} | {'holds' if ok else '**fails**'} |"
        )
    if s4:
        out.append(
            f"| S4 | weighted >= plain in CKA | weighted minus plain: {', '.join(f'{x:+.3f}' for x in s4)} | {'holds' if min(s4) >= 0 else '**fails**'} |"
        )
    return out


def figure(by: dict, path: Path, target: str = "dino") -> None:
    import matplotlib.pyplot as plt

    meas = [("knn", "10"), ("knn", "1000"), ("cka", None)]
    fig, axes = plt.subplots(len(meas), 2, figsize=(11, 9.5), sharex=True)
    for col, cap in enumerate(("orig", "gemini")):
        sets = by.get(("full", target, cap), {})
        if not sets:
            continue
        llm = [r for n, r in sets.items() if r["kind"] == "llm"]
        for row, (m, k) in enumerate(meas):
            ax = axes[row, col]
            for fam in ("T", f"N{BIG}", f"W{BIG}"):
                for w, ls in (("plain", "-"), ("weighted", "--")):
                    pts = sorted(
                        (r["k_s"], best(r, m, k))
                        for n, r in sets.items()
                        if family_of(n, r) == fam and n.endswith(w)
                    )
                    if pts:
                        x, y = zip(*pts, strict=True)
                        lab = f"{fam} ({w})" if fam[0] != "W" else fam
                        ax.plot(x, y, ls, color=COLORS[fam[0]], lw=2, marker="o", ms=4, label=lab)
            fb = max(best(r, m, k) for n, r in sets.items() if family_of(n, r) == "F")
            ax.axhline(fb, color=COLORS["F"], lw=1.5, ls=":", label="best word floor")
            for vals, key, lab in (
                ([float(grid(r, m, k)[:, 0].max()) for r in llm], "L0", "LLM layer 0"),
                ([best(r, m, k) for r in llm], "LLM", "LLM best layer"),
            ):
                if vals:
                    ax.axhspan(min(vals), max(vals), color=COLORS[key], alpha=0.15, lw=0)
                    ax.axhline(
                        np.mean(vals), color=COLORS[key], lw=1.5, label=f"{lab} (band = 4 LLMs)"
                    )
            ax.set_xscale("log", base=2)
            ax.set_title(f"{cap} captions, {MNAME[k]}", loc="left", fontsize=10)
            ax.grid(color="#e6e5e0", lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == len(meas) - 1:
                ax.set_xlabel("spectral / word-vector dimension k_s")
            if col == 0:
                ax.set_ylabel("excess over the null")
    axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        f"D2 (n = 10240) against {target}, best cell: kernels with no language model vs LLMs",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    by = load(args.out)
    global BIG, SMALL
    sizes = sorted({int(m[1]) for d in by.values() for n in d if (m := re.match(r"N(\d+)__", n))})
    if sizes:
        BIG, SMALL = sizes[-1], sizes[0]
    lines = ["# exp-028 summary"]
    meta = args.out / "kernels_meta.json"
    if meta.exists():
        info = json.loads(meta.read_text())
        lines += [
            "\n### Graphs\n",
            "| Caption | Graph | nodes | edges | components | largest | isolated | tau | eval captions without anchor edges |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for cap, d in info.items():
            for g, s in d.items():
                if "components" in s:
                    lines.append(
                        f"| {cap} | {g} | {s['n']} | {s['edges']} | {s['components']} | {s['largest_component']} | {s['isolated']} | {s['tau']:.2f} | {s.get('eval_without_anchor_edges', '-')} |"
                    )
    for t in ("dino", "clip"):
        lines += full_table(by, t)
    closures = {}
    for t in ("dino", "clip"):
        tab, res = split_table(by, t)
        lines += tab
        closures[t] = res
    lines += stability(by)
    lines += predictions(by)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    (args.out / "closure.json").write_text(json.dumps(closures, indent=1, default=float))
    figure(by, args.out / "fig_spectral.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
