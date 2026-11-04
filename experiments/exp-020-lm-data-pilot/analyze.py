"""exp-020: summary tables, the decision gate and the contrast figure.

Reads `pairs.jsonl`, `contrasts.jsonl` and `bpb/*.json` of the output folder; writes `summary.md`
and `fig_contrasts.{png,pdf}`. Values are calibrated excesses (score - permutation-null mean) at
the best cell; contrasts are continuation - base with a 95% bootstrap CI over samples.

    uv run python experiments/exp-020-lm-data-pilot/analyze.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GROUPS = {"code": "code / math", "math": "code / math", "images": "images (positive control)"}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def short(repo: str) -> str:
    return repo.split("/")[-1]


def sig(ci) -> int:
    return 1 if ci[0] > 0 else -1 if ci[1] < 0 else 0


def large_k(r: dict) -> dict | None:
    """The k = n/100 entry of a row's k sweep (the one k other than 10)."""
    other = [v for k, v in r.get("knn_sweep", {}).items() if k != "10"]
    return other[0] if other else None


def median(rows: list[dict], tag: str, get) -> str:
    """Median over rows (ViTs) of the best cell's excess over the null mean."""
    v = [get(r) for r in rows if r["tag"] == tag]
    v = [d["best"]["score"] - d["best"]["mu0"] for d in v if d is not None]
    return f"{np.median(v):.3f}" if v else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pairs = [r for r in load(args.out / "pairs.jsonl") if r["group"] == "LV"]
    contrasts = load(args.out / "contrasts.jsonl")
    bpb = {
        json.loads(f.read_text())["model"]: json.loads(f.read_text())
        for f in sorted((args.out / "bpb").glob("*.json"))
    }
    lines = ["# exp-020 summary", ""]

    # 1. capability covariates
    lines += [
        "## Bits per byte",
        "",
        "| model | generic | original captions | Gemini captions |",
        "|---|---|---|---|",
    ]
    for m, b in bpb.items():
        lines.append(
            f"| {short(m)} | {b['generic']['bpb']:.3f} | {b['orig']['bpb']:.3f} | {b['gemini']['bpb']:.3f} |"
        )

    # 2. alignment per model (median over ViTs)
    lines += ["", "## LM x ViT alignment (median over the 4 ViTs, best cell, excess over null)", ""]
    lines += [
        "| model | captions | mkNN k=10 (n=1024) | ORC given density (n=1024) | mkNN k=10 (n=10240) | mkNN k=n/100 (n=10240) | CKA (n=10240) |",
        "|---|---|---|---|---|---|---|",
    ]
    models = sorted({r["a"].rsplit("__", 1)[0] for r in pairs})
    for m in models:
        for c in ("orig", "gemini"):
            rs = [r for r in pairs if r["a"] == f"{m}__{c}"]
            cols = [
                median(rs, "d2s", lambda r: r["mutual_knn"]),
                median(rs, "d2s", lambda r: r.get("orc_k10")),
                median(rs, "d2l", lambda r: r["mutual_knn"]),
                median(rs, "d2l", lambda r: large_k(r)),
                median(rs, "d2l", lambda r: r["cka_lin"]),
            ]
            lines.append(f"| {m.split('__')[-1]} | {c} | " + " | ".join(cols) + " |")

    # 3. contrasts and the gate
    lines += ["", "## Contrasts: continuation - base (best cell; 95% CI over samples)", ""]
    lines += [
        "| base -> continuation | captions | n | ViT | mkNN k=10 | mkNN k=n/100 | CKA |",
        "|---|---|---|---|---|---|---|",
    ]
    gate = {}
    for c in contrasts:
        big = [
            k for k in c if k.startswith("mknn_k") and k.endswith("_best") and k != "mknn_k10_best"
        ]

        def cell(d):
            return f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"

        lines.append(
            f"| {short(c['base'])} -> {short(c['cont'])} | {c['caption']} | {c['n']} | {c['vit'].split('_patch')[0]}{'_' + c['vit'].split('.')[-1]} | "
            f"{cell(c['mknn_k10_best'])} | {cell(c[big[0]]) if big else '-'} | {cell(c['cka_best'])} |"
        )
        key = (c["base"], c["cont"], c["adds"], c["caption"], c["tag"])
        gate.setdefault(key, []).append(sig(c["mknn_k10_best"]["ci"]))

    lines += ["", "## Decision gate (mutual kNN k = 10, best cell)", ""]
    lines += [
        "| base -> continuation | adds | captions | n | ViTs higher / lower / n.s. |",
        "|---|---|---|---|---|",
    ]
    for (b, c, adds, cap, tag), s in gate.items():
        s = np.array(s)
        lines.append(
            f"| {short(b)} -> {short(c)} | {adds} | {cap} | {tag} | {(s > 0).sum()} / {(s < 0).sum()} / {(s == 0).sum()} |"
        )
    lines += [
        "",
        "Gate (written before running): (1) both image-trained continuations higher for most ViTs;",
        "(2) code / math - base significant in one direction for both Qwen sizes and >= 3 of 4 ViTs;",
        "(3) not explained by generic bits per byte (table above).",
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(contrasts, args.out)


def plot(contrasts: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    if not contrasts:
        return
    keys = sorted({(c["base"], c["cont"]) for c in contrasts}, key=lambda k: k[1])
    panels = [(cap, tag) for cap in ("orig", "gemini") for tag in ("d2s", "d2l")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), sharey=True)
    vits = sorted({c["vit"] for c in contrasts})
    for ax, (cap, tag) in zip(np.atleast_1d(axes), panels, strict=False):
        for vi, v in enumerate(vits):
            xs, ys, lo, hi = [], [], [], []
            for ki, (b, cn) in enumerate(keys):
                m = [
                    c
                    for c in contrasts
                    if (c["base"], c["cont"], c["vit"], c["caption"], c["tag"])
                    == (b, cn, v, cap, tag)
                ]
                if m:
                    d = m[0]["mknn_k10_best"]
                    xs.append(ki + 0.15 * (vi - 1.5))
                    ys.append(d["diff"])
                    lo.append(d["diff"] - d["ci"][0])
                    hi.append(d["ci"][1] - d["diff"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3, capsize=2, label=v.split(".")[0])
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(
            range(len(keys)),
            [f"{short(b)}\n-> {short(c)}" for b, c in keys],
            fontsize=6,
            rotation=60,
        )
        ax.set_title(f"{cap} captions, n = {1024 if tag == 'd2s' else 10240}", fontsize=9)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("mutual kNN excess: continuation - base")
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_contrasts.{ext}", dpi=200)


if __name__ == "__main__":
    main()
