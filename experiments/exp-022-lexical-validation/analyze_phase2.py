"""exp-022 phase 2: tables from the condensed results; writes phase2/summary.md.

uv run python experiments/exp-022-lexical-validation/analyze_phase2.py --out <results dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DINO = "vit_giant_patch14_dinov2.lvd142m__cls"
CLIP = "vit_huge_patch14_clip_224.laion2b__cls"
MAE = "vit_huge_patch14_224.mae__cls"
AUGREG = "vit_large_patch16_224.augreg_in21k__cls"
RANDOM = "random-vit_base_patch16_224.augreg_in21k__cls"
VIT = {DINO: "DINOv2", CLIP: "CLIP", MAE: "MAE", AUGREG: "AugReg", RANDOM: "random ViT"}
FINAL = "step143000"
PY = ["pythia-160m", "pythia-160m-seed1", "pythia-410m", "pythia-410m-seed1"]


def table(head: list[str], rows: list[list]) -> list[str]:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows],
        "",
    ]


def ex(g: dict, m: str) -> np.ndarray:
    return np.array(g[m]) - np.array(g[f"{m}_mu0"])


def cells(g: dict, m: str) -> dict:
    e = ex(g, m)
    return {
        "L0": float(e[0].max()),
        "deep": float(e[1:].max()),
        "last": float(e[-1].max()),
        "lastlast": float(e[-1, -1]),
        "peak_layer": g["names_a"][int(e.max(1).argmax())],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    p2 = args.out / "phase2"
    grids = json.loads((p2 / "phase2_grids.json").read_text())
    # larger-LM shuffle pairs were first scored against DOCCI's ViT files; the rerun replaces them
    fixed = {g["id"]: g for g in json.loads((p2 / "big_shuffle_grids.json").read_text())}
    grids = [fixed.get(g["id"], g) for g in grids]
    by = {(g["group"], g["a"], g["b"]): g for g in grids}
    old = {g["id"]: g for g in json.loads((args.out / "grids_full.json").read_text())}
    rows1 = {}
    for line in (args.out / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows1[(r["stage"], r["caption"], r["variant"], r["target"])] = r
    lines = ["# exp-022 phase 2 summary", ""]

    # 1. Pythia on original captions, against its lexical floor and on Gemini
    lines += ["## Pythia on original captions (final and over training)", ""]
    rows = []
    for s in PY:
        for t in (DINO, CLIP):
            for rev in ("step0", "step1000", "step4000", "step16000", "step64000", FINAL):
                g = by.get(("LV", f"EleutherAI__{s}@{rev}__orig", t))
                if g is None:
                    continue
                ck, k10 = cells(g, "cka"), cells(g, "knn10")
                k1000 = cells(g, "knn1000")
                rows.append(
                    [
                        s,
                        VIT[t],
                        rev,
                        k10["L0"],
                        k10["deep"],
                        k1000["L0"],
                        k1000["deep"],
                        ck["L0"],
                        ck["deep"],
                        ck["peak_layer"],
                    ]
                )
    lines += table(
        [
            "series",
            "target",
            "step",
            "k10 L0",
            "k10 deep",
            "k1000 L0",
            "k1000 deep",
            "CKA L0",
            "CKA deep",
            "CKA peak layer",
        ],
        rows,
    )

    def baseline(a, caption, t):
        """Unshuffled, untouched: the stored grid of the same LM and captions."""
        if "EleutherAI" in a:
            if caption == "gemini":
                return old.get(f"LV__{a}__{caption}__{t}__d2l")
            return by.get(("LV", f"{a}__{caption}", t))
        g = old.get(f"B3_{caption}__{a}__{caption}__{t}__d2l")
        extra = rows1.get(("trained", "", f"{a}__{caption}", t))  # phase 1's large-k mutual kNN
        if g is not None and extra is not None and "knn" in extra:
            g = {**g}
            for k in ("100", "300", "1000"):
                g[f"knn{k}"] = extra["knn"][k]
                g[f"knn{k}_mu0"] = np.full_like(np.array(extra["knn"][k]), extra["knn_mu0"][k])
        return g

    # 2. Token shuffle
    lines += ["## Token shuffle within captions (excess, best target layer)", ""]
    rows = []
    for g in sorted((g for g in grids if g["group"] == "SHUF"), key=lambda x: x["a"]):
        stem, cap = g["a"].rsplit("__", 1)
        caption, seed = cap.split("-shuf")
        base = baseline(stem, caption, g["b"])
        row = [stem.split("__")[1], caption, seed, VIT[g["b"]]]
        for m in ("knn10", "knn1000", "cka"):
            s = ex(g, m).max(1)
            if base is not None:
                b = ex(base, m).max(1)
                row += [
                    float(b[0]),
                    float(s[0]),
                    float(b[1:].max()),
                    float(s[1:].max()),
                    float(b[-1]),
                    float(s[-1]),
                ]
            else:
                row += ["-", float(s[0]), "-", float(s[1:].max()), "-", float(s[-1])]
        rows.append(row)
    head = ["LM", "captions", "seed", "target"]
    for m in ("k10", "k1000", "CKA"):
        head += [
            f"{m} L0",
            f"{m} L0 shuf",
            f"{m} deep",
            f"{m} deep shuf",
            f"{m} last",
            f"{m} last shuf",
        ]
    lines += table(head, rows)

    # 3. Factorial
    lines += ["## Table x blocks factorial (Pythia, excess at best target layer, by LM layer)", ""]
    rows = []
    for short, repo in (("160m", "pythia-160m"), ("410m", "pythia-410m")):
        for caption in ("gemini", "orig"):
            for t in (DINO, CLIP):
                sets = {
                    "random table, random blocks": baseline(
                        f"EleutherAI__{repo}@step0", caption, t
                    ),
                    "trained table, trained blocks": baseline(
                        f"EleutherAI__{repo}@{FINAL}", caption, t
                    ),
                    "trained table, random blocks": by.get(
                        ("FACT", f"EleutherAI__{repo}@blocks-step0-table-{FINAL}__{caption}", t)
                    ),
                    "random table, trained blocks": by.get(
                        ("FACT", f"EleutherAI__{repo}@blocks-{FINAL}-table-step0__{caption}", t)
                    ),
                }
                for name, g in sets.items():
                    if g is None:
                        continue
                    k = np.round(ex(g, "knn10").max(1), 3).tolist()
                    c = np.round(ex(g, "cka").max(1), 3).tolist()
                    rows.append(
                        [short, caption, VIT[t], name, " ".join(map(str, k)), " ".join(map(str, c))]
                    )
    lines += table(["size", "captions", "target", "cell", "k10 by layer", "CKA by layer"], rows)

    # 4. DOCCI
    stats = json.loads((p2 / "docci_stats.json").read_text())
    lines += ["## DOCCI (human long captions, different images)", "", f"Stats: {stats}", ""]
    lex = {}
    for line in (p2 / "docci_lexical.jsonl").read_text().splitlines():
        r = json.loads(line)
        lex[(r["variant"], r["target"])] = r

    def lex_ex(v, t, m):
        r = lex[(v, t)]
        if m == "cka":
            return float((np.array(r["cka"]) - np.array(r["cka_mu0"])).max())
        if m == "u":
            return float(np.array(r["cka_u"]).max())
        return float((np.array(r["knn"][m]) - r["knn_mu0"][m]).max())

    rows = []
    for t in VIT:
        for v in (
            "tf",
            "tfidf",
            "binary",
            "tfidf_sublinear",
            "word_tfidf",
            "word_tfidf_nostop",
            "first12_tfidf",
            "first50_tfidf",
            "first100_tfidf",
        ):
            rows.append(
                [
                    VIT[t],
                    v,
                    lex_ex(v, t, "10"),
                    lex_ex(v, t, "1000"),
                    lex_ex(v, t, "cka"),
                    lex_ex(v, t, "u"),
                ]
            )
    lines += table(["target", "lexical variant", "k10", "k1000", "CKA", "debiased"], rows)
    rows = []
    for g in sorted((g for g in grids if g["group"] == "DOCCI"), key=lambda x: (x["b"], x["a"])):
        row = [g["a"].split("__")[1], VIT[g["b"]]]
        for m in ("knn10", "knn1000", "cka"):
            c = cells(g, m)
            row += [c["L0"], c["deep"], c["last"]]
        row.append(" ".join(map(str, np.round(ex(g, "cka").max(1), 3).tolist())))
        rows.append(row)
    lines += table(
        [
            "LM",
            "target",
            "k10 L0",
            "k10 deep",
            "k10 last",
            "k1000 L0",
            "k1000 deep",
            "k1000 last",
            "CKA L0",
            "CKA deep",
            "CKA last",
            "CKA by layer",
        ],
        rows,
    )

    # 5. Ridge
    r = json.loads((p2 / "ridge.json").read_text())
    lines += ["## Held-out ridge", "", "Calibration (R^2):", ""]
    lines += table(
        ["case", "lexical", "deep", "joint"],
        [
            [k, v["lexical"]["r2"], v["deep"]["r2"], v["lexical+deep"]["r2"]]
            for k, v in r["calibration"].items()
        ],
    )
    # The run's records lost the LM file and lexical definition to a key clash (fixed in phase2.py).
    # They are recoverable: the 72 records follow the nested loop order, and layer names confirm it.
    sets = [
        (c, lm, last)
        for lm, last in (("pythia-410m", "24"), ("open_llama_7b", "32"), ("bloomz-7b1", "30"))
        for c in ("gemini", "orig")
    ]
    order = [
        (t, c, lm, last, depth, w)
        for t in (DINO, CLIP)
        for c, lm, last in sets
        for depth in ("last", "deep5", "layer0")
        for w in ("tfidf", "tfidf_sublinear")
    ]
    assert len(order) == len(r["pairs"])
    head = [
        "LM",
        "captions",
        "target",
        "depth",
        "lexical def",
        "R2 words",
        "R2 LM",
        "R2 both",
        "top1 words",
        "top1 LM",
        "top1 both",
        "mknn words",
        "mknn LM",
        "mknn both",
    ]
    rows = []
    for x, (t, c, lm, last, depth, w) in zip(r["pairs"], order, strict=True):
        assert (x["target"], x["caption"], x["depth"]) == (t, c, depth)
        assert depth != "last" or x["layer"] == last
        rows.append([lm, c, VIT[t], depth, w, *_ridge(x)])
    lines += table(head, rows)
    (p2 / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(len(lines), "lines")


def _ridge(x: dict) -> list[float]:
    w, m, b = x["lexical"], x["lm"], x["lexical+lm"]
    if not isinstance(w, dict) or not isinstance(m, dict):
        raise ValueError("record keys overwritten")
    return [
        w["r2"],
        m["r2"],
        b["r2"],
        w["top1"],
        m["top1"],
        b["top1"],
        w["mknn"],
        m["mknn"],
        b["mknn"],
    ]


if __name__ == "__main__":
    main()
