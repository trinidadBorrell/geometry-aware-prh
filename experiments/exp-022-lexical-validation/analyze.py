"""exp-022: score phase 1 against the README's fixed pass rules; writes summary.md.

uv run python experiments/exp-022-lexical-validation/analyze.py --out <results dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DINO = "vit_giant_patch14_dinov2.lvd142m__cls"
CLIP = "vit_huge_patch14_clip_224.laion2b__cls"
SHORT = {DINO: "DINOv2", CLIP: "CLIP"}
PY = ["pythia-160m", "pythia-160m-seed1", "pythia-410m", "pythia-410m-seed1"]
LLMS = [
    "bigscience__bloomz-1b7",
    "bigscience__bloomz-7b1",
    "openlm-research__open_llama_7b",
    "huggyllama__llama-13b",
]
FINAL = "step143000"


def excess(row: dict, m: str) -> np.ndarray:
    if m == "cka":
        return np.array(row["cka"]) - np.array(row["cka_mu0"])
    if m == "cka_u":
        return np.array(row["cka_u"])
    return np.array(row["knn"][m[3:]]) - row["knn_mu0"][m[3:]]


def table(head: list[str], rows: list[list]) -> list[str]:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows],
        "",
    ]


class Data:
    def __init__(self, out: Path):
        self.rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
        self.idx = {
            (r["stage"], r["caption"], r["variant"], r["target"], r.get("subset", "")): r
            for r in self.rows
        }
        self.grids = {r["id"]: r for r in json.loads((out / "grids_full.json").read_text())}

    def get(self, stage, caption, variant, target, subset=""):
        return self.idx.get((stage, caption, variant, target, subset))

    def best(self, stage, caption, variant, target, m, subset=""):
        r = self.get(stage, caption, variant, target, subset)
        return float(excess(r, m).max()) if r else float("nan")

    def last(self, stage, caption, variant, target, m, subset=""):
        """At the target's last layer (fixed, no selection)."""
        r = self.get(stage, caption, variant, target, subset)
        return float(excess(r, m)[-1, 0]) if r else float("nan")


def v1(out: Path, lines: list[str]) -> None:
    r = json.loads((out / "identity.json").read_text())
    lines += ["## V1 identity", ""]
    head = [
        "set",
        "row cos mean",
        "table norm CV",
        "mean-vec ratio",
        "top PC",
        "row PR",
        "rho(norm, idf)",
    ]
    rows = []
    for s in r["sets"]:
        t = s["table"]
        rows.append(
            [
                f"{s['repo'].split('/')[1]}@{s['rev']}",
                f"{s['row_cosine']['mean']:.6f}",
                t["norm_cv"],
                t["mean_vector_norm_ratio"],
                t["top_pc_share"],
                t["pr_rows"],
                t["spearman_norm_idf"],
            ]
        )
    lines += table(head, rows)
    head = ["set", "version", "CKA with TF", "x DINOv2", "x CLIP", "x 1.4b"]
    rows = []
    for s in r["sets"]:
        for name, v in s["versions"].items():

            def m(key, v=v):
                x = v[key]
                c, mu = np.array(x["cka"]), np.array(x["cka_mu0"])
                return float((c - mu).max())

            rows.append(
                [
                    f"{s['repo'].split('/')[1]}@{s['rev']}",
                    name,
                    float(np.array(v["cka_tf"]["cka"]).max()),
                    m("cka_vit_giant_pa"),
                    m("cka_vit_huge_pat"),
                    m("cka_EleutherAI__"),
                ]
            )
    lines += table(head, rows)
    head = ["Gaussian d", "CKA with TF (mean, min-max)", "x DINOv2", "x CLIP", "x 1.4b"]
    rows = []
    for d in sorted({g["d"] for g in r["gaussian"]}):
        gs = [g for g in r["gaussian"] if g["d"] == d]

        def rng(f, gs=gs):
            v = [f(g) for g in gs]
            return f"{np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})"

        def e(key):
            return lambda rec: float(
                (np.array(rec[key]["cka"]) - np.array(rec[key]["cka_mu0"])).max()
            )

        rows.append(
            [
                d,
                rng(lambda g: float(np.array(g["cka_tf"]["cka"]).max())),
                rng(e("cka_vit_giant_pa")),
                rng(e("cka_vit_huge_pat")),
                rng(e("cka_EleutherAI__")),
            ]
        )
    lines += table(head, rows)


def v2(out: Path, lines: list[str]) -> None:
    r = json.loads((out / "numerics.json").read_text())
    worst = max(
        max(abs(x["cka_exact"] - x["cka_stored"]), abs(x["knn_exact"] - x["knn_stored"])) for x in r
    )
    lines += ["## V2 numerics", "", f"{len(r)} cells; worst difference {worst:.5f}", ""]
    p = json.loads((out / "perm_check.json").read_text())
    rows = []
    for x in p:
        c = np.abs(np.array(x["cka_perm_mu0"]) - np.array(x["cka_analytic_mu0"])).max()
        k = max(
            np.abs(np.array(v) - x["knn_analytic_mu0"][key]).max()
            for key, v in x["knn_perm_mu0"].items()
        )
        rows.append([x["caption"], x["variant"], float(c), float(k)])
    lines += table(["caption", "variant", "worst CKA null error", "worst kNN null error"], rows)


def v3(d: Data, lines: list[str]) -> None:
    lines += ["## V3 estimator", "", "Excess vs debiased CKA at n = 10240 (best target layer).", ""]
    rows = []
    for c in ("gemini", "orig"):
        for t in (DINO, CLIP):
            for v in ("tf", "tfidf"):
                ex, u = d.best("lexical", c, v, t, "cka"), d.best("lexical", c, v, t, "cka_u")
                rows.append([c, SHORT[t], v, ex, u, u - ex])
    lines += table(["captions", "target", "variant", "excess", "debiased", "difference"], rows)
    lines += ["Trained LMs, debiased CKA: layer 0 and best deeper layer (best target layer).", ""]
    rows = []
    for s in PY:
        key = f"EleutherAI__{s}@{FINAL}__gemini"
        for t in (DINO, CLIP):
            r = d.get("trained", "", key, t)
            u = np.array(r["cka_u"])
            rows.append([s, "gemini", SHORT[t], float(u[0].max()), float(u[1:].max())])
    for m in LLMS:
        for c in ("gemini", "orig"):
            for t in (DINO, CLIP):
                r = d.get("trained", "", f"{m}__{c}", t)
                u = np.array(r["cka_u"])
                rows.append([m.split("__")[1], c, SHORT[t], float(u[0].max()), float(u[1:].max())])
    lines += table(["LM", "captions", "target", "layer 0", "best deep"], rows)
    lines += ["Disjoint blocks of 1024 and nested sets, mean (min-max) over blocks.", ""]
    rows = []
    variants = sorted({(r["caption"], r["variant"]) for r in d.rows if r["stage"] == "subsets"})
    for c, v in variants:
        for t in (DINO, CLIP):
            bl = [
                (
                    d.best("subsets", c, v, t, "cka", f"block{i}"),
                    d.best("subsets", c, v, t, "cka_u", f"block{i}"),
                    d.best("subsets", c, v, t, "knn10", f"block{i}"),
                )
                for i in range(10)
            ]
            arr = np.array(bl)
            rows.append(
                [
                    c,
                    v.split("__")[-2] + ":" + v.split(":")[-1] if ":" in v else v,
                    SHORT[t],
                    f"{arr[:, 0].mean():.3f} ({arr[:, 0].min():.3f}-{arr[:, 0].max():.3f})",
                    f"{arr[:, 1].mean():.3f} ({arr[:, 1].min():.3f}-{arr[:, 1].max():.3f})",
                    f"{arr[:, 2].mean():.3f}",
                    d.best("subsets", c, v, t, "cka", "first2048"),
                    d.best("subsets", c, v, t, "cka_u", "first2048"),
                    d.best("subsets", c, v, t, "cka", "first4096"),
                    d.best("subsets", c, v, t, "cka_u", "first4096"),
                ]
            )
    lines += table(
        [
            "captions",
            "variant",
            "target",
            "1024 excess",
            "1024 debiased",
            "1024 knn10",
            "2048 excess",
            "2048 debiased",
            "4096 excess",
            "4096 debiased",
        ],
        rows,
    )


def v4(d: Data, out: Path, lines: list[str]) -> None:
    lines += [
        "## V4 selection",
        "",
        "Split-half: cell chosen on one half, scored on the other.",
        "",
    ]
    rows = []
    for x in json.loads((out / "split.json").read_text()):
        for m in ("knn10_excess_layer0", "knn10_excess_deep", "cka_layer0", "cka_deep"):
            rows.append(
                [
                    x["set"].replace("EleutherAI__", "").replace(f"@{FINAL}__gemini", ""),
                    SHORT[x["target"]],
                    m,
                    x[m]["in_sample"],
                    x[m]["held_out"],
                    x[m]["in_sample"] - x[m]["held_out"],
                ]
            )
    lines += table(["set", "target", "measure", "in sample", "held out", "difference"], rows)
    lines += [
        "Fixed cells: target last layer; TF, TF-IDF, and trained LM at layer 0, depth 0.75, last.",
        "",
    ]
    rows = []
    for c in ("gemini", "orig"):
        for t in (DINO, CLIP):
            for v in ("tf", "tfidf"):
                rows.append(
                    [
                        c,
                        SHORT[t],
                        v,
                        d.best("lexical", c, v, t, "knn10"),
                        d.last("lexical", c, v, t, "knn10"),
                        d.best("lexical", c, v, t, "cka"),
                        d.last("lexical", c, v, t, "cka"),
                    ]
                )
    lines += table(
        [
            "captions",
            "target",
            "variant",
            "knn10 best",
            "knn10 last ViT layer",
            "CKA best",
            "CKA last ViT layer",
        ],
        rows,
    )
    rows = []
    for s in PY:
        for t in (DINO, CLIP):
            g = d.grids[f"LV__EleutherAI__{s}@{FINAL}__gemini__{t}__d2l"]
            for m in ("knn10", "cka"):
                e = np.array(g[m]) - np.array(g[f"{m}_mu0"])
                i75 = int(
                    np.abs(np.arange(len(g["names_a"])) / (len(g["names_a"]) - 1) - 0.75).argmin()
                )
                rows.append(
                    [
                        s,
                        SHORT[t],
                        m,
                        float(e[0, -1]),
                        float(e[i75, -1]),
                        float(e[-1, -1]),
                        float(e[0].max()),
                        float(e[1:].max()),
                    ]
                )
    lines += table(
        [
            "LM",
            "target",
            "measure",
            "L0 x last",
            "0.75 x last",
            "last x last",
            "L0 best",
            "deep best",
        ],
        rows,
    )


def v5(d: Data, out: Path, lines: list[str]) -> None:
    c = json.loads((out / "concentration.json").read_text())
    lines += [
        "## V5 concentration",
        "",
        f"Near-duplicate image rows {c['near_duplicate_images']}; duplicate captions gemini {c['duplicate_captions_gemini']}, orig {c['duplicate_captions_orig']}; removed {c['removed_total']}.",
        "",
    ]
    rows = []
    for r in d.rows:
        if r["stage"] != "dedup":
            continue
        for m in ("knn10", "cka"):
            full = _full(d, r["caption"], r["variant"], r["target"], m)
            rows.append(
                [r["caption"], r["variant"], SHORT[r["target"]], m, full, float(excess(r, m).max())]
            )
    lines += table(
        ["captions", "variant", "target", "measure", "all rows", "without duplicates"], rows
    )
    rows = [
        [k, v["layer"], v["mean"], v["zero_fraction"], v["top5pct_share"], v["top25pct_share"]]
        for k, v in c["overlap"].items()
    ]
    lines += table(
        ["set", "DINOv2 layer", "mean overlap", "zero share", "top 5% share", "top 25% share"], rows
    )
    lines += ["Clusters (k-means on DINOv2 last layer):", ""]
    lines += table(
        ["cluster", "size", "most over-represented tokens"],
        [
            [
                x["cluster"],
                x["size"],
                " ".join(x["top_tokens"][:8]).replace("|", "/").replace("\n", " "),
            ]
            for x in c["clusters"]
        ],
    )
    rows = []
    keys = sorted(
        {(r["caption"], r["variant"], r["target"]) for r in d.rows if r["stage"] == "cluster"}
    )
    sizes = {f"c{x['cluster']}": x["size"] for x in c["clusters"]}
    for cap, v, t in keys:
        cs = [
            r
            for r in d.rows
            if r["stage"] == "cluster" and (r["caption"], r["variant"], r["target"]) == (cap, v, t)
        ]
        w = np.array([sizes[r["subset"]] for r in cs], dtype=float)
        k = np.array([excess(r, "knn10").max() for r in cs])
        ka = np.array([excess(r, "cka").max() for r in cs])
        u = np.array([excess(r, "cka_u").max() for r in cs])
        rows.append(
            [
                cap,
                v,
                SHORT[t],
                len(cs),
                float((k * w).sum() / w.sum()),
                float((ka * w).sum() / w.sum()),
                float((u * w).sum() / w.sum()),
                _full(d, cap, v, t, "knn10"),
                _full(d, cap, v, t, "cka"),
            ]
        )
    lines += table(
        [
            "captions",
            "variant",
            "target",
            "clusters",
            "within knn10",
            "within CKA excess",
            "within CKA debiased",
            "whole-set knn10",
            "whole-set CKA",
        ],
        rows,
    )


def _full(d: Data, caption, variant, target, m):
    if variant in ("tf", "tfidf"):
        return d.best("lexical", caption, variant, target, m)
    name, layer = variant.split(":")
    if name == "pythia-410m":
        g = d.grids[f"LV__EleutherAI__pythia-410m@{FINAL}__gemini__{target}__d2l"]
    else:
        g = d.grids[f"B3_orig__huggyllama__llama-13b__orig__{target}__d2l"]
    e = np.array(g[m]) - np.array(g[f"{m}_mu0"])
    return float(e[g["names_a"].index(layer)].max())


def v6_v7(d: Data, out: Path, lines: list[str]) -> None:
    cap = json.loads((out / "captions.json").read_text())
    lines += [
        "## V6 template and V7 definition",
        "",
        f"Wrapped Gemini captions: {cap['n_wrapped_gemini']}; contains 500w: {cap['gemini']['contains_500w']}.",
        "",
    ]
    for c in ("gemini", "orig"):
        x = cap[c]
        tops = ", ".join(f"{t['tok']!r} {t['share']:.3f}" for t in x["top_tokens"][:12])
        lines += [
            f"- {c}: median tokens {x['tokens_median']}, vocab {x['vocab']}, r_tf {x['r_tf']:.1f}, r_tfidf {x['r_tfidf']:.1f}, duplicates {x['duplicate_rows']}, non-ASCII rows {x['non_ascii_rows']}. Top tokens: {tops}"
        ]
    lines += ["", f"Original-caption language: {cap.get('orig_language')}", ""]
    lines += ["Gemini heads:", *[f"- `{h[:140]}`" for h in cap["gemini"]["head"][:4]], ""]
    variants = [
        v
        for c, v in sorted(
            {(r["caption"], r["variant"]) for r in d.rows if r["stage"] == "lexical"}
        )
        if not v.startswith(("first", "random"))
    ]
    seen = []
    rows = []
    for v in variants:
        if v in seen:
            continue
        seen.append(v)
        for c in ("gemini", "orig"):
            row = [v, c]
            ok = False
            for t in (DINO, CLIP):
                r = d.get("lexical", c, v, t)
                if r is None:
                    row += ["-", "-", "-"]
                    continue
                ok = True
                row += [
                    float(excess(r, "cka").max()),
                    float(excess(r, "cka_u").max()),
                    float(excess(r, "knn10").max()),
                ]
            if ok:
                rows.append(row)
    lines += table(
        [
            "variant",
            "captions",
            "DINO CKA",
            "DINO debiased",
            "DINO knn10",
            "CLIP CKA",
            "CLIP debiased",
            "CLIP knn10",
        ],
        rows,
    )
    rows = []
    for c in ("gemini", "orig"):
        for t in (DINO, CLIP):
            o, x = (d.get("crossfit", c, f"tfidf_idf_{w}", t, "half") for w in ("own", "cross"))
            rows.append(
                [
                    c,
                    SHORT[t],
                    float(excess(o, "cka").max()),
                    float(excess(x, "cka").max()),
                    float(excess(o, "knn10").max()),
                    float(excess(x, "knn10").max()),
                ]
            )
    lines += table(
        [
            "captions",
            "target",
            "CKA idf own",
            "CKA idf other half",
            "knn10 own",
            "knn10 other half",
        ],
        rows,
    )


def v8(d: Data, lines: list[str]) -> None:
    lines += [
        "## V8 length vs content",
        "",
        "Gemini cut to L tokens, against the original captions (excess).",
        "",
    ]
    rows = []
    for mode in ("first", "random"):
        for w in ("tf", "tfidf"):
            for t in (DINO, CLIP):
                for L in ("12", "matched", "25", "50", "100", "200", "400"):
                    v = f"{mode}{L}_{w}"
                    rows.append(
                        [
                            mode,
                            w,
                            SHORT[t],
                            L,
                            d.best("lexical", "gemini", v, t, "cka"),
                            d.best("lexical", "gemini", v, t, "cka_u"),
                            d.best("lexical", "gemini", v, t, "knn10"),
                            d.best("lexical", "gemini", v, t, "knn1000"),
                        ]
                    )
                rows.append(
                    [
                        "full gemini",
                        w,
                        SHORT[t],
                        "613",
                        d.best("lexical", "gemini", w, t, "cka"),
                        d.best("lexical", "gemini", w, t, "cka_u"),
                        d.best("lexical", "gemini", w, t, "knn10"),
                        d.best("lexical", "gemini", w, t, "knn1000"),
                    ]
                )
                rows.append(
                    [
                        "original",
                        w,
                        SHORT[t],
                        "12",
                        d.best("lexical", "orig", w, t, "cka"),
                        d.best("lexical", "orig", w, t, "cka_u"),
                        d.best("lexical", "orig", w, t, "knn10"),
                        d.best("lexical", "orig", w, t, "knn1000"),
                    ]
                )
    lines += table(
        ["mode", "weights", "target", "tokens", "CKA", "debiased", "knn10", "knn1000"], rows
    )


def v9_v11(d: Data, lines: list[str]) -> None:
    lines += ["## V9 generality", ""]
    rows = []
    targets = sorted(
        {r["target"] for r in d.rows if r["stage"] == "targets" and "pythia" not in r["target"]}
    )
    for t in targets:
        for c in ("gemini", "orig"):
            for v in ("tf", "tfidf"):
                rows.append(
                    [
                        t[:32],
                        c,
                        v,
                        d.best("targets", c, v, t, "cka"),
                        d.best("targets", c, v, t, "cka_u"),
                        d.best("targets", c, v, t, "knn10"),
                    ]
                )
    lines += table(["target", "captions", "variant", "CKA", "debiased", "knn10"], rows)
    lines += ["## V11 depth: retention by LM layer", ""]
    rows = []
    for s in ("pythia-160m", "pythia-410m"):
        for rev in ("step0", FINAL):
            key = f"EleutherAI__{s}@{rev}__gemini"
            g = d.grids[f"LV__{key}__{DINO}__d2l"]
            vis = (np.array(g["cka"]) - np.array(g["cka_mu0"])).max(1)
            tf_, ti = (
                np.array(d.get("targets", "gemini", v, key)["cka"])[:, 0] for v in ("tf", "tfidf")
            )
            b_tf, b_ti = (
                d.best("lexical", "gemini", "tf", DINO, "cka"),
                d.best("lexical", "gemini", "tfidf", DINO, "cka"),
            )
            for name, arr in (
                ("layers", g["names_a"]),
                ("vision CKA", np.round(vis, 3).tolist()),
                ("CKA with TF", np.round(tf_, 3).tolist()),
                ("CKA with TF-IDF", np.round(ti, 3).tolist()),
                ("TF retention x floor", np.round(tf_ * b_tf, 3).tolist()),
                ("TF-IDF retention x floor", np.round(ti * b_ti, 3).tolist()),
            ):
                rows.append([f"{s}@{rev}", name, " ".join(str(x) for x in arr)])
    lines += table(["set", "row", "by layer"], rows)


def v10(d: Data, lines: list[str]) -> None:
    lines += [
        "## V10 other families: trained LMs, layer 0 vs deep (excess at best target layer)",
        "",
    ]
    rows = []
    for m in LLMS:
        for c in ("gemini", "orig"):
            for t in (DINO, CLIP):
                r = d.get("trained", "", f"{m}__{c}", t)
                row = [m.split("__")[1], c, SHORT[t]]
                for k in ("knn10", "knn100", "knn1000"):
                    e = excess(r, k)
                    row += [float(e[0].max()), float(e[1:].max())]
                g = d.grids[f"B3_{c}__{m}__{c}__{t}__d2l"]
                e = np.array(g["cka"]) - np.array(g["cka_mu0"])
                row += [float(e[0].max()), float(e[1:].max())]
                row += [
                    d.best("lexical", c, f"tok-{m}_tf", t, "knn10"),
                    d.best("lexical", c, f"tok-{m}_tfidf", t, "knn10"),
                    d.best("lexical", c, f"tok-{m}_tf", t, "cka"),
                    d.best("lexical", c, f"tok-{m}_tfidf", t, "cka"),
                ]
                rows.append(row)
    lines += table(
        [
            "LM",
            "captions",
            "target",
            "k10 L0",
            "k10 deep",
            "k100 L0",
            "k100 deep",
            "k1000 L0",
            "k1000 deep",
            "CKA L0",
            "CKA deep",
            "k10 TF",
            "k10 TF-IDF",
            "CKA TF",
            "CKA TF-IDF",
        ],
        rows,
    )
    rows = []
    for s in PY:
        for t in (DINO, CLIP):
            g = d.grids[f"LV__EleutherAI__{s}@{FINAL}__gemini__{t}__d2l"]
            row = [s, SHORT[t]]
            for k in ("knn10", "knn100", "knn1000", "cka"):
                e = np.array(g[k]) - np.array(g[f"{k}_mu0"])
                row += [float(e[0].max()), float(e[1:].max())]
            rows.append(row)
    lines += table(
        [
            "Pythia final (Gemini)",
            "target",
            "k10 L0",
            "k10 deep",
            "k100 L0",
            "k100 deep",
            "k1000 L0",
            "k1000 deep",
            "CKA L0",
            "CKA deep",
        ],
        rows,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    d = Data(args.out)
    lines = ["# exp-022 phase 1 summary", ""]
    v1(args.out, lines)
    v2(args.out, lines)
    v3(d, lines)
    v4(d, args.out, lines)
    v5(d, args.out, lines)
    v6_v7(d, args.out, lines)
    v8(d, lines)
    v9_v11(d, lines)
    v10(d, lines)
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(lines)} lines")


if __name__ == "__main__":
    main()
