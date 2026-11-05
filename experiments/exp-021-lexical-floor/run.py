"""exp-021: is the step-0 alignment of exp-018 a word-count effect? No network, CPU only.

Stage `grids`: per-LM-layer profiles from exp-018's stored `pairs/*.npz`: for each series,
checkpoint, target and measure, the best excess over target layers at every LM layer. Separates
layer 0 (the embedding table) from the deep layers without re-running anything.

Stage `kernel`: caption features built from token counts alone, scored against cached features
(calibrated mutual kNN at every k, linear CKA, every target layer):
- bow_tf: l2-normalised token-frequency rows, the d -> infinity limit of a random table
  (no outlier clamp: a 0.95 quantile clamp is meaningless on sparse rows; large d bridges this);
- bow_tfidf: the same with smoothed idf weights;
- table_d<d>_s<seed>: frequency rows x a Gaussian table, then the PRH clamp and l2 norm;
- length: caption length only, as a point on a quarter circle (cosine order = length order);
- destroyed: one fresh random vector per caption (no token identity), through the PRH steps.
A real step-0 Pythia layer 0 must reproduce the score stored by exp-018, or the job stops.

Stage `summary`: `summary.md` from both.

    uv run python experiments/exp-021-lexical-floor/run.py --out <dir> --exp016 <dir> --exp018 <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "exp-016-positive-controls"))
import measure as M16  # noqa: E402  PRH preprocessing, as every earlier experiment

from geoprh import knn_sweep as ksw  # noqa: E402

DINO = "vit_giant_patch14_dinov2.lvd142m__cls"
CLIP = "vit_huge_patch14_clip_224.laion2b__cls"
REF = "EleutherAI__pythia-1.4b__gemini"
SERIES = ["EleutherAI__pythia-160m", "EleutherAI__pythia-410m"]
TOKENIZER = "EleutherAI/pythia-160m"
MAX_TOKENS = 2048  # as exp-020 extract_lm.py
FINAL = "step143000"
# Stated before the run (reports/Embedding layer alignment with vision.md).
PREDICTED = {DINO: 0.42, CLIP: 0.33, REF: 0.72}


def load_extract_lm():
    spec = importlib.util.spec_from_file_location(
        "extract_lm", ROOT / "exp-020-lm-data-pilot" / "extract_lm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- stage: grids


def stage_grids(out: Path, exp018: Path) -> None:
    rows = []
    for line in (exp018 / "pairs.jsonl").read_text().splitlines():
        r = json.loads(line)
        z = np.load(exp018 / "pairs" / f"{r['id']}.npz")
        m = re.match(r"(.+)@(step\d+)__", r["a"])
        measures = {"cka": "cka_lin", "knn10": "mutual_knn"}
        for k in r.get("knn_sweep", {}):
            measures[f"knn{k}"] = f"sweep_k{k}"
        res = {}
        for name, key in measures.items():
            s, mu = z[f"{key}_score"], z[f"{key}_mu0"]
            e = s - mu
            j = e.argmax(1)  # best target layer for each LM layer
            i = np.arange(len(j))
            res[name] = {"excess": e[i, j].tolist(), "score": s[i, j].tolist(), "col": j.tolist()}
        rows.append(
            {
                "series": m[1],
                "rev": m[2],
                "step": int(m[2][4:]),
                "target": r["b"],
                "group": r["group"],
                "tag": r["tag"],
                "n": r["n"],
                "caption": r["caption"],
                "layers": r["names_a"],
                "measures": res,
            }
        )
    (out / "layer0_grids.json").write_text(json.dumps(rows))
    print(f"grids: {len(rows)} pairs", flush=True)


# ---------------------------------------------------------------- features


def token_counts(texts: list[str], tok) -> tuple[sp.csr_matrix, np.ndarray]:
    """[n, V] token counts over the V tokens present, and caption lengths in tokens."""
    ids = tok(texts, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)["input_ids"]
    lengths = np.array([len(x) for x in ids])
    assert lengths.min() > 0
    rows = np.repeat(np.arange(len(ids)), lengths)
    vocab, cols = np.unique(np.concatenate([np.asarray(x) for x in ids]), return_inverse=True)
    counts = sp.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(ids), len(vocab))
    )
    counts.sum_duplicates()
    return counts, lengths


def prh(x: torch.Tensor) -> torch.Tensor:
    """exp-016 `prepared_layers` for one layer: clamp at the 0.95 quantile, l2 norm."""
    from aristotelian.prh.preprocess import prepare_features

    return F.normalize(prepare_features(x[:, None], q=0.95, exact=False), dim=-1)[:, 0]


def variant(name: str, counts: sp.csr_matrix, lengths: np.ndarray) -> torch.Tensor:
    n = counts.shape[0]
    tf = sp.diags(1 / lengths.astype(np.float32)) @ counts
    if name == "bow_tf":
        return F.normalize(torch.from_numpy(tf.toarray()), dim=-1)
    if name == "bow_tfidf":
        df = np.asarray((counts > 0).sum(0)).ravel()
        idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        return F.normalize(torch.from_numpy((tf @ sp.diags(idf)).toarray()), dim=-1)
    if name == "length":
        rank = pd.Series(lengths).rank().to_numpy() / n
        t = torch.from_numpy(rank * math.pi / 2).float()
        return torch.stack([t.cos(), t.sin()], 1)
    if name == "destroyed":
        gen = torch.Generator().manual_seed(0)
        return prh(torch.randn(n, 768, generator=gen))
    m = re.fullmatch(r"table_d(\d+)_s(\d+)", name)
    assert m, name
    d, seed = int(m[1]), int(m[2])
    gen = torch.Generator().manual_seed(seed)
    table = torch.randn(counts.shape[1], d, generator=gen).numpy()
    return prh(torch.from_numpy(np.asarray(tf @ table)).float())


def participation_ratio(x: torch.Tensor) -> tuple[float, torch.Tensor]:
    k = (x @ x.T).double()  # float32 mean / norm over n^2 entries drift on the CPU (~2% in CKA)
    k = k - k.mean(0, keepdim=True)
    k = k - k.mean(1, keepdim=True)
    return float(k.trace() ** 2 / (k * k).sum()), k


# ---------------------------------------------------------------- alignment


class Side:
    """Unit-norm centred Grams [L, n, n] and neighbour ranks of a list of prepared layers."""

    def __init__(self, layers: list[torch.Tensor], names: list[str], k_max: int):
        self.names = names
        self.sweep = ksw.build_sweep_cache(layers, k_max)
        grams = []
        for x in layers:
            k = (x @ x.T).double()  # see participation_ratio
            k = k - k.mean(0, keepdim=True)
            k = k - k.mean(1, keepdim=True)
            grams.append((k / k.norm()).float())
        self.grams = torch.stack(grams)


def align(v: Side, t: Side, ks: list[int], permutations: int, seed: int) -> dict:
    """Variant (one layer) against every target layer; the variant's rows are permuted."""
    assert v.grams.shape[0] == 1
    n = v.grams.shape[1]
    gram = v.grams[0]
    tflat = t.grams.flatten(1)
    cka = tflat @ gram.flatten()
    knn = ksw.scores(t.sweep, v.sweep, ks)[:, :, 0]  # [ks, target layers]
    gen = torch.Generator().manual_seed(seed)
    null_cka, null_knn = [], []
    for _ in range(permutations):
        perm = torch.randperm(n, generator=gen)
        null_cka.append(tflat @ gram[perm][:, perm].flatten())
        null_knn.append(ksw.scores(t.sweep, v.sweep, ks, perm)[:, :, 0])
    res = {
        "names": t.names,
        "cka": {
            "score": cka.tolist(),
            "mu0": torch.stack(null_cka).mean(0).tolist(),
        },
        "knn": {},
    }
    mu = torch.stack(null_knn).mean(0)
    for i, k in enumerate(ks):
        res["knn"][str(k)] = {"score": knn[i].tolist(), "mu0": mu[i].tolist()}
    j = int((knn[0] - mu[0]).argmax())
    o = ksw.overlaps(
        ksw.SweepCache(t.sweep.ranks[j : j + 1], t.sweep.idx[j : j + 1]), v.sweep, ks[:1]
    )[0, 0, 0]
    boot = torch.randint(0, n, (1000, n), generator=gen)
    means = o[boot].mean(1) - mu[0, j]
    res["knn"][str(ks[0])]["excess_ci"] = torch.quantile(
        means, torch.tensor([0.025, 0.975])
    ).tolist()
    return res


def stage_kernel(args, out: Path) -> None:
    from transformers import AutoTokenizer

    feats = {DINO: args.exp016, CLIP: args.exp016}
    ks = [int(k) for k in args.ks.split(",")]
    k_max = max(ks)
    x20 = load_extract_lm()
    caps = x20.captions(args.exp016 / "d2.parquet", args.limit)
    tok = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=args.cache_dir)
    widths = [int(d) for d in args.widths.split(",")]
    vit = [DINO, CLIP]
    trio = [DINO, CLIP, REF]
    retention = [f"{s}@{r}__gemini" for s in SERIES for r in ("step0", FINAL)]
    plan = [
        ("gemini", "bow_tf", trio + retention),
        ("gemini", "bow_tfidf", trio),
        ("gemini", "length", trio),
        ("gemini", "destroyed", [DINO]),
    ]
    for d in widths:
        if d in (768, 1024):
            plan += [
                ("gemini", f"table_d{d}_s0", trio),
                ("gemini", f"table_d{d}_s1", [DINO]),
            ]
        else:
            plan += [("gemini", f"table_d{d}_s0", vit)]
    plan += [("orig", v, vit) for v in ("bow_tf", "bow_tfidf", "length")]
    plan += [("orig", f"table_d{d}_s0", vit) for d in widths if d in (768, 1024)]

    path = out / "kernel.jsonl"
    done = set()
    if path.exists():
        done = {
            (r["caption"], r["variant"], r["target"])
            for r in map(json.loads, path.read_text().splitlines())
        }
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    targets: dict[str, Side] = {}

    def target(key: str) -> Side:
        if key not in targets:
            while len(targets) >= args.keep_targets:
                targets.pop(next(iter(targets)))
            f = args.exp018 / "features" / f"{key}__d2l.pt"
            if key in feats:
                f = feats[key] / "features" / f"{key}__d2l.pt"
            layers, names = M16.prepared_layers(f)
            targets[key] = Side(layers, names, k_max)
        return targets[key]

    counts_of = {c: token_counts(caps[c], tok) for c in ("gemini", "orig")}
    for c, (cnt, lengths) in counts_of.items():
        stats.setdefault(c, {})
        stats[c].update(
            n=cnt.shape[0],
            vocab=cnt.shape[1],
            median_tokens=float(np.median(lengths)),
            truncated=int((lengths >= MAX_TOKENS).sum()),
        )

    # Pipeline check first: exp-018's real step-0 layer 0 must reproduce its stored scores.
    series0 = f"{SERIES[0]}@step0__gemini"
    rec_path = args.exp018 / "pairs.jsonl"
    if rec_path.exists():
        rid = f"LV__{series0}__{DINO}__d2l"
        rec = next(
            (r for r in map(json.loads, rec_path.read_text().splitlines()) if r["id"] == rid), None
        )
        layers, _ = M16.prepared_layers(args.exp018 / "features" / f"{series0}__d2l.pt")
        real = Side(layers[:1], ["0"], k_max)
        res = align(real, target(DINO), ks, 5, 1)
        # compare at the stored cell: `best` is the largest excess, not the largest score
        got = res["knn"]["10"]["score"][rec["mutual_knn"]["best"]["cell"][1]]
        want = rec["mutual_knn"]["best"]["score"]
        got_c = res["cka"]["score"][rec["cka_lin"]["best"]["cell"][1]]
        want_c = rec["cka_lin"]["best"]["score"]
        print(f"check knn10 {got:.4f} vs {want:.4f}; cka {got_c:.4f} vs {want_c:.4f}", flush=True)
        assert rec["mutual_knn"]["best"]["cell"][0] == 0, "stored best cell is not LM layer 0"
        assert abs(got - want) < 2e-3 and abs(got_c - want_c) < 2e-3, "pipeline check failed"
        del real

    t0 = time.time()
    for caption, name, tkeys in plan:
        todo = [t for t in tkeys if (caption, name, t) not in done]
        if not todo:
            continue
        cnt, lengths = counts_of[caption]
        x = variant(name, cnt, lengths)
        if name.startswith("bow") or name == "length":
            stats[caption][f"r_{name}"], kc = participation_ratio(x)
            if name == "bow_tf":
                lt = variant("length", cnt, lengths)
                _, kl = participation_ratio(lt)
                stats[caption]["cka_bow_length"] = float((kc * kl).sum() / (kc.norm() * kl.norm()))
            del kc
            stats_path.write_text(json.dumps(stats, indent=1))
        v = Side([x], [name], k_max)
        del x
        for t in todo:
            res = align(
                v, target(t), ks, args.permutations, zlib.crc32(f"{caption}{name}{t}".encode())
            )
            with path.open("a") as fh:
                fh.write(json.dumps({"caption": caption, "variant": name, "target": t, **res}) + "\n")
            e = np.array(res["knn"][str(ks[0])]["score"]) - np.array(res["knn"][str(ks[0])]["mu0"])
            print(
                f"{caption} {name} x {t}: CKA {max(res['cka']['score']):.3f} "
                f"(null {max(res['cka']['mu0']):.3f}), knn{ks[0]} excess {e.max():.3f}  "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        del v


# ---------------------------------------------------------------- stage: summary


def best(row: dict, measure: str) -> tuple[float, float]:
    """(score, excess) at the target layer with the largest excess."""
    m = row["cka"] if measure == "cka" else row["knn"][measure[3:]]
    s, mu = np.array(m["score"]), np.array(m["mu0"])
    j = int((s - mu).argmax())
    return float(s[j]), float(s[j] - mu[j])


def stage_summary(out: Path) -> None:
    lines = ["# exp-021 summary", ""]
    kp = out / "kernel.jsonl"
    if kp.exists():
        rows = [json.loads(x) for x in kp.read_text().splitlines()]
        stats = json.loads((out / "stats.json").read_text())
        lines += ["```", json.dumps(stats, indent=1), "```", "", "## Kernels", ""]
        ks = sorted(int(k) for k in rows[0]["knn"])
        lines += [
            "| captions | variant | target | CKA | CKA null | "
            + " | ".join(f"knn{k} excess" for k in ks)
            + " |",
            "|" + "---|" * (5 + len(ks)),
        ]
        for r in rows:
            c, ce = best(r, "cka")
            lines.append(
                f"| {r['caption']} | {r['variant']} | {r['target']} | {c:.3f} | {c - ce:.3f} | "
                + " | ".join(f"{best(r, f'knn{k}')[1]:.3f}" for k in ks)
                + " |"
            )
        lines += ["", "## Predictions stated before the run (bag of tokens, Gemini captions)", ""]
        for t, want in PREDICTED.items():
            r = next(
                (
                    x
                    for x in rows
                    if (x["caption"], x["variant"], x["target"]) == ("gemini", "bow_tf", t)
                ),
                None,
            )
            if r:
                got = best(r, "cka")[0]
                verdict = "holds" if abs(got - want) <= 0.02 else "FAILS"
                lines.append(f"- {t}: predicted {want:.2f}, measured {got:.3f}: {verdict}")
    gp = out / "layer0_grids.json"
    if gp.exists():
        rows = json.loads(gp.read_text())
        lines += ["", "## Layer 0 vs best deeper layer (n = 10240, best target layer)", ""]
        lines += [
            "| series | target | step | measure | layer 0 excess | best deeper excess | deeper layer |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
        for r in sorted(rows, key=lambda x: (x["series"], x["target"], x["step"])):
            if r["tag"] != "d2l":
                continue
            for m in ("knn10", "knn1000", "cka"):
                if m not in r["measures"]:
                    continue
                e = np.array(r["measures"][m]["excess"])
                j = int(e[1:].argmax()) + 1
                lines.append(
                    f"| {r['series'].split('__')[-1]} | {r['target'][:18]} | {r['step']} | {m} | "
                    f"{e[0]:.3f} | {e[j]:.3f} | {r['layers'][j]} |"
                )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"summary: {len(lines)} lines", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp016", type=Path, required=True)
    p.add_argument("--exp018", type=Path, required=True)
    p.add_argument("--stages", default="grids,kernel,summary")
    p.add_argument("--ks", default="10,30,100,300,1000")
    p.add_argument("--widths", default="768,1024,64,256,4096,16384")
    p.add_argument("--permutations", type=int, default=30)
    p.add_argument("--keep-targets", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    if "grids" in stages:
        stage_grids(args.out, args.exp018)
    if "kernel" in stages:
        stage_kernel(args, args.out)
    if "summary" in stages:
        stage_summary(args.out)


if __name__ == "__main__":
    main()
