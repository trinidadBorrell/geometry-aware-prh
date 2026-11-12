"""exp-023: alignment above the TF-IDF floor, Gemini vs original captions on the same 10240 images.

Differences of CKA or overlap are level differences, not variance components. Also reports the gap
scaled by the room left above the floor, because a high floor leaves less room to gain.

    uv run python experiments/exp-023-prh-lexical-floor/gap_d2.py --exp022 <exp-022 results dir> --out <dir>
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
VIT = {DINO: "DINOv2", CLIP: "CLIP", MAE: "MAE", AUGREG: "AugReg"}
LMS = [
    "bigscience__bloomz-1b7",
    "bigscience__bloomz-7b1",
    "openlm-research__open_llama_7b",
    "huggyllama__llama-13b",
]
PY = ["pythia-160m", "pythia-160m-seed1", "pythia-410m", "pythia-410m-seed1"]
FINAL = "step143000"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp022", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    old = {g["id"]: g for g in json.loads((a.exp022 / "grids_full.json").read_text())}
    new = {g["id"]: g for g in json.loads((a.exp022 / "phase2" / "phase2_grids.json").read_text())}
    rows = {}
    for line in (a.exp022 / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["stage"] in ("lexical", "targets"):
            rows[(r["caption"], r["variant"], r["target"])] = r

    def floor(cap, lm, t, m, last):
        r = rows.get((cap, f"tok-{lm}_tfidf", t)) or rows[(cap, "tfidf", t)]
        s = (
            np.array(r["cka"]) - np.array(r["cka_mu0"])
            if m == "cka"
            else np.array(r["knn"][m[3:]]) - r["knn_mu0"][m[3:]]
        )
        return float(s[-1, 0] if last else s.max())

    def grid(cap, lm, t):
        if lm.startswith("pythia"):
            k = f"LV__EleutherAI__{lm}@{FINAL}__{cap}__{t}__d2l"
            return old[k] if cap == "gemini" else new[k]
        return old[f"B3_{cap}__{lm}__{cap}__{t}__d2l"]

    out = []
    for lm, vits in [
        *[(x, [DINO, CLIP, MAE, AUGREG]) for x in LMS],
        *[(x, [DINO, CLIP]) for x in PY],
    ]:
        for t in vits:
            for m in ("knn10", "cka"):
                for last in (False, True):
                    row = {
                        "lm": lm.split("__")[-1],
                        "vit": VIT[t],
                        "measure": m,
                        "cell": "last" if last else "best",
                    }
                    for cap in ("orig", "gemini"):
                        g = grid(cap, lm, t)
                        e = np.array(g[m]) - np.array(g[f"{m}_mu0"])
                        deep = float(e[-1, -1] if last else e[1:].max())
                        f = floor(cap, lm, t, m, last)
                        row[cap] = {
                            "floor": f,
                            "deep": deep,
                            "gap": deep - f,
                            "scaled": (deep - f) / (1 - f),
                            "l0": float(e[0, -1] if last else e[0].max()),
                        }
                    out.append(row)
    (a.out / "gap_d2.json").write_text(json.dumps(out, indent=1))

    def show(rs, title):
        o, g = ([r[c] for r in rs] for c in ("orig", "gemini"))
        val = lambda x, k: np.array([v[k] for v in x])  # noqa: E731
        d = val(g, "gap") - val(o, "gap")
        ds = val(g, "scaled") - val(o, "scaled")
        print(
            f"{title:42s} n={len(rs):2d} | floor {val(o, 'floor').mean():.3f} > {val(g, 'floor').mean():.3f} | deep {val(o, 'deep').mean():.3f} > {val(g, 'deep').mean():.3f} | "
            f"gap {val(o, 'gap').mean():.3f} > {val(g, 'gap').mean():.3f} (diff {d.mean():+.3f}, {d.min():+.3f}..{d.max():+.3f}, Gem larger {int((d > 0).sum())}/{len(d)}) | scaled {val(o, 'scaled').mean():.3f} > {val(g, 'scaled').mean():.3f} ({int((ds > 0).sum())}/{len(ds)})"
        )

    for m in ("knn10", "cka"):
        for c in ("best", "last"):
            base = [r for r in out if r["measure"] == m and r["cell"] == c]
            show(base, f"{m} {c} all")
            show([r for r in base if not r["lm"].startswith("pythia")], f"{m} {c} larger LMs")
            show([r for r in base if r["lm"].startswith("pythia")], f"{m} {c} Pythia")
            for v in VIT.values():
                show(
                    [r for r in base if r["vit"] == v and not r["lm"].startswith("pythia")],
                    f"{m} {c} larger x {v}",
                )


if __name__ == "__main__":
    main()
