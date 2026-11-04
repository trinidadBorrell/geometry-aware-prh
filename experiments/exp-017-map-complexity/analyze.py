"""exp-017: summarise linear vs nonlinear held-out prediction (real.jsonl, synthetic.jsonl).

Per pair and direction, each readout is averaged over folds. The nonlinear gain of a readout is
(best of knn / krr / mlp) - ridge, where "best" is chosen per pair and direction by mean R^2. The
fold-level sign test counts folds where the chosen nonlinear method beats ridge on R^2.

    uv run python experiments/exp-017-map-complexity/analyze.py --root results/exp-017-map-complexity
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NONLIN = ("knn", "krr", "mlp")
READ = ("r2", "top1", "mknn")


def per_pair(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["id"], r["dir"])].append(r)
    out = []
    for (pid, d), rs in groups.items():
        mean = {m: {k: float(np.mean([r["methods"][m][k] for r in rs])) for k in READ}
                for m in ("ridge", *NONLIN)}  # fmt: skip
        best = max(NONLIN, key=lambda m: mean[m]["r2"])
        wins = sum(r["methods"][best]["r2"] > r["methods"]["ridge"]["r2"] for r in rs)
        gains = [r["methods"][best]["r2"] - r["methods"]["ridge"]["r2"] for r in rs]
        out.append({
            "id": pid, "dir": d, "group": rs[0]["group"], "folds": len(rs), "best": best,
            "mean": mean, "wins": wins,
            "gain": {k: mean[best][k] - mean["ridge"][k] for k in READ},
            "gain_sd": float(np.std(gains, ddof=1)) if len(gains) > 1 else float("nan"),
            "layers": (rs[0].get("layer_a"), rs[0].get("layer_b")),
            "kept": (float(np.mean([r["kept_a"] for r in rs])), float(np.mean([r["kept_b"] for r in rs]))),
        })  # fmt: skip
    return out


def fmt_group(pairs: list[dict], key: str) -> str:
    lines = [
        f"| {key} | dir | pairs | ridge R^2 | best nonlinear R^2 | gain R^2 [min, max] | gain top-1 | gain mkNN "
        "| best method | folds won |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    by = defaultdict(list)
    for p in pairs:
        by[(p["group"], p["dir"])].append(p)
    for (g, d), ps in sorted(by.items()):
        ridge = np.median([p["mean"]["ridge"]["r2"] for p in ps])
        nl = np.median([p["mean"][p["best"]]["r2"] for p in ps])
        gain = [p["gain"]["r2"] for p in ps]
        g1 = np.median([p["gain"]["top1"] for p in ps])
        gk = np.median([p["gain"]["mknn"] for p in ps])
        best = defaultdict(int)
        for p in ps:
            best[p["best"]] += 1
        won = sum(p["wins"] for p in ps)
        tot = sum(p["folds"] for p in ps)
        lines.append(
            f"| {g} | {d} | {len(ps)} | {ridge:.3f} | {nl:.3f} | {np.median(gain):+.3f} "
            f"[{min(gain):+.3f}, {max(gain):+.3f}] | {g1:+.3f} | {gk:+.3f} | "
            + ", ".join(f"{k} {v}" for k, v in sorted(best.items()))
            + f" | {won}/{tot} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    parts = []
    for name in ("real", "synthetic"):
        f = args.root / f"{name}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open()]
        pairs = per_pair(rows)
        if name == "synthetic":
            for p in pairs:
                p["group"] = p["id"]
        parts.append(
            f"## {name} ({len(rows)} fold rows, {len(pairs)} pair-directions)\n\n"
            + fmt_group(pairs, "group")
        )
        (args.root / f"{name}_pairs.json").write_text(json.dumps(pairs, indent=1))
    text = "\n\n".join(parts) + "\n"
    (args.root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
