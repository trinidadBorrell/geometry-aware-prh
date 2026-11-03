"""exp-016 arm C: synthetic dose-response of every measure to the amount of shared content.

Two "models" see n = 1024 samples through different encoders. Each sample has a latent per
model made of shared coordinates (the same for A and B) and private coordinates (drawn
independently for A and B), all standard normal, the private ones scaled by sigma. Each model maps
its latent through its own random 3-layer tanh network (width 1024; independent weights for A and
B), standardizes each layer and adds Gaussian noise (sd 0.3); the three layers are its "layers".

Two sweeps, 3 seeds each (new latents, weights and noise):
- `count`: a latent of m = 12 coordinates, the first m_s = round(rho * 12) shared and the other
  12 - m_s private (sigma = 1), rho in {0, 0.1, 0.25, 0.5, 0.75, 1}. At rho = 1 both models encode
  the same content with different nonlinear encoders (the best two encoders can agree when
  nothing is private); at rho = 0 they share nothing and every measure must be at its null.
- `scale`: all 12 shared coordinates plus 12 private ones scaled by sigma in
  {0, 0.1, 0.25, 0.5, 0.75, 1}: the same shared content, with private content of growing variance
  (private variance fraction 12 sigma^2 / (12 + 12 sigma^2), 0 to 0.5).

Measures as `measure.py` (mutual kNN k = 10, linear CKA, ORC given density and density at
k = 10, 30; 200-permutation nulls). Outputs `dose_response.jsonl` and `fig_dose_response.png`.

    uv run python experiments/exp-016-positive-controls/dose_response.py --out results/exp-016-positive-controls/dose --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SWEEPS = {
    "count": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],  # rho, shared fraction of the 12 coordinates
    "scale": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],  # sigma, scale of 12 private coordinates
}
M, WIDTH, DEPTH, NOISE, N = 12, 1024, 3, 0.3, 1024


def _measure():
    # imported by name (not from a spec) so worker processes can unpickle its functions
    sys.path.insert(0, str(HERE))
    import measure

    return measure


def encoder(latent: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """[n, m] -> [n, DEPTH, WIDTH]: random tanh network, each layer standardized, plus noise."""
    x, layers = latent, []
    for _ in range(DEPTH):
        w = torch.randn(x.shape[1], WIDTH, generator=gen) * (2.0 / x.shape[1] ** 0.5)
        x = torch.tanh(x @ w)
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
        layers.append(x + NOISE * torch.randn(x.shape, generator=gen))
    return torch.stack(layers, 1)


def latents(sweep: str, v: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    m_s, m_p, sigma = (round(v * M), M - round(v * M), 1.0) if sweep == "count" else (M, M, v)
    shared = torch.randn(N, m_s, generator=gen)
    za = torch.cat([shared, sigma * torch.randn(N, m_p, generator=gen)], 1)
    zb = torch.cat([shared, sigma * torch.randn(N, m_p, generator=gen)], 1)
    return za, zb


def make_pair(sweep: str, v: float, seed: int, root: Path) -> tuple[str, str]:
    gen = torch.Generator().manual_seed(seed)
    keys = []
    for name, z in zip("AB", latents(sweep, v, gen), strict=True):
        key = f"{sweep}{v:.2f}_s{seed}_{name}__d1"
        obj = {"layers": encoder(z, gen).half(), "layer_idx": list(range(DEPTH)), "meta": {}}
        (root / "features").mkdir(parents=True, exist_ok=True)
        torch.save(obj, root / "features" / f"{key}.pt")
        keys.append(key)
    return keys[0], keys[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    ms = _measure()
    pairs = [
        (sweep, v, s, *make_pair(sweep, v, 1000 * j + 100 * s + i, args.out))
        for j, (sweep, values) in enumerate(SWEEPS.items())
        for i, v in enumerate(values)
        for s in range(args.seeds)
    ]
    ms.stage_geometry(args.out, args.workers, args.out / "tmp")
    caches = ms.Caches(args.out, "cpu", keep=2)
    rows = []
    for i, (sweep, v, seed, a, b) in enumerate(pairs):
        ca, names_a = caches.get(a)
        cb, names_b = caches.get(b)
        _, row = ms.align_pair(ca, cb, names_a, names_b, args.permutations, 0.05, seed=i)
        ga, gb = (np.load(args.out / "geometry" / f"{x}.npz") for x in (a, b))
        for k in ms.KS:
            oa, da = ms.fill_degenerate(ga[f"orc_k{k}"], ga[f"dens_k{k}"])
            ob, db = ms.fill_degenerate(gb[f"orc_k{k}"], gb[f"dens_k{k}"])
            pairs_k = {
                "orc": (ms.paired.residualize(oa, da), ms.paired.residualize(ob, db)),
                "dens": (ms.paired.standardized_ranks(da), ms.paired.standardized_ranks(db)),
            }
            for q, (za, zb) in pairs_k.items():
                _, row[f"{q}_k{k}"] = ms.rank_pair(
                    za, zb, args.permutations, 0.05, i, names_a, names_b
                )
        row.update({"sweep": sweep, "value": v, "seed": seed, "twonn_a": ga["twonn"].tolist()})
        rows.append(row)
        print(
            f"{sweep} {v:.2f} seed {seed}: mkNN {row['mutual_knn']['best']['score']:.3f}, "
            f"CKA {row['cka_lin']['best']['score']:.3f}, "
            f"ORC|dens {row['orc_k10']['best']['score']:+.3f}, "
            f"density {row['dens_k10']['best']['score']:+.3f}, "
            f"TwoNN A {np.round(ga['twonn'], 1).tolist()}",
            flush=True,
        )
    with (args.out / "dose_response.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    plot(rows, args.out)


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    measures = [
        ("mutual_knn", "mutual kNN (k = 10)"),
        ("cka_lin", "linear CKA"),
        ("orc_k10", "ORC | density, Spearman (k = 10)"),
        ("dens_k10", "density, Spearman (k = 10)"),
    ]
    xlabels = {
        "count": "shared fraction of the 12 latent coordinates (rho)",
        "scale": "scale of 12 private coordinates (sigma), 12 shared",
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharey="col")
    for row_axes, (sweep, values) in zip(axes, SWEEPS.items(), strict=True):
        sub = [r for r in rows if r["sweep"] == sweep]
        for ax, (m, title) in zip(row_axes, measures, strict=True):
            for which, colour, label in (
                ("score", "#2a78d6", "best layer pair"),
                ("tau", "#52514e", "null 95% threshold"),
            ):
                vals = np.array(
                    [[r[m]["best"][which] for r in sub if r["value"] == v] for v in values]
                )
                ax.fill_between(values, vals.min(1), vals.max(1), color=colour, alpha=0.18, lw=0)
                ax.plot(values, vals.mean(1), color=colour, marker="o", ms=3, label=label)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xlabels[sweep], fontsize=8)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Arm C: synthetic dose-response, two random nonlinear encoders (mean and min-max over 3 seeds)",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_dose_response.{ext}", dpi=200)


if __name__ == "__main__":
    main()
