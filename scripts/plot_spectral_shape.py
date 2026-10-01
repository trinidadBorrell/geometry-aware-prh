#!/usr/bin/env python3
"""Shape of fitted log-spectra against random spectra on the same budget.

Reads saved one-sided factors. Does not refit. The random draw has a Haar
eigenbasis and isotropic log-eigenvalues, clipped to [1/4, 4] and scaled to
the distortion budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.plots import _save  # noqa: E402

LOG4 = float(np.log(4.0))
FAMILIES = (
    ("exp_s", "na", "exp(S)"),
    ("oblique_diag", "basis", "oblique diagonal"),
    ("fixed_diag", "na", "fixed diagonal"),
    ("block4", "strong", "block size 4"),
)
COLORS = {
    "dinov2-small": "#1b4f72",
    "vit-in21k-small": "#b9770e",
    "clip-laion-base": "#196f3d",
}


def spectrum(b: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigvalsh(0.5 * (b + b.T))
    return np.sort(np.log(np.clip(ev, 1e-12, None)))


def skewness(ell: np.ndarray) -> float:
    z = ell - ell.mean()
    m2 = float(np.mean(z**2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(z**3) / m2**1.5)


def participation(ell: np.ndarray) -> float:
    """1 means equal |log eigenvalues|. 1/q means one spike carries the distortion."""
    s2 = float(np.sum(ell**2))
    s4 = float(np.sum(ell**4))
    if s4 <= 0.0:
        return float("nan")
    return (s2**2) / (ell.size * s4)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def random_ell(rho: float, rng: np.random.Generator, q: int = 32) -> np.ndarray:
    target = float(rho) * q
    direction = rng.normal(size=q)
    direction *= np.sqrt(target) / np.linalg.norm(direction)
    clipped = np.clip(direction, -LOG4, LOG4)
    norm = float(np.dot(clipped, clipped))
    clipped = np.clip(clipped * np.sqrt(target / max(norm, 1e-12)), -LOG4, LOG4)
    return np.sort(clipped)


def load_one_sided(runs: Path) -> list[dict]:
    rows = []
    allowed = {(family, profile) for family, profile, _ in FAMILIES}
    for path in runs.glob("*__a_only__*__true.factors.npz"):
        parts = path.name.replace(".factors.npz", "").split("__")
        side_a, side_b, mode, family, profile, rho, seed, corr = parts
        if mode != "a_only" or corr != "true" or (family, profile) not in allowed:
            continue
        blob = np.load(path)
        ell = spectrum(np.asarray(blob["selected_parts_0_B"], float))
        rows.append(
            {
                "side_a": side_a,
                "side_b": side_b,
                "family": family,
                "profile": profile,
                "rho": 0.1 if rho == "rho0p1" else 0.4,
                "seed": seed,
                "ell": ell,
                "skew": skewness(ell),
                "participation": participation(ell),
            }
        )
    return rows


def partner_rms(rows: list[dict]) -> float:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["side_a"], row["seed"])].append(row["ell"])
    dists = []
    for items in buckets.values():
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                dists.append(rms(left, right))
    return float(np.median(dists)) if dists else float("nan")


def summarise(rows: list[dict], rng: np.random.Generator) -> dict:
    out = {}
    for family, profile, label in FAMILIES:
        for rho in (0.1, 0.4):
            part = [r for r in rows if r["family"] == family and r["profile"] == profile and r["rho"] == rho]
            draws = [random_ell(rho, rng) for _ in range(40)]
            fitted_vs_random = []
            seen = {}
            for row in part:
                if row["seed"] != "seed0":
                    continue
                seen.setdefault(row["side_a"], row["ell"])
            for ell in seen.values():
                for draw in draws:
                    fitted_vs_random.append(rms(ell, draw))
            random_vs_random = [rms(a, b) for i, a in enumerate(draws) for b in draws[i + 1 :]]
            out[f"{family}:{rho}"] = {
                "label": label,
                "rho": rho,
                "n": len(part),
                "partner_rms": partner_rms(part),
                "fitted_vs_random_rms": float(np.median(fitted_vs_random)),
                "random_vs_random_rms": float(np.median(random_vs_random)),
                "skew_fitted": float(np.median([r["skew"] for r in part])),
                "skew_random": float(np.median([skewness(d) for d in draws])),
                "participation_fitted": float(np.median([r["participation"] for r in part])),
                "participation_random": float(np.median([participation(d) for d in draws])),
            }
    return out


def _median_curve(rows: list[dict]) -> np.ndarray:
    return np.median(np.stack([r["ell"] for r in rows], axis=0), axis=0)


def plot(rows: list[dict], rng: np.random.Generator, dest: Path) -> None:
    langs = sorted({r["side_a"] for r in rows if r["family"] == "exp_s"})
    fig, axes = plt.subplots(2, len(langs), figsize=(2.3 * len(langs), 5.6), sharey=True, sharex=True)
    x = np.arange(1, 33)
    for row_i, rho in enumerate((0.1, 0.4)):
        draws = [random_ell(rho, rng) for _ in range(25)]
        for col, lang in enumerate(langs):
            ax = axes[row_i, col]
            for draw in draws:
                ax.plot(x, draw, color="0.75", lw=0.6, zorder=1)
            part = [r for r in rows if r["family"] == "exp_s" and r["rho"] == rho and r["side_a"] == lang]
            for partner, color in COLORS.items():
                curves = [r for r in part if r["side_b"] == partner]
                if not curves:
                    continue
                ax.plot(x, _median_curve(curves), color=color, lw=1.6, zorder=2, label=partner if row_i == 0 and col == 0 else None)
            if row_i == 0:
                ax.set_title(lang, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"budget {rho:g}\nsorted log eigenvalue")
            ax.axhline(0.0, color="0.5", lw=0.4)
            ax.set_xlim(1, 32)
    axes[0, 0].plot([], [], color="0.75", lw=1.2, label="random spectrum")
    axes[0, 0].legend(fontsize=7, loc="upper left", frameon=False)
    axes[-1, 0].set_xlabel("eigenvalue order")
    fig.suptitle("One-sided exp(S). Each coloured curve is the seed-median spectrum of one vision partner.", fontsize=10)
    _save(fig, dest)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    positions = []
    labels = []
    skew_data = []
    part_data = []
    random_skew = {}
    random_part = {}
    for family, profile, label in FAMILIES:
        for rho in (0.1, 0.4):
            part = [r for r in rows if r["family"] == family and r["profile"] == profile and r["rho"] == rho]
            positions.append(len(positions))
            short = {"exp(S)": "exp(S)", "oblique diagonal": "oblique", "fixed diagonal": "fixed PCA", "block size 4": "block 4"}[label]
            labels.append(f"{short}\n{rho:g}")
            skew_data.append([r["skew"] for r in part])
            part_data.append([r["participation"] for r in part])
            draws = [random_ell(rho, rng) for _ in range(40)]
            random_skew[len(positions) - 1] = float(np.median([skewness(d) for d in draws]))
            random_part[len(positions) - 1] = float(np.median([participation(d) for d in draws]))
    for ax, data, random_ref, ylab in (
        (axes[0], skew_data, random_skew, "skewness of log eigenvalues"),
        (axes[1], part_data, random_part, "participation ratio"),
    ):
        ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        ax.scatter(positions, [random_ref[i] for i in positions], color="#922b21", zorder=3, label="random feasible spectrum")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylab)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("One-sided fits. Boxes are vision–language pairs and seeds.", fontsize=10)
    _save(fig, dest.with_name("spectral_shape_summary.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    args = parser.parse_args()
    runs = Path(args.work) / "results" / "structured_metric_factors" / "runs"
    fig_dir = Path(args.work) / "results" / "structured_metric_factors" / "figures"
    rng = np.random.default_rng(1)
    rows = load_one_sided(runs)
    if not rows:
        raise SystemExit(f"no one-sided factors in {runs}")
    summary = summarise(rows, rng)
    dest = fig_dir / "spectral_shape.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, np.random.default_rng(1), fig_dir / "spectral_shape.png")
    for key, row in summary.items():
        print(
            key,
            "partner",
            round(row["partner_rms"], 3),
            "random_pair",
            round(row["random_vs_random_rms"], 3),
            "fitted_vs_random",
            round(row["fitted_vs_random_rms"], 3),
            "skew",
            round(row["skew_fitted"], 3),
            "vs",
            round(row["skew_random"], 3),
            "participation",
            round(row["participation_fitted"], 3),
            "vs",
            round(row["participation_random"], 3),
        )
    print(f"wrote {fig_dir / 'spectral_shape.png'}", flush=True)


if __name__ == "__main__":
    main()
