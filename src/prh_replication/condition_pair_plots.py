"""Holdout excess by model pair for identity, one-sided, shared, and separate fits.

Reads the joined structured-metric table. Does not refit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter

LANG_ORDER = (
    "qwen2-7b",
    "qwen2.5-7b",
    "qwen3-8b-base",
    "olmo-7b-0724",
    "olmo2-1124-7b",
    "olmo-3-1025-7b",
)
VIS_ORDER = ("dinov2-small", "vit-in21k-small", "clip-laion-base")
FAMILIES = (
    ("exp_s", "na", "exp(S)"),
    ("fixed_diag", "na", "fixed PCA diagonal"),
    ("oblique_diag", "basis", "oblique diagonal"),
    ("block4", "strong", "block 4"),
)
MODES = (
    ("identity", "Identity", "#4d4d4d"),
    ("a_only", "One-sided", "#2c5aa0"),
    ("shared_pc", "Shared", "#d68910"),
    ("separate", "Separate", "#1e8449"),
)
SHORT = {
    "qwen2-7b": "Qwen2-7B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen3-8b-base": "Qwen3-8B",
    "olmo-7b-0724": "OLMo-7B",
    "olmo2-1124-7b": "OLMo-2-7B",
    "olmo-3-1025-7b": "OLMo-3-7B",
    "dinov2-small": "DINOv2-S",
    "vit-in21k-small": "ViT-IN21k-S",
    "clip-laion-base": "CLIP-B",
}


def true_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("correspondence", "true") == "true"]


def ordered_pairs(rows: list[dict], kind: str) -> list[tuple[str, str]]:
    found = {(row["side_a"], row["side_b"]) for row in rows if row["kind"] == kind}
    ordered: list[tuple[str, str]] = []
    if kind == "vl":
        for lang in LANG_ORDER:
            for vis in VIS_ORDER:
                if (lang, vis) in found:
                    ordered.append((lang, vis))
    else:
        for left in LANG_ORDER:
            for right in LANG_ORDER:
                if (left, right) in found:
                    ordered.append((left, right))
    rest = sorted(found - set(ordered))
    return ordered + rest


def mode_values(
    rows: list[dict],
    *,
    kind: str,
    family: str,
    profile: str,
    mode: str,
    rho: float,
    pair: tuple[str, str],
    field: str = "exploratory_excess",
) -> list[float]:
    side_a, side_b = pair
    if mode == "identity":
        return [
            float(row[field])
            for row in rows
            if row["kind"] == kind
            and row["family"] == "identity"
            and row["side_a"] == side_a
            and row["side_b"] == side_b
        ]
    return [
        float(row[field])
        for row in rows
        if row["kind"] == kind
        and row["family"] == family
        and row["profile"] == profile
        and row["mode"] == mode
        and abs(float(row["rho"]) - float(rho)) < 1e-9
        and row["side_a"] == side_a
        and row["side_b"] == side_b
    ]


def pair_label(side_a: str, side_b: str) -> str:
    return f"{SHORT.get(side_a, side_a)}\n{SHORT.get(side_b, side_b)}"


def plot_conditions_by_pair(
    plt,
    rows: list[dict],
    *,
    kind: str,
    family: str,
    profile: str,
    title: str,
    path: Path,
    field: str = "exploratory_excess",
) -> None:
    rows = true_rows(rows)
    pairs = ordered_pairs(rows, kind)
    if not pairs:
        return
    rhos = (0.1, 0.4)
    width = 0.18
    x = np.arange(len(pairs))
    fig_w = 16.5 if kind == "vl" else 8.2
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 8.4), sharex=True, sharey=True)
    for ax, rho in zip(axes, rhos):
        if kind == "vl":
            for group in range(0, len(pairs), 3):
                if (group // 3) % 2 == 1:
                    ax.axvspan(group - 0.5, min(group + 3, len(pairs)) - 0.5, color="#f3f4f6", zorder=0)
        for index, (mode, label, color) in enumerate(MODES):
            offset = (index - (len(MODES) - 1) / 2) * width
            medians = []
            for pair_i, pair in enumerate(pairs):
                vals = mode_values(
                    rows,
                    kind=kind,
                    family=family,
                    profile=profile,
                    mode=mode,
                    rho=rho,
                    pair=pair,
                    field=field,
                )
                medians.append(float(np.median(vals)) if vals else np.nan)
                if mode != "identity" and vals:
                    ax.scatter(
                        np.full(len(vals), x[pair_i] + offset),
                        vals,
                        s=12,
                        color=color,
                        zorder=3,
                        linewidths=0,
                    )
            ax.bar(
                x + offset,
                medians,
                width=width * 0.92,
                color=color,
                label=label,
                zorder=2,
            )
        ax.set_ylabel("Holdout excess CKA")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(f"{title},  ρ = {rho:g}")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    top = max(ax.get_ylim()[1] for ax in axes)
    axes[0].set_ylim(0, top * 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    axes[1].set_xticks(x, [pair_label(*pair) for pair in pairs], fontsize=7.5)
    axes[1].set_xlabel("Bars are seed medians. Dots are the three fit seeds. Identity is M = I and has one score.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _write(fig, path)


def _write(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_condition_pair_figures(rows: list[dict], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, kind_name in (("vl", "Vision–language"), ("ll", "Language–language")):
        for family, profile, label in FAMILIES:
            path = out / f"{kind}_{family}_{profile}_excess_by_pair.png"
            plot_conditions_by_pair(
                plt,
                rows,
                kind=kind,
                family=family,
                profile=profile,
                title=f"{kind_name}, {label}",
                path=path,
            )
            if path.exists():
                written.append(path)
    return written
