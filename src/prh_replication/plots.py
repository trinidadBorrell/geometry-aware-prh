"""Publication-style matplotlib figures written to disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def raster_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    path: Path,
    cbar: str,
    cmap: str,
    vmin: float,
    vmax: float,
    xlabel: str = "original feature ID",
    ylabel: str = "original feature ID",
    xticks: list[str] | None = None,
    yticks: list[str] | None = None,
    caption: str = "",
    square: bool = True,
) -> dict:
    """Heatmap with the image rasterised (safe for large d×d)."""
    d0, d1 = matrix.shape
    fig_w = 8.2 if xticks is None else max(7.0, 0.18 * len(xticks) + 2.5)
    fig_h = 7.4 if yticks is None else max(5.5, 0.18 * len(yticks) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        matrix,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        aspect="equal" if square else "auto",
        interpolation="nearest",
        rasterized=True,
    )
    if xticks is None:
        ax.set_xticks([0, d1 // 2, d1 - 1], [0, d1 // 2, d1 - 1])
    else:
        ax.set_xticks(range(len(xticks)), xticks, rotation=90, ha="right", fontsize=6)
    if yticks is None:
        ax.set_yticks([0, d0 // 2, d0 - 1], [0, d0 // 2, d0 - 1])
    else:
        ax.set_yticks(range(len(yticks)), yticks, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar, fraction=0.046, pad=0.04)
    if caption:
        fig.text(0.01, 0.01, caption, fontsize=7, va="bottom")
    finite = matrix[np.isfinite(matrix)]
    n_sat = int(np.sum(np.abs(finite) > vmax + 1e-15)) if finite.size and np.isfinite(vmax) else 0
    frac = float(n_sat / finite.size) if finite.size else 0.0
    _save(fig, path)
    return {"n_saturated": n_sat, "frac_saturated": frac, "vmin": vmin, "vmax": vmax}


def heatmap(
    matrix: np.ndarray,
    xticks: list[str],
    yticks: list[str],
    title: str,
    path: Path,
    cbar: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(xticks) + 2), max(5, 0.45 * len(yticks) + 2)))
    im = ax.imshow(matrix, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right")
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar)
    _save(fig, path)


def mean_sd_lines(
    xs: np.ndarray,
    partners: dict[str, np.ndarray],
    mean: np.ndarray,
    sd: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    logy: bool = False,
    ylim: tuple[float, float] | None = None,
    identity: float | None = 1.0,
    sd_available: bool = True,
    note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, ys in partners.items():
        ax.plot(xs, ys, color="0.65", lw=0.8, alpha=0.55)
    ax.plot(xs, mean, color="C0", lw=2.0, label="partner-sample mean")
    if sd_available and np.isfinite(sd).any():
        lo = mean - sd
        hi = mean + sd
        if logy:
            lo = np.clip(lo, 1e-12, None)
        ax.fill_between(xs, lo, hi, color="C0", alpha=0.22, label="±1 sample SD (weight space)")
    elif not sd_available:
        ax.plot([], [], " ", label="sample SD unavailable (n<2)")
    if identity is not None:
        ax.axhline(identity, color="k", ls="--", lw=1.0, label="identity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if note:
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)


def bars(labels: list[str], actual: list[float], null_mean: list[float], title: str, ylabel: str, path: Path) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(labels) + 3), 4.5))
    ax.bar(x - 0.18, actual, 0.35, label="actual")
    ax.bar(x + 0.18, null_mean, 0.35, label="shuffle mean")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def lines(xs: list[float], series: dict[str, list[float]], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, path)
