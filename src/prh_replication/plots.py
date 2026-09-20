"""Publication-style matplotlib figures written to disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def heatmap(matrix: np.ndarray, xticks: list[str], yticks: list[str], title: str, path: Path, cbar: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(xticks) + 2), max(5, 0.45 * len(yticks) + 2)))
    im = ax.imshow(matrix, cmap="viridis", origin="upper")
    ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right")
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar)
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
