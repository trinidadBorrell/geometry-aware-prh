import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = "/workspace/results/emily/alignment"
FIGURES_DIR = "/workspace/results/emily/figures-0921"
TOPK = 10
CELL = 0.30  # inches per matrix cell

METRICS = ["cycle_knn", "mutual_knn", "cka", "unbiased_cka", "cknna"]

llm_models = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
    "openlm-research/open_llama_13b",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",
]

lvm_models = [
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b",
    "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k",
    "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
]

N_LLM = len(llm_models)
N_LVM = len(lvm_models)

TASK_SUBSTRINGS = [
    ("laion2b_ft_in12k", "laion2b_ft_in12k"),
    ("laion2b", "laion2b"),
    ("augreg_in21k", "augreg_in21k"),
    ("lvd142m", "lvd142m"),
    ("mae", "mae"),
]


def task_for_model(name):
    for substr, task in TASK_SUBSTRINGS:
        if substr in name:
            return task
    raise ValueError(f"no task match for {name}")


def short_label(name):
    return name.split("/")[-1]


def compact_lvm_label(name):
    task = task_for_model(name)
    base = name.split(".")[0]
    parts = base.split("_")
    size = parts[1] if len(parts) > 1 else base
    patch_match = re.search(r"patch(\d+)", base)
    patch = f"p{patch_match.group(1)}" if patch_match else ""
    variant = f"{size}_{patch}" if patch else size
    return f"{task}\n{variant}"


def to_alignment_filename(metric, topk, null_calibrate):
    m = metric + "_NC" if null_calibrate else metric
    fname = f"{m}_k{topk}.npy" if "knn" in m else f"{m}.npy"
    return os.path.join(RESULTS_DIR, fname)


def load_scores(metric, topk, null_calibrate):
    path = to_alignment_filename(metric, topk, null_calibrate)
    data = np.load(path, allow_pickle=True).item()
    return data["scores"]


def draw_panel(ax, mat, row_labels, col_labels, cmap, title, label_fontsize, diverging=False):
    if diverging:
        m = np.abs(mat).max()
        m = m if m > 0 else 1e-6
        im = ax.imshow(mat, cmap=cmap, vmin=-m, vmax=m, aspect="auto")
    else:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")  # no normalization — auto-scaled per panel
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=label_fontsize)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=label_fontsize)
    ax.set_title(title, fontsize=8)
    return im


def save_full(g, metric, out_path):
    n_rows, n_cols = len(g["row_labels"]), len(g["col_labels"])
    fig_h = max(n_rows, 2) * CELL + 2.0
    fig_w = n_cols * 3 * CELL + 3.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, 3, wspace=0.5, figure=fig)

    panels = [
        (g["baseline"], "Baseline", "viridis", False),
        (g["nc"], "Null-calibrated", "viridis", False),
        (g["delta"], "Delta (baseline - NC)", "RdBu_r", True),
    ]
    for col_idx, (mat, subtitle, cmap, diverging) in enumerate(panels):
        ax = fig.add_subplot(gs[col_idx])
        im = draw_panel(ax, mat, g["row_labels"], g["col_labels"], cmap, subtitle, label_fontsize=6, diverging=diverging)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{metric} — {g['title']}", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_delta_across_metrics(key, title, row_labels, col_labels, deltas_by_metric, out_path):
    n_rows, n_cols = len(row_labels), len(col_labels)
    n_metrics = len(deltas_by_metric)
    fig_h = max(n_rows, 2) * CELL + 2.0
    fig_w = n_cols * n_metrics * CELL + 2.5 * n_metrics
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, n_metrics, wspace=0.6, figure=fig)

    for col_idx, (metric, delta) in enumerate(deltas_by_metric.items()):
        ax = fig.add_subplot(gs[col_idx])
        im = draw_panel(ax, delta, row_labels, col_labels, "RdBu_r", metric, label_fontsize=6, diverging=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title} — delta across metrics", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


os.makedirs(FIGURES_DIR, exist_ok=True)

llm_labels = [short_label(m) for m in llm_models]
vis_labels = [compact_lvm_label(m) for m in lvm_models]

deltas_by_grouping = {
    "llmxllm": {"title": "LLM x LLM", "row_labels": llm_labels, "col_labels": llm_labels, "deltas": {}},
    "visxvis": {"title": "Vision x Vision (all tasks)", "row_labels": vis_labels, "col_labels": vis_labels, "deltas": {}},
    "llmxvis": {"title": "LLM x Vision (all tasks)", "row_labels": llm_labels, "col_labels": vis_labels, "deltas": {}},
}

for metric in METRICS:
    baseline_full = load_scores(metric, TOPK, False)
    nc_full = load_scores(metric, TOPK, True)

    expected_n = N_LLM + N_LVM
    assert baseline_full.shape == (expected_n, expected_n), (
        f"{metric}: expected {(expected_n, expected_n)}, got {baseline_full.shape} "
        "— check this run used --modality_x all --modality_y all"
    )
    assert nc_full.shape == baseline_full.shape

    groupings = {
        "llmxllm": {
            "title": "LLM x LLM",
            "row_labels": llm_labels, "col_labels": llm_labels,
            "baseline": baseline_full[:N_LLM, :N_LLM],
            "nc": nc_full[:N_LLM, :N_LLM],
        },
        "visxvis": {
            "title": "Vision x Vision (all tasks)",
            "row_labels": vis_labels, "col_labels": vis_labels,
            "baseline": baseline_full[N_LLM:, N_LLM:],
            "nc": nc_full[N_LLM:, N_LLM:],
        },
        "llmxvis": {
            "title": "LLM x Vision (all tasks)",
            "row_labels": llm_labels, "col_labels": vis_labels,
            "baseline": baseline_full[:N_LLM, N_LLM:],
            "nc": nc_full[:N_LLM, N_LLM:],
        },
    }

    for key, g in groupings.items():
        g["delta"] = g["baseline"] - g["nc"]
        save_full(g, metric, os.path.join(FIGURES_DIR, f"{metric}_{key}_full.png"))
        deltas_by_grouping[key]["deltas"][metric] = g["delta"]

for key, info in deltas_by_grouping.items():
    save_delta_across_metrics(
        key, info["title"], info["row_labels"], info["col_labels"], info["deltas"],
        os.path.join(FIGURES_DIR, f"{key}_delta_across_metrics.png"),
    )