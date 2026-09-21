import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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
    """(task, ViT variant) — e.g. 'augreg_in21k\\ntiny_p16'"""
    task = task_for_model(name)
    base = name.split(".")[0]  # e.g. vit_tiny_patch16_224
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


def draw_panel(ax, mat, row_labels, col_labels, cmap, vlo, vhi, title, label_fontsize):
    im = ax.imshow(mat, cmap=cmap, vmin=vlo, vmax=vhi, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=label_fontsize)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=label_fontsize)
    ax.set_title(title, fontsize=8)
    return im


os.makedirs(FIGURES_DIR, exist_ok=True)

llm_labels = [short_label(m) for m in llm_models]
vis_labels = [compact_lvm_label(m) for m in lvm_models]  # already task-contiguous in lvm_models order

for metric in METRICS:
    baseline_full = load_scores(metric, TOPK, False)
    nc_full = load_scores(metric, TOPK, True)

    expected_n = N_LLM + N_LVM
    assert baseline_full.shape == (expected_n, expected_n), (
        f"{metric}: expected {(expected_n, expected_n)}, got {baseline_full.shape} "
        "— check this run used --modality_x all --modality_y all"
    )
    assert nc_full.shape == baseline_full.shape

    groupings = [
        {
            "title": "LLM x LLM",
            "row_labels": llm_labels,
            "col_labels": llm_labels,
            "baseline": baseline_full[:N_LLM, :N_LLM],
            "nc": nc_full[:N_LLM, :N_LLM],
        },
        {
            "title": "Vision x Vision (all tasks)",
            "row_labels": vis_labels,
            "col_labels": vis_labels,
            "baseline": baseline_full[N_LLM:, N_LLM:],
            "nc": nc_full[N_LLM:, N_LLM:],
        },
        {
            "title": "LLM x Vision (all tasks)",
            "row_labels": llm_labels,
            "col_labels": vis_labels,
            "baseline": baseline_full[:N_LLM, N_LLM:],
            "nc": nc_full[:N_LLM, N_LLM:],
        },
    ]
    for g in groupings:
        g["delta"] = g["baseline"] - g["nc"]

    # --- normalize across all three groupings for this metric ---
    all_bn = np.concatenate([np.concatenate([g["baseline"].ravel(), g["nc"].ravel()]) for g in groupings])
    vmin, vmax = all_bn.min(), all_bn.max()

    all_deltas = np.concatenate([g["delta"].ravel() for g in groupings])
    dmax = np.abs(all_deltas).max()
    dmax = dmax if dmax > 0 else 1e-6

    # ============ Figure 1: 3x3 (baseline, NC, delta) x (llm-llm, vis-vis, llm-vis) ============
    height_ratios = [max(len(g["row_labels"]), 2) for g in groupings]
    max_cols = max(len(g["col_labels"]) for g in groupings)

    fig_h = sum(height_ratios) * CELL + 2.5
    fig_w = max_cols * 3 * CELL + 3.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer_gs = GridSpec(3, 1, height_ratios=height_ratios, hspace=1.0, figure=fig)

    bn_axes, d_axes = [], []
    last_im_bn, last_im_d = None, None

    for row_idx, g in enumerate(groupings):
        inner_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[row_idx], wspace=0.4)
        panels = [
            (g["baseline"], "Baseline", "viridis", vmin, vmax),
            (g["nc"], "Null-calibrated", "viridis", vmin, vmax),
            (g["delta"], "Delta (baseline - NC)", "RdBu_r", -dmax, dmax),
        ]
        for col_idx, (mat, subtitle, cmap, vlo, vhi) in enumerate(panels):
            ax = fig.add_subplot(inner_gs[col_idx])
            im = draw_panel(
                ax, mat, g["row_labels"], g["col_labels"], cmap, vlo, vhi,
                f"{g['title']}\n{subtitle}", label_fontsize=5,
            )
            if col_idx < 2:
                bn_axes.append(ax)
                last_im_bn = im
            else:
                d_axes.append(ax)
                last_im_d = im

    fig.suptitle(metric, fontsize=14)
    fig.colorbar(last_im_bn, ax=bn_axes, fraction=0.02, pad=0.02, label="score")
    fig.colorbar(last_im_d, ax=d_axes, fraction=0.04, pad=0.02, label="delta")
    fig.savefig(os.path.join(FIGURES_DIR, f"{metric}_full.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ Figure 2: 1x3, delta only ============
    col_counts = [len(g["col_labels"]) for g in groupings]
    row_counts = [len(g["row_labels"]) for g in groupings]

    fig_h2 = max(row_counts) * CELL + 2.0
    fig_w2 = sum(col_counts) * CELL + 2.5
    fig2 = plt.figure(figsize=(fig_w2, fig_h2))
    gs2 = GridSpec(1, 3, width_ratios=col_counts, wspace=0.5, figure=fig2)

    last_im_d2 = None
    for col_idx, g in enumerate(groupings):
        ax = fig2.add_subplot(gs2[col_idx])
        last_im_d2 = draw_panel(
            ax, g["delta"], g["row_labels"], g["col_labels"], "RdBu_r", -dmax, dmax,
            g["title"], label_fontsize=5,
        )

    fig2.suptitle(f"{metric} — delta only", fontsize=13)
    fig2.colorbar(last_im_d2, ax=fig2.axes, fraction=0.03, pad=0.02, label="delta")
    fig2.savefig(os.path.join(FIGURES_DIR, f"{metric}_delta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)