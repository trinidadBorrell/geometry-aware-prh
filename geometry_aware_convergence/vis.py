import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

RESULTS_DIR = "/workspace/results/emily/alignment"
FIGURES_DIR = "/workspace/results/emily/figures-0921"
TOPK = 10

CELL = 0.32  # inches per matrix cell, drives figure sizing

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


def to_alignment_filename(metric, topk, null_calibrate):
    m = metric + "_NC" if null_calibrate else metric
    fname = f"{m}_k{topk}.npy" if "knn" in m else f"{m}.npy"
    return os.path.join(RESULTS_DIR, fname)


def load_scores(metric, topk, null_calibrate):
    path = to_alignment_filename(metric, topk, null_calibrate)
    data = np.load(path, allow_pickle=True).item()
    return data["scores"]


os.makedirs(FIGURES_DIR, exist_ok=True)

task_to_idx = {}
for i, m in enumerate(lvm_models):
    task_to_idx.setdefault(task_for_model(m), []).append(i)

for metric in METRICS:
    baseline_full = load_scores(metric, TOPK, False)
    nc_full = load_scores(metric, TOPK, True)

    expected_n = N_LLM + N_LVM
    assert baseline_full.shape == (expected_n, expected_n), (
        f"{metric}: expected {(expected_n, expected_n)}, got {baseline_full.shape} "
        "— check this run used --modality_x all --modality_y all"
    )
    assert nc_full.shape == baseline_full.shape

    # --- build the list of groupings for this metric: llm-llm, then vis-vis and llm-vis per task ---
    groupings = []

    llm_labels = [short_label(m) for m in llm_models]
    groupings.append({
        "title": "LLM x LLM",
        "row_labels": llm_labels,
        "col_labels": llm_labels,
        "baseline": baseline_full[:N_LLM, :N_LLM],
        "nc": nc_full[:N_LLM, :N_LLM],
    })

    for task, idxs in task_to_idx.items():
        idxs_full = [N_LLM + i for i in idxs]
        task_labels = [short_label(lvm_models[i]) for i in idxs]

        groupings.append({
            "title": f"Vision x Vision ({task})",
            "row_labels": task_labels,
            "col_labels": task_labels,
            "baseline": baseline_full[np.ix_(idxs_full, idxs_full)],
            "nc": nc_full[np.ix_(idxs_full, idxs_full)],
        })

        groupings.append({
            "title": f"LLM x Vision ({task})",
            "row_labels": llm_labels,
            "col_labels": task_labels,
            "baseline": baseline_full[np.ix_(range(N_LLM), idxs_full)],
            "nc": nc_full[np.ix_(range(N_LLM), idxs_full)],
        })

    # --- normalize across all groupings for this metric: one baseline/NC scale, one delta scale ---
    all_baseline_nc = np.concatenate(
        [np.concatenate([g["baseline"].ravel(), g["nc"].ravel()]) for g in groupings]
    )
    vmin, vmax = all_baseline_nc.min(), all_baseline_nc.max()

    all_deltas = np.concatenate([(g["baseline"] - g["nc"]).ravel() for g in groupings])
    dmax = np.abs(all_deltas).max()
    dmax = dmax if dmax > 0 else 1e-6  # guard against an all-zero delta collapsing the norm

    # --- layout: one row per grouping, 3 columns (baseline, NC, delta), row heights follow matrix size ---
    height_ratios = [max(len(g["row_labels"]), 2) for g in groupings]
    max_cols = max(len(g["col_labels"]) for g in groupings)

    fig_h = sum(height_ratios) * CELL + 2.0
    fig_w = max_cols * 3 * CELL + 2.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_gs = GridSpec(len(groupings), 1, height_ratios=height_ratios, hspace=0.9, figure=fig)

    baseline_nc_axes = []
    delta_axes = []
    last_im_bn, last_im_d = None, None

    for row_idx, g in enumerate(groupings):
        inner_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[row_idx], wspace=0.35)
        delta = g["baseline"] - g["nc"]
        panels = [
            (g["baseline"], "Baseline", "viridis", vmin, vmax),
            (g["nc"], "Null-calibrated", "viridis", vmin, vmax),
            (delta, "Delta (baseline - NC)", "RdBu_r", -dmax, dmax),
        ]

        for col_idx, (mat, subtitle, cmap, vlo, vhi) in enumerate(panels):
            ax = fig.add_subplot(inner_gs[col_idx])
            im = ax.imshow(mat, cmap=cmap, vmin=vlo, vmax=vhi, aspect="auto")
            ax.set_xticks(range(len(g["col_labels"])))
            ax.set_xticklabels(g["col_labels"], rotation=90, fontsize=6)
            ax.set_yticks(range(len(g["row_labels"])))
            ax.set_yticklabels(g["row_labels"], fontsize=6)
            ax.set_title(f"{g['title']}\n{subtitle}", fontsize=7)

            if col_idx < 2:
                baseline_nc_axes.append(ax)
                last_im_bn = im
            else:
                delta_axes.append(ax)
                last_im_d = im

    fig.suptitle(metric, fontsize=13)
    fig.colorbar(last_im_bn, ax=baseline_nc_axes, fraction=0.02, pad=0.02, label="score")
    fig.colorbar(last_im_d, ax=delta_axes, fraction=0.04, pad=0.02, label="delta")

    save_path = os.path.join(FIGURES_DIR, f"{metric}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)