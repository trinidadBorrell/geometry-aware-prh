import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/workspace/results/emily/alignment"
FIGURES_DIR = "/workspace/results/emily/figures-0921"
TOPK = 10

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

# order-sensitive: check the more specific suffix before its substring
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


def plot_triplet(baseline, null_calib, row_labels, col_labels, title, save_path):
    delta = baseline - null_calib

    vmin = min(baseline.min(), null_calib.min())
    vmax = max(baseline.max(), null_calib.max())
    dmax = np.abs(delta).max()

    fig_w = max(9, 0.5 * len(col_labels) * 3)
    fig_h = max(3, 0.5 * len(row_labels))
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    panels = [
        (baseline, "Baseline", "viridis", vmin, vmax),
        (null_calib, "Null-calibrated", "viridis", vmin, vmax),
        (delta, "Delta (baseline - NC)", "RdBu_r", -dmax, dmax),
    ]

    for ax, (mat, subtitle, cmap, vlo, vhi) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vlo, vmax=vhi, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(subtitle, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


os.makedirs(FIGURES_DIR, exist_ok=True)

for metric in METRICS:
    # Step 1: load baseline and null-calibrated matrices
    baseline_full = load_scores(metric, TOPK, False)
    nc_full = load_scores(metric, TOPK, True)

    expected_n = N_LLM + N_LVM
    assert baseline_full.shape == (expected_n, expected_n), (
        f"{metric}: expected {(expected_n, expected_n)}, got {baseline_full.shape} "
        "— check this run used --modality_x all --modality_y all"
    )
    assert nc_full.shape == baseline_full.shape

    # Step 2: language-language
    b = baseline_full[:N_LLM, :N_LLM]
    nc = nc_full[:N_LLM, :N_LLM]
    labels = [short_label(m) for m in llm_models]
    plot_triplet(
        b, nc, labels, labels,
        title=f"{metric}: LLM x LLM",
        save_path=os.path.join(FIGURES_DIR, f"{metric}_llm_llm.png"),
    )

    # group vision models by task, in lvm_models order
    task_to_idx = {}
    for i, m in enumerate(lvm_models):
        task_to_idx.setdefault(task_for_model(m), []).append(i)

    # Step 3: vision-vision per task
    for task, idxs in task_to_idx.items():
        idxs_full = [N_LLM + i for i in idxs]
        b = baseline_full[np.ix_(idxs_full, idxs_full)]
        nc = nc_full[np.ix_(idxs_full, idxs_full)]
        labels = [short_label(lvm_models[i]) for i in idxs]
        plot_triplet(
            b, nc, labels, labels,
            title=f"{metric}: {task} vision x vision",
            save_path=os.path.join(FIGURES_DIR, f"{metric}_vision_{task}.png"),
        )

    # Step 4: language vs vision per task
    row_labels = [short_label(m) for m in llm_models]
    for task, idxs in task_to_idx.items():
        idxs_full = [N_LLM + i for i in idxs]
        b = baseline_full[np.ix_(range(N_LLM), idxs_full)]
        nc = nc_full[np.ix_(range(N_LLM), idxs_full)]
        col_labels = [short_label(lvm_models[i]) for i in idxs]
        plot_triplet(
            b, nc, row_labels, col_labels,
            title=f"{metric}: LLM x {task} vision",
            save_path=os.path.join(FIGURES_DIR, f"{metric}_lang_vision_{task}.png"),
        )