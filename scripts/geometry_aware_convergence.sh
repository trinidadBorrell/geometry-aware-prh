#!/usr/bin/env bash
pip install -r requirements.txt
cd geometry_aware_convergence

LLAMA3B_PATH=/workspace/hf/prh_llms/wit_1024/openlm-research_open_llama_3b_pool-avg.pt
LLAMA7B_PATH=/workspace/hf/prh_llms/wit_1024/openlm-research_open_llama_7b_pool-avg.pt

VITBASE_PATH=/workspace/hf/prh_vlms/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt
VITLARGE_PATH=/workspace/hf/prh_vlms/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-cls.pt

LLAMA3B_PATH_BIG=/workspace/hf/wit_1m/shards/openlm-research_open_llama_3b/shard_0000.pt
VITBASE_PATH_BIG=/workspace/hf/wit_1m/shards/vit_base_patch14_dinov2.lvd142m/shard_0000.pt


LLAMA7B_PATH_BIG=/workspace/hf/wit_1m/shards/openlm-research_open_llama_7b/shard_0000.pt
VITLARGE_PATH_BIG=/workspace/hf/wit_1m/shards/vit_large_patch14_dinov2.lvd142m/shard_0000.pt

OUT_PATH="/workspace/results/emily/mutual_knn_all_k"
# OUT_PATH="/workspace/results/emily/nc_metrics_1025_streamline"

# python distance_metric_alignment.py --input_file_x "$LLAMA3B_PATH" --input_file_y "$VITBASE_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "standard" --n-evecs 20
# python distance_metric_alignment.py --input_file_x "$LLAMA7B_PATH" --input_file_y "$VITBASE_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "standard" --n-evecs 20

# python distance_metric_alignment.py --input_file_x "$LLAMA3B_PATH" --input_file_y "$VITLARGE_PATH" --output_dir "$OUT_PATH" --normalize  --gw_mode "standard" --n-evecs 20
# python distance_metric_alignment.py --input_file_x "$LLAMA7B_PATH" --input_file_y "$VITLARGE_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "standard" --n-evecs 20

# python distance_metric_alignment.py --input_file_x "$LLAMA3B_PATH_BIG" --input_file_y "$VITBASE_PATH_BIG" --output_dir "$OUT_PATH"


# python distance_metric_alignment.py --input_file_x "$LLAMA3B_PATH" --input_file_y "$LLAMA7B_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "standard"
# python distance_metric_alignment.py --input_file_x "$LLAMA3B_PATH" --input_file_y "$LLAMA7B_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "entropic"

# python distance_metric_alignment.py --input_file_x "$VITBASE_PATH" --input_file_y "$VITLARGE_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "standard"
# python distance_metric_alignment.py --input_file_x "$VITBASE_PATH" --input_file_y "$VITLARGE_PATH" --output_dir "$OUT_PATH" --normalize --gw_mode "entropic"

# python distance_metric_alignment.py --input_file_x "$LLAMA3B_PATH_BIG" --input_file_y "$VITBASE_PATH_BIG" --output_dir "$OUT_PATH" --normalize


# FEATS=/workspace/hf
# # RESULTS=/workspace/results/emily
# LLM=openlm-research_open_llama_3b
# LVM=vit_base_patch14_dinov2.lvd142m

# LLAMA3B=openlm-research_open_llama_3b
# LLAMA7B=openlm-research_open_llama_7b

# VITBASE=vit_base_patch14_dinov2.lvd142m
# VITLARGE=vit_large_patch14_dinov2.lvd142m

# # "x_file|y_file|output_dir" per dataset
SETTINGS=(
    "$LLAMA3B_PATH|$VITBASE_PATH|$OUT_PATH"
    "$LLAMA3B_PATH|$VITLARGE_PATH|$OUT_PATH"
    "$LLAMA7B_PATH|$VITBASE_PATH|$OUT_PATH"
    "$LLAMA7B_PATH|$VITLARGE_PATH|$OUT_PATH"
    # 10k pair: all-k + null on every layer pair is very expensive at this N; enable only deliberately
    # "$LLAMA3B_PATH_BIG|$VITBASE_PATH_BIG|$OUT_PATH"
)

align() {  # align X_FILE Y_FILE OUT_DIR [extra measure_alignment args]
    local x=$1 y=$2 out=$3; shift 3
    python measure_alignment.py --input_file_x "$x" --input_file_y "$y" --output_dir "$out" "$@"
}

sweep() {  # sweep METRIC FLAG VALUE...  (baseline + null-calibrated, both datasets)
    local metric=$1 flag=$2; shift 2
    for v in "$@"; do
        for setting in "${SETTINGS[@]}"; do
            IFS='|' read -r x y out <<< "$setting"
            align "$x" "$y" "$out" --metric "$metric" "$flag" "$v"
        done
    done
}

# mutual_knn at every k=1..N-1 in one call per model pair (no --topk sweep needed).
# Null: NUM_PERMS shuffles per layer pair; override with e.g. `NUM_PERMS=1000 bash run.sh`.
# NUM_PERMS=${NUM_PERMS:-200}
# for setting in "${SETTINGS[@]}"; do
#     IFS='|' read -r x y out <<< "$setting"
#     align "$x" "$y" "$out" --metric mutual_knn_all_k --num-permutations "$NUM_PERMS"
# done

# ---- 10k pair: k/N scaling ----
# One all-k call returns mKNN at every k=1..N-1 (so every k/N), for every layer pair. Rank matrices are
# computed once per call, so this costs about as much as 2-3 single-k runs and is cheaper than any
# multi-k sweep. Error bars from adjacent k, or any k/N grid, are then computed offline from the saved arrays.
# Cost warning: the null runs P shuffles on EVERY layer pair at N=10k. Time one run before launching.
NUM_PERMS_BIG=${NUM_PERMS_BIG:-200}
OUT_BIG="/workspace/results/emily/mutual_knn_all_k_10k"
align "$LLAMA7B_PATH_BIG" "$VITBASE_PATH_BIG" "$OUT_BIG"  --metric mutual_knn_all_k --num-permutations "$NUM_PERMS_BIG"
align "$LLAMA3B_PATH_BIG" "$VITLARGE_PATH_BIG" "$OUT_BIG"  --metric mutual_knn_all_k --num-permutations "$NUM_PERMS_BIG"
align "$LLAMA3B_PATH_BIG" "$VITLARGE_PATH_BIG" "$OUT_BIG"  --metric mutual_knn_all_k --num-permutations "$NUM_PERMS_BIG"



# Alternative (much more expensive for >3 k values): single-k runs at k matched in k/N to the 1024 runs.
# N_SMALL=1024; N_BIG=10000   # set N_BIG to the actual shard size
# for k in 5 10 20 50 100; do
#     k_big=$(( (k * N_BIG + N_SMALL / 2) / N_SMALL ))
#     align "$LLAMA3B_PATH_BIG" "$VITBASE_PATH_BIG" "$OUT_BIG" --metric mutual_knn --topk "$k_big" --num-permutations "$NUM_PERMS_BIG"
# done

# sweep mutual_knn --topk 5 10 20 50 100
# sweep mutual_nd --dist 0.995 0.99 0.98 0.95 0.9

# sweep cknna_local --topk 5 10 20 50 100 200 500 1000
# sweep cknda_local --dist 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.9

# for metric in mutual_knn mutual_nd cknna_local cknda_local; do
#     python measure_alignment.py --modality_x language --modality_y vision --metric "$metric" --later-layers --output_dir "revision_1001/alignment"
#     python measure_alignment.py --modality_x language --modality_y vision --metric "$metric" --null-calibrate --later-layers --output_dir "revision_1001/alignment"
# done