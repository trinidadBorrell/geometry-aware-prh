#!/usr/bin/env bash
pip install -r requirements.txt
cd geometry_aware_convergence

FEATS=/workspace/hf
RESULTS=/workspace/results/emily
LLM=openlm-research_open_llama_3b
LVM=vit_base_patch14_dinov2.lvd142m

# "x_file|y_file|output_dir" per dataset
SETTINGS=(
    "$FEATS/wit_1024/${LLM}_pool-avg.pt|$FEATS/wit_1024/${LVM}_pool-cls.pt|$RESULTS/alignment_1024"
    "$FEATS/wit_1m/shards/$LLM/shard_0000.pt|$FEATS/wit_1m/shards/$LVM/shard_0000.pt|$RESULTS/alignment_10k"
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
            align "$x" "$y" "$out" --metric "$metric" "$flag" "$v" --null-calibrate
        done
    done
}

sweep mutual_knn --topk 5 10 20 50 100
sweep mutual_nd --dist 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.9

# for metric in cycle_knn mutual_knn cka unbiased_cka cknna; do
#     python measure_alignment.py --modality_x all --modality_y all --metric "$metric" --null-calibrate
# done