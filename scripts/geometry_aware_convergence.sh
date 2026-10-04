#!/usr/bin/env bash
pip install -r requirements.txt
cd geometry_aware_convergence

FEATS=/workspace/hf
RESULTS=/workspace/results/emily
LLM=openlm-research_open_llama_3b
LVM=vit_base_patch14_dinov2.lvd142m

# "x_file|y_file|output_dir" per dataset
SETTINGS=(
    "$FEATS/wit_1024/${LLM}_pool-avg.pt|$FEATS/wit_1024/${LVM}_pool-cls.pt|$RESULTS/revision_1004/alignment_1024"
    "$FEATS/wit_1m/shards/$LLM/shard_0000.pt|$FEATS/wit_1m/shards/$LVM/shard_0000.pt|$RESULTS/revision_1004/alignment_10k"
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
            align "$x" "$y" "$out" --metric "$metric" "$flag" "$v" --later-layers
            align "$x" "$y" "$out" --metric "$metric" "$flag" "$v" --null-calibrate --later-layers
        done
    done
}

# sweep mutual_knn --topk 5 10 20 50 100 200 500 1000
sweep mutual_nd --dist 0.85 0.8 0.75 0.7 0.65 0.6

# sweep cknna_local --topk 5 10 20 50 100 200 500 1000
# sweep cknda_local --dist 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.9

# for metric in mutual_knn mutual_nd cknna_local cknda_local; do
#     python measure_alignment.py --modality_x language --modality_y vision --metric "$metric" --later-layers --output_dir "revision_1001/alignment"
#     python measure_alignment.py --modality_x language --modality_y vision --metric "$metric" --null-calibrate --later-layers --output_dir "revision_1001/alignment"
# done