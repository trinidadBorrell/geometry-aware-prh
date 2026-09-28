#!/usr/bin/env bash
pip install -r requirements.txt
cd geometry_aware_convergence

FEATS=/workspace/hf
RESULTS=/workspace/results/emily

# "input_dir|output_dir" for each dataset
SETTINGS=(
    "$FEATS/wit_1024/|$RESULTS/alignment_1024"
    "$FEATS/wit_1m/shards/vit_base_patch14_dinov2.lvd142m/|$RESULTS/alignment_10k"
)

align() {  # align IN_DIR OUT_DIR [extra measure_alignment args]
    local in_dir=$1 out_dir=$2; shift 2
    python measure_alignment.py \
        --input_dir_x "$in_dir" --input_dir_y "$in_dir" \
        --output_dir "$out_dir" \
        --modality_x all --modality_y all "$@"
}

sweep() {  # sweep METRIC FLAG VALUE...  (baseline + null-calibrated, both datasets)
    local metric=$1 flag=$2; shift 2
    for v in "$@"; do
        for setting in "${SETTINGS[@]}"; do
            IFS='|' read -r in_dir out_dir <<< "$setting"
            align "$in_dir" "$out_dir" --metric "$metric" "$flag" "$v"
            align "$in_dir" "$out_dir" --metric "$metric" "$flag" "$v" --null-calibrate
        done
    done
}

sweep mutual_knn --topk 5 10 20 50 100
sweep cknda --dist 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.9

# for metric in cycle_knn mutual_knn cka unbiased_cka cknna; do
#     python measure_alignment.py --modality_x all --modality_y all --metric "$metric" --null-calibrate
# done