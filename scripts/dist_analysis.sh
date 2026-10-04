#!/usr/bin/env bash
pip install -r requirements.txt
cd geometry_aware_convergence

# LLM_MODELS=(
#     "bigscience/bloomz-560m"
#     "bigscience/bloomz-1b1"
#     "bigscience/bloomz-1b7"
#     "bigscience/bloomz-3b"
#     "bigscience/bloomz-7b1"
#     "openlm-research/open_llama_3b"
#     "openlm-research/open_llama_7b"
#     "openlm-research/open_llama_13b"
#     "huggyllama/llama-7b"
#     "huggyllama/llama-13b"
# )

# LVM_MODELS=(
#     "vit_tiny_patch16_224.augreg_in21k"
#     "vit_small_patch16_224.augreg_in21k"
#     "vit_base_patch16_224.augreg_in21k"
#     "vit_large_patch16_224.augreg_in21k"
#     "vit_base_patch16_224.mae"
#     "vit_large_patch16_224.mae"
#     "vit_huge_patch14_224.mae"
#     "vit_small_patch14_dinov2.lvd142m"
#     "vit_base_patch14_dinov2.lvd142m"
#     "vit_large_patch14_dinov2.lvd142m"
#     "vit_giant_patch14_dinov2.lvd142m"
#     "vit_base_patch16_clip_224.laion2b"
#     "vit_large_patch14_clip_224.laion2b"
#     "vit_huge_patch14_clip_224.laion2b"
#     "vit_base_patch16_clip_224.laion2b_ft_in12k"
#     "vit_large_patch14_clip_224.laion2b_ft_in12k"
#     "vit_huge_patch14_clip_224.laion2b_ft_in12k"
# )

# for model in "${LLM_MODELS[@]}"; do
#     python characterize_manifold.py --input_dir "/workspace/hf" --model_name "$model" --modality "language"
# done

# for model in "${LVM_MODELS[@]}"; do
#     python characterize_manifold.py --input_dir "/workspace/hf" --model_name "$model" --modality "vision"
# done

FEATS=/workspace/hf
RESULTS=/workspace/results/emily
LLM=openlm-research_open_llama_3b
LVM=vit_base_patch14_dinov2.lvd142m

python characterize_manifold.py --input_file "$FEATS/wit_1024/${LLM}_pool-avg.pt" --model_name "$LLM\_1024" --modality "language"
python characterize_manifold.py --input_file "$FEATS/wit_1m/shards/$LLM/shard_0000.pt" --model_name "$LLM\_10k"  --modality "language"
python characterize_manifold.py --input_file "$FEATS/wit_1024/${LVM}_pool-cls.pt" --model_name "$LVM\_1024"  --modality "vision"
python characterize_manifold.py --input_file "$FEATS/wit_1m/shards/$LVM/shard_0000.pt" --model_name "$LVM\_10k"  --modality "vision"