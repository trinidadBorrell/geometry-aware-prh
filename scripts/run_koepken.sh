pip install -r requirements.txt

cd geometry_aware_convergence

python extract_features.py --model openlm-research/open_llama_7b --config wit_1024 --output_dir /workspace/hf/wit_1024 --shard_id 0
python extract_features.py --model vit_large_patch14_dinov2.lvd142m --config wit_1024 --output_dir /workspace/hf/wit_1024 --shard_id 0

# python extract_features.py --modality language --config wit_1m --shard_id 0 --num_shards 100 --output_dir /workspace/hf/wit_1m/shards