pip install -r requirements.txt

cd geometry_aware_convergence

python extract_features.py --modality vision --config wit_1024 --output_dir /workspace/hf/wit_1024

python extract_features.py --modality language --config wit_1m --shard_id 0 --num_shards 100 --output_dir /workspace/hf/wit_1m/shards