#!/usr/bin/env bash
pip install -r requirements.txt
cd transferable_steering

python run_steering.py --mode additive --graph mknn --out-dir /workspace/results/emily/steering_pca/additive_steering_transfer_mknn_pca_11 --pca 11
python run_steering.py --mode additive --graph mknn --out-dir /workspace/results/emily/steering_pca/additive_steering_transfer_mknn_pca_32 --pca 32
python run_steering.py --mode additive --graph mknn --out-dir /workspace/results/emily/steering_pca/additive_steering_transfer_mknn_pca_64 --pca 64



# python run_steering.py --mode replace --graph mknn --out-dir /workspace/results/emily/steering/replacement_steering_transfer_mknn


# python run_steering.py --mode additive --graph ring --out-dir /workspace/results/emily/steering/additive_steering_transfer_ring
# python run_steering.py --mode replace --graph ring --out-dir /workspace/results/emily/steering/replacement_steering_transfer_ring