#!/usr/bin/env bash
pip install -r requirements.txt
cd transferable_steering

# python run_steering.py --mode additive --graph mknn --out-dir /workspace/results/emily/additive_steering_transfer_mknn
python run_steering.py --mode replace --graph mknn --out-dir /workspace/results/emily/replacement_steering_transfer_mknn