#!/usr/bin/env bash
pip install -r requirements.txt
cd transferable_steering

python run_steering.py --mode additive --graph mknn --out-dir additive_steering_transfer_mknn
python run_steering.py --mode replacement --graph mknn --out-dir replacement_steering_transfer_mknn