import os
import argparse 
import glob

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
from pprint import pprint

from utils import *
KS = [1, 5, 10, 20, 50, 100]
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

def compute_distances(feat_path, model, normalize=True, q=0.95, row_chunk=16, device="cuda:0"):
    save_path = os.path.join(args.output_dir, f"{model.replace('/', '_')}_stats.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feats = torch.load(feat_path, map_location="cpu")["feats"]  # (N, L, D)
    n, num_layers, _ = feats.shape

    row_qs = []
    for i in range(0, n, row_chunk):
        block = feats[i:i + row_chunk].to(device).float().abs().flatten(start_dim=1)
        row_qs.append(torch.quantile(block, q, dim=1))
        del block
    q_val = torch.cat(row_qs).mean()

    eye = torch.eye(n, dtype=torch.bool, device=device)
    q_idx = [int(round(p * (n - 2))) for p in QS]
    k_idx = [n - 1 - k for k in KS]

    out = {k: [] for k in ["min", "max", "mean", "std", "quantiles", "knn_sims", "cutoff_counts"]}
    for l in range(num_layers):
        x = feats[:, l, :].to(device).float().clamp(-q_val, q_val)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        K = x @ x.T
        s = K[~eye].view(n, n - 1).sort(dim=1).values
        del K, x

        out["min"].append(s[:, 0])
        out["max"].append(s[:, -1])
        out["mean"].append(s.mean(dim=1))
        out["std"].append(s.std(dim=1))
        out["quantiles"].append(s[:, q_idx])
        out["knn_sims"].append(s[:, k_idx])
        out["cutoff_counts"].append(torch.stack([(s >= c).sum(dim=1) for c in CUTOFFS], dim=1))
        del s

    arrays = {k: torch.stack(v).cpu().numpy() for k, v in out.items()}
    np.savez_compressed(save_path, **arrays, qs=QS, ks=KS, cutoffs=CUTOFFS)

if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")
    parser.add_argument("--input_file",      type=str, default=None)
    parser.add_argument("--input_dir",      type=str, default=None)
    parser.add_argument("--modality",      type=str, default="language")
    parser.add_argument("--model_name",      type=str)
    
    parser.add_argument("--output_dir",     type=str, default="/workspace/results/emily/manifold_dists")
    

    args = parser.parse_args()

    assert (args.input_file is None) + (args.input_dir is None) == 1, "only 1 of input_dir or input_file can be specified"
    if args.input_file is not None:
        feat_path = args.input_file
    else:
        feat_path = to_feature_filename(args.input_dir, args.modality, args.model_name)

    compute_distances(feat_path, args.model_name, args.layer)