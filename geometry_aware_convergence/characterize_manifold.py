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

# Saves a dict with key = layer num, and N x N_K samples where each 1 of N_K rows is KS[i] nearest neighbor
def compute_distances(feat_path, model, normalize=True, q=0.95, row_chunk=16, device="cuda:0"):
    save_path = os.path.join(args.output_dir, f"{model.replace('/', '_')}_stats.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feats = torch.load(feat_path, map_location="cpu")["feats"]
    n, num_layers, _ = feats.shape

    row_qs = []
    for i in range(0, n, row_chunk):
        block = feats[i:i + row_chunk].to(device).float().abs().flatten(start_dim=1)
        row_qs.append(torch.quantile(block, q, dim=1))
        del block
    q_val = torch.cat(row_qs).mean()

    eye = torch.eye(n, dtype=torch.bool, device=device)

    KS = [1, 5, 10, 20, 50, 100] if '1024' in model else [10, 50, 100, 200, 500, 100]
    k_idx = [n - 1 - k for k in KS]

    out = {}
    for l in range(num_layers):
        x = feats[:, l, :].to(device).float().clamp(-q_val, q_val)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        K = x @ x.T
        s = K[~eye].view(n, n - 1).sort(dim=1).values
        del K, x

        out[l] = s[:, k_idx]

    arrays = {k: v.cpu().numpy() for k, v in out.items()}
    np.save(save_path, arrays, ks=KS)

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

    compute_distances(feat_path, args.model_name)