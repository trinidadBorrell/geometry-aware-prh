import ot
import numpy as np
from pydiffmap import diffusion_map
from scipy.spatial.distance import pdist, squareform


import os
import argparse 
import glob

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from tasks import get_models
from pprint import pprint

from utils import *


def causal_inner_product(X, Y):
    pass

def isomap(X, Y):
    pass

def diffusion(X, n_evecs=3, k=64, alpha=0.5, t=1):
    def pairwise_dist(X):
        dist_sq = 2 * torch.sum(X**2) - 2 * X @ X.T
        dist_sq = torch.clip(dist_sq, min=0)
        return torch.sqrt(dist_sq)
    X = X.detach().double()
    D = pairwise_dist(X)
    sorted_vals = torch.sort(D, 1)[0]
    eps = torch.median(sorted_vals[:, k]) ** 2
    W = torch.exp(-D**2 / eps)
    q = W.sum(1)
    W = W / torch.outer(q**alpha, q**alpha)
    d = W.sum(1)
    S = W / torch.sqrt(torch.outer(d, d))
    evals, evecs = torch.linalg.eigh(S)
    idx = torch.argsort(-evals)[: n_evecs + 1]
    evals, evecs = evals[idx], evecs[:, idx]
    psi = evecs / torch.sqrt(d)[:, None]
    emb = psi[:, 1:] * evals[1:] ** t
    return pairwise_dist(emb)

def diffusion_cpu(X, n_evecs=3, k=64, alpha=0.5, t=1):
    X = X.detach().cpu().double().numpy()
    D = squareform(pdist(X))
    eps = np.median(np.sort(D, 1)[:, k]) ** 2
    W = np.exp(-D**2 / eps)
    q = W.sum(1)
    W = W / np.outer(q**alpha, q**alpha)
    d = W.sum(1)
    S = W / np.sqrt(np.outer(d, d))
    evals, evecs = np.linalg.eigh(S)
    idx = np.argsort(-evals)[: n_evecs + 1]
    evals, evecs = evals[idx], evecs[:, idx]
    psi = evecs / np.sqrt(d)[:, None]
    emb = psi[:, 1:] * evals[1:] ** t
    return squareform(pdist(emb))

def diffusion_fit_transform(X, dmap):
    X_transform = dmap.fit_transform(X.cpu())
    return squareform(pdist(X_transform))

def gromov_wasserstein(X, Y, dist="diffusion", n_permutations=200, quantile=0.95):
    if dist == "diffusion":
        # dmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=3, epsilon='bgh', alpha=0.5, k=64)
        # dist_metric = lambda x: diffusion(x, dmap)
        dist_metric = lambda x: diffusion(x)
    C_x = dist_metric(X)
    C_y = dist_metric(Y)

    T, log = ot.gromov.gromov_wasserstein(C_x, C_y, log=True)
    gw_dist = log['gw_dist']

    true_score = torch.mean(T.diagonal())

    null_scores = []

    for _ in range(n_permutations):
        randperm1 = torch.randperm(T.shape[0])
        randperm2 = torch.randperm(T.shape[1])

        T_shuffle = T[randperm1][randperm2]
        null_scores.append(torch.mean(T_shuffle.diagonal()))

    null_scores = torch.stack(null_scores)

    null_quantile = torch.quantile(null_scores, quantile)
    gated_score = max(0, true_score - null_quantile)
    return true_score, gated_score, gw_dist
    

def compute_score(x_feats, y_feats, normalize=True, num_permutations=200, quantile=0.95):
    if isinstance(x_feats, torch.Tensor):
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

    if isinstance(y_feats, torch.Tensor):
        y_feats = [y_feats[:, j, :] for j in range(y_feats.shape[1])]

    n_layers_x = len(x_feats)
    n_layers_y = len(y_feats)

    true_score_ret = np.zeros((n_layers_x, n_layers_y))
    nc_score_ret = np.zeros((n_layers_x, n_layers_y))
    gw_dist_ret = np.zeros((n_layers_x, n_layers_y))

    pbar = tqdm(total=n_layers_x*n_layers_y)

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y


            avg, nc, gw = gromov_wasserstein(x_aligned, y_aligned, n_permutations=num_permutations, quantile=quantile)
            true_score_ret[i, j] = avg
            nc_score_ret[i, j] = nc
            gw_dist_ret[i, j] = gw
            pbar.update(1)
    return {
        "true_score": true_score_ret,
        "nc_score": nc_score_ret,
        "gw_dist": gw_dist_ret
    }
    
def compute_alignment(x_fp, y_fp, num_permutations, quantile, precise=True):
    """
    Args:
        x_fp: x features
        y_fp: y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    os.makedirs(args.output_dir, exist_ok=True)

    raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]

    if isinstance(raw_x, torch.Tensor):
        x_feats = prepare_features(raw_x.float(), exact=precise)
    else:
        x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
    

    raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]

    if isinstance(raw_y, torch.Tensor):
        y_feats = prepare_features(raw_y.float(), exact=precise)
    else:
        y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]
    
    alignment_scores = compute_score(
        y_feats, x_feats,
        normalize=True,
        num_permutations=num_permutations,
        quantile=quantile
    )

    del x_feats, y_feats
    torch.cuda.empty_cache()

    return alignment_scores



def to_alignment_filename(output_dir, model_x, model_y):
    save_path = os.path.join(
        output_dir,
        f"{model_x}_{model_y}.npy"
    )
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--ot_metric",         type=str, default="gromov_wasserstein", choices=['gromov_wasserstein'])
    parser.add_argument("--dist_metric",         type=str, default="diffusion", choices=["diffusion"])

    parser.add_argument("--input_dir",      type=str, default="/workspace/hf")
    parser.add_argument("--input_file_x",      type=str, default=None)
    parser.add_argument("--input_file_y",      type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default="/workspace/results/emily/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--quantile", type=float, default=0.95)

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


    model_x_name = args.input_file_x.split('/')[-1][:-3]
    model_y_name = args.input_file_y.split('/')[-1][:-3]
    print('MODEL X', model_x_name, 'MODEL Y', model_y_name)

    save_path = to_alignment_filename(args.output_dir, model_x_name, model_y_name)
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()
    
    
    print(f"dataset:\t{args.dataset}")
    
    print('\nmeasuring alignment')
    alignment = compute_alignment(
        args.input_file_x, 
        args.input_file_y, 
        args.num_permutations,
        args.quantile, 
        args.precise
    )
    alignment["x_model"] = args.input_file_x
    alignment["y_model"] = args.input_file_y
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, alignment)
    print(f"saved to {save_path}")
    