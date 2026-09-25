import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
from pprint import pprint

from utils import *

# Copied over from PRH directory.


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, dist=None, normalize=True, null_calibration=False, num_permutations=200, quantile=0.95):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    if isinstance(x_feats, torch.Tensor):
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

    if isinstance(y_feats, torch.Tensor):
        y_feats = [y_feats[:, j, :] for j in range(y_feats.shape[1])]

    best_alignment_indices = None
    best_alignment_score = 0

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk
            if 'knd' in metric:
                assert dist is not None, 'dist must be defined'
                kwargs['cutoff'] = dist

            score = metrics.AlignmentMetrics.measure(metric, x_aligned, y_aligned, **kwargs)

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    if null_calibration:
        i, j = best_alignment_indices
        x, y = x_feats[i], y_feats[j]

        if normalize:
            x_aligned = F.normalize(x, p=2, dim=-1)
            y_aligned = F.normalize(y, p=2, dim=-1)
        else:
            x_aligned = x
            y_aligned = y
        best_alignment_score = metrics.null_calibrate(metric, x_aligned, y_aligned, topk=topk, dist=dist, num_permutations=num_permutations, quantile=quantile)['gated']
    return best_alignment_score, best_alignment_indices

    
def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, dist, null_calibration, num_permutations, quantile, precise=True):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
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

    symmetric_metric = (sorted(x_feat_paths) == sorted(y_feat_paths))
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
        if isinstance(raw_x, torch.Tensor):
            x_feats = prepare_features(raw_x.float(), exact=precise)
        else:
            x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
        
        # x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            
        for j, y_fp in enumerate(y_feat_paths):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue           

            raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]
            if isinstance(raw_y, torch.Tensor):
                y_feats = prepare_features(raw_y.float(), exact=precise)
            else:
                y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]
            best_score, best_indices = compute_score(
                y_feats, x_feats,
                metric=metric, topk=topk, dist=dist,
                null_calibration=null_calibration, 
                num_permutations=num_permutations,
                quantile=quantile
            )
            
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()

    return alignment_scores, alignment_indices



def to_alignment_filename(output_dir, metric, topk, dist, null_calibrate):
    if null_calibrate:
        metric += "_NC"
    dist_flag = f'_d{dist}' if dist > 0 else ''
    save_path = os.path.join(
        output_dir,
        f"{metric}_k{topk}{dist_flag}.npy" if 'knn' in metric else f"{metric}.npy"
    )
    return save_path

if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--modality_x",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--modality_y",     type=str, default="all", choices=["vision", "language", "all"])

    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)
    parser.add_argument("--dist",           type=float, default=0.0)

    parser.add_argument("--input_dir",      type=str, default="/workspace/hf")
    parser.add_argument("--output_dir",     type=str, default="/workspace/results/emily/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    parser.add_argument("--null-calibrate", action="store_true")
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--quantile", type=float, default=0.95)

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    save_path = to_alignment_filename(args.output_dir, args.metric, args.topk, args.dist, args.null_calibrate)
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()
    
    llm_models, lvm_models = get_models(args.modelset, modality='all')

    def _models_for(modality):
        if modality == "language":
            return [(m, "language") for m in llm_models]
        elif modality == "vision":
            return [(m, "vision") for m in lvm_models]
        return [(m, "language") for m in llm_models] + [(m, "vision") for m in lvm_models]

    models_x = _models_for(args.modality_x)
    models_y = _models_for(args.modality_y)

    models_x_paths = [to_feature_filename(args.input_dir, mod, m) for m, mod in models_x]
    models_y_paths = [to_feature_filename(args.input_dir, mod, m) for m, mod in models_y]
    
    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn
    
    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if 'knn' in args.metric:
        print(f"topk:\t{args.topk}")
    if 'knd' in args.metric:
        print(f"dist:\t{args.dist}")

    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')
    alignment_scores, alignment_indices = compute_alignment(models_x_paths, models_y_paths, args.metric, args.topk, args.dist, args.null_calibrate, args.num_permutations, args.quantile, args.precise)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {
        "scores": alignment_scores,
        "indices": alignment_indices,
        "x_models": [m for m, _ in models_x],
        "y_models": [m for m, _ in models_y],
    })
    print(f"saved to {save_path}")
    