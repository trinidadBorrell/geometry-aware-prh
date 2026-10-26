import os
import argparse 
import glob

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models, READABLE_NAMES
from pprint import pprint

from utils import *

# Copied over from PRH directory.


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, dist=None, normalize=True, null_calibration=True, num_permutations=200, quantile=0.95, later_layers = False):
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
    best_alignment_score = None

    alignment_scores =  np.zeros((len(x_feats), len(y_feats)))

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            # if using later layers only, restrict to relative depth > 0.5
            if later_layers and i <= len(x_feats) // 2 or j <= len(y_feats) // 2:
                continue
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk
            if 'nd' in metric:
                assert dist is not None, 'dist must be defined'
                kwargs['cutoff'] = dist

            score = metrics.AlignmentMetrics.measure(metric, x_aligned, y_aligned, **kwargs)
            alignment_scores[i, j] = score

            if best_alignment_score is None or score > best_alignment_score:
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
        null_scores = metrics.null_calibrate(metric, x_aligned, y_aligned, topk=topk, dist=dist, num_permutations=num_permutations, quantile=quantile)['null_scores']
    else:
        null_scores = []
    return alignment_scores, null_scores

    
def compute_alignment(
        x_feat_paths, 
        y_feat_paths, 
        metric, 
        topk, 
        dist, 
        num_permutations, 
        quantile,  
        later_layers, 
        out_dir,
        null_calibration=True, 
        precise=True,
    ):
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

    a, b =get_models('val', 'all')
    models = a+b

    def parse_fname(fname):
        for model in models:
            if model.replace('/', '_') in fname:
                return READABLE_NAMES[model]
        assert False, f"model name {fname} is not supported!"

    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = (sorted(x_feat_paths) == sorted(y_feat_paths))
    if metric == "cycle_knn":
        symmetric_metric = False

    # alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    # alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    all_null = []

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

            # alignment scores: per-layer alignment scores
            alignment_scores, null_values = compute_score(
                y_feats, x_feats,
                metric=metric, topk=topk, dist=dist,
                null_calibration=null_calibration, 
                normalize=True,
                num_permutations=num_permutations,
                quantile=quantile,
                later_layers = later_layers
            )

            x_model_name = parse_fname(x_fp)
            y_model_name = parse_fname(y_fp)

            save_path = to_alignment_filename(
                out_dir,
                x_model_name,
                y_model_name,
                metric,
                topk, dist
            )

            np.save(save_path, {
                    "scores": alignment_scores,
                    # "indices": alignment_indices,
                    "x_model": x_fp, 
                    "y_model": y_fp
                })

            all_null.append(null_values)
            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()


    if null_calibration:
        np.save(
            os.path.join(out_dir, f'{metric}_{topk}_{dist}_null_values.npy'), {
                "values": np.stack(all_null).flatten()
            }
        )
    return alignment_scores#, alignment_indices



def to_alignment_filename(output_dir, model_x, model_y, metric, topk, dist):
    dist_flag = f'_d{dist}' if dist > 0 else ''
    save_path = os.path.join(
        output_dir,
        f"{model_x}_{model_y}_{metric}_k{topk}{dist_flag}.npy" if ('knn' in metric or 'nd' in metric) else f"{metric}.npy"
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
    parser.add_argument("--input_file_x",      type=str, default=None)
    parser.add_argument("--input_file_y",      type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default="/workspace/results/emily/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    # parser.add_argument("--null-calibrate", action="store_true")
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--quantile", type=float, default=0.95)

    parser.add_argument("--later-layers",  action="store_true")

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # save_path = to_alignment_filename(args.output_dir, args.metric, args.topk, args.dist)
    
    # if os.path.exists(save_path) and not args.force_remake:
    #     print(f"alignment already exists at {save_path}")
    #     exit()


    if args.input_file_x is not None:
        assert args.input_file_x is not None and args.input_file_y is not None, \
            "pass both --input_file_x and --input_file_y"
        models_x_paths = [args.input_file_x]
        models_y_paths = [args.input_file_y]
        models_x = [(args.input_file_x, args.modality_x)]
        models_y = [(args.input_file_y, args.modality_y)]
    else:
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
    if 'nd' in args.metric:
        print(f"dist:\t{args.dist}")

    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    compute_alignment(
        models_x_paths, 
        models_y_paths, 
        args.metric, 
        args.topk, 
        args.dist,
        args.num_permutations, 
        args.quantile, 
        args.later_layers, 
        args.output_dir
    )

    # np.save(save_path, {
    #     "scores": alignment_scores,
    #     "indices": alignment_indices,
    #     "x_models": [m for m, _ in models_x],
    #     "y_models": [m for m, _ in models_y],
    # })
    print(f"saved to {args.output_dir}")
    