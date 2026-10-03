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

    
def compute_distances(feat_path, model, normalize=True):
    
    os.makedirs(args.output_dir, exist_ok=True)

    raw_x = torch.load(feat_path, map_location="cuda:0")["feats"]
    if isinstance(raw_x, torch.Tensor):
        x_feats = prepare_features(raw_x.float(), exact=False)
    else:
        x_feats = [prepare_features(layer.float(), exact=False) for layer in raw_x]

    pairwise_dists = []
    for l, x in enumerate(x_feats):
        if normalize:
            x_aligned = F.normalize(x, p=2, dim=-1)

        # TODO: More distance metrics!

        K = x_aligned @ x_aligned.T
        pairwise_dists.append(K)
    pairwise_dists = np.array(pairwise_dists)
    
    np.save(os.path.join(args.output_dir, f'{model}_dists.npy'), {
            "pairwise_dists": pairwise_dists,
    })


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
        feat_path = to_feature_filename(args.input_dir, args.modality, args.model_nae)

    compute_distances(feat_path, args.model_name)