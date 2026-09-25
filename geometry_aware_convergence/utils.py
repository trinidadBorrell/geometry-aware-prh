import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
from pprint import pprint

def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    if isinstance(feats, torch.Tensor):
        feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
        return feats.cuda()
    elif isinstance(feats, list):
        return [metrics.remove_outliers(f.float(), q=q, exact=exact).cuda() for f in feats]
    else:
        raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")


def to_feature_filename(input_dir, modality, model_name):
    save_name = f"{model_name.replace('/', '_')}"
    pooling = "_pool-avg" if modality == 'language' else '_pool-cls'


    save_name += pooling
    mode = 'prh_llms' if modality == 'language' else 'prh_vlms/prh'

    save_path = os.path.join(input_dir, mode, 'wit_1024', f"{save_name}.pt")
    return save_path
