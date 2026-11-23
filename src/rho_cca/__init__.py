"""Ridge metrics, correlation-weighted CCA, centred CKA, and steering transport.

The numerical conventions are taken from the audited angus-prh-replication
code named in each module docstring. Importing this package does not load a
model, a result directory, or an experiment runner.
"""

from rho_cca.cca import RhoCCAConfig, RhoCCAFit, fit_rho_cca
from rho_cca.cka import CKAScore, linear_cka, linear_cka_from_feature_grams, shared_neighbour_fraction
from rho_cca.extract import hashed_embedding
from rho_cca.io import load_fit, save_fit
from rho_cca.providers import ArrayProvider, MapExtractor, collect_pair
from rho_cca.records import (
    Endpoint,
    EndpointMismatch,
    FeatureTable,
    ModelPair,
    NonlinearTransportError,
    PairedDataset,
    RhoCCAError,
    Split,
)
from rho_cca.ridge import RidgeConfig, RidgeFit, fit_ridge
from rho_cca.splits import split_groups, split_rows, stratified_group_folds
from rho_cca.steering import (
    DifferenceInMeans,
    NormMatch,
    SteeringConfig,
    SteeringFit,
    attach_steering_target,
    fit_steering,
    match_norm,
)

__all__ = [
    "ArrayProvider",
    "CKAScore",
    "DifferenceInMeans",
    "Endpoint",
    "EndpointMismatch",
    "FeatureTable",
    "MapExtractor",
    "ModelPair",
    "NonlinearTransportError",
    "NormMatch",
    "PairedDataset",
    "RhoCCAConfig",
    "RhoCCAError",
    "RhoCCAFit",
    "RidgeConfig",
    "RidgeFit",
    "Split",
    "SteeringConfig",
    "SteeringFit",
    "attach_steering_target",
    "collect_pair",
    "fit_rho_cca",
    "fit_ridge",
    "fit_steering",
    "hashed_embedding",
    "linear_cka",
    "linear_cka_from_feature_grams",
    "load_fit",
    "match_norm",
    "save_fit",
    "shared_neighbour_fraction",
    "split_groups",
    "split_rows",
    "stratified_group_folds",
]

__version__ = "0.1.0"
