"""Fit ridge and ρ-CCA on synthetic paired features, then reload them.

Provenance: source A and partner B are stand-ins for two pre-block residual
streams. No checkpoint is loaded. The rows are independent, so the split is
``train_test_split``.
"""

from tempfile import TemporaryDirectory

import numpy as np

from rho_cca import (
    Endpoint,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RhoCCAConfig,
    RidgeConfig,
    fit_rho_cca,
    fit_ridge,
    load_fit,
    save_fit,
    split_rows,
)


def main() -> None:
    rng = np.random.default_rng(0)
    source = Endpoint("synthetic-a", "example", 1, "pre_block", "mean", 6)
    partner = Endpoint("synthetic-b", "example", 2, "pre_block", "mean", 4)
    ids = tuple(f"row-{i}" for i in range(32))
    groups = tuple(f"row-{i}" for i in range(32))
    x = rng.normal(size=(32, 6))
    y = x[:, :4] + 0.1 * rng.normal(size=(32, 4))
    bare = PairedDataset(
        FeatureTable(source, ids, groups, x),
        FeatureTable(partner, ids, groups, y),
    )
    data = PairedDataset(bare.source, bare.partner, split_rows(bare, test_size=0.25, validation_size=0.25, seed=0))
    pair = ModelPair(source, partner)
    ridge = fit_ridge(pair, data, config=RidgeConfig(t=1e-3))
    cca = fit_rho_cca(pair, data, config=RhoCCAConfig())
    with TemporaryDirectory() as directory:
        ridge_path = save_fit(ridge, f"{directory}/ridge.npz")
        cca_path = save_fit(cca, f"{directory}/cca.npz")
        loaded_ridge = load_fit(ridge_path)
        loaded_cca = load_fit(cca_path)
    train = data.select(data.split.train_ids)
    ridge_gap = np.max(np.abs(loaded_ridge.metric_coordinates(train.source) - ridge.metric_coordinates(train.source)))
    cca_gap = np.max(np.abs(loaded_cca.coordinates(train.source) - cca.coordinates(train.source)))
    print(f"ridge support {ridge.weight.shape}, reload max abs gap {ridge_gap:.3e}")
    print(f"rho-CCA components {cca.n_components}, reload max abs gap {cca_gap:.3e}")
    print(f"training correlations min {cca.rho.min():.3f} max {cca.rho.max():.3f}")


if __name__ == "__main__":
    main()
