"""Fit a source difference-in-means direction, decode it into B, then attach C.

The C decoder sees unlabelled bridge pairs only. Norm matching is applied
afterwards and does not refit the direction.
"""

import numpy as np

from rho_cca import (
    DifferenceInMeans,
    Endpoint,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RhoCCAConfig,
    SteeringConfig,
    attach_steering_target,
    fit_rho_cca,
    fit_steering,
    match_norm,
    split_rows,
)


def main() -> None:
    rng = np.random.default_rng(2)
    source = Endpoint("qwen-like", "example", 17, "pre_block", "positions", 6)
    partner = Endpoint("olmo-like", "example", 11, "pre_block", "positions", 6)
    target = Endpoint("moonlight-like", "example", 13, "pre_block", "positions", 5)
    ids = tuple(f"review-{i}" for i in range(36))
    groups = ids
    x = rng.normal(size=(36, 6))
    y = x + 0.05 * rng.normal(size=(36, 6))
    bare = PairedDataset(FeatureTable(source, ids, groups, x), FeatureTable(partner, ids, groups, y))
    data = PairedDataset(bare.source, bare.partner, split_rows(bare, test_size=0.25, validation_size=0.25, seed=2))
    pair = ModelPair(source, partner)
    alignment = fit_rho_cca(pair, data, config=RhoCCAConfig())
    objective = DifferenceInMeans(data.split.train_ids[:6], data.split.train_ids[6:12])
    steered = fit_steering(alignment, pair, objective, data, config=SteeringConfig(decoder="pseudoinverse"))
    bridge_x = y + 0.01 * rng.normal(size=y.shape)
    bridge_y = rng.normal(size=(36, 5)) + bridge_x[:, :5]
    bridge_bare = PairedDataset(
        FeatureTable(partner, ids, groups, bridge_x),
        FeatureTable(target, ids, groups, bridge_y),
    )
    bridge = PairedDataset(
        bridge_bare.source,
        bridge_bare.partner,
        split_rows(bridge_bare, test_size=0.25, validation_size=0.25, seed=3),
    )
    attached = attach_steering_target(
        steered,
        bridge,
        anchor="partner",
        target=target,
        config=SteeringConfig(decoder="ridge", t=1e-3),
    )
    applied = match_norm(attached.vector_for(target), 1.5)
    print(f"source norm {np.linalg.norm(steered.source_vector):.4f}")
    print(f"partner decoded norm {steered.raw_norm:.4f} weights_cancelled={steered.weights_cancelled}")
    print(f"attached raw norm {attached.raw_norm:.4f} applied norm {applied.applied_norm:.4f}")
    print(f"command unchanged {np.allclose(attached.command, steered.command)}")


if __name__ == "__main__":
    main()
