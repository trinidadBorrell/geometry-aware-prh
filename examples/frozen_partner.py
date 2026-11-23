"""Fit an A–B metric on published ids and score it against native C.

C is a third basis. Its rows are joined by sample id and are not passed
through the A–B factor. The split below is an explicit published assignment,
not a fresh shuffle.
"""

import numpy as np

from rho_cca import (
    Endpoint,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RhoCCAConfig,
    RidgeConfig,
    Split,
    fit_rho_cca,
    fit_ridge,
    linear_cka,
)


def main() -> None:
    rng = np.random.default_rng(1)
    source = Endpoint("language-a", "example", 17, "pre_block", "mean", 5)
    donor = Endpoint("vision-b", "example", None, "pooled", "image", 5)
    third = Endpoint("vision-c", "example", None, "pooled", "image", 5)
    ids = tuple(f"image-{i:03d}" for i in range(30))
    groups = ids
    latent = rng.normal(size=(30, 5))
    x = latent + 0.05 * rng.normal(size=(30, 5))
    b = latent @ rng.normal(size=(5, 5))
    c = latent @ rng.normal(size=(5, 5))
    bare = PairedDataset(FeatureTable(source, ids, groups, x), FeatureTable(donor, ids, groups, b))
    split = Split(
        train_ids=ids[:18],
        validation_ids=ids[18:24],
        test_ids=ids[24:],
        group_by_sample=tuple((sample_id, sample_id) for sample_id in ids),
        seed=None,
        fingerprint=bare.fingerprint(),
        method="published",
    )
    data = PairedDataset(bare.source, bare.partner, split)
    pair = ModelPair(source, donor)
    ridge = fit_ridge(pair, data, config=RidgeConfig(t=1e-3))
    cca = fit_rho_cca(pair, data, config=RhoCCAConfig(beta=1.0, floor=1e-6))
    held = data.select(split.test_ids)
    native_full = FeatureTable(third, ids, groups, c)
    native = native_full.take(native_full.index(split.test_ids))
    ridge_score = ridge.cka_with_native(held.source, native)
    weighted = cca.cka_with_native(held.source, native, weights="reliability")
    equal = cca.cka_with_native(held.source, native, weights="equal")
    native_only = linear_cka(held.source.values, native.values)
    print(f"held-out native excess {native_only.excess:+.4f}")
    print(f"ridge metric excess {ridge_score.excess:+.4f}")
    print(f"rho-CCA weighted excess {weighted.excess:+.4f}")
    print(f"rho-CCA equal-weight excess {equal.excess:+.4f}")
    print("C was not a fitting input; donor training ids stay on B.")


if __name__ == "__main__":
    main()
