# Conventions

Rows are examples. A factor `F` with shape `(d, k)` maps a row `x` by `x @ F`.
The metric is `F @ F.T`. A decoder `G` has shape `(k, d_target)`, and a
displacement moves as `v_B = v_A @ F_A @ G_B`.

## Ridge

Centred source and partner supports `Xc`, `Yc` give

```text
(C_X + λ I) W = Xcᵀ Yc / n
λ = t · trace(C_X) / q
C_X = Xcᵀ Xc / n
```

`W` is `(q_source, q_partner)`. Metric coordinates are `(support - μ_source) @ W`.
Predictions add `μ_partner`. Those are different objects. With `t=None`, the
penalty minimises validation prediction MSE and ties keep the larger penalty.
Test rows are not a selection input. This is the row convention from
`objective_independent.py`, not the spectral column convention.

## ρ-CCA

Covariances divide by `n - 1`. Each side is whitened with `τ = η trace(C) / d`
and the whitened cross-covariance is decomposed by SVD. Weights come from the
Pearson correlations of the paired training canonical scores:

```text
w = mean_normalise(max(ρ², floor) ** β)
```

They are not the regularised singular values. `components=None` keeps every
finite coordinate in the prepared support. Equal-weight coordinates reuse the
same directions with scale 1. One-sided scoring transforms one fitted endpoint
and leaves the other matrix as it was supplied. A third basis C is never given
the source factor merely because its width matches.

Clipping and row normalisation are nonlinear. A metric learned after them is
constant in the prepared coordinates, not as the same matrix on raw activations.

## CKA

After column centring, `raw` is the usual linear CKA. `chance` is

```text
||X||_F² ||Y||_F² / ((n - 1) ||XᵀX||_F ||YᵀY||_F)
```

and `excess = raw - chance`. A zero-variance side raises. `chance` matches the
mean of `raw` over row permutations of one side on a fixed Gram. Orthogonal
mixes of columns, shared row permutations, and translations do not change the
score. A finite permutation sample is not expected to land on zero raw CKA.

`shared_neighbour_fraction` is the mean fraction of each row's k nearest
neighbours that also appear in the other row's neighbour list.

## Steering

`fit_steering` starts from a difference of source training means. It does not
subtract a fitted feature mean, clip the difference, or add a decoder
intercept. Pseudoinverse decoding and learned ridge decoding are separate
`decoder` settings. A ridge alignment transports by its regression factor; that
path is called `ridge`, not a pseudoinverse.

A shared positive diagonal inside a full-column-rank encode/decode pair
cancels. The fit records whether the weighted and equal-weight displacements
agreed. `match_norm` is a later rescaling. A vector below `near_zero_norm` is
replaced by zero rather than scaled up. `attach_steering_target` fits only the
new decoder from unlabelled anchor/target pairs. The source command stays put.

## Splits

`split_rows` calls `train_test_split`. `split_groups` calls
`GroupShuffleSplit`, whose sizes count groups. `stratified_group_folds` calls
`StratifiedGroupKFold`. Validation is a second split of the training subset.
A saved `Split` rejects a group that appears in two partitions. Fitting reads
the training ids only.

## What this package does not do

It does not ship datasets, model weights, or the historical galleries. It does
not implement multifamily consensus or MAXVAR. Inverse-metric steering is not
a hidden default. Hooks and generation stay optional; `preblock_rows` imports
PyTorch only when called.
