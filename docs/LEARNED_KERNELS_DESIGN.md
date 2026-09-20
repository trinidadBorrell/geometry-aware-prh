# Learned-kernel extension — frozen design

Machine-readable copy: `configs/learned_kernels.json`.

Operational PRH reference remains `platonic-rep@dcd76ba`. This experiment does not replace `results/prh_released_code`.

## Data

Existing COCO val2017 splits: train 2048 / val 1024 / test 1024. Leftover 904 images are not a 1024 confirmatory gallery, so **test evaluation is labelled exploratory** (the same gallery used in the PRH baseline).

Equal n=1024 on val and test. Train landscapes use n=2048; shuffled-CKA ratios are **not** compared raw across train vs eval without noting n.

## Representations

PRH clip q=0.95 then L2, all cached layers. Clip threshold is gallery-level (computed on the tensor being prepared), matching official `prepare_features`. New fitted quantities (layers, s, θ) use **train only**.

Each of the 18 V–L pairs: choose the layer pair by **maximum training-set extension linear CKA**. Freeze those layers for every kernel and objective. PRH test-gallery maxima stay in `results/prh_released_code` as a separate table.

## Kernels and objectives

See the JSON for the exact λ (25 values, 0.1–10 including 1) and α (9 values, 0.1–100) grids. Shared dimensionless λ (and α) act on r=||x−x′||/s; raw bandwidths are λ s_A and λ s_B.

Objectives, independently, from one shared candidate table: a, a/b, a−b with analytic unrestricted one-sided permutation mean b.

## What we will not do

No grid expansion from evaluation. No mNN tuning claims. No scaling-law language. No overwriting PRH baseline directories.
