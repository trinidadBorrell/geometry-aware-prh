# Metric stability protocol (frozen before evaluation)

Scientific question: does learned anisotropic reweighting recover stable directions associated with a model, or does it mainly adapt to a particular training sample and comparison partner?

This document and `configs/metric_stability.json` are the contract for `results/metric_stability/`. They were written before any new exploratory-test scores from this study. Historical result directories listed in the config are not modified. The freeze tag `prh-release-alignment-freeze-20260921` is not moved.

## Audit

Reusable unchanged:

- Final-layer caches and `load_final_prepared`: exact 0.95 absolute-value clip, then per-row L2, on each full split matrix (`prh_prepare_layer`). Subsets do not recompute the clip.
- Split IDs in `coco_val2017_splits.json`. One caption per image (`coco-val2017-{image_id}`), so disjoint IDs are disjoint images. The 904-image `heldout` list has no final-layer features and is not used.
- `fit_pca` (economy SVD of train-mean-centred features). This study then fixes signs; it does not edit `fit_pca`.
- Contracted float64 scores: `make_pack`, `_scores_from_deltas`, `scores_numpy`. Objective is excess `a−b`. Analytic shuffle baseline of a frozen kernel is `b`.
- `metric_gram`, `metric_sq_distances`, `mnn_from_knn` (k=10).
- Distortion `D=||S||_F^2/q`, eigenvalue clip of `S` to `±log 4` (so eigenvalues of `B=exp(S)` stay in `[1/4, 4]`), then scale toward zero if the distortion budget is exceeded.
- `decompose_s`, `metric_diagnostics`, incumbent retention (higher train excess, then lower distortion).

Small extensions, new code only:

- Modes identity, A-only, B-only, separate, and shared-PC (one `S`, each model’s own 32-PC basis).
- Equal-total distortion, plus a labelled per-side budget only at full training size.
- `torch.matrix_exp` on the gradient path. Spectral projection uses `eigh` outside the graph. Identity starts are always scored.
- Canonical PCA signs, nested subset IDs, partner-mean fitting, shuffled correspondence fits, and a seeded basis rotation.
- Ambient `ΔM` comparisons via the `q×q` factor, without forming `d×d` matrices.

Not used for this objective, and not retuned:

- `eval_onesided` casts Grams to float32 before `extension_stats`. That path is the recorded ~2×10⁻⁵ gap against float64 contractions. This study scores in float64 and checks contracted scores against float64 direct Grams. Historical scores are not rewritten.
- `fit_one_sided` can lose the identity-start gradient because it differentiates through `eigh` at repeated eigenvalues. New fits do not call that routine. Repaired `S` matrices stay in their original unsigned-SVD coordinates and are not loaded as solutions.
- Ratio is reported and is not optimised.

Fitted on training rows only: mean, PCA basis, and `S`. Per example, before any split: clip and L2. Per gallery: Gram sample-centring inside CKA, and neighbour distances on `x−μ_train`. Validation and exploratory-test rows never enter `fit_anisotropic` or PCA. The manifest split named `test` is exploratory held-out evaluation; it has already been inspected in earlier experiments. It does not select initialisations, iterates, budgets, rank, optimiser settings, partners, subsets, or rotations.

## Design locked by the config

- Six base language models and three vision anchors from `release_models.json`. No supplementary instruction-tuned models.
- Primary grid: 18 vision–language pairs and three language–language pairs (Qwen2/OLMo-1, Qwen2.5/OLMo-2, Qwen3/OLMo-3); five modes; `ρ∈{0,0.1,0.4}` with equal total distortion. Side A is language on vision–language pairs and Qwen on language–language pairs.
- Shared-PC total distortion is `2||S||_F^2/q`. Separate mode uses `D_A+D_B≤ρ`.
- Secondary, full training size only: separate and shared-PC with 0.4 on each side (twice the total allowance of one-sided `ρ=0.4`).
- Sample-size panel: six language models against DINOv2-S, plus the three language–language pairs; A-only and separate; sizes 128, 256, 512, 1024, 2048; three nested sequences; budgets 0.1 and 0.4. At 2048 every sequence is the same full training set and is fit once. Fixed-basis and subset-basis protocols are separate. If numerical rank cannot support `q=32`, the case is recorded and not silently reduced.
- Partner transfer: one language metric against two vision partners, third partner excluded from fitting and from validation selection. Sizes 128, 512, 2048. Fixed basis. Same subset sequences.
- Controls, predetermined in the config: shuffled correspondence on three pairs; three basis rotations of side A at full size and `ρ=0.4`.
- Optimiser: full-batch Adam, 120 steps, learning rate 0.03, gradient clip 5, three starts (continuation when a smaller budget exists, identity, one perturbation). Best feasible iterate, including iteration zero, is kept. Monotone training-excess tolerance across nested budgets: 1×10⁻⁸. The training-only smoke showed a nonzero identity gradient, and that unnegated Adam steps decreased excess. Steps ascend excess by passing Adam the negative gradient, as in the repaired runner’s `(-excess).backward()`. The iteration cap was not changed.

The report’s closing sentence uses descriptive cutoffs chosen here, not significance tests: a correction is called stable only if the median replicate Frobenius cosine of ΔM at n=1024 is at least 0.9 for both basis protocols; subspaces are called stable if that fails but the median amplified overlap at rank 4 is at least 0.9. Otherwise the supported claim is about alignment scores. Gains smaller than 0.01 excess CKA are described as small.
