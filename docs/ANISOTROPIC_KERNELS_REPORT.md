# Anisotropic kernels — report

**Status:** complete on the existing 18 COCO vision–language pairs. Exploratory evaluation on the audited test gallery (n=1024). `results/prh_released_code`, `results/learned_kernels`, and `results/flex_kernels` were not modified.

**Question.** Does modest, learned anisotropic reweighting improve held-out alignment, and how much geometric distortion is required?

**Answer.** Yes, under the tested bounds: a full metric inside the leading 32 training PCs raises mean test CKA a from **0.508 (identity / linear)** to **0.798**, while diagonal reweighting of those PCs only reaches **0.581** (excess) or **0.522** (ratio). The gain needs the true pairing (shuffle-fit metrics score **worse** than identity on true test). It is **not** PCA truncation (truncation a=0.504). Distortion is large: eigenvalues often sit on `[1/4, 4]`, centred-Gram effective rank falls from **48 to ~9–16**, and mutual kNN at k=10 does **not** improve. A small aligned component is amplified; this is not broader neighbour agreement and not a model-size law.

## Lead answers

1. **Does anisotropy improve held-out alignment over identity?** Yes. Frozen-layer identity a=0.50766 matches the previous linear CKA on these layers. Excess-selected full metrics: a=0.798 (Δa=+0.290). Diagonal excess: a=0.581 (Δa=+0.073). Ratio-selected full: a=0.735 (Δa=+0.228). Ratio-selected diagonal: a=0.522 (Δa=+0.014).
2. **Does that survive validation and beat shuffled fitting?** Val and test track for full excess (val a=0.799, test 0.798). Shuffle-fit (3 pairs × 3 seeds, frozen PCA, same τ search): mean true-test Δa vs identity is **−0.23 to −0.27**. Shuffled-test Δa vs shuffled identity is ≈0. The improvement is correspondence-specific, not a generic reweighting artifact. Analytic b (permutations of a **fitted** kernel) is a different control.
3. **Is diagonal PCA reweighting enough?** No. Mixing directions inside the subspace (full B=exp(S)) more than triples the CKA gain relative to diagonal B.
4. **How much distortion?** Substantial, at the allowed cap. Full excess: mean R=1.59, 51% of learned eigenvalues at the ±log 4 bound, condition numbers using the bound ratio 16. Residual directions stay at eigenvalue 1. Uniform subspace boost relative to the residual is allowed (no trace(S)=0).
5. **Does improvement concentrate rank?** Yes. Identity r_eff=tr(Kc)²/‖Kc‖_F² = 47.8. Full ratio 9.3; full excess 12.7; diagonal ratio 8.8. Centred diagonal-energy fraction falls (0.048 → 0.014). **A small shared component is amplified.** A large J alone is not broad agreement (identity J=11.67 vs full ratio J=89.1 on n=1024; do not compare raw J to train n=2048).
6. **Does local mNN improve?** No. Identity metric-distance mNN k=10 = 0.1754, matching inner-product ranking on the unit-normalised features. Full excess mNN=0.1775; several pairs decrease. Independently maximised PRH mNN on **different** layers is 0.221 — not this frozen-layer comparator.
7. **Are the fitted metrics stable?** Conditional on frozen U: validation a is stable across three 682-point subsets. Induced val Grams have high Frobenius cosine (~0.89–0.97 for full excess). Language-side B matrices are only moderately similar (fro cosine ~0.76–0.85); principal angles of the four most-distorted eigenvectors are ~1.2 rad. **Matching spectra are not matching semantic directions.** No cross-model metric transfer.

## Compact test means (exploratory n=1024, 18 pairs)

| Kernel | a | a/b | a−b | Δa vs id | R | r_eff | mNN k=10 | kNN overlap vs id |
|---|---|---|---|---|---|---|---|---|
| Identity (full linear) | 0.508 | 11.67 | 0.455 | 0 | 0 | 47.8 | 0.175 | 1 |
| PCA truncation q=32 | 0.504 | — | — | −0.004 | — | — | — | — |
| Official RBF σ=1 (same layers) | ~0.522 | — | — | +0.014 | — | — | — | — |
| Previous RBF excess-selected | 0.512 | 10.07 | 0.455 | +0.004 | — | — | — | — |
| Diag, ratio-selected | 0.522 | 65.1 | 0.512 | +0.014 | 1.92 | 8.8 | 0.150 | 0.62 |
| Diag, excess-selected | 0.581 | 42.0 | 0.564 | +0.073 | 0.79 | 16.0 | 0.168 | 0.76 |
| Full, ratio-selected | 0.735 | 89.1 | 0.725 | +0.228 | 1.85 | 9.3 | 0.168 | 0.64 |
| Full, excess-selected | 0.798 | 70.7 | 0.784 | +0.290 | 1.59 | 12.7 | 0.178 | 0.68 |

τ selection (unpenalised val): full excess always τ=0; full ratio mostly 0.1 (12/18); diagonal ratio almost always τ=0 with **all** diagonal eigenvalues at the bound.

## Setup (frozen)

Config: `configs/anisotropic_kernels.json`. Layers from `results/learned_kernels/layers.json`. Splits from `data/manifests/coco_val2017_splits.json` (the 3 MB work-disk `coco_val2017.json` is truncated JSON; IDs match the existing feature caches).

- z=x−μ_train; U from train SVD, q=32 on every pair (never rank-reduced). Train PCA covers ~59% (language) / ~57% (vision) of variance; the residual stays at M=I.
- M=I+U(B−I)Uᵀ. Families: B=diag(exp(h)); B=exp(S). Eigenvalues of B in [1/4,4]. Identity is an explicit candidate.
- Primary J=a/b; secondary a−b. R=½(‖log B_A‖_F²/q_A+‖log B_B‖_F²/q_B). Train uses the specified normalised gain −τR. Three Adam starts; L² gradient clip; parameter clamp.
- Contractions on q×q matrices; scores checked against direct Grams in tests.
- mNN uses d_M²=(x−x′)ᵀM(x−x′), self-excluded. Identity recovers the original neighbour ranking.
- Shuffle pairs (chosen in config): bloomz-560m/dinov2-small, qwen3-0.6b-base/clip-laion-base, olmo-1b-0724/vit-in21k-small.

## Interpretation bounds

Held-out CKA **improves under bounded directional reweighting**, and **a small shared component is amplified**. Training improvement **does generalise** to val/test when correspondence is real, and **does not** when metrics are selected on shuffled train/val.

This does **not** show semantic identity, native geometric convergence, or a scaling law. Linear CKA is already invariant to rotations and coordinate permutations; the experiment only reweights the leading-variance subspace with a factor-of-four cap. A negative result in other directions is still possible. Neighbour overlap with identity ~0.65–0.76 means local order **does** change, but mutual kNN does not rise — do not sell CKA a as retrieval.

## Reproduction

```bash
sudo mount /dev/sdb1 /mnt/sdb1   # if needed
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
cd /mnt/sdb1/prh-replication-work/repo
/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests/test_anisotropic_kernels.py -q
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_anisotropic_kernels.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 40
```

Smoke: add `--smoke`. Outputs: `/mnt/sdb1/prh-replication-work/results/anisotropic_kernels/` and `results/anisotropic_kernels/` in git. Figures: `delta_a_*.png`, `distortion_*.png`, `rank_*.png`, `test_a_by_pair.png`.
