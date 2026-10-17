# exp-012: does ID mismatch predict misalignment? (WP1)

**Status:** running · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-012-id-mismatch/`

WP1 of the [proposal](../proposal-geometry-aware-prh.md).

## Idea

In exp-010 the LLM and ViT intrinsic dimensions move toward each other and meet at ~20-35 in late
layers, where cross-modal alignment usually peaks. If a mismatch in dimension limits alignment
(H2 in the proposal), alignment across layer pairs should fall as the ID gap grows, beyond what
depth alone explains.

## Hypothesis

- H2: calibrated alignment correlates negatively with |log ID_a - log ID_b| across layer pairs,
  after partialling out relative depth, in the confound-free tracks (T1 LLM x LLM, T2 ViT x ViT)
  as well as cross-modally (T3); and the best-aligned cell has a small ID gap.
- Falsified if the partial correlation is near zero (median |rho| < 0.1) in all tracks. A
  negative relation in T3 only points first at the pooling confound.

## Method

- Data: PRH `wit_1024`, 10 LLMs (mean-pooled caption, every hidden state) and 17 ViTs (CLS,
  every block); PRH features (q = 0.95 clamp, l2 norm). Pairs: T1 45, T2 136, T3 170.
- Alignment (`geoprh.layer_calibration`): mutual kNN (k = 10) and linear CKA on every layer
  pair, each cell calibrated against a 200-permutation null (the same permutation for all of
  B's layers; per-cell Algorithm 1 of Groger et al.). Tested cell by cell against Aristotelian's
  metrics and `calibrated_similarity.calibrate`.
- ID per layer from exp-010 (TwoNN, Levina-Bickel, PCA90; PRH features, n = 1024).
- Per pair: Spearman and partial Spearman (relative depths, their difference and product
  partialled out of the ranks) between the null-corrected score S - mu0 and three mismatches:
  |log ID ratio| (TwoNN, Levina-Bickel) and |log (PCA90/ID) ratio| (linear vs nonlinear
  structure); and the percentile of the best cell's ID gap among all cells.

## Commands

See exp-011 (both run on one pod); locally, on copied outputs:

```bash
uv run python experiments/exp-012-id-mismatch/analyze.py --root results/exp-012-id-mismatch --id-root results/exp-010-intrinsic-dim
```

## Results

Pending.
