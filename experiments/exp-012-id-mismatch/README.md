# exp-012: does ID mismatch predict misalignment? (WP1)

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-012-id-mismatch/`

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

Run 2026-10-17, commit `e91fabf`, RTX PRO 4500 pod, ~22 min for the 351 pairs (3-4 s per pair on
the GPU). Outputs in `/workspace/results/oddharak/exp-012-id-mismatch/` (`pairs/*.npz`,
`pairs.jsonl`, `analysis.jsonl`, `summary.txt`, fig1-fig3), copied to
`results/exp-012-id-mismatch/`. Check: the raw max mutual kNN of ViT-tiny x BLOOMZ-560m (0.087)
matches `prh_val_calibrated` (0.0867).

Median over pairs (partial = depth partialled out; best pct = percentile of the best cell's ID
gap among all cells, 0.5 = no relation):

| track | metric | rho, TwoNN | partial, TwoNN [q25, q75] | partial < 0 | partial, LB | best pct, TwoNN |
|---|---|---|---|---|---|---|
| T1 LLM x LLM (45) | mutual kNN | -0.65 | -0.36 [-0.54, -0.15] | 93% | -0.22 | 0.30 |
| T2 ViT x ViT (136) | mutual kNN | -0.40 | -0.26 [-0.41, -0.15] | 95% | -0.26 | 0.22 |
| T3 ViT x LLM (170) | mutual kNN | -0.76 | -0.27 [-0.44, -0.00] | 75% | -0.29 | 0.07 |
| T1 | linear CKA | -0.57 | -0.09 [-0.36, +0.13] | 64% | -0.02 | 0.19 |
| T2 | linear CKA | -0.41 | -0.32 [-0.48, -0.14] | 90% | -0.29 | 0.19 |
| T3 | linear CKA | -0.84 | -0.27 [-0.50, -0.02] | 78% | -0.18 | 0.11 |

Linearity mismatch |log (PCA90/ID) ratio|, partial, mutual kNN / CKA: T1 -0.18 / -0.17,
T2 -0.28 / -0.23, T3 +0.10 / +0.05.

**1. ID mismatch predicts lower alignment beyond depth, in every track.** The raw correlation is
strong (-0.4 to -0.84) but mostly depth; after partialling out depth, a moderate negative
relation remains (median -0.22 to -0.36 for mutual kNN, for both ID estimators), in 93-95% of
the confound-free T1 and T2 pairs. So it is not an artefact of the LLM pooling.

**2. Cross-modally, alignment peaks where IDs match.** In T3 the best-aligned cell has one of the
smallest ID gaps of its layer grid (median percentile 0.07 for mutual kNN, 0.04 with
Levina-Bickel IDs). This is the exp-010 observation (LLM and ViT IDs meet at 20-35 in late
layers, where alignment peaks) confirmed over all 170 pairs. Fig2 shows it for DINOv2-giant x
LLaMA-13B: both scores are highest in the late-layer block where the ID ratio is smallest.

**3. Global (CKA) alignment within LLMs is not tied to ID.** For T1 the CKA relation is near zero
(-0.09 TwoNN, -0.02 LB), while mutual kNN gives -0.36. Within LLMs, the local neighbourhood
structure tracks ID, the global linear structure does not.

**4. Linear-vs-nonlinear mismatch matters within modality, not across.** The PCA90/ID mismatch
is negative in T1 and T2 but slightly positive in T3: cross-modal alignment is no lower between
an LLM layer and a ViT layer that differ in how linear they are.

**How strong is this evidence for H2?** Moderate and correlational. ID mismatch explains part of
where alignment is high, but correlations of -0.3 leave most of the variation to other factors.
Two caveats:
- **Common cause.** Layers that compute similar things may have both similar ID and similar
  neighbourhoods, without the ID limiting alignment. The depth control removes only the part
  explained by position in the network.
- **Biased ID.** exp-011 shows the estimators give 0.7-0.9 of the true ID, less at higher ID,
  which compresses the log ratios but does not reorder them.
The WP5 invariant tests (GW lower bounds, trustworthiness/continuity) are needed to say whether
ID differences *bound* alignment rather than accompany it.
