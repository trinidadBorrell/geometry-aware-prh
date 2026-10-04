# exp-010: intrinsic dimension of the representations, per layer and vs sample size

**Status:** stages 1-2, TwoNN check and PCA comparison done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-010-intrinsic-dim/`

## Idea

Characterise the intrinsic dimension (ID) of each layer's embedding with the Levina-Bickel MLE.
The estimator works on nearest-neighbour distances, so it depends on the number of samples n
(neighbourhoods get finer as n grows). Stage 1 measures that dependence on one small LLM,
for every layer. Stage 2 fixes n and a few layers and compares the ID of the LLMs and ViTs that
have enough samples cached.

## Hypothesis

- The ID profile over depth rises and then falls, peaking in the middle layers, as reported for
  LLMs (Valeriani et al. 2023; Cheng et al. 2025: a high-dimensional "abstraction" phase).
- The absolute ID changes with n (the MLE is biased low at finite n in high dimension), but if the
  *shape* of the layer profile is stable (high rank correlation with the largest n), a smaller n
  is enough to compare models.

## Method

- **Estimator:** Levina & Bickel 2004, Eqs. 8-9: per-point MLE m_k(x), averaged over points,
  then over k = 10..20. Computed with scikit-dimension's `skdim.id.MLE` (noise-free defaults,
  `comb="mean"` = the paper's mean; skdim's default `"mle"` is MacKay-Ghahramani's harmonic
  mean) on one sklearn kNN search per set; the average over k is done in
  `geoprh.intrinsic_dim`. Exact duplicate rows (identical captions) are dropped first, since a
  zero distance makes the log-ratio infinite; `n_unique` is recorded.
- **Data:** exp-006's cached activations of the first 10240 WIT-1M samples (LLM: every hidden
  state, mean-pooled over the caption; ViT: every block's CLS token).
- **Sample sets:** disjoint row blocks, 10 x 1000, 5 x 2000, 2 x 5000, 1 x 10000. The spread at
  each n is across sets.
- **Geometries:** `raw` (activations as cached) and `prh` (q = 0.95 outlier clamp + l2 norm, what
  the alignment metrics see).
- **Stage 1 model:** bigscience/bloomz-560m, all 25 layers.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp010.sh > /workspace/results/oddharak/exp-010.log 2>&1 < /dev/null &"
uv run python experiments/exp-010-intrinsic-dim/plot.py --root results/exp-010-intrinsic-dim
```

## Results (stage 1: bloomz-560m)

Run 2026-10-04, commit `93399a7`, RTX PRO 4000 pod used as a 12-vCPU machine (CPU pods were
unavailable), ~5 min. Outputs (`intrinsic_dim.jsonl`, `summary.txt`, figure, cache inventory) are in
`/workspace/results/oddharak/exp-010-intrinsic-dim/`, copied to `results/exp-010-intrinsic-dim/`.
No sample had an exact duplicate (`n_unique = n` everywhere).

ID (mean over sets) for selected layers:

| layer | raw 1k | raw 2k | raw 5k | raw 10k | raw 1k/10k | prh 1k | prh 2k | prh 5k | prh 10k | prh 1k/10k |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (embeddings) | 47.3 | 53.8 | 62.6 | 68.4 | 0.69 | 51.4 | 54.8 | 57.8 | 58.7 | 0.88 |
| 4 | 43.1 | 45.7 | 47.5 | 47.8 | 0.90 | 42.1 | 42.7 | 42.6 | 42.4 | 0.99 |
| 5 | 23.5 | 28.2 | 32.9 | 35.5 | 0.66 | 38.4 | 39.5 | 40.1 | 40.2 | 0.96 |
| 12 | 21.8 | 25.4 | 28.7 | 30.4 | 0.72 | 32.2 | 33.2 | 34.0 | 34.7 | 0.93 |
| 20 | 19.6 | 21.6 | 23.7 | 24.9 | 0.79 | 26.4 | 27.2 | 28.2 | 29.0 | 0.91 |
| 23 | 19.0 | 20.3 | 21.8 | 23.0 | 0.83 | 24.9 | 25.8 | 26.9 | 27.7 | 0.90 |
| 24 (last) | 13.7 | 14.9 | 16.4 | 17.4 | 0.78 | 34.9 | 36.0 | 37.4 | 38.4 | 0.91 |

**1. No middle-layer peak.** The ID is highest at the embedding layer (~50-70) and falls almost
monotonically with depth, to ~17 (raw) at the last layer. The hypothesised rise-and-fall is not
there for mean-pooled caption embeddings of this model. The studies that report a mid-depth peak
use per-token / last-token representations of longer text; mean pooling over a short caption
makes layer 0 a bag-of-words average, which is plausibly the most spread-out representation.

**2. The raw and PRH geometries disagree in two places.** Raw ID drops abruptly between layers 4
and 5 (43 -> 24 at n = 1k), and the last layer drops again. The PRH clamp + l2 norm removes the
first drop (42 -> 38, smooth decline) and turns the second into a jump up (25 -> 35). The pattern is
consistent with a few outlier dimensions of very large magnitude appearing at layer 5 (the
"massive activations" of LLMs) and dominating Euclidean distances until the final LayerNorm.
This is an inference, not checked here. Either way, the ID of the raw activations depends on
which coordinates dominate the norm, and the geometry the alignment metrics see is the PRH one.

**3. ID grows with n and has not converged at 10k, but the layer profile is fixed.**
- Raw: ID at 1k is 0.66-0.90 of the 10k value; it is still rising at 10k (e.g. layer 12:
  21.8 -> 30.4).
- PRH: much less sensitive, 0.88-0.99 of the 10k value; layers 4-5 are flat in n.
- The *shape* of the profile is the same at every n: Spearman of the layer profile against
  n = 10k is >= 0.99 for both geometries.
- Set-to-set variability is small next to the n effect: sd across the ten 1k sets is 0.2-0.6
  (2 at layer 0), against changes of 3-20 between n = 1k and 10k.

**Implication for stage 2.** Absolute ID is a function of n, so models must be compared at the
same n. At a fixed n the layer ordering is stable from n = 1000 on, and the PRH geometry is within
~10% of its 10k value. So n = 1024 is usable for comparisons, and it is the only size cached for
every model: the PRH `wit_1024` set covers all 10 LLMs and 17 ViTs, while 10240 samples exist only
for bloomz-560m/1b1/1b7 and five small ViTs (`inventory.txt`). One caveat: the MLE's downward
bias at finite n is larger for higher true ID, so cross-model differences at n = 1024 are
compressed, not reordered, as long as the ordering holds across n here.

Caveats: one model; mean pooling only; Levina-Bickel only (no TwoNN/GRIDE cross-check); the
per-k values (stored in `id_k`) fall with k (layer 0 raw at 10k: 74 at k = 10, 64 at k = 20),
so the k = 10..20 average is itself scale dependent.

## Results (stage 2: all 27 models at n = 1024)

Run 2026-10-04, commit `e752f73`, RTX PRO 4000 pod (CPU pods unavailable), ~25 min. Data: PRH
`wit_1024` (10 LLMs, 17 ViTs), one set. Plus a check on the 7 other wit1m10k models at n = 1000
(10 sets) and 10000. Outputs in `.../exp-010-intrinsic-dim/prh1024/` (`summary_models.txt`,
`fig2_profiles_{raw,prh}`, `fig3_family_means`); the check models were appended to the stage-1
`intrinsic_dim.jsonl`. Relative depth = layer / (L-1); ViT layer 0 is block 0 (no patch-embedding
layer in the cache).

**1. LLMs and ViTs have opposite ID profiles.** In the PRH geometry:
- LLMs: ID is highest at the embeddings (49-61) and falls with depth to ~27-30 around 80% depth,
  with a small rise before the last layer (LLaMA, OpenLLaMA) and a final value of 28-35. The three
  families nearly coincide; model size matters little (bloomz 560m -> 7b1 shifts the profile up by
  ~3).
- ViTs: ID is lowest early (7-11) and rises with depth to 20-44 at the last block. Families
  differ: supervised IN21k and CLIP-ft-IN12k rise steadily to 35-44; CLIP to ~25-30; MAE stays
  flat at ~20; DINOv2 is flat at ~15 until the last few blocks, then jumps to 24-39. Larger ViTs
  end higher (IN21k: tiny 21 -> large 44).
- So the two modalities move toward each other and meet in the ~20-35 range in their late layers,
  where cross-modal alignment is usually measured to peak. Whether that coincidence is meaningful
  is untested here.

**2. Raw vs PRH.** For LLMs the raw profile has an abrupt drop early (bloomz between 15% and 25%
depth, OpenLLaMA/LLaMA at ~10%), to ~20-25, where PRH declines smoothly, as in stage 1 for
bloomz-560m. ViT profiles are nearly the same in both geometries.

**3. The model ordering holds across n.** On the 8 models with 10240 samples, Spearman of the
model ordering at n = 1000 vs n = 10000, at relative depth 0 / 0.25 / 0.5 / 0.75 / 1:
PRH 0.98 / 0.98 / 0.98 / 1.00 / 1.00; raw 1.00 / 0.95 / 0.90 / 0.90 / 0.93. Comparing models at
n = 1024 is therefore sound for the PRH geometry, as the absolute values are biased low.

Caveats: one sample set at n = 1024 (stage 1 puts set-to-set sd at ~0.5-2); mean-pooled text vs
CLS-token images, so pooling differs between modalities; Levina-Bickel only.

## Results (TwoNN check)

Question: Valeriani et al. 2023 report an early/intermediate ID peak using TwoNN; is our
monotone LLM profile an artefact of Levina-Bickel? Run 2026-10-04, commit `a5839f6`, RTX PRO 4000,
~25 min, same data and sets as stages 1-2 (all 8 wit1m10k models at every n). Outputs in
`.../exp-010-intrinsic-dim/twonn/` (`fig1_id_by_layer`, `prh1024/fig4_estimators_{raw,prh}`).

**Estimator.** TwoNN as DADApy's `compute_id_2NN` (the implementation Valeriani et al. use):
mu = r2/r1, largest 10% dropped, least-squares slope through the origin of -log(1 - i/N) on
log mu. Computed with `skdim.id.TwoNN` (DADApy does not build on Windows); a test checks it
against DADApy's formula written out. DADApy's decimation (`data_fraction`, mean over 1/f random
subsets) is what our disjoint sets at each n already do.

**1. LLMs: same shape, no peak.** TwoNN gives the same decline from the embeddings
(PRH, n = 1024: 40-58 at layer 0, 22-32 at mid depth, 23-33 at the end), ~5-10 lower than
Levina-Bickel. The maximum is at relative depth <= 0.04 for all 10 LLMs. The estimator does not
explain the missing peak.

**2. ViTs: the estimator matters in late layers.** Levina-Bickel's late-layer rise (to 30-44
for IN21k, CLIP-ft, DINOv2) is mostly absent with TwoNN, which plateaus at ~15-25. The ViT maximum
moves into the middle for several models (CLIP-huge 0.39, DINOv2-base 0.36, IN21k-small 0.36,
DINOv2-giant 0.69); at n = 10000, IN21k-small peaks at block 4/11 and DINOv2-small at 6/11 with
a dip after. That is closer to Valeriani's iGPT shape. Levina-Bickel at k = 10..20 looks at larger
neighbourhoods than TwoNN (k = 2), so the late-layer ViT ID is scale dependent.

**3. n dependence.** TwoNN moves the other way from Levina-Bickel for LLMs: in the PRH geometry,
early-layer ID falls with n (bloomz-560m layer 0: 49 at 1k -> 43 at 10k) where Levina-Bickel
rose. The model ordering at 1k vs 10k is mostly stable (Spearman 0.90-1.00) except PRH at mid
depth (0.69).

**What likely explains the difference from the literature (not tested).** Valeriani et al. take
the input of each block after its first LayerNorm, on protein LMs and iGPT. Cheng et al. 2025,
who find the mid-depth peak in text LLMs, use the residual stream as we do but the **last token**
of 20-token sequences, GRIDE at a plateau scale (k ~ 32), 10k sequences. We mean-pool short
captions: layer 0 is then a bag-of-words average, which plausibly carries the highest ID, whereas
a single last token at layer 0 is one vocabulary embedding with low ID. Pooling is the most likely
cause; testing it needs a new extraction with last-token pooling.

## Results (linear vs nonlinear dimension: PCA)

Question: do the representations lie in a linear subspace of about the ID, or is the structure
nonlinear? Run 2026-10-04, commit `af77628`, RTX PRO 4000, ~45 min, same data and sets as before.
Outputs in `.../exp-010-intrinsic-dim/pca/` (`prh1024/fig4_estimators_{raw,prh}` puts all four
measures on one log axis per family; `prh1024/summary_models.txt` ends with the table below).

**Measures.** sklearn PCA spectrum (centred), dimensions from `skdim.id.lPCA`: the number of
components for 90% of the variance ("PCA90", the PC-ID of Ansuini et al. 2019) and the
participation ratio PR = (sum lambda)^2 / sum lambda^2. Reading: data on a flat m-dimensional
subspace gives PCA90 <= m, so PCA90 / ID ~ 1 (a test on a flat 5-d subspace gives 5 / 5).
PCA90 / ID >> 1 means the data is not flat at the scale of the dataset.

PRH features, n = 1024, family mean at relative depth 0 / 0.5 / 1:

| family | PCA90 | PR | TwoNN | PCA90 / TwoNN |
|---|---|---|---|---|
| BLOOMZ | 510 / 391 / 455 | 186 / 102 / 120 | 52 / 31 / 29 | 9.7 / 12.8 / 16.1 |
| OpenLLaMA | 565 / 517 / 542 | 173 / 59 / 117 | 45 / 27 / 24 | 12.7 / 19.0 / 22.7 |
| LLaMA | 534 / 560 / 500 | 144 / 163 / 44 | 40 / 22 / 25 | 13.3 / 25.0 / 20.0 |
| ViT IN21k | 7 / 76 / 294 | 5 / 24 / 161 | 12 / 19 / 23 | 0.6 / 3.8 / 12.3 |
| MAE | 6 / 96 / 143 | 4 / 19 / 26 | 8 / 18 / 20 | 0.8 / 5.0 / 7.0 |
| DINOv2 | 9 / 51 / 366 | 6 / 13 / 188 | 10 / 16 / 14 | 0.9 / 3.1 / 26.2 |
| CLIP ft-IN12k | 7 / 122 / 427 | 5 / 26 / 216 | 12 / 20 / 24 | 0.6 / 5.8 / 17.9 |
| CLIP | 7 / 174 / 350 | 6 / 28 / 144 | 12 / 21 / 20 | 0.5 / 8.1 / 17.1 |

**1. ViTs go from linear to nonlinear with depth.** In the first block PCA90 (4-15) is at or
below the ID: the CLS representations sit in a low-dimensional, essentially flat subspace. The
ratio then grows steadily, to 7 (MAE) - 26 (DINOv2) at the last block. MAE stays the most linear
(PR ~ ID ~ 20 over the second half of the network).

**2. LLMs are strongly nonlinear at every depth, including the embeddings.** PCA90 is 400-560
against an ID of 22-52 (ratio 10-25), and even PR, which is dominated by the top eigenvalues, is
2-7x the ID.

**3. Sample size: the ViT linear dimension is converged, the LLM one is not.** On the wit1m10k
models from n = 1000 to 10000, ViT PCA90 is unchanged in the first block (e.g. 4 -> 4, 10 -> 10)
and grows 10-40% in the last; bloomz PCA90 roughly doubles (bloomz-1b7 layer 0: 534 -> 1107; last
layer: 461 -> 988) while TwoNN stays flat. The LLM spectrum is heavy-tailed: no linear dimension
captures 90% of the variance at these n, so the LLM ratios above are lower bounds. PR moves much
less (+10-30%).

**4. Raw LLM activations: one direction carries the variance.** In the raw geometry PCA90 is
1-2 at mid depth for every LLM family (PR ~ 1): a single outlier direction holds > 90% of the
variance. This supports the massive-activation reading of the abrupt raw-ID drop in stages 1-2.
The PRH clamp + l2 norm removes it.

**What this says about linearity.** A flat subspace of about the ID is ruled out for every LLM
layer and for mid-to-late ViT layers: such a subspace would hold 90% of the variance in about ID
components, and 3-26x more are needed (lower bounds for LLMs). Early ViT blocks are consistent
with a near-linear representation. PCA vs ID cannot tell *which* nonlinearity: a curved manifold,
a union of separated clusters (each locally low-dimensional), or a heavy-tailed spread of
low-variance directions all give PCA90 >> ID. For PRH this matters directly: linear CKA is driven
by the global (PCA) structure and mutual kNN by the local one, and for LLMs these differ by an
order of magnitude in dimension at every layer.

## Next

- Separate curvature from clustering: local PCA on growing kNN neighbourhoods (on a curved
  manifold the local linear dimension is ~ID at small radius and grows with it), and graph-geodesic
  vs Euclidean distances. Scale-dependent ID (GRIDE) for the plateau.
- Last-token pooling for the LLMs, to test whether pooling explains the missing mid-depth peak.
- Relate the per-layer ID and PCA90 / ID of each model pair to their alignment (CKA, mutual kNN)
  at that layer pair.
