# exp-005: sweeping k - how mutual kNN and CKNNA meet CKA, and why CKNNA exceeds 1

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-005-cknna-large-k/`

## Idea

exp-004 found CKNNA at k=200 already CKA-like in magnitude. Here we sweep k over the whole range
(10 -> n-1, n = 1024) and overlay mutual kNN and CKA, to see how the local and global metrics
converge at the two ends. Separately, CKNNA scores above 1 were observed at large k (around
k~500 for MAE models), which an alignment score should not do.

## Hypothesis

CKNNA is `hsic(M*K, M*L) / sqrt(hsic(Mk*K, Mk*K) hsic(Ml*L, Ml*L))`, where Mk and Ml are each
model's own top-k mask and M = Mk*Ml is their intersection. The numerator and the denominators
use different masks, so Cauchy-Schwarz does not bound the ratio. Masking sets kernel entries to
0. When the kernel has a large mean (anisotropic features, cos ~ c for all pairs), those zeros
are the largest deviations in the matrix, and the centred "energy" of `M*K` is dominated by
c^2 x (variance of the mask pattern) rather than by the geometry. For random masks of density
p = k/(n-1) this gives CKNNA ~ p(1+p), which crosses 1 at p ~ 0.62. Prediction: scores above 1
follow the mask-only statistic and the permutation null, not alignment, and appear earliest for
the most anisotropic representations. MAE is the obvious candidate: its last-block CLS has a
mean cosine of ~0.9.

**What would falsify it:** CKNNA above 1 when the kernels are centred before masking, or a
permutation null that stays well below the observed score at the k where it crosses 1.

## Method

- `cknna_k_sweep.py` scores every layer pair at k in {10, 25, 50, 100, 200, ..., 900, 950, 1000,
  1024}. k=1024 means all off-diagonal entries (k = n-1), where CKNNA is unbiased CKA exactly.
  Metrics: mutual kNN(k), CKNNA(k) (platonic-rep, unbiased), linear CKA. Scores are the max over
  layer pairs, with platonic-rep preprocessing (q=0.95 clamp, l2 norm). The layer x layer terms
  are matrix products; `tests/test_cknna_sweep.py` checks all three metrics against
  `metrics.AlignmentMetrics`.
- `--diagnostics` adds three CKNNA variants that reuse its masks: *same-mask* (both denominators
  on the joint mask), *centred* (Grams HKH, HLH before masking) and *mask-only* (K = L = 1).
  It also adds a 10-permutation null of CKNNA at the best layer pair, and per-layer anisotropy
  (mean off-diagonal cosine).
- Grids (run 2026-09-27, CPU pod cpu3c x16):
  - *diagnostics*: 4 LLMs (bloomz-560m, bloomz-7b1, open_llama_13b, llama-13b) x 5 ViTs
    (MAE base/huge, IN21k large, DINOv2 large, CLIP huge);
  - *grid*: bloomz-560m and bloomz-1b1 x all 17 ViTs (+10 bloomz-1b7 pairs). This is the part of
    the full 170-pair grid that finished before it was cut to keep the run near 30 min;
  - *vision_vision*: last-block CLS of all 17 ViTs against each other (the PRH Fig. 12 setting,
    here with all 1024 images).
- Near-constant layers (std of off-diagonal cosines < 0.01, e.g. ViT block-0 CLS with mean
  cosine 0.999) are skipped when taking the max in the figures. For them the unbiased HSIC is a
  float32 cancellation of O(n^2) terms down to ~1e-7, so CKNNA/unbiased CKA near k=n is round-off:
  the same layer pair gave 1.91 on one machine and 0.27 on another, and upstream's own
  summation gives 0.03 (float64: 0.036).

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=8 nohup bash infra/runpod/run_exp005_k_sweep.sh > /workspace/results/oddharak/exp-005.log 2>&1 &"
# what was actually run (reduced for time):
uv run python experiments/exp-005-cknna-large-k/cknna_k_sweep.py --out $OUT/diagnostics --llms diag --lvms diag --diagnostics --workers 10
uv run python experiments/exp-005-cknna-large-k/cknna_k_sweep.py --out $OUT/vision_vision --vision-vision
uv run python experiments/exp-005-cknna-large-k/plot.py --root results/exp-005-cknna-large-k
```

## Results

Run 2026-09-27, commit `126c34c` (diagnostics, grid) / `8dfcb12` (vision_vision; no change to the
metric code in between), CPU pod cpu3c x16, ~35 min. Raw outputs are in
`/workspace/results/oddharak/exp-005-cknna-large-k/{diagnostics,grid,vision_vision}`, copied to
`results/exp-005-cknna-large-k/`. Figures come from `plot.py` (local only, regenerable).

Mean over the 20 diagnostic pairs (max over layer pairs):

| k | 10 | 200 | 400 | 500 | 600 | 700 | 800 | 900 | 1024 (= n-1) |
|---|---|---|---|---|---|---|---|---|---|
| mutual kNN | 0.12 | 0.34 | 0.50 | 0.57 | 0.65 | 0.72 | 0.80 | 0.89 | 1.00 |
| CKNNA (upstream) | 0.14 | 0.37 | 0.59 | 0.71 | 0.85 | **1.03** | **1.23** | **1.37** | 0.30 |
| CKNNA permutation null (mean) | 0.00 | 0.15 | 0.38 | 0.52 | 0.74 | 0.98 | 1.20 | 1.33 | 0.00 |
| mask-only CKNNA (K = L = 1) | 0.12 | 0.36 | 0.61 | 0.76 | 0.97 | 1.24 | 1.59 | 2.06 | - |
| random masks p(1+p) | 0.01 | 0.23 | 0.54 | 0.73 | 0.93 | 1.15 | 1.39 | 1.65 | - |
| CKNNA on centred Grams | 0.17 | 0.34 | 0.39 | 0.38 | 0.36 | 0.33 | 0.31 | 0.29 | 0.30 |

Linear CKA 0.38 and unbiased CKA 0.30. CKNNA at k = n-1 equals unbiased CKA exactly, which the
tests check.

**Convergence at the extremes (fig1).** At k=10, CKNNA (0.14) sits on mutual kNN (0.12). At
k = n-1 it is unbiased CKA (0.30), just below linear CKA (0.38). In between it does not
interpolate: it rises past both endpoints and crosses 1 at k = 700-800 in every one of the 20
pairs (and 44/44 grid pairs), peaks at k = 800-950 (up to 2.4 for open_llama_13b x CLIP-huge),
then falls to unbiased CKA over the last ~50 k. Mutual kNN just converges to 1, its chance level
k/(n-1). **CKNNA on centred Grams is the curve that actually bridges the two ends:** it starts at
the mKNN level, peaks around k = 300-500 (0.39), comes down to unbiased CKA at k = n-1, and never
exceeds 1 on any pair or k.

**Why CKNNA exceeds 1 (fig2).** From k ~ 700 on, observed CKNNA is within 0.05 of its permutation null (0.98
vs 1.03 at k=700, 1.33 vs 1.37 at k=900). Pairing images with the wrong captions gives the same
value above 1, so the score is not measuring alignment. It follows mask-only CKNNA, which uses no
kernel values at all, and the random-mask prediction p(1+p) (crosses 1 at p ~ 0.62, k ~ 630). The
mechanism: masking writes zeros into a kernel whose entries are all ~c > 0. After centring, those
zeros are the dominant deviations, so hsic(M*K, M*K) ~ c^2 x var(mask). The numerator's joint mask
M = Mk*Ml is sparser than either own mask, and at large k its zero fraction (1 - p^2) is closer to
1/2 than the own masks' (1 - p). So the numerator's "energy" exceeds the denominators', and the
ratio has no Cauchy-Schwarz bound. Centring the Grams before masking removes the c^2 term, and
the effect disappears.

**Above chance (fig3).** Both local metrics carry their largest above-chance signal at k ~ 100-400
(CKNNA minus null ~0.2-0.3, mKNN minus k/(n-1) ~0.1-0.2). By k ~ 700 CKNNA's excess over its null
has collapsed to ~0.05. The rank agreement of CKNNA with CKA across pairs drops from 0.97 (k=10)
to 0.44 (k=800); the centred variant keeps 0.71-0.95 at every k.

**MAE and k ~ 500 (fig5).** In the cross-modal max-over-layers setting MAE crosses 1 at k = 700-800
like every other family (fig4). The k ~ 500 observation comes from the vision-vision setting (PRH
Fig. 12, last-block CLS). MAE's last block is extremely anisotropic (mean cosine 0.89 vs 0.10 for
the other ViTs), and there CKNNA *is* the mask-only statistic: MAE x MAE CKNNA 0.99 vs mask-only 0.98
at k=500, and 1.29 vs 1.31 at k=800. All three MAE x MAE pairs cross 1 at k=600 here (n=1024).
The earlier `prh_val` vision-vision run (1000 images) reached 1.00 at k=500. The centred variant stays flat at ~0.8 and ends at unbiased CKA (0.82). For
the other ViTs, whose kernels are nearly centred already, CKNNA never averages above 1 (peak 0.91).

**Numerical caveat.** Separately, near k = n a near-constant layer (ViT block-0 CLS, cosine std
< 0.001) can produce any value, because its unbiased HSIC is float32 round-off (see Method). The
old `prh_val` cross-modal CKNNA values of 1.4-2.1 at k=1000 mix this with the mask effect. Several
of them come from ViT blocks 0-2, selected by the max over layer pairs.

Caveats: 20 diagnostic pairs plus 44 grid pairs, not the full 170; the null is 10 permutations
at the observed best layer pair, not the max-over-layers null of exp-003/004.

## Next

- Use centred-Gram CKNNA (mask HKH, HLH instead of K, L) wherever a k sweep is needed: it is
  bounded in practice, equals unbiased CKA at k = n-1 and keeps the model ranking at every k.
  Calibrate it with the exp-003 permutation machinery to see whether exp-004's "keeps the scaling
  trend" result survives.
- Report CKNNA at k <= ~400 only (n = 1024), or as excess over its permutation null.
- Compute unbiased HSIC in float64, or drop near-constant layers, before taking maxima over layers.
