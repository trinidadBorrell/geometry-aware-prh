# exp-010: intrinsic dimension of the representations, per layer and vs sample size

**Status:** stage 1 done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-010-intrinsic-dim/`

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

## Next (stage 2)

- Fix n = 1024 (PRH `wit_1024` cache, all 27 models) and compare the ID profile over relative
  depth across LLM and ViT families, PRH geometry as primary.
- Check on the 8 models with 10240 samples that the model ordering at n = 1024 matches n = 10240.
