# exp-010: intrinsic dimension of the representations, per layer and vs sample size

**Status:** running (stage 1) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-010-intrinsic-dim/`

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

## Results

Pending.
