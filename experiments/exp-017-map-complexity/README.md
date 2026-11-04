# exp-017: is the map between representations nonlinear? (H1 as a prediction test)

**Status:** done (run at `4ac66bb`, last-layer references at `0993e90`) · **Owner:** oddharak ·
**Outputs:** `/workspace/results/oddharak/exp-017-map-complexity/`, local copy
`results/exp-017-map-complexity/` (`real.jsonl`, `synthetic.jsonl`, `summary.md`, `*_pairs.json`)

## Idea

exp-011 / exp-013 showed that curvature and geodesics cannot be estimated at ID 20-40, so the
"wrong ruler" hypothesis (H1) was left untested. But H1 does not need curvature, only a map: if two
models share structure related by a smooth nonlinear warp that linear / Euclidean rulers miss, a
flexible regressor predicts one model from the other (on held-out samples) clearly better than a
linear one, and its predictions recover more of the target's neighbourhoods. If it does not, no
ruler that relies on a learnable smooth correspondence can find more agreement at this n.

## Method (`maps.py`, `analyze.py`)

- Data: exp-016 `d2l` features (WIT-1M subset, n = 10240, 7 layers per model), PRH preprocessing.
- Pairs: B3 LLM x ViT with Gemini and original captions (16 + 16); dual encoders' final embeddings
  (A, 3 + 3); B1 (one LLM, original x Gemini caption, 4); within-modality W_llm / W_vit (6 + 6,
  Gemini captions); random-init ViT x LLMs (D_random, 4); B3-Gemini with the pairing shuffled
  (D_shuffle, 16, one fold). Layer pair: the cell with the largest mutual kNN (k = 10) on all
  samples; dual encoders final x final. That rule picked input-level cells for the within-modality
  pairs (LLM layer 0 x 0, the embeddings of the same caption text; ViT block 4 x 10), so they were
  rerun at the last layer x last layer (W_llm_last, W_vit_last), the comparable reference for the
  last-layer B3 cells (LLM 20-24 x ViT 23).
- 5-fold CV, both directions. PCA to 256 dimensions on the training fold (input and target
  separately; it keeps 0.88-0.94 of LLM variance, 0.62-0.92 of ViT variance). Methods, tuned on an
  inner 10% validation split and refitted on the training fold: ridge; kNN regression; Nystroem RBF
  kernel ridge (2048 landmarks); a residual MLP (ridge map plus a 2 x 1024 GELU MLP whose last layer
  starts at zero; AdamW, early stopping).
- Held-out readouts in the target's PCA space: R^2, retrieval top-1 among the ~2048 test samples,
  mutual kNN (k = 10) between predicted and true targets. Nonlinear gain = best nonlinear method
  (by mean R^2, per pair and direction) minus ridge.

## Results

One RTX PRO 6000 pod, 2026-11-04, ~45 min including the last-layer rerun (~$2). Fits took 2-3 s
per fold. Medians over pairs (a2b / b2a):

### Synthetic controls (n = 10240, m latent coordinates, one seed)

| condition | ridge R^2 | best nonlinear R^2 | gain |
|---|---|---|---|
| null (nothing shared) | 0.00 | 0.00 | 0 |
| linear relation, m = 12 / 32 | 0.97 | 0.97 | 0.000-0.001 |
| A flat, B mildly curved (RFF), m = 12 | 0.51 / 0.96 | 0.91 / 0.97 | **+0.39** / +0.005 |
| same, m = 32 | 0.58 / 0.95 | 0.90 / 0.96 | **+0.32** / +0.005 |
| strongly wound (RFF), m = 12 | 0.04 / 0.39 | 0.23 / 0.50 | +0.19 / +0.10 |
| strongly wound, m = 32 | 0.01 / 0.06 | 0.01 / 0.06 | ~0 (nothing learnable) |
| mild curve + as much private latent, m = 12 / 32 | 0.27 / 0.32 | 0.35 / 0.33 | +0.08 / +0.01 |
| two random tanh networks, m = 12 / 32 | 0.77 / 0.71 | 0.85 / 0.71 | +0.08 / 0.00 |
| tanh + private, m = 12 / 32 | 0.35 / 0.33 | 0.35 / 0.33 | 0.00 |

- The test is specific: a linear relation and the null give no gain.
- It has power for a mild warp of a shared latent, even at m = 32 (+0.32).
- Its power collapses as the warp becomes high-dimensional: strong winding at m = 32, and any warp
  once half of the latent is private (the regime exp-016 arm C points to). A small gain on real
  data therefore rules out a mild, learnable warp, not every warp.

### Real pairs

| group | ridge R^2 | best nonlinear R^2 | gain R^2 | share of unexplained variance closed | mutual kNN ridge -> best | top-1 ridge -> best |
|---|---|---|---|---|---|---|
| D_shuffle (16) | 0.00 / 0.00 | 0.00 / 0.00 | 0.000 (0/32 folds won) | 0 | 0.005 -> 0.005 | 0 -> 0 |
| D_random: random ViT x LLM (4) | 0.19 / 0.31 | 0.22 / 0.33 | +0.03 / +0.03 | 0.04 | 0.04 -> 0.05 / 0.03 -> 0.03 | 0.01 -> 0.02 |
| B3_orig: LLM x ViT, WIT captions (16) | 0.23 / 0.19 | 0.26 / 0.21 | +0.02 / +0.02 | 0.03 | 0.14 -> 0.15 / 0.12 -> 0.13 | 0.07 -> 0.09 |
| **B3_gemini: LLM x ViT, Gemini captions (16)** | 0.47 / 0.64 | 0.51 / 0.70 | **+0.06 / +0.06** | 0.10 / 0.18 | 0.37 -> 0.39 / 0.39 -> 0.44 | 0.51 -> 0.59 / 0.55 -> 0.61 |
| A_gemini: dual encoders (3) | 0.36 / 0.42 | 0.41 / 0.47 | +0.04 / +0.04 | 0.07 | 0.32 -> 0.34 / 0.34 -> 0.36 | 0.57 -> 0.58 |
| B1: one LLM, two captions (4) | 0.46 / 0.27 | 0.50 / 0.29 | +0.03 / +0.01 | 0.06 / 0.02 | 0.18 -> 0.19 / 0.17 -> 0.18 | 0.19 -> 0.20 |
| W_llm_last: LLM x LLM, last layers (6) | 0.93 / 0.93 | 0.95 / 0.95 | +0.02 / +0.02 | 0.25 / 0.23 | 0.80 -> 0.82 / 0.81 -> 0.83 | 1.00 |
| **W_vit_last: ViT x ViT, last layers (6)** | 0.38 / 0.40 | 0.44 / 0.47 | **+0.07 / +0.07** | 0.13 / 0.12 | 0.33 -> 0.37 / 0.38 -> 0.41 | 0.60 -> 0.69 / 0.56 -> 0.68 |

(W_llm / W_vit at their mutual-kNN cells, input-level layers: ridge 0.92 / 0.85, gains +0.015 /
+0.035; full table in `results/exp-017-map-complexity/summary.md`.)

1. **Every non-shuffled pair has a small, systematic nonlinear component.** Kernel ridge (and in
   some pairs the MLP) beats ridge in every fold of every pair, and the shuffled pairing gives
   exactly zero. The gain is +0.015 to +0.07 R^2, i.e. 2-25% of the variance ridge leaves
   unexplained.
2. **It is not specific to crossing modalities.** LLM x ViT with Gemini captions gains +0.06 R^2
   and +0.02 to +0.05 mutual kNN; two ViTs with different training objectives gain as much or more
   (+0.07, +0.03 to +0.04 mutual kNN); a randomly initialised ViT against LLMs gains +0.03.
3. **The nonlinear map does not close the cross-modal gap in local structure.** After the best map,
   LLM x ViT neighbourhoods agree at 0.39-0.44 (from 0.37-0.39 linearly), against 0.80-0.83 for two
   LLMs with a *linear* map. The gain moves with shared content (Gemini vs WIT captions:
   +0.06 vs +0.02), as the linear level does.
4. **At the last layers, a ViT is not better predicted by another ViT than by an LLM reading a
   descriptive caption.** B3_gemini ridge R^2 0.47 / 0.64 and mutual kNN 0.37-0.39 against
   W_vit_last 0.38 / 0.40 and 0.33-0.38. Last-layer ViTs of different objectives (supervised
   IN21k, MAE, DINOv2, CLIP) differ from each other about as much as from LLMs; MAE pairs are the
   lowest, but DINOv2 x CLIP is also only 0.40 / 0.42 linearly.

## What this means

- **H1 in its testable form is not supported.** The relation between LLM and ViT representations
  is mostly linear (ridge recovers ~90% of the R^2 any method reaches). The nonlinear part is small,
  appears in every kind of pair at about the same size, and raises neighbourhood agreement by a few
  hundredths. There is no learnable "curved" correspondence that a geodesic or other nonlinear ruler
  could exploit to bring cross-modal agreement up to the within-modality level.
- **What is left of H1 is not identifiable.** Arm C showed that private variance is large; with
  half of the latent private, or a warp in ~32 dimensions, a planted nonlinear warp is invisible to
  every regressor at n = 10240. Any ruler estimated from these samples is blind to it in the same
  way, so it cannot explain measured disagreement either. This is the same conclusion as exp-011 /
  exp-013, reached through prediction instead of curvature.
- Together with exp-016, the cross-modal gap is best described as informational (unpredictable
  variance, which moves with caption content) plus a small, generic nonlinear residual, not as a
  geometric mismatch.

## Caveats

- R^2 and neighbourhoods are measured in each target's 256-dimensional PCA space (62-94% of the
  variance), not the full space.
- One layer pair per model pair (chosen on all samples by mutual kNN for B3, last x last for the
  references); the comparison between methods is within the same cell, so selection does not favour
  either.
- One seed, one MLP architecture and hyperparameter set; kernel ridge, which needs no
  architecture choice, gives the gain in almost all pairs.
- The synthetic controls use one seed per condition; their purpose is the sign and rough size of
  the power, not exact values.

## Commands

```bash
uv run python experiments/exp-017-map-complexity/maps.py synthetic --out results/exp-017-map-complexity
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp017.sh > /workspace/results/oddharak/exp-017.log 2>&1 < /dev/null &"
# last-layer references (added after the first run)
uv run python experiments/exp-017-map-complexity/maps.py real --features <exp-016 features> --out <dir> --groups W_llm_last W_vit_last
uv run python experiments/exp-017-map-complexity/analyze.py --root results/exp-017-map-complexity
```

## Run notes

On 2026-11-04 the first attempts found no GPU capacity in EU-RO-1; a later retry got an RTX PRO
6000 ($2.49/h). The job wrapper's self-termination was disabled by hand so the last-layer rerun
could finish on the same pod; the pod was then terminated manually (`13iqwap07l8r7n`, confirmed
gone).
