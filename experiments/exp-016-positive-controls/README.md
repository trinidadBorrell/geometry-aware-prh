# exp-016: positive controls, ceilings and negative controls (WP8)

**Status:** running · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-016-positive-controls/` (features, geometry, pairs), local copy `results/exp-016-positive-controls/`

## Idea

exp-012 to exp-015 found weak cross-modal agreement between LLMs and ViTs (mutual kNN ~0.1 above
null, per-sample ORC agreement 0.10-0.16) against strong within-modality agreement (ORC 0.57-0.68),
and matched pooling did not close the gap. Read against the PRH this is only meaningful if the
same measures, on the same data, *can* show strong cross-modal agreement where we know it exists,
and stay at null where we know it does not. Without that, a low value could come from the data
(WIT captions describe little of the image), the measures, or the sample size.

## Arms

- **A (positive control):** contrastive dual encoders, trained to align text and image: OpenAI
  CLIP ViT-L/14, OpenCLIP ViT-H/14 (laion2b), SigLIP SO400M. Text tower x its own image tower,
  every layer and the final projected embedding, on D1 (PRH wit_1024) and D2 (WIT-1M subset,
  original and Gemini captions). A_cross: text tower of one model x image tower of another.
- **B (ceilings):** B1 one LLM, original x Gemini caption of the same image (same model,
  different text); B2 one ViT, image x augmented image (same model, different view); B3 LLM x ViT
  with Gemini instead of original captions (does a caption that describes the image raise it?).
- **C (dose-response, synthetic):** two random nonlinear encoders of latents with a controlled
  amount of shared content (`dose_response.py`): how each measure responds to shared content.
- **D (negative controls):** the sample pairing shuffled once (must be null); a randomly
  initialised ViT-B/16 against LLMs, ViTs and text towers (architecture / input statistics only).
- **X, W (references):** LLM x ViT and within-modality pairs on D1, as exp-012 / exp-014.

Primary readouts: (1) arm A's final-embedding pair; (2) the best layer pair under the
max-over-layers null; (3) the B3 difference (Gemini minus original) per LLM x ViT pair.

## Predictions (written before the runs)

| Outcome | Expectation | Interpretation |
|---|---|---|
| **A high:** dual-encoder final embeddings well above the LLM x ViT values (mutual kNN ~0.25-0.5 above null, ORC agreement ~0.3-0.5) | likely | The measures detect cross-modal alignment on WIT; the low LLM x ViT values reflect a real gap between non-contrastive models. |
| **A low** (like LLM x ViT, ~0.1-0.2) | unlikely; possible with WIT's captions | Ceiling problem: data or measures cannot show alignment even where it exists. No cross-modal claim from D1; move to D2's Gemini captions. |
| **A high on mutual kNN / CKA, low on ORC** | plausible | ORC agreement is not a sensitive cross-modal measure; keep it for within-modality comparisons only. |
| **B3 raises LLM x ViT** toward arm A | plausible for mutual kNN | Supports H4: shared content, not geometry, limits alignment. |
| **B3 raises arm A but not LLM x ViT** | possible | The models diverge beyond content: stronger evidence against PRH-style convergence. |
| **B1 / B2 ceilings** | B2 high (~0.6-0.9), B1 moderate (~0.3-0.5) | Cross-modal values are read as a fraction of the relevant ceiling, not of 1. |
| **C** monotone, null at no shared content | expected | Translates observed values into amounts of shared content; a non-monotone measure is dropped. |
| **D** random-init and shuffled near null | expected | Otherwise the measure picks up architecture or input statistics and needs a floor correction. |

## Method

- Data: D1 = `minhuh/prh` wit_1024; D2 = the exp-006 WIT-1M subset (first 10240 images with an
  original caption) plus each image's Gemini recaption (`prepare_d2.py`, urls checked row by row).
  Gemini captions are long (median ~3000 characters, ~650 tokens); CLIP / SigLIP text towers
  truncate them to 77 / 64 tokens, LLMs read up to 2048 tokens.
- Features (`extract.py`): dual encoders via open_clip, every block at the pooled position (EOT
  token for CLIP, last position for SigLIP; CLS for CLIP image towers, token mean for SigLIP,
  which has none) plus `encode_text` / `encode_image`; check: the hooked last block passed
  through each tower's own head reproduces `encode_text` / `encode_image` exactly (ViT-B-32, max
  abs difference 0). LLMs: every hidden state mean-pooled over the caption (platonic-rep loading,
  bf16, float32 accumulation; agrees with exp-015's extraction to 1.3% relative, bf16 noise).
  ViTs: timm CLS as exp-015. Augmentation: random resized crop (60-100% of the area, at the
  image's own size), colour jitter (0.4, 0.4, 0.4, 0.1), horizontal flip, one fixed view per
  image and seed, seeds 1-3. D2 is computed once at n = 10240 and saved as `d2s` (first 1024,
  all layers) and `d2l` (10240, 7 evenly spaced layers).
- Measures (`measure.py`), PRH preprocessing: mutual kNN (k = 10) and linear CKA per layer pair,
  each cell calibrated against a 200-permutation null of the pairing, and the max over cells
  against the null of the max (Algorithm 2); at n = 1024 also ORC given density and density
  agreement (k = 10, 30; exp-014 `layer_task` and `paired`), calibrated the same way;
  near-duplicate samples (nearest-neighbour distance < 1% of the median) left out of the ORC
  statistics; bootstrap 95% CI of the mutual-kNN excess at the best and final cells.
- Smoke test on fake features: correlated "towers" give mutual kNN 0.54 (null 0.01), the shuffled
  and random pairs stay at null, a planted duplicate is flagged and dropped.

## Commands

```bash
uv run python experiments/exp-016-positive-controls/extract.py --check
uv run python experiments/exp-016-positive-controls/dose_response.py --out results/exp-016-positive-controls/dose --workers 16
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=30 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp016.sh > /workspace/results/oddharak/exp-016.log 2>&1 < /dev/null &"
```

## Results

### Arm C: synthetic dose-response (local, `results/exp-016-positive-controls/dose/`)

Two random 3-layer tanh encoders (width 1024, noise sd 0.3; TwoNN ID 15-36, the range of the real
layers), n = 1024, 3 seeds. Best layer pair, mean over seeds (null 95% threshold: mutual kNN
~0.012, CKA 0.02-0.04, ORC / density 0.04-0.06):

| `count`: shared fraction of 12 coordinates | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|
| mutual kNN | 0.011 | 0.018 | 0.036 | 0.104 | 0.270 | 0.810 |
| linear CKA | 0.023 | 0.10 | 0.24 | 0.48 | 0.72 | 0.98 |
| ORC given density | +0.02 | +0.03 | +0.05 | +0.06 | +0.07 | +0.58 |
| density | +0.00 | +0.03 | +0.07 | +0.11 | +0.23 | +0.70 |

| `scale`: 12 shared + 12 private scaled by sigma | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|
| private variance fraction | 0 | 0.01 | 0.06 | 0.2 | 0.36 | 0.5 |
| mutual kNN | 0.83 | 0.82 | 0.74 | 0.46 | 0.24 | 0.11 |
| linear CKA | 0.99 | 0.99 | 0.98 | 0.90 | 0.73 | 0.49 |
| ORC given density | +0.60 | +0.59 | +0.45 | +0.25 | +0.08 | +0.07 |
| density | +0.74 | +0.68 | +0.55 | +0.38 | +0.19 | +0.13 |

- Every measure is monotone in shared content and at its null when nothing is shared (prediction
  met).
- Two encoders of *identical* content agree far from perfectly on local structure: mutual kNN
  ~0.8, ORC given density ~0.6 (the level of the within-modality pairs in exp-014 / exp-015).
- The local measures fall steeply with private content, CKA slowly: with half of the variance
  private (`scale` sigma = 1, or `count` rho = 0.5) mutual kNN is 0.1 and ORC agreement ~0.07,
  near null, while CKA is still ~0.5. ORC agreement is near null already at 36% private variance.
- So the measures do not translate into a single "shared fraction": the same synthetic pair reads
  very differently on CKA and on the local measures, and a weak local agreement (as LLM x ViT,
  ORC ~0.1-0.16, mutual kNN ~0.1 above null) is compatible with a large shared component plus
  substantial private variance. Low local agreement is evidence of private content, not of
  absent shared content.
