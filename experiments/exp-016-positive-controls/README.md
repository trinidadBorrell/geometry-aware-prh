# exp-016: positive controls, ceilings and negative controls (WP8)

**Status:** done (run at `4a46073`) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-016-positive-controls/` (features, geometry, pairs), local copy `results/exp-016-positive-controls/`

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
# on a 24 GB GPU (as run): WORKERS=12 LLM_TOKENS=2048 KEEP_CACHES=2 EXTRACT_TIMEOUT=180m BATCH=8, and
# llama-13b with: extract.py --only huggyllama/llama-13b --llm-tokens 2048 --llm-gpu-gib 14
uv run python experiments/exp-016-positive-controls/analyze.py --pairs results/exp-016-positive-controls/pairs.jsonl --out results/exp-016-positive-controls
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

### Arms A, B, D and references (`results/exp-016-positive-controls/summary.md`, `fig_groups`, `fig_b3`)

183 model pairs. Values are the excess over the shuffled-pairing null (score - mu0) at the best
layer pair, median [min, max] over the group's pairs. The best cell is a maximum over all layer
pairs, so it is compared with the null 95% threshold *of that maximum* (Algorithm 2): ~0.004 for
mutual kNN, 0.03-0.15 for CKA and ~0.11 for ORC given density at n = 1024. The shuffled pairs show
why that matters: their best ORC cell is +0.09 by selection alone, and none is significant.

| n = 1024 (D1 / D2 small) | mutual kNN | linear CKA | ORC given density | density |
|---|---|---|---|---|
| X: LLM x ViT, PRH captions (16) | 0.12 [0.07, 0.19] | 0.27 | 0.15 (15/16 sig.) | 0.29 |
| B3: LLM x ViT, D2 original captions (16) | 0.12 [0.07, 0.18] | 0.25 | 0.16 (14/16) | 0.25 |
| A: dual encoder, own towers, original captions (3) | 0.22-0.24 | 0.39 | 0.31-0.34 | 0.51-0.60 |
| A_cross: text tower of one, image tower of another (6) | 0.21 | 0.40 | 0.33 | 0.50 |
| A: dual encoder, Gemini captions (3) | 0.30 [0.19, 0.33] | 0.53 | 0.40 | 0.61 |
| **B3: LLM x ViT, Gemini captions (16)** | **0.34 [0.24, 0.41]** | **0.58** | **0.38** | **0.60** |
| B1: one LLM, original x Gemini caption (4) | 0.16 | 0.36 | 0.24 | 0.42 |
| B2: one ViT, image x augmented image (12) | 0.57 [0.49, 0.85] | 0.84 | 0.74 | 0.94 |
| W: LLM x LLM / ViT x ViT (6 / 6) | 0.51 / 0.49 | 0.71 / 0.84 | 0.58 / 0.59 | 0.83 / 0.88 |
| D_shuffle (23) | 0.003 (0/23) | 0.002 (0/23) | 0.09 (0/23) | 0.06 (1/23) |
| D_random: random ViT x text models (7 + 8) | 0.01-0.025 | 0.03-0.13 | 0.05-0.14 | ~0 |
| D_random: random ViT x pretrained ViTs (4) | 0.16-0.23 | 0.50-0.66 | 0.20-0.30 | 0.18-0.23 |

At n = 10240 (D2 large, 7 layers per model, no ORC) every mutual-kNN value is lower (the
neighbourhood k = 10 is a 10x smaller fraction of the data) and CKA is unchanged, but the order is
the same: LLM x ViT 0.066 with original captions and 0.233 [0.13, 0.28] with Gemini captions
(16/16 pairs higher, +0.10 to +0.19), against 0.145 / 0.195 for the dual encoders' final
embeddings with original / Gemini captions; CKA 0.29 -> 0.63 (dual encoders 0.45 / 0.57).

Per model:

- B3 with Gemini captions raises **all 16** LLM x ViT pairs on every measure (n = 1024: mutual kNN
  +0.17 to +0.25, CKA +0.22 to +0.46, ORC +0.13 to +0.34). With original captions the
  language-supervised ViT (CLIP ViT-H laion2b) aligns best with the LLMs (mutual kNN 0.17 vs
  0.08-0.14); with Gemini captions DINOv2 does (0.39 vs 0.26-0.35), and MAE has the highest ORC
  agreement (0.42).
- Dual encoders: the final embedding is within 0.01-0.04 mutual kNN of the best layer pair; the
  cross-model pairs (A_cross) are as aligned as each model's own towers, so the alignment is not
  specific to the jointly trained pair. SigLIP's text tower reads 64 tokens, so it sees only the
  start of a Gemini caption (median ~650 tokens); it is the one model that loses with Gemini
  captions (mutual kNN 0.21 -> 0.19, ORC 0.24 -> 0.16).
- B2 by ViT: DINOv2 0.85 (ORC 0.90), CLIP 0.62, AugReg 0.52, MAE 0.49; three augmentation seeds
  agree to 0.01.
- D_random: a randomly initialised ViT-B/16 against a pretrained ViT on the same images reaches
  mutual kNN 0.16-0.23 and CKA 0.50-0.66: pixel statistics alone produce that much of the
  ViT x ViT agreement (W_vit 0.49 / 0.84 is well above it). Against text models it gives a small,
  significant cross-modal floor (mutual kNN <= 0.025, CKA <= 0.15; ORC 0.05-0.14, around the
  null-max threshold of ~0.10, significant for 7 of 15).

### Predictions vs outcomes

| Prediction (written before the runs) | Outcome |
|---|---|
| **A high:** dual encoders ~0.25-0.5 above null, ORC agreement ~0.3-0.5 | **Partly.** With WIT's original captions CLIP / OpenCLIP / SigLIP reach mutual kNN 0.19-0.25 and ORC 0.30-0.39: about twice LLM x ViT, at the bottom of the predicted range. The measures detect cross-modal alignment, but WIT captions cap it even for contrastive models |
| A low (ceiling problem) | Not for the measures; for the data, partly (see B3) |
| A high on mutual kNN / CKA, low on ORC | **No.** ORC agreement follows mutual kNN in every arm (A 0.34, B3 Gemini 0.38, B2 0.74); it is a usable cross-modal measure |
| **B3 raises LLM x ViT toward arm A** | **Exceeded.** With Gemini captions LLM x ViT (0.34) reaches or passes the dual encoders (0.30 with the same captions, 0.22-0.24 with the original ones), on all measures and at n = 10240. Supports H4: shared content, not geometry, limits the alignment we measured in exp-012 to exp-015 |
| B3 raises arm A but not LLM x ViT | No |
| B2 high (~0.6-0.9), B1 moderate (~0.3-0.5) | B2 as predicted (0.49-0.85, model-dependent). B1 lower (0.16): an LLM's representations of the original and the Gemini caption of the same image agree *less* than its Gemini-caption representation agrees with a ViT's image representation (0.34) |
| C monotone, null at no shared content | Met (see Arm C) |
| D near null | Shuffled pairs: met. Random-init ViT: near null against text, but a large floor against ViTs (above) |

### Interpretation

- The weak cross-modal agreement of exp-012 to exp-015 (mutual kNN ~0.1, ORC agreement 0.10-0.16)
  is mostly a property of the data: WIT's reference captions (median 47 characters) share little
  content with their images. With captions that describe the image, independently trained LLMs and
  ViTs agree as much as contrastive dual encoders do, and about 60% of the image x augmented image
  ceiling (0.34 / 0.57), on neighbourhoods, CKA and per-sample ORC alike.
- So, on this evidence, there is no geometric incompatibility to explain: the same local-structure
  measures that show weak agreement on WIT captions show strong agreement on descriptive captions.
  "Platonic vs Aristotelian" readings of a single benchmark depend on how much the paired inputs
  share before any geometry is measured. Read with arm C, the remaining gap to the within-modality
  level (~0.5) is what modality-private variance alone produces on these measures.
- The PRH claim proper is a trend with capability; this experiment is about levels (WP9 is the
  trend test, now to be run on descriptive captions).

### Caveats

- One dataset (WIT images) and one recaptioner (Gemini). Gemini captions are model-generated and
  long (~650 tokens); part of their effect could be style shared with the LLMs' training data
  rather than image content. A second recaption source or COCO (WP11) separates the two.
- The best-layer-pair values are maxima over up to 41 x 41 cells; compare them with the null of
  the max (given above), not with zero. Mutual kNN at fixed k = 10 falls with n, so values at
  n = 1024 and n = 10240 are not comparable to each other, only within n.
- Random-init and dual-encoder arms use a single initialisation / checkpoint each.

### Run notes

One RTX PRO 4000 (24 GB GPU, 31 GB RAM cap), ~15 h at $0.57/h (~$9), about half of it lost to
failures:
host-RAM OOM when D1 images were decoded up front (fixed by lazy decoding, identical pixels),
volume quota (llama-13b read from the older complete HF cache, no download), llama-13b weights
filling the GPU (`--llm-gpu-gib`), DINOv2-g at 518 px (`BATCH=8`), and GPU-memory thrashing of
the permutation null at n = 10240 (B permuted one layer at a time; identical scores). Features
and per-pair arrays stay on the volume; `pairs.jsonl`, the logs and the figures are in
`results/exp-016-positive-controls/`.
