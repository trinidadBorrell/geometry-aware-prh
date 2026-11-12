# exp-027: does a latent-prediction ViT (I-JEPA) align better with LLMs than a pixel-reconstruction ViT (MAE)?

**Status:** done, 2026-11-12 · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-027-ijepa-vs-llm/` (copy in local `results/exp-027-ijepa-vs-llm/`), WIT-1024 activations in `/workspace/activations/facebook__ijepa_vith14_{1k,22k}/` · **Code:** `extract.py`, `score_d2.py`, `wit.py`, `analyze.py`, `infra/runpod/run_exp027.sh`, uncommitted, on `ef2baa9`

## Results (2026-11-12)

**Summary:** I-JEPA aligns better with LLMs than MAE, as predicted, on every LLM and both datasets. But I-JEPA aligns better with plain word counts by almost the same amount. Across all 19 mean-pooled ViTs, a ViT's alignment with word counts predicts its alignment with LLMs (R² = 0.94 for mutual 10-NN). The objective changes how much caption-nameable content a ViT keeps, mostly in its last blocks. Beyond that, I-JEPA is not special.

All numbers are excess over the null: the exact null mean on D2, and the mean of the null of the max on WIT-1024 (Gröger's Algorithm 2, 200 permutations; every pair p = 0.005). Tables: `summary.md`. Figures: `fig_wit.png`, `fig_floor_scatter.png`, `fig_d2.png`.

**Checks, all passed:**
- **I-JEPA:** 256 patch tokens, no prefix token; `layernorm(hidden_states[-1])` equals `last_hidden_state` exactly.
- **New code vs exp-015's cached mean-pooled WIT features:** min row cosine 0.99998 (MAE) and 1.000000 (DINOv2-g).
- **D2 scorer vs exp-016's stored bloomz-7b1 Gemini x DINOv2 grid:** max difference 7e-6 (CKA), 2e-5 (10-NN).
- **WIT calibration vs `prh_val_calibrated`:** raw score identical, null mean within 2e-6.

### Predictions

| | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | I-JEPA 1k > MAE, mutual 10-NN | WIT 10/10 LLMs (0.078 vs 0.067); D2 8/8 cells (Gemini 0.150 vs 0.113, original 0.040 vs 0.032) | holds |
| P2 | Same for CKA | WIT 10/10 (0.209 vs 0.194); D2 8/8 (Gemini 0.682 vs 0.619) | holds |
| P3 | I-JEPA 1k < DINOv2-g | DINOv2-g above for 10/10 LLMs (0.111 vs 0.078) | holds |
| P4 | I-JEPA 22k >= 1k | 1k above 22k for 10/10 LLMs on WIT (by 0.005); on D2 22k is higher at the best cell, lower at last x last | **fails** |
| P5 | I-JEPA's gain over MAE exceeds its gain in word floor | D2 Gemini k = 10: LLM gap +0.037, word-floor gap +0.034, TF-IDF floor gap +0.029 | holds, barely |

### 1. The objective's effect runs through the same channel as word counts

| WIT-1024 | LLMs, mutual 10-NN | Word floor, 10-NN | Above floor | LLMs, CKA | Word floor, CKA | Above floor |
|---|---:|---:|---:|---:|---:|---:|
| I-JEPA H/14, IN-1k | 0.078 | 0.035 | 0.043 | 0.209 | 0.044 | 0.165 |
| MAE H/14, IN-1k | 0.067 | 0.026 | 0.041 | 0.194 | 0.040 | 0.154 |
| DINOv2 g/14 | 0.111 | 0.047 | 0.064 | 0.259 | 0.057 | 0.202 |
| CLIP H/14 (LAION) | 0.144 | 0.054 | 0.090 | 0.277 | 0.057 | 0.220 |

- **Above the floor, I-JEPA and MAE nearly coincide:**
  - mutual 10-NN: +0.003 (positive for 10/10 LLMs, but small);
  - CKA: +0.010;
  - D2 Gemini CKA: the floor gap (+0.078) exceeds the LLM gap (+0.063).
- **Across the pool, the floor predicts the ranking** (`fig_floor_scatter.png`). Mutual 10-NN, all 19 mean-pooled ViTs:
  - Spearman 0.95; linear fit R² 0.94;
  - LLM ≈ 2.55 x floor − 0.007; the ratio LLM / floor spans 2.1-2.7;
  - residuals: I-JEPA −0.003, MAE +0.007, CLIP-H +0.012 (the largest, as expected for a text-supervised tower).
  - CKA is weaker: Spearman 0.69, R² 0.51.
- **What this says about PRH's vision side.** Across ViTs, alignment with LLMs is mostly how much word-describable structure a ViT encodes, scaled by a near-constant factor. This extends exp-023's observation that the TF-IDF floor rises with ViT size in every family.

### 2. Where it happens: MAE loses nameable content in its last blocks

Mutual 10-NN on D2, Gemini captions, mean over 4 LLMs (best LLM layer) / word floor, at each stored ViT depth (`fig_d2.png`):

| ViT | 0.33 | 0.50 | 0.67 | 0.83 | 1.00 |
|---|---:|---:|---:|---:|---:|
| I-JEPA 1k | 0.047 / 0.040 | 0.078 / 0.064 | 0.109 / 0.089 | **0.150** / 0.125 | 0.140 / 0.123 |
| MAE | 0.068 / 0.056 | **0.113** / 0.092 | 0.103 / 0.084 | 0.096 / 0.078 | 0.084 / 0.069 |
| DINOv2-g | 0.076 / 0.062 | 0.135 / 0.104 | 0.193 / 0.153 | 0.234 / 0.198 | **0.236** / 0.207 |

- **Different depth profiles.** MAE peaks at mid-depth and falls in the last blocks. I-JEPA and DINOv2 rise to the end or near it. At depth 0.5, MAE is above I-JEPA.
- **The same picture holds** on original captions and in CKA. It is consistent with reports that reconstruction-trained ViTs keep their most semantic features before the last block (Walmer et al., CVPR 2023; not checked further here).
- **The ratio is almost constant.** At every depth and for every ViT, LLM / floor is about 1.2 on Gemini captions (range 1.0-1.3) and 1.5-2.0 on original captions.

### 3. Other observations

- **I-JEPA 1k and 22k are nearly the same representation.** ViT x ViT on D2: mutual 10-NN 0.658, CKA 0.972, the highest pair by far (others 0.33-0.46 / 0.82-0.90). Ten times more pretraining data barely changed it, which fits P4 failing.
- **Pooling matters as much as the objective.** MAE's CLS token aligns better than its patch mean on D2 (Gemini k = 10: 0.141 vs 0.113). Against MAE CLS, I-JEPA's advantage shrinks to 0.150 vs 0.141. The PRH pool uses CLS, so the published MAE deficit is partly pooling.
- **I-JEPA's place in the pool.** Mean-pooled on WIT-1024, I-JEPA ranks below all DINOv2, CLIP and AugReg S/B/L models and above all MAE models, next to AugReg-tiny.
- **HF image processor check.** HF's processor (mean/std 0.5, no crop) gives 0.074 / 0.204 against 0.078 / 0.209 with MAE's transform. No conclusion changes.

### What this changes

- **For our lexical-floor line.** The floor is not only a language-side baseline. A ViT's alignment with word counts predicts its PRH alignment with LLMs almost perfectly at k = 10. Every vision-side PRH claim (size, objective, data) should be read against the ViT's own floor.
- **For Balestriero and LeCun's view argument.** At fixed architecture and data, the latent-prediction objective keeps more caption-nameable content in the last blocks than pixel reconstruction. It does not give a more language-like geometry beyond that content. Both are bad-view models by their definition; this run does not test good-view image JEPAs.

### Limits

- HF's I-JEPA weights are the context encoder, not the target encoder the paper evaluates.
- **Best-cell numbers.** Best cells take a maximum over layer pairs on the same samples; last x last is reported beside them on D2.
- **Small and shared samples:**
  - 10 LLMs from three families on WIT and 4 LLMs on D2;
  - pairs share models, so "10/10" is not 10 independent tests;
  - differences of 0.003-0.01 are within what one more model family could move.
- **The floor is itself a ViT measurement** (word counts against the ViT). A shared factor can drive both the floor and LLM alignment, so the regression says "co-varies", not "causes".
- Linear-probe quality differs between I-JEPA and MAE, so the objective is not isolated from capability.

### Run

- **Pod:** one L4 (24 GB, $0.59/h; RTX PRO 4500 and 4000 had no capacity), EU-RO-1, 10:35-14:03 UTC, about $2.05.
- **Attempt 1:** ran out of GPU memory on DINOv2-g at batch 64, because all 40 block outputs are kept (log: `exp-027.attempt1.log`). Its completed features were kept.
- **Attempt 2:** `BATCH=16 GROUP=1`, exit 0. DINOv2-g at batch 16 took 79 min on the L4.
- The pod continued with exp-028 after this job.

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
infra/runpod/push_code.sh <ip> <port>
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && BATCH=16 GROUP=1 nohup bash infra/runpod/run_exp027.sh > /workspace/results/oddharak/exp-027.log 2>&1 < /dev/null &"
uv run python experiments/exp-027-ijepa-vs-llm/analyze.py --out results/exp-027-ijepa-vs-llm
```

## Idea

Balestriero and LeCun (2610.39302) argue that what a joint-embedding model learns is set by its views. By their definition image JEPAs are "bad-view" models, because positives are corruptions of one photo. MAE is not a JEPA: it reconstructs pixels. exp-016 found that pixel statistics alone carry a lot of shared structure (F6: random ViT vs pretrained ViT, mutual kNN 0.16-0.23). exp-022 found MAE the lowest of the four ViTs at k = 10. The PRH pool has no JEPA.

I-JEPA ViT-H/14 and MAE ViT-H/14 were both trained on ImageNet-1k with the same architecture and patch size. The pair isolates the objective, latent-space prediction vs pixel reconstruction, at fixed architecture and data, which no PRH comparison can do. I-JEPA ViT-H/14 on ImageNet-22k adds a data-scale step at fixed architecture and objective.

## Models

| Key | Weights | Objective | Data | Pooling |
|---|---|---|---|---|
| I-JEPA 1k | `facebook/ijepa_vith14_1k` (HF `IJepaModel`) | latent prediction | IN-1k | mean of patch tokens (no CLS) |
| I-JEPA 22k | `facebook/ijepa_vith14_22k` | latent prediction | IN-22k | mean of patch tokens |
| MAE | `vit_huge_patch14_224.mae` (timm) | pixel reconstruction | IN-1k | mean of patch tokens, CLS as a reference |
| DINOv2-g | `vit_giant_patch14_dinov2.lvd142m` (timm) | self-distillation | LVD-142M | mean of patch tokens, CLS as a reference |

- **Layers:** every block output (HF `hidden_states[1:]`, timm `blocks.{i}.add_1`), before the final norm, as all cached ViTs.
- **Pooling:** I-JEPA has no CLS token, so every model in the comparison is mean-pooled over patch tokens, prefix tokens excluded (exp-015's recipe). Cached CLS features stay as a reference only.
- **Preprocessing:** I-JEPA is fed MAE's exact timm eval transform (bicubic resize, centre crop 224 at crop_pct 0.9, ImageNet mean/std). I-JEPA was trained with ImageNet normalisation (`facebookresearch/ijepa` `src/transforms.py`), but the HF repo ships a `ViTImageProcessor` with mean/std 0.5 and no crop. The HF processor is run on WIT-1024 as a check, not as the main result.
- **Weights:** the HF conversion took the checkpoint's context `encoder`, not the EMA `target_encoder` that the I-JEPA paper evaluates (`convert_ijepa_to_hf.py`). Stated as a limit.

## Data and comparisons

1. **WIT-1024 (PRH's setup):** 10 PRH LLMs (mean-pooled, cached) x I-JEPA 1k / 22k and all 17 PRH ViTs, all mean-pooled (exp-015's cache). PRH preprocessing (q = 0.95 clamp, l2), max over layer pairs, Gröger's Algorithm 2 with 200 permutations (`geoprh.prh_calibration`), mutual kNN (k = 10) and linear CKA. Word TF-IDF floors (with and without stopwords) against each ViT, calibrated the same way.
2. **D2 (n = 10240, original and Gemini captions):** the 4 exp-016 LLMs (bloomz-1b7, bloomz-7b1, open_llama_7b, llama-13b; 7 stored layers) x the four mean-pooled ViTs plus MAE and DINOv2 CLS. Mutual kNN at k = 10, 100, 1000 and linear CKA, excess over the exact null (exp-022), and floors: token TF-IDF (Pythia tokenizer, as exp-022) and word TF-IDF without stopwords. ViT x ViT pairs among the six for reference.
   - **Readouts:** best cell, and the fixed last x last cell (exp-022 V4: best cells are selected on the same samples).

## Predictions (written before the run)

| | Prediction | Falsified if |
|---|---|---|
| P1 (primary) | I-JEPA 1k > MAE (both mean-pooled) in calibrated mutual kNN against LLMs | I-JEPA is not above MAE for at least 8 of 10 LLMs on WIT-1024, or for at least 7 of the 8 D2 cells (4 LLMs x 2 caption sets) at k = 10 excess |
| P2 | Same direction for CKA | Same rule, CKA |
| P3 | I-JEPA 1k < DINOv2-g (mean-pooled) for most LLMs | I-JEPA above DINOv2 for 6 or more of 10 LLMs on WIT-1024 |
| P4 | I-JEPA 22k >= I-JEPA 1k | 22k below 1k for 6 or more of 10 LLMs (weak prediction) |
| P5 | The I-JEPA minus MAE gap in LLM alignment is larger than the gap in their word floors, so the ranking survives the floor | Floor gap >= LLM gap at k = 10 on D2 Gemini |

**Reading:**
- **P1 holds:** at fixed architecture and data, the objective matters for cross-modal alignment, in the direction "pixel detail is private variance".
- **P1 fails:** objective does not order alignment at fixed data, and MAE's low PRH scores need another explanation.

**Not separable in this design:** I-JEPA also reports higher ImageNet linear-probe accuracy than MAE at ViT-H/14 (Assran et al. 2023). A P1 win is consistent with "better model, better aligned" as much as with "latent target, less pixel nuisance".

## Checks (the job stops if one fails)

- I-JEPA: `layernorm(hidden_states[-1])` equals `last_hidden_state`, and the token count is 256 patches (no prefix token).
- MAE mean-pooled WIT-1024 features from this code vs exp-015's cached ones: min row cosine >= 0.999 at every layer.
- D2 scorer: bloomz-7b1 Gemini x DINOv2 CLS reproduces exp-016's stored grid (`grids_full.json`) within 0.003 for CKA and mutual 10-NN.
- WIT-1024 calibration: one cached CLS pair reproduces `prh_val_calibrated` raw score and null mean within 0.002.

## Cost

One GPU pod (RTX PRO 4000/4500 class), priced live before creation. Extraction: 4 ViT-H/g forward passes over 11264 images, about 15 min. D2 scoring: about 20-30 min. WIT-1024 calibration: about 200 pairs at ~12 s each (prh_val_calibrated timing), about 40 min. About 1.5 h in total. New storage on the volume: about 1.2 GB.
