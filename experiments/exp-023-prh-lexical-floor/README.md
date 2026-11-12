# exp-023: how much of PRH's alignment does a word kernel reach?

**Status:** done, 2026-11-06 · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-023-prh-lexical-floor/` (copy in local `results/exp-023-prh-lexical-floor/`) · **Code:** `run.py`, `analyze.py`, uncommitted, on `ef2baa9`

## Idea

Run the lexical floor on PRH's own setup: WIT-1024 original captions, the published 10 LLM x 17 ViT pool, cached activations, PRH's procedure (q = 0.95 clamp, l2 norm, max over layer pairs). The first results below are raw scores; null calibration follows.

Compared for every pair: word kernels with no model; LLM layer 0; the best LLM layer. All at their best ViT layer. The best-pair score reproduced stored `prh_val` values within 0.002 on every one of the 170 pairs, or the job would have stopped.

## Results

Median (min-max) over 170 pairs:

| Measure | TF | TF-IDF | Word TF-IDF, no stopwords | Layer 0 | Best layer |
|---|---:|---:|---:|---:|---:|
| Mutual kNN, k = 10 (chance 0.010) | 0.035 (0.027-0.044) | 0.046 (0.031-0.059) | 0.060 (0.037-0.074) | 0.042 (0.025-0.061) | 0.125 (0.056-0.202) |
| Unbiased CKA (chance 0) | 0.054 (0.035-0.082) | 0.082 (0.049-0.110) | 0.129 (0.072-0.148) | 0.088 (0.042-0.129) | 0.309 (0.209-0.433) |
| Raw CKA | 0.184 (0.077-0.238) | 0.359 (0.132-0.457) | 0.408 (0.154-0.490) | 0.230 (0.085-0.312) | 0.382 (0.243-0.530) |

1. **Words reach about the embedding layer and well below the best layer.** Word TF-IDF without stopwords is at or above layer 0 in 100% of pairs on mutual kNN and on unbiased CKA, and never reaches the best layer on either. Median ratio to the best layer: 0.29 (TF), 0.38 (TF-IDF), 0.48 (word) on mutual kNN; 0.19, 0.26, 0.38 on unbiased CKA. Ratios are descriptive, not shares of variance.
2. **Raw CKA cannot be used for this comparison.** Raw TF-IDF CKA nearly equals the best layer (ratio 0.83), and passes it in 25% of pairs (word: 42%), while unbiased CKA shows a ratio of 0.26 and no pair above. This is biased CKA inflating high-rank sparse kernels at n = 1024, seen earlier in exp-022. A word kernel does not beat PRH's models; the estimator favours it.
3. **The best layer is late:** median relative depth 0.89 (mutual kNN), 0.96 (unbiased CKA). Layer 0 is never the best layer.

### LLM capability

| LLM (by 1 - bpb) | 1 - bpb | mKNN layer 0 | mKNN best | uCKA layer 0 | uCKA best |
|---|---:|---:|---:|---:|---:|
| bloomz-560m | -1.076 | 0.036 | 0.097 | 0.077 | 0.271 |
| bloomz-1b7 | -0.939 | 0.043 | 0.107 | 0.098 | 0.288 |
| bloomz-3b | -0.894 | 0.040 | 0.110 | 0.090 | 0.298 |
| bloomz-1b1 | -0.885 | 0.036 | 0.104 | 0.061 | 0.295 |
| bloomz-7b1 | -0.839 | 0.037 | 0.118 | 0.071 | 0.309 |
| open_llama_3b | -0.718 | 0.046 | 0.137 | 0.096 | 0.328 |
| open_llama_7b | -0.677 | 0.046 | 0.141 | 0.096 | 0.339 |
| open_llama_13b | -0.656 | 0.046 | 0.145 | 0.089 | 0.335 |
| llama-7b | -0.648 | 0.048 | 0.144 | 0.096 | 0.332 |
| llama-13b | -0.621 | 0.048 | 0.143 | 0.101 | 0.330 |

Mean over 17 ViTs. Plain TF-IDF is flat here (mKNN 0.045-0.046; uCKA 0.080-0.085), and word-level TF-IDF is one constant per ViT.

- **The trend lives in deep layers.** Best layer vs 1 - bpb, Spearman per ViT, median: 0.93 (mKNN), 0.85 (uCKA). Layer 0 vs 1 - bpb: 0.84 (mKNN) but 0.38 (uCKA).
- **That layer-0 mKNN value is mostly between families:** within families it is nearly flat (bloomz 0.036-0.043 with no order; open_llama 0.046; llama 0.048). Best layer rises with size inside bloomz (0.097 to 0.118, Spearman 1.00) and open_llama (0.137 to 0.145), not across the two llama sizes.
- **Family is a bigger step than size:** bloomz-7b1 (0.118) sits below open_llama_3b (0.137). Size, family, bpb and depth are confounded across this pool.

### Vision capability

Within ViT families, TF-IDF mutual kNN rises with ViT size in every family (Spearman 1.00: e.g. DINOv2 0.043 to 0.049, CLIP 0.053 to 0.058), so part of the vision-side trend appears with no language model. Best-layer mutual kNN rises too (DINOv2 0.111 to 0.139, CLIP 0.158 to 0.173), by more in absolute terms, except AugReg (flat, Spearman 0.20) and MAE (non-monotonic). Unbiased CKA disagrees with raw CKA on size: AugReg best falls with size (0.300 to 0.250, Spearman -1.00) and DINOv2 is mixed (-0.40), while raw CKA rises in both.

## Null calibration (2026-11-06)

Gröger's Algorithm 2, 200 permutations (p floor 0.005). TF-IDF with each LLM's tokens is the chosen floor. Each series is calibrated over the cells it chooses from; best-layer values are the stored `prh_val_calibrated`. A recomputed full grid reproduced the stored raw score and null mean within 0.0001 for both measures. Tables: `calibrated_summary.md`; figure: `fig_calibrated.png`.

Median (min-max) of raw minus the null of the max, 170 pairs:

| Measure | TF-IDF, no model | Layer 0 | Best layer |
|---|---:|---:|---:|
| Mutual kNN, k = 10 | 0.035 (0.019-0.048) | 0.031 (0.014-0.050) | 0.113 (0.044-0.189) |
| Linear CKA | 0.043 (0.027-0.067) | 0.066 (0.039-0.106) | 0.217 (0.103-0.342) |

- All three series pass in 100% of pairs (p < 0.05).
- **Raw CKA's inflation is confirmed:** TF-IDF raw 0.359 has a null of 0.314. The best layer's raw 0.382 has a null of 0.176.
- Median ratio to the best layer: TF-IDF 0.31 (mKNN), 0.20 (CKA); no pair reaches the best layer. Excess may understate high-rank word kernels, since unbiased CKA gives 0.26. Word TF-IDF without stopwords gives 0.41 and 0.25, as a sensitivity check.
- **Capability, after calibration:** mutual kNN keeps its trend (best layer vs 1 - bpb, Spearman 0.93; best minus TF-IDF 0.92). CKA's weakens (0.67; vs params 0.43, range -0.20 to 0.51) and is mostly a gap between families: within bloomz it is not ordered by size (0.203, 0.230, 0.175, 0.185, 0.202) and open_llama and llama are flat.
- **ViT size:** TF-IDF mKNN rises with size in all 5 families; the best layer does in 3. CKA falls with size for AugReg (best layer 0.231 to 0.167).

## Above the floor: Gemini vs original captions

Same 10240 WIT images, TF-IDF floor, excess over null, n = 10240 (`gap_d2.py`, `gap_d2.json`). Means over 16 pairs: bloomz-1b7, bloomz-7b1, open_llama_7b, llama-13b x 4 ViTs. Floors use each LM's tokens for DINOv2 and CLIP, Pythia's for MAE and AugReg. This is a level difference, not a decomposition, and pairs share models, so there is no confidence interval.

| Measure, cell | Floor, orig to Gemini | Deep, orig to Gemini | Deep minus floor, orig to Gemini | Gemini gap larger in |
|---|---:|---:|---:|---:|
| k = 10, best | 0.029 to 0.156 | 0.065 to 0.215 | 0.036 to 0.059 | 16/16 |
| k = 10, last x last | 0.028 to 0.150 | 0.055 to 0.195 | 0.026 to 0.045 | 16/16 |
| CKA, best | 0.075 to 0.549 | 0.296 to 0.650 | 0.221 to 0.102 | 0/16 |
| CKA, last x last | 0.072 to 0.499 | 0.226 to 0.556 | 0.154 to 0.057 | 1/16 |

- **The gaps do not match.** Long captions raise the floor far more than the gap: +0.127 vs +0.023 at k = 10, and on CKA the gap shrinks.
- **Most of what longer captions add is already visible in word statistics.** LMs add somewhat more above the floor at k = 10, and less in CKA.
- **Room left matters.** CKA has a ceiling, and a floor near 0.55 leaves less to gain. Scaled by the room left, (deep - floor) / (1 - floor), CKA at the best cell is about equal (0.240 vs 0.233, 8/16), while last x last is still lower (0.166 vs 0.113, 3/16).
- **Pythia** agrees in direction: k = 10 gap 0.005 to 0.023, CKA gap 0.129 to 0.056.
- **Held-out ridge (exp-022)** gives the same picture without subtracting CKA: adding an LM to words raises R^2 by about 0.03-0.07 on Gemini and 0.08-0.14 on original captions (DINOv2).
- **Not separated:** caption length and caption content move together here.

## Limits

- Raw scores differ in chance level by kernel; use the calibrated section for raw CKA.
- Layer pairs are chosen on the same samples. exp-022's split-half check found this harmless at n = 10240; not re-checked at n = 1024.
- bpb is on WIT captions, not OpenWebText as in the paper (exp-002).
- One caption per image, original WIT captions. Ten LLMs from three families are few, and one pool.

## Next

Sample size (Koepke). Truncated Gemini through LMs, to separate length from content.

## Commands

```bash
uv run python experiments/exp-023-prh-lexical-floor/run.py --out <dir> --stored /workspace/results/oddharak/prh_val --workers 3
uv run python experiments/exp-023-prh-lexical-floor/analyze.py --out <dir> --stored <local prh_val>
```

Run: `cpu3c` 8 vCPU, 16 GB. Six workers ran out of memory at 16 GB; three finished the remaining pairs. Under $0.25 of pod time.
