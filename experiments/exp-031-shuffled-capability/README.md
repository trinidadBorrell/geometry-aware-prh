# exp-031: does the PRH capability trend need word order? (shuffled captions through PRH's 10 LLMs)

**Status:** done, 2026-11-28 (pod run at commit 6d4d3e5, RTX PRO 4500, about 2.7 h, about $2) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-031-shuffled-capability/` (copy in local `results/exp-031-shuffled-capability/`); shuffled features in `/workspace/activations/<llm>/layer_XX/*_tf-shufw{0,1}.pt` · **Code:** `extract.py`, `analyze.py`, `infra/runpod/run_exp031.sh`, `tests/test_exp031_shuffle.py`; `geoprh.activation_cache.Variant.transform` and `prh_calibration --text-transform / --text-root / --llms`

## Result in one paragraph

Shuffling the words of each caption keeps most of the alignment level, but it removes most of the capability trend. On mutual 10-NN, shuffled captions keep about 68% of the intact excess (median 0.077 vs 0.113), and intact beats shuffled in 170/170 pairs on both measures. The slope of excess on 1 - bpb over the 10 LLMs falls to 30% (0.115 to 0.034). The LLaMA-type vs BLOOMZ gap falls to about 25% (0.033 to 0.008). The size trend within BLOOMZ mostly survives (slope 0.064 to 0.050-0.054). On linear CKA the trend disappears entirely (Spearman 0.67 to 0.01-0.03, slope 0.170 to 0.004-0.009). In short, the between-family part of PRH's trend needs word order, and the within-family size trend is mostly lexical. The Spearman-based P1 fails (0.77 vs the 0.6 limit), but that is because the shuffled trend is still monotone yet much flatter: Spearman sees ranks, not slope.

## Question

PRH's central evidence is a trend: LLMs that model language better align better with vision models (exp-023, WIT-1024: best-layer mutual 10-NN vs 1 - bpb, median Spearman 0.93 over 17 ViTs).

Is that trend carried by the words, or by how LLMs use word order?

## Why this design (and not baseline subtraction)

A word-count or word-vector baseline does not depend on the LLM. Per ViT it is one constant, so subtracting it cannot change the ranking of 10 LLMs. exp-023's "the trend survives the floor" is near-circular for this reason (exp-030 README).

Shuffling each caption's words gives a baseline that is **specific to each LLM**:
- the same model, the same tokenizer and word knowledge, the same bag of words;
- no word order.

Comparing two LLM-specific trends (intact vs shuffled) is not circular, and it adds or subtracts no scores.

## Design

- **Data:** WIT-1024 (PRH's benchmark), original captions; PRH's 10 LLMs x 17 ViTs.
- **ViT features:** cached CLS features. Intact LLM features: cached (`prh_val`).
- **New inputs, one GPU pass per LLM, all hidden states, mean-pooled as PRH:**

| Code | Input |
|---|---|
| I | intact caption (cached; re-extracted for 1 LLM as a pipeline check) |
| S0, S1 | words shuffled within the caption, seeds 0 and 1 |

- **Word-level shuffle:** split on whitespace, permute, re-join, then tokenise. This keeps multi-token words intact, unlike exp-022 / exp-026's token-level shuffle. Special tokens (BOS) are added by the tokenizer as usual.
- **Scoring:** exactly exp-023's calibrated path (`geoprh.prh_calibration`, Gröger's Algorithm 2, 200 permutations, max over layer pairs, mutual 10-NN and linear CKA).

## Checks (the job stops if one fails)

- **Layer 0 must not change under shuffling.** No PRH LLM adds position embeddings at the input (BLOOM uses ALiBi, LLaMA / OpenLLaMA use RoPE), so the mean-pooled layer 0 of I and S must agree (row cosine >= 0.999). Word-level shuffling can change tokenisation at word boundaries, so the check is on a token-level shuffle of 50 captions.
- **Cache check:** re-extracted intact features for bloomz-560m reproduce the cached ones (row cosine >= 0.999 at every layer).
- **Score check:** the intact calibrated score for one pair reproduces `prh_val_calibrated` within 0.002.

## Readouts

1. **Trends:** Spearman(score, 1 - bpb) over the 10 LLMs, per ViT, median over the 17 ViTs, for I, S0 and S1. Also within families: BLOOMZ (5 sizes) and OpenLLaMA (3).
2. **Levels:** I vs S per LLM x ViT pair, with win counts (I > S in how many of 170 pairs), and the median level of each.
3. **Depth:** the best LLM layer for I vs S (relative depth). exp-026 found that order-dependent gains are late.

## Predictions and reading rules (fixed before running)

| | Prediction | Reading if it holds |
|---|---|---|
| P1 | Mutual 10-NN: the shuffled trend is clearly weaker (median Spearman S <= 0.6, against about 0.93 for I) | The capability trend in local alignment is carried by how LLMs use word order |
| P2 | Mutual 10-NN: I > S in at least 160/170 pairs | Order adds fine-scale alignment on PRH's own benchmark, not only on WIT-10k (exp-022, exp-026) |
| P3 | CKA: the shuffled trend is about as strong as the intact one (within 0.15) | Global alignment and its family differences are lexical |

- **If P1 fails** (S keeps the trend): better LLMs align better because of better word-level knowledge. This is the "lexical trend" outcome; it is just as reportable.
- S0 and S1 must agree within 0.1 in median Spearman, or the run is too noisy to read.

## Limits known in advance

- 10 LLMs from three families; size, family and training data are confounded (as in PRH).
- Shuffled text is out of distribution, so S underestimates what an order-free model could do with the same words. P1 is therefore an upper bound on how much the trend needs order. A phrase-preserving shuffle (exp-032) narrows this.
- One caption per image, short captions only.

## Cost

- **GPU extraction:** 10 LLMs (0.56B-13B, bf16) x 1024 captions x 2 seeds, all hidden states, plus the checks.
  - One RTX PRO 4500 (32 GB, about $0.72/h). 13B in bf16 fits.
  - Weight downloads dominate (about 115 GB in total). Keep `HF_HOME` on the container disk and delete each model after use, since the volume is near quota.
  - About 1.5-2 h, about $1.5.
- **Calibration:** 170 pairs x 2 seeds x 2 measures, on the same pod or a CPU pod. exp-023's calibration of 170 pairs cost under $0.25.

## Implementation

- **Extraction (`extract.py extract`)** calls platonic-rep's own `extract_llm_features` on the transformed captions, so loading (bf16, `device_map="auto"`), left padding to the longest caption of the set, masked mean pooling (BOS included, as PRH), loss and bpb are exactly PRH's. The feature file is split into the cache under the key `minhuh_prh_wit_1024_pool-avg_cid-0_tf-shufw<seed>`, with `transform` and `seed` in each `meta.json`.
- **Word shuffle:** whitespace split, permute, join with single spaces; the rng is keyed by (seed, row). Punctuation stays attached to its word, and one-word captions are unchanged. `captions_changed` in `extract.jsonl` records the share of captions that moved.
- **Scoring:** `geoprh.prh_calibration --text-transform shufw<seed> --text-root <dir>`. ViT features and model info (params, layers, intact bpb) come from the intact cache, so the capability axis is the intact captions' 1 - bpb, as in exp-023. The shuffled bpb is logged in `extract.jsonl` as an out-of-distribution measure.
- **Analysis (`analyze.py`):** intact rows from `<out>/reext/` when present (same GPU), with `prh_val_calibrated` as a reproduction check; otherwise from `prh_val_calibrated`. Shuffled rows come from `<out>/shufw*/`. It writes `summary.md`, `trends.json` and `fig_trend.{png,pdf}`, plus the P1-P3 verdicts and the seed-agreement check. P2's rule is applied as the share 160/170 of whatever pairs are present.
- **Checks (`extract.py check-*`, all in the `checks` stage; any failure stops the job):** layer 0 under a token-level shuffle of 50 captions for bloomz-560m and llama-7b (min row cosine >= 0.999); re-extracted intact bloomz-560m against the cache (>= 0.999 at every layer); calibration of bloomz-560m x ViT-tiny against `prh_val_calibrated` (raw, null mean and g within 0.002).

## Local smoke test (2026-11-28, laptop GPU, bloomz-560m only)

- **All three checks pass.** Layer 0 under a token shuffle: min row cosine 1.000 (50 of 50 captions changed). Re-extracted intact features reproduce the cache at every layer (min row cosine 1.000; bpb identical, 2.0759). Calibration reproduces the stored bloomz-560m x ViT-tiny pair within 1e-4.
- **Shuffled extraction:** 96% of captions change; bpb rises from 2.08 to 2.67.
- **First numbers (one LLM, the 4 locally cached ViTs, not a result):** shuffled excess is below intact in 4/4 pairs, on both measures. But it keeps most of it: mutual 10-NN 0.058-0.084 vs intact 0.074-0.110, against a TF-IDF floor of about 0.035 (exp-023). On WIT-10k the shuffled LLMs fell to the floor (exp-026). If this holds for the whole pool, order-free LLM features carry much of PRH's local alignment on WIT-1024, which bears on P1.

## Change during the pod run: intact re-extracted on the same GPU

The first pod attempt stopped at the cache check. Re-extracted intact bloomz-560m on the RTX PRO 4500 reached only a min row cosine of 0.9988 at the last layer (bpb 2.081 vs 2.076 cached), while the laptop GPU had reproduced the cache exactly. bf16 kernels differ across GPUs. So that I and S differ only in word order, the intact captions were re-extracted for all 10 LLMs on the same GPU (`reext`, key `..._tf-reext`, kept on container disk only), and these are the primary intact condition. The stored scores (`Ist`) become a reproduction check, and the cache check is logged but not fatal. The other checks passed: layer 0 under a token shuffle (bloomz-560m, llama-7b) and the calibrated score of one stored pair (difference < 1e-9).

**Reproduction:** same-GPU vs stored intact excess over 170 pairs: mutual 10-NN median |d| 0.0002, max 0.0008, Spearman 0.9998; CKA max |d| 0.0001. The feature drift does not change any score.

## Results (WIT-1024, 10 LLMs x 17 ViTs)

Full tables: `results/exp-031-shuffled-capability/summary.md` and `trends.json`; figure `fig_trend.png`.

G = I - mean(S0, S1) is the alignment that needs word order (the "order gain"). It is a difference of scores, read only for its trend. Its level is not evidence of image-relevant order information: in exp-030's toy, intact minus shuffled is about 0.055 mutual 10-NN even when order carries no image information.

**Trend over the 10 LLMs, per ViT, median (min to max) over ViTs:**

| measure | condition | Spearman(1 - bpb, excess) | slope on 1 - bpb |
|---|---|---|---|
| Mutual 10-NN | intact | 0.93 (0.88-0.95) | 0.115 |
| Mutual 10-NN | shuffled s0 / s1 | 0.77 (0.35-0.88) / 0.78 (0.70-0.85) | 0.034 / 0.034 |
| Mutual 10-NN | order gain G | 0.94 (0.85-0.99) | 0.083 |
| Linear CKA | intact | 0.67 (0.13-0.72) | 0.170 |
| Linear CKA | shuffled s0 / s1 | 0.01 (-0.82-0.21) / 0.03 (-0.77-0.21) | 0.004 / 0.009 |
| Linear CKA | order gain G | 0.90 (0.56-0.96) | 0.145 |

Shuffled captions keep 29% of the intact slope on mutual 10-NN and 9% on CKA (per-ViT ratio, median); the order gain carries the other 71% and 91%.

**Within vs between families** (slope; within = pooled over families with >= 2 LLMs, family means removed; between = over the 3 family means):

| measure | condition | within-family | between-family |
|---|---|---|---|
| Mutual 10-NN | intact | 0.066 | 0.123 |
| Mutual 10-NN | shuffled (mean) | 0.052 | 0.029 |
| Mutual 10-NN | order gain G | 0.022 | 0.095 |
| Linear CKA | intact | 0.001 | 0.181 |
| Linear CKA | shuffled (mean) | -0.037 | 0.007 |
| Linear CKA | order gain G | 0.043 | 0.163 |

The size trend within a family is mostly kept by shuffled captions (mKNN: 0.052 of 0.066). The trend between families is mostly order gain (0.095 of 0.123; CKA 0.163 of 0.181). The share of alignment that survives shuffling falls with capability: 0.77-0.79 for BLOOMZ, 0.61-0.65 for the LLaMA-type models (Spearman with 1 - bpb -0.90 on both measures).

**Levels:** intact > shuffled in 170/170 pairs on both measures and both seeds. Median excess: mutual 10-NN 0.113 intact vs 0.077 shuffled; CKA 0.217 vs 0.144-0.149. Shuffled mutual 10-NN per LLM is flat across the LLaMA-type models (0.078-0.085), against 0.124-0.132 intact.

**bpb on shuffled captions** rises from 1.62-2.08 to 2.50-2.68 and hardly separates the LLMs any more (`extract.jsonl`).

**Depth:** the best LLM layer moves little (relative depth 0.68-0.97 intact, 0.75-0.96 shuffled). Order does not shift alignment to a different depth at this resolution.

**Verdicts (fixed before the run):**

| | result | verdict |
|---|---|---|
| P1 | mutual 10-NN shuffled Spearman 0.77 / 0.78 (limit 0.6) | fails as worded; the slope falls to 30%, see above |
| P2 | intact > shuffled in 170/170 pairs | holds |
| P3 | CKA Spearman falls by 0.64-0.67 (limit 0.15) | fails: the CKA trend is not lexical, it disappears without order |
| seeds | S0 vs S1 median Spearman differ by 0.01 (mKNN) and 0.02 (CKA) | ok |

**Reading.**
- **Level vs trend.** The order-free part of each LLM's representation carries most of the alignment (about 70%, twice exp-023's TF-IDF floor of 0.035) but little of the capability trend. The trend lives in the order gain, which tracks capability as closely as total alignment on mutual 10-NN (0.94 vs 0.93) and more closely on CKA (0.90 vs 0.67): the lexical part adds model-to-model variation unrelated to capability (e.g. bloomz-1b1's high CKA).
- **External lexical baselines cannot show this.** They are one constant per ViT, so they cannot change a ranking of LLMs (exp-023's 0.93 to 0.92). The same LLM on shuffled words is the baseline that can.
- **Between vs within.** The order-dependent trend is mostly between families (LLaMA-type vs BLOOMZ); the size trend within BLOOMZ is mostly lexical. With three families, two of them LLaMA-architecture, "capability" here is largely "BLOOMZ vs LLaMA-type". The extension below adds three families to test this.
- **What "needs order" means.** Shuffled captions are still contextualised (every token attends to the others); what they lose is order. Two readings fit the gain: (a) order-dependent composition carries image-relevant structure, which better LMs extract better; (b) better LMs encode the same words more precisely in well-formed text, and shuffling puts every model out of distribution (bpb 2.50-2.68 for all). This experiment does not separate them; exp-030's conditional permutation test does.
- Most of the absolute alignment level survives shuffling. Unlike on WIT-10k (exp-026), order-free LLM features keep much of PRH's local alignment on WIT-1024.
- Limit: shuffled text is out of distribution, so the order effect here is an upper bound (see "Limits known in advance"). Family and size are confounded with the training data, as in PRH.

## Extension: three newer families (2026-11-28)

**Why:** the order-dependent trend above is mostly a contrast between BLOOMZ and LLaMA-type models. Newer families with much lower bpb test whether it holds beyond that one contrast.

**Models (5):**

| model | repo used | why this repo |
|---|---|---|
| Qwen2.5-1.5B, Qwen2.5-7B | `Qwen/Qwen2.5-*` | official, ungated |
| Llama-3.2-3B | `unsloth/Llama-3.2-3B` | ungated copy of Meta's base weights |
| Llama-3.1-8B | `NousResearch/Meta-Llama-3.1-8B` | ungated copy; Gröger et al.'s pool also uses NousResearch Llama-3 |
| Gemma-2-9B | `unsloth/gemma-2-9b` | ungated copy of Google's base weights |

All base models. Same pipeline and conditions (reext, shufw0, shufw1) on one GPU; old rows are not recomputed (GPU differences do not move scores, see the reproduction above).

**Pipeline changes:**
- Gemma-2 runs with eager attention: transformers 4.57's SDPA path drops Gemma-2's attention-logit soft-capping.
- Tokenizers keep their defaults, as in PRH, for the features.
- `prh_calibration --llms` accepts LLMs outside the "val" modelset; their info comes from their intact re-extraction (`_tf-reext`).
- `analyze.py` reads their bpb from `extract.jsonl`, reports PRH's 10 and all LLMs separately, adds the order gain G, slopes and the within / between-family split; the pre-registered verdicts stay on PRH's 10.

**Found while preparing: PRH's bpb is not comparable across tokenizers.**
- platonic-rep scores each token from the position before it. Tokenizers that add no BOS (BLOOMZ, Qwen2.5) therefore leave the first caption token unscored, and on short captions that token carries many bits.
- Local measurements on WIT-1024 (`extract.py bpb`):

| model | bpb, PRH default | bpb, every token scored (EOS prepended) |
|---|---|---|
| bloomz-560m | 2.076 (= stored) | 2.283 |
| Qwen2.5-1.5B | 1.616 | 1.966 |

- PRH's own capability axis thus favours BLOOMZ by about 0.2 bpb against the LLaMA-type models, which score every token. That bias works against the BLOOMZ-LLaMA gap, so it does not create PRH's trend. For Qwen2.5 it is decisive: by default it sits next to llama-7b, scored fully next to BLOOMZ.
- The run computes both values for the five BLOOMZ models and every new LLM (`bpb.jsonl`; the other families add a BOS, so both values agree). `analyze.py` reports the trends on both capability axes.

**Local smoke test (2026-11-28, Qwen2.5-1.5B, laptop GPU):**
- Layer 0 under a token shuffle: min row cosine 1.000.
- Extraction: intact bpb 1.616, shuffled 2.418 (96% of captions changed).
- Scoring through `--llms` against the 4 local ViTs. Mutual 10-NN excess: intact 0.084-0.128, shuffled 0.057-0.087. CKA: 0.233-0.345 intact, 0.151-0.260 shuffled. Same pattern as PRH's pool: shuffled is lower in every pair but keeps most of the level.

**Readouts:**
- Do the new families sit on the same lines?
- Does G keep tracking capability over 15 LLMs and 6 families, with a flat shuffled curve?
- Do the answers hold on both capability axes?

## Run plan

- **Pod:** one RTX PRO 4500 Blackwell (32 GB), EU-RO-1, template `sw0wyyvy63` with the volume. 13B in bf16 is 26 GB, so if it does not fit, the next rung is an A100 80GB (ask first). Container disk: 60 GB rather than the template's 30 GB, in case weights have to be downloaded.
- **Weights:** read offline from the volume's second HF cache (`/workspace/.cache/huggingface/hub`, which held all ten PRH LLMs in exp-016). A model missing there is downloaded to container disk and deleted after use.
- **Features:** written to container disk (`/root/exp031/activations`, about 2.3 GB per seed in bf16), scored there, and copied into `/workspace/activations` in the last stage. A quota failure on that copy is logged, not fatal. Calibrated rows go straight to the volume and the score stage resumes.
- **Launch:**

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp031.sh > /workspace/results/oddharak/exp-031.log 2>&1 < /dev/null &"
uv run python experiments/exp-031-shuffled-capability/analyze.py --out results/exp-031-shuffled-capability --stored results/prh_val --stored-cal results/prh_val_calibrated
```

`STAGES`, `SEEDS`, `LLMS` and `GRACE` override the defaults; the pod terminates itself `GRACE` seconds after the job, whatever its exit status.
