# exp-031: does the PRH capability trend need word order? (shuffled captions through PRH's 10 LLMs)

**Status:** done, 2026-11-28 (pod run at commit 6d4d3e5, RTX PRO 4500, about 2.7 h, about $2) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-031-shuffled-capability/` (copy in local `results/exp-031-shuffled-capability/`); shuffled features in `/workspace/activations/<llm>/layer_XX/*_tf-shufw{0,1}.pt` · **Code:** `extract.py`, `analyze.py`, `slopes.py`, `infra/runpod/run_exp031.sh`, `tests/test_exp031_shuffle.py`; `geoprh.activation_cache.Variant.transform` and `prh_calibration --text-transform / --text-root / --llms`

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

Full tables: `results/exp-031-shuffled-capability/summary.md`; figure `fig_trend.png`; slopes `slopes.json`.

**Trend, Spearman(1 - bpb, excess), median (min to max) over ViTs:**

| measure | intact | shuffled seed 0 | shuffled seed 1 |
|---|---|---|---|
| Mutual 10-NN | 0.93 (0.88-0.95) | 0.77 (0.35-0.88) | 0.78 (0.70-0.85) |
| Linear CKA | 0.67 (0.13-0.72) | 0.01 (-0.82-0.21) | 0.03 (-0.77-0.21) |

**Size of the trend (`slopes.py`), median over ViTs:**

| measure | condition | slope, all 10 LLMs | slope, within BLOOMZ | LLaMA-type minus BLOOMZ |
|---|---|---|---|---|
| Mutual 10-NN | intact | 0.115 | 0.064 | 0.033 |
| Mutual 10-NN | shuffled s0 / s1 | 0.034 / 0.035 | 0.050 / 0.054 | 0.008 / 0.009 |
| Linear CKA | intact | 0.170 | 0.011 | 0.053 |
| Linear CKA | shuffled s0 / s1 | 0.004 / 0.009 | -0.019 / -0.036 | 0.003 / 0.006 |

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
- The cross-family part of PRH's capability trend depends on word order: LLaMA-type models beat BLOOMZ because of how they use order, not because of better word knowledge.
- The within-BLOOMZ size trend is mostly lexical.
- Most of the absolute alignment level survives shuffling, well above exp-023's TF-IDF floor (about 0.035). Unlike on WIT-10k (exp-026), order-free LLM features keep much of PRH's local alignment on WIT-1024.
- Limit: shuffled text is out of distribution, so the order effect here is an upper bound (see "Limits known in advance"). Family and size are confounded with the training data, as in PRH.

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
