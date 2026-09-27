# exp-006: is 1024 samples enough? Sample-set variability and sample size

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-006-sample-size/`

## Idea

Koepke et al. 2026 ("Back into Plato's Cave", arXiv 2604.18572) argue that the 1024 WIT pairs
used by Huh et al. are too sparse to represent the underlying space. They keep the 1024 queries
fixed, grow the gallery to 1M (WIT) and 15M (LAION), and find that mutual kNN collapses (e.g.
DINOv2 x OpenLlama-3b: 0.135 -> 0.008 at k=10). We ask two narrower questions with all-vs-all
alignment on our existing pipeline:

1. **Variability:** how much do CKA and mutual kNN change between five disjoint 1024-sample sets
   drawn from the same distribution?
2. **Sample size:** how much do they change when the same pairs are measured on 2048, 4096 and
   10240 samples?

Both questions are asked of the raw scores and of the Groger null-calibrated ones (null mean
mu0, calibrated g = (raw - tau) / (1 - tau)).

## Hypothesis

- If 1024 samples are enough, the five disjoint sets agree closely (spread much smaller than the
  differences between model pairs) and so does the pair ranking.
- Raw mutual kNN at fixed k=10 should fall with n, since chance overlap is ~k/n and the
  neighbourhood becomes a smaller fraction of the data. Calibration corrects only the chance part.
  If the calibrated score also falls, the loss is structural and not only chance, which is
  Koepke et al.'s point. CKA is a global statistic and should be much less sensitive to n.

## Method

- **Data:** the first 10240 rows of `askoepke/wit_1m_recaptioned`, config `wit_1m`, with the
  original WIT caption. This config excludes the 1024 PRH query images; the rows are Koepke et
  al.'s deduplicated WIT-1M gallery. The subset is saved once as
  `wit_1m_first10240.parquet` on the volume.
- **Models (15 pairs):** the 3 smallest LLMs (bloomz-560m, -1b1, -1b7) x the smallest ViT of each
  family (ImageNet21K tiny and small, MAE base, DINOv2 small, CLIP laion2b base).
- **Activations:** `extract.py`, the platonic-rep recipe (every LLM layer mean-pooled over the
  caption, every ViT block's CLS token). They are stored in the shared cache under
  `askoepke_wit_1m_recaptioned_first10240-original_pool-{avg,cls}`. None were cached before.
- **Metrics:** `calibrate.py`. Mutual kNN (k=10) and linear CKA, max over layer pairs, q=0.95
  clamp and l2 norm per sample set, 200 permutations, alpha = 0.05, seed 0, as exp-003.
  The CKA null is computed in feature space, which matches Aristotelian's Gram version
  (`tests/test_exp006_calibrate.py`). The CKA null is skipped at n=10240 for now.
- **Sample sets:** rows [1024 s, 1024 (s+1)) for s = 0..4 (disjoint), and the nested prefixes
  2048, 4096, 10240. The PRH `wit_1024` result for the same 15 pairs (exp-003) is shown as a
  reference.
- **Difference from Koepke et al.:** they fix the queries and grow the gallery. Here every sample
  is both a query and a neighbour candidate (all-vs-all), the setting of Huh et al. and Groger
  et al.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp006_sample_size.sh > /workspace/results/oddharak/exp-006.log 2>&1 &"
uv run python experiments/exp-006-sample-size/plot.py --root results/exp-006-sample-size \
    --prh results/prh_val_calibrated/calibrated_pairs.jsonl
```

## Results

Run 2026-09-27, commit `98149d7`, RTX PRO 4500. It took ~25 min: extraction ~10 min,
calibration ~15 min. Raw outputs are `calibrated.jsonl` and the subset parquet in
`/workspace/results/oddharak/exp-006-sample-size/`, copied to `results/exp-006-sample-size/`.
Activations for the 8 models x 10240 samples are in the shared cache. A first pass included one
sample with an empty caption, which gives NaN text features. It was discarded, and `extract.py`
now skips empty captions and refuses non-finite features.

Mean over the 15 pairs (max over layer pairs; every p-value is at the 0.005 floor of 200
permutations):

| | 5 disjoint 1024 sets | per-pair range across sets | PRH wit_1024 | n = 2048 | n = 4096 | n = 10240 |
|---|---|---|---|---|---|---|
| mKNN raw | 0.096 | 0.012 | 0.095 | 0.073 | 0.058 | 0.042 |
| mKNN null mean | 0.012 | 0.000 | 0.012 | 0.006 | 0.003 | 0.001 |
| mKNN calibrated g | 0.083 | 0.012 | 0.082 | 0.067 | 0.054 | 0.040 |
| CKA raw | 0.313 | 0.016 | 0.329 | 0.290 | 0.278 | 0.271 |
| CKA null mean | 0.115 | 0.006 | 0.116 | 0.063 | 0.032 | not run |
| CKA calibrated g | 0.223 | 0.018 | 0.240 | 0.243 | 0.254 | not run |

(The n = 1024 point of the nested series is disjoint set 0.)

**1. Variability across disjoint 1024-sample sets is small (fig1).** The typical spread of one
pair across the five sets is 0.012 for mKNN (CV 5%) and 0.016 for CKA (CV 2%). The spread between
model pairs is 4x (mKNN) and 9x (CKA) larger than the spread within a pair (between-pair sd
0.022 vs within 0.005; 0.058 vs 0.007). The pair ranking is stable: Spearman between any two
disjoint sets is 0.91-0.99 for mKNN and 0.99 for CKA. The original PRH `wit_1024` set sits inside
the disjoint-set spread for mKNN. For CKA it is slightly above it, most visibly for the MAE pairs
(raw 0.329 vs 0.313 on average). So at n = 1024 the numbers are reproducible: 1024 samples are
not too few to *estimate* these scores.

**2. But the scores depend on n itself (fig2).** Going from 1024 to 10240 samples:
- **mKNN (k=10)** falls to 0.41x of its n=1024 value (0.100 -> 0.042, range 0.37-0.47x across
  pairs). Calibration does not remove this: the null mean only falls from 0.012 to 0.001 (chance
  ~ k/n), so the calibrated score falls almost as much (0.087 -> 0.040, 0.46x). The loss is
  structural, not chance overlap, which agrees with Koepke et al. With a fixed k, a neighbourhood
  covers 1% of the data at n=1024 and 0.1% at n=10240, and the two models agree much less on
  those fine-grained neighbours. The pair ranking survives (Spearman 0.94 between n=1024 and
  n=10240), and every family drops by about the same factor (CLIP 0.130 -> 0.059, MAE 0.061 ->
  0.025).
- **Linear CKA** is much more stable. Raw CKA falls 13% (0.313 -> 0.271, 0.78-0.95x), and most
  of that is its chance level: the null mean halves with every doubling of n (0.115 -> 0.063 ->
  0.032). The calibrated CKA therefore *rises* slightly (0.220 -> 0.254 at n=4096). Raw CKA at
  n=1024 is inflated by finite-sample bias; the global alignment itself does not shrink with n.
  The pair ranking moves more than mKNN's (Spearman 0.80 between n=1024 and 10240).

**Reading.** Sample-set noise at n = 1024 is not the issue. The issue is that the local metric
measures a different quantity at each n: mKNN at fixed k asks about ever finer neighbourhoods as
n grows, and cross-modal agreement at that scale is weak. The global metric, once corrected for
its own chance level, does not decline over this range. This is consistent with Koepke et al.'s
"coarse semantic overlap rather than fine-grained structure".

Caveats: 15 small-model pairs (bloomz <= 1.7B, ViTs <= base); n only up to 10240; all-vs-all
rather than Koepke et al.'s fixed-query / growing-gallery setup; the CKA null at n = 10240 was
skipped.

## Next

- CKA null at n = 10240 (the feature-space null is cheap, ~1 min per pair on the GPU).
- mKNN with k scaled to n (k = n/100, which Koepke et al. report is stable) versus fixed k.
  That separates "neighbourhood size relative to the data" from "number of samples".
- Larger models from the val grid, to see whether the n-dependence changes with scale.
