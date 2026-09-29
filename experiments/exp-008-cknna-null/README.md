# exp-008: how much of CKNNA(k) is chance? Permutation-null calibration at every k

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-008-cknna-null/`

## Idea

exp-005 found the published CKNNA peaking at k = 400-600 for every pair, and concluded that
CKNNA interpolates between mutual kNN (local) and CKA (global). But none of the CKNNA
definitions is chance-corrected. For j among i's own top-k neighbours, the centred kernel entry
is positive, so on the shared neighbours both factors tend to be positive even for unrelated
models. For two independent random feature sets (n = 1024) the published formula scores 0.01 at
k = 10, 0.32 at k = 500 and 0 at k = n - 1: a hump at the same place as exp-005's peak. Mutual
kNN has the same issue (chance k/(n-1)). Here every metric of exp-007 is calibrated against a
permutation null at every k.

## Hypothesis

- The null mean of `paper` rises from ~0 at k = 10 to ~0.3 at k ~ n/2 and falls back to ~0 at
  k = n - 1; the null of `code` and `eq36` is larger and does not fall to 0 as fast.
- After calibration the k = 400-600 peak of exp-005 disappears or shrinks strongly: calibrated
  CKNNA should be largest at small k (local alignment) or at k = n - 1 (CKA), not in between.
- If the calibrated curve still peaks at intermediate k, intermediate neighbourhoods carry real
  cross-modal structure beyond chance.

## Method

- `calibrate.py`: Groger et al.'s Algorithm 2 per (LLM, ViT) pair: permute the image-caption
  pairing, recompute the whole layer x layer matrix, take its max; 200 permutations,
  alpha = 0.05, seed 0. The same permutations (Aristotelian `batched_perms`, CPU) are used for
  every metric and every k. Reports raw, null mean `mu0` and sd, the cutoff `tau` (exact
  order statistic), the p-value and the calibrated `g = (raw - tau) / (1 - tau)` (0 if
  raw <= tau), all from Aristotelian's `_build_gated_summary`. `g` assumes a maximum of 1,
  which `code` and `eq36` do not respect; for those, read raw against `mu0`/`tau`.
- Metrics and grid as exp-007: mutual kNN and the four CKNNA definitions at 15 values of k,
  linear CKA once; 170 pairs. The null reuses exp-007's matrices: a permutation acts on the
  language model's masked Grams, and the self terms do not change.
- `tests/test_cknna_variants.py` checks that the calibration of `code` and of CKA reproduces
  Aristotelian's own `compute_alignment_gated_cknna_cached` / `..._cka_cached` (raw, tau, mu0,
  g, p) with the same permutations.
- Null maxima per pair, metric and k are saved in `nulls/*.npz` for later analysis.

## Commands

```bash
# on the pod, after exp-007 (same script)
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp007_exp008.sh > /workspace/results/oddharak/exp-007-008.log 2>&1 < /dev/null &"
uv run python experiments/exp-008-cknna-null/plot.py --root results/exp-008-cknna-null
```

## Results

Run 2026-09-29, commit `bcced83`, A100 80GB, 1.6 h of calibration for 170 pairs (after
exp-007 on the same pod). Raw outputs are `calibrated_pairs.jsonl` and `nulls/*.npz` in
`/workspace/results/oddharak/exp-008-cknna-null/`, copied to `results/exp-008-cknna-null/`.
Figures come from `plot.py`.

Mean over 170 pairs (max over layer pairs; 200 permutations):

| k | 10 | 100 | 200 | 400 | 500 | 600 | 800 | 1000 | 1024 (= n-1) |
|---|---|---|---|---|---|---|---|---|---|
| **raw** | | | | | | | | | |
| mutual kNN | 0.12 | 0.26 | 0.35 | 0.50 | 0.57 | 0.64 | 0.80 | 0.98 | 1.00 |
| CKNNA `paper` | 0.16 | 0.29 | 0.37 | 0.45 | 0.45 | 0.44 | 0.34 | 0.29 | 0.29 |
| CKNNA `centred` | 0.17 | 0.30 | 0.36 | 0.40 | 0.39 | 0.37 | 0.32 | 0.31 | 0.31 |
| CKNNA `code` | 0.15 | 0.27 | 0.38 | 0.60 | 0.72 | 0.87 | 1.26 | 0.90 | 0.31 |
| CKNNA `eq36` | 0.15 | 0.31 | 0.45 | 0.72 | 0.88 | 1.06 | 1.50 | 1.62 | 0.49 |
| **null mean** | | | | | | | | | |
| mutual kNN | 0.01 | 0.10 | 0.20 | 0.40 | 0.50 | 0.59 | 0.79 | 0.98 | 1.00 |
| CKNNA `paper` | 0.01 | 0.09 | 0.18 | 0.32 | 0.36 | 0.38 | 0.27 | 0.02 | 0.02 |
| CKNNA `centred` | 0.01 | 0.08 | 0.14 | 0.20 | 0.20 | 0.17 | 0.07 | 0.00 | 0.00 |
| CKNNA `code` | 0.01 | 0.07 | 0.16 | 0.42 | 0.60 | 0.80 | 1.23 | 0.86 | 0.00 |
| CKNNA `eq36` | 0.01 | 0.13 | 0.28 | 0.64 | 0.84 | 1.05 | 1.49 | 1.62 | 0.44 |
| **calibrated g** | | | | | | | | | |
| mutual kNN | 0.11 | 0.17 | 0.18 | 0.16 | 0.14 | 0.11 | 0.06 | 0.01 | 0.00 |
| CKNNA `paper` | 0.15 | 0.21 | 0.23 | 0.18 | 0.12 | 0.07 | 0.09 | 0.26 | 0.26 |
| CKNNA `centred` | 0.16 | 0.24 | 0.26 | 0.25 | 0.24 | 0.24 | 0.26 | 0.31 | 0.31 |

Linear CKA: raw 0.39, null mean 0.17, g 0.27. `g` is not shown for `code` and `eq36` (their
null exceeds 1, so `(raw - tau) / (1 - tau)` is meaningless).

- **The null is large and shaped like exp-005's "peak" (fig1).** For `paper` it rises from 0.01
  (k = 10) to 0.38 (k = 600) and falls to 0.02 at k = n - 1, as predicted from random features.
  At k = 400-600, 71-86% of the raw `paper` score is chance.
- **After calibration the k = 400-600 peak is gone (fig2).** Calibrated `paper` has a hump at
  k = 200 (0.23), its *minimum* at k = 600-800 (0.07-0.09), and its maximum at k = 1000-1023 (0.26)
  for 167 of 170 pairs. exp-005's conclusion that CKNNA peaks at intermediate k, and the
  "interpolates between local and global" reading built on it, came from the chance level.
- **`centred` gives the cleanest picture.** Its null is smaller (<= 0.20) and its calibrated
  score rises from 0.16 (k = 10) to 0.26 by k = 200, stays flat at 0.24-0.26 up to k = 800, and
  ends at 0.31 (calibrated unbiased CKA) at k = n - 1, the maximum for all 170 pairs. There is no
  intermediate-k structure beyond chance: once the neighbourhood holds ~20% of the data, the
  above-chance alignment is already at its global level.
- **Mask-first definitions are chance at large k.** For `code` and `eq36` the null mean tracks
  the raw score (`eq36`: 1.62 vs 1.62 at k = 1000; `code`: 1.23 vs 1.26 at k = 800). Only 5% of
  pairs are significant for `eq36` at k = 800 and 46% for `code`, while `centred` is significant
  for every pair at every k and `paper` for >= 93%. Their values above 1 carry no evidence of
  alignment.
- **Mutual kNN** calibrated peaks at k = 200 (0.18) and goes to 0 at k = n - 1 by construction
  (every point is everyone's neighbour). Its pair ranking at large k is unrelated to k = 10
  (Spearman 0.07 at k = 900).
- **Rankings.** Calibrated `centred` keeps a Spearman >= 0.64 with calibrated mKNN(10) at every
  k; calibrated `paper` falls to -0.11 at k = 700, where its signal is smallest relative to
  its null.
- **Families (fig3).** With `centred`, CLIP is highest and MAE lowest at every k, and DINOv2 is
  second from k = 200 on (the PRH ordering). With mKNN the families converge as the
  signal fades at large k, and with `paper` in the 600-800 dip.

**Answer to the question.** Calibration removes the intermediate-k peak. What remains, with
the properly centred CKNNA, is a nearly monotone local-to-global curve (a plateau at 0.24-0.26 for k = 200-800): neighbourhood alignment at
k = 10 (0.16) grows to the global, unbiased-CKA level (0.31) and the vision-family ordering does
not change with k. The mask-first CKNNA used in the platonic-rep and Aristotelian code should
not be used beyond k ~ 500, where they become chance-dominated and exceed 1.

## Notes

- The self-terminate step in `run_exp007_exp008.sh` failed on this image (`runpodctl config`:
  `.runpod.yaml` not found), so the pod was terminated by hand after the job finished.
