# exp-005: sweeping k - how mutual kNN and CKNNA meet CKA

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-005-cknna-large-k/`

## Idea

exp-004 found CKNNA at k=200 already CKA-like in magnitude. Here we sweep k over the whole range
(10 -> n-1, n = 1024) for the full val grid and overlay mutual kNN and CKA, to see how the local
and global metrics meet at the two ends.

## Which CKNNA

We use CKNNA **as published** (Huh et al. 2024, Appendix A, Eqs. 12, 16-18):

```
Kbar_ij     = K_ij - E_l[K_il]                                   centre each row over the dataset
Align(K, L) = sum_ij alpha_ij Kbar_ij Lbar_ij                    sum over mutual neighbours only
alpha_ij    = 1[j in knn_K(i) and j in knn_L(i) and i != j]
CKNNA       = Align(K, L) / sqrt(Align(K, K) Align(L, L))
```

It is bounded by 1: the numerator sums over a subset of each model's own neighbours, so
Cauchy-Schwarz bounds it by the denominators. At k = n-1 it is the row-centred CKA of Eq. 15.

The released code (`platonic-rep/metrics.py`, `AlignmentMetrics.cknna`) computes something
different. It masks the *raw* kernel first (non-neighbours set to 0) and then centres the masked
matrix inside `hsic_unbiased`. The code gives no reason; most likely the existing HSIC routine was
reused. The two nearly agree at small k (the paper's headline uses k = 10). They do not agree at
k = n - 1: the code then gives unbiased CKA and the formula the row-centred sum of Eq. 15
(exp-007). At large k, however, the inserted zeros dominate the centring whenever the features are
anisotropic (all cosines ~ c > 0), and the code's score exceeds 1. On a first pass of this
experiment it reached 1.0-2.4 at k = 700-950, and the permutation null reached the same values, so
those numbers are not alignment. This is also where the "CKNNA > 1 for MAE around k ~ 500"
observation came from: in the vision-vision setting MAE's last block has a mean cosine of 0.89.
The published formula does not have this problem. It is the only CKNNA used below. (First-pass
code: commit `e24f58d`.)

## Method

- `cknna_k_sweep.py`: for every (LLM, ViT) pair, every layer pair, and k in {10, 25, 50, 100,
  200, ..., 900, 950, 1000, 1024}, computes mutual kNN(k) and CKNNA(k) (published), plus linear
  CKA (platonic-rep) once per pair. Each metric reports the max over layer pairs (platonic-rep
  `compute_score`, q=0.95 clamp, l2 norm). k = 1024 means all off-diagonal entries (k = n-1).
  The layer x layer terms are matrix products. `tests/test_cknna_sweep.py` checks CKNNA against
  a literal implementation of Eqs. 16-18 (and that it is <= 1), and mKNN and CKA against
  `metrics.AlignmentMetrics`.
- `--vision-vision`: last-block CLS of all 17 ViTs against each other (PRH Fig. 12 setting).
- Grid: the full cached val set, 10 LLMs x 17 ViTs.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=5 nohup bash infra/runpod/run_exp005_k_sweep.sh > /workspace/results/oddharak/exp-005.log 2>&1 &"
uv run python experiments/exp-005-cknna-large-k/plot.py --root results/exp-005-cknna-large-k
```

## Results

> **Update (exp-007, exp-008):** most of the CKNNA peak at k = 400-600 below is the chance
> level of the formula (null mean 0.32-0.38 there). After permutation calibration the published
> formula has its minimum at k = 600-800, and a properly centred CKNNA rises almost
> monotonically from local alignment to unbiased CKA. The "interpolates between local and
> global" reading below should be taken from exp-008, not from the raw curve.

Run 2026-09-27, commit `12db226`, CPU pod cpu5c x8 (16 GB; two restarts with fewer workers
after out-of-memory kills, resumed from the per-pair log), all 170 pairs. Raw outputs are in
`/workspace/results/oddharak/exp-005-cknna-large-k/{grid,vision_vision}`, copied to
`results/exp-005-cknna-large-k/`. Figures come from `plot.py` (local, regenerable).

Mean over 170 pairs (10 LLMs x 17 ViTs), max over layer pairs:

| k | 10 | 100 | 200 | 400 | 500 | 600 | 800 | 1000 | 1024 (= n-1) |
|---|---|---|---|---|---|---|---|---|---|
| mutual kNN | 0.12 | 0.26 | 0.35 | 0.50 | 0.57 | 0.64 | 0.80 | 0.98 | 1.00 |
| CKNNA | 0.16 | 0.29 | 0.37 | 0.45 | 0.45 | 0.44 | 0.34 | 0.29 | 0.29 |

Linear CKA: 0.39.

- **Bounded, as it should be.** The largest CKNNA over all 170 pairs and all k is 0.53
  (vision-vision: 0.92).
- **Meets both ends (fig1).** At small k, CKNNA sits on mutual kNN (0.16 vs 0.12 at k=10) and
  ranks the pairs the same way (Spearman 1.00 with CKNNA k=10). It peaks at k = 400-600 for every
  pair (400: 59, 500: 96, 600: 15), then falls to the row-centred CKA of Eq. 15 at k = n-1 (0.29).
  That is below platonic-rep's linear CKA (0.39), which also centres columns and keeps the
  diagonal. Mutual kNN instead converges to 1, its chance level k/(n-1). So CKNNA, not mutual
  kNN, is the metric that interpolates between local and global.
- **Rankings.** The rank agreement across pairs with CKNNA at k=10 falls from 0.95 (k=100) to
  0.29 (k=500), then recovers to 0.72 (k=1000). In the middle range all pairs have similar
  scores (0.40-0.50), so the ranking is mostly noise. The CLIP > DINOv2 > others ordering at small
  k is kept up to k ~ 600 (fig2).
- **MAE (fig2, fig3).** Cross-modal, MAE is the lowest family at small k (0.11 vs 0.16 at k=10)
  and joins the others by k=500 (0.45). Vision-vision, MAE x MAE peaks at 0.86 (k=500). Nothing
  exceeds 1: the >1 values seen before came from the code's formula.

**Does this affect Huh et al.'s figures?** Their CKNNA figures use the code. Comparing the code's
CKNNA (our `prh_val` replication) with the published formula on the same pairs:

| k | 10 | 50 | 100 | 500 | 600 | 700 | 800 | 900 | 950 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| vision-vision (Fig. 12), Spearman code vs paper | 1.00 | 1.00 | 0.99 | 0.78 | 0.47 | 0.13 | -0.10 | -0.16 | -0.10 | 0.95 |
| cross-modal (Fig. 10), Spearman code vs paper | 0.99 | 0.99 | 0.97 | 0.79 | - | - | 0.41 | - | - | 0.45 |

- **Fig. 12** (Spearman correlations between metrics over vision-vision pairs, bsz 1000). The
  CKNNA rows for k <= 100 are unaffected. k = 1000 = bsz is the unmasked case (CKA) and is also
  fine. The rows for k = 500-975 do not measure what the paper defines: at k >= 700 the code's
  ranking is uncorrelated with the published formula's. The paper's formula keeps a 0.93-0.95
  correlation with mutual kNN k=10 at every k; the code's drops to -0.16. So the low correlations
  Fig. 12 shows for large-k CKNNA reflect the implementation, not the metric. The paper's
  conclusions do not rest on those rows (its vision analysis uses mutual kNN k=10).
- **Fig. 10** (cross-modal alignment trend vs k). The lines for k = 500, 800 and 1000 (< 1024,
  so masked) come from the code version. The caption's "high values of k show less conclusive
  alignment" is therefore partly an artefact of the implementation.
- Caveat: this is our replication on 17 ViTs and 10 LLMs, not their 78 vision models. The
  code-vs-formula disagreement is a property of the formulas, but we cannot reproduce their exact
  numbers.
