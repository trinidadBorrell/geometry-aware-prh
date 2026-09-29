# exp-007: four CKNNA definitions swept over k, against mutual kNN and CKA

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-007-cknna-variants/`

## Idea

"CKNNA" is written down four different ways, and they only agree at small k:

| name | source | computes | bounded by 1 |
|---|---|---|---|
| `paper` | Huh et al. 2024, App. A, Eqs. 12, 16-18, as printed (= exp-005) | centre each row, `Kr = K - row mean`; `sum(a Kr Lr) / sqrt(sum(m_K Kr^2) sum(m_L Lr^2))` | yes |
| `centred` | new: the same with the CKA centring of Eq. 14, `Kc = H K H` | `sum(a Kc Lc) / sqrt(sum(m_K Kc^2) sum(m_L Lc^2))` | yes |
| `code` | `platonic-rep/metrics.py` `AlignmentMetrics.cknna`, copied unchanged into Aristotelian (`metrics/other_metrics.py`, `experiments/layerwise_engine.py`) | mask the raw kernel, then unbiased HSIC: `hsic_u(a K, a L) / sqrt(hsic_u(m_K K, m_K K) hsic_u(m_L L, m_L L))` | no |
| `eq36` | Groger et al. 2026 (Aristotelian), App. B.2.3, Eq. 36 | mask the raw kernel, then double-centre: `<H(aK)H, H(aL)H>_F / (‖H(m_K K)H‖ ‖H(m_L L)H‖)` | no |

`m_K` is the k-NN indicator of K (self excluded), `a = m_K * m_L` the mutual-neighbour mask.

Why `centred`: PRH Appendix A is inconsistent with itself. Eq. 13 writes `Trace(Kbar Lbar)`, Eq. 15
the elementwise sum, and Eq. 12 only removes row means, so Eq. 16 at k = n is not the CKA of
Eq. 14, although the paper (and the Fig. 10 caption) says that k = |X| recovers CKA. Centring
properly first and then masking keeps the paper's structure (centre, then restrict to mutual
neighbours) and its bound, and is the most defensible reading of what was intended. Because
Eq. 17 excludes i = j, at k = n - 1 it is CKA without the diagonal terms, not CKA itself.

Two details of `code` beyond "mask first": `hsic_unbiased` multiplies K by Lᵀ, so with the
asymmetric k-NN masks its first term only counts reciprocal edges (j in knn(i) and i in knn(j),
in both models); and U-centring a masked, non-PSD matrix loses the Cauchy-Schwarz bound.

## Hypothesis

- All four agree at small k (the PRH headline uses k = 10).
- `paper` and `centred` stay <= 1 at every k; `code` and `eq36` exceed 1 at intermediate and
  large k once features are anisotropic.
- At k = n - 1, `code` is unbiased CKA and `centred` is off-diagonal CKA; `paper` is the
  row-centred sum of Eq. 15 and is lower (exp-005: 0.29 vs CKA 0.39).

## Method

- `geoprh.cknna_variants`: all four definitions, mutual kNN and linear CKA over whole layer
  stacks. Every layer x layer term is a matrix product (Frobenius inner products of flattened
  n x n matrices, and batched products over the sample index for the row and column sums of the
  masked products), so it runs on a GPU. `tests/test_cknna_variants.py` checks each variant
  against a literal float64 implementation of its definition, `code` against
  `AlignmentMetrics.cknna`, mutual kNN and CKA against platonic-rep, and that the permuted
  scores (exp-008) equal scores recomputed on permuted features.
- `sweep.py`: the full cached val grid (10 LLMs x 17 ViTs = 170 pairs), every layer pair,
  k in {10, 25, 50, 100, 200, ..., 900, 950, 1000, 1024 (= n - 1)}; linear CKA once per pair.
  metric(vision, language) as upstream; max over layer pairs. q = 0.95 clamp, l2 norm.
- `--vision-vision`: last-block CLS of the 17 ViTs against each other (PRH Fig. 12 setting).
- `plot.py`: figures and the summary table.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp007_exp008.sh > /workspace/results/oddharak/exp-007-008.log 2>&1 < /dev/null &"
uv run python experiments/exp-007-cknna-variants/plot.py --root results/exp-007-cknna-variants
```

## Results

Run 2026-09-29, commit `bcced83`, A100 80GB (~20 min, mostly loading activations from the
volume). Raw outputs are in `/workspace/results/oddharak/exp-007-cknna-variants/{grid,vision_vision}`,
copied to `results/exp-007-cknna-variants/`. Figures come from `plot.py`.

Mean over 170 pairs (10 LLMs x 17 ViTs), max over layer pairs (fig1):

| k | 10 | 100 | 200 | 400 | 500 | 600 | 800 | 1000 | 1024 (= n-1) |
|---|---|---|---|---|---|---|---|---|---|
| mutual kNN | 0.12 | 0.26 | 0.35 | 0.50 | 0.57 | 0.64 | 0.80 | 0.98 | 1.00 |
| CKNNA `paper` | 0.16 | 0.29 | 0.37 | 0.45 | 0.45 | 0.44 | 0.34 | 0.29 | 0.29 |
| CKNNA `centred` | 0.17 | 0.30 | 0.36 | 0.40 | 0.39 | 0.37 | 0.32 | 0.31 | 0.31 |
| CKNNA `code` | 0.15 | 0.27 | 0.38 | 0.60 | 0.72 | 0.87 | 1.26 | 0.90 | 0.31 |
| CKNNA `eq36` | 0.15 | 0.31 | 0.45 | 0.72 | 0.88 | 1.06 | 1.50 | 1.62 | 0.49 |

Linear CKA: 0.39 (unbiased CKA: 0.31).

- **`paper` reproduces exp-005 exactly** (same numbers at every k), an end-to-end check of the new
  engine against the old one.
- **Agreement at small k.** Up to k = 100 the four definitions give nearly the same scores
  (0.15-0.17 at k = 10) and the same ranking of the 170 pairs (Spearman >= 0.97 with `paper`).
  PRH's headline (k = 10) is therefore not affected by which definition is used.
- **Mask-first definitions exceed 1.** `code` exceeds 1 on 71% of pairs at k = 700 and on all
  pairs at k = 800-900, up to 2.54; `eq36` exceeds 1 on every pair for 600 <= k <= 1000, up to
  2.20. Their rankings decouple from the bounded definitions at intermediate k (Spearman with
  `paper` 0.06-0.40 for `eq36` at k >= 600, 0.40-0.57 for `code`), and `eq36` becomes
  uncorrelated with mutual kNN(10) and CKA (-0.19 to 0.04 at k = 500-700). Both definitions of
  "CKNNA" that the code and the Aristotelian paper use are not alignment scores at large k.
- **`centred` ends at unbiased CKA.** At k = n - 1 it equals unbiased CKA (= `code` at n - 1)
  to within 4e-4, Spearman 1.00: leaving out the diagonal (Eq. 17's i != j) removes exactly the
  bias term of the biased HSIC. So `centred` runs from mutual kNN at small k to unbiased CKA at
  k = n - 1, and is <= 0.50 everywhere. Biased CKA differs noticeably (0.39 vs 0.31, Spearman
  0.74 across pairs), which is the known diagonal bias of biased HSIC at this sample size.
- **`centred` is the most stable ranking.** Its rank agreement with mutual kNN(10) never drops
  below 0.77 over k, and with CKA below 0.72; `paper` drops to 0.32 / 0.23 at k = 500.
- **Vision-vision (fig2).** Same picture: `paper` <= 0.92 and `centred` <= 0.90; `code` and
  `eq36` reach 1.56 and 1.68; the MAE x MAE pairs are the highest (mean ~1.5 near k = 950).

Which parts of these curves are above chance is exp-008.
