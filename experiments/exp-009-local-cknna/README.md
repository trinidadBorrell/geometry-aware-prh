# exp-009: local CKA on mutual neighbourhoods (Emily's `cknna_local`) vs the CKNNA definitions

**Status:** done (stopped at 15 of 25 pairs) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-009-local-cknna/`

## Idea

Emily proposed a local version of CKNNA (branch `emily`,
`geometry_aware_convergence/metrics.py#L252-L302`): for every point i take its mutual neighbours
S_i = N_K(i) ∩ N_L(i), compute CKA with unbiased HSIC on the Gram blocks of {i} ∪ S_i, skip
points with fewer than 4 mutual neighbours, and average the kept points. Unlike CKNNA, each
neighbourhood is centred on its own, so the anisotropy problem of the mask-first CKNNA
(exp-007/008) cannot occur, and it gives a distribution of per-point scores.

We compare it with mutual kNN, the four CKNNA definitions of exp-007 and linear CKA, add
- `local_union`: the same local CKA on {i} ∪ N_K(i) ∪ N_L(i), which keeps every point and lets
  neighbours only one model has count against the score, and
- `coverage`: the fraction of points `local_mutual` keeps,

and look at per-point distributions and permutation nulls.

## Method

- `geoprh.local_cknna`: both local metrics vectorised over points and layer pairs (five n³
  products per layer pair instead of a Python loop over points). `tests/test_local_cknna.py`
  checks `mutual` point by point against Emily's loop (with the fix below) and `union` against a
  loop; at k = n - 1 both equal unbiased CKA.
- `run.py`: 15 (LLM, ViT) pairs (bloomz-560m, bloomz-7b1, open_llama_3b × one large ViT per
  vision family; the run was stopped after 15 of the planned 25), every layer pair, 15 values of
  k. Max over layer pairs as elsewhere. At each metric's best layer pair: per-point scores
  (k = 10, 50, 200, 500, 1000) and a 200-permutation null. **This null is at a fixed layer pair**
  (chance level of the score), not the max-over-layers null of exp-008.
- `plot.py` (figures), `diagnostics.py` (the numbers in "Problems" below).
- Run 2026-09-30, commit `b0b4764`, L4. Two earlier attempts were discarded (float32, see below).

## Results (15 pairs)

Mean over pairs of the max over layer pairs:

| k | 10 | 100 | 200 | 500 | 800 | 1023 |
|---|---|---|---|---|---|---|
| mutual kNN | 0.12 | 0.25 | 0.34 | 0.57 | 0.80 | 1.00 |
| CKNNA `centred` | 0.17 | 0.29 | 0.36 | 0.38 | 0.31 | 0.30 |
| local CKA, mutual (Emily) | 0.45 | 0.29 | 0.29 | 0.30 | 0.31 | 0.30 |
| its coverage | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| local CKA, union | 0.32 | 0.30 | 0.29 | 0.30 | 0.30 | 0.30 |
| null mean, local mutual | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| null mean, local union | 0.24 | 0.07 | 0.03 | 0.01 | 0.00 | 0.00 |

Linear CKA 0.39; unbiased CKA (every metric at k = n - 1 except mKNN) 0.30.

## Problems

1. **The code does not run as written.** `sqrt` is `math.sqrt`, which returns a Python float,
   and the next call is `.item()` on it: `AttributeError` at the first kept point. Fix:
   `score_i = (hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item()`.
2. **float32 is not precise enough.** A block is a point and its nearest neighbours, all with
   cosines ~0.9, so the three unbiased-HSIC terms nearly cancel. In float32 per-point scores were
   off by up to 30 at k = 50 and exceeded 1 (a pair scored 1.65 on the first run), although the
   score is bounded by 1 in exact arithmetic (unbiased HSIC is an inner product of U-centred
   matrices). Emily's loop is float32 too. Fix: float64, or (used here) subtract
   ½(s_a + s_b) from every Gram entry, with s_a the mean similarity of a to its k neighbours;
   unbiased HSIC is exactly invariant to additive row + column effects, and this brings float32
   within 5e-5 of float64 (tested). On the corrected run, the mean at k = 200 fell from 0.48 to
   0.29.
3. **At small k it averages a handful of hand-picked points.** Only points with >= 4 mutual
   neighbours count. At k = 10 that is 3% of the points: the best layer pair's score is a mean
   over a median of **10 points out of 1024**. Its spread across layer pairs is large (sd 0.13),
   so the max over ~600 layer pairs picks a lucky one: 0.45, 1.5× unbiased CKA, and the ranking
   of the 15 pairs is unrelated to every other metric (Spearman −0.09 with mKNN, −0.12 with CKA,
   −0.28 with unbiased CKA, against ≥ 0.96 among the others).
4. **It measures agreement only where the models already agree.** Points whose neighbourhoods
   disagree are dropped rather than penalised. Toy: two models identical on one cluster of 5% of
   the points and unrelated elsewhere give local mutual **0.99** (coverage 0.05), against
   mKNN 0.06, centred CKNNA 0.27 and CKA 0.16. The score has to be read together with coverage
   (or mKNN); on its own it can call two mostly unrelated models aligned.
5. **Its null is uninformative where it differs from CKA.** Under the permutation null almost no
   point has 4 mutual neighbours at k = 10, so the null is 0 by convention (no points), not
   because chance agreement is 0.
6. **Where it is reliable, it is unbiased CKA.** From k = 50 the mean is 0.87-1.02× the pair's
   unbiased CKA and the pair ranking converges to it (Spearman 0.75 at k = 50, 0.83 at 200,
   ≥ 0.95 from 500). Per-point scores concentrate around the global value as k grows (fig3). The
   local geometry among shared neighbours agrees about as much as the global geometry does, so
   the metric adds nothing CKA does not already give.
7. **The union variant trades selection for chance.** Keeping every point removes problem 3-4,
   but at small k its null is 0.24 of an observed 0.32: a block made of i's neighbours in K and
   i's neighbours in L has built-in shared structure (each model sees its own group as tight),
   which unbiased HSIC reads as alignment. From k = 100 it also equals unbiased CKA.
8. **Cost.** Five n³ products per layer pair and k (vs n² for CKNNA): ~10× the cost of all
   CKNNA definitions together, and the original loop would be orders of magnitude slower.

## Is it worth pursuing?

Not as an alignment score. In the regime where it is numerically and statistically sound
(k >= 50) it reproduces unbiased CKA; in the regime where it differs (k ~ 10) the difference is
selection (3% of points kept, max over layer pairs) and noise, and by construction it ignores
the disagreement that mKNN measures. The centred CKNNA (exp-007/008) already covers the local to
global range with a bound, a well-behaved null and n² cost.

What could be kept: the per-point view. Per-point scores could be used as a diagnostic (where in
the data do two models agree?), provided (i) blocks are large enough to be stable (k >= 50),
(ii) the computation is float64 or shifted, and (iii) the score is always reported with coverage,
or replaced by a neighbourhood definition that does not condition on agreement and has a
calibrated null. The row-wise centred CKNNA gives such per-point scores at n² cost (fig3) and
would be the simpler starting point.

Caveats: 15 pairs from three LLMs up to 7B; the nulls here are at a fixed layer pair, not
calibrated max-over-layers p-values.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp009.sh > /workspace/results/oddharak/exp-009.log 2>&1 < /dev/null &"
uv run python experiments/exp-009-local-cknna/plot.py --root results/exp-009-local-cknna
uv run python experiments/exp-009-local-cknna/diagnostics.py --root results/exp-009-local-cknna
```
