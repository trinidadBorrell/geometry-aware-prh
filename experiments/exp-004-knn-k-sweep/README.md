# exp-004: neighbourhood size k - does local similarity become global?

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-004-knn-k-sweep/`

## Idea

Groger et al. Fig. 26 sweeps the neighbourhood size of mutual kNN over k in {10, 20, 50, 100} and
reports that alignment stays significant after calibration while "the scaling trend is clearest at
small k and flattens at large k". We push past their largest k (here k = 200, with n = 1024
samples) and ask what happens to the calibrated signal when the neighbourhood stops being local.

## Hypothesis

As k grows a neighbourhood covers a larger part of the dataset, so the permutation null grows too:
chance overlap of two random k-sets is about k/n. Two competing readings:

1. *(the original idea)* larger k probes more of the space, so mutual kNN starts to behave like a
   global metric, and its calibration loss should approach that of linear CKA (~32%).
2. *(the objection)* mutual kNN keeps only set membership and discards distances, so as k -> n it
   saturates at 1 and carries no geometry. The metric that genuinely interpolates local -> global
   is CKNNA, which weights by centred kernel values and provably recovers CKA at k = n
   (Huh et al., Appendix A, Eq. 18-19). Then mutual kNN's calibrated score should *collapse*
   toward 0 at large k rather than imitate CKA.

**What would falsify (2):** at k = 200 mutual kNN keeps a calibrated signal comparable to k = 10 in
normalised terms, and its vision-model ranking starts to reorder the way CKA's does.

The metrics live on different scales at different k (the null mean moves with k), so the comparison
uses normalised quantities - the null threshold tau and the z-score (raw - mu0)/sd0 - alongside the
paper-style gated score.

## Method

- Metrics: mutual kNN (k = 200) and CKNNA (k = 200); CKNNA is the local->global bridge, with the
  k = 10 results of exp-003 and linear CKA as the two reference points.
- 200 permutations, alpha = 0.05, q = 0.95 outlier clamp, max over layer pairs - identical to
  exp-003, so the numbers are directly comparable.
- Reduced grid (40 pairs): 5 LLMs spanning 0.56B to 13B across three families, 8 ViTs covering
  ImageNet21K / MAE / DINOv2 / CLIP at two sizes each.
- Mutual kNN calibration uses the mask-based variant, whose cost does not depend on k (the
  index-based one would allocate an n x k x k tensor). `tests/test_prh_calibration.py` checks the
  two agree at k = 10.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp004_k_sweep.sh > /workspace/results/oddharak/exp-004.log 2>&1 &"
```

Figures come from Aristotelian's own plotting, unchanged:

```bash
cd Aristotelian && uv run python -m scripts.plots.experiments \
    --sections prh_alignment --assets-dir /workspace/results/oddharak/exp-004-knn-k-sweep
```

## Results

Run 2026-09-20, commit `5c1b36f`, RTX PRO 4500, ~20 min. Outputs (payloads, 14 PDFs from
Aristotelian's plotting, per-pair JSONL) are in
`/workspace/results/oddharak/exp-004-knn-k-sweep/`, copied to `results/exp-004-knn-k-sweep/`.
Compared against exp-003 (mKNN k=10, linear CKA) on the same 40 pairs.

| metric | raw | null tau | calibrated | loss | z = (raw-mu0)/sd0 | scaling trend after calibration |
|---|---|---|---|---|---|---|
| mKNN k=10 | 0.123 | 0.014 | 0.111 | 11% | 164 | +0.87 |
| mKNN k=200 | 0.349 | 0.207 | 0.180 | 49% | 63 | +0.65 |
| CKNNA k=200 | 0.385 | 0.168 | 0.261 | 33% | 107 | +0.85 |
| linear CKA | 0.387 | 0.157 | 0.275 | 29% | 200 | +0.50 |

(means over the 40 pairs; scaling trend = Spearman of score vs log LLM size, averaged over ViTs)

**The prediction held.** Raw mutual kNN rises 0.123 -> 0.349 and the calibration loss rises from
11% to 49%. The null threshold tau = 0.207 is almost exactly k/n = 200/1024 = 0.195, the chance
overlap of two random 200-sets, so most of the extra raw score is chance.

**But mutual kNN does not become CKA.** Its loss (49%) overshoots CKA's (29%) while its evidence
against the null collapses (z from 164 to 63, the weakest of the four), and the scaling trend
decays (+0.87 to +0.65), continuing the flattening Groger et al. report up to k=100. Large-k mutual
kNN is not "global", it is noisy: it keeps set membership and discards distances, so the
neighbourhood stops carrying geometry. Hypothesis (2) is supported, (1) is not.

**CKNNA at k=200 is already essentially CKA.** tau 0.168 vs CKA's 0.157, calibrated 0.261 vs 0.275:
with n=1024 the kernel-weighted local metric has converged to the global one by k=200, as Huh et
al.'s k -> n argument predicts.

**The interesting asymmetry.** CKNNA keeps the scaling trend after calibration (+0.85, close to
mKNN k=10's +0.87) while CKA loses half of it (+0.50), even though the two have nearly identical
nulls and calibrated magnitudes. Restricting the kernel to neighbours preserves *which models align
better* even when the magnitude has gone global. That is a geometry effect rather than a magnitude
effect, and it is the most promising thread here.

Caveats: 40 pairs only; the CLIP (INet ft) panel is empty because the reduced grid omits those
models; every p-value sits at the 0.005 floor of 200 permutations, so significance cannot be
compared across metrics, only magnitude and trend.

## Next

- Sweep CKNNA across the full k range (10 -> 1000) to locate where local turns into global, and
  check whether the scaling trend survives all the way to k = n (where CKNNA is CKA by
  construction). That tension is the core question.
- n = 1024 bounds this: at k = 200 the neighbourhood is already a fifth of the dataset. A larger
  WIT subset would separate "large neighbourhood" from "most of the data".
- Re-run on the full 10 x 17 grid if the effect holds, for comparability with exp-003.
