# exp-004: neighbourhood size k — does local similarity become global?

**Status:** running · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-004-knn-k-sweep/`

## Idea

Gröger et al. Fig. 26 sweeps the neighbourhood size of mutual kNN over k ∈ {10, 20, 50, 100} and
reports that alignment stays significant after calibration while "the scaling trend is clearest at
small k and flattens at large k". We push past their largest k (here k = 200, with n = 1024
samples) and ask what happens to the calibrated signal when the neighbourhood stops being local.

## Hypothesis

As k grows, a neighbourhood covers a larger part of the dataset, so the permutation null grows
too: chance overlap of two random k-sets is ≈ k/n. Two competing readings:

1. *(the original idea)* larger k probes more of the space, so mutual kNN starts to behave like a
   global metric, and its calibration loss should approach that of linear CKA (~32%).
2. *(the objection)* mutual kNN keeps only set membership and discards distances, so as k → n it
   saturates at 1 and carries no geometry. The metric that genuinely interpolates local → global
   is CKNNA, which weights by centred kernel values and provably recovers CKA at k = n
   (Huh et al., Appendix A, Eq. 18–19). Then mutual kNN's calibrated score should *collapse*
   toward 0 at large k rather than imitate CKA.

**What would falsify (2):** at k = 200, mutual kNN keeps a calibrated signal comparable to k = 10
in normalised terms, and its vision-model ranking starts to reorder the way CKA's does.

Because the two metrics live on different scales at different k (the null mean moves with k), the
comparison uses normalised quantities — excess over the calibrated threshold τ and the z-score
(raw − μ₀)/σ₀ — alongside the paper-style gated score.

## Method

- Metrics: mutual kNN (k = 200) and CKNNA (k = 200); CKNNA is the local→global bridge, with the
  k = 10 results of exp-003 and linear CKA as the two reference points.
- 200 permutations, α = 0.05, q = 0.95 outlier clamp, max over layer pairs — identical to
  exp-003, so the numbers are directly comparable.
- Reduced grid (40 pairs): 5 LLMs spanning 0.56B → 13B across three families, 8 ViTs covering
  ImageNet21K / MAE / DINOv2 / CLIP at two sizes each.
- Mutual kNN calibration uses the mask-based variant, whose cost does not depend on k
  (the index-based one would allocate an n×k×k tensor). `tests/test_prh_alignment.py` checks the
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

_pending_

## Next

- If mutual kNN collapses at k = 200 while CKNNA tracks CKA, sweep CKNNA across the full k range
  to locate where local turns into global.
- n = 1024 bounds this: at k = 200 the neighbourhood is already a fifth of the dataset. A larger
  WIT subset would separate "large neighbourhood" from "most of the data".
