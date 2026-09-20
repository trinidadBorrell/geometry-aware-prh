# Learned-kernel extension — report

**Status:** complete on the existing 18 V–L pairs. Exploratory evaluation on the audited COCO test gallery (n=1024). PRH baselines in `results/prh_released_code` were not modified.

**Question.** Do model pairs prefer stable, nondegenerate kernel functions that reveal correspondence beyond shuffled controls, and do those functions resemble one another and transfer?

**Answer in one paragraph.** Maximising ordinary CKA always selected the narrowest grid RBF (λ=0.1), producing near-identity Grams with CKA≈1 and shuffle ratio≈1 — a degenerate fit that ignores correspondence. The ratio a/b always selected the widest grid RBF (λ=10), which is essentially the linear kernel on these L2-normalised features (test a=0.508 vs linear 0.508). Excess a−b was the only objective with interior, pair-dependent λ. Those ratio “functions” are identical across pairs because every pair hit the same boundary, so profile similarity and transfer look perfect for a trivial reason. A confirmatory unused 1024-image gallery does not exist (904 leftovers). Evaluation is labelled **exploratory**.

## Setup (frozen before eval)

Config: `configs/learned_kernels.json`. Design: `docs/LEARNED_KERNELS_DESIGN.md`.

- Representations: PRH clip q=0.95 then L2; layers = max **train** extension linear CKA; frozen for all kernels.
- s_A, s_B: train median off-diagonal Euclidean distances. Shared λ acts on r=||x−x′||/s, not on raw σ.
- Grid: 25 λ ∈ [0.1, 10] including 1; 9 α ∈ [0.1, 100]. Not expanded after seeing eval.
- Estimator: a=<Kc,Lc>_F / (||Kc||_F ||Lc||_F) without the official +1e-6. Analytic b for unrestricted one-sided permutations. Official CKA with +1e-6 stored alongside. MC vs analytic |b−mean| ≈ 8e-5 on test (100 perms).
- Hook check (8 images × 3 ViTs): block-output CLS vs FX `blocks.{i}.add_1` **max abs diff 0**. No re-extract.

Train n=2048, val/test n=1024. **Do not compare raw a/b across train vs test** without noting n: mean linear ratio is 21.3 on train and 11.7 on test from the (n−1) factor, not from overfitting.

## 1. Do the three objectives select different kernels?

Yes, at the grid boundary.

| Objective | RBF λ (all 18 pairs) | RQ (λ, α) | Boundary |
|---|---|---|---|
| a (CKA) | always 0.1 | always (0.1, 100) | 18/18 |
| a/b (ratio) | always 10 | always (10, 100) | 18/18 |
| a−b (excess) | median 1.47, 9 distinct values, 3/18 at 10 | mixed | 3/18 RBF, 11/18 RQ |

On exploratory test (means):

| Kernel | a | b | a/b | a−b |
|---|---|---|---|---|
| Linear | 0.508 | 0.052 | 11.67 | 0.455 |
| RBF σ=1 (raw) | 0.522 | 0.077 | 8.59 | 0.445 |
| RBF λ=1 | 0.524 | 0.077 | 8.10 | 0.447 |
| RBF CKA-selected (λ=0.1) | 0.998 | 0.998 | 1.000 | ≈0 |
| RBF ratio-selected (λ=10) | 0.508 | 0.053 | 11.63 | 0.455 |
| RBF excess-selected | 0.512 | 0.057 | 10.07 | 0.455 |

Ratio optimisation does **not** beat linear CKA on held-out a or a/b. It refuses the degenerate Dirac-like kernel that absolute CKA prefers. Excess sits between λ=1 and the wide boundary and slightly raises a (0.512) while lowering the ratio (10.07).

## 2. Do gains generalise?

Val and test track each other for nondegenerate kernels (ratio a: val 0.510, test 0.508; linear a: 0.510 / 0.508). Absolute-CKA “gains” to a≈1 also appear on val and test: they generalise the **degeneracy**, not correspondence.

Relative to linear on the same frozen layers, ratio Δa = +0.00018 (max +0.00023). That is not a material held-out gain.

## 3. Nondegenerate and reproducible?

**CKA-selected RBF:** centred Gram diagonal energy 0.993; off-diagonal kernel median ~10⁻²²; shuffle ratio 1. High nonnegative effective rank (~1016 of n=1024) because a near-identity matrix is full rank after centering — this is **not** a single shared direction, and it is still degenerate as a correspondence kernel.

**Ratio-selected RBF:** diagonal energy 0.049; off-diagonal median 0.995; effective rank ~48; U-centred CKA (unbiased HSIC, not fitted) ~0.50–0.65 where defined. Reproducible: all three train subsets of 682 picked λ=10.

**Excess:** subset λ std mean 0.29; 12/18 pairs identical across subsets. More identifiable than ratio’s flat wide-kernel region, but several fits sit on λ=10.

Shuffle-fit control (3 pairs × 2 seeds; **resolution: 6 draws**): after permuting train correspondence, CKA still picks λ=0.1 and test a≈0.999, ratio≈1. Ratio sometimes collapses to a narrow kernel (test ratio≈1) and sometimes to λ=10 with test a≈0.10–0.12, i.e. no usable correspondence. Fitting on real labels is required for the nondegenerate wide kernel; the control is coarse.

## 4. Are functions similar beyond fitting uncertainty?

For ratio and CKA, every pair selected the **same** θ, so centred profile inner products under ν=U[0,3] are identically 1. That is grid-boundary collapse, not evidence that distinct radial functions agree. Training-distance mass outside [0,3] is 0 (features are L2-normalised; s≈0.8–1.05).

Excess kernels do vary; they were not forced to a single profile. Between-pair excess transfer gaps are small (~10⁻³) relative to a itself.

Mean-centring a radial profile under ν is **not** Gram double-centring.

## 5. Transfer

For CKA and ratio, source θ is constant, so transfer equals the target-fitted kernel for every share class (same pair / shared LM / shared ViT / neither). That is not generalisation to unseen models.

Excess RBF: neither-model cells have mean transfer a=0.513 vs target-fitted 0.512 (gap defined as fitted minus transfer; **negative allowed**; mean gap −0.001). Descriptive only; cells are dependent (repeated models).

## What this does not show

- No mNN improvement (strictly decreasing radial kernels preserve neighbour order).
- No scaling law or generation convergence (short ladders only).
- No claim that λ>10 would help; the ratio optimum is a **boundary** of the frozen grid.
- No numerical recovery of PRH WIT Figure 3.

## Reproduction

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
export HF_HOME=/mnt/sdb1/prh-replication-work/hf
cd /mnt/sdb1/prh-replication-work/repo
/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests -q
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/check_vision_fx_hooks.py --work /mnt/sdb1/prh-replication-work
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_learned_kernels.py --work /mnt/sdb1/prh-replication-work --skip-hooks --n-perm 100
```

Smoke (1 pair, thinned grid): add `--smoke`. Outputs: `/mnt/sdb1/prh-replication-work/results/learned_kernels/` and `results/learned_kernels/` in the repo.
