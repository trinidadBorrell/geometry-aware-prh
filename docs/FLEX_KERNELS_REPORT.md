# Flexible polynomial and spectral kernels — report

**Status:** complete on the existing 18 COCO vision–language pairs. Exploratory evaluation on the audited test gallery (n=1024). Directories `results/prh_released_code` and `results/learned_kernels` were not modified.

**Question.** Do flexible, regularised kernels reveal reproducible correspondence beyond linear CKA, and do model pairs prefer distinct similarity functions that transfer selectively?

**Answer in one paragraph.** On this frozen panel the primary objective (centred CKA ratio a/b) returns the exact linear kernel for every pair and every family. Validation-selected polynomial mixtures are also linear under the excess objective. Linear-plus-Fourier and spherical-plus-Fourier mixtures can pick a small nonlinear mass under excess (mean 5–6%), raising held-out CKA a from 0.508 to 0.511 — the same scale as the previous fitted-RBF excess result, and still below the unfitted official RBF σ=1 score (0.522). That increment does not change mutual kNN, neighbour order versus linear, or shuffled-fitting scores (shuffle-fit test a typically 0.02–0.18, with occasional high-a / ratio≈1 degeneracies). Pair-specific nonlinear coefficients are redundant (basis Gram condition numbers ~10¹⁶), unstable as geometry, and do not beat a common linear kernel by a scientifically useful margin. Negative result: richer valid dictionaries do not overturn the RBF-grid conclusion.

## Lead answers

1. **Which family, if any, improves over linear on held-out data?** Ratio-selected: none (all a=0.50766, identical to linear). Excess-selected: monomial and spherical do not; `lin_fourier` and `sph_fourier` raise mean test a by +0.0028.
2. **Does regularisation preserve that improvement?** Validation usually prefers nonlinear-mass shrinkage (`nl`) or `unreg`, never `moderate`/`strong` on real pairs. Shrinkage **removes** the unregularised polynomial excess “gain” (unreg a≈0.513 → selected a=0.508). For Fourier families it **keeps a sliver** (unreg a≈0.514–0.515 → selected 0.5105).
3. **Is improvement meaningful in absolute CKA as well as the ratio?** The only nonzero selected gain is in a (and a tiny excess). The ratio **falls** (11.67 → 10.74) because b rises with the spectral mass. Official CKA with +1e-6 agrees with a to ~10⁻¹⁰ on linear Grams.
4. **Do shuffled-fitting controls show comparable gains?** No. Full coefficient + regulariser search on independently shuffled train/val, evaluated on shuffled test (3 prespecified pairs × 2 seeds × 4 families): mean test a ≈ 0.07–0.12. Actual-correspondence linear a on those pairs is ~0.47–0.52. Occasional shuffle draws put all mass off linear and inflate a while driving a/b → 1 (degeneracy), which is not the selected real-data behaviour.
5. **Are nonlinear functions identifiable and stable?** Ratio: nonlinear mass is identically 0; three training subsets of 682 also return mass 0. Excess Fourier mass is small and pair-dependent; dictionaries are extremely collinear (median max off-diagonal basis correlation 0.9998). Do not read coefficients as distinct geometry.
6. **Does pair-specific fitting outperform a common kernel?** A common coefficient vector (equal pair weights, moderate penalty) is the linear vertex (mean test a=0.50765). Pair-specific excess Fourier is +0.0028 on a. Leave-one-pair-out common kernels are linear (mean a=0.50765). The gap is not a useful pair-specific kernel.
7. **Do functional differences predict transfer differences?** Ratio transfer is tautological (everyone is linear). Excess `sph_fourier` transfer a is 0.5105 on the same pair and 0.5098 when sharing neither model (fitted−transfer gap ~7×10⁻⁴). Share-class cells are dependent; no generation or scaling claim.
8. **Are polynomial and Fourier families genuinely different on these embeddings?** As **selected** kernels under ratio, no. As **dictionaries**, Fourier shells at d∈{384,…,2048} match the Gaussian limit max |κ−exp(−ν²r²/2)| ≲ 3×10⁻⁴ over the verification grid: they are a **multiscale RBF mixture**, not a new angular geometry. Monomial t^p and Gegenbauer Z_{p,d} are numerically interchangeable here (same selected linear vertex; similar unreg excess a). Shared harmonic-order weights are **not** the same scalar polynomial in different dimensions; they simply never left the degree-1 term after validation.

## Setup (frozen before eval)

Config: `configs/flex_kernels.json`. Design: `docs/FLEX_KERNELS_DESIGN.md`. Implementation: `src/prh_replication/flex_kernels.py`, `scripts/run_flex_kernels.py`. Tests: `tests/test_flex_kernels.py` (17 passed on the Ubuntu venv with the rest of the suite).

- Representations, layers, train median distance scales s, manifests, and splits: reused from `results/learned_kernels` (train-max linear CKA layers; PRH clip q=0.95 then L2). No reselection, no downloads, no pooling change.
- Families (simplex, nonnegative, sum 1; linear included once; unit diagonal before centring; no eigenvalue clipping):
  - monomial: k=Σ_{p=1}^{12} w_p t^p
  - spherical: shared order weights on Z_{p,d_A} and Z_{p,d_B} (normalised Gegenbauer; p=0 excluded)
  - lin_fourier: linear + 12 isotropic spectral shells
  - sph_fourier: Z_{1:12} + 12 shells
- Frozen ν grid (log from 0.1 to 8, last entry snapped to 8): see config. d_ref=1024, s_ref=1.0 for canonical t-domain plots only.
- κ_{ν,d}(r)=₀F₁(;d/2; −d ν² r²/4) via power series for |z|<80, else the Gaussian high-d limit. Deterministic; no random Fourier features.
- Penalty: R=λ_nl(1−c_lin)+λ_complexity(poly degree cost and/or Fourier ν/ν_max cost)+λ_smooth Σ(Δc^Fourier)². Polynomial families omit Fourier terms; lin_fourier omits poly degree cost.
- Regimes: unreg / nl / moderate / strong. Train maximises J(c)/J(linear)−R; val maximises unpenalised J; test unused for selection. Families reported separately.
- Optimiser: projected finite differences, 180 steps, six starts (linear vertex, uniform, best vertex, three Dirichlet). Allow boundary solutions.
- Contracted stats ⟨Kc,Lc⟩=cᵀMc, ||Kc||_F²=cᵀAc, traces linear in c. Eval rebuilds mixed Grams for diagnostics, mNN, and official CKA.
- Evaluation: exploratory test n=1024, 100 permutations. **Do not compare raw a/b across train vs test** (n−1 factor).

Observed embedding sizes all d>2 (language 896–2048, vision 384 or 768). Gegenbauer and spectral shells are defined.

## Mean exploratory-test scores (18 pairs)

| Kernel | a | a/b | a−b | nl mass | mNN k=10 |
|---|---|---|---|---|---|
| Linear | 0.50766 | 11.673 | 0.45530 | 0 | 0.1754 |
| Official RBF σ=1 | 0.52195 | — | — | — | — |
| Previous RBF excess-selected (same reps) | 0.512 | 10.07 | 0.455 | — | — |
| Any family, **ratio**-selected | 0.50766 | 11.673 | 0.45530 | 0 | 0.1754 |
| Monomial / spherical **excess**-selected | 0.50766 | 11.673 | 0.45530 | 0 | 0.1754 |
| lin_fourier **excess**-selected | 0.51050 | 10.743 | 0.45576 | 0.056 | 0.1754 |
| sph_fourier **excess**-selected | 0.51051 | 10.742 | 0.45576 | 0.053 | 0.1754 |
| Unreg excess monomial (not val-selected) | 0.51311 | — | — | 0.083 mean | — |
| Unreg excess sph_fourier (not val-selected) | 0.51526 | — | — | 0.460 mean | — |
| Best single Fourier vertex (excess, train) | 0.51230 | — | — | 1 at that ν | — |

kNN overlap with the linear baseline is 1.0 for every selected kernel. Linear centred diagonal-energy fraction 0.0485; effective participation rank 47.8. Excess Fourier: 0.0510 and 50.4. Unbiased U-centred CKA 0.483 → 0.485. Neighbour order is unchanged at the selected solutions.

Ratio always selected `unreg` (all regimes already linear). Excess selected `nl` 16/18 (polynomials), 9/18 (`lin_fourier`), 13/18 (`sph_fourier`); remainder `unreg`.

## Verification and redundancy

Focused checks in `tests/test_flex_kernels.py` and `results/flex_kernels/verification.json`:

- Degree-1 monomial and Z_1 equal the linear kernel; Z_p(1)=1; recurrence vs independent step.
- Unit diagonal, symmetry, small-fixture PSD within tolerance.
- Contracted CKA vs direct centred Grams.
- κ(0)=1; Fourier vs Gaussian limit on the frozen ν grid: max abs error 2.9×10⁻⁴ (d=384), 2.0×10⁻⁵ (d=1024), 2.7×10⁻⁶ (d=2048). High-|z| series fallback is the Gaussian, so mid/high ν on large d **are** RBFs by construction when the series is skipped.
- All model d>2; no special-case path used in the run.

Basis products are ill-conditioned (median cond(A) ~10¹⁶). Coefficient vectors in a redundant dictionary are not identifiable kernels.

Canonical profile comparison (fixed d_ref=1024, s_ref=1, t∈[−1,1], r=√(2−2t)) is a **convention**, not a data inner product. Figures: `results/flex_kernels/figures/`. On actual target representations, selected ratio kernels are the linear Gram; selected excess Fourier kernels stay close to linear (knn overlap 1).

## Stability, shuffle, transfer

- **Subsets:** three deterministic 682-point training draws, frozen selected regulariser from the full train set. Nonlinear mass 0 on the first three pairs for all families/objectives — coefficient fitting, given the val-chosen shrinkage, returns the linear vertex.
- **Shuffle-fit:** includes the regulariser search. Polynomial and spherical families often stay linear with test a matching the shuffled linear baseline (~0.02–0.07). Some seeds put nl mass 0.5–1.0 and raise a to ~0.15–0.21 with ratio≈1. One `lin_fourier` excess shuffle (`qwen3-0.6b-base`/`clip-laion-base`, seed 0) reached a=0.40 with nl mass 1 and ratio≈1. That is not a transferred correspondence kernel; it is the same degeneracy absolute CKA prefers.
- **Transfer:** ratio gaps ~10⁻⁵. Excess `sph_fourier` neither-model mean a=0.5098 vs same-pair 0.5105. LOPO excludes a **pair**, not its models; those kernels are linear.

## What this does not show

- No mNN improvement; do not treat the +0.003 CKA bump as local neighbour agreement.
- No claim that a wider ν grid, signed (non-simplex) series, or final-layer-only features would help.
- No scaling or generation convergence.
- Evaluation gallery remains **exploratory** (no unused 1024-image confirmatory split).

## Reproduction

On Ubuntu, after rsyncing this repo to `/mnt/sdb1/prh-replication-work/repo`:

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
export HF_HOME=/mnt/sdb1/prh-replication-work/hf
cd /mnt/sdb1/prh-replication-work/repo
/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests -q
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_flex_kernels.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 100
# Rewrite shuffle_fit.json for all four families without refitting pairs:
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_flex_kernels.py \
  --work /mnt/sdb1/prh-replication-work --shuffle-only --n-perm 100
```

Smoke (1 pair, P=4, 4 frequencies, 2 regimes): add `--smoke`. Outputs: `/mnt/sdb1/prh-replication-work/results/flex_kernels/` and `results/flex_kernels/` in the git workspace. Do not overwrite `results/prh_released_code` or `results/learned_kernels`.
