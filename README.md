# PRH replication

Bounded reimplementation of Platonic Representation Hypothesis (PRH) alignment measurements on COCO val2017, plus kernel extensions. Historical result directories stay frozen. The metric-stability extension below is the one authorised new sweep; it does not add models, layers, or features.

Snapshot: **`prh-release-alignment-freeze-20260921`** (commit `ed966f5`). Do not move that tag.

## What we can claim

On this COCO gallery, final-layer features, and frozen `q=32` one-sided metric:

- Native alignment is not monotone in release recency (Qwen2.5 dip; OLMo-2 vision peak).
- Nested-budget one-sided CKA excess increases above identity at every `ρ>0` after incumbent retention. The parent conclusion that large one-sided budgets collapse to identity is **not** a geometric fact; it was an optimiser that dropped feasible smaller-budget solutions.
- Extra alignment is carried by directional `S̃`, not uniform PCA-subspace gain `μ`.
- Partner-transfer and LOPO still beat identity on average and lag partner-specific fits.
- Cross-release amplified Grams `G+` sit above a 50-draw Haar orientation reference on the pre-registered comparison set.
- Shuffled correspondence does not recover true-test CKA. Subset refits at `ρ=0.1` with frozen PCA have moderately aligned `S̃` (mean cosine 0.86); that is not unique-direction identification.

Binding limits: already-used COCO gallery; final layer only; observational releases; dependent pairwise cells; effective rank ≠ concept count; directional uniqueness unresolved.

## Artifacts

JSON, PNG, and per-fit dumps are **not** versioned. The freeze tag still contains a historical git copy; do not rewrite that tag.

| Role | Path |
|---|---|
| Authoritative one-sided fits | `/mnt/sdb1/prh-replication-work/results/release_anisotropy_repair/` |
| Parent run (optimisation superseded) | `/mnt/sdb1/prh-replication-work/results/release_anisotropy/` |
| Feature caches | `/mnt/sdb1/prh-replication-work/features/coco_val2017_final_block_pre_norm/` |
| Checksums | `/mnt/sdb1/prh-replication-work/freeze/prh-release-alignment-freeze-20260921/` |
| Frozen configs | `configs/release_anisotropy_repair.json`, `data/manifests/release_models.json` |

Do not delete feature caches or parent result directories.

## Since the 21 September freeze

Each later study has its own directory under `results/` on the work disk. Nothing here refits the frozen repair.

| Date | Experiment | Results |
|---|---|---|
| 21 Sep | Learned anisotropic weights. `diag(B)` on frozen training PCs, from repaired one-sided fits. | `results/learned_anisotropic_weights/` |
| 22 Sep | Feature-selection frequency. Top 20% of original coordinates, and ΔM heatmaps. | `results/feature_selection_frequency/` |
| 22 Sep | CKA versus version ordinal within a family. | `results/cka_vs_recency/` |
| 23 Sep | RBF/RQ kernels rescored on an 80/20 holdout of the cached gallery. | `results/learned_kernel_cka_80_20/` |
| 27 Sep | Metric stability: one-sided, shared-PC, separate, sample size, partner transfer. | `results/metric_stability/` |
| 27–28 Sep | Full-gallery versus quarter-gallery training of the one-sided metric. | `results/resampled_metric_training/` |
| 29 Sep–1 Oct | Structured factors: identity, one-sided, shared, separate, plus spectra and per-pair bars. | `results/structured_metric_factors/` |

`scripts/plot_learned_anisotropic_weights.py` reads repaired `onesided_fits` and writes figures under `/mnt/sdb1/prh-replication-work/results/learned_anisotropic_weights/`. It does not refit, extract, or overwrite freeze artifacts.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/plot_learned_anisotropic_weights.py \
  --work /mnt/sdb1/prh-replication-work
```

Primary figures use fixed ρ=0.1 with vision and language partners separate. Weights are `diag(B)` on stored training-PCA axes (not sorted eigenvalues). PC j is not identified across models.

`scripts/plot_feature_selection_frequency.py` is a second post-process of the same repaired fits: original-coordinate top-20% selection frequency and ΔM=M−I heatmaps under `/mnt/sdb1/prh-replication-work/results/feature_selection_frequency/`.

```bash
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/plot_feature_selection_frequency.py \
  --work /mnt/sdb1/prh-replication-work
```

`scripts/plot_cka_vs_recency.py` plots native test CKA against **approximate version ordinality of each pair** (mean of 1-based family ranks; VL uses the language rank only), not calendar dates.

## Reproduction

Default: **do not rerun**. Completed `summary.json` directories skip unless you pass `--force`.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy_repair.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 0
```

Parent `scripts/run_release_anisotropy.py` is an **archive wrapper**. It will not write a different estimator into `results/release_anisotropy/`. To inspect the superseded optimiser, check out the freeze tag.

`--force` on the parent archive exits 2 and does not overwrite frozen parent fits.

## Historical experiments

Do not mix these with the final-layer release-alignment numbers.

| Experiment | Runner |
|---|---|
| PRH released-code protocol | `scripts/run_prh_released_code.py` |
| Learned RBF/RQ kernels | `scripts/run_learned_kernels.py` |
| Polynomial / Fourier mixtures | `scripts/run_flex_kernels.py` |
| Two-sided anisotropic kernels (frozen layers) | `scripts/run_anisotropic_kernels.py` |
| Archive original/modern panels | `scripts/run_phase1.py` |

## Metric stability extension

Authorised after the freeze. Historical result directories are unchanged. The new run is `results/metric_stability/` on the work disk. Protocol: `configs/metric_stability_protocol.md`.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_metric_stability.py \
  --work /mnt/sdb1/prh-replication-work --stage all
```

The manifest split named `test` is exploratory held-out evaluation in this study.

## Structured metric factors

Identity, one-sided, shared-PC and separate fits for exp(S), fixed diagonal weights, and regularised nonorthogonal diagonal or block-diagonal factors. It does not rerun metric stability or resampled-quarter training. Protocol: `configs/structured_metric_factors_protocol.md`. Output: `results/structured_metric_factors/`.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_structured_metric_factors.py \
  --work /mnt/sdb1/prh-replication-work --stage all --workers 8
```

## Resampled metric training

Authorised comparison of full-gallery updates with quarter-gallery resampling for the one-sided language metric. It does not rerun metric stability. Protocol: `configs/resampled_metric_training_protocol.md`. Output: `results/resampled_metric_training/`.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_resampled_metric_training.py \
  --work /mnt/sdb1/prh-replication-work --stage all
```

## Known limitations

- Exploratory COCO gallery (already used in earlier stages).
- Final-layer release study ≠ max-over-layers PRH means.
- Direct vs contracted CKA `a` can differ by ~2×10⁻⁵ (float32 Gram path vs float64 contractions).
- Identity-start autograd can drop at `S=0` (degenerate `eigh`); repaired fitting keeps incumbents.
- Pairwise cells are dependent. Recency is observational.
