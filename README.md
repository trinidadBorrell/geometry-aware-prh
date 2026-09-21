# PRH replication

Bounded reimplementation of Platonic Representation Hypothesis (PRH) alignment measurements on COCO val2017, plus kernel extensions. **Experimental scope is frozen.** Do not add models, layers, or new sweeps.

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

## Known limitations

- Exploratory COCO gallery (already used in earlier stages).
- Final-layer release study ≠ max-over-layers PRH means.
- Direct vs contracted CKA `a` can differ by ~2×10⁻⁵ (float32 Gram path vs float64 contractions).
- Identity-start autograd can drop at `S=0` (degenerate `eigh`); repaired fitting keeps incumbents.
- Pairwise cells are dependent. Recency is observational.
