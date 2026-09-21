# PRH replication

Bounded reimplementation of Platonic Representation Hypothesis (PRH) alignment measurements on COCO val2017, plus documented kernel extensions. **Experimental scope is frozen.** Do not add models, layers, or new sweeps.

## Start here

1. `FREEZE.md` — what the frozen release-alignment study can claim, and where artifacts live.
2. `docs/CODEBASE_AUDIT.md` — source layout, historical vs authoritative runners, leftover scientific notes.
3. This file — which command to run (usually: none).

## Authoritative frozen results

Snapshot: **`prh-release-alignment-freeze-20260921`** (commit `ed966f5`). Do not move that tag.

| Role | Path |
|---|---|
| Authoritative one-sided fits | `/mnt/sdb1/prh-replication-work/results/release_anisotropy_repair/` (gitignored JSON/PNG; not in HEAD after hygiene) |
| Parent run (optimisation superseded) | same work-disk tree + freeze tag |
| Checksums | `/mnt/sdb1/prh-replication-work/freeze/prh-release-alignment-freeze-20260921/` |
| Corrected report | `docs/RELEASE_ANISOTROPY_REPORT.md` |

Feature caches stay on the work disk. Do not delete them.

## Current reproduction entry point

Default: **do not rerun**. Completed `summary.json` directories skip unless you pass `--force`.

Authoritative nested-budget fitting (already complete):

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy_repair.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 0
```

Parent `scripts/run_release_anisotropy.py` is an **archive wrapper**. It will not write a different estimator into `results/release_anisotropy/`. To inspect the superseded optimiser, check out the freeze tag.

## Historical experiments

Each has its own config, runner, `results/<name>/`, and report. None of these should be mixed with the final-layer release-alignment numbers.

| Experiment | Runner | Report |
|---|---|---|
| PRH released-code protocol | `scripts/run_prh_released_code.py` | `docs/PRH_BEFORE_AFTER.md` |
| Learned RBF/RQ kernels | `scripts/run_learned_kernels.py` | `docs/LEARNED_KERNELS_REPORT.md` |
| Polynomial / Fourier mixtures | `scripts/run_flex_kernels.py` | `docs/FLEX_KERNELS_REPORT.md` |
| Two-sided anisotropic kernels (frozen layers) | `scripts/run_anisotropic_kernels.py` | `docs/ANISOTROPIC_KERNELS_REPORT.md` |
| Archive original/modern panels | `scripts/run_phase1.py` | `results/original`, `results/modern` |

## Known limitations

- Exploratory COCO gallery (already used in earlier stages).
- Final-layer release study ≠ max-over-layers PRH means.
- Direct vs contracted CKA `a` can differ by ~2×10⁻⁵ (float32 Gram path vs float64 contractions).
- Identity-start autograd can drop at `S=0` (degenerate `eigh`); repaired fitting keeps incumbents.
- Pairwise cells are dependent. Recency is observational.

See `FREEZE.md` for the claim list and `docs/CODEBASE_AUDIT.md` for deferred artifact-in-git questions.
