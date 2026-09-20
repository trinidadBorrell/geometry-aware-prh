# Freeze: `prh-release-alignment-freeze-20260921`

Final bounded pass on the release-alignment experiment. **No further sweep.** Negative and mixed controls are part of the frozen record.

## What we can claim

On this COCO gallery, final-layer features, and frozen `q=32` one-sided metric:

- Native alignment is not monotone in release recency (Qwen2.5 dip; OLMo-2 vision peak).
- Nested-budget one-sided CKA excess increases above identity at every `ρ>0` after incumbent retention. The parent conclusion that large one-sided budgets collapse to identity is **not** a geometric fact; it was an optimiser that dropped feasible smaller-budget solutions.
- Extra alignment is carried by directional `S̃`, not uniform PCA-subspace gain `μ`.
- Partner-transfer and LOPO still beat identity on average and lag partner-specific fits.
- Cross-release amplified Grams `G+` sit above a 50-draw Haar orientation reference on the pre-registered comparison set.
- Shuffled correspondence does not recover true-test CKA. Subset refits at `ρ=0.1` with frozen PCA have moderately aligned `S̃` (mean cosine 0.86); that is not unique-direction identification.

Limitations (binding): already-used COCO gallery; final layer only; observational releases; dependent pairwise cells; effective rank ≠ concept count; directional uniqueness unresolved.

## Artifact locations

| Item | Location | Status |
|---|---|---|
| Authoritative repair outputs | `/mnt/sdb1/prh-replication-work/results/release_anisotropy_repair/` | authoritative |
| Fitted `S` / `B` (360 files) | `.../release_anisotropy_repair/onesided_fits/` (work disk only) | authoritative |
| Repo mirror (JSON/PNG, no fit matrices) | `results/release_anisotropy_repair/` | copy |
| Parent run (do not delete) | `/mnt/sdb1/prh-replication-work/results/release_anisotropy/` and `results/release_anisotropy/` | historical; **optimisation-dependent files superseded** |
| Native / two-sided | parent `native_matrix.json`, `native_anisotropy.json`, `two_sided.json` (checksummed in repair `native_reuse.json`) | reused |
| Feature caches | `/mnt/sdb1/prh-replication-work/features/coco_val2017_final_block_pre_norm/` | reused; SHA-256 in `feature_hashes.json` |
| Checksums | `results/freeze/prh-release-alignment-freeze-20260921/SHA256SUMS.txt` and `INVENTORY.json` (full copy also under `/mnt/sdb1/prh-replication-work/freeze/prh-release-alignment-freeze-20260921/`) | freeze |
| Report | `docs/RELEASE_ANISOTROPY_REPORT.md` | corrected |
| Design | `configs/release_anisotropy_repair.json`, `data/manifests/release_models.json` | frozen |

## Software (Ubuntu repair process)

- Python 3.12.3, numpy 2.2.0, torch 2.10.0+cu128, transformers 5.3.0
- Host: Linux 6.14.0-24-generic x86_64
- Work venv: `/mnt/sdb1/prh-replication-work/venv`
- Command: `python scripts/run_release_anisotropy_repair.py --work /mnt/sdb1/prh-replication-work --n-perm 0` (tail completed with `--resume-after lopo`)
- `git_head` in the run’s `software_versions.json` is null because the work-disk copy is not a git repo; the freeze tag is on the laptop workspace.

## Verification

- `tests/test_release_anisotropy.py`: 12 passed on the work venv.
- Train monotonicity failures: 0/360.
- `D≤ρ` and `ρ=0` identity: true.
- Parent `ρ=0.1` fit re-eval: identical train excess at all larger `ρ`.
- PCA variance vs parent: abs_diff 0.
- Direct vs contracted `a`: ~2×10⁻⁵ on three sampled pairs (marked in `checks.json`; not expanded).
- Shuffle: 2/12 identity, mean true-test Δ`a` −0.027 (kept).
- No background experiment jobs left. vLLM was not changed.

## Resume (do not start casually)

Rerunning a **completed** result needs an explicit flag:

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
export HF_HOME=/mnt/sdb1/prh-replication-work/hf
cd /mnt/sdb1/prh-replication-work/repo
# Parent (historical):
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy.py --work /mnt/sdb1/prh-replication-work --force
# Repair:
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy_repair.py --work /mnt/sdb1/prh-replication-work --n-perm 0 --force
```

Default invocation exits if `summary.json` already exists.

Sync code from the laptop with `rsync -e 'ssh -F /dev/null -i ~/.ssh/id_ed25519_cursor'`. Do not overwrite feature caches or parent result directories.
