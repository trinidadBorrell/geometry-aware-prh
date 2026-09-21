# Codebase audit (post-freeze hygiene)

Cleanup commit is **not** the freeze. Frozen science remains `prh-release-alignment-freeze-20260921` (`ed966f5182f5e1fe434d34c2165494d93c439c41`). Backup of that starting tree: branch `backup/pre-code-hygiene-ed966f5`. The freeze tag was not moved.

## Diff baseline

`origin/main` on `https://github.com/trinidadBorrell/geometry-aware-prh.git` is essentially empty (`.gitignore` + `README.md`). Local history is a **single root commit** (the freeze). There is no meaningful `main...HEAD` merge-base.

**Chosen baseline for this audit:** `origin/main` as the review remote; **implementation delta** measured against the freeze commit for this cleanup.

Untracked local result dumps (`results/flex_kernels/cache/*.npz`, `results/anisotropic_kernels/pca.json`, etc.) were left on disk and not incorporated.

## What made “the diff” large

Against `origin/main`, freeze-era tracked files were ~319k insertions: **~9k Python**, **~308k JSON/PNG**. The review problem is primarily **generated results committed with the freeze**, then **duplicated runners**, not a missing framework.

| Category | Files (freeze) | Approx. lines / bytes | Purpose |
|---|---:|---:|---|
| Core scientific implementation | 16 `src/*.py` | ~4.3k lines | metrics, kernels, extract, anisotropic math |
| Experiment orchestration | 9 `scripts/*.py` | ~4.2k lines (freeze) | CLI runners |
| Tests | 6 | ~0.8k | unit tests, no downloads |
| Configs/manifests | 8 | small | frozen designs + COCO splits |
| Documentation | 12 | reports | one report per experiment |
| Generated results/figures | 55 json + pngs | **8.4 MB json** | pair tables, transfer, consistency |
| Temporary/debug | 0 confirmed | — | none removed as debug-only |
| Dependency changes | `pyproject.toml` | none vs freeze | no lockfile in this repo |

Largest Python files at freeze: `run_release_anisotropy.py` (1032), `run_release_anisotropy_repair.py` (728), `release_anisotropy.py` (641), `run_flex_kernels.py` (640).

## Execution map

Shared: `registry.Paths`, `io_utils`, `datasets`, `metrics`/`kernels`/`prh_ref`, `extract` / `extract_final`.

| Entry | Class | Outputs |
|---|---|---|
| `run_release_anisotropy_repair.py` | **1. Authoritative frozen workflow** | `results/release_anisotropy_repair/` |
| `run_release_anisotropy.py` | **5→thin archive** | parent dir; **no silent refit** |
| `run_prh_released_code.py` | 2 historical | `results/prh_released_code/` |
| `run_learned_kernels.py` | 2 historical | `results/learned_kernels/` |
| `run_flex_kernels.py` | 2 historical | `results/flex_kernels/` |
| `run_anisotropic_kernels.py` | 2 historical (two-sided `fit_metrics`) | `results/anisotropic_kernels/` |
| `run_phase1.py` | 2 historical original/modern | `results/original`, `results/modern` |
| `check_vision_fx_hooks.py` | 4 diagnostic | hook check JSON |
| `write_freeze_inventory.py` | 4 | work-disk checksums |

Anisotropic **math** is not duplicated: one-sided nested budgets live in `release_anisotropy.py`; two-sided spectral-clip `fit_metrics` stays in `anisotropic_kernels.py` (different estimator; used by the two-sided bridge).

## Cleanup performed

- Moved load/eval helpers (`pack_ab`, `eval_onesided`, `partner_sets`, `load_splits`, `dump_fit`) to `release_protocol.py`. Repair runner no longer imports the parent CLI module.
- Replaced the 1032-line parent runner with a **160-line archive**: skip if complete; `--force` **refuses** to overwrite parent fits; `--extract-only` still extracts if the parent run is incomplete.
- Canonical `jsonable` + `skip_if_complete` in `io_utils`. Learned/flex/anisotropic/PRH runners skip completed `summary.json` unless `--force`.
- Dropped unused local `jsonable` wrappers and unused imports.
- `.gitignore`: `*.npz` and `results/**/cache/`.
- Comments on why `excess.requires_grad` can be false at `S=0` (degenerate `eigh` + constant `torch.where` branches), without changing the estimator.

**Intentionally not merged:** `eval_onesided` (ambient Gram, float32 `extension_stats`) vs `scores_numpy` (float64 contractions). `fit_metrics` vs `fit_one_sided`. Historical reports.

## Scientific notes (not silently “fixed”)

1. **`requires_grad=False` at identity.** Repeated eigenvalues of `S=0` make `eigh` derivatives undefined; PyTorch may drop the graph. `_scores_from_deltas` also uses `torch.where(..., constant_nan)`. Parent last-iterate + no incumbents is still the reason large-`ρ` collapsed to identity. Repair keeps incumbents; algorithm of `project_S` unchanged.
2. **Direct vs contracted CKA `a` ~2e-5.** `eval_onesided` scores `metric_gram` then `extension_stats(k.float())`; fitting uses float64 contractions. Frozen `checks.json` flagged 1e-6. Test now asserts `<5e-5` on a synthetic pair so a real break is caught without pretending 1e-6.
3. **Parent CLI vs library.** After the freeze, `fit_one_sided` in `src/` is the **repaired** optimiser. Rerunning the old 1032-line script would **not** have reproduced parent identity-collapse. The archive wrapper prevents that silent mismatch.

## Deferred (do not rewrite the freeze tag)

Removing generated JSON/PNG from **future** commits is done: they are gitignored; `git rm --cached` untracks the working copy. The freeze tag still contains the historical blobs so that snapshot stays reproducible. Do not `git clean` the local `results/` tree.

## Verification

Ubuntu venv: `pytest tests -q` → **54 passed**. No extraction, no `--force` experiment runs, no vLLM changes.

## After cleanup (working tree vs freeze)

Python: **−1018 / +191** lines (`git diff ed966f5 -- '*.py'`). Parent runner 1032 → 160. Cumulative vs `origin/main` remains dominated by frozen JSON/PNG (~318k insertions).
