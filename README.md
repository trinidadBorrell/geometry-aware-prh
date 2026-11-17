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
| 13 Oct | CKA excess (raw − shuffle baseline) versus version ordinal on the combined 9-model language panel. | `results/cka_vs_recency_excess/` |
| 23 Sep | RBF/RQ kernels rescored on an 80/20 holdout of the cached gallery. | `results/learned_kernel_cka_80_20/` |
| 27 Sep | Metric stability: one-sided, shared-PC, separate, sample size, partner transfer. | `results/metric_stability/` |
| 27–28 Sep | Full-gallery versus quarter-gallery training of the one-sided metric. | `results/resampled_metric_training/` |
| 29 Sep–1 Oct | Structured factors: identity, one-sided, shared, separate, plus spectra and per-pair bars. | `results/structured_metric_factors/` |
| Oct 2026 | Caption reliability → CCA + correlation weighting (filter, weighted CCA, mKNN, replication). Code: `experiments/caption_cca_reliability/`. | `results/caption_reliability_filter_v1/`, `results/crosscov_weighted_cka_v1/`, `results/cca_reliability_replication_v1/`, … |

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

`scripts/plot_cka_vs_recency_excess.py` is the redo: **each OLMo → temporally nearest Qwen** (plus Qwen→nearest-OLMo anchors for ≥7 points), y-axis **excess CKA** (raw − analytic shuffle baseline). Headline: `temporal_match__excess_cka_vs_release_date.png`.

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

## Spectral transform calibration

Compares a covariance-spectrum map with a correspondence-fitted ridge map and a matched q=32 exp(S) metric, then applies singular-value and eigenvalue powers. Compatible archived metrics are calibrated on a separate full-space track. Protocol: `configs/spectral_transform_calibration_protocol.md`. Output: `results/spectral_transform_calibration/`.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_spectral_transform_calibration.py \
  --work /mnt/sdb1/prh-replication-work --stage all
```

## Unrestricted metric comparison

Asks whether ridge beat the budgeted one-sided metric because of the distortion box or because the objectives differ. It reuses the frozen q=32 scores and the existing ridge and covariance maps. Protocol: `configs/unrestricted_metric_comparison_protocol.md`. Output: `results/unrestricted_metric_comparison/`.

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_unrestricted_metric_comparison.py \
  --work /mnt/sdb1/prh-replication-work --stage all
```

## Sentiment steering

These runs ask whether a sentiment edit fit in one model can be carried into another. Paths in the configs are relative to `--work-root`. Weights, Hugging Face datasets, activation caches, and `results/` stay on that disk and are not in git.

Pinned checkpoints, looked up under `$WORK/hf`:

| Family | Hugging Face id | Revision | Pre-block layer |
|---|---|---|---:|
| Qwen2-7B Base | `Qwen/Qwen2-7B` | `453ed1575b739b5b03ce3758b23befdb0967f40e` | 17 |
| OLMo-7B-0724 Base | `allenai/OLMo-7B-0724-hf` | `1ee306df318ee15bfe4a76ebd5c002b0105b1ab6` | 11 |
| Gemma 2 2B Base | `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | 9 |
| Mistral 7B Base | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | 11 |
| Moonlight-16B-A3B | `moonshotai/Moonlight-16B-A3B` | `476b36a473d4467f94469414bef6cee75c9c8172` | 9 |

SiEBERT is `siebert/sentiment-roberta-large-english` at `74cea614e245b0832c770ec9aa51bd58df965b9c`, truncated at 512 tokens. The host runs used transformers 4.44.2. Moonlight needs `trust_remote_code`, eager attention, and a process-local acceptance of safetensors archives that omit the format tag. That patch is in the Moonlight runner. It is not a transformers upgrade.

Matched passages are `data/imdb_matched_pairs_v1.json`, rebuilt by `data/build_imdb_matched_pairs_v1.py`. Later panels also read earlier `results/*/artifacts` split files when those directories exist. A missing split file shrinks the review exclusion set. Those artifacts are not in git, so a fresh machine does not reproduce the archived panels until the earlier runs have been executed in order.

```bash
export PYTHONPATH=src
python3 -m prh_replication.multifamily_shared_subspace
python3 scripts/run_gemma_layer_anchor.py --help
```

The first command is the CPU self-check. `--help` checks the CLI. Neither loads a model. A full run is:

```bash
export PYTHONPATH=src HF_HOME="$WORK/hf"
python3 scripts/run_imdb_matched_pair_sentiment.py \
  --config configs/imdb_matched_pair_sentiment_v1.json --work-root "$WORK"
```

Recommended order. Each runner reads the previous run's artifacts and does not overwrite them.

| Step | Config | What it decides |
|---|---|---|
| Matched-pair sentiment | `configs/imdb_matched_pair_sentiment_v1.json` | Qwen and OLMo native vectors |
| CCA transfer | `configs/imdb_matched_pair_cca_transfer_v1.json` | Whether the Qwen edit carries into OLMo |
| Native retention | `configs/imdb_cca_native_retention_v1.json` | 80% retention against OLMo's own edit |
| Gemma native screen | `configs/imdb_cca_third_model_transfer_v1.json` | Stopped: the prespecified Gemma dose failed validation |
| Moonlight transfer | `configs/imdb_cca_moonlight_third_model_v1.json` | Decoder into Moonlight, Qwen command held fixed |
| Five-family geometry | `configs/multifamily_cca_shared_subspace_v1.json` | Shared subspace, Moonlight excluded from the fit |
| Anchor swap | `configs/multifamily_cca_anchor_swap_v1.json` | Same command, Gemma or Mistral as the anchor |
| Gemma layer chart | `configs/gemma_layer_anchor_v1.json` | Gemma layer 9 versus layer 13 as the anchor |

`scripts/run_actadd_imdb_sentiment_transfer.py` is the earlier activation-addition entrypoint. Later panels exclude its review ids. `scripts/run_imdb_cca_blinded_quality_review.py --prepare` writes a blank rating sheet. It does not fill ratings.

On the 128-review Gemma-layer panel, the matched Gemma layer-9 route scored +0.078 over unedited and Moonlight's own edit scored +0.080, at the same injection norm. That is a comparison of point estimates. It is not an 80% retention result. This layer-9 route is the chart fit in `gemma_layer_anchor_v1`. It is not the earlier archived Gemma layer-9 consensus direction from the anchor swap. Gemma is the attachment anchor. The edited model is Moonlight. Document retrieval is a geometry diagnostic, not a steering test. Layer 13 did not clearly beat unedited on that panel.

Archived numeric panels cannot be recreated from git alone. They need the local weights and the result directories named above. Re-running a full panel rescores generations. It is not a smoke test.

## Known limitations

- Exploratory COCO gallery (already used in earlier stages).
- Final-layer release study ≠ max-over-layers PRH means.
- Direct vs contracted CKA `a` can differ by ~2×10⁻⁵ (float32 Gram path vs float64 contractions).
- Identity-start autograd can drop at `S=0` (degenerate `eigh`); repaired fitting keeps incumbents.
- Pairwise cells are dependent. Recency is observational.
