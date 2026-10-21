# Caption reliability → CCA + correlation weighting

Experiment code for the caption-filter → CCA → ρ²-weighting line of work (presentation label **CCA + correlation weighting**; archived reports called this “CCA reliability”).

**Not included:** rendered slide decks under `results/cca_reliability_slide_deck_*` (local presentation artifacts only).

## Layout

| Path | Role |
|---|---|
| `configs/` | Experiment JSON configs |
| `scripts/` | CLI entry points |
| `tests/` | Unit tests |
| `../../src/prh_replication/` | Library modules (package imports) listed below |

Compatibility symlinks keep the usual top-level `configs/`, `scripts/`, and `tests/` paths working.

## Experiments (run order)

1. **Caption reliability filter** — `run_caption_reliability_filter.py` / `caption_reliability_filter_v1.json`
2. **SAE × cross-covariance CCA** (shared CCA machinery) — `run_sae_crosscov.py` / `sae_crosscov_v1.json`
3. **CCA + correlation weighting** — `run_crosscov_weighted_cka.py` / `crosscov_weighted_cka_v1.json`
4. **mKNN under the same metrics** — `run_crosscov_metric_mknn.py` / `crosscov_metric_mknn_v1.json`
5. **Extracted-diagonal ablation** — `run_cca_metric_diagonal_ablation.py` / `cca_metric_diagonal_ablation_v1.json`
6. **Feature vs PCA metric / caption×metric 2×2** — `run_feature_vs_pca_metric.py` / `feature_vs_pca_metric_v1.json`
7. **Procedure replication** (new pairs + Flickr30k) — `run_cca_reliability_replication.py` / `cca_reliability_replication_v1.json` (+ `extract_cca_reliability_replication_features.py`)

Results live on the work disk under `results/<experiment_id>/` (gitignored).

## Package modules

- `caption_reliability.py` / `caption_reliability_study.py`
- `sae.py` / `sae_crosscov.py` / `sae_crosscov_study.py`
- `crosscov_weighted_cka.py` / `crosscov_weighted_cka_study.py`
- `crosscov_metric_mknn.py` / `crosscov_metric_mknn_study.py`
- `cca_metric_diagonal_ablation.py` / `cca_metric_diagonal_ablation_study.py`
- `feature_vs_pca_metric.py` / `feature_vs_pca_metric_study.py`
- `cca_reliability_replication.py` / `cca_reliability_replication_study.py`
- `feature_prep.py` — clip / L2-prep helpers
- `spectral_calibration.py` — CKA/mKNN scoring helpers used by the above

Model-manifest additions for new LMs/VMs: `data/manifests/release_models.json`, `registry.py`, `extract_final.py`.
