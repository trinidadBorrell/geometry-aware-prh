# Acceptance status

Default checks are CPU-only. They do not download datasets, load model weights, or generate text. Full-gallery medians stay not-run.

## Dataset helper

`resolve_dataset` uses a supplied path as given. A missing supplied path raises and does not download. Omitting the path returns a complete cache, or raises. `--download` on `experiments/fetch_datasets.py` is required to fill an empty cache under `--cache` (default `~/.cache/rho-cca/datasets`). An incomplete cache directory is left in place.

## Cases

| Case | Original run | API | Parity | Max error | Tolerance | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A. OLMo-3 / CLIP LAION → DINOv2, COCO val2017 q=32 | `frozen_partner_cka_audit_v1` smoke; scores from `spectral_transform_calibration` | `fit_ridge`, `cka_with_native`, `save_fit` / `load_fit` | Real-data fitting on preprocessed arrays | Weight `2.33e-15`; dense metric `2.78e-15`; CKA excess `5.55e-17` | `atol=1e-10` for the weight (max entry `0.658`); CKA `atol=1e-6` | Pass |
| B. Same triple | Legacy `regularized_cca` and `reliability_weights` from `angus-prh-replication` `8fd211a`, run once on these arrays | `fit_rho_cca`, weighted and equal `cka_with_native`, save/load | Real-data fitting on the same preprocessed arrays | Factors `2.94e-13`; correlations `2.55e-15`; weights `1.24e-14`; dense weighted metric `2.05e-12`; CKA excess `6.11e-16` | Factors inside `atol=1e-10`. Dense metric magnitude `288`, so `rtol=1e-8` allows `2.88e-6`. CKA `atol=1e-6` | Pass |
| C. Qwen source vector, OLMo encoder, Moonlight decoder | `imdb_matched_pair_cca_transfer_v1` and `imdb_cca_moonlight_third_model_v1` only | `fit_steering` on the archived vector, `match_norm`, save/load. `attach_steering_target` not called | Application-only. Encoders and `D` were loaded, not fit | OLMo raw `0`; OLMo matched `2.78e-16`; frozen command `1.04e-17`; Moonlight raw `4.34e-19`; Moonlight matched `1.39e-17` | `atol=1e-8`, `rtol=1e-8` | Pass as application parity |
| C fitting | Same run | `fit_steering` from review differences; `attach_steering_target` from bridge rows | Unavailable | — | — | Not-run. No cached activation rows |
| Synthetic steering | None | `fit_steering`, `attach_steering_target`, `match_norm`, save/load | Synthetic correctness against an independent NumPy mean, pseudoinverse, and ridge solve | Inside the acceptance assertions | `atol=1e-10`, `rtol=1e-8` | Pass. Not an IMDb or Moonlight result |
| Flickr / nocaps gallery medians | Audit reports | Not wired to feature files | Optional replay | — | Rounded report values, not `1e-6` references | Not-run |

The smoke triple was selected because its train, validation, and test PCA scores and the archived donor map were all present. That choice was made before reading CKA. Validation selected `t=0.01` with MSE `0.0049359189737058955`, the same penalty and MSE as `smoke_fit.json`. The legacy weight differs from `smoke_donor_W.npy` by `2.33e-15`.

These fits use the saved q=32 coordinates. They do not rerun clipping, row normalisation, or the PCA basis. C is the DINOv2 COCO test split, not a Flickr or nocaps gallery, so the scores are not the gallery medians.

Held-out and test rows were scrambled and the ridge map was unchanged. For ρ-CCA, scrambling validation and test left the training weights and source projector unchanged. Scoring C did not change the stored factor. Save/load repeated the CKA score.

## Fixture

`tests/fixtures/smoke_olmo_clip_dinov2_q32.npz`, 2,312,974 bytes, SHA-256 `1c744712b563a272b0e66ffb70e3b98de334b65353d9e07b108681a4c0be8a1a`.

It holds OLMo and CLIP train, validation, and test scores, the DINOv2 test scores, the COCO split ids, and the legacy reference arrays. No images, captions, weights, or generations. The coordinates are derivatives of COCO val2017 and the three frozen bases, included so the acceptance test can run offline.

Host score paths: `/mnt/sdb1/prh-replication-work/results/spectral_transform_calibration/scores/{train,val,test}/{olmo-3-1025-7b,clip-laion-base,dinov2-small}.npy`. IDs come from `angus-prh-replication:data/manifests/coco_val2017_splits.json`.

## Steering files

The maps stay outside the package. The application check reads one root containing both result directories. Hashes from the run:

| File | SHA-256 |
| --- | --- |
| `imdb_matched_pair_cca_transfer_v1/artifacts/maps.npz` | `a90eccbc716acfd061f2a88f4efd21114cbee951bfc7a4035d6c850570781c8f` |
| `dir_qwen_native.npy` | `2ee4d7a38a849074efa33bf7f2290b3d8111106aff1bad3f8d5831b8b7f8ee27` |
| `dir_olmo_aligned64.npy` | `beb10048a3750ddd3649f25b482856d398116a36fee7917d7ba8f1884db6cfd8` |
| `geometry.json` | `f411b3ffad8a1e5c97327f8cd0072bcda4fd08bdb150a46711869d953026831d` |
| `imdb_cca_moonlight_third_model_v1/artifacts/decoder.npz` | `1afa6daddf0bf57d9e738973646ac853df87039f346dcf9dda9b8fcde5601da1` |
| `z_edit.npy` | `39ef34849ffe0115a68a2a761a1e56e18bf8963aa6811c5750b134baf70d4685` |
| `dir_transfer.npy` | `11c26df633584aa67e7d680afa071aa073ee3830d05aceb2df1ff8452d98d0d1` |
| `decoder.json` | `4ea3ea3f7604d1db5b40bbe462ba8d7661049f753f753edd850437232a1389e2` |
| `frozen_command.json` | `71259a34dfa2aa6996666d23fd44d39ddfc4b1b49a1be1334988f89c6da4e00f` |

The Qwen dose-scaled vector is decoded with the loaded encoders and then norm-matched to the archived OLMo direction. The frozen command is `E_qwen` times the unit Qwen vector and matches `z_edit.npy`. The Moonlight displacement is `z_edit @ D.T` with no intercept, then norm-matched to `dir_transfer.npy`. Two-model, multifamily, and layer-13 files were not used.

## Optional commands

```bash
python experiments/replay_cached.py
RHO_CCA_ARTIFACT_ROOT=/path/to/results python experiments/replay_cached.py
python experiments/replay_moonlight_application.py --artifact-root /path/to/results
```

The gallery command stays not-run until the feature files are mounted. It does not treat a saved median as a new measurement. The steering command is application parity, not fitting parity.

## Research source

Untracked research source that is not in `angus-prh-replication` (`8fd211a`) is copied at `/home/angus/Documents/Code/rho-cca-untracked-source-snapshot`. The copy has 363 files, 4,420,343 bytes, and `sha256sum -c MANIFEST.sha256` succeeded. Originals were left in the working tree. That snapshot is not part of this package.
