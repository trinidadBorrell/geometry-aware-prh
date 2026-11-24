# Optional reproductions

These scripts call `rho_cca`. They are not part of `python -m pytest tests`.
`fetch_datasets.py` downloads only when `--download` is passed, and only into
`--cache`. It does not generate text.

## Dataset locations

Pass a directory or file you already have. Omitting a path does not download.
Add `--download` to store omitted datasets in `--cache` (default
`~/.cache/rho-cca/datasets`). An incomplete cache directory is not overwritten.

```bash
python experiments/fetch_datasets.py \
  --coco-images /data/coco/val2017 \
  --coco-captions /data/coco/annotations/captions_val2017.json \
  --flickr-captions /data/flickr/dataset_flickr30k.json \
  --flickr-images /data/flickr/flickr30k_images \
  --nocaps /data/nocaps \
  --imdb /data/aclImdb
```

`--only flickr-captions nocaps` limits the set. COCO images are the val2017 zip, COCO captions are
`annotations_trainval2017.zip`, Flickr captions are the Karpathy
`caption_datasets.zip`, Flickr images use the Kaggle CLI for
`eeshawn/flickr30k`, nocaps is the Hugging Face set `lmms-lab-encoder/NoCaps`,
and IMDb is Stanford `aclImdb_v1.tar.gz`.

## Cached fixture

```bash
python experiments/replay_cached.py
```

`experiments/fixtures/ridge_normal_equation.npz` is a 24-by-4 / 24-by-3 ridge
problem. The stored weight was solved with the audited normal equation
`(C_X + λ I) W = Xcᵀ Yc / n`, `λ = t trace(C_X) / q`, `t = 0.001`, outside the
package. The script fits that matrix through `fit_ridge` and prints the max
absolute gap. A gap above `1e-8` is a failure.

## Historical audits

The report checks, which are rounded medians and not 1e-6 references, are:

| Check | Report value | Result directory |
| --- | --- | --- |
| Flickr ridge excess lift | +0.1647 | `frozen_partner_cka_audit_v1` |
| Flickr one-sided ρ-CCA excess lift | +0.1381 | `frozen_partner_cka_audit_v1` |
| nocaps ridge excess lift | +0.1312 | `rho_cca_nocaps_frozen_partner_v1` |
| nocaps ρ-CCA excess lift | +0.1207 | `rho_cca_nocaps_frozen_partner_v1` |
| nocaps weighted minus equal | +0.1679 | `rho_cca_nocaps_frozen_partner_v1` |

Those runs live on the geometry work tree, not in this package. Point
`RHO_CCA_ARTIFACT_ROOT` at the parent of those result directories. The current
script reports `not-run` until a feature adapter for the saved galleries is
added. It does not rescore a JSON summary and call that a reproduction.

```bash
python experiments/replay_cached.py
RHO_CCA_ARTIFACT_ROOT=/path/to/results python experiments/replay_cached.py
```

`RHO_CCA_ARTIFACT_ROOT` is the parent of `frozen_partner_cka_audit_v1` and
`rho_cca_nocaps_frozen_partner_v1`. Without the feature galleries the
historical rows stay `not-run`.

Application of one archived steering run, without refitting and without
generation:

```bash
python experiments/replay_moonlight_application.py \
  --artifact-root /path/to/results
```

The root is the parent of `imdb_matched_pair_cca_transfer_v1` and
`imdb_cca_moonlight_third_model_v1`. A missing directory is `not-run`.

IMDb matched-pair transfer, Moonlight attachment, multifamily consensus, and
the layer-13 retention run also need saved activations or generations.
Rescoring saved generations is a different command from generating new text.
Layer 13's strong native comparison is a recorded negative result, not a
target to beat. Full behavioural reruns are optional and are not byte-stable
across machines.

Expected cost once arrays and models are already local: the CKA replays are
CPU minutes; a full IMDb or Moonlight generation pass is a GPU job of about
an hour and is outside this package's default command.
