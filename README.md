# rho-cca

Research library for a ridge-induced metric, correlation-weighted CCA, centred
linear CKA, and explicit steering transport.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q
.venv/bin/python examples/fit_save_load.py
```

The smallest command, once the package is installed, is:

```bash
python examples/fit_save_load.py
```

## Fit

```python
from rho_cca import (
    ModelPair, PairedDataset, RidgeConfig, RhoCCAConfig,
    fit_ridge, fit_rho_cca, split_rows,
)

pair = ModelPair(source_endpoint, partner_endpoint)
split = split_rows(dataset, test_size=0.25, validation_size=0.25, seed=0)
ridge = fit_ridge(pair, PairedDataset(dataset.source, dataset.partner, split), config=RidgeConfig(t=1e-3))
cca = fit_rho_cca(pair, PairedDataset(dataset.source, dataset.partner, split), config=RhoCCAConfig())
```

`RidgeConfig(t=None)` selects the penalty on the validation split. Test rows
are not used for that choice. `RhoCCAConfig` keeps every canonical coordinate
in the prepared support unless `components` is set. Reliability weights use the
training Pearson correlations, with the archived floor `1e-6` and `beta=1`.

Rows are examples. `x @ F` applies a factor, and the metric is `F @ F.T`.
Save and reload with `save_fit` / `load_fit` (NPZ plus a JSON header, no
pickle). A reloaded object does not need the training rows to transform new
rows from the same feature basis.

## Frozen partner and steering

`fit.cka_with_native(source, native_c)` applies the source factor and leaves C
as supplied. Matching width is not enough to accept another checkpoint.

```python
steered = fit_steering(cca, pair, DifferenceInMeans(positive_ids, negative_ids), data, config=SteeringConfig())
attached = attach_steering_target(steered, bridge, anchor="partner", target=target, config=SteeringConfig(t=1e-3))
applied = match_norm(attached.vector_for(target), dose)
```

`v_B = v_A @ F_A @ G_B`. The displacement does not subtract a fitted mean or
add a decoder intercept. Pseudoinverse decoding and learned ridge decoding are
named separately. Full-rank correlation weights can cancel; the fit records
that instead of claiming the weights moved the vector. `match_norm` is
optional and refuses to amplify a near-zero decode.

See `examples/frozen_partner.py` and `examples/steering_transport.py`.

## Layout

| Path | Role |
| --- | --- |
| `ONBOARDING.md` | API, conventions, and validation workflow |
| `src/rho_cca/` | Importable package |
| `tests/test_library_*.py`, `tests/acceptance/` | Default pytest suite |
| `examples/` | Runnable public-API examples |
| `experiments/` | Optional cached and historical replays |
| `docs/conventions.md` | Row, penalty, weight, and displacement rules |
| `docs/PROVENANCE.md` | Map back to `angus-prh-replication` |

`python -m pytest tests -q` collects `test_library_*.py` and
`test_acceptance_*.py`. Older research tests under other names stay out of
that run. The suite is CPU-only and does not load a model. PyTorch is imported
only inside `preblock_rows`.

Numerical acceptance checks reproduce archived ridge and ρ-CCA fits on cached
PCA representations, held-out alignment scores, and steering-vector transport
through saved maps. Full model extraction, preprocessing and behavioural
reproduction are optional experiment workflows. What version 0.1.0 verifies
is recorded in `docs/ACCEPTANCE.md`.

Historical Flickr, nocaps, IMDb, and Moonlight numbers are optional and are
not claimed by the default tests. Dataset paths are arguments to
`experiments/fetch_datasets.py`. An omitted path is not downloaded unless
`--download` is set. Commands are in `experiments/README.md`.
