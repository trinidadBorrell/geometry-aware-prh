# Optional reproductions

These scripts call `rho_cca`. They are not part of `python -m pytest tests`.
None of them downloads data or generates text.

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

IMDb matched-pair transfer, Moonlight attachment, multifamily consensus, and
the layer-13 retention run also need saved activations or generations.
Rescoring saved generations is a different command from generating new text.
Layer 13's strong native comparison is a recorded negative result, not a
target to beat. Full behavioural reruns are optional and are not byte-stable
across machines.

Expected cost once arrays and models are already local: the CKA replays are
CPU minutes; a full IMDb or Moonlight generation pass is a GPU job of about
an hour and is outside this package's default command.
