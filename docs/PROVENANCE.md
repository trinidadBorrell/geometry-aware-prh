# Provenance

The installed package does not import `prh_replication` or read a result
directory. The conventions below were taken from `angus-prh-replication` at
`8fd211a` unless noted.

| Library piece | Source |
| --- | --- |
| Ridge normal equation, validation-MSE tie break, prediction intercept | `src/prh_replication/objective_independent.py` (`ridge_weight`, `choose_index`, `predict`) |
| PCA, `n - 1` covariances, regularised CCA, training Pearson correlations | `src/prh_replication/sae_crosscov.py` |
| Reliability weights `mean(max(ρ², floor)^β)` and one-sided coordinate scaling | `src/prh_replication/crosscov_weighted_cka.py` |
| Clip quantile then row L2 | `src/prh_replication/feature_prep.py` |
| Minimum-norm decoder `D = pinv(E)` and the row form `v @ F @ G` | `src/prh_replication/privileged_cca_steering_transport.py` |
| Displacement encoding without subtracting the feature mean | `src/prh_replication/cca_subspace_steering_transport.py` in the angus working tree (`encode_displacement`) |
| Ridge decoder slope applied without its intercept | `scripts/run_imdb_cca_moonlight_third_model.py` |
| Raw, analytical chance, and excess CKA | `src/prh_replication/frozen_partner_cka_audit.py` `scores` in the angus working tree. That file was not part of `8fd211a`. The committed `linear_cka_features` reports raw CKA only. |
| Shared-neighbour fraction | the same audit's neighbour overlap: mean `|N_k(left) ∩ N_k(right)| / k` |

`rho-CCA` was branched from `origin/main` (`ad16470`). The library commits on
this branch are the packaging of those formulas, not a rerun of the audits.
