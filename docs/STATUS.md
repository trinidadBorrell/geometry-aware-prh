# Run status

- [x] Source audit (`docs/audit_prh_vs_ours.md`)
- [x] PRH released-code baselines (`results/prh_released_code/`)
- [x] Vision FX `add_1` vs block-output hook check (exact match; no re-extract)
- [x] Learned-kernel extension (`results/learned_kernels/`; exploratory test n=1024)
- [x] Report: `docs/LEARNED_KERNELS_REPORT.md`
- [x] Flexible polynomial / Gegenbauer / Fourier mixtures (`results/flex_kernels/`; exploratory test n=1024)
- [x] Report: `docs/FLEX_KERNELS_REPORT.md`
- [x] Anisotropic PCA-subspace metrics (`results/anisotropic_kernels/`; exploratory test n=1024)
- [x] Report: `docs/ANISOTROPIC_KERNELS_REPORT.md`
- [x] Release alignment / one-sided anisotropic corrections (`results/release_anisotropy/`; final-layer protocol; exploratory test n=1024)
- [x] Repair pass + freeze (`results/release_anisotropy_repair/`, `FREEZE.md`, tag `prh-release-alignment-freeze-20260921`)
- [x] Source hygiene after freeze (`docs/CODEBASE_AUDIT.md`); freeze tag not moved

Do not overwrite `results/original`, `results/modern`, `results/prh_released_code`, `results/learned_kernels`, `results/flex_kernels`, `results/anisotropic_kernels`, or `results/release_anisotropy`. Authoritative fitted numbers: `results/release_anisotropy_repair/`.
