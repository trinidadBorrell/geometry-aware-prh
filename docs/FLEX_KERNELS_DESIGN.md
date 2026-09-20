# Flexible polynomial and spectral kernels — frozen design

Machine-readable: `configs/flex_kernels.json`.

Reuses frozen train linear-CKA layers and train distance scales from `results/learned_kernels`. Does not modify PRH or RBF experiment directories. Evaluation remains the exploratory COCO test gallery.

Four simplex families: monomial t^p (p=1..12), spherical-harmonic Z_{p,d} with shared order weights, linear+Fourier shells, spherical+Fourier. Regularisation panel {unreg, nl, moderate, strong} as in the config. Primary objective a/b, secondary a−b. Val selects the regulariser; test is not used for selection.

Completed results and answers: `docs/FLEX_KERNELS_REPORT.md`, `results/flex_kernels/`.
