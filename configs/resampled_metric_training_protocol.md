# Resampled metric training protocol

Question: does fitting one language-side metric on fresh random quarters of the training pool improve exploratory held-out alignment or reproducibility, compared with optimising on the complete training gallery?

Written before any exploratory-test scores from this comparison. Historical result directories, including `results/metric_stability/`, are not modified and are not rerun.

## What is reused

- Final-layer caches and clip-then-L2 on each full split (`load_final_prepared`). Subsets do not re-clip.
- Train, validation, and exploratory-test IDs from the existing COCO manifest. One image per sample id.
- Full-training mean and signed 32-PC basis (`fit_basis`). Frozen for every update and every evaluation. PCA is not refit on quarters.
- One-sided metric \(M_A=I+U_A(\exp(S)-I)U_A^\top\), partner metric \(I\), excess \(a-b\), contracted scores, `torch.matrix_exp` on the gradient path, spectral projection outside the graph, Adam learning rate 0.03, gradient clip 5.
- The manifest split named `test` is exploratory held-out evaluation. It does not select steps, budgets, seeds, or schedules.

## Procedures

Both procedures keep one persistent \(S\) and the same projected initial matrix for a given language model, budget, and initialisation seed.

- Full gallery: every update maximises excess on all 2048 training pairs.
- Resampled quarter: every update draws 512 training positions uniformly without replacement, uses those positions on both sides, and maximises excess after centring inside that subset. The next update draws again. Examples may recur. This targets \(\mathbb E_I[a_I-b_I]\). That expectation is not full-gallery excess, and the subset gradient is not an unbiased estimator of the full-gallery gradient, because centring and the normalising norms depend on \(I\).

There is no cross-budget continuation. \(\rho=0.1\) and \(\rho=0.4\) each start from the seed-determined matrix. Seed 0 is exact identity. Seeds 1 and 2 are perturbations of scale 0.05 from the existing initialiser, then projected onto the budget.

The stochastic path always applies the projected Adam step. A lower score on a different quarter does not revert \(S\). Deployable checkpoints are chosen by full-validation excess, then lower distortion, then the earlier step. Iteration zero is included. Subset scores are logged and are not used for selection.

## Budget

Primary comparison: 120 updates, the established fitter step count, with the same constant learning rate in both procedures. Validation is scored after every update, including step zero (121 checks).

Secondary comparison: the same resampled trajectory continues to 480 updates at the same learning rate and clip. Cumulative paired examples are then about \(480\times512 = 120\times2048\). The extension has more updates and more validation checks, so it has more chances to pick a checkpoint. Final iterates and validation-selected iterates are both reported. Neither comparison is labelled compute-matched.

## Panel

Six base language models against DINOv2-S only. Language-side \(S\) only. Three initialisation seeds. Resampled runs also use three sampling seeds per initialisation. Every run is kept. Optimiser-seed variation and sampling-stream variation are reported separately. These are repeated optimisations on one dataset, not independent data replicates.

## Stability

Comparisons use \(\Delta M\) in the frozen language PCA basis: full-gallery starts against each other; sampling streams at a fixed start; initialisations under resampling; full-gallery against resampled solutions. Cosines are reported as magnitudes. Near-zero corrections stay missing. This does not re-estimate PCA and does not resample the corpus.
