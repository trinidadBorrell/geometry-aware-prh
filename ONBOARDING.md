# Agent onboarding

This repository provides a Python library for learned representation geometry and steering-vector transport. Start with the public API and the recorded acceptance checks before changing the numerical implementation.

This guide describes the library's design and working conventions. Follow applicable `AGENTS.md` instructions and the current user request. `README.md` and `pyproject.toml` describe installation. `docs/ACCEPTANCE.md` records what version 0.1.0 verifies, including numerical errors and stages that were not run.

## Read first

- Applicable `AGENTS.md` files.
- `README.md` and `pyproject.toml`.
- `docs/ACCEPTANCE.md`.
- `docs/conventions.md` and `docs/PROVENANCE.md`.
- The relevant public implementation, examples, and tests.

Inspect the current branch and working-tree changes before editing. The library was developed on `rho-CCA`. Do not switch branches or discard existing work automatically. Research snapshots and legacy result trees may live outside the package. The package does not import `prh_replication`.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/rho_cca/` | Installable public library and numerical implementation |
| `tests/` | Fast offline tests, including small acceptance cases |
| `examples/` | Small runnable demonstrations of the public API |
| `experiments/` | Optional acquisition helpers and historical reproductions |
| `docs/conventions.md` | Row convention, penalties, weights, CKA, and displacements |
| `docs/PROVENANCE.md` | Which legacy modules supplied each formula |
| `docs/ACCEPTANCE.md` | Tested stages, references, numerical errors, and missing inputs |

Examples are `examples/fit_save_load.py`, `examples/frozen_partner.py`, and `examples/steering_transport.py`. Optional commands are `experiments/fetch_datasets.py`, `experiments/replay_cached.py`, and `experiments/replay_moonlight_application.py`. Inspect the tree before assuming another filename. Do not copy legacy experiment runners into the library.

## Public operations

- `fit_ridge(model_pair, dataset, *, config)` fits a paired ridge predictor and its induced source metric.
- `fit_rho_cca(model_pair, dataset, *, config)` fits regularised CCA and correlation-weighted geometry.
- `fit_steering(alignment, model_pair, objective, dataset, *, config)` constructs a source steering command and decoded directions.
- `attach_steering_target(fit, bridge, *, anchor, target, config)` attaches a new target using paired bridge activations.

`save_fit` and `load_fit` persist a fitted object as NPZ plus a JSON header. There is no pickle. Fitting entry points are module-level functions. Fitted objects hold reusable state and application methods. Use the existing signatures and examples rather than adding another estimator framework.

`Endpoint` identity is the model, revision, layer, hook site, pooling, width, and string extraction settings. Feature width alone does not establish compatibility. `cka_with_native` transforms the fitted source and leaves the supplied native features unchanged.

## Theory: what the library is learning

The research question is whether different models encode some of the same relationships between inputs, but express them in different geometries. The library learns a way to measure those relationships using paired examples. Steering asks the stronger question of whether corresponding coordinates also support corresponding behavioural interventions.

These are empirical questions. The linear algebra below defines the objects. It does not guarantee successful generalisation or steering.

### Features, factors, and metrics

Let \(X \in \mathbb{R}^{n \times q_X}\) and \(Y \in \mathbb{R}^{n \times q_Y}\) contain matched training features from models A and B. Rows are examples. Columns are coordinates after the chosen preprocessing and optional PCA. Means and PCA bases are fitted on training data only.

For a factor \(F \in \mathbb{R}^{q_X \times k}\), transformed coordinates and their induced metric are

\[
Z = XF, \qquad M = FF^\top.
\]

For row vectors \(x, x'\),

\[
\langle x, x' \rangle_M = x M x'^\top, \qquad d_M(x, x')^2 = (x - x') M (x - x')^\top.
\]

Learning a metric changes which combinations of coordinates contribute to similarity. It does not change the pretrained model.

\(M\) is positive semidefinite. A nonzero displacement with \(xF = 0\) has zero induced distance, so the ambient space carries a seminorm. The retained column space of \(F\) can still have a genuine inner product. A constant linear metric does not by itself identify a curved manifold.

The range of \(F\) is the feature subspace this metric depends on. Calling that subspace privileged describes the fitted construction. It does not establish that every semantic distinction or useful intervention lies there.

Clipping and row normalisation are nonlinear. A factor fitted after them is constant in the prepared coordinates, not the same matrix on raw activations. Raw steering transport through those preparations raises `NonlinearTransportError`.

### Ridge prediction induces source geometry

For centred paired features, ridge learns a linear predictor of B from A. The library uses

\[
C_X = X_c^\top X_c / n, \qquad C_{XY} = X_c^\top Y_c / n,
\]

and solves

\[
(C_X + \lambda I) W = C_{XY}, \qquad \lambda = t \, \operatorname{tr}(C_X) / q_X.
\]

This is the normal equation for

\[
W = \operatorname{argmin}_W \frac{1}{n}\lVert X_c W - Y_c \rVert_F^2 + \lambda \lVert W \rVert_F^2.
\]

\(W\) has shape \((q_X, q_Y)\). The divisor is \(n\), the number of training rows. ρ-CCA uses \(n-1\) instead. Keep those divisors separate when moving a penalty between the two fits.

Predicted coordinates have Gram

\[
(X_c W)(X_c W)^\top = X_c (WW^\top) X_c^\top,
\]

so the induced A-side metric is \(M_{A \mid B} = WW^\top\), not \(W^\top W\).

Two A activations are close under this metric when the predictor sends them to similar B features. If B shares structure with another model C, that geometry might also improve A–C alignment. That is a testable possibility, not a theorem about every partner.

Metric coordinates are \((\text{support} - \mu_X) W\). Predictions add \(\mu_Y\). A displacement does not include that intercept. A common offset also disappears under centred CKA. With `t=None`, the penalty minimises validation prediction MSE and ties keep the larger penalty. Test rows are not a selection input.

### Regularised CCA and correlation weighting

CCA finds paired linear readouts whose scores agree across matched examples. The library centres the training rows and divides by \(n-1\):

\[
C_X = X_c^\top X_c / (n-1), \qquad C_Y = Y_c^\top Y_c / (n-1), \qquad C_{XY} = X_c^\top Y_c / (n-1).
\]

Regularisation uses the mean-eigenvalue scale

\[
A_X = C_X + \tau_X I, \qquad \tau_X = \eta \, \operatorname{tr}(C_X) / q_X,
\]

and the same form for \(Y\). The default \(\eta\) is `RhoCCAConfig.eta` (`0.01`). The library does not run a generic optimiser. It whitens with the symmetric inverse square root of \(A_X\) and \(A_Y\), takes the SVD of the whitened cross-covariance, and applies one joint sign convention: paired columns flip together so the largest-magnitude entry of the source column is nonnegative.

If every eigenvalue of \(A_X\) is above `support_tol` times the largest eigenvalue, the returned source directions \(L_X\) satisfy

\[
L_X^\top A_X L_X = I,
\]

and likewise for \(Y\). These directions need not be Euclidean-orthogonal. Below that threshold the implementation replaces the eigenvalue by 1 before the inverse square root. It does not delete the mode, and the returned support mask does not by itself truncate \(L\). `components=None` keeps every finite canonical coordinate.

The covariance inner product \(A_X\) is not the learned feature metric \(M_X\). Do not assume the readout vectors are orthonormal under \(M_X\).

For ρ-CCA, the weights come from the Pearson correlations of the paired training canonical scores, not from the regularised singular values. Correlations are clipped to \([0, 1]\) for the weight. A training correlation below `-1e-8` is counted and recorded as `algebraic_failure`; it is not left as a negative weight. With the configured floor \(\epsilon\) and power \(\beta\),

\[
r_i = \max(\operatorname{clip}(\rho_i, 0, 1)^2, \epsilon), \qquad w_i = \frac{r_i^\beta}{\operatorname{mean}_j(r_j^\beta)}.
\]

The defaults are \(\beta = 1\) and \(\epsilon = 10^{-6}\). Define

\[
F_X = L_X \operatorname{diag}(\sqrt{w}), \qquad M_X = F_X F_X^\top = L_X \operatorname{diag}(w) L_X^\top,
\]

and likewise for \(Y\). Coordinate scaling uses \(\sqrt{w_i}\). The metric contribution is \(w_i\). Applying the square a second time is a different method.

These weights are not probabilities of semantic relevance or of steering success. Equal-weight coordinates reuse the same \(L\) with scale 1. That is not native Euclidean geometry, whose factor would be \(I\). Compare weighting only when the directions stay fixed. Changing rank at the same time tests a different construction.

### CKA measures agreement between example relationships

`linear_cka` column-centres the evaluation features and then computes

\[
\operatorname{raw}(X, Y) = \frac{\lVert X_c^\top Y_c \rVert_F^2}{\lVert X_c^\top X_c \rVert_F \, \lVert Y_c^\top Y_c \rVert_F}.
\]

For the Grams \(K_c = X_c X_c^\top\) and \(L_c = Y_c Y_c^\top\), this is the same number as

\[
\operatorname{CKA}(K_c, L_c) = \frac{\langle K_c, L_c \rangle_F}{\lVert K_c \rVert_F \, \lVert L_c \rVert_F}.
\]

`linear_cka_from_feature_grams` instead double-centres \(XX^\top\) and \(YY^\top\) for features that have not already been column-centred. Centring uses the evaluation rows only. It does not update the fitted transformation. Matching feature widths is unnecessary. \(n\) here is the evaluation count and can differ from the training count.

For a fixed centred Gram and a uniform permutation of one side's row identities, the expected raw score is

\[
b(X, Y) = \frac{\lVert X_c \rVert_F^2 \, \lVert Y_c \rVert_F^2}{(n-1) \, \lVert X_c^\top X_c \rVert_F \, \lVert Y_c^\top Y_c \rVert_F}, \qquad E(X, Y) = \operatorname{raw}(X, Y) - b(X, Y).
\]

In Gram form the numerator is \(\operatorname{tr}(K_c)\operatorname{tr}(L_c)\). This is defined for \(n > 1\) and nonzero centred Gram norms. A zero-variance side raises `RhoCCAError`. Compute the chance term separately for each arm. Excess CKA is this subtraction, not division by the baseline and not a significance test.

A metric can concentrate on a few shared modes and raise CKA without preserving every semantic distinction or local neighbourhood. `shared_neighbour_fraction` is a separate neighbour-overlap score. Use it, or a behavioural evaluation, when neighbourhood structure is the scientific target.

### Three distinct transfer questions

1. New examples or a new dataset: apply a frozen A–B fit to new inputs from the same endpoints.
2. New partner: fit on A–B, then compare A under \(M_{A \mid B}\) with native C. The source Gram is \(X_A M_{A \mid B} X_A^\top\) and the native Gram is \(X_C X_C^\top\). C does not enter the donor fit, weight selection, or preprocessing fitted for A.
3. New feature space: apply an intervention in B or another model. That requires a model-specific encoder, decoder, or attachment. A's numeric metric has no automatic meaning in an unrelated basis.

The first two evaluate geometry. The third can evaluate intervention transport. Replacing both endpoints is not a frozen A–B to A–C partner swap.

Shuffled fitting with genuine evaluation pairs tests whether the pairing mattered. A rotation of a factor's output coordinates leaves \(FF^\top\) unchanged, so it checks that invariance. It is not a negative control for the learned orientation.

### Steering is transport of a displacement

`fit_steering` sets the source vector to the difference of the source training class means:

\[
v_A = \operatorname{mean}(h^+) - \operatorname{mean}(h^-).
\]

When the two classes have the same number of rows, this equals the average of paired differences. The library does not require the classes to have equal size, and it does not look for an explicit pairing beyond the supplied ids. Labels name source training rows. They are not applied to the partner.

The returned vector is the displacement. Applying it inside a live residual stream is a caller-side edit, not part of fitting. Optional extraction helpers such as `preblock_rows` import PyTorch only when called.

For an affine encoding \(z = (h - \mu) F_A\),

\[
z(h + v_A) - z(h) = v_A F_A.
\]

The mean cancels in that difference. The library's displacement therefore does not subtract a fitted feature mean. A constant anisotropic metric does not, by itself, change this difference of means or justify multiplying it by an inverse metric. A classifier gradient is a covector and transforms differently.

Here \(F_A\) means the linear factor actually used for transport. In the implementation that is `linear_factor`: optional PCA composed with the stored coordinate scaling. It need not be the factor used in a separate CKA call. Score standardisation and any further scaling have to be recorded explicitly. Clipping and row normalisation are rejected for this raw transport.

Encode the displacement and decode it:

\[
\Delta z = v_A F_A, \qquad v_B = \Delta z \, G_B,
\]

with \(F_A\) shaped \((d_A, k)\) and \(G_B\) shaped \((k, d_B)\). Decoding omits a fitted intercept, including a ridge-decoder intercept that was stored while fitting activations.

For a partner encoder \(F_B\) of full column rank, the minimum-Euclidean-norm edit that reproduces the requested coordinate change is

\[
v_B = \Delta z \, F_B^+.
\]

The library calls this decoder `pseudoinverse` and sets \(G_B = \operatorname{pinv}(F_B)\). "Smallest" means native Euclidean norm, not semantic distance or behavioural disruption. With rank deficiency, only attainable coordinates are reproduced exactly. A learned ridge decoder is a different reconstruction and is labelled `ridge`. A ridge alignment transports by its regression factor with \(G_B = I\). That path is not reported as a pseudoinverse.

A self-roundtrip \(v_A F_A F_A^+\) projects onto the encoder's column space. PCA or canonical truncation can remove part of an intervention before cross-model transport. `match_norm` changes dose after decoding. A vector below `near_zero_norm` becomes zero rather than being scaled up. Norm matching does not restore discarded coordinates.

### Why weighting may cancel in steering

Let \(E_A, E_B\) be unweighted row encoders and \(S\) an invertible diagonal scaling used on both sides. When \(E_B\) has full column rank,

\[
(E_B S)^+ = S^{-1} E_B^+,
\]

so

\[
v_A E_A S (E_B S)^+ = v_A E_A E_B^+.
\]

The weighted and unweighted transports agree. This can happen in a retained space that is smaller than the residual. Correlation weighting can still change the Gram used for CKA. The fit compares the weighted and equal-weight displacements and records `weights_cancelled` when they agree. A one-sided filter between an unweighted encode and decode is a different operation.

### Attaching a third model

`attach_steering_target` freezes the source command and one anchor factor. It fits only a ridge decoder from unlabelled anchor–target pairs. Means and the decoder intercept may be used to fit activations. The displacement applied afterwards is `command @ G` and does not add the intercept.

Independent CCA fits have their own coordinate signs and scales. A separately fitted A–C frame cannot replace the frozen shared frame without an explicit alignment.

Target sentiment labels, target-native steering vectors, and generation scores are not arguments of the attachment. If a later evaluation uses them to choose a layer or a dose, record that separately. It is not part of the unlabelled fit.

Decoding coordinates is an algebraic property. Behavioural preservation still needs an evaluation at a documented site, schedule, and dose, with a usable native reference and quality checks.

### Related research extension: caption variance

A historical extension penalised directions that varied between captions of the same image. If \(W_{\mathrm{caption}}\) is their average within-image covariance, one language-side denominator used in that work was

\[
A_X = C_X + \lambda_{\mathrm{caption}} W_{\mathrm{caption}} + \tau_X I.
\]

In that extension, \(C_X\) and \(C_{XY}\) were built from per-image caption centroids paired with image features, while \(W_{\mathrm{caption}}\) was estimated from the training captions around those centroids. The quadratic term penalises within-image caption variability along a readout, and the cross-covariance still rewards image–caption agreement. That variation can be wording, emphasis, or both.

This shows how a task-specific prior can enter the geometry. The current library API does not expose a caption-covariance penalty. Dataset helpers can locate caption files. That is acquisition, not this CCA variant.

### Boundaries on interpretation

Canonical readout directions are not established causal intervention directions. Shared CKA geometry does not imply a universal semantic space, shared mechanisms, a uniquely identified manifold, or preserved steering.

Assess transfer and quality on held-out examples. Report raw scores and paired contrasts. Similar point estimates are not proof of equivalence. Keep algebraic identities, archived numerical parity, and new empirical findings distinct. `docs/ACCEPTANCE.md` is the record of which library stages were tested.

## Data and evaluation

Use the scikit-learn wrappers `split_rows`, `split_groups`, and `stratified_group_folds` for new splits. `split_rows` uses `train_test_split`. `split_groups` uses `GroupShuffleSplit`, whose sizes count groups. `stratified_group_folds` uses `StratifiedGroupKFold`. Validation is a second split of the training subset. A saved `Split` rejects a group that crosses partitions.

Draw the assignment once and store sample ids, so both endpoints receive it. Keep related captions, positions, paraphrases, and contrast pairs in the same group. Fit preprocessing, PCA, and alignment on training ids only. Penalty selection uses the explicit validation ids. Test ids do not change fitted state.

Match examples by stable sample ids. Preserve a supplied split and its feature metadata when replaying a research case.

`resolve_dataset` and `experiments/fetch_datasets.py` use a supplied path as given. A missing supplied path raises and does not download. Omitting a path does not download. `--download` fills an empty directory under `--cache`, default `~/.cache/rho-cca/datasets`, and refuses to overwrite an incomplete directory. Importing `rho_cca` and the default tests do not download data or load a model.

## Local validation

Run the ordinary offline suite:

```bash
python -m pytest tests -q
```

Pytest collects `test_library_*.py` and `test_acceptance_*.py`. Other research tests in the checkout stay outside that run.

For a public-API or packaging change, also run the documented examples and a non-editable install and import outside the checkout. Confirm that the import resolves to the installed package.

For a numerical change, run the relevant cached acceptance case that is actually available. Follow `docs/ACCEPTANCE.md` for artifacts, commands, and tolerances. Investigate a discrepancy before loosening a tolerance or replacing an expected value.

Keep these evidence levels separate:

- Synthetic correctness.
- Real fitting parity on saved activations or PCA scores.
- Application parity using already-fitted maps.
- Full extraction or behavioural reproduction.

Loaded-map parity does not establish decoder-fitting parity. Fitting on saved PCA scores does not establish raw preprocessing or PCA-fitting parity. A historical full-gallery result is not, by itself, a newly reproduced library result.

Optional reproductions stay outside default test discovery. Missing optional data is `not-run`, with the missing inputs named. It is not a pass and not an unexplained skip of a mandatory test.

## Change and handoff

Keep changes focused and call the public API from examples and experiment adapters. Preserve numerical provenance and the versioned save/load schema. Ordinary maintenance needs the local checks above. A broad CI or benchmark programme is not required for that maintenance.

Preserve untracked research source and existing artifacts. Stage explicit paths. Do not fold caches, weights, activations, generations, or an entire research tree into the package.

When handing off, report the changed behaviour, the tests that were actually run, numerical errors where they matter, and the stages that remain unverified. Update `docs/ACCEPTANCE.md` when its evidence changes. Do not copy test counts or acceptance errors into this file.
