# Structured metric factors

Frozen before any exploratory-test score from this comparison.

## Question

At equal total distortion, how do identity, one-sided, shared-PC and separate fits compare, and do regularised nonorthogonal factorizations change alignment, mutual kNN, or the spectrum of the induced metric?

## Data

Final-layer caches, paired COCO val2017 order, and the existing train / validation / exploratory-test split. Image identities are disjoint. Each model’s training mean and signed 32-PC basis are fit once and reused. PCA is not recomputed on subsets or on evaluation rows.

The manifest split named `test` is exploratory held-out evaluation. It does not select regularisation, block size, optimiser settings, or checkpoints.

## Metric

For a learned basis \(A\) and positive weight \(D\),

\[
B = A^{-\top} D A^{-1}, \qquad M = I + U(B-I)U^\top.
\]

\(A^{-\top}DA^{-1}\) is the inner product in coordinates \(z=A^{-1}u\). It is not \(ADA^{-1}\). \(ADA^{-1}\) is generally not symmetric and is not a kernel metric here.

\(U\) is the frozen training PCA basis. The residual subspace stays at identity. \(A\) may be nonorthogonal. It is not replaced by a QR factor.

The exp(S) benchmark is \(B=\exp(S)\) with \(S=S^\top\), using the same quarter-gallery schedule. Historical full-gallery fits are not pooled in.

## Families

- `exp_s`: full SPD benchmark. No basis penalty and no off-diagonal penalty.
- `fixed_diag`: \(A=I\), \(D=\mathrm{diag}(e^{h})\).
- `oblique_diag`: learned \(A\), diagonal \(D\). \(R_{\mathrm{off}}=0\), so the off-diagonal coefficient is not a second run.
- `block2`, `block4`: learned \(A\), block-diagonal \(D=\mathrm{blockdiag}(\exp H_j)\). Membership is consecutive learned-basis coordinates, not semantic groups.

An unrestricted \(A\) with diagonal \(D\) can already represent any SPD \(B\). Singular values of the column-normalised \(A\) are clipped to \([0.5, 2]\), so the condition number is at most 4. This is a neighbourhood restriction, not a search over every nonorthogonal basis. Column normalisation removes the diagonal scale gauge by absorbing column norms into \(D\) before the clip. The clip itself can change \(B\). Factors remain non-unique after normalisation.

## Penalties

\[
R_{\mathrm{basis}}=\|A^\top A-I\|_F^2/q, \qquad R_{\mathrm{off}}=\|H-\mathrm{diag}(H)\|_F^2/q,
\]

with \(H=\log D\) and \(q=32\). \(R_{\mathrm{off}}\) penalises off-diagonal log-weight entries, not the entries of \(D\) directly.

Profiles, fixed before fitting: strong \((\lambda_{\mathrm{basis}},\lambda_{\mathrm{off}})=(0.1,1)\); stronger \((0.1,10)\). Diagonal families use \(\lambda_{\mathrm{basis}}=0.1\) for oblique \(A\) and zero penalties for fixed \(A=I\).

Both transformed sides contribute. A tied shared factor is counted twice. The training objective is expected subset excess minus these penalties.

## Constraints

Eigenvalues of each induced \(B\) lie in \([1/4,4]\). Distortion is \(d(B)=\|\log B\|_F^2/32\).

Equal total budgets \(\rho\in\{0.1,0.4\}\): one-sided \(d(B_A)\le\rho\); shared-PC \(2d(B)\le\rho\); separate \(d(B_A)+d(B_B)\le\rho\).

exp(S) uses the audited spectral clip and Frobenius scaling. Factor families use the same clip and the same equal-total scale on the induced \(B\), then refactor into the declared \((A,D)\). The refactor diagonalises \(A^\top B^\star A\), with eigenvectors aligned toward \(I\), so the new factors reproduce \(B^\star\) before the singular-value clip. Block families then apply a block-diagonal rotation that gives \(D\) the eigenvectors of the proposed block weight. That rotation preserves \(B^\star\) and leaves off-block entries at zero. It does not give the block family a larger set of induced metrics than diagonal \(D\). If the singular-value clip would leave the factorization, the orthogonal eigenbasis of \(B^\star\) is stored instead. A line search toward the previous feasible factors is the fallback when refactoring cannot land inside the constraints. Accepted \(A\) and \(D\) are the factors that produce the accepted \(B\). Retractions, backtracks and rejected steps are counted. A step that cannot move keeps the previous factors and still consumes its training subset, so matched runs stay on the same batch sequence.

## Training

Uniform quarters of the 2048-row training pool (512 rows), without replacement inside a batch, same indices on both sides, one persistent parameter state. Subset Grams are centred inside the subset. The subset gradient is not an unbiased full-gallery gradient.

Adam learning rate 0.03, gradient clip 5, 120 updates. These match the resampled-training quarter schedule. Seed 0 starts at identity. Seeds 1 and 2 use perturbation scale 0.05. The same seed is the initialisation seed and the sampling seed. Sampling draws do not depend on the parameter RNG. Shuffle controls use a separate permutation RNG.

Checkpoint selection reads full-validation excess only, then lower total distortion, then the earlier step. The penalised objective is recorded and is not the selection score.

## Conditions

Identity is \(M=I\). One-sided learns side A (language on V–L pairs, Qwen on L–L pairs) and keeps side B at identity. Shared-PC ties the reduced metric \(B\). For factor families that ties both \(A\) and \(D\). Each model still lifts \(B\) through its own \(U\). This does not identify a shared ambient metric. Separate learns both sides. Reverse one-sided is not refit.

## Panel

Six base language models crossed with three vision anchors (18 V–L pairs), plus the three release-matched Qwen/OLMo L–L pairs. Three seeds. No supplementary product models.

Shuffle control: the three `control_pairs` already named in the metric-stability config, \(\rho=0.4\), one-sided oblique diagonal and block-4 strong, three seeds. Partner rows are permuted on the training pool and on the validation gallery. Exploratory evaluation uses true correspondence.

## Complexity rule

Secondary only, not the checkpoint rule and not a reason to hide fixed-family scores:

\[
J_{\mathrm{val,select}}=(a-b)_{\mathrm{val}}-0.01\frac{C_{\mathrm{total}}}{q(q-1)/2}.
\]

\(C_{\mathrm{total}}\) counts untied off-diagonal weighting parameters. A shared \(D\) counts once. Separate \(D\) matrices count separately. Diagonal \(D\) contributes 0. exp(S) counts the off-diagonal entries of each untied \(S\). The term does not penalise basis parameters.

## Spectra

Compare sorted log-eigenvalues of induced \(B\). Also report the mean-subtracted log spectrum. Eigenvalues of \(D\) are factorization diagnostics. They are not eigenvalues of \(B\) unless \(A\) is orthogonal. The check is the generalised problem \(Dv=\lambda A^\top A v\).

## What this run is not

It does not extract features, download models, refit metric stability, or rewrite the resampled-training report. Shared-PC is not a shared semantic basis. Similar spectra are not shared semantic directions. Column normalisation does not uniquely identify \(A\) and \(D\).
