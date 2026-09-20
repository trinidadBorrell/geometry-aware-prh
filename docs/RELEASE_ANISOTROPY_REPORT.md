# Release alignment and anisotropic corrections — corrected report

**Status:** frozen (`prh-release-alignment-freeze-20260921`). Exploratory evaluation on the already-used COCO val2017 gallery (test n=1024). Final-layer protocol; **not** comparable to previous max-over-layers PRH means.

**Authoritative fitted outputs:** `/mnt/sdb1/prh-replication-work/results/release_anisotropy_repair/` (also mirrored as small JSON/PNG under `results/release_anisotropy_repair/` in this repo).

**Historical outputs retained:** `results/release_anisotropy/` and the same path on the work disk. Optimisation-dependent conclusions there are **superseded**. Native alignment, native anisotropy, and the two-sided max-budget bridge were **reused**, not refit.

**Question.** As approximately size-matched Qwen and OLMo *base* checkpoints advance, does native final-layer alignment rise, and are the anisotropic corrections that recover extra CKA consistent across partners and releases?

**Answer (after optimiser repair).** Native alignment is **not monotonic in recency** (unchanged). One-sided subspace reweighting **does** raise held-out CKA at every nested budget in the frozen grid; the original claim that large budgets collapse to identity was an **optimiser bug**, not a geometric fact. Uniform PCA-subspace weighting (`μ`) is a negligible share of distortion; transfer and partner agreement track the traceless part `S̃`. Amplified response signatures `G+` agree across releases more than spectrum-preserving Haar rotations of the same `S`. Shuffled correspondence does **not** produce true-test gains. Conditional stability of *directions* at `ρ=0.1` is moderate (mean `S̃` cosine 0.86 on frozen PCA), not a proof of unique semantic axes.

**Primary fitting objective (frozen before inspecting these test numbers):** CKA excess \(a-b\). Distortion is a hard budget \(D(M)=\|S\|_F^2/q\), not a Lagrange penalty. Selected budget `ρ` and attained `D` are reported separately. Validation-selected `D` is **not** “required correction.” Ratio \(a/b\) is reported but was not used for selection.

No further sweep is proposed.

## Lead answers (supported after repair)

1. **Native mNN / linear CKA vs recency.** Unchanged from the parent run. Not a monotone law. Qwen2.5-7B is the weakest of the three Qwen bases on every fixed partner; Qwen3-8B-Base is the strongest. OLMo-2-1124 is the strongest OLMo checkpoint against vision, not Olmo 3. See §1 (native tables reused).
2. **Held-out alignment after bounded one-sided reweighting.** Yes, at every `ρ>0` in the grid after incumbent retention. Base+supp VL mean test `a`: identity 0.379 → `ρ=0.1` 0.506 → `ρ=0.4` 0.537 → `ρ=1` 0.543 → max 0.546. LL: 0.751 → 0.877 → 0.886 → 0.887 → 0.887. Validation-selected operating points often use `ρ∈{0.4,1,(log 4)²}` (72/72 pairs `ρ≥0.4`); mean headline Δ`a` vs identity is +0.168 VL (all LMs). **Matched-budget** numbers, not a claim that extra `ρ` is required. Test tracks validation on the VL mean (no material overfit of the *mean* curve); 18/72 pairs have val-argmax `ρ` ≠ test-argmax `ρ`, with mean test-excess gap vs the test-best grid point of ~−3×10⁻⁵.
3. **Performance at matched distortion budgets.** Report `ρ` (cap) and attained `D` separately. At `ρ=0.1`, every VL pair attains `D=0.1` (0/27 identity). At max `ρ`, mean VL `D=0.81` (still below the cap for many pairs). mNN rises more slowly than CKA (VL 0.185 → 0.208 → 0.224 → 0.231 → 0.234).
4. **Cross-partner consistency beyond uniform weighting.** At `ρ=0.1`, `D ≈ μ² + \|S̃\|_F²/q` with mean VL `μ²=4.5×10⁻⁴` vs directional 0.0996. Vision-partner `S̃` cosine 0.794 (**27/27 defined**); language-partner 0.484 (90/90 defined). Raw `S` cosines are almost the same (0.791 / 0.475), so agreement is **not** an artefact of uniform subspace gain. Per-model vision `S̃` (3 pairs each) ranges 0.73–0.84; do not replace that panel with a mean over a handful of nonzero cells — after repair, all vision cells at `ρ=0.1` are defined.
5. **Transfer to excluded partners.** At `ρ=0.1`, freeze `M_{A|B}` and eval vs `C`: VL Δexcess vs identity +0.047 (n=189), gap vs a metric fit on `C` −0.079. Uniform-only transfer Δexcess ≈ 0. Direction-only (unfitted; 165/189 within original eig bounds) +0.065. LL transfer vs identity +0.015, gap −0.111. LOPO (held-out partner out of fit **and** val `ρ`): vision-anchor Δexcess +0.156 (n=27, mean selected `ρ=1.55`, mean `D=0.79`); OLMo partners of Qwen/supp +0.141; Qwen partners of OLMo +0.092; language-shared → vision +0.047. Unseen partners ≠ unseen examples.
6. **Cross-release `G+` vs orientation reference.** At `ρ=0.1`, 18 defined family×partner comparisons (6 per family). Mean `G+` cosine: Qwen-base 0.880, OLMo-base 0.845, Qwen3x-supp 0.893. Haar reference (50 seeded rotations per correction, same eigenvalues/`D`/identity residual): mean 95th percentile 0.31–0.41; **18/18** observed `G+` exceed that percentile. `G−` is weaker (0.58–0.65) and is not claimed as a match. This is a descriptive orientation reference, not a selection-aware test or a concept-identity test.
7. **Conditional stability and shuffled-fit.** At frozen `U`, `ρ=0.1`, three equal training subsets, earliest/latest bases × three vision anchors: all 36 subset fits are direction-defined; held-out Δexcess +0.119; pairwise `S̃` cosine 0.862 (n=36). At full-data val-selected `ρ∈{1.0,(log 4)²}` when it differs from 0.1, `S̃` cosine falls to ~0.69. These are **conditional** on frozen PCA and a frozen `ρ`; they do not rerun PCA or the full selection procedure. Near-zero `S̃` would be direction-undefined; that did not occur at `ρ=0.1`. Shuffle (four models × DINOv2 × three seeds; independent train/val shuffles; val-select on shuffled val; eval on true and shuffled test): 2/12 selected identity; 10/12 fit nonzero `D`. Mean true-test Δ`a` vs identity **−0.027**. Shuffled-test `a` ~0.08 vs identity true-test `a` ~0.33. Successful true-correspondence fits are **not** themselves null controls.

## Before / after (what changed)

| Earlier claim (parent `results/release_anisotropy`) | After repair | Disposition |
|---|---|---|
| Large one-sided budgets collapse to identity | Nested incumbents keep `ρ=0.1` solutions; larger `ρ` attains `D>0` (VL mean `D=0.81` at max cap) | **Withdrawn** as geometry; optimiser bug |
| One-sided gains only at `ρ=0.1`; max-ρ means = identity | Gains at every `ρ>0`; max-ρ VL `a=0.546` vs identity 0.379 | **Superseded** |
| Shuffle/stability at max `ρ` are identity, so they do not test interpreted corrections | Shuffle and stability rerun with nested fitting; shuffle true-test Δ`a` negative; `ρ=0.1` directions moderately stable | **Replaced** |
| Vision `S` cosine 0.84 (n=10 defined) | Vision `S̃` 0.79 (n=27/27 defined) | **Updated counts**; not a mean over a few survivors |
| Uniform vs directional not separated | `μ` negligible; uniform-only ablation ≈ identity; direction-only ≈ full at `ρ=0.1` VL (27/27 in bounds) | **New diagnostic** |
| No orientation reference for `G±` | Haar `Q diag(s) Qᵀ` in the frozen `q=32` subspace; `G+` above ref 95th in all defined `ρ=0.1` cells | **New diagnostic** |
| Two-sided LL `a` ~0.95 | Unchanged (not a nested-`ρ` fit) | **Retained** |
| Native recency ranking | Unchanged (same caches, PCA variance match abs_diff=0) | **Retained** |
| “Required correction” from val-selected `ρ` | Forbidden wording; report selected `ρ` and attained `D` | **Withdrawn language** |

## Optimiser diagnosis (not “nonconvexity”)

Recorded `qwen2-7b` vs DINOv2 `ρ=0.1` parent fit, re-evaluated **without updates**:

- Same train excess 0.31477 at every `ρ≥0.1`.
- Same `D=0.1`, feasible, spectral bounds OK.
- At `ρ=0` the projected `S` is zero (budget), so the score is not supposed to match.

Cause in the parent fitter: independent Adam per `ρ`; last iterate only; identity start hits repeated eigenvalues of `S=0` so `obj.requires_grad` is false and that start aborts; perturbation last-iterates then lose to `identity_explicit`; the feasible `ρ=0.1` `S` was never a candidate at larger `ρ`. `ρ` is an upper bound (`project_S` scales only if `D>ρ`), not an equality and not a change of objective. `params.clamp` on vech is not the budget.

Repair: evaluate identity and every smaller-budget incumbent; warm-start continuation as one of three starts; keep the best **feasible training** iterate including iteration 0; do not discard a valid incumbent because a later step is worse or nonfinite. Train excess is nondecreasing in `ρ` on all 360 pair×budget cells (0 failures). Two-sided bridge was a single spectral-clip `fit_metrics` call and already returned non-identity metrics; **not recomputed**.

## Model inventory and protocol

Unchanged from the parent report: same revisions, captions, splits, final-block pre-LN hook, PRH clip+L2, `q=32`, CKA-excess objective, partner sets. Native matrices and `pca.json` variance fractions were provenance-checked (recomputed train PCA matches parent `variance_fraction` to 0). Feature-cache SHA-256 values: `results/release_anisotropy_repair/feature_hashes.json`.

Supplementary 4B product panel remains **not** a matched base series.

## 1. Native alignment vs recency

Reused from `results/release_anisotropy/native_matrix.json`. Test n=1024.

**Qwen-base × OLMo-base (CKA a / mNN k=10)**

|  | OLMo-7B-0724 | OLMo-2-1124 | Olmo-3-1025 |
|---|---|---|---|
| Qwen2-7B | 0.703 / 0.542 | 0.588 / 0.460 | 0.645 / 0.495 |
| Qwen2.5-7B | 0.610 / 0.511 | 0.469 / 0.431 | 0.539 / 0.474 |
| Qwen3-8B-Base | 0.749 / 0.502 | 0.748 / 0.478 | 0.801 / 0.517 |

**Vision–language**

|  | DINOv2 | IN21k | CLIP-LAION |
|---|---|---|---|
| Qwen2-7B | 0.234 / 0.151 | 0.279 / 0.159 | 0.319 / 0.161 |
| Qwen2.5-7B | 0.181 / 0.144 | 0.215 / 0.152 | 0.246 / 0.157 |
| Qwen3-8B-Base | 0.346 / 0.160 | 0.397 / 0.171 | 0.448 / 0.172 |
| OLMo-7B-0724 | 0.347 / 0.191 | 0.416 / 0.196 | 0.472 / 0.205 |
| OLMo-2-1124 | 0.428 / 0.234 | 0.503 / 0.245 | 0.572 / 0.250 |
| Olmo-3-1025 | 0.393 / 0.205 | 0.464 / 0.211 | 0.526 / 0.224 |

Base VL mean CKA 0.377. Recency is observational: architecture, data, post-training, and size also change.

## 2. One-sided alignment vs distortion (repaired)

All language models, test split. Identity at `ρ=0` matches native CKA/mNN.

| ρ (cap) | VL a | VL excess | VL mNN | VL mean D | VL # identity | LL a | LL excess | LL mean D |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.379 | 0.301 | 0.185 | 0 | 27/27 | 0.751 | 0.700 | 0 |
| 0.1 | 0.506 | 0.428 | 0.208 | 0.100 | 0/27 | 0.877 | 0.826 | 0.100 |
| 0.4 | 0.537 | 0.468 | 0.224 | 0.400 | 0/27 | 0.886 | 0.839 | 0.346 |
| 1.0 | 0.543 | 0.476 | 0.231 | 0.660 | 0/27 | 0.887 | 0.840 | 0.454 |
| (log 4)² | 0.546 | 0.480 | 0.234 | 0.811 | 0/27 | 0.887 | 0.840 | 0.468 |

**Validation-selected operating point vs identity (test)** — selected `ρ` is a grid index, not a physical requirement.

| panel | pairing | mean selected ρ | mean attained D | Δa | Δexcess |
|---|---|---|---|---|---|
| base | VL | 1.61 | 0.85 | +0.175 | +0.185 |
| base | LL | 1.21 | 0.53 | +0.158 | +0.162 |
| supp | VL | 1.61 | 0.73 | +0.154 | +0.166 |
| supp | LL | 0.88 | 0.35 | +0.091 | +0.098 |

**Two-sided bridge** (parent, Qwen-base × OLMo-base only; **not** mixed into one-sided curves): test `a` 0.922–0.966. Per-side `D` 0.58–1.35. Joint reweighting is a different estimator than one-sided `M_{A|B}`.

Figure: `results/release_anisotropy_repair/excess_vs_rho_vl.png`.

## 3. Uniform vs directional decomposition

`S = μ I_q + S̃`, `tr(S̃)=0`, `D = μ² + \|S̃\|_F²/q`.

Unfitted ablations `B=exp(μ)I` and `B=exp(S̃)` (eigenvalues of the latter may leave `[1/4,4]`; those cases are excluded from matched-budget rankings, not clipped).

At `ρ=0.1` VL (n=27): full Δexcess +0.127; uniform-only ≈ 0; direction-only +0.127 (27/27 in bounds). LL: 8/45 direction-only out of bounds; in-bound mean Δexcess +0.111 vs full +0.127. Transfer follows the same split: uniform-only Δexcess ≈ 0.

## 4. Consistency, transfer, LOPO

See lead items 4–5. All-partner `S̃` cosine at `ρ=0.1` is 0.417 (n=252/252 defined). That all-partner mean mixes vision and language groups and is **not** the headline consistency number.

## 5. Signatures and Haar orientation reference

`G± = H Z ΔM± Zᵀ H` with the `q×q` PSD split of `B−I` (equivalent to the ambient split of `U(B−I)Uᵀ`). Haar: `S_random = Q diag(s) Qᵀ`, `Q` Haar in the frozen PCA subspace; 50 rotations; seed 20260921. Not used to select metrics.

At `ρ=0`, all `G+` near-zero (identity). At `ρ≥0.1`, every defined comparison in the pre-registered family×partner set has observed `G+` above the Haar 95th percentile (n=6 per family per `ρ`).

## 6. Native anisotropy

Unchanged (`native_anisotropy.json`). Qwen participation rank 33.0 → 21.4 → 76.3; OLMo 60.2 → 75.4 → 76.3. Effective rank is not a count of semantic concepts. One-sided `ρ=0.1` does not collapse `r_eff` the way the earlier two-sided experiment did.

## Controls and remaining numerical note

- `ρ=0` identity reproduces native CKA/mNN; train monotonicity 0 failures; `D≤ρ` on all saved fits; parent `ρ=0.1` vech is budget-invariant for `ρ≥0.1`.
- Direct vs contracted CKA `a` on three sampled headline pairs differs by ~2×10⁻⁵ (check threshold was 1e-6). Frozen as a **numerical discrepancy**, not a new analysis. No additional sweep.
- Shuffle and subset stability: see lead item 7. Failed/weak shuffle seeds are kept (qwen3-8b seed 2 and olmo-7b seed 0 selected identity).
- Units: `tests/test_release_anisotropy.py` (12 passed on the Ubuntu venv after the factor-gram patch).

## What this does not show

- This is exploratory evaluation on an already-used COCO gallery.
- Final-layer results do not establish all-layer trends.
- Release comparisons are observational.
- Pairwise cells are dependent.
- Effective rank is not a count of semantic concepts.
- Directional stability may remain unresolved even if alignment is stable. Conditional `S̃` cosine 0.86 at `ρ=0.1` is not unique-axis identification.
- A higher version number does not cause alignment.
- Higher CKA is not higher retrieval (mNN gains remain smaller).
- Haar exceedance is not a semantic-identification test.

## Reproduction

Completed results **do not rerun** unless you pass `--force` (parent) or `--force` / `--resume-after` (repair). No experiment jobs are left running. vLLM was not started or stopped in this pass.

```bash
export HF_HOME=/mnt/sdb1/prh-replication-work/hf
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
cd /mnt/sdb1/prh-replication-work/repo
/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests/test_release_anisotropy.py -q
# Repair (skips extract; reuses native/two_sided). Explicit rerun only:
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy_repair.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 0 --force
# Resume signatures/shuffle/stability only (fits already on disk):
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy_repair.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 0 --resume-after lopo
```

Parent (historical; do not use for fitted one-sided claims): `scripts/run_release_anisotropy.py --force`.
