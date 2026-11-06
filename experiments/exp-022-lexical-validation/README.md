# exp-022: is the lexical floor real? (validation before any scale-up)

**Status:** phases 1 and 2 done · **Owner:** oddharak · **Outputs (planned):** `/workspace/results/oddharak/exp-022-lexical-validation/`

Reviews `reports/lexical_floor_research_implementation_plan.md` and replaces its milestones A-F for now. Everything in phase 1 runs on data already on the volume, on one CPU pod.

## Phase 2 results (2026-11-06)

Runs: RTX PRO 4000, git `60ce80c` plus uncommitted `phase2.py` and `extract_lm.py` (copies with the outputs). About $4.15 in total across three pods. Tables: `results/exp-022-lexical-validation/phase2/summary.md`; figure: `phase2/fig_phase2.png`. Numbers are excess over the null at n = 10240 and the best ViT layer, unless marked. Pairwise numbers use 50 permutations, for means only.

**Checks that passed:** shuffled layer 0 equals unshuffled layer 0 (row cosine >= 0.999999); unshuffled open_llama_7b and bloomz-7b1 reproduce exp-016's features (row cosine >= 0.9999 on all layers); shuffle seed 0 vs 1 differ by <= 0.002; ridge recovers the three known synthetic answers.

### 1. Word order: deep alignment drops when tokens are shuffled

Pythia-410M, open_llama_7b, bloomz-7b1 x DINOv2, CLIP (6 pairs), mean over pairs. "Late" is depth 0.83, "last" the last layer.

| Captions | Measure | TF-IDF variants, no model | Layer 0 | Late, intact | Late, shuffled | Last, intact | Last, shuffled |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini | k = 10 | 0.176-0.252 | 0.182 | 0.235 | 0.167 | 0.239 | 0.164 |
| Gemini | CKA | 0.508-0.656 | 0.563 | 0.649 | 0.606 | 0.645 | 0.587 |
| Original | k = 10 | 0.032-0.037 | 0.027 | 0.070 | 0.030 | 0.065 | 0.025 |
| Original | CKA | 0.078-0.081 | 0.105 | 0.295 | 0.139 | 0.306 | 0.161 |

- **k = 10:** with order destroyed, deep layers fall back to about the layer-0 level or below on both caption sets. Their gain at k = 10 needs word order.
- **CKA on original captions:** about half the deep value survives shuffling. The rest needs order, and what survives is still above layer 0 (0.139-0.161 vs 0.105).
- **CKA on Gemini captions:** shuffling costs much less (0.649 to 0.606).

Shuffling also gives each caption an unnatural sequence, which trained blocks never saw, so this is an upper bound on what order contributes, not a clean decomposition.

### 2. Long human captions also have a large word floor (DOCCI)

DOCCI: 10240 photos, median 132 tokens. Different images from WIT, so no pairing with Gemini.

| x DINOv2 | k = 10 | k = 1000 | CKA |
|---|---:|---:|---:|
| TF | 0.043 | 0.066 | 0.136 |
| TF-IDF | 0.158 | 0.137 | 0.325 |
| Sublinear TF-IDF | 0.176 | 0.167 | 0.332 |
| Word TF-IDF, no stopwords | 0.170 | 0.186 | 0.367 |
| Trained layer 0 (4 LMs) | 0.068-0.072 | 0.106-0.129 | 0.240-0.308 |
| Pythia 160M / 410M, deep | 0.125 / 0.136 | 0.243 / 0.263 | 0.453 / 0.488 |
| open_llama_7b / bloomz-7b1, deep | 0.178 / 0.192 | 0.298 / 0.306 | 0.550 / 0.539 |

- **Not just the generator:** human long captions also give a large floor at k = 10, passing Pythia's deep layers, so the Gemini floor is not only an artefact of the generator.
- **Smaller than Gemini:** CKA is well below Gemini's (0.33-0.37 vs 0.59-0.67), though the images and length differ.
- **Depth adds clearly in CKA and at large k.** Trained layer 0 sits far below TF-IDF at k = 10 here, unlike on Gemini.
- **Other encoders:** CLIP and AugReg follow DINOv2. MAE is lower at k = 10 (0.079-0.091). A random ViT stays low (k = 10 <= 0.010, CKA <= 0.062).

### 3. Pythia on original captions

| x DINOv2, final checkpoint | k = 10 layer 0 | k = 10 deep | CKA layer 0 | CKA deep |
|---|---:|---:|---:|---:|
| 160M, two seeds | 0.024 | 0.028 | 0.110-0.111 | 0.128-0.150 |
| 410M, two seeds | 0.023 | 0.042 | 0.110-0.113 | 0.248-0.261 |

For comparison, TF-IDF gives 0.032 and 0.079, word TF-IDF 0.038 and 0.112. Random init is at the word-count level (k = 10 0.018-0.019, CKA 0.054-0.057; TF 0.020, 0.063). So Pythia-160M barely beats word statistics on original captions, while 410M and the larger LMs (CKA 0.278-0.355) clearly do: on human short captions, alignment beyond words scales with model size in these few models. Sizes differ in peak learning rate, so this is not a clean size effect.

### 4. Table and blocks need each other

Pythia-410M x DINOv2, Gemini / original captions, last layer:

| Table | Blocks | k = 10 | CKA |
|---|---|---:|---:|
| Trained | Trained | 0.218 / 0.042 | 0.683 / 0.261 |
| Trained | Random | 0.068 / 0.017 | 0.355 / 0.074 |
| Random | Trained | 0.042 / 0.010 | 0.316 / 0.032 |
| Random | Random | 0.043 / 0.015 | 0.158 / 0.050 |

Neither alone reaches the trained pair at depth. Trained blocks on a random table do not beat fully random k = 10. 160M gives the same pattern with erratic CKA in random-table cells (e.g. 0.071, 0.100, 0.168, 0.047 across layers), probably dominant activation directions, so read CKA cautiously in those cells.

### 5. Random-depth signal needs token order (claim E)

Shuffling removes most of what a random network keeps at depth (160M x DINOv2, Gemini, last layer: k = 10 0.046 to 0.003; CKA 0.160 to 0.080). 410M matches. So the gap above word-count retention is not a bag-of-words effect. It could reflect shared phrases or positions passing through random blocks (inference, not tested). It is not evidence of learned content.

### 6. Held-out prediction: words and LMs carry partly different information

Ridge from PCA inputs to the ViT's last layer, 5 folds, R^2 (retrieval top-1):

| Target, captions | Words | Pythia-410M last | Larger LMs last | Words + LM |
|---|---:|---:|---:|---:|
| DINOv2, Gemini | 0.41-0.42 (0.36-0.38) | 0.36 (0.31) | 0.41-0.43 (0.38-0.43) | 0.44-0.48 (0.43-0.52) |
| CLIP, Gemini | 0.39 (0.59-0.63) | 0.38 (0.62) | 0.43-0.44 (0.72-0.75) | 0.43-0.46 (0.74-0.80) |
| DINOv2, original | 0.08-0.09 (0.02) | 0.15 (0.04) | 0.20-0.23 (0.07-0.09) | 0.16-0.22 (0.05-0.09) |
| CLIP, original | 0.08 (0.03) | 0.18 (0.13) | 0.24-0.26 (0.23-0.29) | 0.19-0.26 (0.15-0.27) |

- **Gemini:** words predict the ViT about as well as an LM, and the two together do better than either. Neither contains the other.
- **Original:** the LM makes words redundant: adding them changes little (at most 0.01 in R^2, some of it fit noise).
- **Limit:** PCA to 256 dimensions may discard rare-word signal, which penalises word inputs, and 'last' is one layer choice.

### Claims after phase 2

| Claim | Status |
|---|---|
| Lexical alignment is non-negligible | **Supported.** Well above null everywhere. On long captions, Gemini or human, word statistics are comparable to or above small LMs at k = 10 |
| Lexical alignment explains everything | **No.** Larger LMs stay above word statistics in CKA and at large k, and on original captions LMs are far above them |
| The floor comes only from Gemini | **No,** DOCCI shows it with human captions, though at lower CKA |
| Depth adds beyond words | **Yes, mostly through word order at k = 10,** and most clearly on short human captions |

**Limits:** few LMs, two ViT targets for shuffle and factorial, one human dense dataset on different images. Shuffle and DOCCI features were on pod disks and are gone (regenerable); condensed results are in `phase2/results.tgz` and `phase2/big_shuffle_grids.json` on the volume. The volume is near quota, so `d2s` files and DOCCI copies from this run were removed.

## Phase 1 results (2026-11-05)

Run: `cpu3g` 8 vCPU, git `60ce80c`, all stages exit 0, 44 min of job time, about $0.26 of pod time. Scoring: `analyze.py`; tables in `results/exp-022-lexical-validation/summary.md`. Numbers are excess over the null at n = 10240 unless marked debiased. Exact null means agreed with 30 real permutations to 3 decimals on all four checked rows.

| Check | Verdict | Result |
|---|---|---|
| V1 identity | **Pass** | Frequencies x real table vs stored layer 0: mean row cosine >= 0.999998 on all 5 tables |
| V2 numerics | **Pass** | 8 stored GPU cells vs exact float64: worst difference 0.00008 |
| V3 estimator | **Pass** | Debiased within 0.016 of excess. TF < TF-IDF < trained layer 0 for DINOv2. For CLIP TF-IDF is above layer 0 (0.523 vs 0.492-0.500), as it was before debiasing. Debiased values are stable from n = 1024 up; biased TF-IDF is not (0.507 at 1024, 0.613 at 10240) |
| V4 selection | **Partial fail** | Split-half selection changes nothing (<= 0.001). Fixed cells do move: DINOv2 CKA at its last layer is TF 0.300, TF-IDF 0.485, Pythia layer 0 0.457-0.467, against 0.411, 0.613, 0.654-0.663 at its mid-layer best. CLIP does not move. So best-cell numbers are dropped from claims |
| V5 concentration | **Pass** | 25 rows removed, nothing moves > 0.001. Within 19 image-type clusters TF-IDF keeps mutual 10-NN 0.221 and debiased CKA 0.577 (Gemini), 0.048 and 0.097 (original) |
| V6 template | **Pass** | 0 of 10240 captions carry the JSON wrapper |
| V7 definition | **Fail, upward** | TF-IDF moved by more than 0.03, always towards a higher floor except plain word-level TF-IDF (-0.020). Tokenizer and out-of-sample idf do not matter (<= 0.017 and <= 0.003 in CKA) |
| V8 length vs content | **Both** | At 12 tokens, first tokens of Gemini give TF-IDF CKA 0.290 vs 0.079 for originals (content); 12 random Gemini tokens give 0.036 (length needed) |
| V9 generality | **Pass** | Floor present for all 4 ViTs and 3 image towers; random ViT low (TF-IDF CKA 0.110 Gemini, 0.012 original) |
| V10 other families | See below | Claim D holds only for small Pythia on Gemini captions |
| V11 depth | **Gap narrows, stays open** | Last random layer vision CKA 0.158-0.160; TF-IDF retention predicts 0.099-0.107, TF 0.065-0.074 |

### Corrections to exp-021

A second float32 bug: exp-021 computed CKA as a float32 matrix-vector product, which loses accuracy at high values on the CPU and depends on the thread count (locally 0.9412 on 1 thread, exact 0.9561). This is why two "open" results existed. Values below about 0.5 are little affected.

| Quantity (Gemini) | exp-021 reported | Correct | Consequence |
|---|---:|---:|---|
| TF x pythia-1.4b CKA | 0.691 | 0.712 | The 0.72 prediction **holds** (off by 0.008), not fails |
| Real step-0 layer 0 x TF CKA | 0.906-0.913 | 0.959-0.968 | Matches the formula (0.957, 0.968) and Gaussian tables (0.956, 0.967) |
| Gaussian table x pythia-1.4b | 0.656, 0.666 | 0.681, 0.684 (5 seeds) | Real tables (0.671-0.695) are not above Gaussian ones |
| TF-IDF x DINOv2 CKA excess | 0.600 | 0.613 | Small |
| TF-IDF x pythia-1.4b CKA | 0.788 | 0.829 | Small |

Still failed: r_B 68.9 against the inferred 83-97, and mutual 10-NN 0.120 below the predicted 0.127-0.14.

**Retracted:** "training adds mostly coarse structure because word overlap is flat at large k". That flatness is specific to linear TF-IDF. With stopwords removed or sublinear counts the lexical floor keeps rising with k.

### The lexical floor depends strongly on weighting

Gemini captions, no model, against what trained LMs reach (Pythia 160M / 410M; exp-016's 1.7B-13B LMs):

| Excess | TF | TF-IDF | Binary | Sublinear TF-IDF | Word TF-IDF, no stopwords | Pythia deep | Larger LMs deep |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 k = 10 | 0.119 | 0.186 | 0.202 | 0.252 | 0.230 | 0.190-0.218 | 0.234-0.280 |
| DINOv2 k = 1000 | 0.208 | 0.231 | 0.362 | 0.383 | 0.391 | 0.392-0.414 | 0.413-0.446 |
| DINOv2 CKA | 0.411 | 0.613 | 0.670 | 0.656 | 0.651 | 0.648-0.704 | 0.713-0.778 |
| CLIP k = 10 | 0.109 | 0.176 | 0.184 | 0.242 | 0.216 | 0.184-0.216 | 0.235-0.264 |
| CLIP k = 1000 | 0.192 | 0.225 | 0.308 | 0.334 | 0.331 | 0.336-0.359 | 0.359-0.369 |
| CLIP CKA | 0.319 | 0.508 | 0.512 | 0.553 | 0.554 | 0.534-0.568 | 0.583-0.594 |

Best cells here for comparability with earlier tables. At the fixed last x last cell against DINOv2, sublinear TF-IDF gives k = 10 0.242 and CKA 0.500; Pythia gives 0.169-0.196 and 0.426-0.478, larger LMs 0.212-0.229 and 0.481-0.529. Several definitions were compared, so quote the range, not the maximum. Sublinear counts and stopword removal are textbook defaults, not tuned here.

### Original captions tell a different story

| x DINOv2 / CLIP | Best lexical definition | Trained layer 0 | Deep layers (best) |
|---|---:|---:|---:|
| Mutual 10-NN | 0.038 / 0.042 | 0.024-0.028 / 0.028-0.033 | 0.055-0.089 / 0.067-0.111 |
| CKA | 0.112 / 0.105 | 0.085-0.114 / 0.085-0.125 | 0.278-0.355 / 0.325-0.387 |

Four LMs (bloomz-1b7, bloomz-7b1, open_llama_7b, llama-13b). Pythia has no original-caption features yet. Depth clearly adds here, and the table does little. These floors are concentrated in few samples: 78-86% of captions have zero neighbour overlap at k = 10 for TF and TF-IDF, against 26-38% on Gemini.

### Claims after phase 1

| Claim | Status |
|---|---|
| A: step-0 layer 0 is frequencies x table | **Confirmed**, exactly |
| B: TF-IDF nearly equals a trained table on Gemini | **Restated:** simple lexical baselines reach or pass trained tables on Gemini, and come close to or pass Pythia's deep layers. Larger LMs stay above them in CKA against DINOv2 and at k = 1000. Plain TF-IDF is not special |
| C: floor is small on original captions | **Confirmed**, for 4 non-Pythia LMs, and depth adds a lot there |
| D: the table carries most of the rise | **Limited** to small Pythia on Gemini. Weaker for larger LMs, absent on original captions |
| E: random depth keeps more than TF retention predicts | **Open**, gap narrowed by TF-IDF retention |

Trained tables are also not simply implementing idf through row norms: Spearman(row norm, idf) is -0.23 to -0.34 at the final checkpoint, i.e. rarer tokens have smaller norms. Trained layers do sit closer to TF-IDF than to TF (160m / 410m last layer: 0.74 vs 0.53).

### Decision

By the rule below: V1-V3 pass, V4's stated remedy is applied (best-cell numbers dropped from claims), and claim B survives V5-V7 in a stronger form. This goes to phase 2. Not supported by phase 1: any large grid.

**Limits:** one image set and one generator; Gemini saw the images, so its words may already encode what the ViT sees. 83% of original captions were detected as English by a detector that is noisy on short text; all are ASCII, and an English-only rerun was not done. `lc.scores` still uses float32 matrix products, so large-n scores must not be run on CPU without fixing it; exp-016 and exp-018 ran on GPU and match exact values at the 8 checked cells.

## Claims under test (from exp-021)

| ID | Claim | Current evidence |
|---|---|---|
| A | Step-0 layer 0 is exactly token frequencies x table | Gaussian table matches against ViTs; one stored score reproduced; CKA with the kernel unexplained (0.91 vs 0.96) |
| B | On Gemini captions, TF-IDF nearly equals a trained table at k = 10 and in CKA, not at large k | One tokenizer, one idf, best-cell scores |
| C | On original captions the floor is small | TF CKA excess 0.063, TF-IDF 0.079; trained rows from other models |
| D | The trained table carries most of the rise | 4 Pythia series, Gemini only |
| E | Random depth keeps more vision signal than TF retention predicts | Rough retention calculation |

## Threats nobody has tested yet

1. **Length vs content.** Gemini captions have 613 tokens (median), originals 12. "Descriptive captions" and "long captions" are fully confounded in exp-021.
2. **Estimator bias.** Biased CKA with kernels of very different rank: r_B is 69 for TF and 2397 for original-caption TF-IDF, whose null mean reaches 0.047-0.060. Subtracting a null mean is not the same as a debiased estimate.
3. **Selection.** Layer 0 takes the max over 7 cells, deep layers over 42, on the same samples used to score them.
4. **Concentration.** The agreement could come from a few image types (maps, portraits, flags) whose words and pixels both cluster, or from near-duplicates. The source deduplicates by perceptual hash and caption text, but details are not given.
5. **Prompt template.** The recaption prompt asks for JSON with a `"500w"` key. The dataset card does not say whether the wrapper was stripped. Fixed template tokens would distort TF and flatter TF-IDF.
6. **Lexical definition.** All results use one tokenizer (GPT-NeoX subwords, case-sensitive) and an idf from the same 10240 captions.
7. **Language.** The source calls this English WIT. Not yet checked on our rows.
8. **Numerics.** The float32 CKA bug found in exp-021 was checked against one stored GPU value only.
9. **One generator.** Everything rests on `gemini-3-flash-preview` and one prompt. Phase 1 cannot fix this; phase 2 can.

## Corrections to exp-021 before anything else

- Remove the "share of trained" and "share of rise" rows and the "gain" wording: CKA differences are not variance components. Report levels and differences only.
- Report full layer trajectories beside best cells.

## Phase 1: checks on cached data (CPU, no forward passes)

| Check | What runs | Pass rule (fixed before running) | If it fails |
|---|---|---|---|
| **V1 identity** | Download the real input tables (160m and 410m at step 0 and final, 1.4b final). Compute frequencies x table and compare with stored layer-0 features. Also CKA with TF without the clamp shared across layers | Mean row cosine >= 0.999 | Stop and find the bug. Separately explains or fails to explain 0.91 vs 0.96 and the 1.4b miss |
| **V2 numerics** | Exact float64 CKA vs stored scores: at least 8 cells from exp-016 and exp-018 | Within 0.001 | Recompute affected results |
| **V3 estimator** | Debiased CKA for every row; 10 disjoint subsets of 1024; nested 2048 / 4096 / 10240 with k scaled to n | Order TF < TF-IDF <= trained layer 0 unchanged; debiased within 0.03 of excess at n = 10240 | Debiased becomes the primary measure and claims B and C are restated |
| **V4 selection** | Same rows at a fixed cell (vision final layer; LM depth 0.75) and with the cell chosen on half the samples, scored on the other half | Headline rows move <= 0.02 | Drop best-cell numbers from claims |
| **V5 concentration** | Per-sample overlap distribution; remove near-duplicates; k-means on DINOv2 (20 clusters), score within clusters | Removing duplicates moves rows <= 0.02. Within-cluster TF-IDF mutual 10-NN stays above null by >= 0.03 | Floor is reframed as image-type words, not fine content |
| **V6 template** | Look at raw caption strings; strip any wrapper; mask the top 10 / 100 / 500 tokens | Stripping moves rows <= 0.02 | Rerun exp-021 on cleaned captions |
| **V7 definition** | Binary presence; idf from the other half of captions; word-level TF-IDF via scikit-learn, with and without English stopwords; each exp-016 LM's own tokenizer | TF-IDF rows move <= 0.03 | Claim B holds only for that definition, and we say so |
| **V8 length vs content** | Gemini cut to 12, 25, 50, 100, 200, 400 tokens, by first tokens and by random sample, plus a language check on originals | No pass or fail. At 12 tokens vs original TF-IDF CKA excess: > 0.05 means content, <= 0.02 means length | Decides how the caption effect is described |
| **V9 generality** | TF and TF-IDF against all 4 cached ViTs, the 3 dual-encoder image towers and the random-init ViT | TF-IDF floor present for all 4 ViTs; random ViT no higher than the LLM values (CKA ~0.15 Gemini, ~0.04 original) | Claim is limited to the encoders where it holds |
| **V10 other families** | Full trajectories from exp-016 `pairs/*.npz`: 4 LLMs x 4 ViTs, both caption sets. Includes trained layer 0 on original captions | No pass or fail. Tests claims C and D without a GPU | Claim D is limited to Pythia |
| **V11 depth** | TF-IDF retention across step-0 layers, beside TF retention | No pass or fail. Does TF-IDF retention close the gap in claim E? | Then the gap is reweighting, not architecture, as far as this test can tell |

Pass rules are judgement calls. Change them now, not after seeing results.

**Readouts:** mutual kNN at k = 10, 30, 100, 300, 1000 and linear CKA, as in exp-018 and exp-021. Null means are exact rather than sampled: k / (n - 1) for mutual kNN and tr(A) tr(B) / (n - 1) for CKA of centred unit kernels, both checked against 30 real permutations on the TF and TF-IDF rows x DINOv2. No p-values are claimed from this. No RSA and no sample bootstrap of kNN: resampled rows become each other's neighbours, so spread comes from disjoint subsets.

## Decision after phase 1

- **V1 or V2 fails:** stop, fix, rerun.
- **V1-V4 pass, and claim B or D survives V5-V7:** go to phase 2.
- **Claims B and D both collapse:** no scale-up. Write up the narrow result: the step-0 floor is lexical, and permutation nulls miss it.
- **A large grid is not an outcome of this phase.** It needs a named question that phase 2 leaves open.

## Phase 2: only after that decision (GPU, priced first)

1. **Pythia on original captions:** 4 series at 6 checkpoints, reusing exp-018's runner with `CAPTIONS=orig`.
2. **Within-caption token shuffle:** shuffle token IDs, seeded, through a final and a step-0 model on both caption sets. Layer 0 must not change at all, which is a check in itself. Any change at depth measures word order and context. This is the most direct test of whether composition matters.
3. **Table x blocks factorial:** random or trained table crossed with random or trained blocks, for claim E.
4. **A second caption source:** DOCCI human dense captions if the images can be matched, and only then new Gemini short / medium / detailed captions if V8 leaves the question open.
5. **Does depth add beyond words?** Held-out ridge from exp-017 `maps.py`: predict the ViT from TF-IDF, from deep features and from both. Calibrate on synthetic data first. Used instead of residual kernels.

## Implementation

- **Same shape as earlier experiments:** `experiments/exp-022-lexical-validation/run.py` with stages, `pairs`-style JSONL plus `summary.md`, `infra/runpod/run_exp022.sh` with self-termination, and results under the volume path above.
- **Reuse:** `M16.prepared_layers` (PRH clamp and norm), `geoprh.knn_sweep`, `d2.parquet` order, exp-016 / exp-018 features and `pairs/*.npz`, exp-006's `_hsic_unbiased` logic in float64.
- **New shared code, small:** `src/geoprh/lexical.py` (token matrix, TF, binary, TF-IDF, truncation, masks), moved out of exp-021, and `tests/test_lexical.py`: exact toy matrices, padding excluded, feature-space CKA equals kernel CKA, frequencies x table equals a manual mean, shuffle leaves the kernel unchanged.
- **Fix:** `lc.build_cache` centres and normalises in float64, with a regression test.
- **Run-time guard:** before scoring, reproduce one stored exp-018 score and assert a match, as exp-021 did. It caught a real bug.

## Cost

Phase 1 is about 170 pairs at ~20 s each plus setup and table downloads: about 2 h on `cpu3g` 8 vCPU at $0.32/h, about $0.65, upper bound $1. Tables are ~4 GB on the container disk. Nothing goes in the volume except results.
