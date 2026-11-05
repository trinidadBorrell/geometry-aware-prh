# exp-022: is the lexical floor real? (validation before any scale-up)

**Status:** phase 1 running · **Owner:** oddharak · **Outputs (planned):** `/workspace/results/oddharak/exp-022-lexical-validation/`

Reviews `reports/lexical_floor_research_implementation_plan.md` and replaces its milestones A-F for now. Everything in phase 1 runs on data already on the volume, on one CPU pod.

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
