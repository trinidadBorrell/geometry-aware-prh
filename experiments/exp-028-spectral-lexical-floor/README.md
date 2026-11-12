# exp-028: the spectral lexical floor. How much of what LMs add above word counts is co-occurrence smoothing?

**Status:** done, 2026-11-12 · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-028-spectral-lexical-floor/` (copy in local `results/exp-028-spectral-lexical-floor/`) · **Code:** `spectral.py`, `run.py`, `analyze.py`, `tests/test_spectral_floor.py`, `infra/runpod/run_exp028.sh`, uncommitted, on `ef2baa9`

## Results (2026-11-12)

**Summary:** Word vectors fitted on 10-20k captions of the same kind, with no language model, reproduce most of what LLMs add above the word floor at the coarse scale:
- **Gemini captions:** 84-92% of the gap in CKA, and 53-92% at k = 1000;
- **original captions:** 46-59% of the gap at k = 1000 and in CKA;
- **k = 10:** none of it. The fine scale needs word order (exp-026).

The mechanism is word similarity, not caption-graph smoothing: the spectral kernels of the caption graph (T, N) fall *below* the plain floor on Gemini captions.

Rows are from the first run (k_s up to 512). The second run recomputed kernels with k_s <= 256 in 2.5 min and reused the finished rows. Pipeline check: the word floor reproduces exp-022's stored row (CKA within 2e-6, 10-NN within 2e-4). Tables: `summary.md`; figure: `fig_spectral.png`.

### Gap closure, split-half (chosen on half A, scored on half B; k halved on halves)

(best inductive kernel - best floor) / (LLM best cell - best floor), LLM = mean of the 4 exp-016 LLMs:

| Caption set | Target | k = 10 | k = 100 | k = 1000 | CKA |
|---|---|---:|---:|---:|---:|
| Original | DINOv2 | -13% | 15% | **57%** | **59%** |
| Original | CLIP | -8% | 14% | 46% | **52%** |
| Gemini | DINOv2 | n/a | 41% | **92%** | **92%** |
| Gemini | CLIP | n/a | 13% | **53%** | **84%** |

On Gemini at k = 10 the denominator is tiny (LLMs 0.250 vs floor 0.244), so the ratio means nothing. The best inductive kernel is W in every cell, usually at k_s = 512, the top of the range.

Levels on half B, DINOv2:

| | Original k = 1000 | Original CKA | Gemini k = 1000 | Gemini CKA |
|---|---:|---:|---:|---:|
| Best word floor (F) | 0.038 | 0.095 | 0.397 | 0.634 |
| Caption graph, transductive (T) | 0.069 | 0.161 | 0.360 | 0.502 |
| Caption graph, Nystrom from 20k anchors (N) | 0.068 | 0.168 | 0.308 | 0.474 |
| PPMI word vectors, 20k anchors (W) | 0.117 | 0.227 | 0.434 | 0.738 |
| LLMs, mean of 4 | 0.176 | 0.317 | 0.437 | 0.748 |

On all 10240 images (best cell, Gemini, CLIP), W also beats trained LLM layer 0 in CKA (0.579 vs 0.485) and at k = 1000 (0.351 vs 0.287).

### Predictions

| | Prediction | Result | Verdict |
|---|---|---|---|
| S1 | T beats the best floor at k = 1000 on Gemini | -0.036, -0.041 | **fails** |
| S2 | No new kernel beats the floor by more than 0.02 at k = 10 | largest difference -0.002 | holds |
| S3 | Original captions: W gains most in CKA, T / N little | W +0.12 to +0.13; T / N +0.06 to +0.08 | holds |
| S4 | Eigenvalue-weighted >= unweighted in CKA | +0.004 to +0.018 | holds |

### Reading

1. **By the rule fixed in advance (>= 50% at k = 1000 or in CKA), the LLMs' coarse gain above the word floor is largely reproducible from co-occurrence statistics** in all four cells. It is strongest on Gemini captions, where W comes within 0.01 of the LLMs.
   - Depth's coarse gain is therefore not evidence of composition.
   - Its fine-scale gain (k = 10) is untouched, and exp-026 ties that to word order.
2. **What carries it is word similarity, not diffusion over captions.**
   - The caption-graph eigenmap smooths away detail that the floor keeps: below the floor on Gemini, and only modestly above it at large k on short original captions.
   - So the spectral motivation of the design fails in the form tested. What reproduces the gain is the word table, learned like word2vec / PPMI from co-occurrence.
   - This fits exp-026's reading of trained layer 0 ("learned word similarity helps coarse neighbourhoods"). W is a better table than the LLMs' own layer 0.
3. **Original captions are not saturated.** From 10k to 20k anchors, W gains +0.016 at k = 1000 and +0.03 in CKA there, while Gemini is flat (within 0.004). The original-caption closure is a lower bound.

### Limits

- **In-domain advantage.** W's co-occurrence comes from captions of the same source and the same generator (for Gemini, written by the same model that wrote the evaluation captions). LLMs see general text. A same-generator vocabulary may carry visual regularities a general corpus does not.
- **Range edge.** The selected k_s is usually 512, the top of the range, so W may not be at its best.
- **Few models.** 4 LLMs and 2 ViTs, one image set.
- **Nystrom oddity.** N gets worse from 10k to 20k anchors on Gemini (-0.03 at k = 1000), not explored.

## Idea

The lexical floor so far is a word kernel used directly: TF-IDF cosine between captions (exp-021 to exp-026). A spectral learner given the same kernel as its similarity graph would not return the kernel. It would return the kernel's top eigenvectors.

- JEPA-type objectives do exactly this at their optimum (Balestriero and LeCun 2610.39302, Prop. 1: Phi(z) = (v_1(z), ..., v_k(z)) / sqrt(p(z)), the top eigenvectors of D^{-1/2} Pi D^{-1/2}).
- The top eigenvectors keep the coarse modes and drop the noisy fine ones.
- Through shared neighbours they also connect captions that share no word.

LMs add most above the floor exactly where such smoothing would act:
- at large k and in CKA (exp-021, exp-022);
- trained layer 0 sits above TF-IDF at k = 1000 on Gemini captions, 0.299 vs 0.230 (exp-026).

**The question:** how much of that gain can word co-occurrence plus one spectral step produce, with no language model at all?

## Kernels (no language model in any of them)

| Code | What | Fitted on | Analogue |
|---|---|---|---|
| F | TF-IDF, sublinear TF-IDF (Pythia tokens), word TF-IDF without stopwords (exp-022 floors) | the evaluation captions' idf | the existing floor |
| T | Laplacian eigenmap of the caption graph over the 10240 evaluation captions | the evaluation captions themselves (transductive) | an upper bound on caption-graph smoothing |
| N | The same eigenmap fitted on disjoint anchor captions, extended to the evaluation captions by the Nystrom formula | anchors only (inductive) | a spectral learner trained on captions of the same kind |
| W | Word vectors from truncated SVD of word-word PPMI (co-occurrence within a caption) on the anchors; caption = TF-IDF-weighted sum | anchors only (inductive) | a learned embedding table (layer 0) |

GloVe was considered and dropped: trained layer 0 already plays the role of a word table learned from general text.

- **Anchors:** the next 20000 rows of `askoepke/wit_1m_recaptioned` after D2's 10240, with non-empty original and Gemini captions. Text only. Disjoint by URL (asserted; 0 shared, 0 duplicate captions). N and W are fitted on the first 10000 and on all 20000 anchors; the 10000 fit is the stability check.
- **Graph:** word TF-IDF (English stopwords removed) cosine, the top 50 neighbours per caption, symmetrised by max.
- **Eigenmaps:** psi = v / sqrt(d), the top eigenvectors of D^{-1/2} A D^{-1/2} with the first dropped. This is Prop. 1's form and `sklearn.manifold.spectral_embedding`'s; a test checks both.
- **Regularisation:** tau / n added to every edge (Amini et al. 2013, tau = mean degree), because the original-caption graph falls apart (191 components, 184 isolated captions).
- **Nystrom extension:** psi_i(x) = sum_j a(x, j) psi_i(j) / (mu_i d(x)), tested to reproduce psi on the anchors.
- **W:** a 20000-word vocabulary (min_df 2), PPMI with context smoothing 0.75, U S^0.5 (Levy, Goldberg and Dagan 2015), `randomized_svd`.
- **Weighting:** spectral kernels unweighted (psi) and eigenvalue-weighted (psi * mu).
- **Dimension:** k_s in {8, ..., 512} in the scored run; capped at 256 in the code afterwards for speed.
- **Normalisation:** rows l2-normalised, no PRH clamp (as the floors).

## Evaluation

- **Data:** D2, n = 10240, original and Gemini captions.
- **Targets:** DINOv2-g and CLIP-H CLS, 7 stored layers (exp-016).
- **Measures:** exp-022's `Rep` / `measure`: mutual kNN at k = 10, 30, 100, 300, 1000 and linear CKA, excess over the exact null.
- **Comparison rows:** the 4 exp-016 LLMs (7 stored layers), and their layer 0.
- **Selection:** on all images and on two fixed halves (seed 0; k halved there so k / n is unchanged). Headline values choose the configuration (floor variant, kernel dimension and weighting, LLM cell) on half A and score it on half B, for kernels and LLMs alike.

## Predictions and reading rules (fixed before running)

| | Prediction | Basis |
|---|---|---|
| S1 | T beats the best F at k = 1000 on Gemini captions | smoothing helps coarse neighbourhoods |
| S2 | At k = 10, no spectral or word-vector kernel beats the best F by more than 0.02 | smoothing discards fine detail |
| S3 | On original captions, T and N help little and W helps CKA most | short captions rarely share words |
| S4 | Eigenvalue-weighted >= unweighted in CKA | CKA rewards matching variance spectra |

**Gap closure:** (best inductive kernel among N and W - best F) / (LLM - best F), half B.
- **>= 50%** at k = 1000 or in CKA: the LMs' coarse gain above words is largely reproducible from co-occurrence statistics.
- **<= 20%:** LMs add coarse structure beyond co-occurrence.

## Run log

**Attempt 1 (2026-11-12, completed; first misread as lost).**
- Ran on exp-027's L4 pod after that job (14:03 UTC), with k_s up to 512.
- Kernels took 1.5 h, almost all of it 513-vector `eigsh` on 6 vCPU.
- Scoring finished at 15:40. The pod terminated itself after the runner's 20-minute wait (16:00 UTC).
- The audit log records that self-termination under the team account user, which I first misread as a manual delete.

**Attempt 2** (CPU pod cpu3g, 16 vCPU, $0.64/h, terminated by hand after copying):
- recomputed the kernels with k_s <= 256 in 2.5 min (kernels now saved to `<out>/kernels/`, resumable);
- skipped scoring, because all 756 rows existed.

## Commands

```bash
uv run pytest -q tests/test_spectral_floor.py
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && STAGES=kernels,score nohup bash infra/runpod/run_exp028.sh > /workspace/results/oddharak/exp-028.log 2>&1 < /dev/null &"
uv run python experiments/exp-028-spectral-lexical-floor/analyze.py --out results/exp-028-spectral-lexical-floor
```
