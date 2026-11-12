# exp-029: do word vectors from general text close the LLMs' coarse gap too? (out-of-domain check for exp-028)

**Status:** done, 2026-11-13 · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-029-ood-word-vectors/` (copy in local `results/exp-029-ood-word-vectors/`) · **Code:** `words.py`, `run.py`, `analyze.py`, `tests/test_ood_words.py`, `infra/runpod/run_exp029.sh`

## Results (2026-11-13)

**Summary:** the answer depends on the caption type.

- **Human (original WIT) captions:** word vectors from general text close **69-79%** of the LLMs' coarse gap above the word floor. That is more than captions of the same kind manage (45-61%), so the general-text rule (>= 50%) passes. On these captions the LLMs' coarse advantage over word counts is mostly ordinary distributional semantics.
- **Gemini captions:** general text closes **almost none of it**: −59% to −37% at k = 1000 (below the floor) and 8-16% in CKA. Only Gemini captions themselves reproduce the LLMs (91-92% for DINOv2). On these captions, exp-028's closure is caption-specific co-occurrence, not general word knowledge, so the rule (< 20%) narrows exp-028's claim.

**Checks:**
- The word floor reproduces exp-022's stored row (CKA within 3e-6, 10-NN exact).
- The exp-028 W20000 refit reproduces exp-028's rows within 0.0002.

**Corpus sizes** (content tokens after stopword removal):
- Gemini anchors: 5.46M. C4 and Wikipedia were streamed to the same count (1x) and to 54.6M (10x).
- Original-caption anchors: only 145k tokens, a 4,922-word vocabulary.

Tables: `summary.md`; figure: `fig_closure.png`.

### Gap closure, half B (chosen on half A)

Closure = (kernel − best word floor) / (LLMs − best word floor), LLMs = mean of 4. Shown as k = 1000 / CKA:

| Fitted on | Original, DINOv2 | Original, CLIP | Gemini, DINOv2 | Gemini, CLIP |
|---|---:|---:|---:|---:|
| Same-type captions, whole caption | 56% / 55% | 45% / 50% | 91% / 92% | 50% / 82% |
| Same-type captions, ±5 window | 61% / 59% | 50% / 53% | 109% / 91% | 61% / 77% |
| Other-type captions, ±5 window | 61% / 63% | 50% / 56% | −128% / 9% | −116% / 31% |
| C4 web, 1x | 51% / 51% | 44% / 53% | −85% / 10% | −62% / −12% |
| C4 web, 10x | 64% / 64% | 58% / 68% | −59% / 16% | −37% / −6% |
| Wikipedia, 1x | 65% / 67% | 58% / 70% | −93% / −7% | −78% / 8% |
| Wikipedia, 10x | **75% / 75%** | **69% / 79%** | −80% / 5% | −70% / 7% |
| GloVe 6B | 60% / 51% | 59% / 57% | −62% / −9% | −46% / −49% |
| Control: words shuffled | −6% / −9% | −6% / −5% | −441% / −125% | −451% / −284% |

Negative closure means the kernel falls below the plain word floor.

At k = 10 on original captions, Wikipedia 10x also closes 31-34% of the fine-scale gap, which in-domain captions never did (−8% to −12%).

### Vocabulary coverage of the evaluation captions

| Fit | Original | Gemini |
|---|---:|---:|
| Same-type captions | 0.64 | 0.99 |
| Other-type captions | 0.72 | 0.64 |
| C4 web, 1x / 10x | 0.81 / 0.83 | 0.97 / 0.98 |
| Wikipedia, 1x / 10x | 0.84 / 0.85 | 0.96 / 0.96 |
| GloVe | 0.94 | 1.00 |

### Predictions

| | Prediction | Verdict |
|---|---|---|
| O1 | Gemini: web 1x closes less than same-type captions | holds (by a wide margin) |
| O2 | Original: Wikipedia >= same-type captions | holds |
| O3 | Gemini: 10x > 1x, and still below same-type | **fails**. 10x is not reliably above 1x (Wikipedia CKA: 5% vs −7% on DINOv2, 7% vs 8% on CLIP), and both stay far below same-type |
| O4 | Shuffled control at or below the floor | holds |

### Reading

1. **On human captions, the LLMs' coarse advantage over word counts is mostly general word similarity.**
   - Wikipedia text (55M tokens) reaches 69-79% of it. It even reaches a third of the fine-scale gap.
   - More text helps (1x → 10x: +10 points), and Wikipedia beats web text. WIT's captions come from Wikipedia, so this is a domain match, not caption leakage: no caption-image pairing is used.
   - Part of general text's edge over same-type captions is coverage. 145k tokens of original captions cover only 64% of the evaluation words.
2. **On Gemini captions, general word similarity adds nothing above the floor. Only the generator's own captions do.**
   - Word co-occurrence in Gemini captions encodes what the captioner saw and how it phrases it: which visual attributes, objects and settings it mentions together.
   - In-domain W learns those regularities from 20k other images' captions, and that substitutes for whatever the LLMs do.
   - The LLMs, trained on general text, must get their Gemini advantage another way, most plausibly from context and composition. A bag of general word vectors cannot replace that.
   - exp-028's 84-92% closure on Gemini is therefore not evidence that the LLMs' gain is lexical. It shows that a generator's caption statistics can stand in for it.
3. **Why do the caption sets differ?** Gemini captions are long and descriptive, and their word floor is already high (0.40 at k = 1000 vs 0.04 on original captions). What is left above it is not word similarity. Short human captions rarely share words, so similarity between different words (synonyms, related objects) is exactly what they lack, and general text supplies it.
4. **The ±5 window and the whole-caption window behave alike** on captions (within a few points), so the window does not confound the corpus contrast.

### What this changes

- exp-028's headline is narrowed:
  - on human captions, "the LLMs' coarse gain above words is largely general distributional semantics" stands, now without the in-domain caveat;
  - on Gemini captions, it is replaced by "a generator's caption statistics reproduce it; general word similarity does not".
- **The floor ladder for human captions:** word floor → general-text word vectors (about 70-80% of the coarse gap, a third of the fine gap) → LLMs.
- **On generated captions, the ladder breaks:** the second step is caption-specific, so floors for generated captions should be fitted on text that does not come from that generator, or not interpreted as general.

### Limits

- 4 LLMs and 2 ViTs, one image set (D2).
- The general corpora are first-come samples of shuffled streams (seed 0), 55M tokens at most; GloVe is the only large general table. More data might move Gemini somewhat, but 1x → 10x barely did.
- The best dimension is often 1024 for the general-text fits, the top of the range.
- The windowed caption fits and the whole-caption fits use different vocabularies from exp-028 (top 50k, count >= 5), and idf comes from the evaluation captions. The `repro` row isolates exp-028's exact setting.

### Run

- CPU pod (cpu3g, 16 vCPU, $0.64/h), 19:27-20:05 UTC, then terminated by hand after copying, about $0.5.
- Corpora took 3 min, vectors 20 min, kernels 3.5 min, scoring 11 min.

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
infra/runpod/push_code.sh <ip> <port>
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp029.sh > /workspace/results/oddharak/exp-029.log 2>&1 < /dev/null &"
uv run python experiments/exp-029-ood-word-vectors/analyze.py --out results/exp-029-ood-word-vectors
```

## Question

exp-028 found that PPMI word vectors fitted on 20k captions of the same kind (W) close 46-92% of the gap between the word floor and LLMs at k = 1000 and in CKA. Those captions came from the same source as the evaluation captions, and for Gemini from the same generator.

**The question:** is the closure general distributional semantics, which an LLM also learns from web text? Or is it caption-specific co-occurrence, i.e. statistics of what appears together in described scenes, plus one generator's vocabulary habits?

Both answers are informative:
- **General text closes the gap:** the LLMs' coarse advantage over words is ordinary word similarity, learnable from any text. The "second floor" is a property of language, not of the captions.
- **Only caption text closes it:** the shared coarse structure is visual co-occurrence written down in image descriptions. That is a more Platonic reading (the world's co-occurrence statistics) and it qualifies exp-028.

## Design

Same evaluation as exp-028 (D2, n = 10240, original and Gemini captions, DINOv2-g and CLIP-H CLS, exp-022 measures, split-half selection, gap closure). Only the corpus that W is fitted on changes.

| Code | Fitted on | Size | What it tests |
|---|---|---|---|
| W-same | 20k captions of the same type (exp-028's W20000) | ~12M tokens (Gemini), ~0.3M (original) | reference |
| W-cross | 20k captions of the other type: Gemini anchors for original evaluation captions, and the reverse | as above | same images, different writer |
| W-web | C4 English (`allenai/c4`, streamed), first documents in order | token-matched to W-same Gemini (~12M), and 10x (~120M) | general web text |
| W-wiki | English Wikipedia (`wikimedia/wikipedia`) | ~12M and ~120M tokens | general text from the source of WIT's original captions (WIT captions come from Wikipedia pages) |
| G | GloVe 6B 300d (Wikipedia 2014 + Gigaword 5), loaded from `sentence-transformers/average_word_embeddings_glove.6B.300d` (the original vectors as safetensors, no extra dependency) | 6B tokens, pretrained | a standard large general-text table |

- **Co-occurrence window.**
  - Captions: the whole caption, as in exp-028.
  - Web and Wikipedia text: a sliding window of ±5 words, because documents are long and whole-document co-occurrence is mostly topic.
  - Captions are also refitted with ±5 windows, so the window is not confounded with the corpus. This doubles the caption rows; one window per corpus type would be cheaper but leaves the confound.
- **Vocabulary.** Top 50k words of each corpus (English stopwords removed, min count 5). Eval-caption coverage is reported per corpus: the share of caption word tokens that are in the vocabulary. A corpus that misses caption words loses for that reason alone, and the coverage column shows it.
- **PPMI and SVD.**
  - At 50k words the dense PPMI matrix is 10 GB, so it is kept sparse; `randomized_svd` accepts sparse input.
  - The same smoothing (0.75) and U S^0.5 as exp-028.
  - A test checks that the sparse path reproduces the dense one on exp-028's toy matrix.
- **Dimension.** k_s in {128, 256, 512, 1024}. 1024 is added because exp-028's W usually chose 512, the top of its range.
- **Caption vector:** TF-IDF-weighted sum of word vectors, over the fit's vocabulary, with idf from the evaluation captions for every fit (exp-028 took idf from its anchors; using the evaluation captions keeps the weighting the same across corpora).
- **Pipeline check (`repro`):** exp-028's W20000 refitted exactly (its vocabulary, anchor idf and dense PPMI). Its rows must match exp-028's stored rows within 0.003, or the job fails. The same fit at 1024 dimensions is the range-edge control.
- **Controls:**
  - shuffled word-to-vector assignment (same vectors, permuted words), which should fall to about the floor or below;
  - W-same rerun with k_s = 1024, to see whether exp-028's edge choice mattered.

## Predictions and reading rules (fixed before running)

| | Prediction |
|---|---|
| O1 | Gemini evaluation: W-web (12M) closes less than W-same at k = 1000 and in CKA, because generator-specific phrasing is missing |
| O2 | Original evaluation: W-wiki is at least as good as W-same, because the original captions come from Wikipedia |
| O3 | 10x more general text closes more than the token-matched version, but stays below W-same on Gemini |
| O4 | The shuffled control is at or below the floor |

**Main readout:** gap closure of the best general-text kernel (W-web or W-wiki, chosen on half A) at k = 1000 and in CKA, half B, per caption set and target.

- **≥ 50%:** "LLMs' coarse gain above words is general distributional semantics." exp-028's claim stands without the in-domain caveat.
- **20-50%:** partly general, partly caption-specific. Report both and use W-same as the upper reference.
- **< 20%:** the closure is caption-specific co-occurrence. exp-028's wording is narrowed to "in-domain caption statistics".

## Cost

CPU only (cpu3g 16 vCPU at $0.64/h, priced live first).
- **Streaming:** about 130M tokens each of C4 and Wikipedia text, a few minutes each.
- **Fitting:** 50k-word sparse PPMI and SVDs, minutes each.
- **Scoring:** the exp-028 path, about 6 corpora x 4 dimensions x 2 caption sets x 2 targets x 3 splits.
- **Total:** under 1 h, about $0.6.

Outputs go straight to the volume, with resumable stages.

## Implementation

- `experiments/exp-029-ood-word-vectors/run.py` reuses exp-028's `run.py` scoring and `spectral.py` PPMI, adding a sparse PPMI path and the corpus streams.
- No spectral (graph eigenmap) kernels: that line was dropped on 2026-11-12.
- Floors and LLM rows are recomputed in the same run, so every comparison is on identical splits.

## Decisions (2026-11-13)

- GloVe included.
- Token budgets: 1x and 10x the Gemini anchors' content tokens (measured at run time, not the ~12M guessed above).
- Word vectors are stored as float16 on the volume; caption vectors are computed in float32.
