# Why averaged random word vectors preserve text similarity: random projection theory and predictions for the step-0 embedding layer

Date: 2026-11-05. Target observation: Pythia at step 0, layer 0, mean-pooled, Gemini captions, n = 10240: linear CKA ~0.39-0.40 and mutual 10-NN ~0.11 against DINOv2-g ([exp-018 README](../../experiments/exp-018-alignment-emergence/README.md), [pairs.jsonl](../../results/exp-018-alignment-emergence/pairs.jsonl)).

**How to read the tags.**
- [FT] means I read the theorem, table or relevant section in the text of the paper.
- [AB] means abstract, poster, or search snippet only. Do not quote these beyond what is written here.
- [DERIVED] means my own derivation, not published as such. Where marked, it was checked by simulation.
- [REPO] means read from this repository's code or results.
- [SYNTH] means a toy caption model of mine, useful for shape and not for levels.

Two statements in the assignment need correcting against the code:
- The embedding init of the Pythia series is `small_init`, so sigma = sqrt(2/(5 d)), not a flat 0.02.
- The PRH clamp is one scalar threshold over all stored layers and features, not a per-feature clamp.

Details are in the initialisation section below.

## Objective: is the step-0 result expected, and what can we test?

### Takeaway
Partly. Theory fully predicts that a random projection at d = 768-1024 keeps almost all of what the bag-of-tokens kernel has: 4-6% of CKA is lost, and more of small-k mutual kNN (my rough estimate is 10-25% at k = 10). Theory does not predict that the bag-of-tokens kernel itself aligns with vision at CKA ~0.42. That is an empirical fact about these captions and images, and it is the real finding.

### Cited Findings
- Layer 0 is exactly a bag of tokens in this pipeline. Pythia uses rotary position embeddings, so nothing is added at the embedding layer ([160M config](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/160M/pythia-160m.yml), [410M config](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/410M/pythia-410m.yml)) [FT]. Extraction uses `add_special_tokens=False` and an attention-mask mean ([extract_lm.py](../../experiments/exp-020-lm-data-pilot/extract_lm.py)) [REPO].
- Step-0 values at n = 10240, two seeds per size ([pairs.jsonl](../../results/exp-018-alignment-emergence/pairs.jsonl)) [REPO]; values are means of the two seeds:

| Target | CKA d=768 (160M) | CKA d=1024 (410M) | CKA null mu0, 768 / 1024 | mKNN k=10, 768 / 1024 | mKNN k=1000, 768 / 1024 |
|---|---|---|---|---|---|
| DINOv2-g | 0.395 | 0.403 | 0.00454 / 0.00461 | 0.1064 / 0.1110 | 0.295 / 0.298 |
| CLIP ViT-H | 0.317 | 0.320 | 0.00975 / 0.00987 | 0.0971 / 0.1008 | 0.280 / 0.283 |
| pythia-1.4b | 0.680 | 0.694 | 0.00502 / 0.00509 | 0.3177 / 0.3330 | 0.521 / 0.528 |

- Mutual 10-NN chance is k/(n-1) = 0.00098, and the stored permutation null is 0.00097-0.00099 ([pairs.jsonl](../../results/exp-018-alignment-emergence/pairs.jsonl)) [REPO]. So 0.11 is ~110x chance.

### Inferences
**Core model [DERIVED].** Let p_i = c_i / L_i be the token frequency vector of caption i, and E (V x d) have i.i.d. N(0, sigma^2) entries. Then x_i = E^T p_i. Let B be the n x V matrix whose rows are p_i / ||p_i||, centred, K_B = B B^T, and r_B = (tr K_B)^2 / ||K_B||_F^2 its participation ratio. For any fixed Y independent of E:

- E<K_X, K_Y> = d sigma^2 <K_B, K_Y>, exactly.
- E||K_X||_F^2 = (d sigma^2)^2 ||K_B||_F^2 (1 + (1 + r_B)/d), exactly. This follows from E[S A S] = (1 + 1/d) A + tr(A) I / d for S = E E^T/(d sigma^2) and symmetric A.
- So **CKA(RP, Y) ≈ A(d) · CKA(BoW, Y), with A(d) = 1 / sqrt(1 + (1 + r_B)/d)**.
- Conditions: Gaussian i.i.d. E independent of Y; linear pooling; ratio of expectations (Jensen gap O(1/d)); l2 normalisation handled by the ||p_i|| rows, accurate to O(1/sqrt(d)); clamp treated as negligible.
- Corollaries:
  - CKA(RP, BoW) = A(d).
  - CKA(RP, RP') = A(d)^2 for two independent seeds at the same d.
  - Permutation null of CKA = sqrt(r_X r_Y)/(n-1), exact. With r_X(d) = r_B / (1 + (1 + r_B)/d), the null also scales as A(d).
  - Seed SD of the numerator ≈ CKA · sqrt(2/(d r_M)), with r_M the participation ratio of B^T K_Y B.

**Simulation check [SYNTH], full pipeline with clamp, l2 and centring, n = 6000, r_B = 143:**

| d | CKA(RP, BoW) | A(d) | CKA(RP, RP') | A(d)^2 |
|---|---|---|---|---|
| 64 | 0.524 | 0.555 | 0.265 | 0.308 |
| 256 | 0.793 | 0.800 | 0.621 | 0.640 |
| 768 | 0.915 | 0.918 | 0.839 | 0.842 |
| 1024 | 0.934 | 0.936 | 0.872 | 0.877 |
| 4096 | 0.982 | 0.983 | 0.966 | 0.966 |
| 8192 | 0.991 | 0.991 | 0.982 | 0.983 |

The formula holds to <1% above d = 256 and over-predicts by ~6% at d = 64. CKA(RP, Y) also tracks CKA(BoW, Y) · A(d) at every d in the same run.

**Applied to the real data [DERIVED + REPO].** The ratio mu0(1024)/mu0(768) equals A(1024)/A(768), and mu0 is very stable across seeds. It gives r_B without touching the captions:

| Target | mu0 ratio | r_B | A(768) | A(1024) | Observed CKA ratio 768/1024 | Predicted | BoW-limit CKA from 768 / from 1024 | Implied r_target |
|---|---|---|---|---|---|---|---|---|
| DINOv2-g | 1.0144 | 97 | 0.942 | 0.956 | 0.980 | 0.986 | 0.419 / 0.422 | 25 |
| CLIP ViT-H | 1.0125 | 83 | 0.949 | 0.961 | 0.991 | 0.988 | 0.334 / 0.333 | 133 |
| pythia-1.4b | 1.0140 | 93 | 0.944 | 0.957 | 0.980 | 0.986 | 0.720 / 0.725 | 32 |

Three separate targets give r_B of 83-97, and the two widths then collapse onto one BoW-limit value per target. This is one out-of-sample check, on a small effect: seed noise on each CKA is ~0.005, so the observed ratios are only good to about +/-0.01. It is consistent, not proof. The seed noise itself implies r_M ≈ 17, which is plausible next to r_target ≈ 25 for DINOv2.

**Testable predictions.** All can be run on CPU, none needs new forward passes except P8. Falsifying thresholds are my judgement.

| # | Prediction | Value | Falsified if |
|---|---|---|---|
| P1 | Exact sparse TF BoW (l2-normalised rows) x vision, CKA, n = 10240 | DINOv2 0.42, CLIP 0.33, pythia-1.4b 0.72 | off by more than 0.02 |
| P2 | r_B computed directly from the captions | 83-97 | outside ~70-110 |
| P3 | CKA between layer 0 of two seeds, same d (already cached) | 0.89-0.90 at d=768; 0.91-0.92 at d=1024; ~0.90-0.91 for 768 x 1024 | off by more than 0.02 |
| P4 | CKA x DINOv2 from a fresh d-dimensional E, with r_B = 90 | d=64: 0.27; 128: 0.32; 256: 0.36; 512: 0.39; 1024: 0.40; 4096: 0.415; 16384: 0.42 | wrong shape, or a plateau other than P1 |
| P5 | The factor CKA(d1)/CKA(d2) does not depend on the target model | same ratio for DINOv2, CLIP, LLM | ratio differs across targets beyond noise |
| P6 | Seed SD of CKA x DINOv2 | ~0.005 at d=768, scaling as 1/sqrt(d) | an order of magnitude off |
| P7 | Exact-BoW mutual 10-NN x vision | DINOv2 0.127-0.14; CLIP 0.114-0.125; pythia-1.4b 0.39-0.43; ratio 768/1024 closer to 1 as k grows | DINOv2 at or below 0.111, or above ~0.16 |
| P8 | Gaussian vs Rademacher E at equal d | indistinguishable | CKA differs beyond seed noise |
| P9 | Removing the clamp at layer 0 (l2 only) | CKA changes by less than ~0.005 | larger change |
| P10 | Assigning fresh random vectors per caption (destroying token identity) | CKA and mKNN fall to the null | anything above the null reveals a channel other than token identity |

### Gaps
- r_B and the BoW limit are indirect here, because captions live on the volume and were not available locally. P1 and P2 are the direct check.
- The l2 and clamp treatment in the derivation is approximate. It is validated only on synthetic text.
- I found no published formula for CKA attenuation under Gaussian projection at fixed n. The closest is the random-matrix analysis of [arXiv 2502.19648](https://arxiv.org/html/2502.19648v2) (below). Treat A(d) as my own result until someone checks it.

## JL and random projection: distortion vs dimension, kNN, unbiasedness, stopwords

### Takeaway
The inner product is exactly unbiased for the bag-of-tokens inner product and the cosine is very slightly biased, by O(1/d). Under Gaussian projection the sparsity of the input and the weight of stopwords do not affect distortion. Stopwords matter because they dominate the quantity being preserved, not the error. Worst-case JL bounds (epsilon ≈ 0.3-0.45 here) are far too loose to be useful; typical-pair error is √(2/d) ≈ 4-6%.

### Cited Findings
- Dasgupta-Gupta form of JL: k ≥ 24 / (3 eps^2 - 2 eps^3) · log n ([Stanford lecture notes](https://cs.stanford.edu/people/mmahoney/cs369m/Lectures/lecture1.pdf)) [AB, search snippet]. Achlioptas gives a version with simple discrete entries (s = 1 or s = 3 in Li et al.'s notation), with beta controlling success probability ([Achlioptas](https://infolab.usc.edu/csci599/Fall2002/paper/DR3_jl.pdf)) [prose read; the formula's symbols were lost in text extraction, so the exact (4 + 2 beta) numerator is not verified].
- JL is tight: any embedding with that guarantee needs m = Ω(eps^-2 lg n), for eps down to about lg^0.5001(n)/sqrt(min(n, d)) ([Larsen and Nelson](https://ar5iv.labs.arxiv.org/html/1609.02094)) [AB].
- Exact variances for Gaussian projection to k dimensions, with m1 = ||u1||^2, m2 = ||u2||^2, a = u1·u2 ([Li, Hastie, Church 2006](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf), eqs 3-5) [FT]:
  - E[v1·v2] = a, Var[v1·v2] = (m1 m2 + a^2)/k.
  - Var||v1||^2 = 2 m1^2/k.
  - Var||v1 - v2||^2 = 2 D^2/k, where D = ||u1 - u2||^2.
- For sparse projections with parameter s, Var[v1·v2] = (m1 m2 + a^2 + (s - 3) Σ u1j^2 u2j^2)/k (same paper, eq 19) [FT]. Under regularity conditions on fourth moments, the variances converge to the Gaussian ones for s = o(D), e.g. s = sqrt(D).
- Nearest-neighbour-preserving embeddings need only k = O(log(2/eps)/eps^2 · log(1/delta) · log lambda_X), where lambda_X is the doubling constant; the bound replaces n by intrinsic dimension. The guarantee is a (1 + eps) approximate nearest neighbour for a given query, with probability 1 - delta, not rank preservation ([Indyk and Naor](https://web.math.princeton.edu/~naor/homepage%20files/lowdim-journal.pdf), Theorem 4.1) [FT, constants hidden in O, some symbols lost in extraction].
- Random projection preserved similarity on text comparably to PCA-style reduction in information-retrieval tests ([Bingham and Mannila 2001](https://cs.uwaterloo.ca/~jhoey/teaching/cs793/papers/p245-bingham.pdf)) [AB, numbers not read].

### Inferences
- **All-pairs JL at n = 10240 [DERIVED from the formula above]:** eps = 0.455 (d=512), 0.355 (d=768), 0.300 (d=1024), 0.141 (d=4096). At n = 1024: 0.381, 0.300, 0.255, 0.121. Worst case over all pairs, and it says nothing about ranks.
- **Typical pair [DERIVED from Li et al.]:** relative SD of a squared distance is sqrt(2/d) = 6.2%, 5.1%, 4.4% at d = 512, 768, 1024.
- **Rank noise [DERIVED]:** two nearly equal distances sharing an anchor i, with difference vectors at angle theta, have relative SD of their gap sqrt(4(1 - cos^2 theta)/d). At theta = 60 degrees this is sqrt(3/d) = 6.2% at d=768 and 5.4% at d=1024. A neighbour keeps its rank only if its relative distance gap exceeds this. It does not depend on V, n, or sparsity.
- **Cosine estimator [DERIVED, Monte Carlo checked]:** with rho = cos(p_i, p_j), mean ≈ rho (1 - (1 - rho^2)/(2d)) and SD ≈ (1 - rho^2)/sqrt(d). At d = 768: rho = 0 gives SD 0.036; 0.3 gives 0.033; 0.6 gives 0.023; 0.9 gives 0.0069. Bias is below 0.0003 at every rho. Simulated values matched these to the third decimal.
- **What is being estimated:** the cosine of raw term-frequency vectors. It is not set overlap (Jaccard) and not IDF-weighted. Length cancels, since the mean and sum give the same cosine.
- **Stopwords:**
  - Gaussian E is rotation-invariant, so only norms enter the error. A vector dominated by "the" has no higher distortion.
  - Stopwords do dominate the target: in my toy model mean off-diagonal cos(TF) was 0.81 for L ≈ 340 and 0.34 for L ≈ 40 [SYNTH]. Long Gemini captions (median ~650 tokens, per the [extract_lm.py](../../experiments/exp-020-lm-data-pilot/extract_lm.py) comment) should sit at the high end.
  - CKA centring removes the mean BoW, so what remains is the fluctuation. Under multinomial sampling Cov(p) ≈ diag(pi)/L, so a token's variance share is its unigram probability pi_w: frequent tokens still hold most of the sampling-noise energy.
  - That model gives r ≈ 1/Σ pi_w^2, which is the same order as the r_B ≈ 90 above if Σ pi_w^2 is near 0.01. I did not measure Σ pi_w^2 on these captions. Measure it.
  - Noise energy falls as 1/L. Prediction: short original WIT captions should align less at layer 0 than Gemini captions, independent of d.
- **Sparse or hashed variants differ.** Stopwords enter through (s - 3) Σ u^2 v^2 for sparse projection and ||x||_inf/||x||_2 for hashing (next section). This does not apply to the Gaussian init used here.

### kNN preservation and mutual kNN vs d
There is no closed form for overlap, so this rests on data plus a toy model.

**Observed dimension effect, ratio of step-0 scores d=768 / d=1024, means of two seeds [REPO]:**

| Target | k=10 | k=30 | k=100 | k=300 | k=1000 | CKA |
|---|---|---|---|---|---|---|
| DINOv2-g | 0.959 | 0.967 | 0.972 | 0.977 | 0.988 | 0.980 |
| CLIP ViT-H | 0.963 | 0.967 | 0.971 | 0.976 | 0.991 | 0.991 |
| pythia-1.4b | 0.954 | 0.962 | 0.969 | 0.974 | 0.985 | 0.980 |

Small-k mutual kNN is the most dimension-sensitive, in the same order for all three targets, and k = 1000 behaves like CKA. This fits rank noise: only near-ties flip, and small neighbourhoods are nearly all near-ties.

**Toy model calibrated to r_B ≈ 100 and CKA ≈ 0.42 [SYNTH]** (L ≈ 650, n = 10240, r_B = 106, r_Y = 30, CKA(BoW, Y) = 0.438, mean cos(TF) = 0.85; two seeds per d). Its mKNN(BoW, Y) of 0.039 is below the real level, so read the shape only:

| d | CKA(RP, Y) | Predicted A(d)·CKA | mKNN(RP, BoW) | mKNN(RP, Y) / mKNN(BoW, Y) |
|---|---|---|---|---|
| 128 | 0.321 | 0.323 | 0.351 | 0.79 |
| 256 | 0.361 | 0.368 | 0.498 | 0.86 |
| 512 | 0.405 | 0.398 | 0.624 | 0.92 |
| 768 | 0.416 | 0.410 | 0.685 | 0.96 |
| 1024 | 0.415 | 0.416 | 0.722 | 0.97 |
| 2048 | 0.426 | 0.427 | 0.797 | 0.99 |
| 4096 | 0.431 | 0.432 | 0.853 | 0.99 |

**What this shows:**
- At d ≈ 1000 the projection replaces roughly 30% of a text's top-10 BoW neighbours, yet alignment with Y falls by only a few percent. Replaced neighbours are near-ties in BoW, so they are about as aligned with Y as the ones they replace. High CKA and poor neighbour identity coexist, and JL-style success does not imply kNN identity is preserved.
- 1 - mKNN(RP, BoW) falls roughly as d^-0.4 to d^-0.45 across all three toy runs, close to the 1/sqrt(d) expected from rank noise. Even d = 4096 keeps only 77-85% of neighbours.
- The toy's 768/1024 ratio for Y is ~0.99 against ~0.96 observed, so real captions are more dimension-sensitive than the toy. Real neighbour gaps may be tighter. The toy cannot set the mKNN level.

**Prediction P7 [DERIVED, weak].** Two-point extrapolation with m(d) = m_inf/(1 + b/d) and m(d) = m_inf (1 - c/sqrt(d)) brackets the exact-BoW mutual 10-NN:
- DINOv2 0.127-0.14;
- CLIP 0.114-0.125;
- pythia-1.4b 0.39-0.43.

Ratios carry about +/-0.01 of seed noise, so widen each by ~0.01. Falsified for DINOv2 if the exact-BoW score is at or below the d=1024 value of 0.111, or above ~0.16. Also expected: overlap with the exact BoW neighbours near 0.6-0.8 at d = 768-1024, and the k-ordering of ratios in the table above persisting at other d.

### Gaps
- No primary source gives a closed form for mutual-kNN overlap under random projection as a function of d and the data. Indyk-Naor bounds approximation factor, not overlap. The result depends on the neighbour-gap distribution, which must be measured.
- Constants in the Indyk-Naor bound are not explicit, so it gives no number for d = 768.

## Random indexing, hashing, and random encoders: what performance do they reach?

### Takeaway
All of these show random token-level codes work, but most famous results do not test our condition. BOREP and RandLSTM use **trained** GloVe inputs with a random pooling layer. The relevant result is Appendix E of Wieting and Kiela, where word vectors are also random: performance sits a few points below pretrained, with a much smaller gap on similarity than on classification.

### Cited Findings
**Hashing**
- Signed feature hashing is unbiased, with variance (1/m) Σ_{i≠j} (x_i^2 x'_j^2 + x_i x'_i x_j x'_j), so O(1/m) for unit vectors. The tail bound requires m ≥ 72 log(1/delta)/eps^2 and small ||x||_inf; symbols in the extracted bound were partly lost, roughly ||x||_inf ≤ eps/(18 sqrt(log(1/delta) log(m/delta))) ([Weinberger et al. 2009](https://arxiv.org/pdf/0902.2206), Lemma 2, Theorem 3) [FT, check the constant against the PDF].
- The exact tradeoff: hashing satisfies JL iff ||x||_inf/||x||_2 ≤ nu(m, eps, delta), with nu = 0 for m < D lg(1/delta)/eps^2, nu = 1 for m ≥ 2/(eps^2 delta), and a Theta form between ([Freksen, Kamma, Larsen 2018](https://ar5iv.labs.arxiv.org/html/1805.08539), Theorem 2) [FT].

**Random indexing**
- Kanerva et al. used 1800-dimensional index vectors with eight non-zeros (four +1, four -1) on TASA and TOEFL: 35-44% correct unnormalised, 48-51% normalised, comparable to 36% for 30,000-dimensional normalised vectors before SVD. SVD-based LSA reached 64%, and narrow context windows gave 62-70% ([Kanerva, Kristoferson, Holst 2000](https://escholarship.org/uc/item/5644k0w6)) [FT, short poster]. Chance is 25%.
- Random Indexing is described as an incremental random projection ([Sahlgren 2005 summary](https://www.talks.cam.ac.uk/talk/index/30064/)) [AB].

**Compressed sensing**
- With i.i.d. Rademacher word vectors, the document sum is A·v_BonG, where A preserves inner products of T-sparse vectors up to eps. Linear classification on it is provably as powerful as on bag-of-n-grams, up to small error, with d = Õ(T/eps^2 · log(1/delta)) ([Arora, Khodak, Saunshi, Vodrahalli 2018 poster](https://pages.cs.wisc.edu/~khodak/docs/ICLR2018Poster.pdf)) [AB, poster text; the full paper was blocked]. GloVe and word2vec work better than random as a sensing matrix ([ICLR page](https://iclr.cc/virtual/2018/poster/96)) [AB].

**Wieting and Kiela 2019** ([ICLR paper](https://arxiv.org/pdf/1901.10444)) [FT, including appendices]
- BOREP is a random matrix W from U[-1/sqrt(d), 1/sqrt(d)] applied to **pre-trained** word vectors, then pooled.
- Scores (accuracy, Pearson for SICK-R and STSB):

| Model | Dim | MR | SUBJ | TREC | SICK-R | STSB |
|---|---|---|---|---|---|---|
| BOE, GloVe | 300 | 77.3 | 91.3 | 81.5 | 80.2 | 70.5 |
| BOREP, GloVe | 4096 | 77.4 | 91.9 | 88.8 | 85.5 | 68.5 |
| RandLSTM, GloVe | 4096 | 77.2 | 91.9 | 86.5 | 85.5 | 72.4 |
| ESN, GloVe | 4096 | 78.1 | 92.6 | 87.9 | 86.1 | 74.4 |
| InferSent-3 (trained) | 4096 | 79.7 | 92.6 | 90.8 | 88.5 | 77.3 |
| **BOE, random Normal words** | 300 | 62.4 | 79.7 | 70.6 | 72.5 | ~63 |
| **BOE, random Normal words** | 4096 | 72.1 | 88.2 | 85.0 | 81.0 | 66.4 |

- The largest gap to trained InferSent-3 in the table for the best random encoder is 3.4 points (Table 1, CR). With random words the mean loss vs pretrained is 5.4 points (Table 8), and the authors state it helps classification more than semantic similarity, because similarity tasks can use a word present in both sentences even if unseen in training. They also report pooling random 4096-d embeddings beats BOREP and RandLSTM with random words.

**Conneau et al. 2018** ([ACL paper](https://ar5iv.labs.arxiv.org/html/1805.01070)) [FT, Table 2]
- BoV-fastText (trained word vectors): word content 91.6, sentence length 66.6, tense 89.1, near chance on order tasks (BShift 50.8, SOMO 54.2, CoordInv 54.8).
- Untrained BiLSTM-max: WC 88.8, SentLen 73.3, BShift 70.6. The authors attribute this to architectural priors.

### Inferences
- The compressed-sensing guarantee does not cover us. Gemini captions have T of several hundred, so d = Õ(T/eps^2) exceeds 1024 for any useful eps. It says "enough dimension makes this as good as BoW", not "1024 is enough". A(d) is the sharper statement for kernel alignment.
- SentEval scores in the table come from models fitted on top of features, so they are not a measure of kernel preservation, and a classifier gains parameters with d. Compare the 300-to-4096 change for random BOE (TREC +14.4, MR +9.7, SICK-R +8.5, STSB about +3) with a kernel view: A(d) saturates near d = 1000 for r_B ≈ 90. Expect classifier numbers to keep rising with d after CKA has plateaued.
- Random indexing results are about word-by-context vectors for synonymy, not text-level kernels. They show near-orthogonal random codes match the orthogonal code they replace (48-51% vs 36% across different corpora, so not a clean comparison).
- Hash and sparse codes would behave worse than Gaussian on stopword-heavy tokens. At the "the"-dominated end ||x||_inf/||x||_2 is large, which is exactly the regime where nu matters. Inference, not tested.

### Gaps
- Conneau's "untrained" encoders: I did not confirm whether their word embedding inputs are pretrained, so they should not be read as fully random.
- Wieting and Kiela report SentEval tasks, which have no vision target. I found no paper running bag-of-random-embeddings against vision features. The repo's note on Balestriero and LeCun (arXiv 2610.39302, not read here) says cross-modal mutual 10-NN rises from chance in their own small text encoders ([exp-018 README](../../experiments/exp-018-alignment-emergence/README.md)). That looks in tension with our step-0 floor of ~100x chance, and is worth checking: they may measure a deep output layer rather than a bag of embeddings.

## Bag-of-random-embeddings vs bag-of-words: measurable difference as a function of dimension

### Takeaway
Yes, measurable and predictable. The difference is the attenuation A(d) with r_B ≈ 90: about 5-6% of CKA at d = 768, 4% at d = 1024, and 1% at d = 4096. CKA between the random embedding and BoW is A(d) itself, and random-projection noise is the only difference in the limit of Gaussian E.

### Cited Findings
- Spectral analysis of finite-neuron similarity: with X = X̃R and Gaussian R, CKA is systematically underestimated, mainly through delocalisation of eigenvectors. Under power-law spectra lambda_i ∝ i^-(1+gamma), about i* ≈ (1 + gamma)/sqrt(8) · sqrt(N) eigenvectors stay localised, and N ≳ 8 k^2/(1 + gamma)^2 is needed to stably recover the top k. Conditions: large P and N with P/N = O(1) ([arXiv 2502.19648](https://arxiv.org/html/2502.19648v2)) [FT for the first 100k characters, via a summariser].
- CKA is a sum over kernel eigenpairs weighted by normalised eigenvalues, so it is dominated by leading components (same paper). It is invariant to orthogonal maps and isotropic scaling, not to general linear maps ([Kornblith et al. 2019](https://arxiv.org/abs/1905.00414v2)) [AB].
- Biased CKA can be driven to its maximum by independent data with extreme feature-to-sample ratios ([arXiv 2405.01012](https://arxiv.org/html/2405.01012v1)) [AB].
- Higher dimensions generally improved random encoders on SentEval over {512, ..., 24576}, with overfitting at the top for STSB and SICK-E on BOREP ([Wieting and Kiela 2019](https://arxiv.org/pdf/1901.10444), Figure 1 text) [FT].

### Inferences
- Attenuation table, A(d) by r_B [DERIVED]:

| r_B | 512 | 768 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| 60 | 0.945 | 0.963 | 0.971 | 0.985 | 0.993 | 0.996 |
| 84 | 0.926 | 0.949 | 0.961 | 0.980 | 0.990 | 0.995 |
| 120 | 0.899 | 0.929 | 0.946 | 0.972 | 0.986 | 0.993 |
| 200 | 0.847 | 0.890 | 0.914 | 0.954 | 0.976 | 0.988 |

- Half-point rule [DERIVED]: A(d)^2 = 1/2 at d = 1 + r_B, so d ≈ 90 is where half the kernel energy is projection noise. A(d) > 0.95 needs d > 9.3 (1 + r_B) ≈ 850. d = 768-1024 is right at the elbow, which is why the 160M-410M difference is small but visible.
- With N = 768 and gamma ≈ 0-1, i* ≈ 10-20 eigenvectors of the BoW kernel stay individually stable (using the formula above, P/N ≈ 13 strictly stretches its stated P/N = O(1) condition). Expect CKA to be nearly saturated while high-index structure, relevant to small-k kNN, is not.
- n does not enter A(d) except via r_B, which is itself bounded by min(n, V). At n = 1024 vs 10240 CKA at step 0 is nearly equal (DINOv2 0.39-0.41 at both, [pairs.jsonl](../../results/exp-018-alignment-emergence/pairs.jsonl)) [REPO], consistent with this, though the best ViT cell differs because n = 1024 uses all layers (`d2s`).
- The biased-CKA inflation issue is not a concern at n/d ≈ 10-13: mu0 is ~0.005. Confirm with the debiased estimator if needed.
- A cheap sweep needs no extra forward pass. Layer 0 is linear in E, so draw any E, compute P·E for any d, and keep P (token counts) from one tokenisation. Concatenating m seeds equals d → m·d exactly; truncation to fewer columns equals a smaller d exactly. So `rand_coord` truncation of step-0 layer 0, already implemented for exp-019 ([truncation.py](../../src/geoprh/truncation.py)) [REPO], is a dimension sweep at no extra cost (it centres before truncating, so compare against its own reference score). Prediction [DERIVED]: at step 0, layer 0, `rand_coord` and `rand_proj` agree within seed noise, because the Gaussian basis is not privileged. Any gap means the pipeline departs from this model.

### Gaps
- No prior paper states A(d), as far as my search went (one targeted search). The result is simple enough that it may exist under other names.
- The formula is for Gaussian E. It should also hold for Rademacher; for sparse codes add the (s - 3) correction, which I did not derive for the kernel norm.

## Tokenisation and frequency weighting

### Takeaway
Random embeddings see only exact token-string match, so tokenisation defines what counts as "overlap". Weighting is just reweighting the BoW before projection. Theory gives its effect on r_B and A(d), but not on alignment with vision; that part is purely empirical.

### Cited Findings
- GPT-NeoX tokenisation depends on the preceding space: a word encodes differently at sentence start, and `add_prefix_space` defaults to False ([HF GPT-NeoX docs](https://huggingface.co/docs/transformers/model_doc/gpt_neox)) [FT]. The 20B tokenizer is used for Pythia 160M and 410M ([410M config](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/410M/pythia-410m.yml)) [FT]. It also allocates extra tokens to whitespace (same docs) [FT].
- SIF weights a/(a + p(w)), then removes the first common component; reported to beat plain averaging on similarity tasks ([Arora, Liang, Ma 2017 abstract](https://mlanthology.org/iclr/2017/arora2017iclr-simple)) [AB; the PDF was blocked].
- uSIF improves on this with a data-derived a (≈1.2e-3 for mean sentence length ≈ 11) and piecewise removal of the first m components. Up to 15.5% on STS'12 yearly average, e.g. GloVe+WR 56.2 vs GloVe+UP 64.9 ([Ethayarajh 2018](https://aclanthology.org/W18-3012.pdf)) [FT, tables hard to read in extraction, so only this checked pair is reported]. Both use trained word vectors.

### Inferences
- [DERIVED] x_i = Σ w(t) e_t / L is a random projection of the weighted BoW. Everything above applies with B rebuilt from weighted counts. Flatter weighting usually raises r_B, which lowers A(d) at fixed d. So a higher-CKA BoW limit can coexist with more attenuation.
- Token orthogonality has sharp edges: `" dog"`, `"dog"`, `" Dog"` and `"Dog"` get independent vectors in a random embedding, with expected similarity 0. Sentence-initial words and capitalisation lose all overlap with their in-sentence forms. Inflections such as `" dog"` and `" dogs"` are also unrelated unless BPE splits them into a shared fragment. Long captions dilute the sentence-start effect, since only one token per sentence is affected.
- BPE splitting of rare words creates shared fragments, which could give both useful morphological overlap and spurious overlap. Theory gives no sign.
- Layer-0 alignment is invariant to any word-order shuffling within a caption, by construction. That is a useful unit test of the pipeline.
- Variants to run, all on CPU from token counts:
  - token vs lowercase/whitespace-normalised word BoW;
  - stopword-only vs content-only partition;
  - TF vs IDF vs SIF a=1e-3.
  Report CKA and mKNN against the same ViTs, plus r_B.

**Weighting in the toy model [SYNTH]** (L ≈ 340, n = 6000; stopwords are pure noise there by construction, so the alignment column shows a possibility, not a forecast):

| Weight | r_B | A(768) | CKA(exact weighted BoW, Y) | Predicted CKA at d = 768 | mKNN k=10, exact BoW |
|---|---|---|---|---|---|
| TF | 143 | 0.918 | 0.053 | 0.049 | 0.005 |
| IDF | 3308 | 0.434 | 0.321 | 0.139 | 0.177 |
| SIF a=1e-3 | 2413 | 0.491 | 0.301 | 0.148 | 0.153 |
| Drop top 150 tokens | 1939 | 0.533 | 0.291 | 0.155 | 0.160 |

The r_B and A(d) columns carry the useful lesson. Down-weighting frequent tokens can raise r_B by an order of magnitude, so d = 768 stops being "effectively BoW" and half the exact-BoW CKA is lost. Prediction [DERIVED]: for IDF or stopword-removed random embeddings, the gap between exact BoW and d = 1024 is much larger than the 4% for TF, and A(d) from the measured r_B accounts for it. Run the exact BoW and the random version for every weighting.

### Gaps
- I found no paper measuring random-embedding alignment under BPE vs whitespace tokenisation. Not searched exhaustively.
- No theory predicts whether stopword fluctuations in captions correlate with images (e.g. plural vs singular "a"/"the" patterns, caption style). The partition test above decides this.
- Sennrich et al. on BPE was not read.

## Initialisation and PRH preprocessing

### Takeaway
Scale does not matter for cosine or CKA, except possibly through the clamp threshold, which is shared across layers. The clamp is near-linear at q = 0.95: 98.9% of the post-clamp kernel is the linear term under Gaussian coordinates. At layer 0 at step 0 it may do nothing at all, which is checkable.

### Cited Findings
- Pythia configs set `init_method: small_init`, no weight tying, rotary position embeddings, hidden size 768 (160M) and 1024 (410M) ([160M](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/160M/pythia-160m.yml), [410M](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/410M/pythia-410m.yml)) [FT].
- `small_init` is sigma = sqrt(2/(5 dim)) with dim = hidden size ([gpt-neox init_functions.py](https://raw.githubusercontent.com/EleutherAI/gpt-neox/main/megatron/model/init_functions.py)) [FT]. The method is passed to `VocabParallelEmbedding` for word embeddings, and the file does not show the call itself ([word_embeddings.py](https://raw.githubusercontent.com/EleutherAI/gpt-neox/main/megatron/model/word_embeddings.py)) [FT].
- The HF `GPTNeoXConfig` default is `initializer_range = 0.02`, and the docs call it truncated normal ([HF docs](https://huggingface.co/docs/transformers/model_doc/gpt_neox)) [FT]. The code draws from a normal with that std ([HF modeling file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py)) [AB, snippet]. This only applies when HF creates fresh weights, not when loading the step-0 checkpoint.
- Clamp code: `q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()`, then `feats.clamp(-q_val, q_val)` ([platonic-rep/metrics.py](../../platonic-rep/metrics.py)) [REPO]. Exp-018 calls this on the [n, L, d] tensor via `prepare_features(x, q=0.95, exact=False)`, then l2-normalises each layer ([measure.py](../../experiments/exp-016-positive-controls/measure.py), [preprocess.py](../../Aristotelian/aristotelian/prh/preprocess.py)) [REPO].

### Inferences
- sigma = 0.0228 at d = 768 and 0.0198 at d = 1024, close to 0.02 but scaling with d. Assuming the method is applied, check it directly: expect std(wte) at step 0 ≈ 0.0228 and 0.0198.
- **Scale [DERIVED]:** cosine, CKA, and kNN are invariant to sigma. The clamp threshold scales with the features, so for a single layer it is also invariant.
- **Where sigma can matter:** the threshold is one scalar computed across all stored layers. Per-caption coordinates at layer 0 have SD sigma ||p_i||, likely small (||p|| ~ 0.1 if Σ pi^2 ~ 0.01). If deeper random layers have larger activations, the threshold sits far above layer 0 and the clamp is a no-op there. Measure the clipped fraction per layer. Both sizes store 7 layers in `d2l`, so the threshold is set by different depths in 160M and 410M.
- **Clamp as Hermite expansion [DERIVED]:** for a Gaussian coordinate clamped at c sigma, post-clamp correlation is Σ a_k^2 rho^k / E[f^2], odd k only, with a_1 = P(|g| < c). Linear share a_1^2 / E[f^2] is:
  - q = 0.90: 0.974;
  - q = 0.95: 0.9885;
  - q = 0.99: 0.998.
  So even if active the clamp barely changes the kernel. Coordinates are conditionally Gaussian given p_i, exactly over E.
- Short captions have larger ||p_i||, so one shared threshold clamps them harder. This is a second-order effect, smaller for long Gemini captions.
- **l2 normalisation [DERIVED]:** ||x_i||^2 = d sigma^2 ||p_i||^2 (1 + O(1/sqrt(d))), so normalising replaces p_i with p_i/||p_i|| plus O(1/sqrt(d)) noise. It removes the length-driven scale of p, but not length's effect on direction: short captions sit farther from the mean, so they can lead the centred kernel's top PCs. Gemini caption length may correlate with image complexity. Test with CKA(K_B, kernel of L).
- **Centring:** in CKA, centring x equals projecting E^T (p_i - p̄), i.e. it drops the mean BoW. Mutual kNN is not centred, but Euclidean rank on the sphere is unaffected by a common direction except through normalisation.
- Rotary embeddings mean no position term here. Models with learned position embeddings would add a random projection of the position histogram, which depends mainly on L, adding a length channel. Not applicable to Pythia.

### Gaps
- I did not inspect `VocabParallelEmbedding` or step-0 weights, so application of `small_init` to the token embedding is unconfirmed (test above).
- The clipped fraction at layer 0 is unknown.
- PyTorch's default embedding init was not checked in this pass; it does not bear on these checkpoints.

## How much of semantic similarity is simple lexical overlap?

### Takeaway
A lot on natural data, little on adversarial data. A binary bag-of-words cosine reached r = 0.73 on STS 2017 English, against 0.85 for the best system. On natural Quora pairs BoW reached 83% accuracy vs BERT 90.5%, but it falls to ~29% on PAWS, which was built to break it. Natural captions are the easy kind, so strong lexical alignment is expected, though I found no direct measurement of BoW against vision features at this scale.

### Cited Findings
- **STS 2017** ([Cer et al.](https://ar5iv.labs.arxiv.org/html/1708.00055)) [FT]:
  - Baseline is cosine of binary word vectors, r × 100: EN-EN 72.78, Primary 53.70, rank 23 of 44 complete submissions.
  - Best EN-EN is 85.47.
  - On the STS Benchmark, test Pearson × 100: GloVe avg 40.6, word2vec avg 56.5, SIF 72.0, Sent2Vec 75.5, InferSent 75.8, ECNU 81.0.
- **PAWS** ([Zhang et al. 2019](https://ar5iv.labs.arxiv.org/html/1904.01130)) [FT], accuracy / precision-recall AUC:

| Model | QQP→QQP | QQP→PAWS-QQP | PAWS-Wiki supervised |
|---|---|---|---|
| BOW | 83.2 / 89.5 | 29.0 / 27.1 | 55.8 / 41.1 |
| BERT | 90.5 / 96.3 | 33.5 / 35.1 | 90.4 / 93.7 |

  Authors state existing paraphrase sets lack pairs with high lexical overlap that are not paraphrases. In QQP, about 80% of high-overlap pairs are paraphrases.
- Word averaging beats LSTMs out of domain on similarity ([Wieting et al. 2016](https://mlanthology.org/iclr/2016/wieting2016iclr-universal)) [AB].
- BoV stores word content very well (91.6) and order not at all, and word content correlated with all downstream tasks ([Conneau et al. 2018](https://ar5iv.labs.arxiv.org/html/1805.01070)) [FT].
- Image-caption ranking work (Hodosh and Hockenmaier) reports more advanced text representations beating simple bag-of-words with KCCA ([IJCAI 2015](https://www.ijcai.org/Abstract/15/593)) [AB]. TF-IDF inputs are still used as a text representation in CCA retrieval ([arXiv 1705.06979](https://arxiv.org/pdf/1705.06979)) [AB]. Weighted bag-of-words is competitive for text-to-image search in VisualSparta ([arXiv 2101.00265](https://arxiv.org/pdf/2101.00265)) [AB, title level only].

### Inferences
- STS baseline EN-EN r = 0.73 and best 0.85 give r^2 of 0.53 vs 0.73 [DERIVED]. So overlap accounts for roughly 70% of the explained variance of the best system, on that one test set. This is a loose analogy, and it is not comparable with STS-B numbers, which use a different split.
- Our analogue: layer-0 mKNN at k=10 vs final model is 0.106-0.111 vs 0.19-0.22, and CKA is 0.39-0.40 vs 0.53-0.70 (README). Random BoW holds 50-60% of final mKNN at the same k. Using the BoW limit (P1) shifts this upward. Training's real gain over lexical overlap is what should be reported, not its gain over the random floor.
- CKA vs pythia-1.4b of 0.68-0.69 at step 0, 0.72 in the BoW limit, says the trained LM's mean-pooled kernel is heavily lexical at its best-matching layer. The best cell there is [0, 0], i.e. the 1.4b embedding layer ([pairs.jsonl](../../results/exp-018-alignment-emergence/pairs.jsonl)) [REPO]: trained layer-0 kernel is itself a bag of (now trained) token vectors. So that 0.68 is largely BoW vs near-BoW and should not be read as LM semantics.

### Gaps
- No primary source found measures CKA or kNN overlap of BoW vs vision features. One search only. Numbers in the retrieval literature I saw were snippets without BoW rows.
- SimLex-999 and Ethayarajh's "How contextual are contextualized word representations" were not read. SimLex is word-level and is unlikely to change the conclusions.
- STS 2012 baseline numbers were not checked.

## What theory does NOT predict, and reading depth

### Takeaway
Theory explains the projection step only. It says nothing about the level of BoW x vision alignment, the sign of weighting effects, or the absolute level of mutual kNN. Read the [AB] items below as leads, not evidence.

### Cited Findings
- JL-type results concern only preservation of input geometry ([Larsen and Nelson](https://ar5iv.labs.arxiv.org/html/1609.02094)) [AB], and Indyk-Naor's concern only approximate neighbours ([paper](https://web.math.princeton.edu/~naor/homepage%20files/lowdim-journal.pdf)) [FT]. Neither includes any target other than the input.
- With random words, an unseen token carries no information, as Wieting and Kiela note for classification ([paper](https://arxiv.org/pdf/1901.10444), Appendix E) [FT].

### Inferences
**Not predicted:**
1. Why CKA(BoW, vision) ≈ 0.42. This is a property of the captions, the images, and how much of caption word statistics track visual content. The theory only passes it through.
2. The absolute mutual kNN level. It needs the neighbour-gap distribution of both BoW and vision.
3. Whether IDF, SIF or stopword removal raises or lowers vision alignment. Only r_B and A(d) are predictable.
4. Which ViT aligns best, or why DINOv2 exceeds CLIP on CKA. Why the best ViT cell is an intermediate layer for DINOv2 and the last layer for CLIP.
5. Whether the layer-0 ranking vs deeper random layers is general. Random non-linear layers fall outside this model. At n = 1024 the best cell is sometimes LM index 1.
6. Synonym or paraphrase effects. Random vectors give expected similarity 0 to synonyms, so anything needing them requires training, including in the first 1000 steps.
7. Anything about the training trajectory, including the sharp transition at steps 512-4000.
8. Exact finite-d effect of clamp and l2 (approximations above), and corrections at d near or below r_B (over-prediction of ~6% at d = 64 in simulation).

**Reading depth:**
- Full text or relevant section read [FT]:
  - Wieting and Kiela 2019 (including Appendices D-E);
  - Weinberger et al. 2009 (Lemma 2, Theorem 3, partly garbled symbols);
  - Freksen et al. 2018 (Theorem 2);
  - Li, Hastie, Church 2006 (eqs 3-5, 12-20, 30);
  - Indyk and Naor (Theorem 4.1, definitions);
  - Conneau et al. 2018 (Table 2);
  - Cer et al. 2017 (Tables 10, 14);
  - Zhang et al. 2019 (PAWS results);
  - Ethayarajh 2018 (partial);
  - Kanerva et al. 2000;
  - Pythia configs, gpt-neox init code, HF GPT-NeoX docs;
  - arXiv 2502.19648 (first ~75%, via summariser).
- Abstract, poster, or snippet only [AB]:
  - Arora et al. 2017 (SIF, PDF blocked by a browser check);
  - Arora et al. 2018 (poster; paper blocked);
  - Larsen and Nelson;
  - Bingham and Mannila;
  - Sahlgren 2005;
  - Kornblith et al. 2019;
  - arXiv 2405.01012;
  - Wieting et al. 2016;
  - Hodosh and Hockenmaier;
  - arXiv 1705.06979, 2101.00265;
  - Dasgupta-Gupta form (via lecture notes);
  - Achlioptas (prose read, formula not legible).
- My own, not from literature:
  - A(d) and its corollaries;
  - the r_B estimate from mu0;
  - cosine moments (standard form, Monte Carlo checked here);
  - rank noise sqrt(3/d);
  - the clamp Hermite calculation;
  - all [SYNTH] numbers.

### Gaps
- This pass used more than the suggested number of tool calls, because many PDFs came back unreadable and had to be extracted locally. Some bounds with lost symbols are flagged above; check those against the PDFs before quoting.
- No test on real caption text was possible here. All of P1-P10 remain open until run on the volume's captions.
