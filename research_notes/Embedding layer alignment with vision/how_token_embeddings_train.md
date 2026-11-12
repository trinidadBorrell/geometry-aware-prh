# How token embeddings are trained in language models, and what they end up encoding

Reading-depth tags on every citation:
- **[FT]** I read the relevant passages in the full text (extracted PDF text checked directly).
- **[FT-s]** Full text was read through a summarising fetch tool. Numbers come from that rendering, not from my own read of the page, so re-check any number you intend to quote.
- **[ABS]** Abstract or landing page only.
- **[CODE]** Config file or source code.
- **[BLOG]** Blog or forum post, not peer reviewed.
- **[SNIP]** Search-result snippet only.

Everything under "Inferences" is mine and has no direct source.

## Q1. How is the input embedding matrix updated in GPT-style models?

### Takeaway
The embedding matrix is an ordinary parameter table. It starts from random values and is trained jointly with the rest of the network by backpropagating next-token cross-entropy, with no word2vec-style pretraining in any open model I checked. Tying varies by family and size: GPT-2 and small Llama/Qwen models are tied, while Pythia, OLMo-7B, Llama-3-8B and Qwen2-7B are untied. This matters because it decides whether a row receives gradient only when its token appears in the input, or at every step.

### Cited Findings

**Tying: origins**
- Press & Wolf show the output matrix is itself a valid word embedding, recommend tying it to the input, and report that the tied matrix's updates resemble those of an untied *output* matrix more than an untied input matrix — [Press & Wolf 2017](https://arxiv.org/abs/1608.05859) [ABS]
- Inan et al. derive tying from a loss framework linking input embeddings and the output projection, and report state-of-the-art Penn Treebank results with fewer parameters — [Inan et al. 2017](https://arxiv.org/abs/1611.01462) [ABS]
- The original Transformer shares one matrix across both embedding layers and the pre-softmax projection, and multiplies embeddings by sqrt(d_model) — [Vaswani et al. 2017](https://ar5iv.labs.arxiv.org/html/1706.03762) [FT-s]

**Tying and initialisation by model**

| Model | Tied? | d | Vocab | Embedding init | Source |
|---|---|---|---|---|---|
| GPT-2 | Yes: `logits = matmul(h, wte, transpose_b=True)` | 768 (small) | 50,257 | N(0, 0.02); position embedding N(0, 0.01) | [OpenAI model.py](https://raw.githubusercontent.com/openai/gpt-2/master/src/model.py), [config](https://huggingface.co/openai-community/gpt2/raw/main/config.json) [CODE] |
| Pythia (all sizes) | No: `no-weight-tying True`; authors say untied eases interpretability research | 512 (70M) to 5120 (12B) | ~50k | Stated `small-init`; see the errata below | [Paper Sec 2.3, Table 6](https://arxiv.org/abs/2304.01373) [FT], [70M](https://huggingface.co/EleutherAI/pythia-70m/raw/main/config.json), [6.9B](https://huggingface.co/EleutherAI/pythia-6.9b/raw/main/config.json), [12B](https://huggingface.co/EleutherAI/pythia-12b/raw/main/config.json) [CODE] |
| GPT-NeoX library default | Tied by default in the library (`no_weight_tying: False`), though the HF config class defaults to untied | n/a | n/a | n/a | [neox_arguments.md](https://raw.githubusercontent.com/EleutherAI/gpt-neox/main/configs/neox_arguments.md), [HF config](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_neox/configuration_gpt_neox.py) [CODE] |
| OLMo-7B | No: `weight_tying: false` | 4096 | 50,280 | `mitchell`: std = 1/sqrt(d) = 0.0156, truncated at ±3 std | [config](https://huggingface.co/allenai/OLMo-7B/raw/main/config.json), [model.py](https://raw.githubusercontent.com/allenai/OLMo/main/olmo/model.py) [CODE] |
| OLMo-1B | Tied in the Feb 2024 release, untied in the July 2024 release | n/a | n/a | n/a | [Lopardo et al. 2026](https://arxiv.org/html/2603.26663) [FT-s] |
| OLMo-2 7B | No | 4096 | 100,352 | Training init not checked | [config](https://huggingface.co/allenai/OLMo-2-1124-7B/raw/main/config.json) [CODE] |
| Llama 2 7B | No (Magikarp table) | n/a | 32,000 | Not public | [Land & Bartolo 2024](https://arxiv.org/abs/2405.05417) [FT]; Meta's HF config returned HTTP 401, so not read directly |
| Llama 3 8B | No | 4096 | 128,256 | Not public | [mirror config](https://huggingface.co/unsloth/llama-3-8b/raw/main/config.json) [CODE, unofficial mirror] |
| Llama 3.2 1B | Yes | 2048 | 128,256 | Not public | [mirror config](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json) [CODE, unofficial mirror] |
| Qwen2.5-0.5B | Yes | 896 | 151,936 | Not public | [config](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json) [CODE] |
| Qwen2-7B | No | 3584 | 152,064 | Not public | [config](https://huggingface.co/Qwen/Qwen2-7B/raw/main/config.json) [CODE] |
| Qwen3 | Tied at 0.6B, 1.7B, 4B; untied at 8B and above | n/a | n/a | n/a | [Lopardo et al. 2026](https://arxiv.org/html/2603.26663) [FT-s] |

- Summary pattern, from the table: within Llama and Qwen, small models tie and large ones do not, while Pythia is untied at every size by design. Lopardo et al. report embeddings as a large parameter share at small scale, but the paper's own numbers disagree internally: 73.4% vs "over 50%" for Pythia-70M, and 9.2% vs "approximately 7%" for 2.8B — [Lopardo et al. 2026](https://arxiv.org/html/2603.26663) [FT-s]

**Initialisation scale**
- GPT-1 states that because layernorm is used throughout, a simple N(0, 0.02) init was sufficient — [Radford et al. 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) [FT]
- **Pythia errata.** The README says the 6.9B and 12B models accidentally used a different initialisation because the value was missing from the config — [Pythia README](https://raw.githubusercontent.com/EleutherAI/pythia/main/README.md) [CODE]
- In the linked issue, measured step-0 std on 6.9B input layers is about 0.02, against 0.00988 from the small-init formula. The paper's Table 6 still lists small-init. The issue does not single out the embedding matrix or give 12B values — [Pythia issue #135](https://github.com/EleutherAI/pythia/issues/135) [FT-s]
- `initializer_range: 0.02` in Hub configs does not settle the training init: Pythia configs show 0.02 for all sizes, including those trained with small-init — [70M config](https://huggingface.co/EleutherAI/pythia-70m/raw/main/config.json) [CODE]

**Optimiser and weight decay**
- Pythia: Adam, betas [0.9, 0.95], eps 1e-8, cosine decay, 143,000 iterations, 2,097,152 tokens per batch, about 300B tokens total — [Biderman et al. 2023](https://arxiv.org/html/2304.01373) [FT]
- **Conflict on Pythia weight decay.** Paper Table 6 lists 0.01 ([paper](https://arxiv.org/abs/2304.01373) [FT]); the repo's 160M deduped YAML rendered as 0.1 ([config](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/160M/pythia-160m-deduped.yml) [CODE, via summarising fetch]). I did not resolve this.
- Decay on embeddings varies by codebase. GPT-NeoX/Pythia applies decay only to weights used in the forward pass, OLMo v1 applies none, and OLMo v1.7 does apply it to embeddings. The authors note that the exact scheme is often undocumented — [Land & Bartolo 2024](https://arxiv.org/abs/2405.05417) [FT]

**Which rows get gradient**
- All output-embedding rows influence predictions at every step, tied or not. In an untied model, input rows for tokens absent from the batch are affected only by any weight decay; they tend to zero under decay or stay at their low initial values otherwise — [Land & Bartolo 2024](https://arxiv.org/abs/2405.05417) [FT]
- In tied OLMo-1B, output-layer gradients make up about 70% of gradient norm in the first 1000 steps. The tied matrix aligns better with an untied output matrix than with an untied input matrix: orthogonal-map cosine 0.669 vs 0.420. Scaling input gradients 5× barely moved this (0.216 → 0.222 similarity to untied input) and gave no consistent downstream gain. Each setting is a single run — [Lopardo et al. 2026](https://arxiv.org/html/2603.26663) [FT-s]

**Pretraining**
- OpenAI's GPT-2 source initialises `wte` from a random normal. OLMo initialises `wte` from a truncated normal. Pythia sets an init method in its training config. None loads external vectors — [GPT-2](https://raw.githubusercontent.com/openai/gpt-2/master/src/model.py), [OLMo](https://raw.githubusercontent.com/allenai/OLMo/main/olmo/model.py), [Pythia config](https://raw.githubusercontent.com/EleutherAI/pythia/main/models/160M/pythia-160m-deduped.yml) [CODE]
- Papadimitriou & Prince describe step 0 of Pythia and OLMo as "just a random initialization" — [Papadimitriou & Prince 2025](https://arxiv.org/abs/2510.07613) [FT]
- Pretrained embeddings did help in low-data RNN-era NMT, up to 20 BLEU points in the most favourable setting — [Qi et al. 2018](https://arxiv.org/abs/1804.06323v2) [ABS]
- Frozen tables are a live research idea, not practice. A 2026 preprint froze a sign-quantised PCA of Gemma 3 270M's embeddings: 0.203 nats worse in loss, ties or wins on some benchmarks, 1.7 to 2.3 nats better on tokens seen 10 to 99 times. One seed per arm, and the codes inherit Gemma's own training — [Embeddings Are Overpriced? 2026](https://arxiv.org/html/2610.22428v1) [FT-s]

### Inferences
- Per-row update asymmetry: in untied models the input row for token t gets gradient only through the deep network and only when t is in the batch. The output row gets a direct softmax gradient at every step. In tied models the single matrix gets both, with the output side dominating. So "the embedding of GPT-2" and "the input embedding of Pythia" are different objects, and neither should be treated as a generic LM embedding.
- Small-init std computed from the formula sqrt(2/(5d)), which reproduces the issue's 0.00988 at d=4096: 0.028 at d=512 and 0.0088 at d=5120. If Pythia-12B's embedding really started at 0.02, that is about 2.3× the intended scale, and it makes the two largest Pythia models a different init condition from the rest. That confounds any Pythia size comparison (see Q6).

### Gaps
- Training-time init for Llama and Qwen is not public; I found no primary source. Their absence of pretraining is an inference from the lack of any such claim, not a documented fact. I found no explicit statement from any lab saying "we do not use pretrained vectors".
- Whether the 0.02 measured on Pythia 6.9B includes the embedding matrix is unconfirmed, but it can be checked directly from the step-0 checkpoints.
- GPT-3/GPT-4 and other closed models: tying status not verified.

## Q2. How do word2vec, GloVe and fastText differ, and what does the factorisation view say?

### Takeaway
Static methods fit a shallow, explicit model of local word-context co-occurrence, which gives them a closed-form target. LM embeddings are the first layer of a deep next-token predictor. There is no equivalent theorem for them except in the trivial zero-layer case.

### Cited Findings

| | word2vec | GloVe | fastText | LM input embedding |
|---|---|---|---|---|
| Objective | Log-linear prediction: CBOW predicts the word from context, skip-gram the reverse; no nonlinear hidden layer. Hierarchical softmax in the first paper, negative sampling in the second | Weighted least squares on log co-occurrence, with f(x) = (x/xmax)^α, xmax = 100, α = 3/4 | Skip-gram where a word is the sum of its character n-gram vectors | Next-token cross-entropy, through the full network |
| Context | C = 10 in experiments, random R in [1, C] so distant words are sampled less | 10 left, 10 right by default; pairs d apart contribute 1/d | Skip-gram window | Preceding sequence, up to max length (e.g. 2048 in Pythia), causal only |
| Optimiser | SGD (from memory, not re-verified in this read) | AdaGrad, lr 0.05, 50 to 100 iterations | Not checked | Adam/AdamW |
| Unit | Word (plus phrases in the second paper) | Word | Word + subword n-grams | BPE fragment |
| Final vector | Word vector | W + W̃ | Sum of n-grams | Row of E |
| Source | [Mikolov 2013a](https://ar5iv.labs.arxiv.org/html/1301.3781) [FT-s], [Mikolov 2013b](https://arxiv.org/abs/1310.4546) [ABS] | [Pennington 2014](https://aclanthology.org/D14-1162.pdf) [FT] | [Bojanowski 2017](https://aclanthology.org/Q17-1010/) [ABS] | Q1 sources |

- Levy & Goldberg: for sufficiently large d, SGNS is optimised at w·c = PMI(w,c) − log k. Below that dimension it becomes weighted matrix factorisation of shifted PMI, with the weights set by pair counts and expected negatives — [Levy & Goldberg 2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/b78666971ceae55a8e87efb7cbfd9ad4-Paper.pdf) [FT]
- NCE analogously factorises a shifted log conditional, log P(w|c) − log k — same source [FT]
- SVD on shifted PPMI matches SGNS on similarity but lags on analogies, and optimising the objective better does not guarantee better linguistic-task performance — same source [FT]
- Zero-layer transformers model bigram statistics, and the bigram table can be read directly from the weights — [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html) [FT-s, first 100k chars only]
- GloVe found better syntactic-subtask performance with small and asymmetric windows — [Pennington 2014](https://aclanthology.org/D14-1162.pdf) [FT]

### Inferences
- A softmax-trained zero-layer model has the optimum u_w·e_t = log P(w|t) + const_t (my derivation, the standard softmax result). That is the same family as the Levy & Goldberg NCE case, but asymmetric: next-token only. This is the only setting where I can state a factorisation target for LM embeddings.
- With depth, E is only required to supply features that later layers can use, so no fixed target matrix exists. The Yao et al. result in Q4 suggests the bigram-like target does describe early training and then decays.
- The narrow, asymmetric next-token signal may be why LM input embeddings favour similarity over relatedness (Q3 and Q5). This is a hypothesis, supported only by the analogy with window-size effects.

### Gaps
- I found no theorem characterising the optimum of E in a deep transformer. The theoretical work I found (Yao et al., Q4) is gradient-flow analysis under small init, not a closed form.

## Q3. What do LM input embeddings encode at the end of training?

### Takeaway
Final input embeddings carry real semantic-similarity structure, plus part-of-speech, frequency and leftovers of the random init. Compared with word2vec they are limited by BPE fragments and token rarity. Several famous "LLM geometry" results (space and time, word2vec-style arithmetic) are about activations, not this matrix.

### Cited Findings

**Semantics and syntax**
- Pythia-12B and OLMo-7B input embeddings correlate with SimLex and WordSim, part of speech, Wiktionary and VerbNet hypotheses. They encode similarity much more than relatedness — [Papadimitriou & Prince 2025](https://arxiv.org/abs/2510.07613) [FT]
- Syntactic RSA values are not expected to be high, because each hypothesis is simple; the authors read them comparatively — same source [FT]

**Frequency, norm, position**
- Only the top 100 words fully lose correlation with their random init. Less frequent words keep some, and lexical classes exceed 0.1 at the end of training. Frequency-rank structure rises gradually across training — [Papadimitriou & Prince 2025](https://arxiv.org/abs/2510.07613) [FT]
- Word frequency and function-word status are confounded, as the authors state — same source [FT]
- In OLMo-1B at 10K steps, untied input norms stay near init (0.8 to 1.0), whereas output and tied norms grow with token frequency up to about 10⁴ occurrences and then fall — [Lopardo et al. 2026](https://arxiv.org/html/2603.26663) [FT-s]
- GPT-2-small (tied): more common tokens have shorter vectors on average, the first SVD direction is almost exactly the mean direction, and the mean is much larger than Gaussian data would suggest. No coefficients are given — [Scherlis 2022](https://www.alignmentforum.org/posts/BMghmAxYxeSdAteDc/an-exploration-of-gpt-2-s-embedding-weights) [BLOG]
- In the zero-layer model W^T W of GPT-2, the top logit is usually the input token itself — same post [BLOG]
- The norm-information-gain result (squared norm tracks KL from the unigram distribution, R² = 0.831 for frequency above 10³) is for SGNS word vectors. Its BERT, RoBERTa, GPT-2 and Llama 2 tests use the vector fed into the softmax, not the input table, so this does not establish the result for E — [Oyama et al. 2023](https://arxiv.org/html/2212.09663) [FT-s]

**Under-trained tokens and centroid findings**
- Land & Bartolo detect under-trained tokens by low input-embedding norm (untied) or by output-embedding cosine distance to known unused tokens (tied). Confirmed counts include GPT-2 XL 67/999, GPT-J 200/999, Pythia 6.7B 14/993, OLMo v1.7 178/993, Llama2 7B 20/639 — [Land & Bartolo 2024](https://arxiv.org/abs/2405.05417) [FT]
- In OLMo v1.7 7B, these embedding-based indicators correlate strongly with the number of times a token appears in the first training epoch, across ten orders of magnitude and not only at low counts. This is the strongest direct evidence I found that embedding position and norm encode training exposure — same source, Fig. 2 [FT]
- Rumbelow & Watkins found anomalous strings among the tokens closest to the centroid of the 50,257 embeddings in GPT-J (distances about 0.06 to 0.14). They suggest these tokens had little involvement in training and admit they lack a good argument for the clustering. A footnote says that in GPT2-xl the anomalous tokens tend to lie as far as possible from the centroid — [Rumbelow & Watkins 2023](https://www.alignmentforum.org/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) [BLOG]

**Low-rank and spectral structure**
- Intrinsic dimension (ID) of embeddings: word2vec 24.75, GloVe 24.77, fastText 13.19, and random Gaussian 130.3, all at 300d — [Kataiwa et al. 2025](https://arxiv.org/html/2503.02142) [FT-s]
- Pythia ID: 14M 35.33, 70M 29.99, 160M 26.97, 410M 24.95, 1B 37.23, 1.4B 32.20, 2.8B 34.18, 6.9B 78.30, 12B 121.82, against ambient d of 128 to 5120. LoRA on the embedding layer of Pythia-410M degrades below the estimated ID — same source [FT-s]
- The Layer 0 (static embedding layer) of GPT-2 and BERT is not dominated by rogue dimensions: removing the top dimension leaves r² about 0.999. Later layers can fall to 0.046 (BERT-11). Word2vec is 0.998, GloVe 0.987. RoBERTa layer 0 is an exception at 0.810 — [Timkey & van Schijndel 2021](https://aclanthology.org/2021.emnlp-main.372) [FT]
- Anisotropy (mean expected cosine) at layer 0: GPT-2 0.484, BERT 0.066; word2vec 0.130, GloVe 0.104 — same source [FT]
- For text-embedding use, the mid-spectrum singular directions of the unembedding matrix do better than the edges. The edge subspace carries frequent-token effects (Qwen2.5-0.5B, Llama-3.1-8B, Mistral-7B) — [Wu et al. 2026](https://arxiv.org/html/2606.07502v1) [FT-s]

**Subword structure**
- BPE tokens often have no meaning matching the word. Llama-2-7b splits "northeastern" into four tokens, none corresponding to "north" or "east". Early layers then "erase" token-level information for multi-token words and named entities — [Feucht et al. 2024](https://arxiv.org/abs/2406.20086) [ABS]
- Papadimitriou & Prince restrict to about 28,000 full-word tokens out of 50,688 — [Papadimitriou & Prince 2025](https://arxiv.org/html/2510.07613) [FT-s]
- The pairs moving closest after 20K steps are rare technical nouns and their inflections ("neutrinos"–"neutrino"), and the pairs moving apart include "of"–"teasp" — [Papadimitriou & Prince 2025](https://arxiv.org/abs/2510.07613) [FT]
- Early layers represent many features in superposition via sparse combinations of neurons — [Gurnee et al. 2023](https://arxiv.org/abs/2305.01610) [ABS]

**Analogies**
- I found no primary study of analogy arithmetic on an LM input matrix.
- A blog reports that king − man + woman on Qwen3-8B's token embeddings has cosine of about 0.05 to "queen", with nearest tokens "king" and "woman", against 0.695 for word2vec — [Wantedly blog](https://www.wantedly.com/companies/wantedly/post_articles/1043592) [BLOG, SNIP]

**Results often cited here that are not about E**
- Linear space and time: Llama-2 "learned representations", not established here as the embedding matrix — [Gurnee & Tegmark 2023](https://arxiv.org/abs/2310.02207) [ABS]
- Word2vec-style arithmetic: an additive update applied by feedforward networks — [Merullo et al. 2023](https://arxiv.org/abs/2305.16130) [ABS]
- Linear representation hypothesis with a causal inner product, tested on LLaMA-2 — [Park et al. 2023](https://arxiv.org/abs/2311.03658) [ABS; the abstract does not say which matrices are used]
- Projecting all parameters into embedding space for zero-pass interpretation, which treats vocabulary space as an interpretive frame — [Dar et al. 2022](https://arxiv.org/abs/2209.02535) [ABS]

### Inferences
- The near-centroid result is not general. GPT-J is untied (Land & Bartolo's table marks it "No", with a bias term in the final layer), and its anomalous tokens sit near the centroid, whereas GPT2-xl (tied) is reported as the opposite in a footnote. Tying could explain that split, because rows in tied models are pushed by the output softmax even if never seen as inputs. This is untested. Neel Nanda's comment on that post attributes the clustering to GPT-2's tied embeddings, which does not obviously fit a list computed on untied GPT-J, so the blog-level explanations are unresolved.
- The GPT-2 layer 0 anisotropy of 0.484 (Timkey) and the large mean (Scherlis) agree, despite different methods. I would expect an untied input matrix to differ, but no source measures that head-to-head. Note Timkey's layer 0 for GPT-2 likely includes position embeddings, since it is computed from contextual token runs. I did not verify this from their code.
- Pythia's ID is flat at 25 to 37 from 14M to 2.8B, then jumps at exactly 6.9B and 12B. Those are the two models with the init errata. So the increase could be an init artefact and not a scale effect. This is my observation; Kataiwa et al. do not discuss it and it is untested.

### Gaps
- I found no paper giving nearest-neighbour statistics for a Llama input matrix.
- No primary, quantified GPT-2 centroid or norm-frequency correlation; only blog-level evidence.
- The Kataiwa ID estimator's sensitivity (e.g. to noise and sample size) was not checked, and the ID numbers should not be read as a rank.

## Q4. How does the embedding matrix change over training, and what is the role of Adam and frequency?

### Takeaway
Linguistic structure in E settles early, around 7 to 15% of training, while the vectors themselves keep moving, especially for rare words. Frequent tokens settle first but move less per exposure. Adam's role is mostly theorised: no paper I found isolates it for embeddings. The four papers on the original list (Crosscoding, Tending, Tracing, Belrose) do not track E. The ones that do are Papadimitriou & Prince, Lopardo et al. and Bernas et al., all found during this search.

### Cited Findings

**Papers that track E directly**
- Papadimitriou & Prince (Pythia-12B 153 checkpoints, OLMo-7B 186 checkpoints; RSA with Kendall's τ on Spearman-distance RDMs; robustness with cosine and Euclidean): correlation with SimLex reaches its approximate final value "within 10,000 steps". Linguistic correlations peak at about 15% of training — [Papadimitriou & Prince 2025](https://arxiv.org/abs/2510.07613) [FT]
- Embeddings "continue to shift substantially" after 20,000 steps. Input-output RSA is low overall, peaks at about 15%, and is stronger for high-frequency words — same source [FT]
- Largest pairwise closing after 20K is 0.476 and largest opening 0.176 (Spearman distance) — same source [FT]
- High-frequency words stabilise earlier in absolute time, but "change more slowly with each exposure", while rare words shift more per occurrence — same source [FT]
- Lopardo et al., cosine to init: OLMo-1B-0724 at step 20K output 0.56, input 0.87. OLMo-7B at 50K output 0.38, input 0.75. Pythia-1B first checkpoint output 0.79, input 0.92; end of training output 0.21, input 0.32 — [Lopardo et al. 2026](https://arxiv.org/html/2603.26663) [FT-s]
- Kataiwa et al.: ID drops sharply early in training, then converges gradually (Pythia 14M to 1.4B) — [Kataiwa et al. 2025](https://arxiv.org/html/2503.02142) [FT-s]
- Bernas et al.: mean distance of a token's realisations across post-warmup checkpoints to its own per-token centroid correlates with log-frequency at Pearson ρ about −0.5 for Pythia and SmolLM2, weaker for EuroBERT (−0.2 for 210M, +0.1 for 610M) — [Bernas et al. 2026](https://arxiv.org/html/2604.08764) [FT-s]
- Embedding updates are enriched along a tangent proxy, far more for frequent tokens: Pythia-1B "to" has early enrichment 128.97 and stable 15.80, while "generator" has early 3.27. This measures direction, not magnitude. Adam is not analysed — same source, Appendix E [FT-s]
- Yao et al. (0.7B decoder, AdamW, 36B tokens, top 200 tokens): embedding similarity vs next-token signature similarity rises from near zero, peaks early, then decays to a nonzero plateau. Their theory says low-order signatures are learned before high-order ones under small init — [Yao et al. 2026](https://arxiv.org/html/2608.30315) [FT-s, first 200k chars]
- Beren: embedding quickly learns a token-correlation pattern that "stays very stable", and embedding norm rises fast and then decays slowly, likely from weight decay. This is across Pythia 70M to 1.4B, with no embedding distance-from-init numbers — [beren 2023](https://www.alignmentforum.org/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1) [BLOG]

**Requested papers that do not track E**
- Crosscoding Through Time: crosscoders on hidden states of the middle block of Pythia-1B, OLMo-1B and BLOOM-1B. Features shift from token-specific detectors to syntactic and group-level features. Pythia-1B BLiMP accuracy goes from near chance to above 90% between 128M and 4B tokens — [Bayazit et al. 2025](https://arxiv.org/html/2509.05291) [FT-s, first 100k chars]
- Tending Towards Stability: linear CKA on Attention and MLP outputs, no embedding analysis. Earlier layers converge faster, and larger models stabilise nearly all layers within the first 20% — [Tending Towards Stability 2024](https://arxiv.org/html/2410.11451) [FT-s]
- Tracing Representation Geometry: RankMe and α_ReQ on last-layer last-token representations, with three phases (warmup collapse, entropy-seeking, compression-seeking). Intermediate layers show the same pattern. No embedding-layer result — [Tracing 2025](https://arxiv.org/html/2509.23024) [FT-s, first 100k chars]
- Belrose et al.: theorems tie n-gram statistics to embedding moments, and Pythia's unigram loss bottoms before bigram loss across checkpoints. No experiment tracks E or tests init — [Belrose et al. 2024](https://arxiv.org/html/2402.04362) [FT-s]

**Adam**
- Adam is designed for "very noisy and/or sparse gradients" and is invariant to diagonal gradient rescaling — [Kingma & Ba 2015](https://arxiv.org/abs/1412.6980) [ABS]
- Beren hypothesises diagonal preconditioning interacting with layernorm produces outlier activations, untested and not embedding-specific — [beren 2023](https://www.alignmentforum.org/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1) [BLOG]

### Inferences
- Token scale, using Pythia's 2,097,152 tokens per batch: 10,000 steps is about 21B tokens and 15% of training about 45B tokens. That is the same order as the corpora used for static vectors (GloVe 6B and 42B, word2vec news 100B). So "early in training" may just mean "once it has seen a static-embedding-sized corpus". This would explain the timing without any special mechanism.
- Adam mechanics (my derivation, ignoring bias correction and ε): a token with one isolated exposure, after second-moment memory has decayed, gets a total per-coordinate move of lr·(1−β1)/sqrt(1−β2)·1/(1−β1/sqrt(β2)), independent of gradient magnitude. That is about 5.8·lr for Pythia's β2 = 0.95, and about 32·lr at the common β2 = 0.999.
- Consequently, per-exposure movement is largest for the rarest tokens. This points the same way as Papadimitriou & Prince's per-exposure finding, though they did not test Adam.
- Sparsity alone cannot explain their top-1000-word results. With 2M tokens per batch, a word at frequency 10⁻⁴ occurs about 200 times in every batch. For those words the likelier driver is gradient consistency: Adam's step scales with mean/rms of the gradient, so if signal-to-noise grows as sqrt(n), speed per step grows as sqrt(n) and per exposure falls as 1/sqrt(n). This gives a checkable prediction from public checkpoints.
- Dense Adam means residual momentum keeps moving a row after its gradient goes to zero, so Land & Bartolo's "only affected by weight decay" is approximate for the steps just after an exposure.
- The cos-to-init values show that "stays near init" holds only early. Pythia-1B input reaches 0.32 by the end. Early stability of *relational* structure and continued *absolute* movement are compatible, since RSA ignores global rotation and scale.
- E vs deeper layers: 7 to 15% for E (RSA to linguistic norms) and under 20% for layers in large models (CKA to final) look similar, but metrics and targets differ. Nothing supports a claim that E converges before the layers.

### Gaps
- **No paper compares E's convergence with deeper layers using one metric on one model.** This is the largest gap for this question. Beren's block-0 remark and Bernas et al.'s embedding-input probe are the closest.
- No controlled experiment varies the optimiser (Adam vs SGD, β2, per-row decay) and measures E. All the Adam links above are inference.
- Tied models: dynamics are thin. Papadimitriou & Prince say they know of no large open model with both checkpoints and tying, whereas Lopardo et al. retrain or use tied OLMo-1B. I did not establish whether tied checkpoints are public.
- Papadimitriou & Prince's numeric curve values exist only in figures, and I could not read them. The 0.56/0.21 values are the only final numbers in their text.
- Their Pythia-12B runs are in the init-errata group, so "rare words retain init" may depend on the larger init. OLMo "shows the same patterns", per the authors, which partly mitigates this.

## Q5. Are LM input embeddings better or worse than word2vec/GloVe on similarity benchmarks?

### Takeaway
There is one direct head-to-head on an input matrix, and it says Pythia-12B is roughly level with GloVe on similarity and clearly worse on relatedness. Everything else measures pooled or contextual vectors, and those comparisons are sensitive to which static baseline was used. "LMs beat word2vec" has no clean support for the input matrix.

### Cited Findings

**Closest to a direct test (input matrix)**
- Final Pythia-12B input embeddings vs GloVe: WordSim-Similarity 0.56 vs 0.59, WordSim-Relatedness 0.21 vs 0.50. This uses Kendall's τ on single-token word pairs and is not the usual Spearman benchmark, so it should not be compared with published numbers — [Papadimitriou & Prince 2025](https://arxiv.org/abs/2510.07613) [FT]

**Layer 0 pooled over contexts (close to E, not identical)**
- Word2Vec / GloVe baselines: WS353 0.6838 / 0.6073, SimLex 0.4420 / 0.3705, SimVerb 0.3636 / 0.2271 — [Bommasani et al. 2020](https://aclanthology.org/2020.acl-main.431.pdf) [FT]
- GPT2-12 at N = 100K: WS353 0.6396 (best layer 0), SimLex 0.4547 (best layer 2) — same source [FT]
- GPT2-24 at N = 100K: WS353 0.6830 (layer 0), SimVerb 0.3056 (layer 0) — same source [FT]
- RoBERTa-12: best at layer 0 on all four datasets, including SimLex 0.5098 and SimVerb 0.4206 — same source [FT]
- Scores depend strongly on N, e.g. GPT2-24 WS353 layer 0 is 0.5209 at N = 10K and 0.6830 at N = 100K. The authors also note that several models "do not reliably outperform" word2vec and GloVe — same source [FT]

**Contextual vectors**
- BERT type vectors (bert-large, 10 sentences per word): best static per dataset vs BERT.L4 gives SL-999 0.49 vs 0.55, WS-353 0.71 vs 0.62, WS-SIM 0.76 vs 0.70, WS-REL 0.66 vs 0.51, Google analogies 0.76 vs 0.66. Static models beat BERT on most out-of-context tasks. The authors argue BERT's earlier "wins" reflected suboptimal static baselines, citing Bommasani's word2vec at 0.44 on SL-999 — [Lenci et al. 2022](https://arxiv.org/abs/2105.09825) [FT]
- BERT-base, concatenated last four layers (Table 4; ×100): SimLex 27.65, GloVe 37.52, word2vec 43.61, fastText 49.24; WS-353 Sim 58.66 vs 62.98, 74.08, 79.74 — [Branco et al. 2020](https://arxiv.org/pdf/2011.07997) [FT]
- Decontextualised single words, final hidden state, BATS analogy 3CosAdd: GloVe 0.3481, word2vec 0.3229, LLaMA2-7B 0.1449, OLMo-8B 0.0525, LLaMA3-8B 0.0496, while ADA-002 0.4123 and PaLM Gecko 0.3981 do better. The abstract's claim that LLMs are better on analogies holds mainly for those two API embedding models — [Mahajan et al.](https://arxiv.org/html/2402.11094v3) [FT-s]. This is the live version; [2502.19607](https://arxiv.org/abs/2502.19607) is withdrawn in its favour.
- API text-embedding models with prompts: SimLex 0.692 (embed-english-v3.0), against word2vec 0.40 and LexVec 0.48 — [Ranjan 2025](https://arxiv.org/abs/2512.06744) [ABS]. These are purpose-trained or pooled text-embedding models, not an LM's E.
- Postprocessing (standardisation) raises similarity scores in almost all layers, except early BERT layers. Scores are only in figures — [Timkey & van Schijndel 2021](https://aclanthology.org/2021.emnlp-main.372) [FT]

**Benchmark ceilings**
- SimLex-999 across 2015-era models ranges from 0.098 to 0.414, skip-gram being best, with inter-annotator agreement 0.67 — [Hill et al. 2015](https://ar5iv.labs.arxiv.org/html/1408.3456) [FT-s]

### Inferences
- Static baselines are not a fixed point. SimLex for word2vec appears as 0.40 to 0.44 across sources, while Lenci's tuned best static reaches 0.49 and fastText 0.49 in Branco. Any "LM vs word2vec" claim needs the baseline specified.
- The relatedness gap (0.21 vs 0.50) is a more distinctive signature than the similarity tie. Input E looks more like a paradigmatic, narrow-context space than GloVe does.
- Bommasani's layer-0 scores change with N. A pure lookup table would change only through word coverage and through mixing of position embeddings and subwords, so these are not a clean read of E.

### Gaps
- **No published SimLex/WordSim table in standard Spearman form for raw E of GPT-2, Pythia, Llama, OLMo or Qwen.** It takes an afternoon to compute, but no one appears to have reported it. Papadimitriou's SimLex curve is figure-only.
- Single-token coverage biases any such test towards frequent words: many benchmark words are multi-token in BPE. Papadimitriou & Prince's restriction (983 SimLex pairs) is one way of handling this. Coverage for WordSim and for other tokenizers is unknown.
- Timkey's layer-0 scores are figure-only and were not extracted.

## Q6. Do embeddings of larger LMs get measurably better on semantic benchmarks, and is dimension the driver?

### Takeaway
This is not known. I found no study of raw E across model sizes on a semantic benchmark. The nearest pooled evidence is mixed, and the static-embedding literature suggests dimension saturates at a few hundred. Pythia offers a clean natural experiment that nobody appears to have run.

### Cited Findings
- All 16 Pythia models (70M to 12B) were trained on the same public data in the same order, with 154 checkpoints each — [Biderman et al. 2023](https://arxiv.org/abs/2304.01373) [ABS]
- Pooled static vectors, deeper vs shallower model of the same family at N = 100K — [Bommasani et al. 2020](https://aclanthology.org/2020.acl-main.431.pdf) [FT]:
  - SimLex: BERT 0.5007 → 0.5044, GPT-2 0.4547 → 0.4505, RoBERTa 0.5098 → 0.4959, XLNet 0.5185 → 0.5503
  - WS353: BERT 0.7023 → 0.7179, GPT-2 0.6396 → 0.6830, RoBERTa 0.6915 → 0.6563, XLNet 0.6629 → 0.7021
- Final-hidden-state analogy accuracy does not track dimension: PaLM (768d) 0.3981, LLaMA2 (4096d) 0.1449, SBERT (384d) 0.2431, OpenELM (3072d) 0.0165 — [Mahajan et al.](https://arxiv.org/html/2402.11094v3) [FT-s]. The authors raise but do not resolve whether gains are "merely because of scale".
- GloVe: diminishing returns above about 200 dimensions on analogies — [Pennington 2014](https://aclanthology.org/D14-1162.pdf) [FT]
- Skip-gram's best SimLex dimension was 200 — [Hill et al. 2015](https://ar5iv.labs.arxiv.org/html/1408.3456) [FT-s]
- Best static model per dataset is mixed across 300 and 2000 dimensions (SL-999 SVD.synt.2000; WS-353 CBOW.w10.300; Google FastText.w2.300) — [Lenci et al. 2022](https://arxiv.org/abs/2105.09825) [FT]
- Dimension has a bias-variance optimum for static embeddings under PIP loss — [Yin & Shen 2018](https://papers.nips.cc/paper/2018/hash/b534ba68236ba543ae44b22bd110a1d6-Abstract.html) [ABS]. A third-party summary gives a skip-gram optimum near 129, with task optima of 56 to 220; I did not verify these against the paper — [Liner summary](https://liner.com/review/on-the-dimensionality-of-word-embedding) [SNIP]
- ID stays between 25 and 37 from 14M to 2.8B (ambient 128 to 2560), then 78 and 122 at 6.9B and 12B — [Kataiwa et al. 2025](https://arxiv.org/html/2503.02142) [FT-s]

### Inferences
- The flat ID against a 20-fold rise in ambient d argues against dimension being the direct driver up to 2.8B, if ID relates to benchmark quality at all. That link is unproven.
- Bommasani's mixed deltas are small relative to the effect of N, so they offer no strong size effect. Those pairs also change depth and possibly d together, and I did not check whether each pair shares a training set.
- Size confounds to control in a Pythia test: d, depth, and the 6.9B and 12B init errata (Q1). Peak learning rate also varies by size, from 1.0×10⁻³ at 70M to 1.2×10⁻⁴ at 6.9B and 12B ([Biderman et al.](https://arxiv.org/html/2304.01373) [FT]), which under Adam directly sets per-row movement (Q4).
- Suggested test, cheap and CPU-only: load `embed_in` from each Pythia size at step 143000, compute standard Spearman on SimLex and WordSim on one common single-token pair set, with GloVe and word2vec on the same pairs, and report with and without standardisation. Repeat with `embed_out`. Add OLMo-1B (tied and untied) to cover tying.

### Gaps
- No direct evidence on the question as asked, for any family.
- Whether training tokens, width or depth matter more for E is untested. Tied vs untied size comparisons are missing as well.
- Benchmarks may be too coarse to detect differences near a ceiling of about 0.67 on SimLex. Papadimitriou's figure-only curves cannot settle this.
