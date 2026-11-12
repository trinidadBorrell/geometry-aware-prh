# Do better or larger word embeddings align more with vision models or brain data?

Reading tags used on every finding:
- **[FT-v]** full text read, and the numbers were checked against text extracted from the PDF.
- **[FT]** full text read through an HTML render with a model-assisted extraction. The numbers were not re-checked by hand.
- **[PART]** only targeted passages of the full text were read.
- **[AB]** abstract, search snippet or a secondary summary only.

"Ours" means captions and images, n = 10240, linear CKA and mutual kNN. **Flag** marks a metric or stimulus that is not comparable to ours.

## Word embeddings vs image representations: what metric was used, and what was the alignment?

### Takeaway
Word-level alignment with vision is real but is mostly measured on tens to thousands of concepts, using RSA, Procrustes retrieval or human-rating proxies, and we found no study using CKA or mutual kNN on static word vectors. Where static vectors are compared directly with contextual models on word-level stimuli, they are not clearly worse: FastText about equals BERT on colour, and GloVe about equals CLIP's text encoder against CLIP images on 80 concepts.

### Cited Findings

**Direct alignment between static word vectors and image features**
- **Luo, Zhang, Xiao, Lake 2024 (CogSci), [PART, table checked]**. 80 ImageNet concepts (20 in each of 4 domains), GloVe 300d (Wikipedia 2014 + Gigaword 5). Spearman RSA ("alignment correlation"), in the inferred row order:
  - Random ViT 0.272
  - ResNeXt supervised 0.396
  - ViT supervised 0.312
  - ResNet DINO 0.332
  - ViT DINO 0.478
  - ViT iBOT 0.486
  - CLIP image x GloVe 0.646
  - CLIP image x CLIP text 0.648
  - Recovery accuracy runs from 89.18% (random) to about 96.5%, and the text confirms the ordering. The PDF table layout is garbled, so the row order was inferred from column order and the narrative; the text confirms ViT-supervised strength of 0.9994. Headcam (SAYCam) vision models reach about 0.16 against the word space. Word embedding quality and dimension were not varied. — [Luo et al.](https://escholarship.org/uc/item/7dz6b64q). **Flag:** 80 concepts, image-class averages, RSA.
- **Roads & Love 2020 (Nature Machine Intelligence), [FT-v]**. GloVe 300d (Common Crawl, 840B tokens) against a 50d GloVe trained on Open Images label co-occurrence (about 9M images), with a VGG-16 DeepCluster 4096d pixel variant. Up to 64 shared text-image-audio concepts per run. The metric is Spearman correlation of similarity-matrix upper triangles.
  - The headline ρ is not alignment between modalities. It is the correlation between mapping accuracy and alignment correlation: .99 text-image, .92 text-audio, .92 image-audio.
  - That correlation grows with system size: ρ = .16, .67, .96, .98 for 10, 30, 100, 300 concepts.
  - Alignment strength is 0.945 for two systems at 59 concepts and 0.50 at 10 concepts. — [Roads & Love](https://arxiv.org/abs/1906.09012). **Flag:** the main "image" space is label co-occurrence, not a pixel model, and 300d against 50d are fixed design choices. The raw text-image alignment correlation was not extracted (see Gaps).
- **Abdou et al. 2021 (CoNLL), [FT-v]**. Colour terms against CIELAB: 330 Munsell chips, 122 terms, 18 terms for RSA. Kendall's τ RSA and linear-map selectivity.
  - Word-type FastText: RSA 0.23*, linear map 0.72.
  - PMI baselines: RSA 0.11-0.17, linear map 0.70-0.72.
  - BERT controlled context: max RSA 0.24, linear map 0.76.
  - RoBERTa no context: max RSA 0.33. — [Abdou et al.](https://arxiv.org/abs/2109.06129). **Flag:** the target is a perceptual colour space, not a vision model. n is very small for RSA.
- **Aho, Roads, Love 2023 (PNAS), [AB]**. Early child concepts allow more than 85% of visual-word mappings to be correctly inferred — [UCL record](https://discovery.ucl.ac.uk/id/eprint/10179591/). A review reports about 80% with about 21 concepts and about 90% with 83, GloVe for words and images — [secondary, 2602.16584](https://arxiv.org/abs/2602.16584). The PDF download failed, so the method is not confirmed.

**Language models rather than static vectors, for contrast**
- **Li, Kementchedjhieva, Sogaard 2023 (TACL; arXiv title changed), [FT-v]**. Procrustes alignment (with PCA), P@k against 79,059 candidate words, 7,194 pairs in the main split. Best is ResNet152 x OPT30B: P@100 = 64.1%, about 1/3 within 10 words, about 1/20 exact. Random baseline P@100 is 0.1531%. No static word embeddings; words are contextual token averages over Wikipedia sentences. — [Li et al.](https://arxiv.org/abs/2302.06555). **Flag:** a supervised mapping fit on seen concepts, retrieval metric.
- **Merullo et al. 2023 (ICLR), [FT, from a text copy]**. One linear layer from vision features into the space of GPT-J (6B, 4096d). COCO CLIPScore / RefCLIPScore: CLIP 0.762 / 0.804, NFRN50 0.689 / 0.741, BEIT 0.636 / 0.697. Random encoders score CLIP-S about 0.49. No embedding-size sweep; the paper states projected inputs do not correspond to discrete tokens. — [Merullo et al.](https://arxiv.org/abs/2209.15162). **Flag:** a functional generation metric, supervised.
- **Patel & Pavlick 2022 (ICLR), [FT, from a text copy]**. Few-shot grounding of spatial and cardinal terms, and colour, in generative prompts. Top-1 spatial, original worlds: GPT-2 124M 0.11, 355M 0.12, 774M 0.08, 1.5B 0.10, GPT-3 175B 0.45. Random-world control for GPT-3: 0.16. — [Patel & Pavlick](https://mlanthology.org/iclr/2022/patel2022iclr-mapping). **Flag:** prompt accuracy on synthetic grids, no vision model, no embeddings.
- **Bavaresco et al. 2024, [FT, partial numbers]**. RSA against the 180 Pereira concepts. The text states that GloVe (840B CommonCrawl) has higher correlations in the language network than all more powerful language-only models in the picture condition, and the authors note that GloVe was used to select those stimuli. — [Bavaresco et al.](https://arxiv.org/abs/2407.17914). No numeric GloVe value was extracted.

**Rating proxies (each modality against human ratings, not against each other)**
- **Lazaridou, Pham, Baroni 2015 (NAACL), [FT]**. Skip-gram 300d on a Wikipedia 2009 dump (about 800M tokens), CNN fc7 averaged over 100 images per word. Spearman:
  - MEN 0.70, SimLex 0.33, SemSim 0.62, VisSim 0.48 for text only.
  - VisSim 0.63 for the multimodal variant A.
  - Zero-shot image labelling with ridge from skip-gram: P@1 1.5%, P@10 14.2%, P@50 36.1%. — [Lazaridou et al.](https://arxiv.org/abs/1501.02598). **Flag:** rating proxy plus supervised mapping.
- **Kiela & Bottou 2014 (EMNLP), [FT-v]**. Skip-gram 100d. MEN-Relevant Spearman: text 0.62, BOVW 0.40, CNN-Mean 0.64, CNN-Max 0.63, multimodal up to 0.72. — [Kiela & Bottou](https://aclanthology.org/D14-1005/). **Flag:** rating proxy.
- **Bruni, Tran, Baroni 2014 (JAIR), [PART, table checked]**. ukWaC (1.9B) and Wackypedia (820M), bag-of-visual-words. MEN Spearman: Window2 text 0.73, Window20 text 0.68, image 0.43, tuned fusion up to 0.78. On the concrete MEN subset: Window20 0.70, image 0.47, tuned fusion 0.78. — [Bruni et al.](https://doi.org/10.1613/jair.4135). The row order in Table 2 was inferred and cross-checked with Table 5. **Flag:** rating proxy, pre-CNN vision.
- **Silberer & Lapata 2014 (ACL), [PART, table checked]**. Stacked autoencoder over text and visual attributes, McRae nouns (7,576 pairs rated). Inferred row order: semantic similarity T 0.65, V 0.60, T+V 0.70; visual similarity T 0.52, V 0.60, T+V 0.64. Human agreement 0.76 and 0.63. — [Silberer & Lapata](https://aclanthology.org/P14-1068/). **Flag:** rating proxy; attributes, not word embeddings.
- **Collell & Moens 2016 (COLING), [PART]**. GloVe 300d (largest available corpus) against 4096d CNN features, predicting concept attributes (F1, Spearman). Vision is better for motion, form and surface, and colour; GloVe is better for encyclopedic and function attributes. Numbers are in figures only, and the authors also report a similar predictive pattern across attribute types. — [Collell & Moens](https://aclanthology.org/C16-1264/). **Flag:** attribute prediction, not alignment.
- **Marjieh et al. 2023 (ICLR), [FT-v for the correlations]**. Human similarity judgments, 31,320 pairs across 5 datasets, Pearson r. ConceptNet Numberbatch over human tags is the word-embedding method: video r = 0.74 for the best multi-tag method at last iteration, 0.35 for a single random tag. GPT-3 prompting on images averages 0.62. Top image model Swin 0.66. — [Marjieh et al.](https://arxiv.org/abs/2206.04105). **Flag:** target is human similarity on small stimulus sets, and words come from human-generated tags.
- **Lewis, Zettersten, Lupyan 2019 (PNAS letter), [FT, figures unread]**. Word-vector similarities correlate significantly with human animal similarity by shape, skin texture and colour. For sparkle-type verbs, model against judgments is ρ = 0.59 sighted and ρ = 0.63 blind. Animal ρ values are in a figure. — [Lewis et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6765286). **Flag:** behavioural only.
- **Merkx, Frank, Ernestus 2022 (CMCL), [FT]**. Caption-trained visually grounded embeddings from ResNet-152 against text vectors. Adding the grounded embedding to text vectors trained on MSCOCO (6.2M tokens) gives LLR 193.72 for Word2Vec, 111.46 for FastText, 168.34 for GloVe. Adding it after also including pretrained vectors gives 69.72, 47.32, 49.80. — [Merkx et al.](https://arxiv.org/abs/2202.10292). **Flag:** reaction-time regression, not alignment.

**Citations in the brief that could not be matched to a word-embedding alignment result**
- "Lewis et al. 2024" resolves to Lewis, Nayak, Yu, Yu, Pavlick, *Does CLIP Bind Concepts?* (arXiv 2212.10537), which concerns CLIP composition. It has no word-embedding alignment result that we saw — [arXiv](https://arxiv.org/abs/2212.10537). [PART]. Lewis et al. 2019 above may be the intended reference.
- "Sucholutsky et al." The closest is the perspective *Getting aligned on representational alignment* — [arXiv](https://arxiv.org/abs/2310.13018). Not read. Sucholutsky is also a co-author of Marjieh et al.

### Inferences
- The strongest word-level signal is that GloVe trained only on text reaches essentially the same RSA with CLIP images as CLIP's jointly trained text encoder on 80 concepts (Luo). If that holds, a word-level ceiling may be reached by a modest static vector. It may also just reflect that coarse category structure dominates at 80 concepts.
- In Luo, swapping the vision model moves RSA with fixed GloVe by a factor of about 1.8 (0.27 random to 0.49 iBOT). That is evidence of vision-side scaling only, since the word side was fixed.
- These RSA values should not be placed on the same axis as our CKA or mutual kNN. For most of these studies n is about 2 orders of magnitude below ours (Li et al., at about 7,000 pairs, is the exception), and concept averages remove the instance variance that drives our measures.

### Gaps
- The raw text-image alignment correlation in Roads & Love (figure or table not extracted).
- Numeric GloVe values in Bavaresco et al.
- Per-model P@k for all 14 x 14 pairs in Li et al. (figure only).

## Does alignment grow with embedding dimension, corpus size or model quality?

### Takeaway
We found no study that sweeps GloVe or word2vec dimension or corpus size and measures alignment with a vision model. On word-similarity benchmarks static-vector quality clearly grows with corpus size and saturates in dimension at about 200, so the x-axis exists; the y-axis has not been measured. The nearest scaling results for vision use transformers (Li: log-linear with size; Abdou: mostly rising with BERT size on small n), and dimension-only evidence is one retrieval comparison.

### Cited Findings

| Result | x-axis | Target and metric | Finding | Read |
|---|---|---|---|---|
| [GloVe, Pennington et al. 2014](https://aclanthology.org/D14-1162/) | Vector size; corpus 1.6B, 6B, 42B | Analogy, word similarity (not alignment) | "Diminishing returns" above about 200d. Analogy total at 300d: 70.3, 71.7, 75.0. WS353: 65.8 (6B), 75.9 (42B). Corpus growth does not guarantee gains for other models; SVD-L at 42B falls to 49.2 from 60.1 at 6B | FT-v |
| [Sahlgren & Lenci 2016](https://arxiv.org/abs/1609.08293) | ukWaC 1M, 10M, 100M, 1B | TOEFL accuracy (not alignment) | SGNS 15.00, 43.75, 65.00, 76.25. CBOW 15.00, 40.00, 61.25, 75.00. PPMI 26.25, 52.50, 66.25, 71.25 | FT |
| [Li et al. 2023](https://arxiv.org/abs/2302.06555) | Transformer LM size within family | Vision, P@k | "Log-linear, with no observed saturation"; effect of vision model size is weaker. No static vectors | FT-v |
| [Abdou et al. 2021](https://arxiv.org/abs/2109.06129) | BERT mini 256, small 512, medium 512, base 768 | Colour, RSA τ max; linear map max | RSA 0.077, 0.106, 0.097, 0.162 (not monotonic). Linear map 0.729, 0.734, 0.739, 0.740. FastText 0.23 / 0.72 | FT-v |
| [Burns et al. 2019](https://arxiv.org/abs/1908.06327) | HGLMM 300d vs 6Kd | Caption-image retrieval, mean recall | Flickr30K 71.0 vs 73.5; MSCOCO 79.8 vs 80.9. 6Kd has higher variance across language models. GloVe vs Word2Vec reported as "insignificant" in preliminary work, no numbers | FT |
| [Patel & Pavlick 2022](https://mlanthology.org/iclr/2022/patel2022iclr-mapping) | GPT-2 124M to GPT-3 175B | Grounding, Top-1 | Flat at about 0.08-0.12 among GPT-2 sizes, then 0.45 at 175B (spatial) | FT |
| [Marjieh et al. 2023](https://arxiv.org/abs/2206.04105) | Image model parameters; ImageNet accuracy | Human similarity, Pearson | r = 0.39 and r = 0.26, p < 0.001. Vision side only | FT-v |

- **Word2VisualVec** uses a word2vec trained on tags of 30M Flickr images at 500d, because Flickr-trained vectors are claimed to capture visual relationships better than web-trained vectors. The claim is cited, not measured, and no sweep is reported. — [Dong et al.](https://arxiv.org/abs/1604.06838) [FT-v]
- Two broad searches for dimension or corpus effects on visual alignment returned no such study, and the review [2602.16584](https://arxiv.org/abs/2602.16584) reports no systematic test of size, dimension or capability among the alignment studies it collects, and explicitly takes no position on whether alignment increases with scale. [FT of the first 100k characters]

### Inferences
- A pretrained GloVe grid confounds several things: 50-300d at 6B share one corpus and vary only rank, while 42B and 840B also change vocabulary, casing and source text. Clean corpus-size scaling would need self-trained vectors on nested subsets, as in Sahlgren & Lenci and Hosseini et al.
- Benchmark quality saturating at about 200d suggests any dimension effect on alignment may be small above that point. A 50d point is where a difference is most likely to show.
- Burns' 300d to 6Kd gain is small and comes with a change of embedding method and supervised fine-tuning, so it is at most weak evidence of a dimension effect.

### Gaps
- No measured curve of static-vector alignment with vision against dimension or corpus size was found. Absence from about 8 targeted searches is not proof; we did not chase citations forward from Roads & Love or Luo in a citation index.

## Brain alignment of word embeddings: does the embedding layer predict as well as deep layers, and does scaling appear?

### Takeaway
Static vectors decode and encode significantly but generally lose to contextual layers: in the studies that tested it, middle layers beat the embedding layer, although one RSA study on GloVe-selected stimuli goes the other way. Scaling is weak and depends on the study: about 4.4% per decade of parameters in one fMRI set, saturation by about 100M words or 8B tokens in others, and no monotonic relation with perplexity in a controlled comparison. The only dimension sweep (GloVe 50-300) and the only GloVe corpus comparison show no clean trend.

### Cited Findings

**Classic word-vector decoding and encoding (no scaling tested)**
- **Mitchell et al. 2008, [FT-v]**. 25 verb co-occurrence features from a trillion-word corpus, 60 nouns, 9 participants. Leave-two-out accuracy 0.83, 0.76, 0.78, 0.72, 0.78, 0.85, 0.73, 0.68, 0.82, mean 0.77. Chance 0.5. — [Mitchell et al.](https://www.cs.cmu.edu/~tom/pubs/science2008.pdf). **Flag:** isolated nouns, fMRI, pairwise decoding.
- **Huth et al. 2016, [FT]**. 985d co-occurrence space, corpus with about 1.55B lexicon-word occurrences, window 15. 7 subjects, more than 2 hours of stories. About 4 shared semantic dimensions. No comparison with LSA, WordNet or word2vec feature spaces and no dimension sweep; the text gives no single summary r. — [Huth et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4852309). **Flag:** voxelwise encoding of narratives.
- **Pereira et al. 2018, [PART, numbers checked]**. GloVe 300d, chosen because it was superior in predicting behavioural data; concepts picked by spectral clustering of that space. Experiment 1: 180 concepts, 16 participants. Mean pairwise accuracies 0.77, 0.73, 0.69, 0.64, and average rank accuracy 0.74 (paradigm order inferred). Sentences: rank accuracy 0.76 and 0.79 over 384 and 243 sentences. — [Pereira et al.](https://www.nature.com/articles/s41467-018-03068-4). **Flag:** a stimulus bias in favour of GloVe. Ryskina et al. confirm the GloVe clustering ([arXiv](https://arxiv.org/abs/2508.11536)), and Bavaresco et al. raise it as an explanation for GloVe's strong result.
- **Abnar et al. 2018, [PART]**. Mitchell's 60 nouns, leave-2-out (chance .5). Dependency-based word2vec is the best neural embedding; dependency word2vec, GloVe and the 25-feature model are each top-ranked for at least one subject; FastText beats word2vec. Results are in a figure. They also cite Murphy et al. 2012 at 83.1 with 1000-dimensional corpus-based vectors. — [Abnar et al.](https://arxiv.org/abs/1711.09285). **Flag:** cross-model differences on 60 nouns are compared on few subjects.

**Static vs contextual, and embedding layer vs depth**
- **Caucheteux & King 2022, [FT]**. 400 isolated Dutch sentences, 102 subjects, fMRI and MEG, ridge encoding, Pearson R. Word-embedding-layer score peaks at fMRI R = 0.052 ± 0.004 in left STG. Middle layers beat input layers by fMRI ΔR = 0.031 ± 0.001 and MEG ΔR = 0.009 ± 0.001. Random embeddings still score fMRI R = 0.019 and MEG R = 0.018. Brain score against word-prediction accuracy across embeddings: MEG 0.77, fMRI 0.57; for middle layers 0.86 and 0.81. Corpus fixed at about 278M words, training steps varied. — [Caucheteux & King](https://pmc.ncbi.nlm.nih.gov/articles/PMC8850612). The paper's model counts are internally inconsistent in the text we read.
- **Caucheteux, Gramfort, King 2021 preprint, [FT]**. Narratives dataset. Using GPT-2 layer 8, the correlation between brain score and comprehension reaches R = 0.50, and it is 0.12 lower with GPT-2's word embedding than with the contextual embedding. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.04.20.440622v3.full).
- **Schrimpf et al. 2021, [FT of first 100k characters]**. "Generally, embedding models such as GloVe do not perform well on any dataset", with values in a figure only. A random context-independent embedding of matched dimensionality predicts on average below 15% of neural data. Intermediate layers outperform first and output layers. Untrained GPT2-xl averages about 51%. Next-word prediction correlates with brain score at r = .44. — [Schrimpf et al.](https://www.biorxiv.org/content/10.1101/2020.06.26.174482v3.full). **Flag:** sentence-level stimuli, and the task and datasets differ.
- **Goldstein et al. 2024, [PART]**. ECoG in IFG during a podcast, zero-shot encoding, 81 electrodes. GPT-2 (1600d reduced to 50d by PCA) is significantly better than GloVe in a figure, FDR q < 0.01. — [Goldstein et al.](https://www.nature.com/articles/s41467-024-46631-y).
- **Bavaresco et al. 2024** is the counter-example: GloVe beats larger language-only models in the language network in the picture condition, but on GloVe-selected stimuli and with RSA. — [Bavaresco et al.](https://arxiv.org/abs/2407.17914) [FT, partial numbers]

**Scaling**
- **Antonello, Vaidya, Huth 2023, [FT]**. OPT 125M to 30B. Performance rises about 4.4% per decade of parameters (r = 0.91), about 15% overall, with a plateau beyond about 30B attributed to width-driven regression conditioning. 3 subjects, about 20 hours of stories. Training-data scaling is much stronger: 122% per decade of stories for OPT-125M layer 9. OPT peaks about 3/4 of the way through the model, LLaMA in relatively early layers. No static embedding baseline and no explicit embedding-layer analysis. — [Antonello et al.](https://arxiv.org/abs/2305.11863).
- **Pasquiou et al. 2022, [FT-v]**. 51 participants, *The Little Prince*, fMRI, all at the same 768 units. Untrained LSTM and GloVe (that is, random embeddings) both average about 6.3%. Trained gain over untrained at 2 layers: GloVe 1.5%, LSTM 3.1%, GPT-2 4.5%, BERT 4.4%. LSTM subset scores do not rise simply with corpus size (1-layer Gutenberg 240M to 4.4G: 0.0973, 0.0957, 0.0934, 0.0920). The paper concludes no simple relation with perplexity. GloVe trained on the Full dataset vs Wikipedia (425M) is a contrast map with no scores, and off-the-shelf Wikipedia models "likely lack statistical power". — [Pasquiou et al.](https://arxiv.org/abs/2207.03380). This is the only corpus manipulation of a static embedding against brain data that we found.
- **Hollenstein et al. 2019 (CogniVal), [FT]**. GloVe 50, 100, 200, 300, word2vec, FastText, WordNet2vec, ELMo, BERT against 15 eye-tracking, EEG and fMRI sources. MSE, lower is better. BERT, ELMo and FastText are best overall. Dimension has no clean effect: GloVe-50 has lower eye-tracking error than GloVe-300 on GECO (0.010 vs 0.020), BERT-base often beats BERT-large. fMRI at 100 / 500 / 1000 voxels: GloVe-300 0.119 / 0.081 / 0.078, BERT-base 0.020 / 0.017 / 0.016. — [Hollenstein et al.](https://arxiv.org/abs/1909.09001). **Flag:** MSE per predicted dimension is not an alignment measure and may not be comparable across input dimensions. This is the only dimension sweep we found, in any target.
- **Hosseini et al. 2024, [FT of first 100k characters]**. GPT-2 (768d, 12 layers) on 1M, 10M, 100M, 1B words, Pereira2018. Against the fully trained model: 1M p = 0.007, 10M p = 0.016, 100M p = 0.99, 1B p = 0.82; untrained p < 0.0001. Numeric scores are only in a figure. Layer 0 is the token embedding, and an exploratory layer analysis shows layers 10-12 improving with data. — [Hosseini et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025646/).
- **AlKhamissi et al. 2025, [FT of first 100k characters]**. 8 Pythia sizes, 34 checkpoints each. Alignment rises sharply and peaks near 8B tokens. Model size is not a reliable predictor once feature count is controlled. Untrained models reach about 50% of pretrained alignment. No layer-by-layer analysis of trained Pythia, so no embedding-layer curve. — [AlKhamissi et al.](https://arxiv.org/abs/2503.01830).

### Inferences
- The embedding layer is typically below the best deep layer in encoding brain data, but above an empty null: random embeddings are significant in Caucheteux & King and about equal to untrained LSTM in Pasquiou. Any word-embedding experiment needs a matched random-vector baseline.
- Brain studies vary the model and read out one layer; none plots embedding-layer score against size or capability. Antonello's point about width and regression conditioning means a ridge-based dimension sweep is confounded. CKA and mutual kNN do not have that ridge issue, though they have their own dimension behaviour.
- Results on Pereira Experiment 1 should not be used as evidence for GloVe's quality, given how its stimuli were picked.

### Gaps
- Numeric brain scores for GloVe, word2vec and others in Schrimpf et al. and Hosseini et al. are in figures.
- Goldstein et al. 2022 (Nature Neuroscience) was not read.
- A search snippet attributes to [Antonello et al. 2021](https://arxiv.org/abs/2106.05426) the claim that BOLD temporal blurring makes GloVe variants nearly identical for predicting brain data. We did not verify this. If true, it would limit word-embedding quality effects in fMRI.

## How do sentence-level averaged word vectors do for caption-image alignment (Word2VisualVec, Dong et al.)?

### Takeaway
This is the closest literature to our stimuli, and averaged vectors do well, but only as inputs to supervised retrieval models. We found no unsupervised geometric alignment measure on mean word vectors for captions and images; retrieval recall is not comparable with our CKA or mutual kNN.

### Cited Findings
- **Word2VisualVec (Dong, Li, Snoek), [FT-v for setup and stop words, FT for tables]**. Mean of 500d Flickr-tag word2vec, mapped by an MLP to ResNet-152 features. Image-to-sentence R@1 / R@10 / median rank:
  - Flickr8k: word2vec 31.9 / 75.3 / 3, bag-of-words 33.6 / 75.3 / 3, word hashing 33.9 / 73.4 / 3.
  - Flickr30k: word2vec 36.6 / 74.2 / 2, bag-of-words 39.7 / 76.7 / 2, word hashing 40.3 / 73.3 / 2.
  - Training data helps bag-of-words overtake word2vec. Train on Flickr8k, test on 30k: word2vec 27.9 / 66.2 / 5 is better at R@1 and R@10 than bag-of-words 26.7 / 60.7 / 5. — [Dong et al.](https://arxiv.org/abs/1604.06838). **Flag:** 1,000-image test sets, supervised, retrieval.
- **Stop words strongly change plain averaged vectors in the same paper:** text-only retrieval mAP for word2vec without the learned map rises from 18.3 to 31.1 (Flickr8k) and 14.7 to 28.4 (Flickr30k) when stop words are excluded. — [Dong et al.](https://arxiv.org/abs/1604.06838) [FT-v]
- **Burns, Kim, Plummer, Saenko 2019 (ICCV), [FT]**. Flickr30K and MSCOCO retrieval, mean recall:
  - From-scratch average embedding: 38.2 and 73.7. Table 1 of the paper shows 44.3 for the Flickr30K value, an internal conflict.
  - Word2Vec average: 62.5 and 75.0, or 71.5 and 78.2 with fine-tuning.
  - FastText average + fine-tuning: 73.0 and 80.7.
  - Sentence-level InferSent: 71.2 and 76.4. BERT: 71.8 and 75.4.
  - Word2Vec LSTM + fine-tuning: 69.0 and 78.2. — [Burns et al.](https://arxiv.org/abs/1908.06327). **Flag:** supervised retrieval, and the choice of embedding is confounded with fine-tuning.
- **Maniparambil et al. 2024 (CVPR), [AB]**. CKA on image-caption benchmarks. Unaligned and aligned encoders have semantically similar spaces, and matching without training is possible. — [Maniparambil et al.](https://arxiv.org/abs/2401.05224). The nearest metric match to ours, but the text encoders are not word-level as far as the abstract shows.
- **Schnaus et al. 2025 (CVPR), [FT of first 100k characters, appendix unread]**. Blind matching over 33 vision and 27 language models, language models from SentenceTransformers plus CLIP; GloVe, word2vec and FastText do not appear in the text read. CIFAR-10 matching accuracy by metric: Gromov-Wasserstein 69.5%, CKA 63.5%, mutual kNN 59.0%. CINIC-10: 79.0%, 80.0%, 70.0%. For vision models, the pre-training method has greater impact than model size. — [Schnaus et al.](https://arxiv.org/abs/2503.24129). **Flag:** class-level matching at N = 10 for these figures.
- **Our own data point (not literature).** In exp-018, random-init Pythia mean-pooled at layer 0 already gives CKA 0.39 against DINOv2-g and mutual kNN at k = 10 of 0.10-0.11 against ViTs, read in the README as a random projection of caption word counts and lexical overlap. Final trained values are mutual kNN 0.19-0.22 and CKA 0.53-0.70, and the best cell stays at layer 0 until step 512-1000. — [exp-018 README](C:/Users/Oddharak/projects/geometry-aware-prh/experiments/exp-018-alignment-emergence/README.md).

### Inferences
- A caption's mean word vector mixes three separable things: lexical overlap (available to random vectors), distributional semantics (added by trained vectors), and weighting (stop words, frequency). The Dong result implies that weighting can matter as much as the vector source.
- Bag-of-words matching or beating word2vec given enough data in a supervised setting is consistent with exp-018's finding that word-count structure alone carries substantial alignment.
- Because captions sharing content words often describe similar images, averaged vectors could look strongly aligned on captions without any semantic quality gain, so any quality scaling has to be measured as excess over a random-vector baseline.

### Gaps
- No measured unsupervised alignment of mean word vectors with image features on captions was found.
- No study relating caption-level alignment to word-level alignment for the same embeddings was found.

## What is the relationship of word-embedding quality to PRH-type bits-per-byte scaling?

### Takeaway
We found no work plotting embedding-layer or static-vector alignment against model capability. PRH itself takes a maximum across average-pooled layer pairs, so the embedding layer is not reported separately, and the nearest work on input embeddings does not touch vision.

### Cited Findings
- **Huh et al. 2024, [FT]**. Mutual kNN with k = 10; cross-modal n = 1,024 from WIT; vision-vision n = 1,000 from Places-365. LM score is 1 − bits per byte over 4M OpenWebText tokens. Reported as a linear relation in a figure, with alignment reaching only 0.16 and no slope or correlation in the text. Each language layer is average-pooled and the maximum over pairwise layer comparisons is taken. The paper does not say whether layer 0 is included, and static vectors are not tested. — [Huh et al.](https://arxiv.org/abs/2405.07987). **Flag:** n is 10x smaller than ours, on Wikipedia captions, and only mutual kNN is the main metric.
- **Papadimitriou & Prince 2025, [AB plus first pages]**. RSA on the input and output vocabulary embeddings of Pythia 12B (153 checkpoints) and OLMo 7B (186 checkpoints) against semantic, syntactic and frequency measures, including GloVe as a reference. The geometry quickly converges to high correlations. High-frequency and function words settle faster. No vision or brain target. — [Papadimitriou & Prince](https://arxiv.org/abs/2510.07613).
- **Llama-2 embedding comparison, [FT]**. Input embeddings at 4096d (7B) and 8192d (70B). Global clustering 0.325 vs 0.215, ASPL 3.392 vs 3.353, correlation with a cross-lingual semantic map 0.466 vs 0.449. Only two sizes and no vision or brain data, and size and dimension are confounded. — [arXiv 2502.11380](https://arxiv.org/abs/2502.11380).
- **Brain analogues.** Caucheteux & King relate brain score to word-prediction accuracy across embeddings (MEG 0.77, fMRI 0.57) — [source](https://pmc.ncbi.nlm.nih.gov/articles/PMC8850612). Pasquiou reports no monotonic relation with perplexity — [source](https://arxiv.org/abs/2207.03380). AlKhamissi reports alignment tracking formal competence and not functional competence — [source](https://arxiv.org/abs/2503.01830). None isolates the embedding layer against capability.
- **Project data.** In exp-018 curves of different Pythia sizes do not collapse when plotted against bits per byte. — [exp-018 README](C:/Users/Oddharak/projects/geometry-aware-prh/experiments/exp-018-alignment-emergence/README.md)

### Inferences
- A static vector has no bits-per-byte. Options for a common x-axis are benchmark quality (SimLex, MEN, analogy, as used in GloVe and Sahlgren & Lenci), corpus size, or a low-order n-gram LM's bits per byte. These are not the same capability axis, so a plot against PRH's axis would be an analogy, not a direct extension.
- If PRH's maximum sometimes selects low layers for weak models, part of its low end could be a lexical floor. exp-018 shows this on Pythia checkpoints. Whether it holds across the final PRH suite is untested as far as we found.

### Gaps
- Whether layer 0 was among the layers maximised over in Huh et al., and how often it wins, is not stated in what we read.
- The relationship between trained token-embedding quality and capability across final models of different sizes was not found.

## What remains unstudied?

### Takeaway
A scaling experiment on word embeddings against vision, measured with calibrated CKA and mutual kNN on captions and images, appears to be new. Confidence is moderate: this is based on targeted searches and full-text checks of the named papers, not an exhaustive citation search.

### Cited Findings
Each item reflects an absence in the papers read above:
- Alignment of static vectors with vision against **dimension**: fixed at 300d in [Luo](https://escholarship.org/uc/item/7dz6b64q) and [Roads & Love](https://arxiv.org/abs/1906.09012), 500d in [Dong](https://arxiv.org/abs/1604.06838). None sweeps it.
- Alignment of static vectors with vision against **corpus size**: corpus scaling exists only on benchmarks ([Sahlgren & Lenci](https://arxiv.org/abs/1609.08293), [GloVe](https://aclanthology.org/D14-1162/)) and, as a two-level contrast map, on brain data ([Pasquiou](https://arxiv.org/abs/2207.03380)).
- **Static vectors with large-n geometric measures on captions**: [Maniparambil](https://arxiv.org/abs/2401.05224) and [Schnaus](https://arxiv.org/abs/2503.24129) use sentence-level encoders. [Li](https://arxiv.org/abs/2302.06555) has no static vectors.
- **Embedding layer against capability, in vision or brain**: not reported in [Huh](https://arxiv.org/abs/2405.07987), [Antonello](https://arxiv.org/abs/2305.11863) or [AlKhamissi](https://arxiv.org/abs/2503.01830).
- **Word-level against caption-level alignment for the same embedding**: not found.
- **A random-vector baseline of matched dimension for word-to-vision alignment**: [Schrimpf](https://www.biorxiv.org/content/10.1101/2020.06.26.174482v3.full) and [Caucheteux & King](https://pmc.ncbi.nlm.nih.gov/articles/PMC8850612) do this for brain data, and [Luo](https://escholarship.org/uc/item/7dz6b64q) randomises the vision side only.

### Inferences
Suggested design for a follow-up (our proposal, untested):
- **Rank:** GloVe 6B at 50, 100, 200, 300d, plus PCA truncation of 300d and 840B as a rank null.
- **Corpus:** self-train SGNS and GloVe at fixed 300d on nested subsets (1M, 10M, 100M, 1B), as in Sahlgren & Lenci and Hosseini. Pretrained 6B, 42B and 840B can serve only as a secondary comparison, because of vocabulary and casing changes.
- **Controls:** random vectors of matched dimension on the same vocabulary as the floor, which gives the lexical-overlap baseline directly and links to exp-018's layer 0; a bag-of-words projection; and permuted word-to-vector assignment. Report excess over these.
- **Pooling:** mean, mean without stop words, and frequency-weighted, given the Dong effect. Track OOV rate by embedding, since coverage changes with vocabulary.
- **Quality axis:** SimLex and MEN measured for each embedding, then alignment against that. Add Pythia layer-0 embeddings from final models and checkpoints for the bridge to bits per byte.
- **Likely outcomes:** Luo's near-ceiling GloVe result and benchmark saturation at about 200d suggest a flat dimension curve above 100-200d. Corpus size is the more likely source of a rising curve. A flat result would itself be informative for the question of whether alignment tracks quality.

### Gaps
- **Search limits.** We did not run a citation-index search, and the search tool returned nothing for several targeted queries. An unindexed or workshop study could exist. [Balestriero and LeCun (2610.39302)](C:/Users/Oddharak/projects/geometry-aware-prh/experiments/exp-018-alignment-emergence/README.md) and Herbst et al. are noted in the exp-018 README but were not re-read here.
- **Abstract-only or secondary reading:** Aho et al. 2023 (PDF download failed), Maniparambil et al. 2024, Sucholutsky et al. (unread), Goldstein et al. 2022 (unread), Antonello et al. 2021 (snippet only), Papadimitriou & Prince 2025 (abstract and first pages).
- **Full texts where nature.com and PNAS pages blocked direct fetch** were read through PDFs or PMC where possible. Schrimpf et al. was read from the bioRxiv full text (first 100k of 289k characters), not the PNAS version, so its later sections (including any further embedding numbers) are unread.
- **Extraction risk.** [FT] numbers came through a model-assisted page extraction and were not hand-checked against PDFs: Antonello, Lazaridou, Sahlgren & Lenci, Hollenstein, Burns, Merkx, Hosseini, AlKhamissi, Schrimpf, both Caucheteux papers, Patel & Pavlick, Merullo, Bavaresco, Huth, Schnaus. Check these before quoting in a paper.
