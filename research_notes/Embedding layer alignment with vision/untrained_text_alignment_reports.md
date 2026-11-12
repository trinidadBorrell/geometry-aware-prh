# Has anyone reported that untrained or embedding-only text representations already align with image representations?

Access labels used throughout:

- **[FT]**: full text retrieved. I keyword-searched all of it (untrained, random init, embedding layer, layer 0, bag-of-words, TF-IDF, GloVe, word2vec, baseline, chance) and read the method and control sections. This is not a cover-to-cover reading.
- **[FT+fig]**: as [FT], and I also inspected the figure image. Values read from a heatmap colourbar are marked "approx." and are my visual estimates, not numbers the authors state.
- **[AB]**: abstract, listing or search snippet only.

## Do the core PRH papers include a random-init or embedding-layer control, and do they report layer-wise alignment including layer 0?

### Takeaway
I found no paper that measures a randomly initialised language model, or its layer 0, against a trained vision model on images and captions. So the Pythia result (layer-0 CKA ~0.39, 10-NN ~0.10, at initialisation) looks unreported, with one caveat: Herbst et al. could only be read as an abstract. The nearest results are Balestriero & LeCun's random-init row, which sits at chance but in a different setup, and two heatmaps of trained models (He et al., Maniparambil et al.) that show high values at the lowest text layer without comment.

### Cited Findings

**Paper-by-paper summary**

| Paper | Access | Untrained or random language model vs vision? | Layer 0 / embedding reported? | What it has instead |
|---|---|---|---|---|
| Huh et al. 2024 | [FT] | No | Possibly searched, never reported | Max over layer pairs; 1 random ResNet-50 in the vision-only pool |
| Gröger et al. 2026 | [FT] | No | No | Permutation null, max-aware |
| Koepke et al. 2026 | [FT] | No | No | k/n chance correction; layer re-selection check |
| You et al. 2026 | [FT] | No | Possibly searched, never reported | Gröger calibration, K = 500 |
| Balestriero & LeCun 2026 | [FT] | **Yes: random-init BERT-S, at chance** | No layer-wise init analysis | TF-IDF and random init on text-only retrieval |
| Han et al. 2026 | [FT] | No | No | Data-mix analysis, uncalibrated mNN k = 20 |
| Maniparambil et al. 2024 | [FT+fig] | No | Heatmap only, first row unlabelled | Shuffling table; averaged GloVe in encoder pool, no number extractable |
| Herbst et al. 2026 | [AB] | **Unknown** | "Across layers", details unknown | "Unimodal and multimodal baselines" |
| He, Trott & Khosla 2025 | [FT+fig] | No | **Yes, layer 0 on the axis of trained models** | Shuffled pairs; scrambled captions |

**Huh et al. 2024 (v5) [FT]**
- **No language-side control.** The keyword search found no random-init language model, no embedding-layer analysis and no BoW baseline. The only untrained model is "1 randomly initialized ResNet-50" in the vision-vision pool of 78 models, and I found no number for it singled out in the text — [arXiv 2405.07987v5](https://arxiv.org/abs/2405.07987v5).
- **Protocol.** Cross-modal alignment uses k = 10 over 1024 WIT samples, class token per vision layer, average-pooled tokens per language layer, then the maximum over pairwise layer comparisons. No layer-indexed result is reported. Last-token features showed no strong signal — [same](https://arxiv.org/abs/2405.07987v5).
- **Layer 0 is probably in the search.** The released code stacks `llm_output["hidden_states"]` — [platonic-rep README](https://raw.githubusercontent.com/minyoungg/platonic-rep/main/README.md). In Hugging Face that tuple contains "the output of the embeddings" plus each layer — [HF docs](https://huggingface.co/docs/transformers/main_classes/output).

**Gröger, Wen & Brbić 2026 (ICML) [FT]**
- **No control of this kind.** The search found no untrained model, embedding-layer analysis or BoW baseline. The null is "absence of a relationship... beyond their marginal statistics", built by holding X fixed and shuffling the rows of Y. It covers the same maximum over layer pairs — [arXiv 2602.14486](https://arxiv.org/abs/2602.14486).
- **Setup.** 204 vision-language pairs, class token and mean pooling as in Huh, maximum across layers reported. The confounders treated are width and depth. Mean-pooled language features are per layer, but layer 0 is not singled out — [same](https://arxiv.org/abs/2602.14486).

**Koepke, Zverev, Ginosar & Efros 2026 [FT]**
- **No control of this kind.** The search found no untrained model, embedding-layer result or BoW/TF-IDF baseline. Chance is k/n, about 9.8×10⁻³ at k = 10, n = 1024 — [arXiv 2604.18572](https://arxiv.org/abs/2604.18572).
- **Layers.** Representations come from all layers following Huh. Layer pairs are chosen on WIT-1024, and a sanity check re-selects them per gallery size. I found no layer-0 result in the text — [same](https://arxiv.org/abs/2604.18572).

**You, Jang, Mo & Jung 2026 [FT]**
- **No control of this kind.** Language features use "all available hidden states", mean-pooled, maximum over layer pairs, calibrated with 500 permutations. No untrained model or embedding-layer result turned up — [arXiv 2609.27252](https://arxiv.org/abs/2609.27252).

**Balestriero & LeCun 2026 [FT]** (preprint, one week old at the time of writing)
- **The only direct random-init row found.** Table 1 is on 1,000 Flickr test pairs, using backbone features, with a separately trained image JEPA. "Random init" gives mutual 10-NN **.010**, R@1 **.001**, R@5 **.011**; chance is .010, .001, .005. Trained encoders reach .042–.062 mutual 10-NN. The authors write "All start at chance" and that alignment "rises from chance" — [arXiv 2610.39302](https://arxiv.org/abs/2610.39302).
- **Why this differs from the Pythia setup.** The text encoder is a BERT from scratch (hidden size 384, 12 layers), and "The JEPAs use the CLS token"; the next-token model uses mean-pooled **final** hidden states. Figure 5 compares image and text encoders "at matched epochs", so at epoch 0 both may be untrained. The Table 1 caption says epoch 199 but does not say which image encoder the Random row uses — [same](https://arxiv.org/abs/2610.39302).
- **Text-only table.** Caption-to-caption R@1 among 5,000 Flickr test captions: TF-IDF **.442**, random init **.178**, next-token .331, bad-view JEPA .381, good-view JEPA .606. Alignment with images is blank for TF-IDF and random init. They state the bad-view JEPA "learns little beyond word overlap (below TF-IDF)" — [same, Table 5](https://arxiv.org/abs/2610.39302).
- **No init row on the PRH benchmarks.** The PRH protocol is mask-weighted mean of each hidden state, clipping, ℓ2 normalisation, mutual 10-NN, maximum over layer pairs. On COCO-5k vs DINOv2-B, Qwen2.5-0.5B scores .100 mutual 10-NN and .571 unbiased CKA. I found no random-init row in Table 17 or on WIT-1024 — [same](https://arxiv.org/abs/2610.39302).

**Han et al. 2026 (ICLR) [FT]**
- **No control of this kind.** The search found no untrained, random-init or embedding-layer result. The metric is mNN with k = 20 on WIT, averaged over ViT-L, DINOv2-G and CLIP-H. Alignment rises with the share of reasoning data, peaking at 75%. No step-0 or layer-wise values are stated in the text — [arXiv 2509.26625](https://arxiv.org/abs/2509.26625).

**Maniparambil et al. 2024 (CVPR) [FT+fig]**
- **Averaged word embeddings are listed, but no number could be attributed to them.** Table A.2 lists `average_word_embeddings_glove.6B.300d` and `average_word_embeddings_komninos` for "the comparison in Figure A.3 and Table A.13". Neither name appears in the Table A.13 rows I could extract, and Figure A.3 does not label individual text encoders, so their CKA cannot be read — [arXiv 2401.05224](https://arxiv.org/abs/2401.05224).
- **Figure A.3 (visual reading).** Kernel CKA with DINOv2 sits at approx. 0.65–0.72 across all text model sizes, including the size bins near 66 MB and 120 MB where these two encoders fall. They share those bins with other encoders, so this does not show their values. The authors conclude that text model size has limited impact — [same](https://arxiv.org/abs/2401.05224).
- **Layer heatmap (Figure A.4, All-Roberta-large-v1 vs DINOv2).** Text layers are numbered 1–26. The lowest text row is visibly lower but still approx. 0.55–0.60 at the last vision layers, against approx. 0.70 at the top. Whether row 1 is the embedding output is not stated. The vision layer dominates: "the image model layer has a far greater effect" — [same](https://arxiv.org/abs/2401.05224).
- **QAP accuracy by layer (Table A.4).** At the last vision column, text layers 6th to 26th give .448, .498, .500, .538, .672. Layers below the 6th are not tabulated. The authors hypothesise that "the text encoder representations do not change as much" — [same](https://arxiv.org/abs/2401.05224).
- **Shuffling control only.** On 5k COCO pairs with DINOv2 and All-Roberta, CKA falls with % shuffled: 0.72, 0.46, 0.27, 0.13, 0.04, 0.01 at 0–100% — [same, Table 1](https://arxiv.org/abs/2401.05224). There is no untrained encoder.

**Herbst, Roschmann, Bouniot & Akata 2026 (NeurIPS SymGeo workshop) [AB]**
- **Abstract content.** It studies where alignment emerges across layers and representation dimensions, and introduces "unimodal and multimodal baselines" because metrics can be confounded by depth and width. Unimodal cross-modal alignment can match or beat aligned VLMs but stays below same-modality alignment — [NeurIPS listing](https://neurips.cc/virtual/2026/157279).
- **Not read.** OpenReview returned a verification page ([paper link](https://openreview.net/forum?id=mSUdvyQ0W3)), which I did not try to get past, and I found no arXiv copy. I cannot say whether it contains a layer-0 or random-init control, and it is the paper most likely to overlap with this result.

**He, Trott & Khosla 2025 (EMNLP; arXiv 2509.20751) [FT+fig]**
- **The prose says the early layers are low.** "Both modalities exhibit low cross-modal predictivity in their earliest layers", though "even early language layers can successfully predict later vision layers". The metric is ridge linear predictivity with cross-validated λ, in both directions, on 1,000 samples — [arXiv 2509.20751](https://arxiv.org/abs/2509.20751).
- **Figure 2 shows layer 0 of trained models.** For Bloomz and OpenLLaMA vs DINOv2 ViT-L, the language axis starts at 0. In the language-to-vision row, predictivity at vision layer 22 is close to flat across language layers, approx. 0.6–0.65 already at layer 0. In the vision-to-language row, language layer 0 at vision layers 19–23 is approx. 0.35–0.45, against a peak of approx. 0.50–0.55 at late language layers — [same](https://arxiv.org/abs/2509.20751). These are trained embeddings, not random ones, and ridge scores are not comparable with CKA or mutual 10-NN.
- **Controls.** Shuffled image-caption pairs give about 0 in both directions. No untrained model. Layer-wise data appear only as figures — [same](https://arxiv.org/abs/2509.20751).

**Other PRH-adjacent papers**
- Tjandrasuwita et al. 2025 [FT]: Appendix D.3 plots randomly initialised networks for alignment vs uniqueness (unbiased CKA, SVCCA, mutual k-NN), figures only, in what its figure captions present as a synthetic-depth setting. The authors report that alignment, performance and depth do not correlate there. It is not a language model on captions — [arXiv 2502.16282](https://arxiv.org/abs/2502.16282).
- Ruthardt et al. 2024 [FT]: a language-free one-hot class baseline gets near-random generalisation of 4.5%. This is class-level generalisation, not an untrained caption encoder — [arXiv 2410.07173](https://arxiv.org/abs/2410.07173).
- Wybitul et al. 2026 [FT]: text and image concepts align from layer 1 inside trained adapter-based VLMs, using synthesised images — [arXiv 2601.08017](https://arxiv.org/abs/2601.08017). No untrained text control found.
- Keyword search found no untrained or embedding-layer alignment control in Bangachev et al. ([2605.28870](https://arxiv.org/abs/2605.28870)), Yoon & Yue ([2507.01201](https://arxiv.org/abs/2507.01201)) or Li et al. on world models ([2608.23720](https://arxiv.org/abs/2608.23720)) [FT, keyword only].

### Inferences
- **The Balestriero result does not contradict the Pythia result.** It uses CLS after 12 random layers (or final-layer pooling), possibly a random image encoder, and Flickr test pairs, not mean-pooled layer 0 against DINOv2. Their random encoder still scores .178 on caption-to-caption R@1, far above what I calculate as chance (about 4/4999, so roughly .0008), so its text kernel keeps lexical signal. This suggests the chance result depends on CLS, the image encoder, or both. The paper does not separate these.
- **"Rises from chance" may be specific to this setup.** Readers may take that sentence as general. A mean-pooled layer-0 control against a trained vision encoder would test it directly.
- **Published maxima may include layer 0.** Huh, You and Balestriero all use hidden states in a max-over-layers protocol. Since no paper reports the winning indices at layer 0, nobody can currently say how often it wins. The crossover at about 1-2B tokens in the Pythia data therefore appears to be new information.
- **The He et al. prose and figure pull apart.** The text says early layers are low, but the language-to-vision panels look nearly flat in language depth. The Maniparambil heatmap and its "text encoder does not change as much" remark point the same way. Both suggest vision depth matters far more than language depth, for trained models.

### Gaps
- Herbst et al. full text is unavailable to me. Read it before claiming priority.
- Whether Maniparambil's first text row is the embedding output, and the GloVe CKA value. The [repo](https://github.com/mayug/0-shot-llm-vision) might settle both; I did not inspect it.
- Which image encoder and pooling Balestriero's Random row uses. It is not stated in the text I read.
- Figure 10 of Han et al. was not inspected, and no numeric values were found in the text.

## Has anyone compared bag-of-words, TF-IDF or averaged word embeddings with image features using CKA, RSA, kNN overlap or Procrustes?

### Takeaway
At the caption level, I found no CKA, RSA, kNN-overlap or Procrustes comparison of explicit BoW or TF-IDF with image features. At the concept level there is an older line of work, where a word vector is compared with class-mean image features. The closest to the Pythia result, mirrored, is Luo et al.: an untrained ViT against GloVe gives Spearman 0.272, against 0.646 for CLIP vision.

### Cited Findings
- **Luo, Zhang, Xiao & Lake 2024 (CogSci) [FT].** 80 categories, class-mean image vectors vs GloVe, Spearman correlation of cosine similarity matrices. Alignment correlation: ViT-B/16 random **0.272**; ResNeXt supervised 0.396; ViT-B/16 supervised 0.312; ResNet DINO 0.332; ViT-B/16 DINO 0.478; ViT-B/16 iBOT 0.486; CLIP-GloVe 0.646; CLIP-CLIP 0.648. The untrained ViT is explicitly "a reference of baselines". Its recovery accuracy is 89.18%, against 91.73–96.55% for the trained models. The untrained side here is vision, not language, and the comparison is over class means, not images and captions — [paper](https://www.cs.princeton.edu/~bl8144/papers/LuoEtAl2024CogSci.pdf).
- **Li, Kementchedjhieva, Fierro & Søgaard 2023 [FT].** Procrustes on word-level LM representations vs image-class representations, with baselines. Against a target of 79,059 words: random retrieval P@1 0.0015%, P@100 0.1531%; **length-frequency alignment** (two-dimensional word length plus frequency) P@1 0.0032%, P@100 0.6053%; shuffled non-isomorphic P@1 0.0000%. The authors explicitly raise the point that models may "pick up on shallow characteristics shared across the two spaces" such as frequency — [arXiv 2302.06555](https://arxiv.org/abs/2302.06555). Trained LMs only; no untrained LM.
- **Dharmaretnam & Fyshe 2018 (NAACL) [FT].** RSA-style 2-vs-2 test of CNN layers against word vectors over 166 ImageNet concepts. They were "surprised" that even the first VGG-16 layer gives above-chance accuracy with SkipGram (p < 0.001), and that this improves on pixels, though with only "weak evidence for semantic relationships" — [ACL Anthology](https://aclanthology.org/N18-2122/).
- **Collell & Moens 2016 (COLING) [FT].** GloVe vs averaged or max-pooled CNN features, compared by predicting attributes (F1, Spearman), not by CKA/RSA. Vision wins on motion, form and colour; GloVe wins on encyclopaedic and function attributes — [ACL Anthology](https://aclanthology.org/C16-1264/).
- **Roads & Love 2020 (Nature Machine Intelligence) [AB].** Unsupervised alignment of systems via similarity matrices, the method Luo extends — [via Luo et al.](https://www.cs.princeton.edu/~bl8144/papers/LuoEtAl2024CogSci.pdf).
- **Sorscher et al. 2022 [AB, via Huh].** Word embeddings of visual concept names can be isometrically mapped to image embeddings for those concepts — [as described in Huh et al.](https://arxiv.org/abs/2405.07987v5). Not read directly.
- **Crisscrossed Captions (Parekh et al. 2021) [FT].** Average GloVe BoW cosine vs human caption similarity: "weak positive correlation" with BoW scores bunched in a high band, mostly .8–1.0. Not an image-feature comparison, but a useful warning on GloVe BoW kernels on COCO — [arXiv 2004.15020](https://arxiv.org/abs/2004.15020).
- **Maniparambil et al. 2024 [FT+fig].** Averaged GloVe is in the CKA encoder pool with no extractable value; see the first question — [arXiv 2401.05224](https://arxiv.org/abs/2401.05224).
- **Balestriero & LeCun 2026 [FT].** TF-IDF appears only in the text-only retrieval table; it has no image alignment value — [arXiv 2610.39302](https://arxiv.org/abs/2610.39302).
- **Searches.** Two searches aimed at BoW/TF-IDF with CKA/RSA returned no such paper. This is absence from what I could find, not proof of absence.

### Inferences
- **Mean-pooled random embeddings are effectively a random projection of bag-of-words.** With fixed random token vectors and averaging, the feature is a random linear map of the normalised token-count vector, so its kernel approximates a BoW kernel (Johnson-Lindenstrauss). This is my reasoning, not a published claim for this setting. It predicts that explicit BoW or TF-IDF kernels on the same captions should match or exceed layer-0 alignment, which is cheap to test.
- **The Li et al. length-frequency baseline is a useful template.** It is the only paper here that tests a trivial lexical feature against vision models by name.

### Gaps
- No caption-level CKA or mutual k-NN between BoW/TF-IDF and vision features, in any source I found.
- Sorscher and Roads & Love were not read directly.

## How well do bag-of-words or random/untrained text encoders perform as baselines in image-text retrieval and visual-semantic embedding work?

### Takeaway
Averaged word vectors and BoW kernels are strong retrieval baselines, sometimes matching or beating recurrent encoders of their time. All these results train a mapping (CCA, KCCA or a ranking loss), so they are not kernel-alignment measures. I found no retrieval paper that uses a frozen random text encoder as a baseline; "random" in this literature means random ranking or random initialisation followed by training.

### Cited Findings
- **Hodosh, Young & Hockenmaier 2013 (Flickr8k) [FT].** KCCA with text kernels. Image annotation / image retrieval, R@1, R@5, R@10, median rank: NN (IDF-weighted BoW F1) 2.5, 7.6, 9.7, 251 / 2.5, 4.7, 7.2, 272. BoW1 4.8, 13.5, 19.7, 64 / 4.5, 14.3, 20.8, 67. BoW5 6.2, 17.1, 24.3, 58 / 5.8, 16.7, 23.6, 60. Best system Tri5Sem 8.3, 21.6, 30.3, 34 / 7.6, 20.7, 30.1, 38 — [JAIR](https://jair.org/index.php/jair/article/view/10833). Expert score at ≥4.0, first caption: Random 0.5%, NN 3.4%, BoW1 6.6%, BoW5 9.7%, Tri5Sem 11.0% — [same, Table 1](https://jair.org/index.php/jair/article/view/10833).
- **Socher et al. 2014 (TACL; 1,000 images, 5 sentences each) [FT].** The BoW baseline averages pretrained word vectors, with a trained image mapping. Mean rank, lower is better. Image search: Random 52.1, BoW 14.6, kCCA 15.9, SDT-RNN 12.5. Describing images: Random 92.1, BoW 21.1, kCCA 18.0, SDT-RNN 16.9. BoW beats CT-RNN and a recurrent NN on all three tasks of Table 1. Recall in Table 3: image search BoW R@1 15.8, R@5 42.2, R@10 60.0 vs SDT-RNN 16.4, 46.6, 65.6 — [ACL Anthology](https://aclanthology.org/Q14-1017/).
- **Kiros, Salakhutdinov & Zemel 2014 [FT].** No BoW baseline as such, but DeViSE represents sentences "as the mean of their word embeddings". Annotation / search R@1, R@5, R@10, median: Flickr8K 4.8, 16.5, 27.3, 28 / 5.9, 20.1, 29.6, 29; Flickr30K 4.5, 18.1, 29.2, 26 / 6.7, 21.9, 32.7, 25. The LSTM model with OxfordNet reaches 18.0, 40.9, 55.0, 8 / 12.5, 37.0, 51.5, 10 on Flickr8K and 23.0, 50.7, 62.9, 5 / 16.8, 42.0, 56.5, 8 on Flickr30K. Random ranking is 0.1, 0.6, 1.1, 631 / 0.1, 0.5, 1.0, 500 — [arXiv 1411.2539](https://arxiv.org/abs/1411.2539). Linear (mean-pool) encoders "perform significantly worse for retrieval than an LSTM".
- **Klein, Lev, Sadeh & Wolf 2015 (CVPR) [FT].** Mean word2vec plus CCA ("Mean Vec"). Flickr8K image search R@1 19.1, R@5 45.3, R@10 60.4; annotation 22.6, 48.8, 61.2. Flickr30K image search 20.5, 46.3, 59.3; annotation 24.8, 52.5, 64.3. On R@1 this beats every deep system listed in the same tables, though NIC has a higher R@10 on Flickr8K image search (64.0). Fisher Vectors go higher still (GMM+HGLMM on Flickr30K R@1: search 25.0, annotation 35.0). This contrasts with Kiros's weak mean-of-words row; the two differ in how the map is fitted, so they are not like-for-like — [paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Klein_Associating_Neural_Word_2015_CVPR_paper.pdf).
- **Faghri et al. VSE++ 2018 [FT].** A GRU over 300-d word embeddings. Keyword search found no BoW, mean-of-word-vectors, TF-IDF or untrained baseline — [arXiv 1707.05612](https://arxiv.org/abs/1707.05612).
- **Burns, Tan, Saenko & Plummer 2019 (ICCV) [FT].** Image-sentence retrieval mean recall, Flickr30K / MSCOCO. Average Embedding, trained from scratch (random init with fine-tuning): **44.3 / 73.7**. Word2Vec: 62.5 / 75.0, with fine-tuning 71.5 / 78.2. FastText: 69.2 / 78.5, with fine-tuning 73.0 / 80.7. LSTM from scratch: 60.0 / 77.5. BERT: 71.8 / 75.4. The paper's main point is that average embeddings often match or beat LSTMs with good word vectors — [arXiv 1908.06327](https://arxiv.org/abs/1908.06327).
- **Wieting & Kiela 2019 (ICLR) [FT, abstract and intro].** Random encoders over pretrained word embeddings (BOREP, random LSTMs, echo state networks) show that modern sentence embeddings gain "surprisingly little". Sentence classification only, no images — [arXiv 1901.10444](https://arxiv.org/abs/1901.10444).
- **"Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers" [AB].** One search hit uses fixed random per-token embeddings as a text-encoder baseline in diffusion transformers. This is image generation, not retrieval — [arXiv 2601.06338](https://arxiv.org/abs/2601.06338).

### Inferences
- **Retrieval evidence supports the claim that word identity carries much of the signal.** Averaged vectors with a fitted map reach most of the performance of richer encoders of the time, in several independent papers.
- **It does not translate directly to the Pythia number.** All these papers train a map and use pretrained or fine-tuned word vectors, whereas the Pythia result fits nothing and uses frozen random vectors.

### Gaps
- No retrieval result found for a frozen, never-trained, random text encoder paired with a trained image encoder.
- Burns's "from scratch" is random init plus training, so it is not an untrained baseline.

## What does the literature say about shared-word structure in captions driving image similarity, including Flickr30k and COCO?

### Takeaway
Caption-level word sharing is treated as a known retrieval signal and, recently, as a theoretical nuisance, but I found no paper that isolates it as a driver of cross-modal alignment scores. The evidence on how far word overlap tracks image similarity is mixed.

### Cited Findings
- **Word overlap retrieves same-image captions well.** On Flickr test captions, TF-IDF retrieves a same-image caption with R@1 .442, above next-token prediction at .331, and random init scores .178 — [Balestriero & LeCun, Table 5](https://arxiv.org/abs/2610.39302) [FT]. Socher reports the same for averaged word vectors, BoW mean rank 11.8 out of 500, close to the best 10.5, saying averaging "already captures a lot of similarity" — [Socher et al.](https://aclanthology.org/Q14-1017/) [FT].
- **Theory treats shared wording as nuisance.** Balestriero & LeCun argue next-token prediction's kernel "relates texts through shared wording" and not shared world states — [arXiv 2610.39302](https://arxiv.org/abs/2610.39302) [FT]. This is a heuristic argument about trained models, by their own account.
- **Image-selected captions often disagree on meaning.** In COCO, most caption pairs selected using image similarity got low human ratings ([0, 3)), though 20,587 positive pairs resulted. BoW vs human similarity has only a weak positive correlation — [Crisscrossed Captions](https://arxiv.org/abs/2004.15020) [FT].
- **Visual denotations link lexically unrelated captions.** On Flickr30k's 30K images and 150K captions, the denotation graph can "capture similarities between syntactically and lexically unrelated descriptions" — [Young et al. 2014](https://aclanthology.org/Q14-1006/) [FT, abstract and intro]. This is a counterpoint: same-image captions need not share words.
- **Manipulating words changes alignment asymmetrically.** On COCO with DINOv2 and trained LLMs at the penultimate layer, nouns-only, nouns+verbs and scrambled captions reduced alignment only in the vision-to-language direction in the primary analysis, with scrambling the largest drop (t(7) = 22.8). In additional LLM families, all manipulations reduced alignment in both directions, including scrambled L→V t(7) = 8.9 — [He et al.](https://arxiv.org/abs/2509.20751) [FT]. This is the nearest published test of "order vs word identity", and no effect sizes are stated in the text.
- **Caption content sets the ceiling.** Denser captions raise alignment — [Huh et al., Figure 9](https://arxiv.org/abs/2405.07987v5) [FT]. Recaptioning to about 500 words raises scores, saturating by 100–250 words — [Koepke et al.](https://arxiv.org/abs/2604.18572) [FT]. Koepke also reports that many-to-many pairing lowers mutual k-NN for fixed k.
- **Frequency is a named alternative at the word level.** Li et al. note that "frequent words may refer to frequently depicted objects" — [arXiv 2302.06555](https://arxiv.org/abs/2302.06555) [FT].

### Inferences
- **Permutation calibration cannot catch this.** Word sharing keeps the true image-caption pairing, so a lexical kernel passes a permutation test by design. If two images have captions sharing "dog", that structure is real and pairing-dependent. That fits the high permutation-calibrated Pythia scores.
- **"Coarse shared structure" and lexical structure may overlap.** Koepke's coarse-structure reading and the layer-0 result are compatible, but no one has measured the overlap.

### Gaps
- No paper found that correlates caption word-overlap directly with image-feature similarity on Flickr30k or COCO for this purpose.
- No published decomposition of a cross-modal score into lexical vs post-lexical parts, such as partialling out a BoW kernel from CKA.

## Is there any paper proposing an untrained-network null as the proper baseline for alignment, rather than a permutation null?

### Takeaway
I found none for vision-language alignment. Within single-modality networks, Cui et al. 2022 come closest, combining random-network nulls with removal of input similarity. In language-brain work, Feghhi et al. 2024 is the strongest analogue: untrained-model scores are explained away by trivial input features, with layer 0 often the best untrained layer.

### Cited Findings
- **Cui, Kumar, Marttinen & Kaski 2022 [FT].** The clearest statement of the confound: RSA and CKA are "confounded by the population structure of data items in the input space, leading to spuriously high similarity for even completely random neural networks". On first ResNet layers, CKA for two noise-perturbed random ResNets is 0.99, above pretrained vs fine-tuned at 0.95; after deconfounding, 0.43 vs 0.72. They propose regressing out input similarity, and test detection against a null of 50 random ResNets (made by permuting weights or adding large noise). In block 4, dCKA detects all model pairs while CKA detects 0 or 0.28 — [arXiv 2202.00095](https://arxiv.org/abs/2202.00095). Vision only (CIFAR/ImageNet), and the random nets are perturbed trained nets.
- **Feghhi et al. 2024 [FT].** Untrained GPT2-XL predictivity on Pereira fMRI is "fully accounted for by sentence length and position". Best untrained layer: EXP1 layer 0 for 6 of 10 seeds, layer 1 for 3, layer 19 for 1; EXP2 layer 1 for 5, layer 2 for 4, layer 5 for 1. Trained-model scores are largely explained by those plus static word embeddings. Only 1.83% and 1.33% of voxels improved significantly with the untrained model — [arXiv 2406.01538](https://arxiv.org/abs/2406.01538). This is a simple-feature control and is brain data, not vision.
- **Schrimpf et al. 2021 [FT, bioRxiv version].** Untrained language models predict brain activity "considerably well"; training improves scores by 53% on average; a random context-independent embedding reaches below 15% of the neural data on average. Untrained performance predicts trained performance. They read this as architecture — [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.06.26.174482v3.full). Feghhi disputes that reading.
- **Usama et al. 2026 [FT].** Language-language only, 14 models, 800 reasoning problems. Random vs random CKA **0.864 ± 0.007**, above trained 0.612 ± 0.011; MNN 0.592 vs 0.435. Embedding-layer CKA **0.000** vs mid-layers 0.944, which they attribute to distinct tokenizer vocabularies. They state the PRH "must therefore be interpreted relative to this architectural baseline" — [arXiv 2605.23315](https://arxiv.org/abs/2605.23315). The zero is striking and I did not establish how activations were pooled, so I would not lean on it. It is a recent preprint.
- **Kapoor, Srivastava & Khosla 2025 [FT].** Vision-vision. Untrained networks serve as a baseline; all metrics are lower than trained, and mean Procrustes alignment rises by 145.26% in early layers and 493.84% in deep layers after training. Early conv layers change little because they compute near-linear input maps "even in untrained networks", and "shared input statistics" drive early convergence — [arXiv 2502.18710](https://arxiv.org/abs/2502.18710).
- **Kornblith et al. 2019 [FT].** Trained CIFAR-10 and CIFAR-100 models share representations that "differ from untrained models" under CKA (Figure 7), shown as a figure — [PMLR](http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf). Not proposed as a null.
- **Luo et al. 2024 [FT].** Uses a random ViT as a reference baseline in practice, without arguing it replaces a permutation null — [paper](https://www.cs.princeton.edu/~bl8144/papers/LuoEtAl2024CogSci.pdf).
- **Ding et al. 2021 [AB].** Uses seed-to-seed distance as a baseline — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/0c0bf917c7942b5a08df71f9da626f97-Paper.pdf).
- **Materials science [AB].** A Nature Machine Intelligence paper reportedly uses a random-weights model as a negative control for "Platonic" geometry in interatomic potentials. I saw this only in a search summary; the page redirected to login — [article](https://www.nature.com/articles/s42256-026-01235-7).
- **Gröger et al. and followers [FT].** The current PRH standard remains the permutation null of independence — [arXiv 2602.14486](https://arxiv.org/abs/2602.14486), [You et al.](https://arxiv.org/abs/2609.27252).

### Inferences
- **The two nulls test different things.** A permutation null asks whether there is any pairing-dependent relationship. An untrained-model null asks whether the relationship exceeds what the input and architecture already give. The Pythia layer-0 result passes the first by a wide margin, so scores above permutation chance do not show learning.
- **A proposal here would be new for vision-language work**, as far as I can find. Cui's input-regression could be adapted by regressing out a BoW or layer-0 kernel on the text side, though Cui's method uses a shared input space, which two modalities do not have.
- **Trivial features deserve checking first.** Feghhi suggests testing caption length alongside BoW, since random mean-pooled features may also encode length, though their mechanism is specific to sum-pooling through a GELU layer and may not apply to mean-pooled layer 0.

### Gaps
- No vision-language paper found that argues for an untrained baseline. Herbst et al.'s "baselines" could be relevant but could not be read.
- Hosseini et al. 2024 and Pasquiou et al. 2022 on untrained or partly trained language models vs brains were seen only in search snippets. The Hosseini link I saw is [this MIT DSpace record](https://dspace.mit.edu/handle/1721.1/155151), which I did not open.
- Coverage: Semantic Scholar's API rate-limited my request, so citation-graph checks of Huh and Gröger were not done. A forward-citation sweep may still find a workshop paper or a short note I missed.
