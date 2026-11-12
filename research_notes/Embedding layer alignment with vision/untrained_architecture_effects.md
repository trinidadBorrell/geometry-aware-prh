# Untrained / random-network alignment: architecture vs input statistics

Reading-depth key, used on every source:
- **[FT]** = I downloaded the full text and extracted passages by keyword search. I did not read it end to end, and values shown only in figures were not read.
- **[FT-sum]** = full text read through a summarising fetch tool. I did not check the numbers against the raw text.
- **[ABS]** = abstract or search-result summary only.

Source wording is paraphrased here; numbers are as printed in the source. Anything else is marked as a gap.

Our setting, for reference: a random-init Pythia, mean-pooled, reaches mutual 10-NN ~0.10 at layer 0 and ~0.05 at a deep layer against vision models. The exp-018 README already describes layer 0 at step 0 as a mean of random token embeddings, i.e. a random projection of the caption's word counts.

## Q1. Language models: what do untrained models achieve, and what does training size add?

### Takeaway
Untrained language models score well above chance against brain data, but whether that reflects architecture is now contested. Two independent critiques attribute it to simple input-derived features (length, position, word identity). One direct comparison finds random static embeddings doing as well as or better than untrained transformers, with depth flat or harmful, which matches our layer-0 vs deep pattern.

### Cited Findings
**Schrimpf et al. 2021 (PNAS) [FT-sum, first 100k of 159k characters]**
- 43 models were scored with initial random weights, still with a trained linear readout on top — [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/)
- Untrained GPT2-xl has average predictivity of ~51%, about 20% below the trained network — [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/)
- Training improves brain scores by 53% on average. Untrained scores predict trained scores across models (r = 0.74, P << 0.00001) — [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/)
- Input-only control: a random context-independent embedding with GPT2-xl's dimensionality predicts on average below 15% of the neural data — [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/). **Contradicted by** Pasquiou 2022 below.
- The authors hedge that minimal-pair model comparisons are still needed to separate architecture from training. They also report that untrained scores on predictivity generalised to the RDM metric, which needs no fitting — [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/)

**Hosseini et al. 2024 (Neurobiology of Language) [FT]**
- Title note: the published title ends "Even After a Developmentally Realistic Amount of Training", not "limited amount" as in the brief — [Hosseini 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025646/)
- Training-size experiment: GPT-2 trained on 1M, 10M, 100M and 1B words. 100M words was not significantly different from the fully trained model (p = 0.99). Untrained, 1M and 10M were significantly below it (p < 0.0001, 0.007 and 0.016, Bonferroni-corrected) — [Hosseini 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025646/)
- Checkpoint experiment: fMRI predictivity rises until the 10% checkpoint, then plateaus. Untrained, 0.1% and 1% are significantly below the fully trained model (ps < 0.001) — [Hosseini 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025646/)
- They built a second untrained GPT-2 (Gaussian, mean 0, SD 0.02) to isolate architecture from initialisation choices; its performance differs from the Hugging Face init — [Hosseini 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025646/)
- Both untrained models are reliably above zero on the Futrell2018 behavioural benchmark (ps < 0.0001). The authors take this to mean that performance there probably does not reflect representation of the linguistic stimuli — [Hosseini 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025646/)

**Caucheteux & King 2022 (Communications Biology) [FT]**
- 32 architectures at 100 training amounts (32,400 embeddings). "Random embeddings", meaning untrained model activations, give significant brain scores: fMRI R = 0.019 ± 0.001, MEG R = 0.018 ± 0.0008, both p < 10⁻¹⁶ — [Caucheteux & King 2022](https://www.nature.com/articles/s42003-022-03036-1)
- For scale, the compositional embedding reaches fMRI R = 0.048 and MEG R = 0.041 — [Caucheteux & King 2022](https://www.nature.com/articles/s42003-022-03036-1)
- Brain score correlates with language accuracy: MEG R = 0.77, fMRI R = 0.57. By layer the correlation is highest in the middle (fMRI 0.81, MEG 0.86), lower at input (fMRI 0.39, MEG 0.73) and output (fMRI 0.63, MEG 0.78) — [Caucheteux & King 2022](https://www.nature.com/articles/s42003-022-03036-1)
- A keyword search of [Caucheteux et al. 2023, Nature Human Behaviour](https://www.nature.com/articles/s41562-022-01516-2) [FT] found no untrained-model result. The "random" hits concern random forecast windows and sampled futures.

**Pasquiou et al. 2022 (ICML) [FT]: the closest parallel to our result**
- Untrained activation matrices included the embedding layer. The baseline, a fixed 768-d vector per word, is described as equivalent to an untrained GloVe model — [Pasquiou 2022](https://arxiv.org/abs/2207.03380)
- Untrained LSTM and random embeddings score about equally, around 6.3% (SE 0.02%), and significantly above transformers (LSTM.4 minus GPT-2.4 = 1.6%, LSTM.4 minus BERT.4 = 0.7%, BERT.4 minus GPT-2.4 = 0.9%) — [Pasquiou 2022](https://arxiv.org/abs/2207.03380)
- With more layers, untrained performance stays flat (LSTM, BERT) or falls (GPT-2) — [Pasquiou 2022](https://arxiv.org/abs/2207.03380)
- Their mechanism: untrained models capture the similarity of brain responses across repeated identical words. The top-10% voxels overlap 79% across untrained LSTM, GPT-2 and BERT — [Pasquiou 2022](https://arxiv.org/abs/2207.03380)

**Critiques attributing untrained scores to simple features**
- [Feghhi et al. 2024](https://arxiv.org/abs/2406.01538) [FT], Pereira dataset: sentence length + sentence position account for 98.4 ± 1.5% (EXP1) and 100.0 ± 0.0% (EXP2) of the neural variance explained by untrained GPT2-XL. After FDR correction, no voxel is better explained by adding the untrained model.
- Mechanisms in [Feghhi et al. 2024](https://arxiv.org/abs/2406.01538): the GELU in the first MLP gives a positive mean output, which under sum-pooling accumulates into sentence length. Untrained absolute positional embeddings make same-position sentences similar under sum-pooling. These are architecture-by-input interactions, not stimulus content.
- [Feghhi et al. 2024](https://arxiv.org/abs/2406.01538) also find that with shuffled splits a non-linguistic autocorrelation feature (OASM) beats GPT2-XL. SP+SL+WORD reaches 72.3 ± 9.2% of GPT2-XL's performance in EXP1 and outperforms it in EXP2.
- [Illusions of Alignment preprint, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.03.09.642245.full.pdf) [FT, not peer reviewed]: positional signals and word rate fully account for untrained LLM predictivity, across three datasets.

**Counter-view: architecture components do matter**
- [AlKhamissi et al. 2024](https://arxiv.org/abs/2406.15109) [FT]: tokenization strategy and multihead attention are identified as the two main drivers of alignment in untrained models. Note that tokenization is an input choice, not architecture.
- [AlKhamissi et al. 2025](https://arxiv.org/abs/2503.01830) [FT]: context integration is reported as the driver of initial alignment in untrained models. Sequence models and mean pooling beat last-token-only readouts, and attention with positional encoding scores highest in a single-block ablation. The checkpoint series includes the untrained model. I did not extract numbers.
- Input-perturbation design in [Kauf et al. 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10187317/) [ABS]: on 627 sentences, word order was perturbed and word subsets removed. Lexical-semantic content, largely from content words, not syntax, drives ANN-brain similarity. This was shown with trained GPT-2 XL, not untrained models.

### Inferences
- Random embeddings at or above untrained transformers (Pasquiou) is the same ordering as our 0.10 vs 0.05. The brain literature therefore already contains precedent for layer 0 beating random depth.
- Schrimpf's below-15% for random embedding vs Pasquiou's parity with LSTM is unresolved. The likely differences are the dataset (sentences vs audiobook) and that repeated words across a long text give random embeddings more to exploit. That second point is my reading, not a tested claim in either paper.
- Word repetition plays the same role in Pasquiou as word overlap between captions does for us.

### Gaps
- Per-model untrained scores for Schrimpf: not in the text I read, and the last 59k characters were not read.
- Hosseini's untrained predictivity values appear only in figures and on OSF.
- No reading confirms whether AlKhamissi 2025's untrained claims survive the Feghhi / Illusions confounds. In my extraction, the 2025 paper cites Feghhi for split design only.

## Q2. Vision: what alignment do random networks achieve?

### Takeaway
Random convolutional networks get substantial brain alignment mainly under fitted linear readouts and in early or monkey data. Under unfitted comparison, or in late human visual cortex, training has a large effect. Random transformers and fully connected networks do far worse than random CNNs, so the vision "architecture effect" is specifically about convolution and locality, and it does not transfer to transformers.

### Cited Findings
- **Conwell et al. 2024 (Nat Commun) [FT]:** 64 randomly initialised counterparts of trained models. Paired trained-vs-random effect is cRSA = 0.30 [0.29, 0.31] and veRSA = 0.56 [0.55, 0.75], p < 0.001, described as the largest and most robust effect in their analyses. The veRSA CI is copied as extracted and looks asymmetric; check against the published page. Untrained models failed to capture the representational structure of late visual stages — [Conwell 2024](https://www.nature.com/articles/s41467-024-53147-y)
- For scale, ResNet50-BarlowTwins scores 0.367 in cRSA and 0.679 in veRSA — [Conwell 2024](https://www.nature.com/articles/s41467-024-53147-y)
- **Kazemian et al. (bioRxiv 2024, version of July 2025) [FT]:** the largest untrained CNN approaches pretrained AlexNet in monkey V4 and IT. In humans it approaches AlexNet in the early visual stream and reaches ~70% of its performance in higher ventral regions — [Kazemian](https://www.biorxiv.org/content/10.1101/2024.05.10.593623.full.pdf)
- In [Kazemian](https://www.biorxiv.org/content/10.1101/2024.05.10.593623.full.pdf), scaling fully connected and transformer architectures gave only small improvements and substantially worse performance, despite matched dimensionality.
- Also there: the linear transformation is reported as crucial for finding the dimensions in wide networks that align with cortex (Extended Data Fig. 2, RSA). Fixed pixel permutation, which removes locality, lowers performance.
- Published version: Nature Machine Intelligence 2025, per a search summary [ABS]; I read only the preprint.
- **Geiger et al. (ICLR 2022; preprint) [FT]:** an improved initial distribution yields 54% of the brain match with no training. Fully trained scores .42. 100,000 supervised updates give 50% of that, and 1,000,000 give 76%. Scored by linear regression — [Geiger](https://www.biorxiv.org/content/10.1101/2020.06.08.140111.full.pdf)
- **Storrs et al. 2021 (JoCN) [FT]:** trained models account for 57% more of the explainable variance than untrained in human IT. They replaced weights with Gaussians matching each trained layer's mean and SD. They also cite one study reporting untrained above trained (Truzzi & Cusack 2020) — [Storrs 2021](https://kriegeskortelab.zuckermaninstitute.columbia.edu/sites/default/files/content/StorrsKriegeskorte_2021_JOCN.pdf)
- **Cichy et al. 2016 (Sci Rep) [FT]:** the untrained DNN correlates significantly with brain representations. It beats the object DNN in V1 but is worse in IT and IPS1&2, and is significantly worse than the object DNN after 166 ms — [Cichy 2016](https://www.nature.com/articles/srep27755)
- **Baek et al. 2021 (Nat Commun) [FT]:** face-selective units arise in untrained AlexNet (Gaussian init): Conv1 0.008%, Conv2 0.047%, Conv3 0.491%, Conv4 0.534%, Conv5 0.579%. Units selective to non-face classes arise too, so this is not face-specific — [Baek 2021](https://www.nature.com/articles/s41467-021-27606-9)
- **Saxe et al. 2011 (ICML) [FT]:** certain convolutional pooling architectures are frequency selective and translation invariant even with random weights. They cite Jarrett 2009: random filters 53% vs trained 54.2% on Caltech101 — [Saxe 2011](https://icml.cc/2011/papers/551_icmlpaper.pdf)
- Also in [Saxe 2011](https://icml.cc/2011/papers/551_icmlpaper.pdf): on CIFAR-10 the top trained architecture got 59.5 ± 0.3% vs 53.2 ± 0.3% for random. Random performance correlates robustly with trained performance. The filter distribution did not matter as long as it was centred at zero. Input-only baseline: an SVM on raw pixels gives 72.6% (NORB-mono) and 28.2% (CIFAR-10-mono).
- **Frankle et al. 2021 (ICLR) [FT, abstract and intro]:** training only BatchNorm affine parameters on frozen random features gives 82% on CIFAR-10 and 32% top-5 on ImageNet — [Frankle 2021](https://arxiv.org/abs/2003.00152). This is not untrained: the rescaling is learned.
- **Ulyanov et al. (Deep Image Prior) [FT, abstract and intro]:** generator structure alone captures much low-level image statistics before any learning — [Ulyanov](https://arxiv.org/abs/1711.10925). This is per-image fitting, not a representation comparison.
- **Elmoznino & Bonner 2024 (PLOS CB) [FT]:** effective dimensionality relates to encoding performance, including within untrained models — [Elmoznino 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10805290/). **In tension with** [Conwell 2024](https://www.nature.com/articles/s41467-024-53147-y): ED shows no correlation with veRSA among trained models (rSpearman = −0.063), and a non-significant slightly negative one among random layers for OTC (−0.142).

### Inferences
- Dropping the fitted readout removes much of the random-network advantage (Conwell's gap is 0.30 for cRSA vs 0.56 for veRSA, though these are different scales; see also Kazemian's RSA point). Our mutual kNN is unfitted, so fitted-regression figures such as 54% or ~70% should not be used as expected values for us.
- Vision's random-architecture alignment comes from a known bias (locality, pooling). Mean-pooled random transformers have no comparable lexical prior beyond the input tokens, so the vision analogy gives little reason to expect a transformer architecture effect. Kazemian's weak random-ViT result points the same way, though it concerns images.

### Gaps
- **Geirhos:** the brief names no specific paper. I found no Geirhos result on untrained networks. The search returned [Geirhos et al. 2020, error consistency](https://arxiv.org/abs/2006.16736) [ABS], which on the abstract does not address random nets.
- Absolute untrained scores in Conwell, Storrs and Kazemian are in figures or supplements, not extracted.

## Q3. Representational similarity with random networks: is shared input known to inflate similarity?

### Takeaway
Yes. Cui et al. 2022 show this directly and propose a fix. Biased CKA can also be high for independent random matrices. The classic papers (Raghu, Morcos, Ding) mostly compare *trained* networks from different seeds, so "random-init comparison" there does not mean untrained, and only Kornblith includes truly untrained nets.

### Cited Findings
- **Cui et al. 2022 [FT]:** RSA and CKA are confounded by the similarity structure of the inputs, which produces spuriously high similarity even for completely random networks — [Cui 2022](https://arxiv.org/abs/2202.00095)
- First-layer ResNet example: CKA of random nets is higher than pretrained vs finetuned (0.99 vs 0.95). dCKA reverses it (0.43 vs 0.72). Caveats: n = 20 CIFAR-10 samples, and the "random" nets are a pretrained ResNet-18 plus N(0,1) noise on each parameter, not fresh inits — [Cui 2022](https://arxiv.org/abs/2202.00095)
- Method: regress vec(K0), the input similarity, out of each layer's vec(K) with a linear model, then compare the residuals. Detection example at block 4: dCKA finds all correct pairs, while CKA finds 0 or 0.28 depending on the random-pair type. After block 7 no method identifies them — [Cui 2022](https://arxiv.org/abs/2202.00095)
- Stated limit: input similarity may not be additively separable in deep layers, so the benefit is larger for shallow layers. They suggest regressing out the previous layer instead — [Cui 2022](https://arxiv.org/abs/2202.00095)
- **Kornblith et al. 2019 [FT]:** models trained on CIFAR-10 and CIFAR-100 develop similar representations, and similarity with untrained networks is much lower. Appendix F.3 shows untrained-vs-untrained heatmaps; values are figure-only — [Kornblith 2019](https://arxiv.org/abs/1905.00414)
- **Murphy et al. 2024 [FT]:** in the low-data, high-dimension regime biased CKA is high even for pairs of random matrices and can be driven to its maximum. Shuffling real data barely changes it. They recommend random *and* shuffled controls — [Murphy 2024](https://arxiv.org/abs/2405.01012)
- **Davari et al. 2022/23 [FT]:** CKA is sensitive to outliers and high-variance directions and can be manipulated to arbitrary values — [Davari](https://arxiv.org/abs/2210.16156)
- **Ding et al. 2021 [FT]:** CKA fails to detect changes in all but the top 10 principal components. CCA is not specific: random-init noise overwhelms differences between far-apart layers — [Ding 2021](https://arxiv.org/abs/2108.01661)
- **Klabunde et al. survey [FT]:** defines k-NN Jaccard and second-order cosine measures, and notes that confounding effects exist for representational measures, citing Cui — [Klabunde](https://arxiv.org/abs/2305.06329)
- **Raghu et al. 2017 [FT]:** networks converge bottom-up during training; no untrained baseline in my extraction — [Raghu 2017](https://arxiv.org/abs/1706.05806). **Morcos et al. 2018 [FT]:** the comparison is generalising vs memorising trained nets; zero text hits for "untrained" — [Morcos 2018](https://arxiv.org/abs/1806.05759)

### Inferences
- All this literature assumes a shared X fed to both models. See Q6: our setting has no shared X, so inflation by shared input cannot be transferred as stated.
- Neighbourhood measures have not been studied for this confound in anything I read. dCKA is defined for kernels and has no kNN version.

### Gaps
- I found no paper measuring input-kernel confounding for mutual kNN specifically.
- Ding, Davari and Raghu do not provide untrained-vs-untrained values in my extraction; "untrained" gets 0, 1 and 0 text hits respectively.

## Q4. Random propagation: does a random net preserve the input kernel, and how does similarity decay with depth?

### Takeaway
In the infinite-width fully connected limit, a layer's kernel is a deterministic function of the previous layer's kernel, so a random net adds nothing except a fixed warping of the input kernel. Depth then pulls that kernel toward a fixed point, at a rate set by the phase. Small input angles are distorted most, which predicts that local (kNN) structure degrades first. Residual connections slow this, and transformer-specific theory exists but does not cover our exact readout.

### Cited Findings
- **Lee et al. 2018 [FT]:** Kˡ(x,x′) = σ_b² + σ_w² F(Kˡ⁻¹(x,x′), Kˡ⁻¹(x,x), Kˡ⁻¹(x′,x′)), where F is deterministic and depends only on the nonlinearity — [Lee 2018](https://arxiv.org/abs/1711.00165)
- **Poole et al. 2016 [FT]:** a correlation map governs pairs of inputs. If its slope χ₁ at 1 is below 1, nearby points converge; if above, they separate. In the ordered phase the manifold contracts, and in the chaotic phase it de-correlates. Lengths reach their fixed point fast, often within 4 layers — [Poole 2016](https://arxiv.org/abs/1606.05340)
- **Schoenholz et al. 2017 [FT]:** the correlation depth scale is ξ_c⁻¹ = −log[σ_w² ∫Dz₁Dz₂ φ′(u₁)φ′(u₂)] and diverges at the order-chaos boundary, which bounds how deep correlated signal propagates. Dropout destroys the critical point — [Schoenholz 2017](https://arxiv.org/abs/1611.01232)
- **Giryes et al. 2016 [FT]:** random Gaussian nets perform a distance-preserving embedding in the Gromov-Hausdorff sense on low-dimensional data, but smaller input angles shrink more, and the shrinkage grows with depth. Assumes semi-truncated linear activations and non-attention layers — [Giryes 2016](https://arxiv.org/abs/1504.08291)
- **Yang & Schoenholz 2017 [FT]:** skip connections give subexponential, often polynomial, dynamics, which preserve input-space geometry better than feedforward nets — [Yang & Schoenholz 2017](https://arxiv.org/abs/1712.08969)
- **Cowsik et al. 2024 [FT]:** random transformers have an order-chaos transition: tokens collapse to a line (ordered) or converge to a regular n-simplex (chaotic). Evolution of the n×n dot-product matrix is derived, with an angle Lyapunov exponent. Model is single-head with tanh MLP, permutation-symmetric ansatz — [Cowsik 2024](https://arxiv.org/abs/2403.02579)
- **Noci et al. 2023 [FT]:** rank collapse: in softmax attention, token representations tend to align perfectly at large depth. Pre-LN is shown as a comparison curve — [Noci 2023](https://arxiv.org/abs/2306.17759)
- **Hron et al. 2020 [FT]:** under the standard d^(−1/2) attention scaling, output columns stay statistically dependent even at infinite width, so the simple GP argument needs modification. They treat d^(−1/2) and d^(−1) scalings, positional encodings and layer norm — [Hron 2020](https://arxiv.org/abs/2006.10540)

### Inferences
- **Random depth is a warped, noisy copy of layer 0, plausibly with no new content.** Under Lee's recursion, infinite-width K^L = F^L(K⁰). If that approximately holds, a random transformer can only lose or reshape caption-kernel information relative to layer 0.
- **That would explain 0.10 → 0.05 without any architecture contribution.** Mutual 10-NN depends on the smallest distances, and Giryes says those are the most distorted. CKA, dominated by large variance directions (Ding), should fall less. This is my prediction and is untested here.
- Across-input theory (Poole, Schoenholz, Lee) and within-sequence token theory (Cowsik, Noci) are different objects. Cowsik's collapse is among tokens in a sequence, while our kernel is among captions after mean-pooling, so applying it to us is an extrapolation.
- Mean-pooling may add its own channel: Feghhi's GELU positive-mean mechanism would behave differently under mean rather than sum pooling, and caption length could still enter through the variance of the mean (∝ 1/length).

### Gaps
- No paper I found derives or measures sample-kernel retention by depth for a Pre-LN, mean-pooled, GPT-NeoX-style random LM. It can be measured directly at low cost.
- Where default Hugging Face / Pythia init sits relative to the Cowsik phase boundary: not established. Cowsik's model differs in activation and head count.
- Finite-width corrections were not reviewed, and the NNGP guarantee is infinite-width only.

## Q5. Input-kernel controls: how do others baseline against pixels, bag of words, or deconfounded measures?

### Takeaway
There are four usable families: a raw-input model as a competitor, covariate regression on the similarity matrix, variance partitioning with simple features, and input perturbation. Cui's regression and Feghhi's partitioning are the most directly reusable. Each answers a different question, and none was built for cross-modal comparison.

### Cited Findings
- **Raw-input baseline, [Khaligh-Razavi & Kriegeskorte 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4222664/) [FT]:** tests a pixel-vector model (Stimulus image, Lab) and untrained models by RDM correlation with brain ROIs, with stimulus-label randomisation tests. From Table 1, reading columns as mIT, hIT, LOC, FFA, PPA, EVC (my alignment from flattened table text; verify):
  - Stimulus image (Lab): 0.044, 0.023 (ns), 0.044, 0.083, −0.033 (ns), −0.119 (ns).
  - Lab joint histogram: 0.147, 0.092, 0.081, 0.094, −0.055 (ns), 0.011 (ns).
  - Untrained convolutional network: 0.132, 0.111, 0.058, 0.083, −0.023 (ns), 0.174.
- **Fixed vs mixed RSA, [Khaligh-Razavi et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5341758/) [FT]:** fixed RSA tests the feature space as is. Mixed RSA first fits a linear remix on a separate training set. Fixed RSA can miss features that would match after linear recombination. This is the formal version of the fitted vs unfitted distinction.
- **Deconfounding, [Cui 2022](https://arxiv.org/abs/2202.00095) [FT]:** linear regression of K⁰ from each representation's K, then HSIC on the residuals (see Q3). They report a linear model suffices especially for shallow layers, per log R² and BIC on ResNet-18 / CIFAR-10.
- **Variance partitioning, [Feghhi 2024](https://arxiv.org/abs/2406.01538) [FT]:** banded ridge on all feature-space subsets, with percentage of LLM predictivity accounted for (their Eq. 3). Feature spaces: sentence position, sentence length, static word embedding, sense, syntax, word position. Includes a correction for regressions that underuse small feature spaces.
- **Random embedding baseline:** [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/), [Pasquiou 2022](https://arxiv.org/abs/2207.03380).
- **Input perturbation:** word-order scrambling and word removal in [Kauf 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10187317/) [ABS]; fixed pixel permutation in [Kazemian](https://www.biorxiv.org/content/10.1101/2024.05.10.593623.full.pdf).
- **Ablation:** component removal in a single untrained block, [AlKhamissi 2025](https://arxiv.org/abs/2503.01830).
- **Pairing control:** on DINOv2 × All-Roberta-large-v1 over 5k COCO image-caption pairs, CKA falls with shuffling: 0.72, 0.46, 0.27, 0.13, 0.04, 0.01 at 0, 20, 40, 60, 80, 100% shuffled — [Maniparambil 2024](https://arxiv.org/abs/2401.05224) [FT]. This tests pairing only; it says nothing about input vs architecture, and the encoders are trained.

### Inferences
- Shuffling (Maniparambil, Murphy) calibrates chance, not attribution. A lexical alignment survives every shuffle-based null because it is real pairing structure. Calibrated excess therefore cannot separate input from architecture. This applies to our calibrated numbers.

### Gaps
- Kriegeskorte's original 2008 RSA paper was not read. I have no sourced statement for a pixel-model convention there, or for partial-correlation RSA as a method.
- Search found no cross-modal paper using a bag-of-words or random-text-encoder control. That is a null from one targeted query plus adjacent searches, not proof of absence.

## Q6. What differs in our setting, and how to separate input from architecture there

### Takeaway
No cited study has our design, in which the two models never see the same input and only one side is random. The confound literature therefore does not transfer directly. For us the real question is how much of the alignment is carried by the caption's token-count kernel, and that can be measured without any network. Layer-0 alignment should also not be called spurious: it is probably real content agreement, attributable to the caption lexicon and not to the LM.

### Cited Findings
Differences from the literature, each grounded in a source above:
- **Shared input:** Cui's derivation uses one X for both networks — [Cui 2022](https://arxiv.org/abs/2202.00095). Klabunde's framework likewise assumes all models get the same inputs — [Klabunde](https://arxiv.org/abs/2305.06329). Brain studies also present the same stimulus to model and subject. Ours pairs a caption with an image.
- **Random on both sides:** Cui's and Kornblith's untrained comparisons are random vs random or random vs trained in the *same* modality — [Cui 2022](https://arxiv.org/abs/2202.00095), [Kornblith 2019](https://arxiv.org/abs/1905.00414). Ours is random LM vs trained ViT, and the trained side already holds learned semantics.
- **Fitted vs unfitted:** most high untrained scores use trained linear readouts — [Schrimpf 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8694052/), [Geiger](https://www.biorxiv.org/content/10.1101/2020.06.08.140111.full.pdf), [Kazemian](https://www.biorxiv.org/content/10.1101/2024.05.10.593623.full.pdf). Mutual kNN fits nothing.
- **Temporal and positional artifacts:** autocorrelation, passage position and word rate — [Feghhi 2024](https://arxiv.org/abs/2406.01538) — arise from experimental stimulus sequences. They have no direct counterpart in an i.i.d. image-caption sample, though caption length may.
- **Sample size:** Cui's headline example uses n = 20 — [Cui 2022](https://arxiv.org/abs/2202.00095). Hosseini's benchmark has a few hundred sentences (batches of 76-78 and 48-49) — [Hosseini 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025646/). Ours is n = 10240.
- **Project-internal, not literature:** a random-init ViT vs pretrained ViT gives mKNN 0.16-0.23 and CKA 0.50-0.66, attributed to pixel statistics (`experiments/next-steps-2026-11-04.md`, F6). So the same-modality inflation exists in our own data.

### Inferences
All of the following are my proposals, none tested.

**Why the confound differs**
- With separate inputs, a model-independent common cause must flow through both K⁰_text and K⁰_image. A caption is a function of its image, so the shared factor is image content expressed in words.
- Cui's single-K⁰ regression needs two covariates here, and regressing out K⁰_text would also remove real semantic signal, because word overlap *is* content. Deconfounded alignment should not be read as true alignment.

**Tests, cheapest first**
1. **Analytic layer-0 kernel, no network.** Compute a token-count kernel from captions (cᵢ·cⱼ/(nᵢnⱼ)), plus TF-IDF and length-normalised variants, and score mutual 10-NN and CKA against the ViTs. Under a random Gaussian embedding table this is the expected kernel of mean-pooled layer 0, up to the projection's finite-dimension noise. If it reaches ~0.10 or above, then the embedding table's specific values and the LM architecture contribute nothing at layer 0.
2. **Embedding-dimension sweep.** Vary the random table's dimension. Alignment should converge toward the analytic kernel's value, as a random projection should.
3. **Vocabulary-permutation null.** Apply a random bijection to token IDs before the random embedding lookup. Token counts are preserved, so layer 0 should be statistically unchanged. This rules out any contribution from which token is which.
4. **Overlap-destroying null.** Give each token occurrence, or each caption's tokens, a fresh ID so that no two captions share tokens, while preserving length. Layer 0 should fall to null. Any remaining excess at any layer is a length or structure channel, not lexical. Partial out caption length separately, following Feghhi's approach.
5. **Factorial embedding × blocks.** Cross embedding-table seed with block-weight seed, and random vs pretrained embeddings with random vs pretrained blocks. Hosseini's two untrained inits are a precedent for checking init dependence.
6. **Depth retention curve.** At each random layer l, measure agreement between the layer's kernel and K⁰ (mKNN(l, token-count) and CKA) alongside alignment with vision.
   - If the depth decline in vision alignment is tracked entirely by retention of K⁰, then the 0.10 → 0.05 drop is signal loss and random architecture adds nothing.
   - Compare local vs global: by Giryes, small k should fall faster than CKA.
7. **Residual analysis at deep layers.** Remove K⁰_text from the deep random kernel (Cui's regression, or the previous-layer variant for deep layers), compute kNN on the residual kernel, and ask whether any excess over null remains. Because kNN on a residual kernel is non-standard, validate it on synthetic data first (exp-016 style).
8. **Vision-side mirror.** Run K⁰_image (pixels or a low-level kernel) against K⁰_text. If token-count vs pixel alignment is near null while token-count vs trained ViT is high, the trained ViT's learned semantics are doing the matching, not a low-level input coincidence.

**How to read outcomes**
- High token-count alignment plus no excess at depth means no evidence for an architecture effect. Report it as a lexical baseline at step 0, and use that baseline as the floor when reading trained-LM alignment.
- Excess at depth that survives overlap destruction and length removal would be a real architecture signal and would be new relative to this literature.

Check before relying on test 3 or 4: confirm that the Pythia / GPT-NeoX embedding layer has no absolute positional component, so that layer 0 really is order-invariant. I did not verify this in this research; rotary embeddings are my recollection only.

### Gaps
- I found no published study of an untrained text encoder against a trained vision model on paired data. The exp-018 README's prior-work list also mentions none. Balestriero and LeCun (per that README) track cross-modal mutual 10-NN from chance during training of small text encoders; I did not read it, and it is outside what I verified.
- Whether the token-count kernel alone reaches the observed 0.10 is an empirical question and could not be answered from the literature.
- Caption repetition and templating in the Gemini captions may inflate token overlap. This depends on our data, not covered here.
