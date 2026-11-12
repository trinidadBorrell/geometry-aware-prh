# Where the project stands and what to do next

2026-11-04 · synthesis of exp-003 to exp-016, the 2026 literature, and a proposed direction.
Draft for discussion; not yet part of the proposal.

## 1. What our experiments established

| # | finding | evidence | status against the literature |
|---|---|---|---|
| F1 | **Uncalibrated scores mislead, and fixed-k mutual kNN measures a different scale at every n.** CKNNA's intermediate-k peak is chance; mask-first CKNNA variants exceed 1. mKNN at k = 10 falls to 0.41x from n = 1024 to 10240 while calibrated CKA is stable | exp-003, 005-008, 006 | Known: Gröger et al. (calibration), Koepke et al. (n). The CKNNA-variant bugs are a small new point |
| F2 | **Geometry is not identifiable from random samples at ID 20-40.** kNN-graph geodesics are off by 50-150%; graph ORC is set by ID, k and density; the moment estimator is only a one-sided flatness test; PersonaManifold's pipeline finds "curvature" on flat data. ID is the one usable invariant | exp-011, 013, 014 | **New.** Nobody else validates the geometric estimators they use. Directly relevant to You et al.'s "Riemannian metric approximation" and to PersonaManifold |
| F3 | **ID mismatch tracks misalignment beyond depth** (partial rho -0.22 to -0.36, 93-95% of confound-free pairs); cross-modal alignment peaks at ID-matched layers; pooling, not modality, explains the opposite LLM / ViT ID profiles | exp-010, 012, 015 | New, correlational |
| F4 | **Per-sample local structure is shared within a modality and refines neighbourhood agreement.** ORC agreement given density 0.57-0.68 within modality (null ~0.05); ~90% survives removing neighbour overlap, but its layer map repeats mutual kNN (0.85-0.96) and it vanishes where neighbourhoods share nothing | exp-014, 015 | New measure; it sits in the "relational structure converges" camp (Gröger; You et al.) |
| F5 | **Shared content bounds cross-modal alignment.** Gemini captions raise all 16 LLM x ViT pairs (mKNN 0.12 -> 0.34, CKA 0.25 -> 0.58, ORC 0.15 -> 0.38), to the dual-encoder level; dual encoders on WIT captions reach only 0.22 | exp-016 B3, A | Headline known (Koepke's recaption, Herbst et al.). The matched dual-encoder comparison is ours |
| F6 | **The ceilings are not where people put them.** Same LLM on the original vs the Gemini caption of one image: 0.16 (below LLM x ViT with Gemini, 0.34). A random-init ViT against a pretrained ViT: mKNN 0.16-0.23, CKA 0.50-0.66 from pixel statistics alone. Image x augmented image: 0.57 | exp-016 B1, B2, D | **New.** The within-modality band contains non-semantic shared content, so it overstates the attainable cross-modal level |
| F7 | **Local measures are fragile to private variance; CKA is not.** Identical content through two random encoders: mKNN ~0.8. With 36% private variance, ORC agreement is at null; with 50%, mKNN 0.11 while CKA is still 0.49 | exp-016 arm C | **New** as a quantified statement |

### What this means for Platonic vs Aristotelian

1. **H1 ("wrong ruler") is not refuted. It cannot be tested with random samples.** At ID 20-40 and n <= 100k, no estimator recovers curvature or geodesics (F2). Every ruler we could validate (Euclidean, cosine, whitened, rank, ORC) gives the same picture. That is a statement about identifiability, not about the manifolds.
2. **Your data do not single out local structure as what converges.** Once the inputs share content (F5), CKA rises as much as mKNN does, relative to each measure's within-modality level: about 0.58 / 0.78 for CKA and 0.34 / 0.50 for mKNN, roughly 0.7 of the band for both. Gröger et al.'s local-vs-global distinction is about the **trend with capability after calibration**, which we have not tested. What our evidence supports is narrower: alignment at every scale is bounded by shared content, and local measures fall faster when private variance is added (F7).
3. **The Aristotelian version worth defending is the refined one: relations converge, metric geometry does not** (You et al., Sept 2026). ORC and density agreement (F4) are relational statistics, and their behaviour fits it.
4. **The open question is Revision 2's question:** is the fine-scale disagreement *informational* (one modality lacks the content) or *organisational* (both have it and arrange it differently)? Koepke et al. assume the latter ("Umwelten"). F6 and F7 show that passive data cannot tell the two apart: private variance alone reproduces the "organisational" pattern.

## 2. The literature since Revision 2

| paper | claim | consequence for us |
|---|---|---|
| Gröger, Wen, Brbić, ICML 2026 (2602.14486) | After calibration the global trend disappears; local neighbourhoods stay aligned | Reproduced (exp-003). Their claim is about trends; ours so far are about levels |
| Koepke, Zverev, Ginosar, Efros (2604.18572) | Fixed-k mKNN collapses from 0.135 (WIT-1024) to 0.008 (LAION-15M), but **is stable at k = n/100**. 500-word recaptions raise scores, saturating at 100-250 words. Cross-modal RSA peaks at the top ~3 eigenmodes. Interpreted as distinct "Umwelten" | **E4 is answered:** the collapse with n is a scale effect. Their top-j eigenmode analysis is a crude shared subspace |
| Herbst, Roschmann, Bouniot, Akata, NeurIPS 2026 SymGeo workshop | Unimodal cross-modal alignment matches or beats VLMs but stays below same-modality; the capability trend holds with clean, informative captions | Confirms F5. Still workshop-level, so the trend question is not closed |
| You, Jang, Mo, Jung (2609.27252, Sept 2026) | 2x2 design: relational structure vs metric geometry, at local and global scale. Relational structure converges at **both** scales (H0-skeleton overlap); metric agreement is weak and flattens the trend; same under a "Riemannian metric approximation" | Closest to your current leaning, and it is already written. Check their Riemannian approximation against exp-011 / 013 |
| Displacement Geometry (2609.24209, NeurIPS 2026) | 44 models: cross-model cosines of difference vectors (directions) beat pointwise agreement (V-L 0.31 vs 0.19 at sample level, 0.49 vs 0.25 at concept level); shared vs private subspace from an agreement matrix | **E3 is partly done.** They use passive differences between random samples and concept centroids, not controlled edits |
| Balestriero, LeCun (2610.39302, Oct 2026) | Theorem: JEPAs trained on "good views" (two observations of one scene) learn the same representation up to rotation; next-token prediction is a JEPA with bad views. They note LLaVA-generated captions leak CLIP information | A theory for F5 and F6: alignment is bounded by view quality. B1 = 0.16 says the original and the Gemini caption are poor views of each other. The same leakage concern applies to our Gemini captions |
| Bangachev, Bresler, Polyanskiy (2605.28870) | Alignment = signal (linear, sparse object-attribute features) + bias + noise; frequent words align better | Supports a factor-level view of alignment |
| Seeing Through Words (EMNLP 2025, 2509.20751) | Many-to-many matches; averaging several captions improves alignment | Supports the paraphrase-ceiling design |
| Schnaus et al., CVPR 2025; vec2vec (2505.12540) | Vision-language matching without pairs works for coarse class sets; unpaired translation works between text models, weakly for CLIP | Option Y1 |

**What nobody has done (from these searches):**

- Every study measures **passive i.i.d. pairs**. None holds the information content fixed and varies one thing at a time.
- None measures geometry **where it is identifiable**.
- None checks convergence **toward known world geometry** factor by factor (the PRH's convergent point Z, made concrete).

The PRH paper's colour experiment is the only precedent for the last point, and it uses a single factor.

## 3. Proposed direction: hold the world fixed

**Idea.** exp-011 and exp-013 show that geometry cannot be estimated from random samples at ID ~30. Along a **designed one- or two-parameter family of inputs** (one factor varied, everything else fixed), the representation instead traces a low-dimensional curve. Its distance profile, length and curvature can be measured from a few dozen points. This is how perceptual straightening is measured (Hénaff et al., 2019; Hosseini and Fedorenko, 2023 for LLMs). If we vary a factor that the caption states and the image shows, the two modalities share that information by construction. Any remaining difference is then organisational. This turns the original "wrong ruler" question and the Revision 2 "informational vs organisational" question into measurements.

**Working model**, stated so that it can fail:

```
x_m  ≈  Σ_f  g_{m,f} · φ_{m,f}(z_f)  +  private_m  +  noise
```

Here z_f are world factors (category, colour, count, position, size, relation, lighting, ...), φ_{m,f} is model m's code for factor f, and g_{m,f} is its gain. Two models can differ in three ways:

- **Informational:** g = 0 for a factor in one model. For example, a caption never mentions the lighting.
- **Exchange rate:** both models encode the factor with the same shape, but their gains differ.
- **Shape:** φ differs, for example a linear vs a log number line, or a hue circle vs a hue line.

Gain differences preserve the order within a factor but change distances, and they reorder fine neighbourhoods wherever several low-gain factors compete. If this is the main mechanism, it would explain several 2026 results at once: relational > metric (You et al.), coarse shared but fine not (Koepke et al.), directions > positions (Displacement Geometry). It would also say that the "Umwelten" differ mainly in exchange rates, and exchange rates can be undone.

### X0: synthetic signatures (week 1, local CPU, free)

- Extend arm C (`dose_response.py`) with gain knobs and shape knobs (monotone warps per factor), alongside the existing private-variance knob.
- For each knob, measure:
  - calibrated mKNN at fixed k and at k = n/100, over n = 1k to 50k;
  - CKA;
  - ORC agreement;
  - a relational vs metric pair (rank-based vs distance-based);
  - displacement cosines.
- Question: does each mechanism leave a distinct signature on passive data?
  - **If not:** that result alone justifies X1, because passive data cannot separate informational from organisational.
  - **If yes:** apply the signatures to the cached exp-016 features (n = 10240, original vs Gemini) right away.

### X1: controlled stimulus families (core, GPU for extraction, ~1 day)

**Factors** with a known world geometry:

- hue (circle; CIELAB distances);
- count 1-12 (numerosity: linear vs log);
- 2-D position (grid);
- object size / distance;
- object category (WordNet distances);
- spatial relation.

**Nuisances**, one modality at a time:

- visual-only: lighting, colour temperature, crop, JPEG, viewpoint;
- text-only: paraphrase, synonyms, word order.

**Stimuli.** Prefer deterministic edits of real photos, because a generator's text encoder leaks information:

- COCO instance masks with hue rotated inside the mask;
- crop / scale for size and position;
- photometric changes for lighting.

Add real controlled datasets for relations and count: What'sUp, CountBench-style sets, CLEVR / Super-CLEVR, PUG:SPAR. Use generative editing only as a secondary source.

**Captions:**

- templated from the factor values;
- 5 LLM paraphrases each (the text ceiling);
- a "factor omitted" variant, which is the informational knob.

**Per factor and model:**

1. Variance share and decodability of the factor (ANOVA / dPCA-style marginalisation).
2. Gain.
3. Trajectory geometry: distance profile over the factor's levels, discrete curvature and straightness, compared with the world geometry (for example CIELAB vs a hue circle, log vs linear count).
4. Cross-model: trajectory RSA, gain ratio, and the angle between factor subspaces after a map fitted on *other* factors.
5. Interactions in 2x2 designs (colour x object, count x category): the parallelogram defect, i.e. does Δcolour depend on the object. This is a discrete holonomy, a curvature-like quantity that the stimulus design makes measurable. Compare it across modalities.

**Nulls and ceilings:** shuffled factor levels; random-init models; paraphrase and augmentation ceilings; max-over-layers nulls as in exp-016.

**Readout:** each factor ends with one of three verdicts: shared (same shape, with a gain ratio), informational (not encoded by one model) or organisational (different shape). That gives a table no one has.

### X2: close the loop on natural data

Use COCO (5 captions, object annotations) and DOCCI (human dense captions) with factor annotations:

- **(i) Prediction.** Predict per-sample neighbour overlap from factor-based predictors: which factors does a sample's neighbourhood vary in?
- **(ii) Gain equalisation, the decisive test.** Estimate factor subspaces and gains from the X1 stimuli only, with no natural pairs. Rescale one model's factor subspaces to the other's, then measure held-out natural mKNN (fixed k and k/n), CKA and ORC.
  - **If equalisation from stimuli closes much of the gap:** the gap is an exchange rate, organisational but trivially invertible. That is a sharp answer to Koepke et al.
  - **If the gap stays:** read the residual against the paraphrase / view ceilings. It is informational or shape-level.
  - **Baselines:**
    - CCA fitted on natural pairs (upper bound);
    - random subspaces of equal dimension;
    - shuffled factor labels.

### X3: per-factor convergence toward the world, over scale and training

Run X1 on model series:

- LLMs: Qwen2.5 0.5-14B, plus Pythia 410M-6.9B, whose public training checkpoints give a trend over training at no extra cost;
- ViTs: DINOv2 S-g, SigLIP 2 sizes.

Does each factor's trajectory approach (a) the world geometry and (b) the other modality as capability grows? This is the PRH trend tested where it is cleanly defined, with Gröger-style calibration per factor. The prediction is that different factors converge at different rates, which would turn "is there a Platonic representation?" into "which parts of the world converge, and how fast?"

### Options, if X0 or X1 disappoint

- **Y1. Blind identifiability.** Can the pairing be recovered *without* pairs? Use entropic GW / QAP on the Gemini-caption set at the cluster level and at the sample level, with LLM x LLM as the control. This is the operational test of the strong Platonic form; local agreement at the dual-encoder level may still not identify the pairing. CPU; builds on Schnaus et al. and vec2vec.
- **Y2. A model organism.** Build a closed synthetic world: CLEVR-like scenes with programmatic captions and an informativeness knob. Train a small MAE ViT and a small next-token LM separately, over several seeds and sizes. Measure convergence to the known scene-graph geometry. This removes the shared-web-data confound and tests Balestriero and LeCun's view prediction outside JEPAs. GPU days; a stretch goal.

### What to keep from Revision 2

| | decision |
|---|---|
| E1 | merged into X0 |
| E2 | **keep, minimal:** DOCCI human captions vs Gemini on the same images, and the captioner-leakage test. F5 rests on it, and Balestriero and LeCun raise the same leakage concern |
| E3 | drop the full version (partly done by Displacement Geometry); X2's gain equalisation is the sharper form |
| E4 | drop: Koepke et al. show stability at k = n/100; reproduce it in X0 at no cost |
| E5 | folded into X1 (paraphrase and view ceilings by design) |
| E6 | keep as a small control (Long-CLIP) |
| E7 | folded into X2 (i) |
| E8 | becomes X3 |

## 4. Paper

**Working title:** *Holding the world fixed: what converges between vision and language models, factor by factor.*

1. Measuring geometry from random samples is not identifiable at representation ID (F2; short section plus appendix). This motivates designed stimuli.
2. Passive data cannot separate informational from organisational sources of misalignment (F6, F7, X0).
3. With information held fixed, the factor table: shared, gain-shifted, shape-different (X1).
4. Gain equalisation estimated from stimuli predicts natural-data alignment (X2).
5. Per-factor convergence toward world geometry with scale and training (X3).

It reconciles the 2026 results under one mechanism, or shows where they cannot be reconciled. Both outcomes are publishable, because the factor table is new either way.

A second, short **workshop paper** (SymGeo / NeurReps style) on estimator identifiability: exp-011, 013 and 014, the PersonaManifold replication, and a check of the "Riemannian approximation" in You et al.

## 5. Timeline (~12 weeks to ICML 2027)

| weeks | work | gate |
|---|---|---|
| 1 | X0 synthetic signatures; apply them to cached exp-016 features | Do mechanisms leave distinct passive signatures? |
| 1-3 | Build stimulus families (deterministic edits first) and captions; E2 extraction in the same GPU session | Pilot: 2 factors x 4 models show a measurable trajectory above the shuffled-level null |
| 3-5 | X1 extraction and per-factor analysis | Factor table for ~15 models |
| 5-7 | X2 gain equalisation on COCO / DOCCI | Does equalisation move held-out natural alignment? |
| 7-9 | X3 model series (Pythia checkpoints, Qwen, DINOv2 sizes) | |
| 9-12 | Robustness, writing, buffer | |

**Compute.** X0 and the analyses run on CPU. Stimulus extraction is a few GPU hours per model set on an RTX PRO 4000 class GPU (~$0.6/h), one pod at a time, with self-termination as in exp-016.

## 6. Risks

- **Designed stimuli are out of distribution.** Deterministic edits of real photos keep them close, and X2 is the bridge back to natural data.
- **Gain equalisation could create agreement.** Fit on stimuli only, evaluate on held-out natural pairs, and compare with shuffled-factor and random-subspace nulls.
- **Factor codes may not be linear.** Trajectory geometry is measured without assuming linearity; the parallelogram defect measures the non-additivity directly.
- **Scooping.** This area moves monthly: three relevant papers appeared in September-October 2026. Read You et al. and Displacement Geometry in full before X1. Check whether either uses controlled variation in an appendix.

## Sources

- [Gröger, Wen, Brbić (ICML 2026): Revisiting the PRH: an Aristotelian view](https://arxiv.org/abs/2602.14486)
- [Koepke, Zverev, Ginosar, Efros (2026): Back into Plato's Cave](https://arxiv.org/abs/2604.18572)
- [Herbst, Roschmann, Bouniot, Akata (NeurIPS 2026 SymGeo workshop): Where does alignment emerge and what does it mean?](https://neurips.cc/virtual/2026/157279)
- [You, Jang, Mo, Jung (2026): What converges in the PRH? Structure over geometry](https://arxiv.org/abs/2609.27252)
- [Displacement Geometry Captures Platonic Shared Reality Across Models and Modalities (NeurIPS 2026)](https://arxiv.org/html/2609.24209v1)
- [Balestriero, LeCun (2026): Proving and improving the PRH with JEPAs](https://www.alphaxiv.org/abs/2610.39302)
- [Bangachev, Bresler, Polyanskiy (2026): Representation alignment rests on linear structure](https://arxiv.org/abs/2605.28870)
- [Seeing Through Words, Speaking Through Pixels (EMNLP 2025)](https://arxiv.org/abs/2509.20751)
- [Schnaus, Araslanov, Cremers (CVPR 2025): It's a (blind) match!](https://arxiv.org/abs/2503.24129)
- [Jha, Zhang, Shmatikov, Morris (2025): Harnessing the universal geometry of embeddings (vec2vec)](https://arxiv.org/abs/2505.12540)
- [Hosseini, Fedorenko (NeurIPS 2023): LLMs implicitly learn to straighten sentence trajectories](https://arxiv.org/abs/2311.04930)
- [COCO-Counterfactuals (2023)](https://arxiv.org/abs/2309.14356)
