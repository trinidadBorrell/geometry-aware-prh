# Implementation Plan: Lexical Floors in Vision-Language Representation Alignment

## 0. Purpose

Build a reproducible experimental pipeline to determine **where vision-language representational alignment emerges on the language side**:

1. image-caption pairing,
2. caption length and lexical structure,
3. random token embeddings,
4. trained token embeddings,
5. contextual transformer layers.

The system must support controlled comparisons across:

- multiple datasets,
- original and generated captions for the **same images**,
- multiple LLMs/tokenizers,
- multiple vision encoders,
- multiple representation-similarity metrics,
- deterministic resampling and uncertainty estimation.

The first milestone is a decisive WiT pilot. Do **not** launch the full 10-LLM x 17-ViT x 3-dataset sweep until the pilot passes the stop/go criteria in Section 16.

---

## 1. Scientific questions

### RQ1: Does an exact lexical alignment floor exist?

For captions tokenized by LLM tokenizer `t`, compare the exact token-frequency kernel with a vision-representation kernel:

\[
K_{TF}^{(t)} = P_{TF}^{(t)} (P_{TF}^{(t)})^T,
\qquad
K_V = YY^T.
\]

Primary quantity:

\[
CKA(P_{TF}^{(t)}, Y).
\]

### RQ2: Does random layer 0 reduce to a random projection of token frequencies?

For random embedding table `E`:

\[
X_{rand} = P_{TF}E.
\]

Test whether, as embedding width increases:

\[
K_{rand}=X_{rand}X_{rand}^T \rightarrow cK_{TF}.
\]

### RQ3: How much additional alignment enters through learned embeddings and contextual layers?

Compare:

```text
length
-> binary token presence
-> token frequency
-> TF-IDF
-> random layer 0
-> trained layer 0
-> contextual layers 1...L
```

### RQ4: Does caption descriptiveness control measured alignment?

For identical images, compare:

```text
original short caption
vs generated short caption
vs generated medium caption
vs generated detailed caption
```

### RQ5: Does deep alignment survive lexical control?

Use cross-fitted ridge residualization to remove the component of LLM representations predictable from lexical features, then compare held-out residuals with vision representations.

---

## 2. Non-goals

The first implementation must **not**:

- claim that CKA differences are additive variance components;
- report `CKA_deep - CKA_lexical` as a “percentage explained”;
- select the best layer on the test set without reporting the complete trajectory;
- mix different image subsets across caption conditions;
- generate synthetic data when a caption/image/model output is missing;
- silently truncate captions differently across models;
- treat all model pairs as statistically independent replicates;
- launch the full model grid before validating the pilot.

---

## 8. Exact lexical features

For each tokenizer, create sparse matrices with shape `(n_captions, vocabulary_size)`.

### 8.1 Count matrix

\[
P_{count}[i,v] = \#\{t : token_{it}=v\}.
\]

### 8.2 Token-frequency matrix

Primary exact layer-0 lexical baseline:

\[
P_{TF}[i,v] = \frac{P_{count}[i,v]}{L_i},
\]

where `L_i` is the number of included, non-padding tokens under the specified special-token policy.

### 8.3 Binary presence matrix

\[
P_{binary}[i,v] = \mathbb{1}[P_{count}[i,v] > 0].
\]

### 8.4 TF-IDF matrix

Use a documented smoothed IDF convention, for example scikit-learn's:

\[
idf(v) = \log\frac{1+n}{1+df(v)}+1.
\]

Record whether rows are L2-normalized. Run both unnormalized and L2-normalized versions only if pilot results show normalization is consequential; otherwise designate one primary definition in the config.

### 8.5 Caption-length features

At minimum:

```python
Z_length = [L, log1p(L)]
```

Optionally add `L**2` after standardizing columns. Do not let a single raw scalar produce unstable comparisons without centering.

### 8.6 Exact lexical kernel

Do not materialize a dense `V x V` matrix.

For sparse `P`:

\[
K_{lex}=PP^T.
\]

For linear CKA, prefer feature-space computation when memory-efficient. For RSA or kernel inspection, construct `n x n` kernels in blocks.

---

## 9. Representation metrics

### 9.1 Linear CKA

Primary implementation:

\[
CKA(X,Y)=
\frac{\|Y^T H X\|_F^2}
{\|X^T H X\|_F\,\|Y^T H Y\|_F},
\quad
H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^T.
\]

Also implement equivalent centered-kernel CKA for testing:

\[
CKA(K,L)=
\frac{\langle HKH,HLH\rangle_F}
{\|HKH\|_F\|HLH\|_F}.
\]

Numerical requirements:

- compute accumulations in float64;
- return a finite scalar in `[0, 1 + tolerance]` for PSD linear kernels;
- fail with a clear error if centered norm is zero;
- clamp only tiny floating-point excursions, not genuine implementation errors.

### 9.2 Mutual k-NN overlap

Primary `k=10`; secondary `k in {5, 20}`.

Document the exact statistic. Recommended:

1. compute cosine-neighbor sets excluding self;
2. for each item, calculate intersection size divided by `k`;
3. average over items.

If “mutual” currently means mutual-neighbor graphs rather than neighborhood overlap, preserve the existing definition but rename it explicitly and test it.

### 9.3 Spearman RSA

- obtain upper-triangular pairwise cosine-distance vectors;
- calculate Spearman correlation;
- exclude diagonal;
- use blockwise computation for large `n`.

### 9.4 Permutation null

Permute image-caption correspondence, not feature dimensions.

Pilot default:

```text
1,000 permutations for final pilot figures
100 permutations for development smoke tests
```

Use the same permutations across compared conditions for paired uncertainty.

Report empirical p-value:

\[
p=\frac{1+\#\{s_{perm}\ge s_{obs}\}}{1+N_{perm}}.
\]

Statistical significance is secondary to effect size.

---

## 10. Required unit and property tests

### 10.1 Token matrix tests

- known toy captions produce exact expected count, TF, binary, and TF-IDF matrices;
- padding never contributes;
- empty captions fail validation or use an explicitly documented policy;
- repeated tokens affect count/TF but not binary features;
- row sums of TF equal 1 within tolerance.

### 10.2 CKA tests

- `CKA(X, X) == 1` within tolerance;
- invariance to isotropic scaling;
- invariance to orthogonal feature rotation;
- feature-space and kernel-space implementations agree within `1e-8` on float64 toy data;
- permuting rows of only one representation lowers alignment on structured toy data;
- jointly permuting rows of both representations leaves CKA unchanged.

### 10.3 Random-projection tests

For synthetic sparse `P`, show empirical convergence:

```text
CKA(P @ E_d, P) increases toward 1 as d increases, averaged over seeds.
```

Do not require strict monotonicity for every seed. Test monotonicity of seed-averaged values within a tolerance or compare low-width versus high-width endpoints.

### 10.4 Manifest tests

- mismatched image order must raise an exception;
- duplicate `(image_id, caption_variant_id)` must raise an exception;
- caption conditions with unequal image sets must fail paired comparison;
- artifact hashes must change if caption text or image ordering changes.

### 10.5 Residualization tests

- ridge model is fit only on training folds;
- each residual row is generated by a model that did not train on that row;
- a toy lexical-only representation yields near-zero residual signal out of sample;
- random targets are not spuriously removed beyond sampling noise.

---

## 11. Experiment 1: exact lexical reproduction

### Goal

Determine whether the current random layer-0 alignment is reproduced by exact caption-token structure.

### Data

- fixed WiT pilot image subset;
- existing Gemini detailed captions;
- identical ordered image IDs for every representation;
- begin with `n=1024` to match existing work, then add sample-size stability analysis.

### Models

At minimum:

- current Pythia tokenizer/model used in preliminary analysis;
- DINOv2 representation used in preliminary analysis;
- CLIP representation used in preliminary analysis.

### Conditions

```text
length
count
binary
TF
TF-IDF
random layer 0, seeds 0...4
trained layer 0
selected contextual layers
```

### Outputs

`results/experiment_01.parquet`, one row per:

```text
dataset x caption_variant x tokenizer x language_model x language_layer
x lexical_condition x vision_model x vision_layer x metric x seed x sample_seed
```

Required columns:

- estimate,
- bootstrap lower/upper CI,
- permutation-null mean/std/p-value,
- `n`,
- hashes for both representation artifacts,
- full config hash.

### Acceptance criteria

- exact TF CKA and random-layer-0 CKA are both computed on exactly the same samples;
- `CKA(random_L0, TF)` is reported directly;
- observed discrepancy is documented rather than forced to fit theory;
- at least five random embedding seeds are included.

---

## 12. Experiment 2: original versus descriptive captions

### Goal

Measure how caption source and descriptiveness affect lexical, embedding, and deep alignment while holding images fixed.

### Primary paired conditions

```text
wit_original
vs
gemini_detailed_existing
```

If controlled generations are available, add:

```text
generated_short
generated_medium
generated_detailed
```

### Controls

For every condition, record:

- mean/median token length,
- vocabulary size,
- unigram entropy,
- mean non-stopword fraction,
- duplicate-caption rate,
- truncation rate.

### Analysis

Plot for each vision model:

```text
x-axis: lexical -> random L0 -> trained L0 -> contextual layer
facet/color: caption condition
y-axis: CKA or neighborhood overlap
```

Do not compare caption conditions with different image sets.

### Acceptance criteria

- paired bootstrap over image IDs;
- original-vs-generated differences reported with confidence intervals;
- length-only baseline shown beside lexical baselines;
- report whether generated-caption gains appear at the exact lexical stage, trained embedding stage, contextual stage, or all three.

---

## 13. Experiment 3: random embedding width sweep

### Goal

Empirically validate the random-projection explanation.

### Widths

```text
32, 64, 128, 256, 512, 1024, 2048, 4096
```

### Seeds

```text
0, 1, 2, 3, 4
```

Increase to at least 20 seeds for the final convergence figure if variance is visibly large.

### Construction

For tokenizer vocabulary size `V` and width `d`, initialize `E in R^(V x d)` using:

1. the model's documented input-embedding initialization distribution as primary;
2. standard Gaussian with variance scaled by `1/d` as a theoretical sensitivity check.

Compute:

\[
X_d=P_{TF}E_d.
\]

Report:

- `CKA(X_d, P_TF)`,
- `CKA(X_d, Y_vision)`,
- local-neighborhood overlap between `X_d` and `P_TF`,
- seed mean and 95% interval.

### Diagnostics if convergence fails

Check, in this order:

1. exact token set included in `P_TF`;
2. special-token inclusion;
3. masked-mean denominator;
4. padding contamination;
5. positional embeddings accidentally included;
6. embedding scaling or layer normalization;
7. tied embedding assumptions;
8. numerical precision.

---

## 14. Experiment 4: trained layer-0 replication

### Goal

Test whether trained embeddings account for much of the best deep-layer alignment beyond one model/seed.

### Pilot scope

Use approximately:

- 3 language-model families,
- at least 2 sizes within one family,
- 4 vision encoders representing distinct objectives:
  - DINO-style self-supervision,
  - CLIP-style image-text contrastive learning,
  - supervised classification,
  - masked-image modeling if available.

### Required output

Complete language-layer trajectories, not only best-layer values.

Report:

- exact TF baseline,
- random layer 0,
- trained layer 0,
- every contextual layer or a clearly documented dense layer grid,
- best layer chosen only for descriptive summary after the complete trajectory is visible.

Use separate panels for each vision objective. Do not average away objective-family differences.

---

## 15. Experiment 5: token identity, order, and composition controls

### Caption transformations

Implement deterministic, seed-recorded transformations:

1. `within_caption_shuffle`
   - preserve token multiset for each caption;
   - destroy word order;
   - exact BoW must remain identical.

2. `cross_caption_reassignment`
   - preserve each caption's token length;
   - approximately preserve corpus unigram counts;
   - destroy token-image association.

3. `mask_top_frequency_k`
   - `k in {10, 50, 100, 500}`.

4. `mask_special_tokens`.

5. `mask_punctuation`.

6. `mask_stopwords` using a documented language-specific list.

7. optional linguistic masks:
   - object nouns,
   - attributes/colors,
   - actions/verbs,
   - named entities,
   - OCR-like strings.

Linguistic masks are secondary because they depend on an external parser and word/subword alignment.

### Expected invariances

`within_caption_shuffle` must satisfy:

```python
K_TF(original) == K_TF(shuffled)
```

within numerical tolerance.

Any change implies a tokenization or implementation bug unless captions are detokenized and retokenized. Prefer shuffling token IDs directly for this control.

### Interpretation

- unchanged BoW plus changed deep alignment -> order/context contribution;
- collapse after cross-caption reassignment -> dependence on image-token association;
- persistence after object-noun masking -> alignment beyond explicit category naming.

---

## 16. Cross-fitted lexical residualization

### Goal

Test whether deep LLM features contain vision-aligned structure not linearly predictable from lexical features.

For LLM representation `H` and lexical feature matrix `P`, fit ridge regression:

\[
\hat H=P B.
\]

Use out-of-fold residuals:

\[
R_i=H_i-\hat H_i^{(-fold(i))}.
\]

### Required procedure

1. Fix five image-level folds.
2. Tune ridge `alpha` using nested CV inside training folds or use a pre-declared logarithmic grid.
3. Fit only on training images.
4. Predict held-out LLM representations.
5. Concatenate residuals in original image order.
6. Compute CKA and neighborhood metrics between residuals and vision representations.

Alpha grid:

```text
1e-6, 1e-5, ..., 1e6
```

Standardize dense targets using training-fold statistics only. For sparse lexical inputs, avoid mean-centering the sparse matrix in memory; use a compatible solver/intercept treatment.

### Primary lexical controls

Run separately with:

- TF,
- TF-IDF,
- trained layer 0.

Do not call this a formal causal decomposition. Describe it as out-of-sample removal of linearly lexical-predictable structure.

### Leakage tests

- folds must be assigned by image, not caption row;
- all caption variants of one image belong to the same fold;
- near-duplicate image groups should remain within one fold where possible.

---

## 17. Sampling, confidence intervals, and stability

### Sample-size sweep

Use nested deterministic subsets where possible:

```text
n = 256, 512, 1024, 2048, 4096, ...
```

Generate at least five subset seeds or use one nested sequence plus bootstrap uncertainty. Report when estimates stabilize.

### Bootstrap

Final pilot default:

```text
1,000 image-level bootstrap replicates
```

For paired caption comparisons, sample image indices once and apply the same sampled indices to all conditions.

If duplicates or multiple captions per image exist, bootstrap at the image/group level.

### Model-family inference

Do not treat 170 model pairs as 170 independent observations. Aggregate and visualize by:

- LLM family,
- tokenizer family,
- LLM size,
- vision training objective,
- vision architecture,
- dataset,
- caption condition.

A later statistical model may use crossed random effects, but this is not required for the first pilot.

---

## 18. Pilot stop/go gate

After Experiments 1-5, create a machine-generated `pilot_report.md` containing all estimates, failures, and configuration hashes.

### Strong go

Proceed to the full scale-up if at least one is robust across samples/seeds:

1. exact lexical kernels produce a nontrivial alignment floor for original human captions;
2. trained layer 0 approaches best deep-layer alignment for multiple model families;
3. controlled caption descriptiveness changes alignment predictably;
4. model or layer rankings materially change under lexical controls;
5. residual alignment differs systematically by vision-training objective;
6. permutation significance remains high while lexical controls substantially alter interpretation.

### Reframe

Reframe around synthetic descriptions if:

- generated captions create a large lexical floor,
- original captions do not,
- and the effect generalizes across generators/prompts or detail levels.

### Stop broad scale-up

Do not launch 10 x 17 x 3 if all are true:

- exact lexical alignment is negligible for original captions;
- trained layer 0 remains far below contextual layers;
- lexical control does not change model/layer conclusions;
- generated-caption effects are specific to one generator/prompt.

The gate decision must be written to a versioned YAML/Markdown record and approved manually before large compute jobs are submitted.

---

## 19. Scale-up design after passing the gate

### Tier 1: broad grid

For all 10 LLMs x 17 ViTs x 3 datasets, run only:

- exact lexical features for each tokenizer,
- trained layer 0,
- a predeclared set of LLM layers such as `{early, middle, late, final}`,
- final/pre-projection vision representation,
- linear CKA,
- `k=10` neighborhood overlap.

### Tier 2: representative subset

For approximately 3 LLMs x 5 ViTs, run:

- all language layers,
- selected/all vision layers,
- pooling variants,
- random-width sweep,
- caption transformations,
- residualization,
- sample-size stability.

### Tier 3: mechanistic case studies

Choose 2-4 pairs after Tier 1 using predeclared selection logic:

- strongest lexical explainability,
- largest contextual gain,
- representative DINO-like encoder,
- representative CLIP-like encoder.

Do not choose only aesthetically convenient examples.

---

## 20. Configuration examples

### `configs/experiments/01_wit_lexical_baselines.yaml`

```yaml
experiment_id: wit_lexical_baselines_v1
seed: 0

dataset:
  id: wit
  manifest: data/manifests/wit_pilot.parquet
  split: pilot
  caption_variants:
    - gemini_detailed_existing
  sample_size: 1024
  sample_seed: 0

language_models:
  - id: EleutherAI/pythia-160m
    revision: REQUIRED_PIN
    layers: [0, 1, 3, 6, 9, 12]
    pooling: masked_mean
    include_special_tokens: true

vision_models:
  - id: dinov2_current_exact_id
    revision: REQUIRED_PIN
    representations: [final_cls]
  - id: clip_current_exact_id
    revision: REQUIRED_PIN
    representations: [pre_projection, post_projection]

lexical_features:
  - length
  - count
  - binary
  - tf
  - tfidf

random_embeddings:
  enabled: true
  seeds: [0, 1, 2, 3, 4]
  width: model_hidden_size
  initializer: model_default

metrics:
  cka:
    enabled: true
    dtype: float64
  knn_overlap:
    enabled: true
    k: [5, 10, 20]
  rsa:
    enabled: true

inference:
  language_batch_size: AUTO
  vision_batch_size: AUTO
  precision: float32

statistics:
  bootstrap_replicates: 1000
  permutation_replicates: 1000
  confidence_level: 0.95

output_dir: outputs/wit_lexical_baselines_v1
```

### `configs/experiments/03_random_width_sweep.yaml`

```yaml
experiment_id: wit_random_width_v1
base_experiment: wit_lexical_baselines_v1
widths: [32, 64, 128, 256, 512, 1024, 2048, 4096]
seeds: [0, 1, 2, 3, 4]
initializers:
  - model_default
  - gaussian_1_over_d
comparisons:
  - [random_l0, exact_tf]
  - [random_l0, vision]
metrics:
  - linear_cka
  - knn_overlap_k10
```

---

## 21. CLI contracts

The coding agent should implement commands similar to:

```bash
python scripts/build_manifest.py \
  --config configs/datasets/wit_pilot.yaml

python scripts/build_lexical_features.py \
  --experiment configs/experiments/01_wit_lexical_baselines.yaml

python scripts/extract_representations.py \
  --experiment configs/experiments/01_wit_lexical_baselines.yaml \
  --modality language

python scripts/extract_representations.py \
  --experiment configs/experiments/01_wit_lexical_baselines.yaml \
  --modality vision

python scripts/run_experiment.py \
  --config configs/experiments/01_wit_lexical_baselines.yaml

python scripts/aggregate_results.py \
  --input outputs/wit_lexical_baselines_v1

python scripts/make_pilot_report.py \
  --input outputs/wit_lexical_baselines_v1 \
  --output outputs/wit_lexical_baselines_v1/pilot_report.md
```

Each command must support:

```text
--dry-run
--force
--resume
--log-level
```

`--dry-run` must validate manifests, hashes, model availability, estimated artifact shapes, and expected output paths without running inference.

`--resume` must skip only artifacts whose data and metadata hashes match exactly.

---

## 22. Result schema

Use long-format Parquet. Required columns:

```text
experiment_id
config_sha256
git_commit
dataset_id
split
caption_variant_id
caption_source
caption_detail_level
sample_size
sample_seed
ordered_image_ids_sha256
language_model_id
language_model_revision
tokenizer_id
tokenizer_revision
language_representation_type
language_layer
language_pooling
vision_model_id
vision_model_revision
vision_representation_type
vision_layer
lexical_feature_type
control_type
metric
metric_parameter
random_seed
estimate
ci_lower
ci_upper
permutation_mean
permutation_std
permutation_p
artifact_left_sha256
artifact_right_sha256
created_at_utc
```

Do not encode critical experimental variables only in filenames.

---

## 23. Required figures for the pilot

Generate publication-quality vector graphics plus source data.

### Figure 1: alignment ladder

For each selected vision encoder:

```text
length -> TF -> random L0 -> trained L0 -> contextual layers
```

Show original and generated captions separately.

### Figure 2: random-width convergence

- x-axis: embedding width on log2 scale;
- y-axis A: CKA with exact TF;
- y-axis B or second panel: CKA with vision;
- line: seed mean;
- band/error bars: 95% interval.

### Figure 3: caption intervention

Paired changes from original to detailed captions at:

- exact TF,
- trained layer 0,
- best/predeclared deep layer.

### Figure 4: layer trajectories

Full LLM layer index against alignment, faceted by vision objective and caption type.

### Figure 5: controls

Original, within-caption shuffled, cross-caption reassigned, category-masked.

### Figure 6: lexical residualization

Raw deep alignment versus out-of-fold residual alignment.

Every figure must write its plotted values to a neighboring CSV/Parquet file.

---

## 24. Implementation order and deliverables

### Milestone A: foundations

Deliver:

- canonical schema and manifest validator;
- artifact metadata/hashing;
- tokenizer count/TF/binary/TF-IDF features;
- CKA, k-NN overlap, RSA;
- unit tests.

Exit criteria:

```bash
pytest -q
```

passes, including feature-vs-kernel CKA equivalence.

### Milestone B: reproduce existing WiT-Gemini result

Deliver:

- exact TF baseline;
- random and trained layer 0;
- DINOv2 and CLIP comparisons;
- five random seeds;
- first alignment-ladder figure.

Exit criteria:

- same `image_id` ordering verified by hashes;
- random-L0 vs exact-TF CKA reported;
- discrepancies documented.

### Milestone C: caption comparison

Deliver:

- original WiT captions on the identical image subset;
- original-vs-Gemini paired analysis;
- caption statistics;
- paired bootstrap intervals.

### Milestone D: random-width theory test

Deliver:

- width sweep;
- convergence figure;
- diagnostics if theory fails.

### Milestone E: mechanistic controls

Deliver:

- order/identity controls;
- token masking;
- cross-fitted residualization;
- leakage tests.

### Milestone F: pilot decision

Deliver:

- `pilot_report.md`;
- figures and source data;
- versioned stop/go decision;
- compute estimate for the broad sweep.

---

## 25. Coding-agent task checklist

The agent should work through these tasks in order and mark each complete in the pull request description.

- [ ] Inspect the existing repository and map current code to the proposed modules.
- [ ] Preserve existing public APIs where reasonable; document migrations.
- [ ] Implement canonical manifest schema and validators.
- [ ] Implement ordered image-ID hashing and fail-closed artifact joins.
- [ ] Implement tokenizer-level sparse count matrix.
- [ ] Implement TF, binary, TF-IDF, and length features.
- [ ] Implement/verify masked-mean trained layer-0 extraction.
- [ ] Implement random layer-0 construction directly as `P_TF @ E`.
- [ ] Implement linear CKA in feature and kernel forms.
- [ ] Add equivalence/property tests.
- [ ] Implement cosine k-NN overlap with explicit self-exclusion.
- [ ] Implement Spearman RSA blockwise.
- [ ] Implement paired permutation and bootstrap procedures.
- [ ] Add model adapter interfaces.
- [ ] Add WiT manifest adapter for existing original and Gemini captions.
- [ ] Add experiment YAML parsing and config hashing.
- [ ] Add resumable representation extraction.
- [ ] Add long-format Parquet result writer.
- [ ] Implement Experiment 1 end to end.
- [ ] Implement original-vs-Gemini paired analysis.
- [ ] Implement random-width sweep.
- [ ] Implement caption transformation controls.
- [ ] Implement cross-fitted ridge residualization without leakage.
- [ ] Implement plots and source-data export.
- [ ] Generate `pilot_report.md` automatically.
- [ ] Document exact commands in repository README.

---

## 26. Pull request acceptance requirements

A PR is not complete unless it includes:

1. code,
2. tests,
3. a small synthetic or toy fixture,
4. configuration example,
5. command used to execute it,
6. expected output schema,
7. no hard-coded local paths,
8. pinned model revisions where available,
9. artifact/hash validation,
10. failure messages that identify mismatched image IDs, dimensions, or revisions.

For model-dependent tests, mark slow/integration tests separately. Unit tests must not download models or datasets.

---

## 27. Open parameters requiring researcher input

The agent should expose these as configuration, not guess or hard-code them:

- exact WiT subset manifest/path;
- exact existing Gemini-caption column/path;
- exact Pythia checkpoint and revision;
- exact DINOv2 and CLIP model IDs/checkpoints;
- current definition of “mutual k-NN” if it differs from neighborhood overlap;
- layers and pooling used in preliminary findings;
- inclusion/exclusion of BOS/EOS tokens in the existing pipeline;
- approved storage backend and output root;
- available GPU/CPU limits;
- additional two datasets for scale-up;
- list of 10 LLMs and 17 ViTs.

The implementation should stop with actionable validation errors when any required item is absent.

---

## 28. First command the coding agent should execute

Before changing code:

```bash
find . -maxdepth 3 -type f | sort
```

Then produce a short implementation map:

```text
existing module -> proposed responsibility -> action: reuse/refactor/create
```

After that, implement **Milestone A only** in the first PR. Keep inference and scientific-result changes in subsequent PRs so mathematical validation can be reviewed independently.

---

## 29. Definition of success

The pipeline succeeds when it can reproducibly answer, for a fixed image set:

1. How aligned is vision geometry with caption length alone?
2. How aligned is it with exact token identity/frequency?
3. Does random layer 0 converge to that lexical geometry with width?
4. How much does the trained embedding table change alignment?
5. How does alignment evolve through contextual layers?
6. How do all five quantities change with caption descriptiveness?
7. What alignment remains after out-of-sample lexical residualization?
8. Which conclusions generalize across datasets, tokenizers, LLM families, and vision objectives?

The output must support these answers without relying on undocumented notebook state or manual row alignment.
