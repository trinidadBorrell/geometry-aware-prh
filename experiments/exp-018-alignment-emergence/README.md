# exp-018: when does a language model's alignment with vision emerge during pretraining? (pilot)

**Status:** pilot done (Gemini captions only), 2026-11-05, git `3fc04e1` · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-018-alignment-emergence/` (summary, curves and figures also in local `results/exp-018-alignment-emergence/`)

## Results (pilot, 2026-11-05)

Run: 4 series x 14 checkpoints, Gemini captions only (volume space), 336 pairs, 200 permutations, RTX PRO 4500, self-terminated with exit 0. All numbers are calibrated excess at the best cell, n = 10240 unless stated.

1. **Seeds agree.** Every curve of the `-seed1` replica lies within ~0.01 of the original. Seed spread is far below the change over training.
2. **Step 0 is already half way.** A random-init Pythia, mean-pooled, already gives CKA 0.39 against DINOv2-g and 0.67 against pythia-1.4b, and mutual kNN at k = 10 gives 0.10-0.11 against the ViTs. Training roughly doubles the cross-modal values: final mutual kNN at k = 10 is 0.19-0.22 and CKA 0.53-0.70.
3. **One sharp transition, then saturation.** Flat up to step 256 (0.5B tokens). Almost all of the change happens between steps 512 and 4000 (1B-8B tokens). After ~10B tokens the curves change little. This supports H-early-saturation (like brain alignment, AlKhamissi et al.) over H-world-knowledge.
   - **Caveat:** the transition coincides with Pythia's LR warmup (the first 1% of training, 1430 steps, about 3B tokens).
4. **Late decline of cross-modal alignment at 160M only.** pythia-160m x ViT mutual kNN at k = 10 peaks at 67-130B tokens and then falls by ~10% by the final step (0.218 to 0.199 for DINOv2; the same in the seed replica and for CLIP). The 160m x pythia-1.4b curves keep rising, and so does every 410m curve. The decline coincides with held-out loss getting worse in both 160m seeds: generic bits per byte goes 1.10 at step 64000, 1.11 at step 100000, then 1.15 (seed1) and 1.29 (seed0, a jump worth checking) at step 143000. The 410m loss does not rise.
5. **Large neighbourhoods first.** Spearman(k, t50) = -1.00 in 12/12 (series x target) cells, for both sizes, both seeds, and the LLM and both ViT targets.
   - Example, 410m x DINOv2: t50 is 2.0B tokens at k = 1000 and 3.7B at k = 10. At 2B tokens, k = 1000 has made ~50% of its change and k = 10 ~8%.
   - CKA onsets fall with the large-k mutual kNN, not with the small-k one.
   - Read naively, this supports H-coarse-first (progressive differentiation) and contradicts H-lexical-first.
6. **Size shifts the onset; loss does not collapse the curves.**
   - The 410m onsets come 1.3-1.7x later in tokens than the 160m ones.
   - Plotted against generic bits per byte, the curves of the two sizes do not overlap: at equal loss the 160m is *more* aligned with pythia-1.4b. H-loss fails in this form.
   - Possible confound: the sizes are trained with different peak learning rates (6e-4 for 160M, 3e-4 for 410M, per the Pythia paper).
7. **ORC given density** moves little against the ViTs (0.25 to 0.32 for 160m x DINOv2) but strongly against pythia-1.4b (0.41 to 0.89).

**Follow-up on the same records (`pairs.jsonl`, using the normalised score `g`, no new runs):**

- **The early plateau is the embedding layer.** The best cell stays at LM layer 0 until step 512-1000, then jumps to deep layers. Layer 0 at step 0 is a mean of random token embeddings, i.e. a random projection of the caption's word counts. So the step-0 alignment is lexical overlap, not learning, and the best-cell curve is a maximum over a fixed floor and rising deep layers.
- **The ordering survives a fixed layer but shrinks.** At fixed depth 0.75, Spearman(k, t50) is still -1.00 in 12/12 cells. For the ViT targets the ratio t50(k=10) / t50(k=1000) falls from 1.6-2.0 at the best cell to 1.25-1.5 at fixed depth (e.g. 160m x DINOv2: 1.63 to 1.28; 410m x DINOv2: 1.85 to 1.36). For the pythia-1.4b target it does not shrink (1.3-1.6 to 1.2-1.9). All t50 fall in steps ~480-2400, where the grid has only 4 points (512, 1000, 2000, 4000).
- **The ViT onset follows cumulative learning rate, not tokens.** Configs (EleutherAI/pythia `models/160M`, `models/410M`): Adam, lr 6e-4 vs 3e-4, 1% warmup, cosine decay, 143000 steps. Best cell, k = 10, x ViT: t50 is 1205-1281 steps (160m) vs 1660-1757 steps (410m), but the cumulative LR at t50 is 0.30-0.33 vs 0.27-0.30. The collapse is weaker at k = 1000 (0.10-0.13 vs 0.08-0.09) and absent for the pythia-1.4b target (0.39-0.40 vs 0.48-0.50). Two LR values only, so suggestive. Tokens are therefore not an identified clock in this series.

**Update from [exp-021](../exp-021-lexical-floor/README.md):** step 0 is reproduced by a Gaussian table with no network, and the same word-count kernel falls from CKA 0.41 on Gemini captions to 0.06 on original captions. TF-IDF alone reaches CKA 0.60 and k = 10 excess 0.186 against DINOv2. Read every curve here as Gemini-caption-specific until the series is run on original captions.

**Gate:**
- **Condition 1 passes** (a large change, far above seed spread).
- **Condition 2 passes on its face** (t50 separates by k, in the same direction in every cell), but it is **not yet interpretable**.
  - The README's argument (compare across k within one measure family) assumed this avoids the arm C sensitivity confound. It does not fully avoid it: small-k mutual kNN is itself the more threshold-like readout. A representation that improves uniformly at every scale could also make k = 1000 cross 50% before k = 10.
  - **Next, before any phase 2:** the interpolation null. Linearly interpolate each LM's features between step 0 and the final checkpoint, and compute calibrated mutual kNN at each k against the same ViT along the path. Coarse-first is real only if the ordering seen over training is stronger than the ordering along this uniform path.
  - Then densify the grid where the action is (steps 512-8000, every 500-1000 steps), and separate warmup from token count, for example with Pythia sizes that share the same learning rate.

## Idea

Every PRH result so far compares *final* models. Public checkpoint series let us watch a language model's alignment with fixed vision models develop over training: Pythia has 154 checkpoints per size, the same data order for every size, and seed replicas; OLMo-2-1B has 195 stage-1 checkpoints up to 4T tokens plus three mid-training anneals.

Four questions:

1. **Shape:** does cross-modal alignment rise early and saturate, or keep growing with tokens? Brain alignment of Pythia models peaks within the first few billion tokens (AlKhamissi et al., EMNLP 2025).
2. **Order across scales:** does fine (local) structure align before coarse (global) structure, or the reverse?
3. **Compute:** across model sizes, is alignment a function of tokens, compute (6ND), or LM loss alone?
4. **Data:** do the data interventions inside the series move alignment? Pythia deduped vs standard; OLMo-2's mid-training anneal adds math and curated data. This links to exp-020.

## Prior work (searches on 2026-11-04)

- **Balestriero and LeCun (2610.39302):** cross-modal mutual 10-NN rises from chance during training of their own small text encoders (BERT-style, 11M-335M, Flickr30k / Recap-DataComp). Uncalibrated, local only; no public LM series; no local vs global comparison over training; pretrained decoders scored only as final models.
- **AlKhamissi et al. (2025):** brain alignment over 34 Pythia checkpoints rises fast and saturates at ~2-8B tokens, tracking formal rather than functional competence. Same method family, different target (fMRI).
- **Crosscoding Through Time (ACL 2026):** language features over Pythia / OLMo / BLOOM checkpoints; language only.
- **Herbst et al. (NeurIPS 2026 workshop):** "where alignment emerges" across layers and representation dimensions, per the abstract. Read the paper to confirm it has no training-time analysis.

We found no study that tracks *calibrated* cross-modal alignment across public LM checkpoints, compares the emergence of local and global alignment, or relates it to compute or loss across sizes.

## Hypotheses (competing predictions)

- **H-coarse-first (progressive differentiation; Saxe, McClelland, Ganguli, PNAS 2019):** networks learn broad distinctions before fine ones. Alignment at large neighbourhood scales (large k/n, CKA) emerges first; small-k mutual kNN later.
- **H-lexical-first:** in the first ~1000 steps Pythia is close to a unigram / bigram model. Mean-pooled caption features are then a weighted bag of word embeddings, and captions sharing content words (and hence images) become neighbours. Local agreement would then appear early and global agreement only once context is composed.
- **H-early-saturation vs H-world-knowledge:** cross-modal alignment either saturates with formal competence (a few B tokens, like brain alignment) or keeps growing as world knowledge accumulates (hundreds of B tokens).
- **H-loss:** curves from different sizes collapse when plotted against the LM's own loss, i.e. alignment is a function of capability, not of size or tokens separately.

## A measurement trap, and the design that avoids it

Comparing *when* mutual kNN rises with *when* CKA rises is confounded. The two measures have different sensitivity curves: exp-016 arm C shows that local measures stay near zero until private variance is small, then rise steeply, while CKA rises gradually. A representation that improves uniformly at every scale would therefore look "global first" on mKNN vs CKA purely through measure sensitivity.

The clean comparison is within one measure family across neighbourhood scale: mutual kNN at n = 10240 for k = 10, 30, 100, 300, 1000, i.e. k/n from 0.1% to 10%.

- Calibrated mutual kNN at every k is the primary scale-resolved readout.
- CKA (n = 10240) is the global end. Centred CKNNA (exp-008) across k would fill in the path from local to global; it is left out of the pilot to keep the metric set small.
- mKNN vs CKA is reported, but read through the arm C dose-response (each measure converted to the private-variance fraction that would produce it), never raw.

## Pilot design

**Language models (checkpoint series)**

- `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`;
- their seed replicas `-seed1` (PolyPythias: same Pile, a different initialisation and data order), for seed variability.

**Checkpoints:** 14, log-spaced: steps 0, 16, 64, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000. One step is ~2.1M tokens, so the series runs from 0 to ~300B tokens. Step 0 is the random-initialisation baseline, which is already a bag of random token embeddings and so not zero.

**Data and vision side (cache first)**

- D2 from exp-016: 10240 WIT-1M images with original and Gemini captions.
- Cached CLS features of DINOv2-g and CLIP ViT-H (`d2s`, `d2l`; MAE ViT-H and AugReg ViT-L with `VITS=...` if time allows). The vision targets are fixed final models, so this measures the LM moving toward them.

**Within-modality reference:** each checkpoint x final `pythia-1.4b` on the same captions. Does LLM x LLM agreement emerge earlier than LLM x ViT agreement?

**Per checkpoint**

- Features: mean-pooled hidden states (exp-020 `extract_lm.py` with `--revisions`), `d2s` all layers, `d2l` 7 layers.
- Alignment, each calibrated with 200 permutations; best cell against the null of the max, plus fixed relative depth 0.75:
  - mutual kNN at k = 10, 30, 100, 300, 1000 (n = 10240; `geoprh.knn_sweep`, one permutation stream for all k);
  - linear CKA;
  - ORC given density and density agreement (n = 1024).
- Diagnostics:
  - LM loss (bits per byte on a fixed held-out text and on the captions);
  - TwoNN ID (from the geometry stage).
  - Not yet in the code: the variance share of the top PC and the outlier-dimension ratio. Anisotropy and massive activations develop during training and move CKA, so add these before reading CKA curves.

**Readouts**

1. Calibrated excess vs tokens, per measure and per k, with bootstrap bands over samples and the spread over two seeds.
2. Onset tokens t25 / t50 / t75: tokens at which progress (e(t) - e(step 0)) / (e(final) - e(step 0)) reaches 25 / 50 / 75%, log-linear interpolation, with a 95% bootstrap CI over samples for mutual kNN. A curve whose total change has a CI including 0 gets no onset. The ordering test: Spearman of t50 against k.
3. Alignment vs LM loss, both sizes on one plot (H-loss).
4. The same curves for LLM x LLM (within-modality emergence).

## Decision gate (written before running)

**Continue to phase 2 if both hold:**

1. Calibrated alignment changes over training by clearly more than seed spread plus bootstrap CI. If it is flat after the first ~1000 steps for every measure, the result is "alignment is set by early lexical learning", which is reportable, but a much smaller project.
2. The t50 values for different k separate by more than their CIs, in the same direction for both sizes and both seeds. Otherwise the local-vs-global question has no answer at this resolution.

## Phase 2 (if the gate passes)

- **Sizes and compute:** Pythia 70M-2.8B on the full log grid, plus 6.9B / 12B at selected checkpoints. Plot alignment vs tokens, vs compute 6ND, and vs loss.
- **Seeds:** PolyPythias, 9 seeds for sizes up to 410M.
- **Data interventions:**
  - Pythia deduped vs standard (same order, deduplicated Pile);
  - OLMo-2-1B: log-spaced stage-1 checkpoints to 4T tokens, then the three stage-2 anneal ingredients (`stage2-ingredient{1,2,3}`). Does the mid-training mix move alignment? This is the exp-020 question inside one training run.
- **Captions:** add DOCCI human dense captions, which needs a small ViT extraction.

## Cost (to be priced live before creating a pod)

- **Extraction:** 4 series x 14 checkpoints = 56 forward passes of <= 410M models on 20480 captions (~7M tokens). About 2-3 GPU-hours on an RTX PRO 4000 / 4500 class GPU.
- **Downloads:** ~0.3 GB (160M) and ~0.8 GB (410M) per checkpoint, deleted after extraction.
- **Measurement:** 56 checkpoints x (2 ViTs + 1 reference) x 2 caption sets x 2 n = 672 pairs. At ~30-40 s per n = 10240 pair (exp-016, 24 GB GPU) about 3-4 h; ORC geometry is of the order of exp-014's 1328 layer tasks (9 min on 28 vCPU).
- **Storage:** ~0.4 GB of features per 410M checkpoint (both caption sets), ~18 GB in total. The volume is close to its quota: check `df` and free space first, or run with `CAPTIONS=gemini`.
- Can share a pod session with exp-020. One pod at a time, with self-termination.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=16 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp018.sh > /workspace/results/oddharak/exp-018.log 2>&1 < /dev/null &"
# pieces, as the runner calls them:
uv run python experiments/exp-020-lm-data-pilot/extract_lm.py --out <out> --d2 <exp016>/d2.parquet --models EleutherAI/pythia-410m --revisions step0,step1000 --cache-dir /root/hf-scratch
uv run python experiments/exp-018-alignment-emergence/align.py --roots <out>,<exp016 out> --stages geometry,pairs --refs EleutherAI__pythia-1.4b
uv run python experiments/exp-018-alignment-emergence/analyze.py --out <out>
```

Outputs: `features/`, `bpb/`, `geometry/`, `pairs.jsonl` + `pairs/*.npz`, `summary.md` (onsets and the ordering test), `curves.json`, `fig_curves`, `fig_vs_bpb`.

## Risks

- **Descriptive by nature.** The hypotheses above, and the data interventions in phase 2, make it a test rather than a set of curves.
- **Early checkpoints** (warmup is the first ~1% of steps) have unusual norms and anisotropy. PRH preprocessing plus the diagnostics; report raw as a check.
- **Best-layer selection changes over training.** Report both the null-of-max best cell and fixed relative depths.
- **The fixed vision target** measures convergence of the LM toward today's vision models, not mutual convergence. No public ViT checkpoint series of comparable density is known to us.
