# exp-019: where in the spectrum does cross-model alignment live? (random and PCA truncation)

**Status:** synthetic calibration and local pilot done (n = 1024, small models); full run on D2 pending ·
**Owner:** oddharak · **Outputs:** `results/exp-019-truncation/{synthetic,pilot}/` (local; full run:
`/workspace/results/oddharak/exp-019-truncation/`)

## Idea

Two 2025 papers show that text embeddings are redundant under dimension removal:

- **Tsukagoshi & Sasano (Findings ACL 2025):** keeping the first 0.5% of dimensions barely hurts classification; random-d and PCA give the same trend.
- **Takeshita et al. (EMNLP 2025):** randomly removing 50% of dimensions costs under 10% on retrieval and classification, and many dimensions *degrade* performance.

Applied to our question, the naive version ("does alignment stay high under truncation?") is mostly predictable:

- Random truncation keeps the Gram matrix up to scale plus noise (a Johnson-Lindenstrauss argument) at ID 20-40.
- Every kernel measure inherits that, from each model's own spectrum, without any information about the other model.

The informative version uses truncation as a **spectral probe**:

- Which band of each model's principal components carries the agreement with the other model?
- Is that band different for cross-modal and within-modality pairs, and for local (mutual kNN) and global (CKA) measures?

Read every curve against **self-retention**: how much truncation changes each model's own neighbourhoods or Gram matrix.

## Hypotheses (written before the pilot)

- **H-T0 (null):** random-truncation curves follow self-retention; nothing cross-modal is learned from them.
- **H-T1:** cross-modal shared content sits in the leading PCs and the tail is private. Prediction: cross-modal mutual kNN is non-monotone under top-m PCA (it peaks at some k*), while within-modality pairs keep rising (the alignment version of Takeshita's degrading dimensions; F7's private variance).
- **H-T2 (full run):** k* is larger with Gemini than with the original captions: informative captions widen the shared band (F5).
- **H-T3 (full run):** k* grows with model size.
- **Koepke et al. check:** they found cross-modal RSA concentrated in the top ~3 eigenmodes. If local agreement survives removing the top ~10 PCs, that result does not carry over to local structure.

## Method (`src/geoprh/truncation.py`, `truncate.py`)

**Pipeline per layer:** raw features → PRH clamp (q = 0.95, upstream `remove_outliers`, one threshold over all layers) → centre → truncate → l2 normalise.

- The clamp comes first because it defines the representation; clamping after truncation would depend on the basis.
- Centring makes PCA well defined. The uncentred PRH score is also reported (`prh`).
- At full size every operator reproduces the centred geometry exactly (tested).

**Operators** (both models truncated the same way):

- `top`: keep the top-m PCs;
- `drop_top`: remove the top m, up to the model's common rank;
- `rand_coord`: keep m random coordinates of the model's own basis;
- `rand_proj`: keep m directions of a Haar-random orthonormal basis (the JL control: it differs from `rand_coord` only if the basis is privileged).

m is absolute, from 2 to 512. m = 1 is excluded: l2 on one coordinate is a sign, and kNN ties broken by sample index produce fake agreement (seen in a first synthetic run: mKNN 0.4 where nothing was shared).

**Measures:**

- Mutual kNN (k = 10) and linear CKA for every layer pair (7 evenly spaced layers per model).
- Each cell is calibrated against a permutation null of the pairing (`layer_calibration` Algorithm 1). The max over cells is compared with the null of the max (Algorithm 2). The same permutations are used at every setting.
- Primary statistic: **excess** = max score − mean of the null of the max.
- `calibrated_pairs` builds each B model's permuted Grams once per chunk for all pairs. Mutual-kNN nulls use `knn_sweep`. Both are tested to match the dense per-pair computation exactly.

**Self-retention:** per model and layer, mutual kNN and CKA between the truncated and the full layer. For a pair, the reference is the geometric mean of the two models' retention at the pair's fixed cell, which is the best layer pair untruncated.

- This is a heuristic reference, not a null.
- If truncation replaced neighbours independently in the two models, the product would be the prediction, and the product is lower still.
- Relative curves use the same fixed cell.

## Synthetic calibration (`truncate.py synthetic`)

Setup: two "models" share 8 latents and each has 16 private latents, rotated into d = 512 with small isotropic noise; n = 1024, 50 permutations. The shared block is placed at a high, equal or low standard deviation relative to the private block.

| scenario | untruncated mKNN / CKA | top-m PCA | remove top-m PCs | random truncation |
|---|---|---|---|---|
| shared_top (sd 3 vs 1) | 0.57 / 0.95 | peak at m = 8 (the shared block): mKNN 0.90, CKA 0.99 | removing 8 → 0 | monotone loss (m = 32: mKNN 0.26); coordinates = projections |
| shared_mid (1.5 vs 1.5) | 0.05 / 0.32 | monotone | monotone | monotone |
| shared_tail (1 vs 3) | −0.00 / 0.01 | nothing | removing 16 (the private block) → mKNN **0.49** | nothing at any m |

- The curves locate the shared block where it was placed.
- Shared content that untruncated measures cannot see at all is found by removing the leading PCs.
- Random truncation cannot reveal it.

## Pilot (`truncate.py real`, local, 2026-11-04)

**Data:** cached PRH `wit_1024` features (original WIT captions, n = 1024), 7 evenly spaced layers per model.

- 3 LLMs: bloomz-560m, 1b1, 1b7;
- 4 ViTs: AugReg tiny / base, DINOv2-S, CLIP ViT-B laion2b.

**Pairs:**

- X: 12 LLM × ViT;
- W_llm: 3;
- W_vit: 6;
- D_shuffle: bloomz-1b7 × CLIP with the samples shuffled once.

**Run settings:** 50 permutations, 2 random draws, 16 min on the laptop CPU. Code at `3fc04e1` plus the uncommitted exp-019 files.

**Untruncated excess:**

| | mutual kNN | CKA |
|---|---|---|
| X | 0.098 [0.08, 0.13] | 0.19 |
| W_llm | 0.66 | 0.75 |
| W_vit | 0.44 | 0.73 |

- Uncentred PRH: 0.087 / 0.200 for X.
- D_shuffle stays within ±0.003 at every setting and operator, so no truncation level manufactures agreement.

**Relative excess / self-retention at the fixed cell** (mean over pairs):

| | m = 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| mKNN, top-m, X | 0.46 / 0.34 | 0.70 / 0.50 | 0.87 / 0.64 | 0.99 / 0.77 | 1.01 / 0.87 |
| mKNN, top-m, W_llm | 0.76 / 0.34 | 0.89 / 0.50 | 0.92 / 0.63 | 0.96 / 0.75 | 0.97 / 0.85 |
| mKNN, remove top-m, X | 0.89 / 0.65 | 0.70 / 0.53 | 0.43 / 0.38 | 0.16 / 0.21 | 0.02 / 0.08 |
| mKNN, remove top-m, W_llm | 0.85 / 0.52 | 0.77 / 0.42 | 0.63 / 0.32 | 0.42 / 0.22 | 0.20 / 0.11 |
| CKA, top-m, X | 1.12 / 0.80 | 1.16 / 0.87 | 1.12 / 0.92 | 1.08 / 0.96 | 1.05 / 0.98 |
| CKA, remove top-m, X (m = 2: 0.56 / 0.73) | 0.23 / 0.49 | 0.15 / 0.39 | 0.07 / 0.29 | 0.02 / 0.21 | 0.01 / 0.13 |
| mKNN, random coordinates, X | 0.04 / 0.07 | 0.11 / 0.14 | 0.23 / 0.24 | 0.45 / 0.39 | 0.67 / 0.55 |

**Findings:**

1. **H-T0 holds.**
   - Under random truncation the cross-modal mKNN curve follows self-retention: 0.23 vs 0.24 at m = 32, 0.45 vs 0.39 at m = 64.
   - In absolute terms, keeping 512 random dimensions retains 95% of X's mKNN excess, and 128 retain 67%. This reproduces the papers' redundancy.
   - It carries no cross-modal information, which is the critique's prediction.
   - `rand_coord` and `rand_proj` agree within 0.03 for X and W_llm.
   - For W_vit, random directions do better than random coordinates at m = 16-32 (0.31 vs 0.22, 0.50 vs 0.40): the ViT channel basis is mildly privileged.
2. **The two measures locate cross-modal agreement in different bands.**
   - **CKA:** cross-modal CKA sits in the top ~2-8 PCs. Removing only the top 2 leaves 56% of X's CKA, against 73% self-retention; removing the top 8 leaves 23% vs 49%. Truncating to 8-16 PCs *raises* it to 1.12-1.16×, which is the CKA version of degrading dimensions. This matches Koepke et al.'s top-eigenmode result.
   - **Mutual kNN:** cross-modal local agreement survives removing the top 8 PCs (89% vs 65% self-retention), is nearly complete in the top 64 (0.99 vs 0.77), and is gone after removing the top 128. It lives in roughly **PCs 8-64**, a band more concentrated than the models' own local structure. So Koepke's top-mode picture does not carry over to local structure.
3. **H-T1 is not supported for mutual kNN.**
   - No top-m truncation raises cross-modal mKNN (X peaks at 1.01× at m = 128, which is noise).
   - The tail beyond ~64 PCs is neither shared nor harmful for neighbourhoods. It only dilutes CKA.
4. **Within-modality agreement extends further into the spectrum than cross-modal agreement.**
   - After removing the top 64 PCs, W_llm keeps 42% of its mKNN excess (self 22%), while X keeps 16% (self 21%).
   - So the "shared band" between LLMs is wider than the one between an LLM and a ViT, consistent with F5 / F6 (shared content bounds alignment).
   - W_vit behaves like X here, but the ViTs are low-dimensional (d = 192-768), which confounds the comparison.

**Caveats:**

- Small models, original WIT captions, and n = 1024: the X excess is only ~0.1.
- 50 permutations instead of 200, and 2 random draws.
- No null of the max over m yet. The CKA gain from selecting m is therefore not yet calibrated, but D_shuffle shows none of it.
- m is absolute while d differs across models (192-2048).
- Self-retention is a heuristic reference.

## Next: full run on D2 (needs the volume, so a CPU pod)

**Data and models:**

- exp-016 `d2s` features: n = 1024, all layers;
- 4 LLMs up to llama-13b;
- DINOv2-g, CLIP ViT-H, MAE ViT-H, AugReg ViT-L;
- original vs Gemini captions;
- dual encoders and the B1 / B2 ceilings.

The current code runs as is on these.

**Additions:**

- 200 permutations;
- a null of the max over m;
- one-model-only truncation;
- per-pair k* and band edges, to test H-T2 (Gemini widens the mKNN band?) and H-T3 (size).

**Scaling to n = 10240 (`d2l`):** CKA would switch to the d × d formula and mKNN to `knn_sweep` only, because the dense n × n caches are too large. Leave that until d2s shows a caption effect.

## Commands

```bash
uv run pytest tests/test_truncation.py tests/test_knn_sweep.py -q
uv run python experiments/exp-019-truncation/truncate.py synthetic --out results/exp-019-truncation/synthetic --draws 2 --permutations 50
uv run python experiments/exp-019-truncation/truncate.py real --root results/activations --out results/exp-019-truncation/pilot --draws 2 --permutations 50
uv run python experiments/exp-019-truncation/plot.py --root results/exp-019-truncation/pilot
```

Figures: `fig_excess` (excess against m) and `fig_retention` (relative to untruncated, with self-retention dashed). Both show the mean over the pairs of each group with a min..max band. `summary.md` lists every value and the per-pair k*.

## Sources

- [Tsukagoshi & Sasano, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.1330.pdf)
- [Takeshita et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1410.pdf)
- [Koepke et al. 2026](https://arxiv.org/abs/2604.18572)
