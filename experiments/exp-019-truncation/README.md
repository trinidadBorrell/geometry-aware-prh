# exp-019: where in the spectrum does cross-model alignment live? (random and PCA truncation)

**Status:** synthetic calibration, local pilot and D2 run done (2026-11-05). Truncation does not reveal hidden cross-modal agreement. Local agreement sits in the leading ~40 PCs, and better captions raise its level without moving the band. The CKA "gain" from truncation is a falling null, not more agreement (this corrects the pilot reading) ·
**Owner:** oddharak · **Outputs:** `results/exp-019-truncation/{synthetic,pilot,d2s}/` (local; D2 also at
`/workspace/results/oddharak/exp-019-truncation/d2s`)

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
   - **CKA:** cross-modal CKA sits in the top ~2-8 PCs. Removing only the top 2 leaves 56% of X's CKA, against 73% self-retention; removing the top 8 leaves 23% vs 49%. Truncating to 8-16 PCs *raises* the excess to 1.12-1.16×. This matches Koepke et al.'s top-eigenmode result. **Corrected by the D2 run:** the rise is in the excess only. Raw CKA falls, and the null falls further, so this is not "degrading dimensions" (D2 finding 2).
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

## D2 run (`truncate.py d2s`, pod, 2026-11-05)

**Data:** exp-016 `d2s` features, n = 1024, 9 evenly spaced layers per model.

- 4 LLMs (bloomz-1b7, 7b1, open_llama_7b, llama-13b) on original and Gemini captions;
- 4 ViTs (AugReg ViT-L, MAE ViT-H, DINOv2-g, CLIP ViT-H laion2b);
- 3 dual encoders (CLIP L/14, OpenCLIP H/14, SigLIP so400m).

**Pairs (61):**

- B3 (LLM × ViT): 16 per caption set;
- W_llm: 6 per caption set;
- W_vit: 6;
- A (a dual encoder's text and image towers): 3 per caption set;
- B1 (same LLM, original × Gemini caption): 4;
- D_shuffle: 1.

**Run settings:** 200 permutations (p floor 0.005), 2 draws of random coordinates, 1 of random directions, 47 settings. Code at `aac9fdf`. cpu3g 16 vCPU, about 1.9 h, about $1.25. D_shuffle has excess within ±0.003 at every setting and operator.

`bands.py` adds the band widths, the caption and size comparisons and the null of the max over m (`d2s/bands.md`). Band widths use the excess at the best cell (log2-interpolated):

- m90: top-m PCs needed to reach 90% of the untruncated excess;
- d50: leading PCs whose removal halves it.

| mutual kNN | untruncated excess | m90 | d50 |
|---|---|---|---|
| B3, original | 0.126 [0.073, 0.188] | 42 [27, 54] | 33 [27, 38] |
| B3, Gemini | 0.345 [0.262, 0.438] | 38 [25, 56] | 39 [28, 51] |
| A, original | 0.207 [0.178, 0.240] | 42 [35, 49] | 39 [31, 50] |
| A, Gemini | 0.260 [0.155, 0.327] | 58 [57, 60] | 46 [39, 56] |
| B1 | 0.167 [0.139, 0.187] | 42 [39, 45] | 45 [39, 51] |
| W_vit | 0.462 [0.372, 0.536] | 13 [8, 19] | 27 [8, 40] |
| W_llm, original | 0.564 [0.470, 0.682] | 31 [19, 47] | 107 [91, 142] |
| W_llm, Gemini | 0.781 [0.710, 0.833] | 14 [10, 15] | 62 [51, 81] |

**Relative excess / self-retention at the fixed cell** (mean over pairs):

| mutual kNN | m = 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| top-m, B3 original | 0.46 / 0.33 | 0.68 / 0.49 | 0.86 / 0.63 | 0.97 / 0.75 | 1.00 / 0.84 |
| top-m, B3 Gemini | 0.48 / 0.42 | 0.71 / 0.58 | 0.88 / 0.72 | 0.97 / 0.82 | 1.01 / 0.89 |
| remove top-m, B3 original | 0.91 / 0.67 | 0.76 / 0.57 | 0.52 / 0.43 | 0.23 / 0.27 | 0.05 / 0.10 |
| remove top-m, B3 Gemini | 0.89 / 0.64 | 0.75 / 0.53 | 0.52 / 0.40 | 0.28 / 0.25 | 0.08 / 0.11 |
| remove top-m, W_llm original | 0.93 / 0.65 | 0.87 / 0.56 | 0.74 / 0.44 | 0.49 / 0.29 | 0.20 / 0.13 |
| remove top-m, W_llm Gemini | 0.89 / 0.63 | 0.83 / 0.53 | 0.67 / 0.40 | 0.47 / 0.26 | 0.24 / 0.13 |

**CKA at top-16: what moves** (means over pairs, untruncated → top-16, at the fixed cell):

| | raw CKA | cell null | excess | pairs with a higher excess |
|---|---|---|---|---|
| B3, original | 0.383 → 0.312 | 0.131 → 0.012 | 0.251 → 0.300 | 16/16 |
| B3, Gemini | 0.635 → 0.633 | 0.041 → 0.009 | 0.594 → 0.623 | 16/16 |
| W_llm, original | 0.850 → 0.824 | 0.108 → 0.010 | 0.743 → 0.814 | 5/6 |
| W_vit | 0.825 → 0.814 | 0.014 → 0.008 | 0.811 → 0.806 | 1/6 |
| A, original | 0.439 → 0.391 | 0.116 → 0.012 | 0.323 → 0.380 | 3/3 |
| D_shuffle | 0.067 → 0.010 | 0.065 → 0.010 | 0.002 → 0.000 | 0/1 |

**Findings:**

1. **The pilot's mutual kNN picture holds on larger models.**
   - B3 agreement is nearly complete in the top 64 PCs (0.97, self 0.75), survives removing the top 8 (0.91, self 0.67), and is almost gone after removing the top 128 (0.05).
   - It runs above self-retention for the leading PCs and falls to or below it by PC 64-128, so the band is more concentrated than the models' own local structure.
   - H-T0 holds again: under random coordinates B3 follows self-retention (m = 64: 0.38 / 0.35 original, 0.48 / 0.46 Gemini), and 512 random coordinates keep 90-92% of the excess. Coordinates and directions agree for B3, W_llm and A.
2. **The CKA gain is a falling null, not more agreement.**
   - Top-16 raises the max-cell excess in every group, and the gain exceeds the 95% quantile of the null of the max over m in all 60 non-shuffled pairs. That test only calibrates selection at zero alignment, and it does not say what moved.
   - Mean raw CKA at the fixed cell does not rise in any group. The tail contributes chance-level CKA: D_shuffle, with nothing shared, drops from 0.067 to 0.010.
   - The null-of-the-max statistic exaggerates this further, because it subtracts the worst layer pair's null from the best pair's score: W_vit gains +0.11 at the best cell and −0.005 at the fixed cell.
   - So untruncated CKA at n = 1024 carries a large dimension-dependent chance level (mean 0.13 for B3 original at the fixed cell). That is a reason to prefer calibrated or debiased CKA, not evidence that some dimensions degrade alignment.
   - The top-PC location does hold at the fixed cell: removing the top 8 leaves 0.32 of B3 original's excess (self 0.58) against 0.91 for mutual kNN, the 10× band difference between the measures seen in d50 (3-4 vs 33-39).
3. **H-T1: right direction, negligible size.** Cross-modal mutual kNN peaks below full size (k* median 256) and W_llm keeps rising (k* = full), as predicted, but the peak gains at most 0.009 (B3 Gemini; 14 of 32 B3 pairs exceed the null's 95% quantile). A fixed m = 16 lowers mutual kNN in all 60 pairs. Dropping the tail does not denoise neighbourhoods.
4. **No hidden shared band.** The synthetic `shared_tail` case (invisible untruncated, 0.49 after removing the private PCs) has no real counterpart: no pair goes from invisible to large when leading PCs are removed. The largest rises are SigLIP's (finding 8).
5. **H-T2 not supported: Gemini changes the level, not the band.**
   - B3 excess rises 2.7× (0.126 → 0.345) while its relative curves barely move (top-16: 0.68 vs 0.71; remove 32: 0.52 vs 0.52). k* is unchanged. d50 rises only from 33 to 39 (15/16 pairs), m90 falls from 42 to 38.
   - W_llm goes the other way: Gemini narrows the band (d50 107 → 62, 0/6 pairs wider; m90 31 → 14), and in absolute terms the tail holds less (remove 128: 0.260 → 0.197). Gemini captions concentrate LLM geometry (top-8 self-retention 0.32 → 0.48).
6. **H-T3 not supported.** Across the 8× LLM size range d50 moves from 31 to 34 (original) and 38 to 41 (Gemini), and m90 is flat. The ViT matters more than its size: the largest (DINOv2-g) has the lowest m90 (29 vs 51 for CLIP ViT-H).
7. **The narrow band is not specific to cross-modal pairs.** B1 (same LLM, different captions) has the same band as B3 and A (d50 33-46). Only W_llm, the same text through different LLMs, reaches deep into the spectrum. W_vit is narrow, but its own local structure is too (remove 8: 0.45 / 0.25) and its best cells are early layers, so its band is not comparable.
8. **Two real exceptions: the leading PCs of some vision-text models look private.** At the fixed cell, where the mutual kNN null is flat:
   - SigLIP towers, removing the top PC: mutual kNN 0.180 → 0.232 (original), 0.157 → 0.187 (Gemini); CKA 0.186 → 0.355, 0.207 → 0.277.
   - B3 pairs with CLIP ViT-H, removing the top 2: mutual kNN +0.014 to +0.017 (original), +0.022 to +0.027 (Gemini), all 8 pairs.
   - Both models are truncated together, so this does not say which model's PC is private.
   - In W_vit, random directions beat random coordinates (m = 64: 0.70 vs 0.48). This comes from the early-layer fixed cells: at the last layer only MAE differs (self 0.40 vs 0.49). One draw of directions.

**Caveats:**

- n = 1024. The CKA chance level and all excess values depend on it.
- Excess is a difference, not a normalised score, so relative curves compare different null levels.
- Pairs share models, so pair counts are not independent evidence, and H-T3 has only 4 LLM sizes.
- m is absolute and both models are truncated with the same m.
- m90 for CKA is censored at the lowest m (2), and CKA m90 / d50 use the max-cell excess, so read CKA from the fixed-cell tables.
- Two draws of coordinates and one of directions.
- The fixed cell is picked on the untruncated data, which favours the untruncated point slightly.

## Next

- **One-model-only truncation on SigLIP and CLIP ViT-H**, to find which model's leading PCs are private, then look at what they encode. This is the one result here that could raise cross-modal alignment.
- **Debiased linear CKA** on the same settings, to confirm the CKA gain goes away (finding 2).
- **Do not scale to n = 10240 (`d2l`) for this question.** That was conditional on a caption effect on the band, and d2s shows a caption effect on the level only.

## Commands

D2 run (CPU pod, 16 vCPU or more, volume attached; about 1.9 h on cpu3g 16 vCPU; `run_exp019.sh` uses 200 permutations, 2 draws of random coordinates, 1 of random projections, 9 layers per model, and terminates the pod 30 min after the job ends so results can be copied):

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp019.sh > /workspace/results/oddharak/exp-019.log 2>&1 < /dev/null &"
scp -P <port> -r root@<ip>:/workspace/results/oddharak/exp-019-truncation/d2s results/exp-019-truncation/
uv run python experiments/exp-019-truncation/bands.py --root results/exp-019-truncation/d2s
```

Local:

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
