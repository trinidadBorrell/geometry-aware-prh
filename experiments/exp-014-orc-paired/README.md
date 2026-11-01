# exp-014: paired per-sample Ollivier-Ricci curvature within modality (Phase A)

**Status:** Phase A done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-014-orc-paired/`

## Idea

Raw Ollivier-Ricci curvature (ORC) values cannot be compared across models: exp-011 showed their
level is set mostly by the intrinsic dimension (ID) and the neighbourhood size k, and only at
matched ID, k and sampling does ORC rank a sphere above a flat torus (correctly, but by a small
amount). A comparison that does not need comparable levels: on the *same* samples, ask whether
the same samples get high or low ORC in both models. A rank correlation across samples ignores
any model- or layer-specific monotone offset (from ID, k, or the PRH normalisation), so it is an
alignment statistic built on local graph structure: do two models agree on *where* the data is
locally clustered, spread or curved?

ORC is also strongly coupled to local sampling density (exp-011: Spearman +0.5 to +0.7 within a
uniformly sampled manifold), so every ORC result is reported next to the same statistic for
density alone, and for ORC with density removed.

This is Phase A: within-modality tracks only (T1 LLM x LLM, T2 ViT x ViT), free of the pooling
confound (LLM mean-pooled captions vs ViT CLS tokens). Cross-modal comparisons (Phase B) wait for
a matched-pooling re-extraction.

## Hypothesis

- Within a family (e.g. BLOOMZ sizes), per-sample ORC agrees strongly across models at
  corresponding depths, beyond what density explains.
- If ORC adds information to the alignment picture, the layer-pair map of ORC agreement differs
  from the mutual kNN map of exp-012 (else it is a re-expression of neighbour overlap).
- Falsified (for our purposes) if ORC agreement is at the level of density agreement and its
  map duplicates mutual kNN.

## Method

- **Synthetic check** (`synthetic_check.py`, local): two noisy, independently rotated copies of
  the same 1024 samples of a manifold (ID 10, noise 0.3 x NN distance). Same samples and geometry:
  ORC rank correlation should be high; with the pairing shuffled, near 0.
- **Per-layer ORC** (`compute.py`, pod): PRH `wit_1024` activations of the 10 LLMs and 17 ViTs,
  PRH features (q = 0.95 clamp, l2 norm). Union kNN graph with Euclidean edges at fixed k = 10 and
  30 (the same for every model: PersonaManifold's k = 5 x ID would make k, and the ORC offset,
  depend on the ID). Node ORC with PersonaManifold's convention (`geometry.ollivier_ricci_exact`,
  exact transport, alpha = 0, node = mean over incident edges; equal to their `edge_curvature` to
  1e-9, tested). Local density = 1 / distance to the k-th neighbour. Flat-torus reference: mean
  node ORC of a uniform Clifford torus with the same n, k and ID (rounded TwoNN ID of the layer).
- **Paired statistics** (`analyze.py`, local; `geoprh.paired`): for every model pair and layer pair,
  Spearman across the 1024 samples of (a) node ORC, (b) density, (c) ORC with each layer's density
  removed on the rank scale (partial). Each cell calibrated against a 200-permutation null of the
  pairing (the same permutation for all of B's layers), as exp-012. Per pair: the best cell, the
  fraction of cells with p < 0.05 (5% by chance), and the Spearman across cells between each map
  and exp-012's null-corrected mutual kNN map.

## Commands

```bash
uv run python experiments/exp-014-orc-paired/synthetic_check.py
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=28 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp014.sh > /workspace/results/oddharak/exp-014.log 2>&1 < /dev/null &"
uv run python experiments/exp-014-orc-paired/analyze.py --root results/exp-014-orc-paired --exp012 results/exp-012-id-mismatch
```

## Results

### Synthetic check

Spearman across samples between the two copies (shuffled pairing in brackets):

| manifold | k | node ORC | density | ORC given density |
|---|---|---|---|---|
| flat cube | 10 | +0.95 (-0.05) | +0.99 (-0.02) | +0.94 (-0.05) |
| flat cube | 30 | +0.99 (+0.03) | +1.00 (+0.07) | +0.97 (+0.02) |
| sphere | 10 | +0.86 (-0.06) | +0.95 (+0.01) | +0.84 (-0.08) |
| sphere | 30 | +0.93 (+0.01) | +0.94 (+0.04) | +0.88 (-0.01) |
| clusters | 10 | +0.98 (+0.01) | +1.00 (+0.01) | +0.98 (-0.01) |
| clusters | 30 | +0.99 (+0.01) | +1.00 (+0.00) | +0.99 (+0.01) |

The statistic behaves as intended: same samples and geometry give 0.84-0.99, even after density
is removed (ORC carries per-sample structure beyond density), and a shuffled pairing gives ~0.

### Models

Run 2026-11-01, commit `e3be91c`, RTX PRO 4500 pod (28 vCPU), 9 min for 1328 layer tasks (27
models x layers x k = 10, 30) and 92 torus references. Outputs in
`/workspace/results/oddharak/exp-014-orc-paired/` (`models/*.npz`, `torus_ref.json`), copied to
`results/exp-014-orc-paired/` with the analysis (`pairs/`, `pairs.jsonl`, `summary.txt`, fig1-fig3).
Two samples (493, 531) are an exact-duplicate caption pair in `wit_1024`: in LLM layers their edge
length is ~0 and kappa = 1 - W1 / d reaches ~-1e5, so they are dropped from all models (1022
samples remain; the results are unchanged to the second decimal).

Median over pairs of the best layer pair's Spearman across samples (null 95% threshold about
0.05 everywhere; fraction of all layer-pair cells with p < 0.05 in brackets; Spearman of the
ORC | density layer-pair map with exp-012's mutual kNN map in the last column):

| track | k | node ORC | density | ORC given density | map vs mutual kNN |
|---|---|---|---|---|---|
| T1 LLM x LLM (45) | 10 | 0.61 (0.88) | 0.84 (0.89) | 0.60 (0.92) | 0.96 |
| T1 | 30 | 0.68 (0.85) | 0.85 (0.87) | 0.68 (0.88) | 0.93 |
| T2 ViT x ViT (136) | 10 | 0.61 (0.98) | 0.89 (0.95) | 0.57 (0.88) | 0.85 |
| T2 | 30 | 0.72 (0.94) | 0.91 (0.95) | 0.64 (0.90) | 0.86 |

Within vs across families (ORC given density, best cell, k = 10 / 30): T1 within (BLOOMZ sizes,
LLaMA sizes, OpenLLaMA sizes; 14 pairs) 0.77 / 0.84, across 0.56 / 0.62; T2 within (same training
objective; 21 pairs) 0.70 / 0.78, across 0.55 / 0.61.

**1. Models agree strongly on which samples have high or low ORC, beyond density.** The best layer
pair reaches 0.57-0.72 (null ~0.05), and removing each model's own density barely lowers it
(0.57-0.68): ORC carries per-sample structure that density does not. Agreement is higher within a
family than across (0.70-0.84 vs 0.55-0.62), as alignment should be.

**2. Density agrees even more (0.84-0.91).** Which samples sit in dense regions is the most shared
property of all; ORC is a weaker but partly independent signal.

**3. The layer-pair pattern of ORC agreement repeats mutual kNN.** Across layer pairs, the ORC
(given density) map correlates 0.85-0.96 with the mutual kNN map of exp-012 (`fig2_examples`:
same block structure, same late-layer and early-layer blocks). As a *layer-level* alignment
score, ORC agreement adds little to mutual kNN; whatever it adds is at the level of individual
samples, which this analysis does not yet separate from neighbour overlap.

**4. ORC levels (descriptive, `fig3_levels`).** Median node ORC is 0.4-0.85 above a uniformly
sampled flat torus of the same ID, n and k in every layer of every model: the representations are
much more locally clustered than uniform sampling. Over depth, LLMs fall from ~0.6-0.85 to ~0.5
and rise at the last layer; ViTs rise to ~0.6-0.75 in the middle and drop in the last blocks.
These levels are not comparable across modalities (pooling confound) and mix curvature with
density and grouping (exp-011), so they are descriptive only.

### Decision gate for Phase B

The gate asked for ORC-beyond-density agreement clearly above the null *and* not a copy of
mutual kNN. The first holds (0.57-0.68 vs ~0.05); the second fails at the layer-pair level (map
correlation 0.85-0.96). Open question before Phase B: at the level of individual samples, does
ORC agreement add anything beyond each sample's neighbour overlap between the two models (e.g.
partial correlation of ORC agreement with per-sample mutual-kNN overlap)? That is a cheap local
analysis on the existing outputs. The matched-pooling re-extraction (Phase B steps 5-7) is
worth doing for the pooling confound in exp-010 / exp-012 regardless of ORC.

### Per-sample check: does ORC agree beyond shared neighbours?

`per_sample.py`, run on the pod for all 181 T1 / T2 pairs (2026-11-01, commit `fc3e747`; a first
local run on 9 pairs gave the same picture). At each pair's best layer pair (by ORC given
density), per sample s: o(s) = share of s's k nearest neighbours the two models have in common
(the per-sample form of mutual kNN). Then (i) ORC agreement given density, (ii) the same with o(s)
also removed from both sides, (iii) ORC-given-density agreement within strata of o(s), each against
a within-stratum permutation null (200 shuffles). Output `per_sample.txt` / `.jsonl`.

Medians over pairs (a stratum counts only for pairs with >= 30 samples in it):

| | k = 10 | k = 30 |
|---|---|---|
| T1 (45): ORC given density -> also given overlap | 0.60 -> 0.54 (90% kept) | 0.68 -> 0.62 (92%) |
| T2 (136): ORC given density -> also given overlap | 0.57 -> 0.52 (91%) | 0.64 -> 0.56 (88%) |
| low overlap 0 < o <= 0.3: agreement (pairs; share significant) | T1 +0.34 (35; 100%), T2 +0.37 (126; 98%) | T1 +0.23 (25; 56%), T2 +0.24 (76; 66%) |
| no overlap o = 0 | T1 +0.14 (2; 0%), T2 +0.06 (9; 22%) | too few samples |
| high overlap o > 0.3 | T1 +0.64, T2 +0.61 (all significant) | T1 +0.69, T2 +0.65 |

Within-family pairs keep 93-95% of their ORC agreement after removing overlap, across-family
pairs 87-91%.

- **Most of the ORC agreement survives removing neighbour overlap** (about 90% of the
  correlation), and it persists, at about half strength, among samples whose neighbourhoods are
  mostly different (0 < o <= 0.3, significant in nearly all pairs at k = 10).
- **But it fades where the models share no neighbours at all** (o = 0: +0.06 to +0.14, rarely
  significant, and few pairs have enough such samples). So per-sample ORC adds information on
  top of partially shared neighbourhoods; it is not an independent channel of agreement.
- Combined with result 3: the layer-pair pattern of ORC agreement follows mutual kNN; per sample,
  ORC refines it (how clustered / curved a sample's neighbourhood is) rather than replacing it.
- Caveat: best layer pairs were chosen on ORC agreement, which favours ORC in absolute terms; the
  full vs partial comparison is on the same cells.

### ORC across neighbourhood sizes (k-sweep)

`k_sweep.py` (pod, same run): BLOOMZ-7b1, LLaMA-13b, DINOv2-giant, CLIP-huge at relative depths
0.1 / 0.5 / 0.9, k = 5, 10, 20, 50, 100, 200 (n = 1024, 2 duplicate samples dropped). Raw ORC rises
with k for any data (neighbourhoods overlap more; a complete graph has kappa -> 1), so two
readouts: the gap between the layer's median node ORC and a uniform flat torus with the same n,
k and ID; and the agreement (ORC given density) between the two LLMs and between the two ViTs at
the same relative depth. Output `k_sweep/k_sweep.json`.

**Gap to the flat reference** (range over the 12 layers):

| k | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| median ORC - flat torus | +0.50 to +0.76 | +0.50 to +0.74 | +0.29 to +0.51 | +0.15 to +0.28 | +0.08 to +0.17 | +0.04 to +0.18 |

All four models are far more locally clustered than uniform sampling at small scales; the
excess fades steadily and is mostly gone by k ~ 100 (10% of the data). The early LLaMA layer
keeps the largest excess at large k (+0.17 to +0.18). The curves are similar across models and
modalities, so this measures the scale of the data's clustering and density heterogeneity (its
"granularity"), not a curvature radius: exp-011 / exp-013 showed smooth curvature moves ORC by
~0.1 at k = 10 and ~0 at large k.

**Agreement by k** (ORC given density / density, Spearman across samples):

| pair, depth | k = 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| BLOOMZ-7b1 x LLaMA-13b, 0.1 | +0.03 / -0.54 | -0.06 / -0.65 | -0.14 / -0.71 | -0.07 / -0.77 | +0.06 / -0.79 | -0.08 / -0.76 |
| BLOOMZ-7b1 x LLaMA-13b, 0.5 | +0.36 / +0.71 | +0.50 / +0.71 | +0.56 / +0.69 | +0.53 / +0.66 | +0.52 / +0.66 | +0.57 / +0.66 |
| BLOOMZ-7b1 x LLaMA-13b, 0.9 | +0.41 / +0.77 | +0.55 / +0.79 | +0.62 / +0.78 | +0.62 / +0.76 | +0.61 / +0.73 | +0.52 / +0.69 |
| DINOv2-g x CLIP-h, 0.1 | +0.40 / +0.84 | +0.52 / +0.87 | +0.58 / +0.90 | +0.63 / +0.92 | +0.59 / +0.93 | +0.45 / +0.92 |
| DINOv2-g x CLIP-h, 0.5 | +0.14 / +0.67 | +0.18 / +0.68 | +0.12 / +0.69 | +0.08 / +0.72 | +0.01 / +0.74 | +0.12 / +0.74 |
| DINOv2-g x CLIP-h, 0.9 | +0.23 / +0.48 | +0.22 / +0.44 | +0.17 / +0.39 | +0.09 / +0.23 | +0.09 / +0.22 | +0.05 / +0.20 |

- Where the models agree, ORC agreement is lowest at k = 5 and plateaus at k ~ 20-100 (2-10% of
  the data): that is the neighbourhood scale at which per-sample structure is shared.
- Agreement depends on the layer pair much more than on k (same-depth layers are not always the
  best-aligned pair; exp-012 / exp-014 find the best ViT pairs early).
- Early-layer BLOOMZ and LLaMA *anti*-agree on density (-0.54 to -0.79): captions that are dense in
  one model's embedding layer are sparse in the other's, presumably from their different
  tokenizers and vocabularies.
