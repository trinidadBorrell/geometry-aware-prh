# exp-015: matched pooling across modalities (Phase B)

**Status:** done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-015-matched-pooling/`, new activations in `/workspace/activations/` (`..._pool-last_cid-0`, ViT `..._pool-avg`)

## Idea

The PRH cache pairs LLM features mean-pooled over the caption tokens with ViT CLS-token features.
exp-010 found opposite intrinsic-dimension (ID) profiles over depth for LLMs (falling) and ViTs
(rising), and exp-012 / exp-014 compare the two modalities, but the two kinds of summary may
themselves differ in geometry: a mean over tokens is a smoothed bag of tokens (layer 0 is a
bag-of-words average), a single token is a learned summary. Cross-modal comparisons should
use the same kind of summary on both sides.

Two matched pairs:
- **one summary token:** LLM last token (new) x ViT CLS (cached)
- **mean of tokens:** LLM mean (cached) x ViT mean of patch tokens (new)

## Hypothesis

- Part of the LLM / ViT difference in ID profile is pooling: LLM last-token features have a
  lower ID at layer 0 than mean-pooled ones (a single vocabulary embedding vs an average), and
  ViT patch means differ from CLS.
- Cross-modal alignment (mutual kNN, per-sample ORC agreement) depends on the matching: if
  pooling is a confound, matched pairs align differently from the mixed PRH pairing.

## Method

- `extract.py`: platonic-rep recipe (as exp-006): every LLM hidden state, platonic-rep
  `load_llm` / `load_tokenizer` (bf16, left padding, so position -1 is the last real token);
  timm ViT `blocks.{i}.add_1`, model eval transform; ViT patch mean skips the prefix tokens
  (CLS, registers). `minhuh/prh` revision `wit_1024`, caption 0.
- Checks (local, `--check` and ad hoc):
  - the code reproduces the cache: ViT CLS identical (max relative error 0); LLM mean median
    relative error 0, max 6e-3 (bf16, different batch composition);
  - ViT patch mean equals timm's own forward through the blocks, patch tokens averaged (max
    relative error 0);
  - LLM last token: the same caption in a padded batch vs alone differs by 2-10% relative
    (also for the longest, unpadded caption: bf16 numerics that depend on batch shape, not
    padding). Neighbourhoods are unaffected: mutual kNN (k = 10) between bf16 and fp32
    last-token features of BLOOMZ-560m is 0.96-0.99 at every layer. bf16 kept, as the cache.
- Per-layer ORC, density and TwoNN ID on the new poolings with exp-014's `compute.py`
  (`--llm-pool last --vit-pool avg`), k = 10, 30.

## Commands

```bash
uv run python experiments/exp-015-matched-pooling/extract.py --models bigscience/bloomz-560m --pools avg --check --root results/activations --limit 256
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=30 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp015.sh > /workspace/results/oddharak/exp-015.log 2>&1 < /dev/null &"
```

## Results

Run 2026-11-01 on one RTX PRO 6000 pod (32 vCPU), commits `fc3e747` (first attempt) to `7f2ce8a`.
The first attempt's extraction failed (the pod enables `hf_transfer` without the package) and a
plain download later stalled; the rerun (`run_exp015b.sh`) extracts one model per process with a
timeout and retries, installs `hf_transfer` (~500 MB/s), and deletes each newly downloaded
model's weights as soon as its features are cached, leaving the shared HF cache at its
pre-Phase-B 24 GB. Outputs: new activations in `/workspace/activations/` (LLM `pool-last`, ViT
`pool-avg`, 27 models), `/workspace/results/oddharak/exp-015-matched-pooling/` (`orc/`,
`align_token/`, `align_mean/`), copied to `results/exp-015-matched-pooling/` with `summary.txt`,
`orc_conditions.json`, fig1-fig3 (`analyze.py`). About 3 h of pod time (~$7, including the stall).

Pairings: **mixed** = LLM mean x ViT CLS (the PRH default), **token** = LLM last x ViT CLS,
**mean** = LLM mean x ViT patch mean.

**1. Pooling explains most of the LLM / ViT difference in ID profile** (`fig1_id`, TwoNN, PRH
features, n = 1024). Family means at relative depth 0 / 0.5 / 1:

| features | depth 0 | 0.5 | 1 |
|---|---|---|---|
| LLM mean-pooled (PRH) | 47.7 | 27.9 | 26.5 |
| LLM last token | 28.9 | 16.3 | 19.3 |
| ViT CLS (PRH) | 10.8 | 19.0 | 20.2 |
| ViT patch mean | 13.0 | 20.0 | 20.2 |

- With last-token pooling the LLM ID drops to the ViT range, and after a collapse in the first
  layers (to ~6 at relative depth ~0.05, where the last tokens of captions ending in the same
  word nearly coincide) it **rises** with depth to ~12-22, as the ViT ID does. The falling LLM
  profile of exp-010 is the signature of mean pooling (layer 0 = a bag-of-words average with the
  highest ID), not of language models as such.
- ViT patch means have nearly the same ID profile as CLS.

**2. Matching the pooling does not increase cross-modal alignment** (`fig2_alignment`, 170 ViT x
LLM pairs, best layer pair, median):

| pairing | mutual kNN: S - null | linear CKA: S - null |
|---|---|---|
| mixed (PRH) | 0.116 | 0.262 |
| token | 0.074 | 0.199 |
| mean | 0.105 | 0.275 |

The mean pairing aligns about as well as the PRH pairing; the single-token pairing aligns less.
Matching the *kind* of summary on both sides is not what drives alignment here: LLM mean
pooling aligns better with either kind of ViT summary than the LLM last token does.

**3. Per-sample ORC agreement across modalities is weak under every pairing** (`fig3_orc`,
ORC given density, k = 10, best layer pair, median; null 95% threshold ~0.05):

| | best-cell Spearman | cells with p < 0.05 | density agreement |
|---|---|---|---|
| ViT x LLM, mixed | +0.16 | 29% | +0.28 |
| ViT x LLM, token | +0.10 | 10% | +0.21 |
| ViT x LLM, mean | +0.15 | 49% | +0.29 |
| LLM x LLM, mean / last | +0.60 / +0.66 | 92% / 88% | +0.84 / +0.95 |
| ViT x ViT, CLS / patch mean | +0.57 / +0.63 | 89% / 99% | +0.89 / +0.88 |

Within a modality, models agree strongly on which samples sit in more clustered / curved local
configurations, under either pooling; across modalities that agreement is a quarter of the size
(still above chance), and matching the pooling does not raise it. The same holds for local
density. Only the exact-duplicate caption pair is dropped; extreme ORC values from
near-identical points in early LLM last-token layers are kept (rank statistics), and undefined
values (points whose neighbours all coincide with them; up to 378 of 1024 samples in a layer
for the LLaMA / OpenLLaMA last token at the first layers) are set to the layer median.

### What this means

- The pooling confound was real for the *geometry* of each modality (ID profiles), but it does
  not explain the weak *cross-modal* agreement: with the same kind of summary on both sides,
  global (CKA), neighbourhood (mutual kNN) and per-sample local-structure (ORC) agreement stay
  far below within-modality agreement.
- So far this favours the Aristotelian reading in its weakest form: language and vision models
  share some local structure above chance, but much less than models of the same modality do,
  and neither a different ruler (exp-012 to exp-014) nor matched pooling closes the gap.
