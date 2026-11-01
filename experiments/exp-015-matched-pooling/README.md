# exp-015: matched pooling across modalities (Phase B)

**Status:** running · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-015-matched-pooling/`, new activations in `/workspace/activations/` (`..._pool-last_cid-0`, ViT `..._pool-avg`)

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

Pending.
