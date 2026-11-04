# exp-020: does a language model's training data set its alignment with vision? (pilot)

**Status:** code ready, not run · **Note:** first planned as exp-017; renumbered because exp-017 (map complexity) and exp-019 (truncation) are taken · **Owner:** oddharak · **Outputs (planned):** `/workspace/results/oddharak/exp-020-lm-data-pilot/`

## Idea

Published results disagree about what makes a language model align with vision models:

- **Huh et al. (2024):** alignment grows with LM capability.
- **Han et al. (*Learning to See Before Seeing*, ICLR 2026):** alignment grows with the share of reasoning data (code, math) in pretraining, peaking at 75%. This comes from LMs trained from scratch, with uncalibrated mutual kNN (k = 20) on short WIT captions, as a side analysis.
- **Koepke et al. (2026):** newer, stronger LLMs, which are trained on much more code, math and synthetic data, do not align better.
- **Balestriero and LeCun (2026):** alignment is set by "views". Text aligns with images to the extent that it is an observation of the same scenes.

The cheapest test uses existing models that share a base and differ only in continued pretraining. This pilot uses the smallest such sets, forward passes only, against the ViT features already cached by exp-016.

## Hypotheses and what each predicts

| hypothesis | what drives alignment | code / math continuation vs base | VLM-trained LM vs its base |
|---|---|---|---|
| H-capability | general modelling quality | follows generic perplexity | follows generic perplexity |
| H-structure (Han et al.) | reasoning-structured data | **higher** | not specified |
| H-views (Balestriero and LeCun) | text that describes visible scenes | **lower or equal** | **higher** |

## Models (all <= 1.5B; base -> continuation from the same weights)

| set | base | continuation(s) | role |
|---|---|---|---|
| Q0.5 | `Qwen/Qwen2.5-0.5B` | `Qwen/Qwen2.5-Coder-0.5B` | code (the Qwen2.5-Coder report: ~5.5T tokens, mostly code, plus text and math; verify the mix) |
| Q1.5 | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-Coder-1.5B`, `Qwen/Qwen2.5-Math-1.5B` | code; math (Qwen Math Corpus v2) |
| V0.5 (positive control) | `Qwen/Qwen2-0.5B-Instruct` | language model of `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | LM trained further with images in context; its `text_config` names `Qwen/Qwen2-0.5B-Instruct` as the starting point (checked) |
| V0.1 (positive control) | `HuggingFaceTB/SmolLM2-135M-Instruct` | language model of `HuggingFaceTB/SmolVLM-256M-Instruct` | the same, smaller; the model card lists `SmolLM2-135M-Instruct` as its base (checked) |

All are on the Hugging Face hub and not gated (checked 2026-11-04). The 7B step comes only if the pilot passes: `meta-llama/Llama-2-7b-hf` (gated, request access first) -> `codellama/CodeLlama-7b-hf` -> `EleutherAI/llemma_7b`.

The VLM language models are run on text only (captions, no image tokens). That measures what image-text training did to the LM's text representations.

## Data and vision side (cache first)

- **D2 from exp-016:** 10240 WIT-1M images with the original and the Gemini caption (`/workspace/results/oddharak/exp-016-positive-controls/d2.parquet`).
- **ViT features already on the volume** (exp-016 `features/*__cls__d2s.pt`, `__d2l.pt`): AugReg ViT-L, MAE ViT-H, DINOv2-g, CLIP ViT-H. No new image extraction is needed.
- **Human dense captions (DOCCI) are not in the pilot.** They follow if the pilot passes, because the Gemini captions may carry Gemini's own visual encoding (the leakage concern Balestriero and LeCun raise for LLaVA captions).

## Method

1. **Extraction** (GPU, `extract_lm.py`; also used by exp-018 with `--revisions`):
   - every hidden state, mean-pooled over the caption (right padding, no BOS, float32 accumulation), for both caption sets;
   - saved as `d2s` (1024 rows, all layers) and `d2l` (10240 rows, 7 layers), as exp-016;
   - for the VLMs, the language model submodule (`model.model.language_model` / `text_model`) on tokenised text only;
   - weights download to the container disk and are deleted after each model (the shared HF cache and the volume quota are untouched).
2. **Capability covariates**, so capability is not confused with alignment: bits per byte (every text preceded by one BOS / EOS context token) on
   - a fixed generic text: the first 1 MB of WikiText-103 test, in 4000-character chunks;
   - the captions themselves: the first 2048 original and 256 Gemini captions (familiarity with descriptive text).
3. **Measures** (`align.py`, reusing exp-016's `align_pair`, `rank_pair` and geometry stage, and `geoprh.knn_sweep`):
   - mutual kNN at k = 10 and at k = n/100;
   - linear CKA;
   - ORC given density and density agreement at n = 1024;
   - each cell calibrated with 200 permutations; the best cell against the null of the max; plus a fixed relative depth (0.75) as a selection-free readout.
4. **Matched-pair contrast.** For each (base, continuation, ViT, caption set): the difference of calibrated excesses, with a paired bootstrap over samples (both models resampled with the same indices; neighbourhoods fixed, as exp-016's CI). Mutual kNN from the per-sample overlaps, CKA recomputed on the resampled rows. Reported at each model's own best cell and at the fixed cell.
5. **Context: LLM x LLM agreement between base and continuation.** How far did continued pretraining move the representation at all? A difference in alignment means little if the representation barely changed.

## Decision gate (written before running)

**Proceed to the causal continued-pretraining phase if all three hold:**

1. The positive controls (V0.5, V0.1) raise calibrated mutual kNN with the ViTs by more than 3 bootstrap SE. Otherwise this measurement cannot detect training-data effects on the LM side, and the project stops here.
2. Code / math minus base exceeds 3 SE in the same direction for both Qwen sizes and for at least 3 of the 4 ViTs, with original or Gemini captions.
3. The direction is not explained by generic bits per byte: the continuation is not simply worse or better at language overall.

**Outcomes:**

- **Code / math higher:** supports H-structure, with calibration and descriptive captions that Han et al. lacked.
- **Lower:** supports H-views.
- **Equal while the positive control moves:** the LM's text domain does not matter at this scale.

That last outcome is also worth reporting, because it contradicts Han et al.

## Cost (to be priced live before creating a pod)

- **Extraction:** 9 small LMs x 20480 captions (Gemini captions ~650 tokens), about 1 h on an RTX PRO 4000 / 4500 class GPU. Downloads (~12 GB) go to the container disk; features on the volume ~3 GB.
- **Measurement:** 144 LM x ViT pairs + 20 base x continuation pairs (n = 1024 with ORC, n = 10240 with k = 10 and k = n/100), 80 contrasts. exp-016 took ~30-40 s per n = 10240 pair on a 24 GB GPU, so about 2-3 h in total.
- For scale: the exp-016 RTX PRO 4000 ran at $0.57/h.
- One pod at a time, with self-termination as in exp-016.

## If the gate passes: phase 2 (sketch)

- **Base:** continued pretraining of `Qwen/Qwen2.5-0.5B` (or Pythia-410M, to share exp-018's model) with equal token budgets (~300M) per arm.
- **Arms:**
  - human-written visual descriptions (Localized Narratives, Visual Genome region descriptions; no image overlap with D2 or DOCCI);
  - the same descriptions with their nouns shuffled (style kept, grounding destroyed);
  - fiction;
  - code;
  - math;
  - generic web (the forgetting control).
- **Runs:** 2 seeds per arm; alignment tracked over steps with exp-018's pipeline.

## Commands

```bash
uv run pytest tests/test_knn_sweep.py
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=16 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp020.sh > /workspace/results/oddharak/exp-020.log 2>&1 < /dev/null &"
# pieces, as the runner calls them:
uv run python experiments/exp-020-lm-data-pilot/extract_lm.py --out <out> --d2 <exp016>/d2.parquet --models Qwen/Qwen2.5-0.5B --cache-dir /root/hf-scratch
uv run python experiments/exp-020-lm-data-pilot/align.py --roots <out>,<exp016 out> --stages geometry,pairs,contrasts
uv run python experiments/exp-020-lm-data-pilot/analyze.py --out <out>
```

Outputs: `features/`, `bpb/`, `geometry/`, `pairs.jsonl` + `pairs/*.npz`, `contrasts.jsonl`, `summary.md` (tables and the gate), `fig_contrasts`.

Checked locally (2026-11-04) on tiny inputs: the k sweep equals `layer_calibration`'s mutual kNN at every k with and without a permutation (`tests/test_knn_sweep.py`), the bootstrap CKA equals `align_pair`'s CKA, and extraction runs for a causal LM, Pythia checkpoints and the SmolVLM text model.

## Risks

- **Continued pretraining changes many things at once:** tokens, data mix, sometimes context length. The positive control and the LLM x LLM shift bound the interpretation; phase 2 is the causal test.
- **"Reading" vs "seeing":** a model trained on descriptive text may represent descriptive captions better without encoding anything visual. Address with the noun-shuffled arm (phase 2), caption-familiarity bits per byte, and both caption sets.
- **Small effects:** model-to-model differences in exp-016 were ~0.02-0.05 mutual kNN, with bootstrap CIs of ~±0.01 at n = 1024. Matched pairs and n = 10240 are needed.
