# exp-026: what helps the LM above the lexical floor? Masking word classes, across depth

**Status:** done, 2026-11-07 · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-026-word-masking/` (copy in local `results/exp-026-word-masking/`) · **Code:** `masks.py`, `run.py`, `analyze.py`, `tests/test_word_masks.py`, uncommitted, on `ef2baa9`

## Idea

Exp-024 masked word classes in TF-IDF only. Here the masked captions go through trained LMs, and every hidden layer is scored against the ViT's 7 stored layers. Comparing a condition's depth curve with its own TF-IDF floor shows what the LM adds, and which input it needs to add it.

## Setup

- **LMs:** open_llama_7b, bloomz-7b1 (no Pythia). **ViTs:** DINOv2, CLIP. That gives 4 pairs. n = 10240 WIT images, Gemini and original captions.
- **Word classes:** spaCy `en_core_web_sm`. Function words are DET, ADP, PRON, AUX, CCONJ, SCONJ, PART.
- **Conditions:**
  - `intact`, and `shuffled` (all tokens);
  - `nofunc`, `nofunc_shuf`;
  - `nouns`, `nouns_shuf` (Gemini only: 2925 original captions have no nouns);
  - `nonouns`;
  - random controls that remove or keep the same number of words per caption: `ctrl_nofunc`, `ctrl_nonouns`, `ctrl_nouns`.
- **Measures:** excess over the exact null for mutual kNN (k = 10, 100, 1000) and linear CKA, with PRH preprocessing. At each LM layer we take the best ViT layer. Curves use a relative-depth grid and bands are min..max over pairs.
- **Floor:** TF-IDF of the same masked tokens. A shuffled condition shares its base's floor.

## Results

Best LM layer, mean over pairs. Tables: `summary.md`; figures: `fig_masking_gemini.png`, `fig_masking_orig.png`.

| Gemini | Floor k = 10 | Layer 0 | Best | Gain | Floor CKA | Layer 0 | Best | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Intact | 0.184 | 0.183 | 0.254 | 0.071 | 0.561 | 0.557 | 0.672 | 0.111 |
| All tokens shuffled | 0.184 | 0.183 | 0.195 | 0.012 | 0.561 | 0.556 | 0.631 | 0.070 |
| No function words | 0.204 | 0.199 | 0.258 | 0.055 | 0.575 | 0.572 | 0.678 | 0.104 |
| No function words, shuffled | 0.204 | 0.199 | 0.203 | -0.001 | 0.575 | 0.572 | 0.623 | 0.048 |
| Nouns only | 0.201 | 0.200 | 0.217 | 0.016 | 0.580 | 0.611 | 0.658 | 0.078 |
| Nouns only, shuffled | 0.201 | 0.200 | 0.201 | 0.001 | 0.580 | 0.611 | 0.616 | 0.036 |
| No nouns | 0.064 | 0.075 | 0.173 | 0.109 | 0.441 | 0.411 | 0.622 | 0.181 |

| Original | Floor k = 10 | Layer 0 | Best | Gain | Floor CKA | Layer 0 | Best | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Intact | 0.034 | 0.027 | 0.083 | 0.049 | 0.087 | 0.099 | 0.353 | 0.266 |
| All tokens shuffled | 0.034 | 0.027 | 0.034 | -0.000 | 0.087 | 0.099 | 0.172 | 0.086 |
| No function words | 0.033 | 0.029 | 0.078 | 0.045 | 0.077 | 0.098 | 0.321 | 0.243 |
| No function words, shuffled | 0.033 | 0.029 | 0.034 | 0.001 | 0.077 | 0.098 | 0.178 | 0.100 |
| No nouns | 0.021 | 0.019 | 0.059 | 0.038 | 0.063 | 0.076 | 0.250 | 0.188 |

**Order among content words accounts for most of the gain, and function words aren't needed.** This is the first reading rule set in advance: `nofunc` is near `intact` and above both `nofunc_shuf` and its random control.

- **Shuffling removes the gain.** At k = 10 the gain falls to about zero in both caption sets, except 0.012 for Gemini with all tokens shuffled. `intact` minus `shuffled` is positive in all 4 pairs: +0.035 to +0.076 (Gemini), +0.031 to +0.070 (original). CKA and k = 1000 keep part of the gain after shuffling, so coarse structure needs order less.
- **Removing function words costs little.** Gemini `nofunc` minus `intact` is -0.000 to +0.009 at k = 10, and -0.001 to +0.013 in CKA. On short original captions it costs a little: about -0.005 at k = 10, -0.03 in CKA. `nofunc` also beats random removal of the same size in all 4 pairs on every measure and caption set.
- **Function words act as noise at the word level only.** Removing them lifts the Gemini floor (0.184 to 0.204), but the LM's best layer stays the same (0.254 vs 0.258). The deep layers appear to discount them already.
- **Nouns set the floor; the rest supplies much of the gain.**
  - Nouns alone reach the intact floor with almost no help from the LM (k = 10 gain 0.016), and stay 0.029 to 0.043 below `intact`.
  - Without nouns, the floor collapses to 0.064, and the LM adds the largest gain of any Gemini condition (0.109, to 0.173). That still sits 0.073 to 0.090 below `intact`, and 0.062 to 0.071 below random removal.
  - This is the only class whose removal is worse than its control. On original captions the gap at k = 10 is small (-0.007 to -0.008).

**Depth.**

- Best layers are late, at relative depth 0.93 to 0.97 for intact Gemini at k = 10, and 0.84 to 0.93 for original captions. Depth 0.83 was only a storage position, and it would have missed the peak slightly.
- The curves are not monotonic on Gemini. At k = 10 intact sits near its floor up to about depth 0.55 (0.190 at depth 0.5, floor 0.184), and all of its gain comes in the last 40% or so.
- Original captions rise steadily from about depth 0.1 and drop at the very last layer (0.083 best, 0.073 last).
- On shuffled input, depth actively hurts at k = 10 on Gemini: 0.183 at layer 0, 0.096 at depth 0.5, 0.177 at the end. So the best layer for shuffled captions is sometimes the embedding layer.
- **Layer 0 is not the same as TF-IDF on every measure.** It matches at k = 10, but on Gemini at k = 1000 it is above (0.299 vs 0.230), so learned word similarity helps coarse neighbourhoods before any context is used.

## Checks

- **Stored scores:** shuffled Gemini reproduces exp-022's stored grid: largest CKA difference 7.6e-06, 10-NN difference 8.4e-05 (open_llama); 6.4e-06 and 7.0e-05 (bloomz).
- **Shuffle at layer 0:** should change nothing, and the largest difference is 0.0022. It appears only for bloomz, only where the clamp threshold differs between the two conditions (threshold 2.71 vs 1.65 gives the largest; 1.8858 vs 1.8857 gives 0). Open_llama's differences are exactly 0.
- **Rows:** all 116 present, with matching md5 on the pod and locally. `tests/test_word_masks.py` passes.

## Limits

- **Inputs are out of distribution.** Masked and shuffled captions are unnatural for the LM, so all drops are upper bounds on what the removed thing contributes. `nofunc` is also ungrammatical yet loses nothing, so being OOD alone doesn't explain the shuffle's damage.
- **"Order" isn't cleanly isolated.** The shuffle is at token level, so it also breaks multi-token words and local phrases, not only sentence order. `nofunc` keeps adjacency, so the result can't separate syntax-free phrase context from longer-range order.
- **Sample of models and measures.** Two LMs and two ViTs, chosen as the better large models. Bands are spread over 4 non-independent pairs, not confidence intervals. Random controls use a single seed.
- **Selection.** "Best" takes a maximum over all LM and ViT layers, while floors take a maximum over ViT layers only. This favours the LM slightly. The shuffled curves are held to the same rule.
- **Gemini saw the images.** Its captions can carry visual detail that a human caption would not. Effects should be read per caption set.
- **Clamp differs by condition.** The threshold is computed over each condition's own layers and shuffled inputs have much larger thresholds on Gemini, so conditions are not preprocessed identically. The layer-0 check bounds this at 0.0022 there, but it isn't measured at deeper layers.

## Run

Pod: RTX PRO 4500 Blackwell, $0.72/h, EU-RO-1. Total about 3.2 h. Terminated and verified.

    uv run python experiments/exp-026-word-masking/run.py --out <dir> --exp016 <dir> --exp018 <dir> --scratch <dir> --stages mask --tag-processes 20
    uv run python experiments/exp-026-word-masking/run.py --out <dir> --exp016 <dir> --exp018 <dir> --scratch <dir> --stages score --grids <big_shuffle_grids.json>
    uv run python experiments/exp-026-word-masking/analyze.py --out results/exp-026-word-masking
