# exp-025: toy examples for the lexical floor

**Status:** done, 2026-11-07 · **Owner:** oddharak · **Outputs:** local `results/exp-025-lexical-toy/` · **Code:** `toy.py`, uncommitted

## Idea

Teaching examples for the write-up: show that word counts align with a vision encoder with no language model, no training and no real data. Both sides are functions of the same scene and share nothing else.

## Six captions

Exact bag-of-words cosines beside a random 2-layer tanh encoder that sees only the scene description. Same block pattern, mutual 1-NN 1.0, CKA 0.998. Dropping "a, on, the" leaves CKA unchanged, because centring removes their constant block. CKA's chance level at n = 6 is 0.62, so the score means little without its null.

## Simulation

1000 scenes, 5 seeds. 8 attributes a caption can mention and 8 it never mentions, 6 values each. A caption names m attributes plus 3 Zipf-distributed function words per content word.

Excess at m = 1, 2, 4, 8:

| Kernel | 10-NN | CKA |
|---|---:|---:|
| Content words only | 0.016, 0.018, 0.058, 0.228 | 0.071, 0.145, 0.300, 0.618 |
| TF-IDF | 0.012, 0.016, 0.034, 0.087 | 0.038, 0.082, 0.173, 0.347 |
| TF | 0.003, 0.007, 0.013, 0.030 | 0.015, 0.032, 0.065, 0.133 |
| TF x random table, width 768 | 0.003, 0.007, 0.013, 0.028 | 0.015, 0.032, 0.064, 0.129 |
| TF-IDF, pairing shuffled | 0.001, 0.000, -0.001, 0.000 | 0.001, 0.000, 0.000, 0.001 |

- More detail raises alignment with no model involved.
- A random table follows TF, as an untrained layer 0 should.
- A shuffled pairing removes it, so a permutation test correctly calls it real and cannot say where it came from.

Captions of 8 random words, unrelated to the scene: raw CKA with the encoder is 0.025, 0.079, 0.184, 0.248 at vocabularies of 10, 100, 1000, 10000, matching the exact null (0.025, 0.078, 0.184, 0.248). High-rank word kernels score high by chance.

## Limits

A hand-built world chosen to make the mechanism visible. It says nothing about the size of the effect on real captions. The encoder is random, not a trained ViT.

## Commands

```bash
uv run python experiments/exp-025-lexical-toy/toy.py --out results/exp-025-lexical-toy
```

Local, under 5 minutes. Figures: `fig_toy_kernels.png`, `fig_toy_sim.png`. Values: `toy.json`, `toy_sim.csv`, `toy_random_words.csv`.
