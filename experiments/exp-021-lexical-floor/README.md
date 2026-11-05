# exp-021: how much of LM x vision alignment is word counts?

**Status:** done, 2026-11-05; being checked in [exp-022](../exp-022-lexical-validation/README.md) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-021-lexical-floor/` (copy in local `results/exp-021-lexical-floor/`) · **Code:** `run.py` at `aac9fdf` plus this uncommitted folder; the exact script is saved as `run.py.txt` with the outputs.

## Idea

exp-018 found that a random-init Pythia, at layer 0 and mean-pooled, already reaches half of its final alignment with vision models. That layer is a random projection of each caption's token counts. So we score the token counts directly, with no network, on the same n = 10240 images. We also read the per-layer grids stored by exp-018 to separate the embedding table from the deep layers.

## Predictions stated before the run

From `reports/Embedding layer alignment with vision.md`, Gemini captions. Falsified if CKA is off by more than 0.02, or DINOv2 mutual 10-NN is <= 0.111 or > 0.16.

| Quantity | Predicted | Measured | Verdict |
|---|---:|---:|---|
| Bag-of-tokens CKA x DINOv2 | 0.42 | 0.415 | holds |
| Bag-of-tokens CKA x CLIP | 0.33 | 0.329 | holds |
| Bag-of-tokens CKA x pythia-1.4b | 0.72 | 0.691 | **fails** (off by 0.029) |
| Bag-of-tokens mutual 10-NN x DINOv2 | 0.127-0.14 | 0.120 | inside the falsification limits, below the predicted range |
| Participation ratio r_B | 83-97 | 68.9 | **outside the range** |

## Results

Numbers are excess over the permutation null at the best target layer unless marked as scores. Null means come from 30 permutations, enough for means, not for small p-values. Table seeds differ by up to 0.008 in CKA, so smaller differences are noise.

**1. Step 0 is word counts, for the vision targets.** A simulated Gaussian table with no network reproduces the real step-0 model:

| Against | Real step 0, d = 768 / 1024 (CKA) | Gaussian table, d = 768 / 1024 (CKA) | Real / Gaussian (mutual 10-NN, d = 768) |
|---|---:|---:|---:|
| DINOv2 | 0.385-0.396 / 0.396-0.401 | 0.390 / 0.398 | 0.105-0.106 / 0.105 |
| CLIP | 0.303-0.311 / 0.308-0.312 | 0.306 / 0.311 | 0.096-0.097 / 0.096 |
| pythia-1.4b | 0.671-0.678 / 0.682-0.695 | 0.656 / 0.666 | 0.316-0.317 / 0.316 |

- **Controls:** one fresh random vector per caption gives excess 0.000 on CKA and on mutual kNN at every k. Caption length alone gives CKA excess 0.010-0.026.
- **Open:** on pythia-1.4b, real step-0 CKA sits 0.01-0.03 above the Gaussian table. This is probably why the 0.72 extrapolation overshot; the cause is unknown. CKA between real layer 0 and the exact kernel is 0.91, below the 0.96-0.97 expected from the width formula.

**2. Width saturates as predicted.** Random table against DINOv2, with the exact kernel at 0.415 (CKA score) and 0.119 (mutual 10-NN excess):

| d | 64 | 256 | 768 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| CKA score | 0.303 | 0.366 | 0.395 | 0.403 | 0.414 | 0.416 |
| Ratio to exact | 0.73 | 0.88 | 0.95 | 0.97 | 1.00 | 1.00 |
| Formula A(d), measured r_B | 0.69 | 0.89 | 0.96 | 0.97 | 0.99 | 1.00 |
| Mutual 10-NN excess | 0.040 | 0.083 | 0.105 | 0.109 | 0.118 | 0.120 |

CLIP gives the same ratios within 0.01 for d >= 256. The formula holds within 0.01 for d >= 256 and underpredicts at d = 64. Mutual kNN saturates more slowly than CKA.

**3. On the original captions the floor nearly vanishes.** Gemini captions have 613 tokens (median), the original ones 12. Trained-LM rows are exp-016's bloomz-1b7, bloomz-7b1, open_llama_7b and llama-13b at n = 10240, not Pythia.

| x DINOv2 | Gemini mutual 10-NN | Gemini CKA | Original mutual 10-NN | Original CKA |
|---|---:|---:|---:|---:|
| Token counts (TF) | 0.119 | 0.411 | 0.020 | 0.063 |
| TF-IDF | 0.186 | 0.600 | 0.032 | 0.079 |
| Trained LMs (exp-016) | 0.234-0.280 | 0.713-0.778 | 0.055-0.089 | 0.278-0.355 |

Gemini captions were generated from the images and are long, so their word overlap can track image content far more closely than a 12-token caption can.

**4. TF-IDF nearly matches trained tables at small k and in CKA, but not at large k.** Gemini captions, final checkpoint, range over 4 Pythia series:

| Excess | TF | TF-IDF | Trained layer 0 | Best deep layer |
|---|---:|---:|---:|---:|
| DINOv2 mutual k = 10 | 0.119 | 0.186 | 0.188-0.192 | 0.190-0.218 |
| DINOv2 mutual k = 100 | 0.177 | 0.240 | 0.290-0.295 | 0.314-0.342 |
| DINOv2 mutual k = 1000 | 0.208 | 0.231 | 0.345-0.349 | 0.392-0.414 |
| DINOv2 CKA | 0.411 | 0.600 | 0.654-0.663 | 0.648-0.704 |
| CLIP mutual k = 10 | 0.109 | 0.176 | 0.171-0.174 | 0.184-0.216 |
| CLIP mutual k = 1000 | 0.192 | 0.225 | 0.304-0.307 | 0.336-0.359 |
| CLIP CKA | 0.319 | 0.507 | 0.488-0.496 | 0.534-0.568 |

- **Fine structure:** at k = 10, TF-IDF nearly matches the trained table against DINOv2 and slightly beats it against CLIP.
- **Coarse structure:** training adds most at large k, where word overlap is flat. This suggests that learned word similarity widens neighbourhoods beyond shared words (inference, not tested).
- **CKA:** TF-IDF sits 0.19 above TF against DINOv2, and trained layer 0 sits 0.05-0.06 above TF-IDF. These are level differences, not components. Whether trained tables actually implement this weighting is not tested.

**5. Trained layer 0 comes close to the deep layers, in all four series.** The earlier one-seed result partly generalises: layer 0 gets close, but it wins outright only in some cells. Layer 0 at the final checkpoint against the peak of any layer and checkpoint (levels only: CKA differences are not variance components):

| Layer 0 path, step 0 -> final (peak over all layers) | 160m | 160m-seed1 | 410m | 410m-seed1 |
|---|---:|---:|---:|---:|
| DINOv2 mutual 10-NN | 0.105 -> 0.192 (0.218) | 0.106 -> 0.190 (0.214) | 0.109 -> 0.189 (0.226) | 0.111 -> 0.188 (0.221) |
| DINOv2 CKA | 0.385 -> 0.661 (0.682) | 0.396 -> 0.663 (0.673) | 0.401 -> 0.658 (0.695) | 0.396 -> 0.654 (0.705) |

The table was trained through the deep network, so this measures what the table holds, not what it could learn alone.

**6. Random depth is not only signal loss.** At step 0, agreement of each layer with the TF kernel falls quickly (160m CKA: 0.91, 0.65, 0.41, 0.31, 0.25, 0.21, 0.18), but vision alignment falls more slowly (0.385, 0.353, 0.269, 0.223, 0.195, 0.178, 0.160). The last layer has about twice the vision CKA that TF retention alone would predict (0.160 vs 0.074). 410m and CLIP show the same gap. Either the random blocks reweight tokens towards something TF-IDF-like, or there is a real architecture signal. This run does not separate these.

## What this changes

- exp-018's "step 0 is half way" is a property of Gemini captions and word counts, not of an untrained network's layers.
- exp-016's "descriptive captions raise LM x ViT alignment" is at least partly a word-overlap effect: on those captions TF-IDF alone reaches mutual 10-NN 0.186 and CKA 0.600, against 0.234-0.280 and 0.713-0.778 for the trained LMs.
- A shuffled-pair null does not detect this. A lexical floor (TF-IDF at minimum) should be reported beside any caption-based alignment.

## Numerical bug found

`lc.build_cache` computed Gram norms in float32 (fixed for exp-022). On CPU this inflates CKA at large n: exact 0.3899 vs 0.399 here at n = 10240 (synthetic data: ratio 1.0225 at n = 10240, 1.0014 at n = 4096, 1.0000 at n = 1024). The stored GPU result matches the exact value at the one cell checked. `run.py` uses float64 for centring and the norm.

## Not done

- No Pythia features on original captions (exp-018 ran Gemini only), so result 3's trained rows come from other, larger models.
- TF-IDF retention through random depth, and row norm vs idf in trained tables.
- r_B and the 1.4b discrepancy above are unexplained.

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && nohup bash infra/runpod/run_exp021.sh > /workspace/results/oddharak/exp-021.log 2>&1 < /dev/null &"
```

Run: cpu3g 8 vCPU, 15 min for the kernels plus setup and two aborted starts, about 25 min of pod time at $0.32/h. The runner terminates the pod 30 minutes after the job ends.
