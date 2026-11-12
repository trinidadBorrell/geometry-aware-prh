# exp-024: which image types and word classes carry alignment, and when?

**Status:** done, 2026-11-07 · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-024-emergence-words/` (copy in local `results/exp-024-emergence-words/`) · **Code:** `run.py`, `analyze_words.py`, `analyze_typed.py`, uncommitted, on `ef2baa9`

## Idea

1. **Word classes.** TF-IDF kernels using only or leaving out one word class (spaCy POS, four fixed lexicons), against frequency-matched random word sets.
2. **Image types over training.** Cluster images with DINOv2 and CLIP last layers (K = 5, 20, 100), score each on the other ViT's types, and follow Pythia checkpoints: between-type share, purity, within-type kNN, CKA between and within types.

## Word classes

n = 10240 WIT images. Excess over the exact null, best of the ViT's last 3 stored layers. Tables: `words_summary.md`; figure: `fig_words.png`. The all-words kernel reproduces exp-022's floor independently (k = 10 0.186, CKA 0.61, DINOv2, Gemini).

| Kernel, DINOv2 | Gemini k = 10 | Gemini k = 1000 | Gemini CKA | Original k = 10 | Original CKA |
|---|---:|---:|---:|---:|---:|
| All words | 0.185 | 0.236 | 0.598 | 0.039 | 0.112 |
| Only nouns | **0.211** | **0.373** | **0.634** | 0.021 | 0.134 |
| Without nouns | 0.060 | 0.149 | 0.457 | 0.021 | 0.057 |
| Without function words | 0.231 | 0.392 | 0.652 | 0.038 | 0.112 |
| Only proper nouns | 0.025 | 0.039 | 0.066 | 0.019 | 0.053 |
| Only medium lexicon | 0.019 | 0.147 | 0.332 | 0.002 | 0.136 |

CLIP agrees in direction (Gemini: nouns 0.187 and 0.530 vs all words 0.176 and 0.503; without nouns 0.062 and 0.370).

1. **On Gemini captions the floor is a noun floor.** Nouns are 27% of tokens, and alone they beat all words on every measure, for both ViTs. Removing them cuts k = 10 by two thirds (0.185 to 0.060). Larger LMs' deep layers reach 0.234-0.280 and CKA 0.713-0.778 (exp-022), so the gap above words narrows but stays.
2. **Function words only add noise for neighbours.** They are 41% of tokens. Removing them raises k = 1000 from 0.236 to 0.392, matching exp-022's no-stopword result.
3. **Original captions lean on names too.** Nouns and proper nouns reach about the same k = 10 (0.021, 0.019), and removing either drops all words from 0.039 to 0.021-0.024. So a name-heavy caption set has a different floor from a descriptive one.
4. **CKA is coarse: few words can drive it.** The 36-word medium lexicon has 1.2% of original-caption tokens and 9091 captions without one, yet excess CKA is 0.136 on DINOv2 with k = 10 at 0.002. Its controls score 0.006-0.013. DINOv2's types show why: maps (772 images at K = 5), documents and diagrams form their own clusters. This is the coarse structure a global kernel measure rewards.
   - **Not a claim that these words beat all words:** raw CKA is 0.141 vs 0.178, and excess favours the sparse kernel through its much lower null (0.005 vs 0.066). It is also the best cell of 3, and weaker for CLIP (0.053; all words 0.104).
5. **Colour, spatial and lighting words do not stand out.** On Gemini, none clearly beats its controls: k = 10 at or near control level, CKA below it. Removing any moves all measures by under 0.005. Captions dense in visual description gain little from these words specifically.
6. **LMs track nouns at every depth, and move off function words.** Against Pythia-410M, nouns are steady at k = 10 (0.242 first, 0.236 last) and rise in CKA (0.636 to 0.681), while function words fall (k = 10 0.050 to 0.025; CKA 0.520 to 0.410). LLaMA-13B shows the same. The function-word kernel has CKA 0.52 with LM layer 0 but 0.30 with DINOv2, so part of layer 0's variance has nothing to do with images.

### Limits of the word analysis

- **Controls are not class-neutral.** A control excludes any word tagged with the class even once, so non-noun classes' controls are mostly nouns, and only the noun control is free of them. "Below controls" for adjectives or verbs means below other, noun-heavy words, not uninformative. Only vs without is the safer contrast.
- **Two control seeds, no intervals,** and CKA controls vary widely (nouns 0.370 and 0.481). Token coverage of a control is not matched, only per-word frequency.
- **Differences are not shares.** Only and without do not add up.
- spaCy tags tokens on small-model accuracy, and tags on caption fragments are imperfect. The lexicons were fixed before results, but they are hand-made.

## Image types over training

n = 10240. Types are k-means clusters of a ViT's last layer, and each ViT is scored on the other's types. Curves: Pythia 160M and 410M, two seeds each (bands: 4 runs x 2 ViTs). Null: pairing broken, 10 permutations, Pythia-410M and TF-IDF. Tables for K = 5, 20, 100: `typed_summary.md`; figure at K = 20: `fig_types.png`.

**The chance level of the between-type part is high (K = 20: 0.36-0.52; K = 5: about 0.7), because both kernels are built on the same K groups, so the comparison is effectively on K points. Raw between-type CKA should not be read on its own.** Everything below on it is excess over that null.

Gemini captions, K = 20, Pythia-410M last layer vs words and larger LMs, excess over null, DINOv2 / CLIP:

| Measure | Step 0 | Final | TF-IDF | Larger LMs, last layer |
|---|---:|---:|---:|---:|
| Between-type CKA | 0.285 / 0.394 | 0.316 / 0.410 | 0.243 / 0.236 | 0.344-0.416 / 0.404-0.432 |
| Within-type CKA | 0.026 / 0.054 | 0.242 / 0.388 | 0.297 / 0.321 | 0.264-0.281 / 0.384-0.413 |
| Within-type mutual kNN | 0.052 / 0.059 | 0.235 / 0.244 | 0.213 / 0.207 | 0.250-0.268 / 0.263-0.278 |
| Between-type share of variance | 0.085 / 0.064 | 0.329 / 0.271 | 0.069 / 0.066 | 0.307-0.318 / 0.258-0.276 |
| Same-type neighbours | 0.274 / 0.236 | 0.479 / 0.492 | 0.390 / 0.410 | 0.486-0.495 / 0.501-0.522 |

Larger LMs use Pythia-410M's null.

1. **How the types are arranged is there before training.** Between-type CKA of an untrained last layer is already well above null, above TF-IDF, and gains little (+0.03, +0.02), after a dip around step 256. K = 5 and K = 100 agree (last layer, DINOv2: 0.093 to 0.103; 0.440 to 0.476). Layer 0 gains somewhat more (K = 20: 0.247 to 0.338, 0.257 to 0.317).
2. **Training changes how much, then what is inside.** Between-type share rises about 4x and within-type structure 4-9x. The arrangement of the types barely changes.
3. **Order, last layer, Gemini, K = 20.** Median t50 (range over 8 curves): between-type share 183 steps (146-268), then within-type CKA 837 (601-921), same-type neighbours 855 (747-949), within-type kNN 1107 (935-1311). Ranges do not overlap between the first and the rest. Layer 0 comes later on all of them (1275-1860). So variance moves onto the types first, and neighbours inside them align later.
4. **Deep layers start below layer 0 and overtake it.** Within-type kNN: last 0.075 to 0.248, layer 0 0.143 to 0.219, crossing around step 1000. The middle layer stays lower on Gemini (0.169), and that dip is absent on original captions.
5. **What LMs add over words.** The largest gap is between-type share, about 4x TF-IDF, then same-type neighbours (0.49-0.52 vs 0.39-0.41) and between-type CKA. Within types, the gap is small for kNN (about 0.05). Within-type CKA is split: words are level with LMs or above on DINOv2, and below on CLIP. This agrees with the shuffle result that local alignment above words needs word order, but it does not test order.
6. **Original captions show the same direction at lower levels.** Between-type share reaches only 0.06-0.08, and the last layer passes TF-IDF on within-type kNN around step 4000. With 6 checkpoints, no ordering is claimed.
7. **Many curves drop slightly at the last checkpoint** (within-type kNN, original captions: 0.086 at 64000 to 0.075 at 143000). Not explored.

### Limits of the image-type analysis

- **Warmup confound.** Warmup ends at step 1430, and nearly every t50 falls inside it or close to it. The order may reflect the learning-rate ramp, not structure, and only a run with another schedule can separate them. Checkpoints are sparse there too (64, 256, 512, 1000, 2000).
- **Between-type share is not all alignment.** Its null is near zero, so it is image-related, but a jump could partly be early loss of dimensions in deep layers, which was not measured.
- **Types depend on the ViT and on K.** DINOv2 and CLIP agree poorly on types (adjusted Rand 0.12-0.13), and DINOv2's largest cluster holds 6519 of 10240 at K = 5. Results hold on both ViTs, but the type definitions are not ground truth.
- **Few runs, small models.** Two sizes and two seeds; only finished checkpoints for larger LMs; ViTs are final only, so this is language-side emergence. Null is 10 permutations on one model, and larger LMs borrow it.
- **Parts do not add up.** Between and within CKA are separate normalised scores, not shares of total CKA. Excess also has unequal room: between-type CKA can gain only about 0.6 above its null.

## Commands

```bash
uv run python experiments/exp-024-emergence-words/run.py --out /workspace/results/oddharak/exp-024-emergence-words --exp016 /workspace/results/oddharak/exp-016-positive-controls --exp018 /workspace/results/oddharak/exp-018-alignment-emergence --pythia-orig /workspace/results/oddharak/exp-022-lexical-validation/phase2/pythia-orig --cache-dir /root/hf-scratch --tag-processes 15 --stages clusters,words --threads 16
uv run python experiments/exp-024-emergence-words/run.py --out /workspace/results/oddharak/exp-024-emergence-words --exp016 /workspace/results/oddharak/exp-016-positive-controls --exp018 /workspace/results/oddharak/exp-018-alignment-emergence --pythia-orig /workspace/results/oddharak/exp-022-lexical-validation/phase2/pythia-orig --cache-dir /root/hf-scratch --tag-processes 15 --stages typed --workers 2 --threads 14
uv run python experiments/exp-024-emergence-words/analyze_words.py --out results/exp-024-emergence-words
uv run python experiments/exp-024-emergence-words/analyze_typed.py --out results/exp-024-emergence-words
```

Run: `cpu3c` 16 vCPU, 32 GB. A first attempt was lost when the account balance ran out and Runpod deleted the pod, so rows now go straight to the volume. The image-type stage then hung for 2h24m with no CPU use: workers forked after torch's thread pool had started and deadlocked, probably in the first attempt too. They now spawn. Five workers ran at about 15 s per set but ran out of memory on the first large LM after 82 of 90 sets, and two workers finished the rest and the nulls, so use 2-3 for those. About $1.60 of pod time, of which $1.15 was the hang; the first attempt cost $0.35. Keep `HF_HOME` on the container disk, since the volume is near quota. Word stage: 446 s.
