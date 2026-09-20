# PRH audit (written before code changes)

Pinned references (2026-09-17):
- Paper: Huh et al., *The Platonic Representation Hypothesis*, arXiv:2405.07987 (ICML 2024).
- Official repo: `minyoungg/platonic-rep` commit `dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9`.
- Operational replication target: **released executable workflow** (`extract_features.py` + `measure_alignment.py`), not the README one-liner and not an unpublished concat variant.

## README vs executable (resolved)

README says “extract last layer features of all vision models” with `--pool cls`. Executable `extract_lvm_features` still iterates `blocks.{i}.add_1` for **every** block and pools CLS per block. `measure_alignment.compute_score` maximises over all stored layers. The README phrase refers to **CLS pooling**, not last-layer-only extraction. **Released-code protocol = all vision blocks + all LM `hidden_states` (including embeddings), CLS / mask-mean, then max over layer pairs.** Concatenated features appear in Appendix C of the paper only; they are **absent** from `compute_score`.

## WIT / MIT limitation (re-verified 2026-09-17)

- `huggingface_hub.list_repo_files("minhuh/prh")` → `['.gitattributes']` only.
- `socket.getaddrinfo("vision14.csail.mit.edu", 80)` → `Name or service not known`.
- No recovery attempt this phase.

## Headline trace (recomputed from pair-level JSON)

Source: `/mnt/sdb1/prh-replication-work/results/original/pair_table.json` and `test_pairs/*.json` (36 pairs, COCO test n=1024). Independently recomputed means match the report:

| Statistic | Recomputed |
|---|---|
| Original V–L mNN | 0.215283 |
| Original V–L mNN shuffle (fixed selected layers) | 0.009836 |
| Original L–L mNN | 0.681 (from earlier group mean) |
| Original V–L linear CKA | 0.507730 |
| Original V–L CKA excess | 0.455441 |
| Modern V–L mNN | 0.043962 |
| Modern V–L linear CKA | 0.257967 |

**Reporting error:** `pair_table.json` stores a single `layers` field copied from **mNN**. In `test_pairs`, **30/36** pairs select a different layer pair for linear CKA than for mNN; **34/36** differ for at least one RBF bandwidth. Displayed CKA scores are the CKA-selected maxima; displayed layer indices were the mNN indices.

Caches used for those scores: `/mnt/sdb1/prh-replication-work/features/coco_val2017/{train,val,test}/<model>/*.pt` with `n=1024` in the cache key. Smoke caches with `n=32` also exist for dinov2-small and qwen3-0.6b-base; the loader keys on sample-id span, so 1024-id files were used. Features are **float16**. Prep at eval: `.float()` → 95th-percentile abs clip (exact, all elements) → L2 (`original`) or identity (`modern`).

Original vs modern on the **same** 1024-id caches and models; they differ in prep, candidate layers, and selection (max on test vs val-frozen relative-depth). Not a different gallery.

## Methodology comparison

Component | Paper description | Official code behaviour | Our implementation | Match status | Likely impact | Required action
---|---|---|---|---|---|---
Gallery | WIT, 1024 pairs, k=10 | `minhuh/prh` revision `wit_1024`; download 4096 / lexical sort / first 1024 noted in `data.py` | COCO val2017, first caption by ann id, hash split, test n=1024 | Dataset substitution | Changes all cross-modal numbers; cannot match Figure 3 numerically | Keep; label clearly. No WIT recovery this phase.
Caption | WIT alt/context fields; `caption_idx=0` default | `text[0]` after `[alt, page, section]` | COCO original English caption, min `ann id` | Dataset substitution | Different text distribution | Document
Sample order | Paired (x_i, y_i) | Dataset row order | Manifest `splits.test` order, shared across models | Verified match (to PRH pairing idea) | — | None
Failed images | Drop unloadable | Timeout URL fetch; skip | COCO local files; 0 failures | Deliberate (local COCO) | None for COCO | None
LM checkpoints (panel A) | Plots say “bloom” | `bigscience/bloomz-*` (instruction-tuned) | Same BLOOMZ IDs | Verified match to **code**; paper–code ambiguity vs BLOOM base | Instruction-tuned vs base | Label BLOOMZ
LM checkpoints (panel B) | Not in Figure 3 | OLMo/Gemma/Llama3 in `modelset=test` | Qwen3-Base, Qwen2.5-0.5B, OLMo-1B-0724 | Dataset/model substitution | New panel, not Figure 3 | Label substituted
Vision IDs | TIMM ViT IN21k / DINOv2 / CLIP LAION | Exact timm names in `tasks.py` | `vit_small_patch14_dinov2.lvd142m`, `vit_small_patch16_224.augreg_in21k`, `vit_base_patch16_clip_224.laion2b` | Verified match for these three | Smaller than 17-model vision panel | None
Vision pooling | CLS each layer | CLS on `blocks.{i}.add_1` | Forward hook on `blocks[i]` output, CLS | Likely match (block return = post-MLP residual; FX `add_1` is that add) | Small if hook ≠ add_1 | Bounded note; no re-extract unless parity fails
LM pooling | Average each layer | Masked mean over tokens, all `hidden_states` incl. embeddings; left pad | Same | Verified match | — | None
LM padding | — | Global `padding="longest"` then batched | Per-batch padding, left, no trunc | Deliberate / minor | Masked mean invariant to pad length | None
Concat features | Appendix C: also compare concatenated features | **Not implemented** | **Not implemented** | Paper–code ambiguity | Unknown; paper claims one extra candidate | Bounded concat-on-cache comparison; primary = no concat
Clip | Truncate elements above 95th percentile | `remove_outliers` exact: sort \|x\|, take q-th, **clamp** to ±q | Same | Verified match | — | None
L2 | Applied before distance | After clip, `F.normalize` | After clip for `original` | Verified match | — | None
mNN | Appendix A: \|∩\|/k; Table 11 says IoU | `mutual_knn`: \|∩\|/k, diag −1e8, argsort | Same as code | Verified match to **code**; paper table is IoU wording | Tiny vs Jaccard | Follow code
CKA | Kornblith; appendix also CKNNA | Biased HSIC, linear or RBF σ=1 default, +1e-6 | Linear via centred Gram Frobenius; RBF with **train-median × {0.5,1,2}**, independent σ per model | Linear ~match; RBF = deliberate deviation | Inflates/changes RBF vs PRH | Recompute PRH RBF at σ=1 both sides from caches
Layer selection | Max pairwise (BrainScore); includes concat | Max over stored layers, **per metric** independently | Same search; **pair_table reported only mNN layers** | Implementation OK; **reporting error** | Misleading layer metadata | Fix table; keep independent selection (PRH does this)
Shuffle | Not in paper figures | Not in `measure_alignment.py` | 100 one-sided perms on **already selected** layers | Deliberate extra control; **mis-specified if used as null for the max statistic** | Shuffle ~10/1023 ≈ 0.0098 matches **fixed-layer** chance, not max-over-layers chance | Add selection-aware null; keep fixed-layer null labelled
Modern protocol | Not PRH | Not PRH | Relative depth + val mNN freeze, no clip | Deliberate supplementary | Large mNN drop | Keep supplementary; do not use as PRH reference
Precision | — | float32 in alignment | Features stored fp16, eval float32 | Minor numerical | Sub-0.001 typical | No re-extract
Competence | OpenWebText 4M tokens, 1−BPB | Caption BPB stored in extract | Caption BPB only, unused in plots | Untested PRH claim | Cannot reproduce Figure 3 x-axis | Document
Places/VTAB 78 models | Figure 2 | Not in this extract script | Not run | Untested | — | Out of scope

## Paper claims vs our results (conditions differ)

- Figure 3 (WIT, 12 LM × 17 LVM, mNN k=10, max layers): peak ~0.16 (paper text). Our COCO V–L mean 0.215 on a **different 6 LM × 3 LVM panel**. Same order, **not a matching estimate**.
- CLIP > vision-only: paper Figure 3 (approximate from description). Our COCO means CLIP 0.224 vs DINOv2 0.209 / IN21k 0.214. Directionally similar, gap smaller. Not digitised Figure 3 values.
- Size vs alignment: paper uses **competence** (bits/byte), not parameters. Our 0.49B / 0.75B / 1.72B are Qwen2.5-0.5B, Qwen3-0.6B-Base, Qwen3-1.7B-Base — **not one family**. Untested: OpenWebText BPB trend.
- CKA “weak trend”: paper qualitative. Our max CKA V–L 0.51 is an absolute score after max-over-layers on COCO, not their trend plot.

## Claims to retract or qualify

- “Layer protocol is the main effect” — compares PRH max-over-layers to a **non-PRH** val-frozen grid. True as a sensitivity of *our* two protocols, not an explanation of PRH. Do not use modern scores as the reference.
- “Narrow RBF inflates because Grams become near-diagonal” — only higher RBF scores observed; no Gram energy diagnostic. Qualify unless measured.
- “CLIP vs DINOv2 vs IN21k is a clean language-supervision contrast” — CLIP is language-supervised; the other two differ in data and objective as well (IN21k labels vs DINOv2 SSL). Not a single-factor contrast.
- “Original protocol” on Qwen/OLMo/COCO — is **PRH released-code protocol applied to substituted data/models**.

## Fixes applied (see `docs/PRH_BEFORE_AFTER.md`)

1. Primary config `configs/prh_released_code.json` and `results/prh_released_code/`.
2. Per-metric layer metadata.
3. Fixed-layer and selection-aware mNN nulls (100 perms; one perm reused across layers).
4. Bounded concat extra on caches; primary remains max-over-layers.
5. `results/original` and `results/modern` left in place.
6. Recomputed from existing 1024 caches; no re-extract.
