# PRH released-code baselines (after audit)

**Operational reference:** Huh et al., arXiv:2405.07987; `minyoungg/platonic-rep@dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9` (`extract_features.py` + `measure_alignment.py`).

**Primary label:** PRH released-code protocol on COCO (test n=1024).  
**Not the reference:** `results/modern` (supplementary modified protocol).  
**Archive:** `results/original` (same mNN / linear CKA on the same caches; RBF used train-median bandwidths, not official σ=1).

Pre-change audit: `docs/audit_prh_vs_ours.md`. Config: `configs/prh_released_code.json`.

## 1. Where did our implementation diverge from PRH?

| Difference | Class | Effect on headline scores |
|---|---|---|
| COCO val2017 instead of WIT-1024 | Dataset substitution | All V–L numbers incomparable to Figure 3 |
| 6 LM × 3 ViT panel vs 12×17 | Model substitution | Means are a different population |
| BLOOMZ checkpoints (code) vs plots labelled “bloom” | Paper–code ambiguity; we followed code | Instruction-tuned vs base untested |
| Qwen3-Base / Qwen2.5 / OLMo-0724 vs paper test-set LMs | Model substitution | Size/competence plots not Figure 3 |
| RBF CKA at train-median × {0.5,1,2} with per-model σ | Deliberate deviation (now corrected in primary) | Inflated 0.5× RBF (0.677); official σ=1 is 0.524 |
| `pair_table` copied mNN layer indices onto CKA | Reporting error (fixed in `results/prh_released_code`) | Scores were already per-metric maxima; labels were wrong (30/36 pairs) |
| Shuffle only on already-selected layers | Extra control, mis-specified as null for max-over-layers | Fixed-layer mean 0.0098 ≈ k/(n−1); selection-aware mean 0.0124 |
| Validation-selected relative-depth “modern” protocol | Deliberate supplementary, not PRH | V–L mNN 0.044; do not use as the PRH reference |
| Concatenated layers (paper Appendix C) absent from released `compute_score` | Paper–code ambiguity | Bounded extra: concat **never** beat max-layer on V–L |
| Features stored fp16 | Minor numerical | mNN identical vs archive; linear CKA Δ < 3e-5 |
| Vision CLS via `blocks[i]` output hook vs FX `add_1` | Unresolved, likely match | No re-extract; metric parity on tensors holds |

WIT-1024 identities and MIT features remain unavailable (`minhuh/prh` is `.gitattributes` only; `vision14.csail.mit.edu` does not resolve).

## 2. Bugs vs substitutions vs ambiguities

- **Bugs / reporting:** CKA layer metadata copied from mNN; RBF presented as if it were the official estimator; shuffle SD treated like uncertainty on the observed score in places.
- **Not bugs:** COCO substitution; modern protocol; BLOOMZ (matches released code); max-over-layers independently per metric (matches released code).
- **Unresolved reference:** paper concat extra vs released max-over-layers-only; paper “bloom” vs code BLOOMZ; README “last layer vision” vs executable all-block CLS (executable wins).

## 3. What changed the reported scores?

- **mNN V–L 0.215 and linear CKA V–L 0.508:** unchanged (recomputed from the same 1024 caches). These already used clip → L2 → max over layers.
- **RBF:** official σ=1 both sides → V–L mean **0.524** (not 0.677). The 0.677 figure was median×0.5, not PRH’s default kernel.
- **Shuffle:** V–L fixed-layer 0.0098; selection-aware 0.0124. Max-over-layers does **not** explain the 0.215 score.
- **Modern 0.044:** still a different protocol on the same caches, not a corrected PRH number.

## 4. What is now faithful to the selected reference?

On COCO, from existing caches, matching released-code prep and metrics:

- clip q=0.95 (exact abs percentile, clamp) then L2
- all stored layers; independent max for mNN, linear CKA, RBF CKA
- mNN = mean \|∩\|/k, k=10, self excluded
- linear CKA = biased HSIC + 1e-6
- RBF CKA σ=1 both sides
- per-metric layer indices
- fixed-layer and selection-aware permutation nulls for mNN (one perm reused across layer candidates)

Parity tests (official clip / mNN / CKA formulas): 15/15 passed.

## 5. What PRH results have we actually reproduced?

| Claim | Status |
|---|---|
| Metric definitions (mNN, linear CKA, clip+L2, max layers) | Metric parity on identical tensors |
| Cross-modal mNN well above chance on a 1024 paired gallery | Qualitative, COCO: 0.215 vs ~0.012 selection-aware null |
| CLIP slightly above DINOv2 / IN21k on V–L mNN | Directional on this panel: 0.224 / 0.209 / 0.214 |
| L–L ≫ V–L | Yes on COCO: 0.681 vs 0.215 |
| Figure 3 WIT numerical values | **Not reproduced** (no WIT) |
| Alignment vs OpenWebText competence | **Untested** |
| 78-model Places/VTAB (Figure 2) | **Untested** |
| Within-family size law | Only two-point ladders: BLOOMZ-560M→1b1 vs DINOv2 0.208→0.223; Qwen3-0.6B→1.7B vs DINOv2 0.196→0.211. The 0.49B / 0.75B / 1.72B plot mixed Qwen2.5 with Qwen3 |

## 6. What cannot be compared directly?

WIT sample IDs, original captions, MIT features, the 12×17 panel, OpenWebText 4M BPB, BLOOM (non-Z) if that is what the figures used, and any digitised Figure 3 y-values.

COCO can show that the **released-code metric stack** yields stable, above-null V–L alignment on a different paired gallery. It cannot show that we recovered the paper’s WIT estimates.

## Before / after headlines (V–L, 18 pairs, n=1024, k=10)

| Statistic | Archive `results/original` | Primary `results/prh_released_code` |
|---|---|---|
| mNN | 0.215283 | 0.215283 |
| mNN fixed-layer shuffle | 0.009836 | 0.009836 |
| mNN selection-aware shuffle | — | 0.012374 |
| linear CKA | 0.507730 | 0.507716 |
| linear CKA fixed-layer shuffle | 0.052 | 0.052275 |
| RBF | 0.677 / 0.525 / 0.512 (median × 0.5/1/2) | **0.524** (σ=1) |
| RBF centred-Gram diag energy fraction | unmeasured | 0.106 mean (not diagonal-dominated) |
| mean off-diagonal RBF kernel | unmeasured | 0.625 |

Supplementary modern protocol (unchanged): V–L mNN 0.044, linear CKA 0.258.

Paper Appendix C concat (not in released `compute_score`; not primary): V–L concat mNN 0.150 < max-layer 0.215 (0/18 beats). Concat beat max-layer only on some L–L pairs (6/15 mNN, 11/15 CKA).

## Reproduce

```bash
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
cd /mnt/sdb1/prh-replication-work/repo
/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests -q
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_prh_released_code.py \
  --work /mnt/sdb1/prh-replication-work --concat --n-perm-select 100 --n-perm-fixed 100
```

Does not overwrite `results/original` or `results/modern`. Kernel learning is next. Final-layer-only extraction is reserved for a later sweep.
