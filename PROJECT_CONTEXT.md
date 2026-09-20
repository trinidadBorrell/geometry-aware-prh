# SPAR phase-1 project context

Pinned sources
- Paper: Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis*. arXiv:2405.07987 (ICML 2024, PMLR 235). Text retrieved 2026-09-16.
- Official code: https://github.com/minyoungg/platonic-rep commit `dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9` (2025-04-12).
- Official dataset repo: https://huggingface.co/datasets/minhuh/prh — inspected via Hub API 2026-09-16: only `.gitattributes` (2.31 kB). Dataset Viewer empty.
- Original features: `http://vision14.csail.mit.edu/prh/wit_1024/` listed in `platonic/__init__.py`. DNS lookup failed on the Ubuntu host (`Could not resolve host: vision14.csail.mit.edu`).

## Research question and this-phase scope

Question for the full SPAR project: do *learned* kernels between model pairs become more similar as models grow and generations succeed?

This phase established PRH released-code baselines on COCO, a first learned-kernel extension (RBF/RQ grids), and a second extension with valid polynomial / Gegenbauer / isotropic spectral-shell mixtures on the same frozen train-selected layers. It does **not** jointly search layers and kernels, expand the model panel, or claim scaling laws.


## Hardware (2026-09-16)

| Machine | Role | Constraint |
|---|---|---|
| Manjaro laptop | git workspace | 741 MB free, no NVIDIA |
| Ubuntu `angus-MS-7C56` | experiments | RTX PRO 6000 98 GB, **89 GB held by `vllm-akkadian.service`**; 125 GB RAM; `/` full; **`/mnt/sdb1` 446 GB free** |

Work root: `/mnt/sdb1/prh-replication-work`. Code is developed in this git repo and rsynced there. Extraction uses leftover GPU only for small ViTs; language models fall back to CPU so we do not evict vLLM.

## Methodological decisions

- **Primary dataset:** COCO val2017, first caption by annotation id. Splits 2048/1024/1024 by `sha256(f"{seed}:{image_id}")`. This is **not** WIT-1024.
- **Original WIT-1024:** preferred exact gallery; **unavailable** (empty HF revision + MIT host unreachable). Do not describe COCO numbers as an exact numerical replication.
- **Flickr30k / Places-365 / full VTAB:** catalogued, not downloaded.
- **Mutual kNN:** mean intersection size / k (Appendix A eq. 11 and `metrics.py`), **not** the IoU wording in paper Table 11.
- **Primary eval:** `configs/prh_released_code.json` — PRH released-code protocol on COCO: clip q=0.95 then L2, max over all stored layers independently per metric, mNN k=10, linear CKA, RBF CKA σ=1 both sides. Paper concat is an optional extra, not the primary score.
- **Archive `results/original`:** same clip/L2/max mNN and linear CKA; RBF used train-median multipliers.
- **Supplementary `results/modern`:** no clip; relative depths {0.25, 0.5, 0.75, 1.0}; val mNN freeze. Not the PRH reference.
- **Language pooling:** attention-mask mean over all `hidden_states` (includes embeddings). Plain captions, left pad, no chat template.
- **Vision pooling:** CLS of each ViT block output (post-residual). Final LN not applied. Language-supervised CLIP is labelled separately from DINOv2 / IN21k.
- **BLOOM vs BLOOMZ:** code uses BLOOMZ; paper figures say bloom. We extract BLOOMZ and label instruction-tuned.
- **Capability:** caption bits-per-byte stored at extraction (reduced). Paper used 4M OpenWebText tokens — not rerun this phase.
- **Kimi:** skipped (weight memory / stage mismatch vs this panel).
- **Qwen3-8B-Base:** audited public and is a true Base checkpoint; **not extracted** because remaining GPU is ~9 GB and CPU 8B would dominate the bound. Next experiment after vLLM is free.

## Commands

On Ubuntu, after rsync:

```bash
export HF_HOME=/mnt/sdb1/prh-replication-work/hf
export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src
cd /mnt/sdb1/prh-replication-work/repo
/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests -q
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_prh_released_code.py \
  --work /mnt/sdb1/prh-replication-work --concat --n-perm-select 100 --n-perm-fixed 100
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/check_vision_fx_hooks.py \
  --work /mnt/sdb1/prh-replication-work
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_learned_kernels.py \
  --work /mnt/sdb1/prh-replication-work --skip-hooks --n-perm 100
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_flex_kernels.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 100
/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_anisotropic_kernels.py \
  --work /mnt/sdb1/prh-replication-work --n-perm 40
```

## Completed / outstanding

See `docs/STATUS.md`, `docs/PRH_BEFORE_AFTER.md`, `docs/LEARNED_KERNELS_REPORT.md`, and `docs/FLEX_KERNELS_REPORT.md`.
Primary PRH numbers: `results/prh_released_code/`.
RBF/RQ extension: `results/learned_kernels/` (exploratory test gallery).
Polynomial + spectral mixtures: `results/flex_kernels/` (same frozen layers; exploratory test gallery).
Anisotropic PCA-subspace metrics: `results/anisotropic_kernels/` (same frozen layers; exploratory test gallery). See `docs/ANISOTROPIC_KERNELS_REPORT.md`.
Release-consistency experiment (final-layer protocol, 7B/8B Qwen–OLMo base panel + supplementary Qwen 3.x 4B): **frozen** as `prh-release-alignment-freeze-20260921`. Authoritative fitted outputs: `results/release_anisotropy_repair/` (work disk `/mnt/sdb1/prh-replication-work/results/release_anisotropy_repair/`). Parent `results/release_anisotropy/` retained; optimisation-dependent conclusions superseded. Report: `docs/RELEASE_ANISOTROPY_REPORT.md`. Inventory: `results/freeze/prh-release-alignment-freeze-20260921/`. See `FREEZE.md`. No further sweep. vLLM was not started or stopped in the freeze pass.

## Deferred

Still deferred: generation/chat-template comparisons, extra layers, 78-model VTAB/Places, OpenWebText 4M BPB, expanding q or eigenvalue bounds after seeing eval, a new confirmation dataset, restarting vLLM after the GPU extraction.
