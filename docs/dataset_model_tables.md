# Dataset and model tables

## Datasets

| Key | Source / revision | Modality | Usable count | Access | Storage | Role |
|---|---|---|---|---|---|---|
| wit_1024_original | minhuh/prh@wit_1024; facebook/pmd WIT seed 42 | image–text | 1024 intended | HF empty; MIT features unresolved | original feats large | exact gallery — **unavailable** |
| coco_val2017 | cocodataset.org val2017 + captions_val2017 | image–text | 5000 images; we freeze 2048/1024/1024 | public HTTP | ~1.2 GB | **primary** modern + substituted original-protocol |
| flickr30k | Illinois Denotation Graph | image–text | ~31k | request | ~4 GB | catalogued only |
| wikitext2 | Salesforce/wikitext wikitext-2-raw-v1 | text | small | public | <50 MB | reduced LM eval (deferred if time) |
| places365 | Zhou et al. 2017 | image | large | academic | large | not used (no VTAB sweep) |

Caption rule (COCO): original English captions only; first by increasing annotation id; all captions of an image stay in the same split because the unit is `image_id`.

Do not call the 1024-image **test** split a WIT-1024 replica. Train/val/test are disjoint 2048+1024+1024 from 5000 val images.

## Panel A (faithful checkpoint subset; **substituted COCO data**)

| Key | Checkpoint | Family / date | Params | Hid / layers | Stage | Lang-sup | Notes |
|---|---|---|---|---|---|---|---|
| bloomz-560m | bigscience/bloomz-560m | BLOOMZ 2022-11 | 559M | 1024 / 24 | instruction-tuned | n/a | paper “bloom” |
| bloomz-1b1 | bigscience/bloomz-1b1 | BLOOMZ 2022-11 | 1.07B | 1536 / 24 | instruction-tuned | n/a | size step |
| dinov2-small | vit_small_patch14_dinov2.lvd142m | DINOv2 2023-04 | 22M | 384 / 12 | SSL | no | vision without language |
| vit-in21k-small | vit_small_patch16_224.augreg_in21k | AugReg 2021 | 22M | 384 / 12 | IN21k classif. | no | not ft_in1k |
| clip-laion-base | vit_base_patch16_clip_224.laion2b | OpenCLIP 2022-10 | 86M | 768 / 12 | CLIP | yes | separate CLIP comparison |

bloomz-3b is registered but not in the default extract list.

Original cached features for these IDs were not verified (host down). Independent regen is on COCO, not WIT.

## Panel B (modern)

| Key | Checkpoint | Rev (Hub, 2026-09-16) | Family / date | Params | Stage | Why |
|---|---|---|---|---|---|---|
| qwen3-0.6b-base | Qwen/Qwen3-0.6B-Base | da87bfb608c1 | Qwen3 2025-04-29 | 0.75B | **Base** | size ladder |
| qwen3-1.7b-base | Qwen/Qwen3-1.7B-Base | ea980cb0a6c2 | Qwen3 2025-04-29 | 1.72B | **Base** | size ladder |
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B | 060db6499f32 | Qwen2.5 2024-09-19 | 0.49B | Base (Instruct is a different repo) | prior generation ~same size |
| olmo-1b-0724 | allenai/OLMo-1B-0724-hf | d7cbab742d80 | OLMo 2024-07 | 1.18B | Base | second family; already on disk |
| + same three vision models | | | | | | overlap with A |

Qwen3-8B-Base (49e3418fbbbc) is a real Base checkpoint; deferred (memory). Unsuffixed `Qwen/Qwen3-0.6B` is a **finetune** of the Base repo — not used.

Kimi: not included (too large / unclear stage match).

Native extraction dtype: bf16 on CUDA if the leftover device is used; fp32 on CPU. No mixed quantized vs native comparison.

## Selection rationale

vLLM occupies 89/98 GB. A 6+3 panel with ≤1.7B LMs on CPU and small ViTs on the remaining GPU is the largest sequential panel that does not require stopping `vllm-akkadian.service`.
