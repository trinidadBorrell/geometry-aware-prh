# Original experiment audit

Sources pinned in `PROJECT_CONTEXT.md`.

## Dataset

`data.py` `prepare_facebook_pmd_dataset`:
- `load_dataset("facebook/pmd", "wit", split="train").shuffle(seed=42)`
- Keep samples with a loadable RGB image and `len(text.split()) > 1`
- Captions list: `[alt text, context_page_description, context_sect_description]` from WIT `meta` JSON
- Default `caption_idx=0` (alt text). Filename omits `_cid` when caption_idx is 0
- Note in code: they downloaded **4096**, **lexicographically sorted filenames** (not natural sort), took first **1024**
- Uploaded that exact set to `minhuh/prh` revision `wit_1024` (was private; now public repo is empty)

Paper Appendix C: k=10 over 1024 WIT samples.

## Features (MIT)

`platonic/__init__.py` maps short names to `.pt` on `http://vision14.csail.mit.edu/prh/wit_1024/`.

## Extraction

Language (`extract_features.py`):
- Tokenize **all** captions with `padding="longest"`
- `padding_side="left"`; Huggyllama pad=`[PAD]`
- `output_hidden_states=True`; **avg** pool with attention mask over every hidden state including embeddings
- `last` pool = index `-1` (valid under left padding)
- Prompt condition exists; default off
- QLoRA optional and documented to shift scores
- Caption NLL and bits-per-byte stored (`utf-8` byte count; nats × log2(e) / nbytes)

Vision:
- timm pretrained ViT only
- `create_feature_extractor` nodes `blocks.{i}.add_1`
- pool **cls** (`[:, 0, :]`)
- No language pooling on vision

## Alignment (`measure_alignment.py`)

1. `remove_outliers` q=0.95 exact
2. L2 normalize
3. Score every layer pair; keep **maximum**
4. Default metric `mutual_knn` k=10

kNN: Gram = inner product, diagonal `-1e8`, `argsort` descending. Equivalent to cosine after L2.

CKA: biased HSIC `tr(KHLH)`, linear kernel `X X^T`, RBF `exp(-cdist^2 / (2 σ^2))` default σ=1. Unbiased HSIC from Song et al. 2012.

## Paper vs code disagreements

| Topic | Paper | Public code |
|---|---|---|
| Mutual kNN | Table 11 says IoU | Appendix A and code: \|∩\| / k |
| Concatenated layers | Appendix C says one comparison is concatenated features | Not implemented |
| BLOOM | Figure 3 labels bloom | `bigscience/bloomz-*` |
| llama-30b | 30B | Comment: actually 33B `huggyllama/llama-30b` |
| OpenWebText BPB | 4M tokens | Extraction BPB is on WIT captions |

## Original model IDs (tasks.py modelset=val)

Language: bloomz-{560m,1b1,1b7,3b,7b1}, open_llama_{3b,7b,13b}, huggyllama/llama-{7b,13b,30b,65b}

Vision: vit_{tiny,small,base,large}_patch16_224.augreg_in21k; MAE b/l/h; dinov2 s/b/l/g; clip laion2b b/l/h; clip laion2b_ft_in12k b/l/h

## Replication classes for this repo

1. Recompute from original cached features — **blocked** (MIT DNS, empty HF).
2. Independent extraction on the original 1024 identities — **blocked** (identities not recoverable).
3. Same metrics/pooling on **COCO val2017** with original and modern checkpoints — **this phase**. Labelled as a substituted-data trend study.
