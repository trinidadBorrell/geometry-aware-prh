"""Frozen dataset and model registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: Literal["language", "vision"]
    source: str
    checkpoint: str
    revision: str | None
    family: str
    release_date: str
    release_source: str
    n_params: int | None
    hidden_size: int | None
    n_layers: int | None
    training_stage: str
    objective: str
    modality: str
    language_supervised: bool
    native_dtype: str
    access: str
    panels: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source: str
    revision: str | None
    modality: str
    role: str
    access: str
    approx_storage: str
    notes: str = ""


# Paper: arXiv:2405.07987 (ICML 2024). Code pin: minyoungg/platonic-rep@dcd76ba.
MODELS: dict[str, ModelSpec] = {
    "bloomz-560m": ModelSpec(
        key="bloomz-560m",
        kind="language",
        source="hf",
        checkpoint="bigscience/bloomz-560m",
        revision=None,
        family="BLOOMZ",
        release_date="2022-11-08",
        release_source="https://huggingface.co/bigscience/bloomz-560m",
        n_params=559_214_592,
        hidden_size=1024,
        n_layers=24,
        training_stage="instruction-tuned (BLOOMZ, not BLOOM base)",
        objective="multitask prompted finetuning after causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="fp16/bf16",
        access="public",
        panels=("A",),
        notes="Paper plots label 'bloom'; official code uses bloomz.",
    ),
    "bloomz-1b1": ModelSpec(
        key="bloomz-1b1",
        kind="language",
        source="hf",
        checkpoint="bigscience/bloomz-1b1",
        revision=None,
        family="BLOOMZ",
        release_date="2022-11-08",
        release_source="https://huggingface.co/bigscience/bloomz-1b1",
        n_params=1_065_307_136,
        hidden_size=1536,
        n_layers=24,
        training_stage="instruction-tuned (BLOOMZ)",
        objective="multitask prompted finetuning after causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="fp16/bf16",
        access="public",
        panels=("A",),
    ),
    "bloomz-3b": ModelSpec(
        key="bloomz-3b",
        kind="language",
        source="hf",
        checkpoint="bigscience/bloomz-3b",
        revision=None,
        family="BLOOMZ",
        release_date="2022-11-08",
        release_source="https://huggingface.co/bigscience/bloomz-3b",
        n_params=3_002_557_440,
        hidden_size=2560,
        n_layers=30,
        training_stage="instruction-tuned (BLOOMZ)",
        objective="multitask prompted finetuning after causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="fp16/bf16",
        access="public",
        panels=("A",),
        notes="Optional; run only if memory allows.",
    ),
    "qwen3-0.6b-base": ModelSpec(
        key="qwen3-0.6b-base",
        kind="language",
        source="hf",
        checkpoint="Qwen/Qwen3-0.6B-Base",
        revision="da87bfb608c1",
        family="Qwen3",
        release_date="2025-04-29",
        release_source="https://huggingface.co/Qwen/Qwen3-0.6B-Base",
        n_params=751_632_384,
        hidden_size=1024,
        n_layers=28,
        training_stage="base (pretrained; not instruction-tuned)",
        objective="causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="bf16",
        access="public",
        panels=("B",),
        notes="Unsuffixed Qwen3-0.6B is a finetune of this base model.",
    ),
    "qwen3-1.7b-base": ModelSpec(
        key="qwen3-1.7b-base",
        kind="language",
        source="hf",
        checkpoint="Qwen/Qwen3-1.7B-Base",
        revision="ea980cb0a6c2",
        family="Qwen3",
        release_date="2025-04-29",
        release_source="https://huggingface.co/Qwen/Qwen3-1.7B-Base",
        n_params=1_720_320_000,
        hidden_size=2048,
        n_layers=28,
        training_stage="base (pretrained)",
        objective="causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="bf16",
        access="public",
        panels=("B",),
    ),
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        kind="language",
        source="hf",
        checkpoint="Qwen/Qwen2.5-0.5B",
        revision="060db6499f32",
        family="Qwen2.5",
        release_date="2024-09-19",
        release_source="https://huggingface.co/Qwen/Qwen2.5-0.5B",
        n_params=494_032_768,
        hidden_size=896,
        n_layers=24,
        training_stage="base (unsuffixed Qwen2.5 checkpoint; Instruct is separate)",
        objective="causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="bf16",
        access="public",
        panels=("B",),
        notes="Near-size prior-generation comparison to Qwen3-0.6B-Base.",
    ),
    "olmo-1b-0724": ModelSpec(
        key="olmo-1b-0724",
        kind="language",
        source="hf",
        checkpoint="allenai/OLMo-1B-0724-hf",
        revision="d7cbab742d80",
        family="OLMo",
        release_date="2024-07",
        release_source="https://huggingface.co/allenai/OLMo-1B-0724-hf",
        n_params=1_177_627_136,
        hidden_size=2048,
        n_layers=16,
        training_stage="base (pretrained)",
        objective="causal LM",
        modality="text",
        language_supervised=True,
        native_dtype="bf16",
        access="public",
        panels=("B",),
        notes="Already cached on host. Not the original paper OLMo-1B-hf.",
    ),
    "dinov2-small": ModelSpec(
        key="dinov2-small",
        kind="vision",
        source="timm",
        checkpoint="vit_small_patch14_dinov2.lvd142m",
        revision=None,
        family="DINOv2",
        release_date="2023-04-14",
        release_source="https://github.com/facebookresearch/dinov2",
        n_params=22_051_968,
        hidden_size=384,
        n_layers=12,
        training_stage="pretrained SSL (no language supervision)",
        objective="DINOv2 self-distillation",
        modality="image",
        language_supervised=False,
        native_dtype="fp32/fp16",
        access="public via timm",
        panels=("A", "B"),
    ),
    "vit-in21k-small": ModelSpec(
        key="vit-in21k-small",
        kind="vision",
        source="timm",
        checkpoint="vit_small_patch16_224.augreg_in21k",
        revision=None,
        family="ViT-AugReg",
        release_date="2021-06",
        release_source="https://github.com/rwightman/pytorch-image-models (augreg in21k)",
        n_params=22_050_664,
        hidden_size=384,
        n_layers=12,
        training_stage="ImageNet-21k supervised (no language)",
        objective="image classification",
        modality="image",
        language_supervised=False,
        native_dtype="fp32",
        access="public via timm",
        panels=("A", "B"),
        notes="Do not substitute augreg_in21k_ft_in1k.",
    ),
    "clip-laion-base": ModelSpec(
        key="clip-laion-base",
        kind="vision",
        source="timm",
        checkpoint="vit_base_patch16_clip_224.laion2b",
        revision=None,
        family="OpenCLIP",
        release_date="2022-10",
        release_source="https://github.com/mlfoundations/open_clip",
        n_params=86_192_640,
        hidden_size=768,
        n_layers=12,
        training_stage="language-supervised contrastive (LAION-2B)",
        objective="CLIP",
        modality="image",
        language_supervised=True,
        native_dtype="fp32/fp16",
        access="public via timm",
        panels=("A", "B"),
        notes="Separately identified language-supervised comparison.",
    ),
}

DATASETS: dict[str, DatasetSpec] = {
    "wit_1024_original": DatasetSpec(
        key="wit_1024_original",
        source="minhuh/prh revision=wit_1024; facebook/pmd WIT seed=42",
        revision="wit_1024",
        modality="image-text",
        role="exact original-protocol gallery",
        access="HF dataset currently empty (.gitattributes only). MIT feature host vision14.csail.mit.edu did not resolve.",
        approx_storage="~1k JPEGs + captions; original features ~tens of GB for full panel",
        notes="Paper used 1024 WIT pairs. Code downloaded 4096 then lexically sorted files and took first 1024.",
    ),
    "coco_val2017": DatasetSpec(
        key="coco_val2017",
        source="http://images.cocodataset.org val2017 + captions_trainval2017",
        revision="2017",
        modality="image-text",
        role="primary modern image-text dataset",
        access="public HTTP",
        approx_storage="val2017.zip ~1GB + annotations ~241MB",
        notes="First listed original caption per image; images grouped by coco image_id.",
    ),
    "flickr30k": DatasetSpec(
        key="flickr30k",
        source="https://shannon.cs.illinois.edu/DenotationGraph/",
        revision=None,
        modality="image-text",
        role="catalogued alternative; not downloaded this phase",
        access="form / academic request",
        approx_storage="~4GB images",
    ),
    "wikitext2": DatasetSpec(
        key="wikitext2",
        source="Salesforce/wikitext wikitext-2-raw-v1",
        revision=None,
        modality="text",
        role="reduced language-model bits-per-byte eval",
        access="public HF",
        approx_storage="<50MB",
        notes="Not the paper's 4M-token OpenWebText eval.",
    ),
    "places365": DatasetSpec(
        key="places365",
        source="Zhou et al. 2017",
        revision=None,
        modality="image",
        role="catalogued; original 78-model VTAB/Places experiment deferred",
        access="academic",
        approx_storage="large",
    ),
}

MODERN_LAYER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
RBF_MULTIPLIERS = (0.5, 1.0, 2.0)
MKNN_K = 10
CLIP_Q = 0.95
N_PERM = 100
SEED = 0


@dataclass
class Paths:
    work: Path
    repo: Path

    @property
    def data(self) -> Path:
        return self.work / "data"

    @property
    def features(self) -> Path:
        return self.work / "features"

    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def hf(self) -> Path:
        return self.work / "hf"


def default_paths(repo: Path | None = None) -> Paths:
    repo = repo or Path(__file__).resolve().parents[2]
    env = Path("/mnt/sdb1/prh-replication-work")
    work = env if env.parent.exists() else (repo / "work")
    return Paths(work=work, repo=repo)
