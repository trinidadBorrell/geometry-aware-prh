"""Shared per-layer activation cache on the team volume.

Layout (see /workspace/README.md):

    <root>/<model>/layer_<ii>/<dataset>.pt          tensor [n_samples, dim]
    <root>/<model>/layer_<ii>/<dataset>.meta.json   model, layer, dataset, seed, dtype, shape, ...

`<model>` is the HF/timm name with "/" replaced by "__"; `<dataset>` encodes everything
that changes the activations (dataset, subset, pooling, caption index), e.g.
`minhuh_prh_wit_1024_pool-avg_cid-0`.

The upstream pipelines (platonic-rep, Aristotelian) cache one `.pt` per model holding all
layers (`{"feats": [n, L, d], "num_params": int}`) and skip extraction when that file
exists. This module bridges the two without touching upstream code:

    push: split upstream feature files into per-layer cache entries (never overwrites)
    pull: rebuild upstream feature files from the cache, so extraction is skipped
    ls:   show which models/variants are cached

Example:
    uv run python -m geoprh.activation_cache pull --modelset small --layout platonic \\
        --dir results/prh_small/platonic/features/minhuh/prh/wit_1024
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

DEFAULT_ROOT = "/workspace/activations"
EXTRA_KEYS = ("loss", "bpb")
LAYER_DEFS = {
    "language": "hidden_states[i] of the causal LM (0 = token embeddings), masked token pooling",
    "vision": "timm ViT blocks.{i}.add_1 (block output), token pooling",
}


@dataclass(frozen=True)
class Variant:
    """What was embedded and how; everything that changes the activations."""

    dataset: str = "minhuh/prh"
    subset: str = "wit_1024"
    pool: str = "avg"
    caption_idx: int | None = 0  # text only
    modality: str = "language"

    @property
    def key(self) -> str:
        key = f"{self.dataset.replace('/', '_')}_{self.subset}_pool-{self.pool}"
        if self.modality == "language" and self.caption_idx is not None:
            key += f"_cid-{self.caption_idx}"
        return key


def default_root() -> Path:
    return Path(os.environ.get("ACTIVATIONS_DIR", DEFAULT_ROOT))


def model_dir(root: Path, model: str) -> Path:
    return root / model.replace("/", "__")


def layer_paths(root: Path, model: str, layer: int, variant: Variant) -> tuple[Path, Path]:
    base = model_dir(root, model) / f"layer_{layer:02d}"
    return base / f"{variant.key}.pt", base / f"{variant.key}.meta.json"


def cached_layers(root: Path, model: str, variant: Variant) -> list[int]:
    """Layer indices with a cached tensor for this model/variant, sorted."""
    layers = []
    for d in model_dir(root, model).glob("layer_*"):
        if (d / f"{variant.key}.pt").exists() and (d / f"{variant.key}.meta.json").exists():
            layers.append(int(d.name.removeprefix("layer_")))
    return sorted(layers)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def push_model(
    root: Path, model: str, variant: Variant, feature_file: Path, *, extractor: str
) -> int:
    """Split an upstream feature file into per-layer cache entries. Returns layers written."""
    payload = torch.load(feature_file, map_location="cpu", weights_only=False)
    feats = payload["feats"]
    layers = [feats[:, i] for i in range(feats.shape[1])] if torch.is_tensor(feats) else feats
    written = 0
    for i, x in enumerate(layers):
        pt_path, meta_path = layer_paths(root, model, i, variant)
        if pt_path.exists() and meta_path.exists():
            continue
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(x.contiguous().clone(), pt_path)
        meta = {
            "model": model,
            "modality": variant.modality,
            "layer": i,
            "num_layers": len(layers),
            "layer_def": LAYER_DEFS[variant.modality],
            "dataset": variant.dataset,
            "subset": variant.subset,
            "pool": variant.pool,
            "caption_idx": variant.caption_idx if variant.modality == "language" else None,
            # Extraction is deterministic (eval mode, fixed dataset revision): no sampling seed.
            "seed": None,
            "dtype": str(x.dtype).removeprefix("torch."),
            "shape": list(x.shape),
            "num_params": int(payload.get("num_params", 0)) or None,
            # per-model scalars saved by the extractor (platonic-rep: caption loss, bits-per-byte)
            "extras": {k: float(payload[k]) for k in EXTRA_KEYS if k in payload},
            "extractor": extractor,
            "source_file": str(feature_file),
            "created_by": os.environ.get("PERSON") or getpass.getuser(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _atomic_write_json(meta, meta_path)
        written += 1
    return written


def load_model(root: Path, model: str, variant: Variant) -> dict | None:
    """Load all cached layers as an upstream-style payload, or None unless fully cached.

    Returns `{"feats": [n, L, d], "num_params": int, **extras}` (extras: loss, bpb).
    """
    layers = cached_layers(root, model, variant)
    if not layers:
        return None
    meta = json.loads(layer_paths(root, model, layers[0], variant)[1].read_text())
    if layers != list(range(meta["num_layers"])):
        return None
    feats = torch.stack(
        [torch.load(layer_paths(root, model, i, variant)[0], map_location="cpu") for i in layers],
        dim=1,
    )
    return {"feats": feats, "num_params": meta["num_params"], **meta.get("extras", {})}


def pull_model(root: Path, model: str, variant: Variant, dest: Path) -> bool:
    """Rebuild an upstream feature file from the cache; False unless fully cached."""
    payload = load_model(root, model, variant)
    if payload is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(payload, dest)
    return True


def upstream_filename(model: str, variant: Variant, layout: str) -> str:
    """Per-model feature filename used by the upstream code (see their `paths`/`utils`)."""
    name = f"{model.replace('/', '_')}_pool-{variant.pool}"
    if layout == "platonic":
        # platonic-rep utils.to_feature_filename: prompt/caption_idx only when truthy
        if variant.modality == "language" and variant.caption_idx:
            name += f"_cid-{variant.caption_idx}"
        return f"{name}.pt"
    if layout == "aristotelian":
        # aristotelian.prh.paths.prh_feature_filename: prompt always, cid for text only
        name += "_prompt-False"
        if variant.modality == "language" and variant.caption_idx is not None:
            name += f"_cid-{variant.caption_idx}"
        return f"{name}.pt"
    raise ValueError(f"unknown layout {layout!r}")


def models(modelset: str) -> list[tuple[str, Variant]]:
    """(model, variant) for every model in an upstream PRH modelset (text: avg, vision: cls)."""
    from aristotelian.prh.prh_models import get_models

    llms, lvms = get_models(modelset)
    return [(m, Variant(pool="avg", caption_idx=0, modality="language")) for m in llms] + [
        (m, Variant(pool="cls", caption_idx=None, modality="vision")) for m in lvms
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared per-layer activation cache.")
    parser.add_argument("command", choices=["push", "pull", "ls"])
    parser.add_argument("--root", type=Path, default=None, help="cache root (env ACTIVATIONS_DIR)")
    parser.add_argument("--modelset", default="small")
    parser.add_argument("--layout", choices=["platonic", "aristotelian"], default="platonic")
    parser.add_argument("--dir", type=Path, nargs="+", default=[], help="upstream feature dir(s)")
    parser.add_argument("--extractor", default="platonic-rep extract_features.py")
    args = parser.parse_args()
    root = args.root or default_root()

    for model, variant in models(args.modelset):
        fname = upstream_filename(model, variant, args.layout)
        if args.command == "ls":
            layers = cached_layers(root, model, variant)
            print(f"{model:<48} {variant.key:<40} layers cached: {len(layers)}")
        elif args.command == "push":
            src = next((d / fname for d in args.dir if (d / fname).exists()), None)
            if src is None:
                print(f"[push] {model}: no {fname} in {[str(d) for d in args.dir]}")
                continue
            n = push_model(root, model, variant, src, extractor=args.extractor)
            print(f"[push] {model}: {n} new layer(s) from {src}")
        elif args.command == "pull":
            if len(args.dir) != 1:
                parser.error("pull takes exactly one --dir")
            dest = args.dir[0] / fname
            if dest.exists():
                print(f"[pull] {model}: {dest.name} already present")
            elif pull_model(root, model, variant, dest):
                print(f"[pull] {model}: rebuilt {dest.name} from cache")
            else:
                print(f"[pull] {model}: not (fully) cached -> will be extracted")


if __name__ == "__main__":
    main()
