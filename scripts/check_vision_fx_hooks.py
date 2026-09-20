#!/usr/bin/env python3
"""Bounded FX add_1 vs block-output CLS hook check (not a metric-parity test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.datasets import load_manifest, split_records
from prh_replication.registry import MODELS, Paths


VISION = ["dinov2-small", "vit-in21k-small", "clip-laion-base"]


def compare_model(spec, records, n_images: int, device: str) -> dict:
    import timm
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision.models.feature_extraction import create_feature_extractor

    model = timm.create_model(spec.checkpoint, pretrained=True).to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    n_blocks = len(model.blocks)
    captured: list[torch.Tensor] = [torch.empty(0) for _ in range(n_blocks)]

    def make_hook(i):
        def _hook(_m, _inp, output):
            captured[i] = output[:, 0, :].detach()

        return _hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.blocks)]
    return_nodes = [f"blocks.{i}.add_1" for i in range(n_blocks)]
    try:
        fx = create_feature_extractor(model, return_nodes=return_nodes)
    except Exception as e:
        for h in handles:
            h.remove()
        return {
            "model": spec.key,
            "checkpoint": spec.checkpoint,
            "error": repr(e),
            "material_mismatch": True,
            "note": "FX graph does not expose blocks.{i}.add_1; hook comparison failed",
        }

    ims = []
    for rec in records[:n_images]:
        with Image.open(rec["abs_path"]) as im:
            ims.append(transform(im.convert("RGB")))
    x = torch.stack(ims).to(device)
    with torch.no_grad():
        _ = model(x)
        hook = torch.stack(captured, dim=1).cpu()
        fx_out = fx(x)
        fx_cls = torch.stack([v[:, 0, :].cpu() for v in fx_out.values()], dim=1)
    for h in handles:
        h.remove()
    diff = (hook.float() - fx_cls.float()).abs()
    rel = diff / (fx_cls.float().abs() + 1e-8)
    return {
        "model": spec.key,
        "checkpoint": spec.checkpoint,
        "n_images": n_images,
        "n_blocks": n_blocks,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "mean_rel_diff": float(rel.mean()),
        "allclose_1e5": bool(torch.allclose(hook.float(), fx_cls.float(), atol=1e-5, rtol=1e-5)),
        "allclose_1e3": bool(torch.allclose(hook.float(), fx_cls.float(), atol=1e-3, rtol=1e-3)),
        "material_mismatch": float(diff.max()) > 1e-3,
        "note": "CLS of blocks[i] output vs torchvision FX nodes blocks.{i}.add_1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--n-images", type=int, default=8)
    args = p.parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    man = load_manifest(paths.data / "manifests" / "coco_val2017.json")
    recs = split_records(man, "train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for key in VISION:
        print("checking", key, flush=True)
        rows.append(compare_model(MODELS[key], recs, args.n_images, device))
        print(json.dumps(rows[-1], indent=2), flush=True)
    out = paths.results / "learned_kernels" / "hook_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    repo_out = ROOT / "results" / "learned_kernels"
    repo_out.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"rows": rows, "any_material_mismatch": any(r["material_mismatch"] for r in rows)}, indent=2)
    out.write_text(text + "\n")
    (repo_out / "hook_check.json").write_text(text + "\n")
    print("material_mismatch", any(r["material_mismatch"] for r in rows))


if __name__ == "__main__":
    main()
