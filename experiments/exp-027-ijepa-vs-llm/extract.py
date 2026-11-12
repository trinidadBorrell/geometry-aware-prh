"""exp-027: mean-pooled ViT features of I-JEPA, MAE and DINOv2 on WIT-1024 (D1) and D2.

Every block output, mean over patch tokens (prefix tokens excluded), as exp-015's `pool-avg`:
- I-JEPA (HF `IJepaModel`, no CLS token): `hidden_states[1:]`, the outputs of the 32 blocks before
  the final layernorm. Fed MAE's timm eval transform (`--preproc timm`, the main result) or the HF
  repo's own `ViTImageProcessor` (`--preproc hf`, mean/std 0.5, a check on D1 only).
- timm ViTs: `blocks.{i}.add_1` (exp-015 `extract_vit` recipe).

Outputs
- D1 -> the activation cache, one tensor per layer with a meta.json (`minhuh_prh_wit_1024_pool-avg`;
  the HF-processor check as `pool-avg-hfproc`). Existing entries are never overwritten: timm models
  whose D1 features exist (exp-015) are recomputed and compared with the cache instead.
- D2 -> `<out>/features/<key>__d2s.pt` (first 1024, all layers) and `__d2l.pt` (10240, 7 layers),
  exp-016's format, with key `<model>__avg`.

    uv run python experiments/exp-027-ijepa-vs-llm/extract.py --out <dir> --d2 <exp016>/d2.parquet
    uv run python experiments/exp-027-ijepa-vs-llm/extract.py --check   # tiny random I-JEPA
"""

from __future__ import annotations

import argparse
import gc
import getpass
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from geoprh import activation_cache as ac

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IJEPA = ["facebook/ijepa_vith14_1k", "facebook/ijepa_vith14_22k"]
TIMM = ["vit_huge_patch14_224.mae", "vit_giant_patch14_dinov2.lvd142m"]
TRANSFORM_FROM = "vit_huge_patch14_224.mae"  # I-JEPA gets MAE's exact eval transform
D1_VARIANT = {
    "timm": ac.Variant(pool="avg", caption_idx=None, modality="vision"),
    "hf": ac.Variant(pool="avg-hfproc", caption_idx=None, modality="vision"),
}
LAYER_DEF = {
    "hf": "HF IJepaModel hidden_states[i + 1] (block output, before the final layernorm), "
    "mean of the patch tokens (no prefix tokens)",
    "timm": "timm ViT blocks.{i}.add_1 (block output), mean of the patch tokens "
    "(prefix tokens excluded)",
}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def timm_transform(name: str = TRANSFORM_FROM):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    cfg = timm.models.get_pretrained_cfg(name).to_dict()
    return create_transform(**resolve_data_config(pretrained_cfg=cfg))


def hf_transform(name: str):
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(name)
    return lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def ijepa_mean(model, transform, images, batch: int, device: str, desc: str) -> torch.Tensor:
    """[n, 32, d]: every block output, mean over the 256 patch tokens."""
    feats = []
    for i in trange(0, len(images), batch, desc=desc):
        x = torch.stack([transform(im.convert("RGB")) for im in images[i : i + batch]]).to(device)
        out = model(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states[1:]
        feats.append(torch.stack([h.mean(1) for h in hs], 1).float().cpu())
    return torch.cat(feats)


def check_ijepa(model, device: str) -> dict:
    """Final layernorm of the last block output is `last_hidden_state`; no prefix token."""
    torch.manual_seed(0)
    size = model.config.image_size
    x = torch.randn(2, 3, size, size, device=device)
    with torch.no_grad():
        out = model(pixel_values=x, output_hidden_states=True)
    n_patch = (size // model.config.patch_size) ** 2
    diff = float((model.layernorm(out.hidden_states[-1]) - out.last_hidden_state).abs().max())
    report = {
        "tokens": int(out.hidden_states[0].shape[1]),
        "patches": n_patch,
        "n_hidden_states": len(out.hidden_states),
        "layernorm_max_abs_diff": diff,
    }
    assert report["tokens"] == n_patch, report
    assert report["n_hidden_states"] == model.config.num_hidden_layers + 1, report
    assert diff < 1e-4, report
    return report


def load_ijepa(name: str, device: str):
    from transformers import IJepaModel

    model = IJepaModel.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    return model


def write_d1(root: Path, model: str, variant: ac.Variant, feats: torch.Tensor, extras: dict) -> int:
    """One cache entry per layer, never overwriting (activation_cache's layout and meta keys)."""
    written = 0
    for i in range(feats.shape[1]):
        pt, meta_path = ac.layer_paths(root, model, i, variant)
        if pt.exists() and meta_path.exists():
            continue
        pt.parent.mkdir(parents=True, exist_ok=True)
        x = feats[:, i].contiguous().clone()
        ac._atomic_save(x, pt)
        meta = {
            "model": model,
            "modality": "vision",
            "layer": i,
            "num_layers": feats.shape[1],
            "layer_def": extras["layer_def"],
            "dataset": variant.dataset,
            "subset": variant.subset,
            "pool": variant.pool,
            "caption_idx": None,
            "seed": None,
            "dtype": str(x.dtype).removeprefix("torch."),
            "shape": list(x.shape),
            "num_params": extras["num_params"],
            "extras": {},
            "preprocessing": extras["preprocessing"],
            "extractor": "experiments/exp-027-ijepa-vs-llm/extract.py",
            "created_by": os.environ.get("PERSON") or getpass.getuser(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        ac._atomic_write_json(meta, meta_path)
        written += 1
    return written


def compare_cached(root: Path, model: str, variant: ac.Variant, feats: torch.Tensor) -> float:
    payload = ac.load_model(root, model, variant)
    assert payload is not None, f"{model} {variant.key} not cached"
    cached = payload["feats"].float()
    assert cached.shape == feats.shape, (cached.shape, feats.shape)
    return float(F.cosine_similarity(cached, feats, dim=2).min())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path)
    p.add_argument("--d2", type=Path, help="exp-016 d2.parquet")
    p.add_argument("--root", type=Path, default=None, help="activation cache")
    p.add_argument("--models", type=lambda s: s.split(","), default=IJEPA + TIMM)
    p.add_argument("--sets", default="d1,d2")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first n images")
    p.add_argument("--check", action="store_true", help="tiny random I-JEPA, no download")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check:
        from transformers import IJepaConfig, IJepaModel

        cfg = IJepaConfig(
            hidden_size=64, num_hidden_layers=3, num_attention_heads=4, intermediate_size=128
        )
        print(json.dumps(check_ijepa(IJepaModel(cfg).eval().to(device), device)))
        return

    M16 = _module("exp016_extract", ROOT / "exp-016-positive-controls" / "extract.py")
    X15 = _module("exp015_extract", ROOT / "exp-015-matched-pooling" / "extract.py")
    root = args.root or ac.default_root()
    sets = args.sets.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    checks_path = args.out / "extract_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}

    def save_checks():
        checks_path.write_text(json.dumps(checks, indent=1))

    images = {}
    if "d1" in sets:
        _, d1 = M16.load_d1()
        images["d1"] = d1[: args.limit] if args.limit else d1
    if "d2" in sets:
        _, _, d2 = M16.load_d2(args.d2, args.limit or M16.N_LARGE)
        images["d2"] = d2

    def d2_done(key: str) -> bool:
        return all((args.out / "features" / f"{key}__{t}.pt").exists() for t in ("d2s", "d2l"))

    for name in args.models:
        kind = "hf" if name in IJEPA else "timm"
        key = f"{name.replace('/', '__')}__avg"
        if kind == "hf":
            model = load_ijepa(name, device)
            checks[name] = {"model_check": check_ijepa(model, device)}
            n_params = sum(p.numel() for p in model.parameters())
            tf = timm_transform()
            for tag, ims in images.items():
                if tag == "d2" and d2_done(key):
                    continue
                if tag == "d1" and ac.cached_layers(root, name, D1_VARIANT["timm"]):
                    log(f"{name} d1 already cached")
                    continue
                feats = ijepa_mean(model, tf, ims, args.batch, device, f"{name} {tag}")
                extras = {
                    "num_params": n_params,
                    "layer_def": LAYER_DEF["hf"],
                    "preprocessing": f"timm eval transform of {TRANSFORM_FROM}",
                }
                if tag == "d1":
                    if args.limit:
                        log(f"smoke test: d1 {tuple(feats.shape)} not written")
                    else:
                        n = write_d1(root, name, D1_VARIANT["timm"], feats, extras)
                        log(f"{name} d1: {n} layers written")
                else:
                    meta = {"model": name, "pool": "avg", **extras}
                    M16.save(args.out, key, "d2", feats, None, meta)
                del feats
            hf_check = "d1" in sets and name == IJEPA[0]  # the shipped processor, as a check
            if hf_check and not ac.cached_layers(root, name, D1_VARIANT["hf"]):
                feats = ijepa_mean(
                    model, hf_transform(name), images["d1"], args.batch, device, "hf proc"
                )
                extras = {
                    "num_params": n_params,
                    "layer_def": LAYER_DEF["hf"],
                    "preprocessing": f"{name} ViTImageProcessor as shipped",
                }
                if not args.limit:
                    write_d1(root, name, D1_VARIANT["hf"], feats, extras)
                del feats
        else:
            model = None
            for tag, ims in images.items():
                if tag == "d2" and d2_done(key):
                    continue
                feats, extras = X15.extract_vit(name, ims, "avg", args.batch, device)
                if tag == "d1":
                    if ac.cached_layers(root, name, D1_VARIANT["timm"]) and not args.limit:
                        cos = compare_cached(root, name, D1_VARIANT["timm"], feats)
                        checks.setdefault(name, {})["d1_vs_exp015_min_cosine"] = cos
                        log(f"{name} d1 vs exp-015 cache: min row cosine {cos:.6f}")
                        assert cos >= 0.999, "mean-pooled features differ from exp-015's cache"
                    elif not args.limit:
                        extras = {**extras, "layer_def": LAYER_DEF["timm"]}
                        extras["preprocessing"] = "timm eval transform"
                        write_d1(root, name, D1_VARIANT["timm"], feats, extras)
                else:
                    meta = {"model": name, "pool": "avg", **extras}
                    M16.save(args.out, key, "d2", feats, None, meta)
                del feats
        save_checks()
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    save_checks()
    log("extract done")


if __name__ == "__main__":
    main()
