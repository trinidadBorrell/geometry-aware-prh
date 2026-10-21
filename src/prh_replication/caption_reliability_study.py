"""Fit caption-reliability filters on frozen language embeddings and score held-out alignment."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from prh_replication.caption_reliability import (
    METHODS,
    PRIMARY,
    apply_scales,
    centred_dot_gap,
    effective_rank,
    feature_norm_stats,
    feature_scales,
    features_vanished,
    fit_pca,
    gram_offdiag_fraction,
    loglog_fit,
    project,
    regroup,
    stabilizer,
    variance_components,
)
from prh_replication.feature_prep import clip_threshold, prepare_with_threshold, sha256_array
from prh_replication.extract import _device_for, _dtype
from prh_replication.extract_final import PROTOCOL, _decoder_layers, _pool_mask_mean, spec_from_manifest
from prh_replication.io_utils import jsonable, load_features, sha256_file, write_json
from prh_replication.registry import Paths
from prh_replication.spectral_calibration import feature_scores_rows, mnn_rows

CONFIG_NAME = "caption_reliability_filter_v1.json"
CACHE_DIR = "coco_val2017_caption_reliability_v1"


def run(work: Path, repo: Path, stage: str, force: bool = False) -> dict:
    started = time.perf_counter()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    out = paths.results / "caption_reliability_filter_v1"
    out.mkdir(parents=True, exist_ok=True)
    spec = repo / "results" / "caption_reliability_filter_v1" / "SPECIFICATION.md"
    if not spec.exists():
        raise FileNotFoundError(f"missing frozen specification {spec}")
    (out / "SPECIFICATION.md").write_text(spec.read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if stage == "full" and (out / "summary.json").exists() and not force:
        print("caption reliability filter already complete; pass --force to rerun", flush=True)
        return json.loads((out / "summary.json").read_text())
    panel = _panel(repo, config)
    splits = _splits(repo)
    captions = _captions(paths, splits, int(config["captions_per_image"]))
    _write_manifest(out, paths, repo, config, splits, captions)
    if stage == "extract":
        _extract_languages(paths, panel, captions, config["languages"])
        return {"extracted": config["languages"]}
    if stage == "pilot":
        _extract_languages(paths, panel, captions, [config["pilot_language"]])
        summary = _score(paths, panel, captions, config, out / "pilot", [config["pilot_language"]], [config["pilot_vision"]])
    elif stage == "full":
        _extract_languages(paths, panel, captions, config["languages"])
        summary = _score(paths, panel, captions, config, out, config["languages"], config["visions"])
    else:
        raise ValueError(f"unknown stage {stage}")
    summary["seconds"] = time.perf_counter() - started
    summary["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    summary["python"] = platform.python_version()
    summary["numpy"] = np.__version__
    summary["torch"] = torch.__version__
    write_json(out / "summary.json" if stage == "full" else out / "pilot" / "summary.json", summary)
    if stage == "full":
        _write_report(out / "REPORT.md", summary)
    print(json.dumps({"stage": stage, "seconds": summary["seconds"], "primary_cka_delta": summary["answers"]["primary_minus_identity_cka"]}), flush=True)
    return summary


def _panel(repo: Path, config: dict) -> dict:
    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    rows = {row["key"]: row for row in manifest["base_panel"] + manifest["vision_anchors"]}
    missing = [key for key in config["languages"] + config["visions"] if key not in rows]
    if missing:
        raise KeyError(f"panel keys missing from release_models.json: {missing}")
    return rows


def _splits(repo: Path) -> dict[str, list[str]]:
    payload = json.loads((repo / "data" / "manifests" / "coco_val2017_splits.json").read_text())
    splits = payload["splits"]
    if set(splits["train"]) & set(splits["test"]):
        raise RuntimeError("train and test image ids overlap")
    if len(splits["train"]) != 2048 or len(splits["test"]) != 1024:
        raise RuntimeError("unexpected split sizes")
    return splits


def _captions(paths: Paths, splits: dict[str, list[str]], n_captions: int) -> dict:
    coco_path = paths.data / "raw" / "coco" / "annotations" / "captions_val2017.json"
    coco = json.loads(coco_path.read_text())
    by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    chosen = {}
    extra = 0
    short = []
    for split, sample_ids in splits.items():
        if split not in ("train", "test"):
            continue
        rows = []
        for sample_id in sample_ids:
            image_id = int(sample_id.rsplit("-", 1)[1])
            annotations = sorted(by_image.get(image_id, []), key=lambda row: int(row["id"]))
            if len(annotations) < n_captions:
                short.append(sample_id)
                continue
            extra += int(len(annotations) > n_captions)
            keep = annotations[:n_captions]
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "caption_ids": [int(row["id"]) for row in keep],
                    "captions": [row["caption"] for row in keep],
                }
            )
        chosen[split] = rows
    if short:
        raise RuntimeError(f"{len(short)} images have fewer than {n_captions} captions; the balanced rule excludes them")
    return {"rows": chosen, "n_images_with_extra_captions": extra, "n_excluded_short": 0, "coco_sha256": sha256_file(coco_path)}


def _write_manifest(out: Path, paths: Paths, repo: Path, config: dict, splits: dict, captions: dict) -> None:
    payload = {
        "dataset": "coco_val2017",
        "fit_split": "train",
        "eval_split": "test",
        "n_train": len(captions["rows"]["train"]),
        "n_test": len(captions["rows"]["test"]),
        "n_val_unused": len(splits["val"]),
        "n_heldout_unused": len(splits["heldout"]),
        "n_images_with_extra_captions": captions["n_images_with_extra_captions"],
        "n_excluded_short": captions["n_excluded_short"],
        "caption_rule": config["caption_rule"],
        "train_ids_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in captions["rows"]["train"]).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in captions["rows"]["test"]).encode()).hexdigest(),
        "coco_sha256": captions["coco_sha256"],
        "split_manifest_sha256": sha256_file(repo / "data" / "manifests" / "coco_val2017_splits.json"),
        "config_sha256": sha256_file(repo / "configs" / CONFIG_NAME),
        "languages": config["languages"],
        "visions": config["visions"],
    }
    write_json(out / "manifest.json", payload)


def _extract_languages(paths: Paths, panel: dict, captions: dict, languages: list[str]) -> None:
    for key in languages:
        for split in ("train", "test"):
            _extract_one(paths, spec_from_manifest(panel[key]), captions["rows"][split], split)


def _cache_path(paths: Paths, key: str, split: str) -> Path:
    return paths.features / CACHE_DIR / split / key / "captions.pt"


def _extract_one(paths: Paths, spec, rows: list[dict], split: str) -> Path:
    dest = _cache_path(paths, spec.key, split)
    sample_ids = [row["sample_id"] for row in rows]
    caption_ids = [row["caption_ids"] for row in rows]
    if dest.exists():
        cached = load_features(dest)
        if list(cached["sample_ids"]) != sample_ids or cached.get("caption_ids") != caption_ids:
            raise RuntimeError(f"caption cache does not match the frozen ids: {dest}")
        if tuple(cached["feats"].shape[:2]) != (len(rows), len(rows[0]["captions"])):
            raise RuntimeError(f"caption cache has the wrong shape: {dest}")
        return dest
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    texts = [caption for row in rows for caption in row["captions"]]
    n_params_est = (spec.n_params or 8_000_000_000) * 2
    device = _device_for("language", n_params_est)
    if device == "cuda":
        free_bytes, _total = torch.cuda.mem_get_info()
        if free_bytes < 18 * (1 << 30):
            print(f"{spec.key} {split} GPU free {free_bytes / (1 << 30):.1f} GiB; using CPU", flush=True)
            device = "cpu"
    dtype = torch.float16 if device == "cpu" else _dtype(device)
    batch_size = 8 if device == "cpu" else 16
    if device == "cpu":
        torch.set_num_threads(16)
    print(f"{spec.key} {split} device={device} dtype={dtype} captions={len(texts)}", flush=True)
    revision = spec.revision if spec.revision and len(spec.revision) >= 16 else None
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kw = dict(revision=revision, dtype=dtype, device_map=None, trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.checkpoint, **load_kw)
        loader = "causal"
    except Exception as exc:
        loader = f"automodel_after_causal:{type(exc).__name__}"
        model = AutoModel.from_pretrained(spec.checkpoint, **load_kw)
    try:
        model = model.to(device).eval()
    except Exception:
        device = "cpu"
        dtype = torch.float16
        model = model.to(dtype=dtype, device="cpu").eval()
    layers, layer_path = _decoder_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden

    handle = layers[-1].register_forward_hook(_hook)
    chunks = []
    try:
        with torch.no_grad():
            started = time.perf_counter()
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[start : start + batch_size], padding=True, truncation=False, return_tensors="pt")
                tokens = {name: value.to(device) for name, value in tokens.items()}
                captured.clear()
                _ = model(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))
                if "h" not in captured or captured["h"].dim() != 3:
                    raise RuntimeError(f"final residual hook failed for {spec.key}")
                chunks.append(_pool_mask_mean(captured["h"], tokens["attention_mask"]).cpu())
                done = start + tokens["input_ids"].shape[0]
                if done <= batch_size * 2 or done % (batch_size * 20) == 0:
                    rate = done / max(time.perf_counter() - started, 1e-6)
                    print(f"{spec.key} {split} {done}/{len(texts)} {rate:.2f} captions/s", flush=True)
    finally:
        handle.remove()
    feats = torch.cat(chunks, dim=0)
    expected = (len(rows), len(rows[0]["captions"]), int(feats.shape[1]))
    feats = feats.reshape(expected)
    if not torch.isfinite(feats.float()).all():
        raise RuntimeError(f"non-finite captions for {spec.key}")
    payload = {
        "feats": feats.half(),
        "sample_ids": sample_ids,
        "caption_ids": caption_ids,
        "protocol": PROTOCOL,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "layer_path": layer_path,
        "hidden_dim": int(feats.shape[-1]),
        "device": device,
        "dtype": str(dtype),
        "loader": loader,
        "model_class": type(model).__name__,
        "pooling": "mask_mean",
        "caption_rule": "smallest five COCO annotation ids",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(dest)
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"wrote {dest}", flush=True)
    return dest


def _load_captions(paths: Paths, key: str, split: str, sample_ids: list[str]) -> np.ndarray:
    payload = load_features(_cache_path(paths, key, split))
    if list(payload["sample_ids"]) != sample_ids:
        raise RuntimeError(f"{key} {split} caption order left the manifest")
    feats = payload["feats"]
    array = feats.float().numpy() if torch.is_tensor(feats) else np.asarray(feats)
    return np.asarray(array, dtype=np.float64)


def _load_archived(paths: Paths, key: str, split: str) -> tuple[np.ndarray, list[str]]:
    directory = paths.features / f"coco_val2017_{PROTOCOL}" / split / key
    files = sorted(directory.glob("*.pt"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one archived cache in {directory}")
    payload = load_features(files[0])
    feats = payload["feats"]
    array = feats.float().numpy() if torch.is_tensor(feats) else np.asarray(feats)
    if array.ndim == 3:
        array = array[:, 0, :]
    return np.asarray(array, dtype=np.float64), list(payload["sample_ids"])


def _prepare(raw: np.ndarray, threshold: float) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))
    return prepare_with_threshold(tensor, threshold).numpy().astype(np.float64)


def _vision_thresholds(paths: Paths, visions: list[str]) -> dict[str, float]:
    lock = json.loads((paths.results / "cross_dataset_metric_transfer" / "LOCK.json").read_text())
    thresholds = lock["clip_thresholds"]
    missing = [key for key in visions if key not in thresholds]
    if missing:
        raise KeyError(f"archived vision clip thresholds missing: {missing}")
    return {key: float(thresholds[key]) for key in visions}


def _score(paths: Paths, panel: dict, captions: dict, config: dict, out: Path, languages: list[str], visions: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train_ids = [row["sample_id"] for row in captions["rows"]["train"]]
    test_ids = [row["sample_id"] for row in captions["rows"]["test"]]
    if set(train_ids) & set(test_ids):
        raise RuntimeError("fitting images overlap the test gallery")
    vision_thresholds = _vision_thresholds(paths, visions)
    vision_test = {}
    for key in visions:
        raw, ids = _load_archived(paths, key, "test")
        order = {sample_id: index for index, sample_id in enumerate(ids)}
        if any(sample_id not in order for sample_id in test_ids):
            raise RuntimeError(f"{key} archived test cache is missing gallery ids")
        stacked = raw[[order[sample_id] for sample_id in test_ids]]
        vision_test[key] = _prepare(stacked, vision_thresholds[key])
    rows: list[dict] = []
    diagnostics = []
    checks = []
    filter_hashes = {}
    modes_path = out / "modes.csv"
    if modes_path.exists():
        modes_path.unlink()
    n_test = len(test_ids)
    take = int(round(float(config["subsample_fraction"]) * n_test))
    generator = np.random.Generator(np.random.PCG64(int(config["subsample_seed"])))
    draws = [generator.choice(n_test, size=take, replace=False) for _ in range(int(config["subsample_count"]))]
    np.save(out / "subsample_indices.npy", np.stack(draws))
    subsample_rows: list[dict] = []
    for language in languages:
        print(f"fit {language}", flush=True)
        raw_train = _load_captions(paths, language, "train", train_ids)
        raw_test = _load_captions(paths, language, "test", test_ids)
        archived, archived_ids = _load_archived(paths, language, "test")
        archived_order = {sample_id: index for index, sample_id in enumerate(archived_ids)}
        archived_aligned = archived[[archived_order[sample_id] for sample_id in test_ids]]
        agreement = float(np.max(np.abs(raw_test[:, 0, :] - archived_aligned)))
        n_images, n_captions, width = raw_train.shape
        flat_train = raw_train.reshape(n_images * n_captions, width)
        threshold = float(clip_threshold(torch.from_numpy(np.ascontiguousarray(flat_train, dtype=np.float32)), float(config["clip_q"])))
        prepared_train = _prepare(flat_train, threshold).reshape(n_images, n_captions, width)
        prepared_test = _prepare(raw_test.reshape(raw_test.shape[0] * n_captions, width), threshold).reshape(raw_test.shape[0], n_captions, width)
        basis = fit_pca(prepared_train.reshape(n_images * n_captions, width), rank_tol=float(config["pca_rank_tol"]))
        train_scores = project(prepared_train.reshape(n_images * n_captions, width), basis["mean"], basis["components"]).reshape(n_images, n_captions, int(basis["width"]))
        parts = variance_components(train_scores, denominator_atol=float(config["denominator_atol"]))
        if float(np.max(parts["decomposition_abs"])) > float(config["decomposition_atol"]):
            raise RuntimeError(f"{language} variance decomposition failed")
        tau = stabilizer(parts["lambda"], coef=float(config["tau_coef"]))
        scales = {method: feature_scales(method, parts["w"], parts["lambda"], tau) for method in METHODS if method != "ORIGINAL"}
        probe = prepared_train.reshape(n_images * n_captions, width)[:64]
        dot_gap = centred_dot_gap(probe, basis["mean"], basis["components"]) if int(basis["width"]) == width else float("nan")
        digest = hashlib.sha256(np.ascontiguousarray(scales[PRIMARY]).tobytes()).hexdigest()
        filter_hashes[language] = digest
        _save_filter(out, language, threshold, basis, parts, scales, tau, digest)
        _append_modes(out, language, parts, scales)
        test_scores = project(prepared_test.reshape(prepared_test.shape[0] * n_captions, width), basis["mean"], basis["components"]).reshape(prepared_test.shape[0], n_captions, int(basis["width"]))
        shuffled_scales = {}
        for seed in config["shuffle_seeds"]:
            shuffled = variance_components(regroup(train_scores, int(seed)), denominator_atol=float(config["denominator_atol"]))
            shuffled_scales[int(seed)] = feature_scales(PRIMARY, shuffled["w"], shuffled["lambda"], tau)
            diagnostics.append(
                {
                    "language": language,
                    "shuffle_seed": int(seed),
                    "zero_weight_fraction": float(np.mean(shuffled_scales[int(seed)] == 0.0)),
                    "negative_signal": int(np.sum(shuffled["negative_signal"])),
                    "mean_reliability": float(np.mean(shuffled["w"])),
                }
            )
        fit_record = {
            "language": language,
            "clip_threshold": threshold,
            "ambient_dim": width,
            "pca_width": int(basis["width"]),
            "numerical_rank": int(basis["numerical_rank"]),
            "tau": tau,
            "negative_signal": int(np.sum(parts["negative_signal"])),
            "excluded_modes": int(np.sum(parts["excluded_mode"])),
            "zero_weight_fraction": float(np.mean(parts["w"] == 0.0)),
            "mean_reliability": float(np.mean(parts["w"])),
            "caption0_max_abs_vs_archive": agreement,
            "centred_dot_gap_64": dot_gap,
            "loglog": loglog_fit(parts["B"], parts["lambda"]),
            "filter_sha256": digest,
        }
        diagnostics.append(fit_record)
        checks.append(fit_record)
        for vision in visions:
            print(f"score {language} {vision}", flush=True)
            target = vision_test[vision]
            for position in range(n_captions):
                original = prepared_test[:, position, :]
                geometry = position == 0
                rows.append(_row(language, vision, "ORIGINAL", "single", position, None, original, target, int(config["knn_k"]), geometry))
                for method, scale in scales.items():
                    transformed = apply_scales(test_scores[:, position, :], scale)
                    rows.append(_row(language, vision, method, "single", position, None, transformed, target, int(config["knn_k"]), geometry))
                for seed, scale in shuffled_scales.items():
                    transformed = apply_scales(test_scores[:, position, :], scale)
                    rows.append(_row(language, vision, PRIMARY, "shuffle", position, seed, transformed, target, int(config["knn_k"]), False))
            for draw, index in enumerate(draws):
                subsample_rows.append(_sub_row(language, vision, "ORIGINAL", draw, prepared_test[index, 0, :], target[index], int(config["knn_k"])))
                for method, scale in scales.items():
                    subsample_rows.append(_sub_row(language, vision, method, draw, apply_scales(test_scores[index, 0, :], scale), target[index], int(config["knn_k"])))
            original_centroid = prepared_test.mean(axis=1)
            rows.append(_row(language, vision, "ORIGINAL", "centroid", -1, None, original_centroid, target, int(config["knn_k"]), False))
            identity_centroid = test_scores.mean(axis=1)
            rows.append(_row(language, vision, "PCA_IDENTITY", "centroid", -1, None, identity_centroid, target, int(config["knn_k"]), False))
            filtered_centroid = apply_scales(identity_centroid, scales[PRIMARY])
            rows.append(_row(language, vision, PRIMARY, "centroid", -1, None, filtered_centroid, target, int(config["knn_k"]), False))
        del raw_train, prepared_train, train_scores
        gc.collect()
    reused = all(row["filter_sha256"] == filter_hashes[row["language"]] for row in checks)
    subsample = _summarize_subsample(subsample_rows)
    _write_rows(out / "metrics.csv", rows)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "checks.json", {"fits": checks, "filter_reused_across_visions": reused, "train_test_disjoint": True})
    summary = _summarize(config, rows, diagnostics, subsample, captions)
    write_json(out / "subsample_summary.json", subsample)
    write_json(out / "summary.json", summary)
    try:
        _figures(out, diagnostics, rows, languages)
    except Exception as exc:
        (out / "figure_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    return summary


def _summarize_subsample(collected: list[dict]) -> dict:
    """Shared 80% image subsamples of caption position 0. Variability, not a confidence interval."""
    summary = []
    for method in METHODS:
        cka = [row["cka"] for row in collected if row["method"] == method and row["cka"] is not None]
        mnn = [row["mnn"] for row in collected if row["method"] == method and row["mnn"] is not None]
        summary.append(
            {
                "method": method,
                "cka_mean": float(np.mean(cka)) if cka else None,
                "cka_std": float(np.std(cka)) if cka else None,
                "mnn_mean": float(np.mean(mnn)) if mnn else None,
                "mnn_std": float(np.std(mnn)) if mnn else None,
                "n": len(cka),
            }
        )
    return {"label": "20 shared 80% image subsamples; not a confidence interval", "draws": summary}


def _save_filter(out, language, threshold, basis, parts, scales, tau, digest) -> None:
    dest = out / "filters" / f"{language}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_threshold": np.asarray(threshold),
        "mean": basis["mean"],
        "components": basis["components"],
        "singular_values": basis["singular_values"],
        "lambda": parts["lambda"],
        "W": parts["W"],
        "Vmean": parts["Vmean"],
        "B_unclipped": parts["B_unclipped"],
        "B": parts["B"],
        "w": parts["w"],
        "negative_signal": parts["negative_signal"],
        "excluded_mode": parts["excluded_mode"],
        "tau": np.asarray(tau),
        "filter_sha256": np.asarray(digest),
    }
    for method, scale in scales.items():
        payload[f"scale_{method}"] = scale
    np.savez_compressed(dest, **payload)


def _append_modes(out: Path, language: str, parts: dict, scales: dict) -> None:
    dest = out / "modes.csv"
    new_file = not dest.exists()
    with dest.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["language", "mode", "lambda", "W", "Vmean", "B_unclipped", "B", "w", "negative_signal", "excluded", "scale_RELIABILITY_SQRT", "scale_RELIABILITY_STRONG", "scale_RELIABILITY_WHITENED", "scale_WHITENED_CONTROL"])
        for index in range(parts["lambda"].shape[0]):
            writer.writerow(
                [
                    language,
                    index,
                    parts["lambda"][index],
                    parts["W"][index],
                    parts["Vmean"][index],
                    parts["B_unclipped"][index],
                    parts["B"][index],
                    parts["w"][index],
                    int(parts["negative_signal"][index]),
                    int(parts["excluded_mode"][index]),
                    scales["RELIABILITY_SQRT"][index],
                    scales["RELIABILITY_STRONG"][index],
                    scales["RELIABILITY_WHITENED"][index],
                    scales["WHITENED_CONTROL"][index],
                ]
            )


def _sub_row(language, vision, method, draw: int, features, target, k: int) -> dict:
    scored = _score_features(features, target, k, geometry=False)
    return {"language": language, "vision": vision, "method": method, "draw": draw, "cka": scored["cka"], "mnn": scored["mnn"]}


def _row(language, vision, method, setting, position, seed, features, target, k: int, geometry: bool) -> dict:
    scored = _score_features(features, target, k, geometry=geometry)
    return {
        "language": language,
        "vision": vision,
        "method": method,
        "setting": setting,
        "position": position,
        "shuffle_seed": seed,
        **scored,
    }


def _score_features(features: np.ndarray, target: np.ndarray, k: int, geometry: bool = True) -> dict:
    norms = feature_norm_stats(features)
    blank = {
        "cka": None,
        "excess": None,
        "permutation_baseline": None,
        "mnn": None,
        "undefined": True,
        "effective_rank": None,
        "offdiag_fraction": None,
        **norms,
    }
    if features_vanished(features):
        return blank
    scored = feature_scores_rows(features, target)
    if not bool(scored["valid_a"]):
        return blank
    return {
        "cka": float(scored["a"]),
        "excess": float(scored["excess"]),
        "permutation_baseline": float(scored["b"]),
        "mnn": float(mnn_rows(features, target, k)),
        "undefined": False,
        "effective_rank": float(effective_rank(features)) if geometry else None,
        "offdiag_fraction": float(gram_offdiag_fraction(features)) if geometry else None,
        **norms,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["language", "vision", "method", "setting", "position", "shuffle_seed", "cka", "excess", "permutation_baseline", "mnn", "undefined", "effective_rank", "offdiag_fraction", "mean", "std", "max"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _select(rows: list[dict], **filters) -> list[dict]:
    chosen = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            chosen.append(row)
    return chosen


def _pair_key(row: dict) -> tuple[str, str]:
    return row["language"], row["vision"]


def _paired_delta(rows: list[dict], left: dict, right: dict, field: str) -> dict:
    a = {_pair_key(row): row[field] for row in _select(rows, **left) if row[field] is not None}
    b = {_pair_key(row): row[field] for row in _select(rows, **right) if row[field] is not None}
    common = sorted(set(a) & set(b))
    deltas = [a[key] - b[key] for key in common]
    return {
        "n_pairs": len(deltas),
        "mean": float(np.mean(deltas)) if deltas else None,
        "median": float(np.median(deltas)) if deltas else None,
        "n_positive": int(sum(value > 0.0 for value in deltas)),
        "n_negative": int(sum(value < 0.0 for value in deltas)),
        "per_pair": [{"language": key[0], "vision": key[1], "delta": value, "left": a[key], "right": b[key]} for key, value in zip(common, deltas)],
    }


def _mean_metric(rows: list[dict], field: str, **filters) -> float | None:
    values = [row[field] for row in _select(rows, **filters) if row[field] is not None]
    if not values:
        return None
    return float(np.mean(values))


def _summarize(config: dict, rows: list[dict], diagnostics: list[dict], subsample: dict, captions: dict) -> dict:
    primary_cka = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": 0, "shuffle_seed": None}, "cka")
    primary_mnn = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": 0, "shuffle_seed": None}, "mnn")
    versus_original = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "ORIGINAL", "setting": "single", "position": 0, "shuffle_seed": None}, "cka")
    whitening = _paired_delta(rows, {"method": "RELIABILITY_WHITENED", "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "WHITENED_CONTROL", "setting": "single", "position": 0, "shuffle_seed": None}, "cka")
    whitening_mnn = _paired_delta(rows, {"method": "RELIABILITY_WHITENED", "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "WHITENED_CONTROL", "setting": "single", "position": 0, "shuffle_seed": None}, "mnn")
    positions = []
    for position in range(int(config["captions_per_image"])):
        delta = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": position, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": position, "shuffle_seed": None}, "cka")
        mnn = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": position, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": position, "shuffle_seed": None}, "mnn")
        positions.append({"position": position, "cka_delta_mean": delta["mean"], "mnn_delta_mean": mnn["mean"], "cka_n_positive": delta["n_positive"], "n_pairs": delta["n_pairs"]})
    shuffles = []
    for seed in config["shuffle_seeds"]:
        delta = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": 0, "shuffle_seed": None}, {"method": PRIMARY, "setting": "shuffle", "position": 0, "shuffle_seed": int(seed)}, "cka")
        shuffles.append({"seed": int(seed), "genuine_minus_shuffle_cka": delta["mean"], "n_pairs": delta["n_pairs"]})
    averaging = _paired_delta(rows, {"method": "PCA_IDENTITY", "setting": "centroid", "position": -1, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": 0, "shuffle_seed": None}, "cka")
    filter_on_centroid = _paired_delta(rows, {"method": PRIMARY, "setting": "centroid", "position": -1, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "centroid", "position": -1, "shuffle_seed": None}, "cka")
    fits = [row for row in diagnostics if "pca_width" in row]
    rank_ratios = []
    for left in _select(rows, method=PRIMARY, setting="single", position=0, shuffle_seed=None):
        right = next(row for row in _select(rows, method="PCA_IDENTITY", setting="single", position=0, shuffle_seed=None, language=left["language"], vision=left["vision"]))
        if left["effective_rank"] and right["effective_rank"]:
            rank_ratios.append(left["effective_rank"] / right["effective_rank"])
    undefined = [row for row in rows if row["undefined"]]
    answers = {
        "primary_minus_identity_cka": primary_cka["mean"],
        "primary_minus_identity_mnn": primary_mnn["mean"],
        "primary_minus_original_cka": versus_original["mean"],
        "genuine_minus_shuffle_cka_mean": float(np.mean([row["genuine_minus_shuffle_cka"] for row in shuffles])) if shuffles else None,
        "reliability_whitened_minus_whitened_cka": whitening["mean"],
        "reliability_whitened_minus_whitened_mnn": whitening_mnn["mean"],
        "caption_position_cka_deltas": [row["cka_delta_mean"] for row in positions],
        "averaging_minus_single_identity_cka": averaging["mean"],
        "filter_minus_identity_on_centroids_cka": filter_on_centroid["mean"],
        "median_effective_rank_ratio": float(np.median(rank_ratios)) if rank_ratios else None,
        "n_undefined": len(undefined),
    }
    return {
        "config": config["name"],
        "n_train": len(captions["rows"]["train"]),
        "n_test": len(captions["rows"]["test"]),
        "n_extra_caption_images": captions["n_images_with_extra_captions"],
        "fits": fits,
        "primary_cka": primary_cka,
        "primary_mnn": primary_mnn,
        "versus_original_cka": versus_original,
        "whitening_cka": whitening,
        "whitening_mnn": whitening_mnn,
        "positions": positions,
        "shuffles": shuffles,
        "averaging_cka": averaging,
        "filter_on_centroid_cka": filter_on_centroid,
        "subsample": subsample,
        "answers": answers,
        "means": {
            method: {
                "cka": _mean_metric(rows, "cka", method=method, setting="single", position=0, shuffle_seed=None),
                "mnn": _mean_metric(rows, "mnn", method=method, setting="single", position=0, shuffle_seed=None),
                "excess": _mean_metric(rows, "excess", method=method, setting="single", position=0, shuffle_seed=None),
            }
            for method in METHODS
        },
    }


def _axes(n_panels: int):
    import matplotlib.pyplot as plt

    columns = 1 if n_panels <= 1 else min(3, n_panels)
    rows = int(math.ceil(n_panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.8 * rows), squeeze=False)
    flat = list(axes.ravel())
    for extra in flat[n_panels:]:
        extra.axis("off")
    return figure, flat[:n_panels]


def _figures(out: Path, diagnostics: list[dict], rows: list[dict], languages: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fits = [row for row in diagnostics if row.get("language") in languages and "pca_width" in row]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for language in languages:
        payload = np.load(out / "filters" / f"{language}.npz")
        axis.plot(np.arange(payload["w"].shape[0]), payload["w"], label=language, linewidth=0.8)
    axis.set_xlabel("PCA rank")
    axis.set_ylabel("Reliability w")
    axis.set_title("Caption reliability by PCA rank")
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "reliability_vs_rank.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for language in languages:
        payload = np.load(out / "filters" / f"{language}.npz")
        rank = np.arange(payload["B"].shape[0])
        axes[0].plot(rank, payload["B"], label=language, linewidth=0.8)
        axes[1].plot(rank, payload["W"], label=language, linewidth=0.8)
    axes[0].set_ylabel("Nonnegative signal B")
    axes[1].set_ylabel("Within-image variance W")
    axes[1].set_xlabel("PCA rank")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "signal_and_within_variance.png", dpi=140)
    plt.close(figure)

    figure, axes = _axes(len(languages))
    for axis, language in zip(axes, languages):
        payload = np.load(out / "filters" / f"{language}.npz")
        keep = (payload["B"] > 0.0) & (payload["lambda"] > 0.0)
        axis.scatter(payload["lambda"][keep], payload["B"][keep], s=4, alpha=0.5)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(language, fontsize=8)
    figure.suptitle("Positive B versus lambda")
    figure.tight_layout()
    figure.savefig(figure_dir / "signal_vs_lambda.png", dpi=140)
    plt.close(figure)

    figure, axes = _axes(len(languages))
    for axis, language in zip(axes, languages):
        payload = np.load(out / "filters" / f"{language}.npz")
        variance = payload["lambda"]
        axis.plot(np.sort(variance)[::-1], label="PCA", linewidth=0.8)
        transformed = variance * np.asarray(payload["scale_RELIABILITY_SQRT"]) ** 2
        axis.plot(np.sort(transformed)[::-1], label="sqrt reliability", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_title(language, fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    figure.suptitle("Training-coordinate variance spectra")
    figure.tight_layout()
    figure.savefig(figure_dir / "covariance_spectra.png", dpi=140)
    plt.close(figure)

    pairs = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": 0, "shuffle_seed": None}, "cka")
    mnn = _paired_delta(rows, {"method": PRIMARY, "setting": "single", "position": 0, "shuffle_seed": None}, {"method": "PCA_IDENTITY", "setting": "single", "position": 0, "shuffle_seed": None}, "mnn")
    labels = [f"{row['language'].split('-')[0]}\n{row['vision'].split('-')[0]}" for row in pairs["per_pair"]]
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
    axes[0].bar(np.arange(len(pairs["per_pair"])), [row["delta"] for row in pairs["per_pair"]], color="#0072B2")
    axes[1].bar(np.arange(len(mnn["per_pair"])), [row["delta"] for row in mnn["per_pair"]], color="#D55E00")
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_ylabel("CKA minus PCA identity")
    axes[1].set_ylabel("Mutual kNN minus PCA identity")
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=6)
    figure.tight_layout()
    figure.savefig(figure_dir / "heldout_deltas.png", dpi=140)
    plt.close(figure)
    del fits


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6f}"


def _write_report(path: Path, summary: dict) -> None:
    answers = summary["answers"]
    fits = summary["fits"]
    position_text = ", ".join(_fmt(value) for value in answers["caption_position_cka_deltas"])
    shuffle_text = ", ".join(f"seed {row['seed']}: {_fmt(row['genuine_minus_shuffle_cka'])}" for row in summary["shuffles"])
    fit_text = "; ".join(
        f"{row['language']} dim {row['ambient_dim']} kept {row['pca_width']}, mean w {row['mean_reliability']:.4f}, negative B {row['negative_signal']}, caption-0 max abs {row['caption0_max_abs_vs_archive']:.3e}"
        for row in fits
    )
    lines = [
        "# Caption reliability filter",
        "",
        "Partner-independent language filters, fit with image-group supervision from five COCO captions. Vision embeddings were not used to fit or select the transformation. This is not a wholly unsupervised result and it is not a nested confirmatory test.",
        "",
        f"Training images {summary['n_train']}. Test images {summary['n_test']}. Images with extra captions beyond five: {summary['n_extra_caption_images']}. No image with fewer than five captions was present, so none were excluded.",
        "",
        f"Fits: {fit_text}.",
        "",
        f"1. Primary sqrt-reliability versus unweighted PCA, caption position 0, mean paired CKA difference {_fmt(answers['primary_minus_identity_cka'])}. Versus the original clipped features: {_fmt(answers['primary_minus_original_cka'])}.",
        f"2. The same comparison for mutual kNN is {_fmt(answers['primary_minus_identity_mnn'])}.",
        f"3. Genuine grouping minus shuffled grouping, mean over the five seeds: {_fmt(answers['genuine_minus_shuffle_cka_mean'])}. Per seed: {shuffle_text}.",
        f"4. Reliability-weighted whitening minus whitening alone: CKA {_fmt(answers['reliability_whitened_minus_whitened_cka'])}, mutual kNN {_fmt(answers['reliability_whitened_minus_whitened_mnn'])}.",
        f"5. Mean paired CKA differences at the five caption positions: {position_text}. Pair counts are in `summary.json`. Pairs share models, so the mean is not eighteen independent replications.",
        f"6. Averaging five captions, unweighted PCA centroid minus the archived caption: {_fmt(answers['averaging_minus_single_identity_cka'])}. The reliability filter on those centroids, minus the unweighted centroid: {_fmt(answers['filter_minus_identity_on_centroids_cka'])}.",
        f"7. Median effective-rank ratio of the primary filter to unweighted PCA: {_fmt(answers['median_effective_rank_ratio'])}. Undefined scores: {answers['n_undefined']}. Scaling reweights modes; it does not raise the signal-to-noise ratio inside one nonzero mode.",
        "",
        "Subsample means are variability across 20 shared 80% image draws, not a confidence interval. Log-log slopes between positive B and lambda are descriptive fits in `summary.json` and are not evidence for a universal power law.",
        "",
        f"Runtime {summary.get('seconds', float('nan')):.1f} seconds. Peak RSS {summary.get('peak_rss_kb', 'unknown')} KB. Python {summary.get('python', '')}, numpy {summary.get('numpy', '')}.",
        "",
        "Reproduce with `PYTHONPATH=src python scripts/run_caption_reliability_filter.py --work /mnt/sdb1/prh-replication-work --stage full`.",
        "",
    ]
    path.write_text("\n".join(lines))
