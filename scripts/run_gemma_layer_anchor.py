#!/usr/bin/env python3
"""Compare Gemma layers 9 and 13 as anchors for one frozen Qwen command."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path

import numpy as np

from prh_replication.multifamily_shared_subspace import (
    activation_consensus,
    apply_linear,
    displacement_consensus,
    document_means,
    fit_pca_whitener,
    fixed_axis_correlation,
    ridge_maps,
    top1_retrieval,
    whiten_activations,
)


def load_parent():
    path = Path(__file__).resolve().parent / "run_multifamily_cca_shared_subspace.py"
    spec = importlib.util.spec_from_file_location("gemma_layer_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARENT = load_parent()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default) + "\n")


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def domain_of(doc_id: str) -> str:
    return "wikitext" if str(doc_id).startswith("wiki:") else "imdb"


def flatten(acts: np.ndarray) -> np.ndarray:
    return np.asarray(acts, dtype=np.float64).reshape(-1, acts.shape[-1])


def repeat_ids(doc_ids, n_pos: int) -> list[str]:
    return [doc for doc in doc_ids for _ in range(n_pos)]


def retrieval_pair(query, gallery, seed: int) -> dict:
    true = top1_retrieval(query, gallery)
    perm = np.random.default_rng(int(seed)).permutation(len(gallery))
    shuffled = top1_retrieval(query, np.asarray(gallery)[perm])
    return {"top1": true, "shuffled_top1": shuffled, "n_docs": int(len(query))}


def fit_ridge_at_alpha(x, y, alpha: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0, ddof=1), 1e-8)
    y_mean = y.mean(axis=0)
    xs = (x - x_mean) / x_scale
    slope = np.linalg.solve(xs.T @ xs + float(alpha) * np.eye(xs.shape[1]), xs.T @ (y - y_mean))
    decode = slope.T / x_scale.reshape(1, -1)
    intercept = y_mean - (x_mean / x_scale) @ slope
    return {"decode": decode, "intercept": intercept, "alpha": float(alpha)}


def chart_predict(pack, ridge, features: np.ndarray) -> np.ndarray:
    white = whiten_activations(features, pack, displacement=False)
    return apply_linear(ridge["decode"], ridge["intercept"], white, displacement=False)


def build_scenes(cfg) -> list[dict]:
    rng = np.random.default_rng(int(cfg["scene_seed"]))
    names = ["Ava", "Noah", "Mia", "Leo", "Eva", "Owen", "Ruby", "Seth", "Nina", "Paul", "Cora", "Hugh", "Ida", "Joel", "Ruth", "Theo"]
    streets = ["King Street", "Mill Road", "Harbor Lane", "Oak Avenue", "River Street", "Station Road", "Garden Way", "Bridge Street"]
    years = ["1994", "1998", "2001", "2004", "2007", "2010", "2012", "2016"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    templates = [
        ("shop", "In {year}, {name} kept a shop at {num} {street}.", " The apples were fresh and the clerk saved the receipt.", " The apples were stale and the clerk lost the receipt."),
        ("museum", "On {day}, {name} visited the museum with a paper map.", " The west wing was open and the guide answered the question.", " The west wing was shut and the guide ignored the question."),
        ("train", "The noon train from {street} reached the station with {name} aboard.", " The seat was clean and the guard found the ticket.", " The seat was dirty and the guard lost the ticket."),
        ("hotel", "{name} booked room {num} at the hotel for {day}.", " The bed was made and the desk kept the booking.", " The bed was bare and the desk lost the booking."),
        ("park", "{name} walked through the park after leaving {street}.", " The path was clear and the bench was dry.", " The path was blocked and the bench was wet."),
        ("class", "{name} attended the history class in room {num}.", " The notes were complete and the clock was right.", " The notes were missing and the clock was wrong."),
        ("concert", "{name} held a ticket for seat {num} at the hall.", " The sound was clear and the usher found the seat.", " The sound was harsh and the usher missed the seat."),
        ("library", "{name} borrowed one book from the library desk.", " The pages were intact and the stamp showed the date.", " The pages were torn and the stamp hid the date."),
    ]
    per = int(cfg["n_scenes"]) // len(templates)
    scenes = []
    for label, context_t, pos, neg in templates:
        if len(pos.split()) != len(neg.split()):
            raise RuntimeError(f"unbalanced template {label}")
        for _ in range(per):
            context = context_t.format(
                year=str(rng.choice(years)),
                name=str(rng.choice(names)),
                num=str(int(rng.integers(2, 80))),
                street=str(rng.choice(streets)),
                day=str(rng.choice(days)),
            )
            scenes.append({
                "scene_id": f"scene:{len(scenes):03d}",
                "template": label,
                "context": context,
                "positive": pos,
                "negative": neg,
                "suffix": cfg["neutral_suffix"],
            })
    if len(scenes) != int(cfg["n_scenes"]):
        raise RuntimeError(f"scene count {len(scenes)}")
    counts = {label: sum(row["template"] == label for row in scenes) for label, *_ in templates}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"unbalanced templates {counts}")
    return scenes


def encoder_from_saved(name, whitened_discovery, saved) -> dict:
    basis = np.asarray(saved[f"basis_{name}"], dtype=np.float64)
    rotation = np.asarray(saved[f"rotation_{name}"], dtype=np.float64)
    scale = np.asarray(saved[f"scale_{name}"], dtype=np.float64)
    flat = flatten(whitened_discovery)
    view_mean = flat.mean(axis=0)
    scores = (flat - view_mean) @ basis
    score_mean = scores.mean(axis=0)
    score_scale = np.maximum(scores.std(axis=0, ddof=1), 1e-8)
    if float(np.max(np.abs(score_scale - scale))) > 1e-6:
        raise RuntimeError(f"{name} score scale does not match the saved consensus")
    return {"basis": basis, "rotation": rotation, "score_mean": score_mean, "score_scale": score_scale, "view_mean": view_mean}


def encode_split(enc, whitened_split) -> np.ndarray:
    return activation_consensus(
        flatten(whitened_split), enc["basis"], enc["score_mean"], enc["score_scale"], enc["rotation"], enc["view_mean"],
    )


def choose_chart_alpha(acts, target, doc_ids, folds, cfg) -> dict:
    grid = [float(a) for a in cfg["alpha_grid"]]
    totals = {alpha: [] for alpha in grid}
    for fold in range(int(cfg["n_folds"])):
        train = folds != fold
        val = folds == fold
        pack = fit_pca_whitener(flatten(acts[train]), int(cfg["pca_dim"]), float(cfg["eta"]))
        x_train = whiten_activations(flatten(acts[train]), pack)
        x_val = whiten_activations(flatten(acts[val]), pack)
        y_train = flatten(target[train])
        y_val = flatten(target[val])
        docs_train = [doc for doc, keep in zip(repeat_ids(doc_ids, 4), np.repeat(train, 4)) if keep]
        docs_val = [doc for doc, keep in zip(repeat_ids(doc_ids, 4), np.repeat(val, 4)) if keep]
        fitted = ridge_maps(x_train, y_train, x_val, y_val, docs_train, docs_val, grid)
        for row in fitted["alphas"]:
            totals[float(row["alpha"])].append(float(row["val_mse"]))
    means = {alpha: float(np.mean(values)) for alpha, values in totals.items()}
    chosen = min(means, key=means.get)
    return {"chosen_alpha": chosen, "mean_fold_mse": {str(alpha): value for alpha, value in means.items()}, "objective": cfg["chart_objective"], "grid": grid}


def final_chart(acts, target, alpha: float, cfg) -> dict:
    pack = fit_pca_whitener(flatten(acts), int(cfg["pca_dim"]), float(cfg["eta"]))
    ridge = fit_ridge_at_alpha(whiten_activations(flatten(acts), pack), flatten(target), alpha)
    ridge["alpha"] = float(alpha)
    return {"pack": pack, "ridge": ridge}


def derange(ids, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(ids))
    for domain in ("imdb", "wikitext"):
        idx = np.asarray([i for i, doc in enumerate(ids) if domain_of(doc) == domain])
        perm = np.roll(np.arange(len(idx)), 1)
        for _ in range(32):
            cand = rng.permutation(len(idx))
            if len(idx) < 2 or np.all(cand != np.arange(len(idx))):
                perm = cand
                break
        order[idx] = idx[perm]
    if np.any(order == np.arange(len(ids))):
        raise RuntimeError("shuffle left a document in place")
    return order


def prepare_direction(raw, dose, near_zero):
    norm = float(np.linalg.norm(raw))
    if norm < near_zero:
        return np.zeros_like(raw), True, norm
    return raw / norm * dose, False, norm


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_root)
    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {"log": []}

    def note(msg):
        state["log"].append({"t": round(time.time() - t0, 1), "msg": msg})
        print(msg, flush=True)

    try:
        resumed = resume_generation(cfg, work, art, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, note)
        if resumed is not None:
            state["generation"] = resumed
            state["diagnostics"] = json.loads((art / "diagnostics.json").read_text()) if (art / "diagnostics.json").exists() else {}
            state["teacher"] = json.loads((art / "teacher.json").read_text()) if (art / "teacher.json").exists() else {}
            state["representation"] = json.loads((art / "representation.json").read_text()) if (art / "representation.json").exists() else {}
            raise SystemExit
        if (art / "representation.json").exists() and (art / "dir_matched_l13.npy").exists() and not (art / "teacher.json").exists():
            note("resume from saved charts and directions")
            scenes = json.loads((art / "scenes.json").read_text())["scenes"]
            directions = {}
            for path in sorted(art.glob("dir_*.npy")):
                name = path.name[len("dir_"):-4]
                raw = np.load(art / f"raw_{name}.npy")
                directions[name] = {"vector": np.load(path), "near_zero": float(np.linalg.norm(raw)) < float(cfg["near_zero_norm"]), "raw_norm": float(np.linalg.norm(raw)), "sha256": sha256_array(np.load(path))}
            state["representation"] = json.loads((art / "representation.json").read_text())
            state["diagnostics"] = json.loads((art / "diagnostics.json").read_text())
            manifest = {split: json.loads((work / cfg["parent_geometry"] / "artifacts" / f"split_{split}.json").read_text()) for split in ("discovery", "attachment_fit", "attachment_val", "geometry_test")}
            state["teacher"] = run_teacher(cfg, work, art, scenes, directions, torch, AutoModelForCausalLM, AutoTokenizer, note)
            state["generation"] = run_generation(cfg, work, art, directions, manifest, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, note)
            raise SystemExit
        save_json(art / "chart_objective.json", {"grid": cfg["alpha_grid"], "objective": cfg["chart_objective"], "folds": cfg["n_folds"], "fold_seed": cfg["fold_seed"], "shuffle_seeds": cfg["shuffle_seeds"]})
        parent = work / cfg["parent_geometry"] / "artifacts"
        swap = work / cfg["parent_swap"] / "artifacts"
        manifest = {split: json.loads((parent / f"split_{split}.json").read_text()) for split in ("discovery", "attachment_fit", "attachment_val", "geometry_test")}
        scenes = build_scenes(cfg)
        save_json(art / "scenes.json", {"n": len(scenes), "scenes": scenes, "frozen_before_scores": True})
        encoders, whitened, z_edit = load_shared_command(work, cfg, parent, manifest)
        state["z_edit_sha256"] = sha256_array(z_edit)
        saved_z = np.load(swap / "z_edit.npy")
        z_err = float(np.max(np.abs(z_edit - saved_z)))
        note(f"z_edit max abs vs archived command {z_err:.3e} hash {state['z_edit_sha256'][:12]}")
        if z_err > 1e-8:
            raise RuntimeError("frozen Qwen command does not match the anchor-swap command")
        np.save(art / "z_edit.npy", z_edit)
        doc_ids = {split: [row["doc_id"] for row in rows] for split, rows in manifest.items()}
        z_ref = {}
        for split in manifest:
            parts = [encode_split(encoders[name], whitened[name][split]) for name in ("qwen", "olmo", "mistral")]
            z_ref[split] = np.mean(parts, axis=0).reshape(len(doc_ids[split]), 4, -1)
        gemma_l9 = load_parent_gemma(parent, manifest)
        fit_deadline = t0 + float(cfg["fit_wall_sec"])
        gemma_l13, hook_info = extract_layer_13(work, cfg, manifest, gemma_l9, art, torch, AutoModelForCausalLM, AutoTokenizer, note)
        state["hook"] = hook_info
        if time.time() > fit_deadline:
            note("extraction exceeded the 45-minute fit budget")
        folds = assign_folds(doc_ids["discovery"], int(cfg["n_folds"]), int(cfg["fold_seed"]))
        charts = {}
        for layer, acts in ((9, gemma_l9), (13, gemma_l13)):
            selection = choose_chart_alpha(acts["discovery"], z_ref["discovery"], doc_ids["discovery"], folds, cfg)
            chart = final_chart(acts["discovery"], z_ref["discovery"], selection["chosen_alpha"], cfg)
            charts[layer] = {"selection": selection, "chart": chart, "acts": acts}
            fold_mse = selection["mean_fold_mse"][str(selection["chosen_alpha"])]
            note(f"layer {layer} chart alpha {selection['chosen_alpha']} fold mse {fold_mse:.5f}")
            save_chart(art / "charts" / f"layer_{layer}.npz", chart)
        moon = load_parent_gemma_like(parent, "moonlight", manifest)
        decoders = {}
        raws = {}
        for layer in cfg["layers"]:
            decoders[layer] = fit_attachment_decoder(charts[layer], gemma_l9 if layer == 9 else gemma_l13, moon, doc_ids, cfg)
            raw = apply_linear(decoders[layer]["backward"]["decode"], decoders[layer]["backward"]["intercept"], z_edit, displacement=True)
            raws[f"matched_l{layer}"] = raw
            note(f"matched L{layer} raw {np.linalg.norm(raw):.5f} alpha {decoders[layer]['backward']['alpha']}")
        shuffles = {}
        for seed in cfg["shuffle_seeds"]:
            order_fit = derange(doc_ids["attachment_fit"], int(seed))
            order_val = derange(doc_ids["attachment_val"], int(seed) + 1000)
            shuffled = fit_attachment_decoder(charts[13], gemma_l13, moon, doc_ids, cfg, fit_order=order_fit, val_order=order_val)
            raw = apply_linear(shuffled["backward"]["decode"], shuffled["backward"]["intercept"], z_edit, displacement=True)
            shuffles[int(seed)] = {"decoder": shuffled, "raw": raw, "norm": float(np.linalg.norm(raw))}
            raws[f"shuffle_{seed}"] = raw
            note(f"shuffle {seed} raw {shuffles[int(seed)]['norm']:.5f}")
        archived = {
            "gemma_l9": np.load(swap / "routes/gemma/consensus_raw.npy"),
            "mistral": np.load(swap / "routes/mistral/consensus_raw.npy"),
        }
        raws["archived_gemma_l9"] = archived["gemma_l9"]
        raws["archived_mistral"] = archived["mistral"]
        qwen_z = encode_split(encoders["qwen"], whitened["qwen"]["geometry_test"])
        state["diagnostics"] = diagnostics(charts, decoders, shuffles, z_ref, qwen_z, moon, doc_ids, raws, cfg)
        save_json(art / "diagnostics.json", state["diagnostics"])
        directions = scale_directions(raws, cfg, art)
        state["directions"] = {name: {"norm": row["raw_norm"], "near_zero": row["near_zero"], "sha256": row["sha256"]} for name, row in directions.items()}
        scene_acts = extract_scene_acts(work, cfg, scenes, torch, AutoModelForCausalLM, AutoTokenizer, note)
        state["representation"] = representation_report(charts, encoders, whitened, scene_acts, z_edit, z_ref["geometry_test"], parent, cfg)
        save_json(art / "representation.json", state["representation"])
        note("representation " + json.dumps({k: v.get("mean_raw") for k, v in state["representation"]["layers"].items()}))
        state["teacher"] = run_teacher(cfg, work, art, scenes, directions, torch, AutoModelForCausalLM, AutoTokenizer, note)
        state["generation"] = run_generation(cfg, work, art, directions, manifest, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, note)
    except SystemExit:
        pass
    except Exception:
        state["error"] = traceback.format_exc()
        note(state["error"])
    state["elapsed_sec"] = time.time() - t0
    save_json(art / "run_state.json", state)
    text = render_md(state, cfg)
    (dest / "RESULTS.md").write_text(text)
    print("wrote RESULTS", flush=True)
    return 0


def load_shared_command(work, cfg, parent, manifest):
    saved = np.load(parent / "space_primary" / "consensus.npz")
    v_q = np.load(work / cfg["sentiment_parent"] / "artifacts/vec_qwen_averaged.npy").astype(np.float64)
    encoders = {}
    whitened = {}
    for name in ("qwen", "olmo", "mistral"):
        cache = np.load(parent / "cache" / f"{name}.npz", allow_pickle=True)
        pack = {key: np.asarray(np.load(parent / "pca" / f"{name}.npz")[key]) for key in ("mean", "components", "whitener")}
        whitened[name] = {}
        for split, rows in manifest.items():
            ids = [str(x) for x in cache[f"ids_{split}"].tolist()]
            if ids != [row["doc_id"] for row in rows]:
                raise RuntimeError(f"{name} {split} order does not match the parent split")
            acts = np.asarray(cache[split])
            whitened[name][split] = whiten_activations(flatten(acts), pack).reshape(acts.shape[0], acts.shape[1], -1)
        encoders[name] = encoder_from_saved(name, whitened[name]["discovery"], saved)
    z_edit = displacement_consensus(
        whiten_activations(v_q, pack_from(parent, "qwen"), displacement=True),
        encoders["qwen"]["basis"], encoders["qwen"]["score_scale"], encoders["qwen"]["rotation"],
    )
    return encoders, whitened, z_edit


def pack_from(parent, name):
    loaded = np.load(parent / "pca" / f"{name}.npz")
    return {key: np.asarray(loaded[key]) for key in ("mean", "components", "whitener")}


def load_parent_gemma(parent, manifest):
    return load_parent_gemma_like(parent, "gemma", manifest)


def load_parent_gemma_like(parent, name, manifest):
    cache = np.load(parent / "cache" / f"{name}.npz", allow_pickle=True)
    out = {}
    for split, rows in manifest.items():
        ids = [str(x) for x in cache[f"ids_{split}"].tolist()]
        if ids != [row["doc_id"] for row in rows]:
            raise RuntimeError(f"{name} {split} order does not match")
        out[split] = np.asarray(cache[split])
    return out


def extract_layer_13(work, cfg, manifest, gemma_l9, art, torch, auto_model, auto_tok, note):
    dest = art / "cache" / "gemma_l13.npz"
    spec = dict(cfg["gemma"])
    spec["layer"] = 9
    model, tok, layer9 = PARENT.load_model(work, spec, torch, auto_model, auto_tok)
    n_layers = len(model.model.layers)
    if not (0 <= 13 < n_layers):
        raise RuntimeError(f"Gemma layer 13 is outside 0..{n_layers - 1}")
    probe_rows = manifest["discovery"][:8]
    probe = PARENT.extract_acts(model, tok, layer9, probe_rows, cfg["boundaries"], torch)
    err = float(np.max(np.abs(probe.astype(np.float64) - gemma_l9["discovery"][:8].astype(np.float64))))
    denom = float(np.linalg.norm(gemma_l9["discovery"][:8].astype(np.float64)))
    cosine = float(np.vdot(probe.astype(np.float64), gemma_l9["discovery"][:8].astype(np.float64)) / max(np.linalg.norm(probe.astype(np.float64)) * denom, 1e-12))
    note(f"L9 parity max abs {err:.4e} cosine {cosine:.6f}")
    if cosine < float(cfg["parity_cosine"]):
        raise RuntimeError(f"L9 extraction does not match the parent cache: {cosine}")
    hook = {
        "indexing": "zero-indexed inputs to Gemma blocks",
        "module": "model.model.layers[index]",
        "hook": "forward_pre_hook on the residual entering the block",
        "n_layers": n_layers,
        "layer_9": "index 9, the 10th block",
        "layer_13": "index 13, the 14th block",
        "l9_parity_max_abs": err,
        "l9_parity_cosine": cosine,
    }
    save_json(art / "hook_convention.json", hook)
    if dest.exists():
        loaded = np.load(dest)
        out = {split: np.asarray(loaded[split]) for split in manifest}
        del model
        torch.cuda.empty_cache()
        note("reuse L13 cache")
        return out, hook
    layer13 = model.model.layers[13]
    arrays = {}
    id_arrays = {}
    for split, rows in manifest.items():
        note(f"extract L13 {split} {len(rows)}")
        arrays[split] = PARENT.extract_acts(model, tok, layer13, rows, cfg["boundaries"], torch).astype(np.float32)
        id_arrays[f"ids_{split}"] = np.asarray([row["doc_id"] for row in rows])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **arrays, **id_arrays)
    del model
    torch.cuda.empty_cache()
    return arrays, hook


def assign_folds(doc_ids, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    folds = np.full(len(doc_ids), -1, dtype=int)
    for domain in ("imdb", "wikitext"):
        idx = np.asarray([i for i, doc in enumerate(doc_ids) if domain_of(doc) == domain])
        rng.shuffle(idx)
        for fold, group in enumerate(np.array_split(idx, n_folds)):
            folds[group] = fold
    if np.any(folds < 0):
        raise RuntimeError("a discovery document missed its fold")
    return folds


def save_chart(path: Path, chart) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pack = chart["pack"]
    ridge = chart["ridge"]
    np.savez(path, mean=pack["mean"], components=pack["components"], whitener=pack["whitener"], decode=ridge["decode"], intercept=ridge["intercept"], alpha=np.asarray(ridge["alpha"]))


def fit_attachment_decoder(chart_row, gemma_acts, moon, doc_ids, cfg, fit_order=None, val_order=None) -> dict:
    chart = chart_row["chart"]
    scores = {}
    moons = {}
    docs = {}
    for split, order in (("attachment_fit", fit_order), ("attachment_val", val_order)):
        acts = gemma_acts[split]
        moon_acts = moon[split]
        if order is not None:
            moon_acts = moon_acts[order]
        scores[split] = chart_predict(chart["pack"], chart["ridge"], flatten(acts))
        moons[split] = flatten(moon_acts)
        docs[split] = repeat_ids(doc_ids[split], 4)
    backward = ridge_maps(scores["attachment_fit"], moons["attachment_fit"], scores["attachment_val"], moons["attachment_val"], docs["attachment_fit"], docs["attachment_val"], cfg["alpha_grid"])
    forward = ridge_maps(moons["attachment_fit"], scores["attachment_fit"], moons["attachment_val"], scores["attachment_val"], docs["attachment_fit"], docs["attachment_val"], cfg["alpha_grid"])
    return {"backward": backward, "forward": forward}


def diagnostics(charts, decoders, shuffles, z_ref, qwen_z, moon, doc_ids, raws, cfg) -> dict:
    out = {"charts": {}, "decoders": {}, "retrieval_vs_qwen": {}, "cosines": {}}
    docs = repeat_ids(doc_ids["geometry_test"], 4)
    q_doc, _ = document_means(qwen_z, docs)
    for layer in cfg["layers"]:
        chart = charts[layer]["chart"]
        pred = chart_predict(chart["pack"], chart["ridge"], flatten(charts[layer]["acts"]["geometry_test"]))
        target = flatten(z_ref["geometry_test"])
        mse = document_mse(pred, target, docs)
        base = document_mse(np.repeat(target.mean(0, keepdims=True), len(target), 0), target, docs)
        pred_doc, _ = document_means(pred, docs)
        target_doc, _ = document_means(target, docs)
        out["charts"][str(layer)] = {
            "alpha": chart["ridge"]["alpha"],
            "fold_mse": charts[layer]["selection"]["mean_fold_mse"],
            "heldout_mse": mse,
            "heldout_baseline_mse": base,
            "heldout_axis_correlation": fixed_axis_correlation(pred, target),
            "heldout_top1_vs_zref": retrieval_pair(pred_doc, target_doc, 0),
        }
        dec = decoders[layer]["backward"]
        out["decoders"][str(layer)] = {"alpha": dec["alpha"], "val_mse": dec["val_mse"], "baseline_mse": dec["baseline_mse"]}
        rev = decoders[layer]["forward"]
        moon_pred = apply_linear(rev["decode"], rev["intercept"], flatten(moon["geometry_test"]), displacement=False)
        moon_doc, _ = document_means(moon_pred, docs)
        out["retrieval_vs_qwen"][str(layer)] = {
            "axis_correlation": fixed_axis_correlation(moon_pred, qwen_z),
            **retrieval_pair(moon_doc, q_doc, 0),
        }
    out["shuffles"] = {
        str(seed): {
            "raw_norm": row["norm"],
            "val_mse": row["decoder"]["backward"]["val_mse"],
            "baseline_mse": row["decoder"]["backward"]["baseline_mse"],
            "alpha": row["decoder"]["backward"]["alpha"],
        }
        for seed, row in shuffles.items()
    }
    labels = list(raws)
    for left in labels:
        out["cosines"][left] = {}
        for right in labels:
            a = np.asarray(raws[left], dtype=np.float64).reshape(-1)
            b = np.asarray(raws[right], dtype=np.float64).reshape(-1)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            out["cosines"][left][right] = float(np.dot(a, b) / denom) if denom else float("nan")
    out["archived_vs_matched_l9"] = out["cosines"]["matched_l9"]["archived_gemma_l9"]
    return out


def document_mse(pred, target, doc_ids) -> float:
    err = np.mean((pred - target) ** 2, axis=1)
    buckets = {}
    for value, doc in zip(err, doc_ids):
        buckets.setdefault(doc, []).append(float(value))
    return float(np.mean([np.mean(values) for values in buckets.values()]))


def scale_directions(raws, cfg, art: Path) -> dict:
    out = {}
    dose = float(cfg["dose"])
    for name, raw in raws.items():
        vec, bad, norm = prepare_direction(np.asarray(raw, dtype=np.float64), dose, float(cfg["near_zero_norm"]))
        np.save(art / f"dir_{name}.npy", vec)
        np.save(art / f"raw_{name}.npy", np.asarray(raw, dtype=np.float64))
        out[name] = {"vector": vec, "near_zero": bad, "raw_norm": norm, "sha256": sha256_array(vec)}
    return out


def extract_scene_acts(work, cfg, scenes, torch, auto_model, auto_tok, note):
    rows = []
    for scene in scenes:
        for polarity in ("positive", "negative"):
            rows.append({"doc_id": f"{scene['scene_id']}:{polarity}", "text": scene["context"] + scene[polarity] + scene["suffix"]})
    out = {}
    spec = dict(cfg["gemma"])
    spec["layer"] = 9
    model, tok, _ = PARENT.load_model(work, spec, torch, auto_model, auto_tok)
    for layer in cfg["layers"]:
        note(f"scene extract gemma L{layer}")
        out[f"gemma_l{layer}"] = PARENT.extract_acts(model, tok, model.model.layers[int(layer)], rows, [10**9], torch)
    del model
    torch.cuda.empty_cache()
    spec = dict(cfg["mistral"])
    model, tok, layer = PARENT.load_model(work, spec, torch, auto_model, auto_tok)
    note("scene extract mistral")
    out["mistral"] = PARENT.extract_acts(model, tok, layer, rows, [10**9], torch)
    del model
    torch.cuda.empty_cache()
    for name, acts in out.items():
        np.save(art_path(cfg, work) / "cache" / f"scenes_{name}.npy", acts.astype(np.float32))
    return out


def art_path(cfg, work):
    path = work / cfg["output"] / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    (path / "cache").mkdir(parents=True, exist_ok=True)
    return path


def representation_report(charts, encoders, whitened, scene_acts, z_edit, z_ref_geometry, parent, cfg) -> dict:
    unit = np.asarray(z_edit, dtype=np.float64).reshape(-1)
    unit = unit / np.linalg.norm(unit)
    common_sd = float((flatten(z_ref_geometry) @ unit).std(ddof=1))
    report = {"common_unlabeled_sd": common_sd, "layers": {}}
    pairs = scene_acts["gemma_l9"].shape[0] // 2

    def summarise(name, raw):
        standardised = raw / common_sd if common_sd > 0 else np.full_like(raw, np.nan)
        report["layers"][name] = {
            "mean_raw": float(np.mean(raw)),
            "mean_over_common_sd": float(np.mean(standardised)),
            "raw": PARENT.boot_mean(raw, int(cfg["n_boot"])),
            "standardised": PARENT.boot_mean(standardised, int(cfg["n_boot"])),
        }

    for layer in cfg["layers"]:
        acts = scene_acts[f"gemma_l{layer}"].reshape(pairs, 2, -1)
        chart = charts[layer]["chart"]
        pos = chart_predict(chart["pack"], chart["ridge"], acts[:, 0, :])
        neg = chart_predict(chart["pack"], chart["ridge"], acts[:, 1, :])
        summarise(f"gemma_l{layer}", (pos - neg) @ unit)
    pack = pack_from(parent, "mistral")
    acts = scene_acts["mistral"].reshape(pairs, 2, -1)
    white_pos = whiten_activations(acts[:, 0, :], pack)
    white_neg = whiten_activations(acts[:, 1, :], pack)
    enc = encoders["mistral"]
    z_pos = activation_consensus(white_pos, enc["basis"], enc["score_mean"], enc["score_scale"], enc["rotation"], enc["view_mean"])
    z_neg = activation_consensus(white_neg, enc["basis"], enc["score_mean"], enc["score_scale"], enc["rotation"], enc["view_mean"])
    summarise("mistral", (z_pos - z_neg) @ unit)
    return report


def score_full(model, layer, ids, cont_start, cont_len, direction, torch):
    positions = list(range(cont_start - 1, cont_start + cont_len - 1))
    handle = None
    if direction is not None:
        delta = torch.tensor(np.asarray(direction), device="cuda", dtype=torch.float32)
        index = torch.tensor(positions, device="cuda", dtype=torch.long)

        def hook(_m, inputs, _delta=delta, _index=index):
            resid = inputs[0]
            out = resid.clone()
            out[:, _index] = out[:, _index] + _delta.to(out.dtype)
            return (out,)

        handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    finally:
        if handle is not None:
            handle.remove()
    window = torch.log_softmax(logits[0, cont_start - 1: cont_start - 1 + cont_len].float(), dim=-1)
    choice = ids[0, cont_start: cont_start + cont_len]
    gathered = window[torch.arange(cont_len, device=ids.device), choice]
    return float(gathered.mean().item())


def score_cached(model, layer, prompt_ids, cont_ids, direction, torch):
    handle = None
    if direction is not None:
        delta = torch.tensor(np.asarray(direction), device="cuda", dtype=torch.float32)

        def hook(_m, inputs, _delta=delta):
            resid = inputs[0]
            out = resid.clone()
            out[:, -1] = out[:, -1] + _delta.to(out.dtype)
            return (out,)

        handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            fwd = model(prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True)
            past = fwd.past_key_values
            values = []
            step_logits = fwd.logits[0, -1]
            for pos in range(cont_ids.shape[1]):
                logp = torch.log_softmax(step_logits.float(), dim=-1)
                values.append(float(logp[cont_ids[0, pos]].item()))
                if pos + 1 == cont_ids.shape[1]:
                    break
                nxt = cont_ids[:, pos: pos + 1]
                attn = torch.ones((1, prompt_ids.shape[1] + pos + 1), device=prompt_ids.device, dtype=prompt_ids.dtype)
                fwd = model(nxt, attention_mask=attn, past_key_values=past, use_cache=True)
                past = fwd.past_key_values
                step_logits = fwd.logits[0, -1]
    finally:
        if handle is not None:
            handle.remove()
    return float(np.mean(values))


def run_teacher(cfg, work, art, scenes, directions, torch, auto_model, auto_tok, note):
    deadline = time.time() + float(cfg["teacher_wall_sec"])
    native = np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy")
    arms = {
        "unedited": None,
        "matched_l9": directions["matched_l9"]["vector"],
        "matched_l13": directions["matched_l13"]["vector"],
        "archived_gemma_l9": directions["archived_gemma_l9"]["vector"],
        "archived_mistral": directions["archived_mistral"]["vector"],
        "native": native,
    }
    for seed in cfg["shuffle_seeds"]:
        arms[f"shuffle_{seed}"] = directions[f"shuffle_{seed}"]["vector"]
    flags = {name: bool(directions[name]["near_zero"]) for name in directions if name in arms}
    model, tok, layer = PARENT.load_model(work, cfg["moonlight"], torch, auto_model, auto_tok)
    model.config.use_cache = True
    smoke = {
        "context": "The blue notebook stayed on the desk.",
        "positive": " The page was clean and the pencil was sharp.",
        "negative": " The page was torn and the pencil was blunt.",
    }
    prompt_ids = tok(smoke["context"], return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
    cont_ids = tok(smoke["positive"], return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
    joined = torch.cat([prompt_ids, cont_ids], dim=1)
    with torch.no_grad():
        bare = model(joined, attention_mask=torch.ones_like(joined), use_cache=False).logits
    zero = score_full(model, layer, joined, int(prompt_ids.shape[1]), int(cont_ids.shape[1]), None, torch)
    full = score_full(model, layer, joined, int(prompt_ids.shape[1]), int(cont_ids.shape[1]), arms["matched_l13"], torch)
    cached = score_cached(model, layer, prompt_ids, cont_ids, arms["matched_l13"], torch)
    cached_zero = score_cached(model, layer, prompt_ids, cont_ids, None, torch)
    window = torch.log_softmax(bare[0, prompt_ids.shape[1] - 1: prompt_ids.shape[1] - 1 + cont_ids.shape[1]].float(), dim=-1)
    bare_mean = float(window[torch.arange(cont_ids.shape[1], device=joined.device), cont_ids[0]].mean().item())
    smoke_report = {
        "zero_edit_abs": abs(zero - bare_mean),
        "cache_unedited_abs": abs(zero - cached_zero),
        "cache_edited_abs": abs(full - cached),
        "full_edit_minus_unedited": full - zero,
        "cached_edit_minus_unedited": cached - cached_zero,
        "sign_vector_sum": float(np.max(np.abs(arms["matched_l13"] + (-arms["matched_l13"])))),
    }
    save_json(art / "teacher_smoke.json", smoke_report)
    note(f"teacher smoke zero {smoke_report['zero_edit_abs']:.3e} cache edited {smoke_report['cache_edited_abs']:.3e}")
    same_sign = smoke_report["full_edit_minus_unedited"] * smoke_report["cached_edit_minus_unedited"] > 0
    if smoke_report["zero_edit_abs"] > 1e-3 or not same_sign or smoke_report["cache_edited_abs"] > 0.05:
        raise RuntimeError(f"teacher-forced smoke failed {smoke_report}")
    rows = []
    ckpt = art / "teacher_scores.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["scene_id"]] = row
    for scene in scenes:
        if scene["scene_id"] in done:
            rows.append(done[scene["scene_id"]])
            continue
        if time.time() >= deadline:
            break
        prompt = scene["context"]
        rec = {"scene_id": scene["scene_id"], "template": scene["template"]}
        for arm, direction in arms.items():
            if flags.get(arm):
                rec[arm] = {"non_evaluable": True}
                continue
            for sign, scale in (("plus", 1.0), ("minus", -1.0)):
                vector = None if direction is None else scale * direction
                pos = score_text(model, layer, tok, prompt, scene["positive"], vector, torch)
                neg = score_text(model, layer, tok, prompt, scene["negative"], vector, torch)
                rec.setdefault(arm, {})[sign] = {"positive_logprob": pos, "negative_logprob": neg, "preference": pos - neg}
        base = rec["unedited"]["plus"]["preference"]
        for arm, payload in rec.items():
            if arm in ("scene_id", "template") or payload.get("non_evaluable"):
                continue
            for sign in ("plus", "minus"):
                payload[sign]["delta_preference"] = payload[sign]["preference"] - base
                payload[sign]["delta_positive"] = payload[sign]["positive_logprob"] - rec["unedited"]["plus"]["positive_logprob"]
                payload[sign]["delta_negative"] = payload[sign]["negative_logprob"] - rec["unedited"]["plus"]["negative_logprob"]
            payload["odd"] = 0.5 * (payload["plus"]["preference"] - payload["minus"]["preference"])
        rows.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if len(rows) % 16 == 0:
            note(f"teacher {len(rows)}/{len(scenes)}")
    contrasts = teacher_contrasts(rows, cfg)
    result = {"ran": True, "n": len(rows), "frozen_n": len(scenes), "partial": len(rows) < len(scenes), "contrasts": contrasts}
    save_json(art / "teacher.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def score_text(model, layer, tok, prompt, continuation, direction, torch):
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
    cont_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
    ids = torch.cat([prompt_ids, cont_ids], dim=1)
    return score_full(model, layer, ids, int(prompt_ids.shape[1]), int(cont_ids.shape[1]), direction, torch)


def teacher_contrasts(rows, cfg) -> dict:
    usable = [row for row in rows if "matched_l13" in row and "matched_l9" in row and not row["matched_l13"].get("non_evaluable")]
    if not usable:
        return {}
    def pref(row, arm, sign):
        return float(row[arm][sign]["preference"])
    out = {
        "l13_minus_l9_plus": PARENT.boot_mean([pref(r, "matched_l13", "plus") - pref(r, "matched_l9", "plus") for r in usable], int(cfg["n_boot"])),
        "l13_minus_unedited_plus": PARENT.boot_mean([r["matched_l13"]["plus"]["delta_preference"] for r in usable], int(cfg["n_boot"])),
        "l9_minus_unedited_plus": PARENT.boot_mean([r["matched_l9"]["plus"]["delta_preference"] for r in usable], int(cfg["n_boot"])),
    }
    for arm in ("matched_l9", "matched_l13", "archived_gemma_l9", "archived_mistral", "native"):
        if arm not in usable[0] or usable[0][arm].get("non_evaluable"):
            continue
        out[f"{arm}_minus_dose"] = PARENT.boot_mean([r[arm]["minus"]["delta_preference"] for r in usable], int(cfg["n_boot"]))
        out[f"{arm}_odd"] = PARENT.boot_mean([r[arm]["odd"] for r in usable], int(cfg["n_boot"]))
        out[f"{arm}_delta_positive"] = PARENT.boot_mean([r[arm]["plus"]["delta_positive"] for r in usable], int(cfg["n_boot"]))
        out[f"{arm}_delta_negative"] = PARENT.boot_mean([r[arm]["plus"]["delta_negative"] for r in usable], int(cfg["n_boot"]))
    first = f"shuffle_{cfg['shuffle_seeds'][0]}"
    for seed in cfg["shuffle_seeds"]:
        arm = f"shuffle_{seed}"
        if arm in usable[0] and not usable[0][arm].get("non_evaluable"):
            out[f"{arm}_plus"] = PARENT.boot_mean([r[arm]["plus"]["delta_preference"] for r in usable], int(cfg["n_boot"]))
    if first in usable[0]:
        out["l13_minus_prespecified_shuffle_plus"] = PARENT.boot_mean(
            [r["matched_l13"]["plus"]["preference"] - r[first]["plus"]["preference"] for r in usable if not r[first].get("non_evaluable")],
            int(cfg["n_boot"]),
        )
    return out


def run_generation(cfg, work, art, directions, manifest, torch, auto_model, auto_tok, pipeline, note):
    native = np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy")
    first = f"shuffle_{cfg['shuffle_seeds'][0]}"
    arms = {
        "unedited": None,
        "native": native,
        "archived_mistral": directions["archived_mistral"]["vector"],
        "matched_l9": directions["matched_l9"]["vector"],
        "matched_l13": directions["matched_l13"]["vector"],
        first: directions[first]["vector"],
    }
    flags = {name: bool(directions.get(name, {}).get("near_zero")) for name in arms if name in directions}
    return steer(cfg, work, art, arms, flags, manifest, torch, auto_model, auto_tok, pipeline, note, panel=None, done=None)


def resume_generation(cfg, work, art, torch, auto_model, auto_tok, pipeline, note):
    panel_path = art / "steering_panel.json"
    ckpt = art / "steering_confirmation.jsonl"
    if not panel_path.exists() or not ckpt.exists() or not (art / "dir_matched_l13.npy").exists():
        return None
    panel = json.loads(panel_path.read_text())
    done = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
    ids = [row["review_id"] for row in done]
    if ids != panel["review_ids"][: len(ids)] or len(ids) >= int(panel["n"]):
        return None
    native = np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy")
    first = f"shuffle_{cfg['shuffle_seeds'][0]}"
    names = ["archived_mistral", "matched_l9", "matched_l13", first]
    arms = {"unedited": None, "native": native}
    flags = {}
    for name in names:
        arms[name] = np.load(art / f"dir_{name}.npy")
        flags[name] = float(np.linalg.norm(arms[name])) < 1e-8
    note(f"resume generation {len(ids)}/{panel['n']}")
    return steer(cfg, work, art, arms, flags, None, torch, auto_model, auto_tok, pipeline, note, panel=panel, done=done)


def steer(cfg, work, art, arms, flags, manifest, torch, auto_model, auto_tok, pipeline, note, panel, done):
    from datasets import load_dataset
    model, tok, layer = PARENT.load_model(work, cfg["moonlight"], torch, auto_model, auto_tok)
    model.config.use_cache = True
    sample = cfg["sample"]
    max_new = int(sample["max_new_tokens"])
    siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
    scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)

    def pos_of(text):
        clipped = scorer.tokenizer((text or " ")[:2000], truncation=True, max_length=512, add_special_tokens=True)["input_ids"]
        raw_scores = scorer(scorer.tokenizer.decode(clipped, skip_special_tokens=True) or " ")
        if raw_scores and isinstance(raw_scores[0], list):
            raw_scores = raw_scores[0]
        table = {item["label"].upper(): float(item["score"]) for item in raw_scores}
        positive = table.get("POSITIVE", 0.0)
        return positive, positive >= table.get("NEGATIVE", 0.0)

    def render(text):
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
        return tok.decode(ids, skip_special_tokens=True).rstrip() + cfg["eval_bridge"]

    def generate(text, direction, seed):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
        handle = None
        if direction is not None:
            delta = torch.tensor(np.asarray(direction), device="cuda", dtype=torch.float32)

            def hook(_m, inputs, _delta=delta):
                resid = inputs[0]
                out = resid.clone()
                out[:, -1] = out[:, -1] + _delta.to(out.dtype)
                return (out,)

            handle = layer.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                fwd = model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
                past = fwd.past_key_values
                new_ids = []
                for _ in range(max_new):
                    nxt = PARENT.sample_next(fwd.logits[0, -1], ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
                    new_ids.append(int(nxt.item()))
                    if tok.eos_token_id is not None and int(nxt.item()) == int(tok.eos_token_id):
                        break
                    ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                    attn = torch.ones((1, ids.shape[1]), device=ids.device, dtype=ids.dtype)
                    fwd = model(nxt.view(1, 1), attention_mask=attn, past_key_values=past, use_cache=True)
                    past = fwd.past_key_values
        finally:
            if handle is not None:
                handle.remove()
        return new_ids, tok.decode(new_ids, skip_special_tokens=True)

    t_gen = time.time()
    if panel is None:
        excluded = generation_excluded(work, cfg, manifest)
        eligible = []
        blocked = set()
        if manifest is not None:
            for rows in manifest.values():
                for row in rows:
                    blocked.add(hashlib.sha256(" ".join(row["text"].split()).encode()).hexdigest())
        for i, row in enumerate(load_dataset("stanfordnlp/imdb")["test"]):
            if int(row["label"]) != 0:
                continue
            review_id = f"test:{i}"
            if review_id in excluded:
                continue
            if hashlib.sha256(" ".join(row["text"].split()).encode()).hexdigest() in blocked:
                continue
            eligible.append({"review_id": review_id, "text": row["text"]})
        np.random.default_rng(int(cfg["generation_seed"])).shuffle(eligible)
        smoke_text = render(eligible[0]["text"])
        enc = tok(smoke_text, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
        with torch.no_grad():
            base = model(enc, attention_mask=torch.ones_like(enc)).logits
        handle = layer.register_forward_pre_hook(lambda _m, inputs: (inputs[0].clone(),))
        try:
            with torch.no_grad():
                zero = model(enc, attention_mask=torch.ones_like(enc)).logits
        finally:
            handle.remove()
        same = generate(smoke_text, None, 12345)[0] == generate(smoke_text, None, 12345)[0]
        zero_err = float(torch.max(torch.abs(base.float() - zero.float())).cpu())
        if zero_err > 1e-3 or not same:
            raise RuntimeError(f"generation smoke failed zero {zero_err} seed {same}")
        timers = eligible[: int(cfg["generation_smoke_blocks"])]
        t_batch = time.time()
        for row in timers:
            text = render(row["text"])
            seed = PARENT.review_seed(row["review_id"])
            for direction in arms.values():
                generate(text, direction, seed)
        sec = (time.time() - t_batch) / max(1, len(timers))
        remain = float(cfg["generation_wall_sec"]) - float(cfg["generation_reserve_sec"]) - (time.time() - t_gen)
        n_panel = min(int(cfg["generation_target"]), int(max(0.0, remain) * 0.9 / sec)) if sec > 0 else 0
        chosen = eligible[len(timers): len(timers) + max(n_panel, 0)]
        panel = {"n": len(chosen), "sec_per_block": sec, "review_ids": [r["review_id"] for r in chosen], "timing_review_ids": [r["review_id"] for r in timers], "frozen_before_test_scores": True}
        save_json(art / "steering_panel.json", panel)
        note(f"generation panel {panel['n']} sec/block {sec:.2f}")
        done = []
        texts = {row["review_id"]: row["text"] for row in chosen}
    else:
        texts = {}
        for i, row in enumerate(load_dataset("stanfordnlp/imdb")["test"]):
            texts[f"test:{i}"] = row["text"]
        chosen = [{"review_id": rid, "text": texts[rid]} for rid in panel["review_ids"][len(done):]]
    deadline = t_gen + float(cfg["generation_wall_sec"]) - float(cfg["generation_reserve_sec"])
    blocks = list(done or [])
    ckpt = art / "steering_confirmation.jsonl"
    for row in chosen:
        if time.time() >= deadline:
            break
        text = render(row["text"])
        seed = PARENT.review_seed(row["review_id"])
        rec = {"review_id": row["review_id"], "seed": seed}
        for arm, direction in arms.items():
            if flags.get(arm):
                rec[arm] = {"non_evaluable": True, "positive_score": None}
                continue
            new_ids, cont = generate(text, direction, seed)
            positive, binary = pos_of(cont)
            rec[arm] = {"prefix": text, "continuation": cont, "n_new": len(new_ids), "positive_score": positive, "positive": bool(binary), "seed": seed, **PARENT.quality(cont, len(new_ids), max_new)}
        blocks.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if len(blocks) % 10 == 0:
            note(f"generation {len(blocks)}/{panel['n']}")
    contrasts = generation_contrasts(blocks, cfg)
    if contrasts:
        contrasts["panel_complete"] = bool(len(blocks) == int(panel["n"]) and int(panel["n"]) > 0)
        recovers = contrasts.get("l13_minus_l9", {}).get("lower95", -1) > 0 and contrasts.get("l13_minus_unedited", {}).get("lower95", -1) > 0
        contrasts["l13_recovers"] = bool(recovers and contrasts["panel_complete"])
    save_json(art / "generation_contrasts.json", contrasts)
    write_blind(blocks[: int(cfg["blind_n"])], list(arms), art)
    del model
    torch.cuda.empty_cache()
    summary = {}
    for arm in arms:
        scored = [block[arm] for block in blocks if block[arm].get("positive_score") is not None]
        summary[arm] = {
            "n": len(scored),
            "positive": int(sum(int(row["positive"]) for row in scored)) if scored else 0,
            "mean_score": float(np.mean([row["positive_score"] for row in scored])) if scored else None,
            "empty": int(sum(int(row.get("empty") or 0) for row in scored)),
            "repetitive": int(sum(int(row.get("repetitive") or 0) for row in scored)),
            "truncated": int(sum(int(row.get("truncated") or 0) for row in scored)),
        }
    return {"ran": True, "n": len(blocks), "frozen_n": int(panel["n"]), "partial": len(blocks) < int(panel["n"]), "summary": summary, "contrasts": contrasts}


def generation_contrasts(blocks, cfg) -> dict:
    usable = [row for row in blocks if row["unedited"].get("positive_score") is not None and row["matched_l13"].get("positive_score") is not None and row["matched_l9"].get("positive_score") is not None]
    if not usable:
        return {}
    def score(row, arm):
        return float(row[arm]["positive_score"])
    first = f"shuffle_{cfg['shuffle_seeds'][0]}"
    out = {
        "l13_minus_l9": PARENT.boot_mean([score(r, "matched_l13") - score(r, "matched_l9") for r in usable], int(cfg["n_boot"])),
        "l13_minus_unedited": PARENT.boot_mean([score(r, "matched_l13") - score(r, "unedited") for r in usable], int(cfg["n_boot"])),
        "l9_minus_unedited": PARENT.boot_mean([score(r, "matched_l9") - score(r, "unedited") for r in usable], int(cfg["n_boot"])),
        "mistral_minus_unedited": PARENT.boot_mean([score(r, "archived_mistral") - score(r, "unedited") for r in usable], int(cfg["n_boot"])),
        "native_minus_unedited": PARENT.boot_mean([score(r, "native") - score(r, "unedited") for r in usable], int(cfg["n_boot"])),
    }
    if first in usable[0] and usable[0][first].get("positive_score") is not None:
        out["l13_minus_shuffle"] = PARENT.boot_mean([score(r, "matched_l13") - score(r, first) for r in usable], int(cfg["n_boot"]))
    l13_over_l9 = out["l13_minus_l9"]["lower95"] > 0
    l13_over_base = out["l13_minus_unedited"]["lower95"] > 0
    out["l13_recovers"] = bool(l13_over_l9 and l13_over_base and len(usable) == int(cfg.get("_frozen_n", len(usable))))
    return out


def generation_excluded(work, cfg, manifest) -> set[str]:
    excluded = set()
    prior_path = work / cfg["prior_actadd_splits"]
    actadd = set()
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        for rows in prior.values():
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and "review_id" in row:
                        actadd.add(row["review_id"])
                        excluded.add(row["review_id"])
    from datasets import load_dataset
    test_neg = [f"test:{i}" for i, row in enumerate(load_dataset("stanfordnlp/imdb")["test"]) if int(row["label"]) == 0]
    pool = [rid for rid in test_neg if rid not in actadd]
    np.random.default_rng(int(cfg["parent_split_seed"])).shuffle(pool)
    excluded.update(pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])])
    for rel in (
        cfg["cca_run"] + "/artifacts/panel.json",
        cfg["cca_run"] + "/artifacts/cca_manifest.json",
        cfg["retention_run"] + "/artifacts/panel_manifest.json",
        cfg["gemma_run"] + "/artifacts/native_splits.json",
        cfg["moonlight_run"] + "/artifacts/panel_manifest.json",
        cfg["moonlight_run"] + "/artifacts/norm_split.json",
        cfg["parent_geometry"] + "/artifacts/steering_panel.json",
        cfg["parent_swap"] + "/artifacts/steering_panel.json",
    ):
        path = work / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for key in ("review_ids", "fit_review_ids", "rank_review_ids", "timing_review_ids", "calibration", "validation"):
            for item in payload.get(key) or []:
                excluded.add(item["review_id"] if isinstance(item, dict) else item)
    for rel in (
        cfg["sentiment_parent"] + "/artifacts/confirmation_qwen.jsonl",
        cfg["sentiment_parent"] + "/artifacts/confirmation_olmo.jsonl",
        cfg["moonlight_run"] + "/artifacts/confirmation.jsonl",
        cfg["parent_geometry"] + "/artifacts/steering_confirmation.jsonl",
        cfg["parent_swap"] + "/artifacts/steering_confirmation.jsonl",
    ):
        path = work / rel
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    excluded.add(json.loads(line)["review_id"])
    return excluded


def write_blind(blocks, arms, art: Path) -> None:
    if not blocks:
        return
    rng = np.random.default_rng(0)
    lines = ["# Blinded continuations", "", "Rate each continuation separately. Leave scores blank.", ""]
    key = []
    letters = "ABCDEFGHIJKLMNOP"
    for index, block in enumerate(blocks, start=1):
        order = list(arms)
        rng.shuffle(order)
        lines += [f"## Case {index}", "", block["unedited"].get("prefix", ""), ""]
        for letter, arm in zip(letters, order):
            lines += [f"### {letter}", "", block[arm].get("continuation") or "", ""]
            key.append({"case": index, "letter": letter, "arm": arm, "review_id": block["review_id"]})
    (art / "blind_sheet.md").write_text("\n".join(lines) + "\n")
    save_json(art / "blind_key.json", key)
    save_json(art / "blind_ratings.json", {"locked": False, "ratings": []})


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}], lower bound {block['lower95']:+.3f}"


def render_md(state, cfg) -> str:
    lines = ["# RESULTS — `gemma_layer_anchor_v1`", ""]
    rep = (state.get("representation") or {}).get("layers") or {}
    teacher = (state.get("teacher") or {}).get("contrasts") or {}
    gen = state.get("generation") or {}
    contrasts = gen.get("contrasts") or {}
    lines.append("Did changing Gemma's layer improve the shared-coordinate signal, the teacher-forced preference, and free generation?")
    if rep:
        lines.append(f"Gemma L9 projection {fmt(rep.get('gemma_l9', {}).get('raw'))}. Gemma L13 projection {fmt(rep.get('gemma_l13', {}).get('raw'))}. Mistral projection {fmt(rep.get('mistral', {}).get('raw'))}.")
        lines.append(f"Divided by the common unlabeled z_ref standard deviation { (state.get('representation') or {}).get('common_unlabeled_sd') }: L9 {fmt(rep.get('gemma_l9', {}).get('standardised'))}, L13 {fmt(rep.get('gemma_l13', {}).get('standardised'))}, Mistral {fmt(rep.get('mistral', {}).get('standardised'))}.")
    lines.append(f"Teacher-forced L13 minus L9 at positive dose: {fmt(teacher.get('l13_minus_l9_plus'))}.")
    lines.append(f"Free generation L13 minus L9: {fmt(contrasts.get('l13_minus_l9'))}. L13 minus unedited: {fmt(contrasts.get('l13_minus_unedited'))}.")
    if contrasts:
        lines.append("An L13 gain over L9 counts as recovery only together with a gain over unedited. The joint flag is " + str(bool(contrasts.get("l13_recovers"))) + ".")
    lines.append(f"L13 minus its prespecified shuffle: {fmt(contrasts.get('l13_minus_shuffle'))}. Archived Mistral minus unedited: {fmt(contrasts.get('mistral_minus_unedited'))}. Native minus unedited: {fmt(contrasts.get('native_minus_unedited'))}.")
    diag = state.get("diagnostics") or {}
    if diag.get("archived_vs_matched_l9") is not None:
        lines.append(f"Cosine between the new matched L9 edit and the archived L9 consensus edit, before scaling: {diag['archived_vs_matched_l9']:.3f}.")
    if gen.get("partial"):
        lines.append(f"Free generation is partial: {gen.get('n')} of {gen.get('frozen_n')}.")
    elif gen.get("n"):
        lines.append(f"Free generation used the full frozen panel, n={gen.get('n')}.")
    summary = gen.get("summary") or {}
    if summary:
        lines += ["", "| arm | n | positive | score | empty | repetitive | truncated |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, row in summary.items():
            if row.get("mean_score") is not None:
                lines.append(f"| {arm} | {row['n']} | {row['positive']} | {row['mean_score']:.3f} | {row['empty']} | {row['repetitive']} | {row['truncated']} |")
    if state.get("error"):
        lines += ["", "## Stop", "", "```", state["error"][-2000:], "```"]
    lines.append("")
    lines.append("These two Gemma layers and this chart are the whole layer comparison. One sentiment command in Moonlight is the behavioral result.")
    lines.append("Reproduction: `python scripts/run_gemma_layer_anchor.py --config configs/gemma_layer_anchor_v1.json --work-root $WORK`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
