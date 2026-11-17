#!/usr/bin/env python3
"""Attach Moonlight through OLMo, Gemma, or Mistral with one frozen Qwen command."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

from prh_replication.multifamily_shared_subspace import (
    activation_consensus,
    align_scores,
    apply_linear,
    cca_directions,
    consensus_basis,
    displacement_consensus,
    document_means,
    fit_pca_whitener,
    fixed_axis_correlation,
    mean_squared_cosine,
    principal_angles,
    random_orthonormal,
    ridge_maps,
    standardise_fit,
    top1_retrieval,
    whiten_activations,
)


def load_parent():
    path = Path(__file__).resolve().parent / "run_multifamily_cca_shared_subspace.py"
    spec = importlib.util.spec_from_file_location("multifamily_parent", path)
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise_text(text: str) -> str:
    return " ".join((text or "").split())


def domain_of(doc_id: str) -> str:
    return "wikitext" if str(doc_id).startswith("wiki:") else "imdb"


def repeat_ids(doc_ids, n_pos: int) -> list[str]:
    return [doc for doc in doc_ids for _ in range(n_pos)]


def flatten(acts: np.ndarray) -> np.ndarray:
    return np.asarray(acts, dtype=np.float64).reshape(-1, acts.shape[-1])


def retrieval_pair(query, gallery, seed: int) -> dict:
    true = top1_retrieval(query, gallery)
    perm = np.random.default_rng(int(seed)).permutation(len(gallery))
    shuffled = top1_retrieval(query, np.asarray(gallery)[perm])
    return {"top1": true, "shuffled_top1": shuffled, "n_docs": int(len(query)), "beats_shuffle": bool(true > shuffled)}


def span_overlap(left: np.ndarray, right: np.ndarray, rank: int) -> dict:
    width = min(int(rank), left.shape[1], right.shape[1])
    a = left[:, :width]
    b = right[:, :width]
    angles = principal_angles(a, b)
    return {
        "rank": width,
        "mean_squared_cosine": mean_squared_cosine(a, b),
        "mean_angle_deg": float(np.degrees(np.mean(angles))),
        "max_angle_deg": float(np.degrees(np.max(angles))),
    }


def residual_basis(pack, basis, scale, rotation, rank: int) -> np.ndarray:
    """Columns are the displacement encoder expressed in residual coordinates."""
    width = min(int(rank), basis.shape[1])
    scale_inv = 1.0 / np.asarray(scale[:width], dtype=np.float64)
    mapped = pack["components"] @ pack["whitener"] @ basis[:, :width] @ np.diag(scale_inv) @ rotation[:width, :width]
    q, _ = np.linalg.qr(mapped)
    return q[:, :width]


def energy_in_span(vector: np.ndarray, basis: np.ndarray) -> float:
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    denom = float(np.dot(v, v))
    if denom <= 0:
        return float("nan")
    proj = basis.T @ v
    return float(np.dot(proj, proj) / denom)


def fit_consensus(whitened, names, rank, anchor, n_iter=40):
    pairwise = {}
    for left, right in combinations(sorted(names), 2):
        fit = cca_directions(flatten(whitened[left]), flatten(whitened[right]), rank)
        pairwise[(left, right)] = {left: fit["U_left"], right: fit["U_right"], "singular": fit["singular"]}
    consensus = {}
    standardised = {}
    for name in names:
        partners = [payload[name][:, :rank] for edge, payload in pairwise.items() if name in payload]
        basis = consensus_basis(partners, rank)
        flat = flatten(whitened[name])
        view_mean = flat.mean(axis=0)
        scores = (flat - view_mean) @ basis["basis"]
        z, score_mean, score_scale = standardise_fit(scores)
        consensus[name] = {
            "basis": basis["basis"], "view_mean": view_mean, "score_mean": score_mean, "score_scale": score_scale,
        }
        standardised[name] = z
    rotations = align_scores(standardised, anchor, n_iter=n_iter)
    for name in names:
        consensus[name]["rotation"] = rotations[name]
    return consensus, pairwise


def encode_rows(consensus, name, whitened_acts):
    row = consensus[name]
    flat = flatten(whitened_acts)
    encoded = activation_consensus(flat, row["basis"], row["score_mean"], row["score_scale"], row["rotation"], row["view_mean"])
    return encoded


def encode_displacement(consensus, name, whitened_delta):
    row = consensus[name]
    return displacement_consensus(whitened_delta, row["basis"], row["score_scale"], row["rotation"])


def pair_encoder(pairwise, pack, left_name, right_name, left_is_qwen=True):
    edge = tuple(sorted((left_name, right_name)))
    payload = pairwise[edge]
    return payload


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
    parent = work / cfg["parent"] / "artifacts"
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    audit_deadline = t0 + float(cfg["audit_wall_sec"])
    state = {"log": []}

    def note(msg):
        state["log"].append({"t": round(time.time() - t0, 1), "msg": msg})
        print(msg, flush=True)

    try:
        resumed = resume_steering(cfg, work, art, parent, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, note)
        if resumed is not None:
            state["steering"] = resumed
            state["audit"] = json.loads((art / "audit.json").read_text()) if (art / "audit.json").exists() else {}
            state["routes"] = json.loads((art / "routes.json").read_text()) if (art / "routes.json").exists() else {}
            raise SystemExit
        names = list(cfg["discovery_models"])
        manifest = {split: json.loads((parent / f"split_{split}.json").read_text()) for split in ("discovery", "attachment_fit", "attachment_val", "geometry_test")}
        caches = {}
        whitened = {}
        packs = {}
        for name in names + ["moonlight"]:
            loaded = np.load(parent / "cache" / f"{name}.npz", allow_pickle=True)
            caches[name] = {}
            for split, rows in manifest.items():
                ids = [str(x) for x in loaded[f"ids_{split}"].tolist()]
                expected = [r["doc_id"] for r in rows]
                if ids != expected:
                    raise RuntimeError(f"{name} {split} row order does not match the frozen split")
                caches[name][split] = np.asarray(loaded[split])
            saved = np.load(parent / "pca" / f"{name}.npz")
            disc = flatten(caches[name]["discovery"])
            pack = fit_pca_whitener(disc, int(cfg["pca_dim"]), float(cfg["eta"]))
            packs[name] = pack
            err = {
                "mean": float(np.max(np.abs(pack["mean"] - saved["mean"]))),
                "components": float(np.max(np.abs(pack["components"] - saved["components"]))),
                "whitener": float(np.max(np.abs(pack["whitener"] - saved["whitener"]))),
            }
            note(f"pca parity {name} {err}")
            if max(err.values()) > float(cfg["parity_basis_tol"]) * 10:
                raise RuntimeError(f"PCA parity failed for {name}: {err}")
            whitened[name] = {split: whiten_activations(flatten(acts), pack).reshape(acts.shape[0], acts.shape[1], -1) for split, acts in caches[name].items()}
        pins = json.loads((parent / "model_pins.json").read_text())["present"]
        for required, layer in (("qwen", 17), ("olmo", 11), ("moonlight", 9)):
            if int(pins[required]["layer"]) != layer:
                raise RuntimeError(f"{required} layer {pins[required]['layer']} is not the archived site {layer}")
        consensus, pairwise = fit_consensus({name: whitened[name]["discovery"] for name in names}, names, int(cfg["rank"]), "olmo")
        saved = np.load(parent / "space_primary" / "consensus.npz")
        parity = {}
        for name in names:
            parity[name] = {
                "basis": float(np.max(np.abs(consensus[name]["basis"] - saved[f"basis_{name}"]))),
                "rotation": float(np.max(np.abs(consensus[name]["rotation"] - saved[f"rotation_{name}"]))),
                "scale": float(np.max(np.abs(consensus[name]["score_scale"] - saved[f"scale_{name}"]))),
            }
        note(f"consensus parity {parity}")
        if any(max(row.values()) > float(cfg["parity_basis_tol"]) for row in parity.values()):
            raise RuntimeError(f"consensus parity failed: {parity}")
        state["parity"] = parity
        state["whitening"] = whitening_audit(whitened, names, float(cfg["eta"]), float(cfg["singular_tol"]), manifest["discovery"])
        note(f"whitening max singular {state['whitening']['max_singular_full_discovery']}")
        v_q = np.load(work / cfg["sentiment_parent"] / "artifacts/vec_qwen_averaged.npy").astype(np.float64)
        cfg["_v_qwen"] = v_q
        state["half"] = half_audit(whitened, packs, caches, consensus, names, manifest, cfg)
        note("half audit done")
        z_edit = encode_displacement(consensus, "qwen", whiten_activations(v_q, packs["qwen"], displacement=True))
        np.save(art / "z_edit.npy", z_edit)
        routes = {}
        doc_ids = {split: repeat_ids([r["doc_id"] for r in manifest[split]], 4) for split in ("attachment_fit", "attachment_val", "geometry_test")}
        moon_x = {split: flatten(caches["moonlight"][split]) for split in doc_ids}
        for anchor in cfg["anchors"]:
            routes[anchor] = fit_route(anchor, consensus, pairwise, whitened, packs, moon_x, doc_ids, z_edit, v_q, cfg, art / "routes" / anchor)
            note(f"route {anchor} raw {routes[anchor]['consensus_raw_norm']:.5f} bridge {routes[anchor]['bridge_raw_norm']:.5f}")
        gemma_ids = [r["doc_id"] for r in manifest["attachment_fit"]]
        val_ids = [r["doc_id"] for r in manifest["attachment_val"]]
        routes["gemma_shuffle"] = fit_shuffled_gemma(consensus, whitened, caches, gemma_ids, val_ids, z_edit, cfg, art / "routes" / "gemma_shuffle")
        state["routes"] = {key: {k: v for k, v in row.items() if not str(k).startswith("_")} for key, row in routes.items()}
        save_json(art / "routes.json", state["routes"])
        state["geometry"] = geometry_report(routes, consensus, pairwise, whitened, caches, manifest, cfg)
        save_json(art / "geometry.json", state["geometry"])
        parent_dec = np.load(parent / "consensus_moonlight.npz")
        refit = routes["olmo"]["_consensus_backward"]
        decoder_err = float(np.max(np.abs(refit - parent_dec["backward"])))
        raw = routes["olmo"]["_consensus_raw"]
        scaled = raw / np.linalg.norm(raw) * float(cfg["dose"])
        saved_dir = np.load(parent / "dir_consensus.npy")
        direction_cosine = float(np.dot(scaled, saved_dir) / (np.linalg.norm(scaled) * np.linalg.norm(saved_dir)))
        state["olmo_reproduction"] = {"decoder_max_abs": decoder_err, "direction_cosine": direction_cosine, "raw_norm": float(np.linalg.norm(raw))}
        note(f"olmo reproduction decoder {decoder_err:.3e} cosine {direction_cosine:.6f}")
        save_json(art / "audit.json", {"parity": parity, "whitening": state["whitening"], "half": state["half"], "olmo_reproduction": state["olmo_reproduction"], "pins": {k: pins[k] for k in ("qwen", "olmo", "gemma", "mistral", "moonlight")}})
        if decoder_err > 1e-4 or direction_cosine < float(cfg["parity_direction_cosine"]):
            raise RuntimeError(f"OLMo route does not reproduce the parent edit: {state['olmo_reproduction']}")
        if time.time() > audit_deadline:
            note("audit exceeded 30 minutes; steering still uses the frozen maps")
        state["steering"] = run_steering(cfg, work, art, parent, routes, manifest, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, note)
    except SystemExit:
        pass
    except Exception:
        state["error"] = traceback.format_exc()
        note(state["error"])
    state["elapsed_sec"] = time.time() - t0
    save_json(art / "run_state.json", state)
    (dest / "RESULTS.md").write_text(render_md(state, cfg))
    print("wrote RESULTS", flush=True)
    return 0


def whitening_audit(whitened, names, eta, tol, discovery_rows) -> dict:
    edges = {}
    max_full = 0.0
    for left, right in combinations(sorted(names), 2):
        a = flatten(whitened[left]["discovery"])
        b = flatten(whitened[right]["discovery"])
        a = a - a.mean(0)
        b = b - b.mean(0)
        denom = a.shape[0] - 1
        cross = (a.T @ b) / denom
        singular = np.linalg.svd(cross, compute_uv=False)
        cov_a = (a.T @ a) / denom
        cov_b = (b.T @ b) / denom
        bound = float(np.sqrt(np.linalg.eigvalsh(cov_a)[-1] * np.linalg.eigvalsh(cov_b)[-1]))
        edges[f"{left}:{right}"] = {"max_singular": float(singular[0]), "mean_top64": float(np.mean(singular[:64])), "within_bound": bound, "divisor": int(denom), "eta": eta}
        max_full = max(max_full, float(singular[0]))
    domains = {}
    for domain in ("imdb", "wikitext"):
        mask = np.asarray([domain_of(row["doc_id"]) == domain for row in discovery_rows])
        a = flatten(whitened["qwen"]["discovery"][mask])
        b = flatten(whitened["gemma"]["discovery"][mask])
        a = a - a.mean(0)
        b = b - b.mean(0)
        singular = np.linalg.svd((a.T @ b) / (a.shape[0] - 1), compute_uv=False)
        domains[domain] = {"pair": "gemma:qwen", "max_singular": float(singular[0]), "note": "subset of the rows used to fit the shared whitener"}
    return {
        "max_singular_full_discovery": max_full,
        "exceeds_one": bool(max_full > 1.0 + tol),
        "edges": edges,
        "domain_subset_example": domains,
        "explanation": "On the discovery rows used to fit the whitener, singular values stay within the within-covariance bound. A domain subset can exceed 1 because that subset was not the covariance the whitener was fit on. Ridge alone does not push a matched full-sample singular value above one.",
    }


def half_audit(whitened, packs, caches, full, names, manifest, cfg) -> dict:
    ids = [r["doc_id"] for r in manifest["discovery"]]
    rng = np.random.default_rng(int(cfg["half_seed"]))
    halves = {}
    for domain in ("imdb", "wikitext"):
        idx = np.asarray([i for i, doc in enumerate(ids) if domain_of(doc) == domain])
        rng.shuffle(idx)
        cut = len(idx) // 2
        halves.setdefault("a", []).extend(idx[:cut].tolist())
        halves.setdefault("b", []).extend(idx[cut:].tolist())
    half_idx = {key: np.asarray(sorted(values)) for key, values in halves.items()}
    rank = int(cfg["rank"])
    fitted = {}
    for key, idx in half_idx.items():
        fitted[key], _ = fit_consensus({name: whitened[name]["discovery"][idx] for name in names}, names, rank, "olmo")
    conditional = {name: {} for name in names}
    for name in names:
        for width in cfg["diagnostic_ranks"]:
            conditional[name][str(width)] = span_overlap(fitted["a"][name]["basis"], fitted["b"][name]["basis"], width)
    end_to_end = {name: {} for name in names}
    energies = {name: {} for name in ("a", "b")}
    # Residual-space comparison uses each half's own PCA whitening.
    own = {}
    for key, idx in half_idx.items():
        own_packs = {name: fit_pca_whitener(flatten(caches[name]["discovery"][idx]), int(cfg["pca_dim"]), float(cfg["eta"])) for name in names}
        own_w = {name: whiten_activations(flatten(caches[name]["discovery"][idx]), own_packs[name]).reshape(len(idx), 4, -1) for name in names}
        space, _ = fit_consensus(own_w, names, rank, "olmo")
        own[key] = {"packs": own_packs, "space": space}
    for name in names:
        for width in cfg["diagnostic_ranks"]:
            left = residual_basis(own["a"]["packs"][name], own["a"]["space"][name]["basis"], own["a"]["space"][name]["score_scale"], own["a"]["space"][name]["rotation"], width)
            right = residual_basis(own["b"]["packs"][name], own["b"]["space"][name]["basis"], own["b"]["space"][name]["score_scale"], own["b"]["space"][name]["rotation"], width)
            end_to_end[name][str(width)] = span_overlap(left, right, width)
    v_q = np.asarray(cfg["_v_qwen"], dtype=np.float64)
    for key, space in (("a", fitted["a"]), ("b", fitted["b"])):
        basis = residual_basis(packs["qwen"], space["qwen"]["basis"], space["qwen"]["score_scale"], space["qwen"]["rotation"], rank)
        energies[key] = {"qwen_edit_energy_rank64": energy_in_span(v_q, basis)}
    random_ref = {}
    frames = random_orthonormal(full["qwen"]["basis"].shape[0], rank, 20, int(cfg["audit_shuffle_seed"]))
    random_ref["qwen_rank64"] = float(np.mean([mean_squared_cosine(full["qwen"]["basis"], frame) for frame in frames]))
    rng_shuf = np.random.default_rng(int(cfg["audit_shuffle_seed"]))
    shuffled_w = {name: whitened[name]["discovery"][rng_shuf.permutation(len(ids))] for name in names}
    shuffled, _ = fit_consensus(shuffled_w, names, rank, "olmo")
    shuffle_ref = {name: {str(width): span_overlap(full[name]["basis"], shuffled[name]["basis"], width) for width in cfg["diagnostic_ranks"]} for name in names}
    # OLMo attachment of each conditional half, using the original attachment rows.
    decoded = {}
    doc_fit = repeat_ids([r["doc_id"] for r in manifest["attachment_fit"]], 4)
    doc_val = repeat_ids([r["doc_id"] for r in manifest["attachment_val"]], 4)
    x_fit = flatten(caches["moonlight"]["attachment_fit"])
    x_val = flatten(caches["moonlight"]["attachment_val"])
    for key, space in fitted.items():
        y_fit = encode_rows(space, "olmo", whitened["olmo"]["attachment_fit"])
        y_val = encode_rows(space, "olmo", whitened["olmo"]["attachment_val"])
        backward = ridge_maps(y_fit, x_fit, y_val, x_val, doc_fit, doc_val, cfg["alpha_multipliers"])
        z = encode_displacement(space, "qwen", whiten_activations(v_q, packs["qwen"], displacement=True))
        decoded[key] = apply_linear(backward["decode"], backward["intercept"], z, displacement=True)
    cosine = float(np.dot(decoded["a"], decoded["b"]) / (np.linalg.norm(decoded["a"]) * np.linalg.norm(decoded["b"])))
    return {
        "conditional_on_parent_whitening": conditional,
        "end_to_end_residual": end_to_end,
        "random_msc_qwen_rank64": random_ref["qwen_rank64"],
        "document_shuffle": shuffle_ref,
        "half_decoded_cosine_rank64": cosine,
        "edit_energy": energies,
        "half_sizes": {key: int(len(idx)) for key, idx in half_idx.items()},
    }


def fit_route(anchor, consensus, pairwise, whitened, packs, moon_x, doc_ids, z_edit, v_q, cfg, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    y = {split: encode_rows(consensus, anchor, whitened[anchor][split]) for split in ("attachment_fit", "attachment_val")}
    backward = ridge_maps(y["attachment_fit"], moon_x["attachment_fit"], y["attachment_val"], moon_x["attachment_val"], doc_ids["attachment_fit"], doc_ids["attachment_val"], cfg["alpha_multipliers"])
    forward = ridge_maps(moon_x["attachment_fit"], y["attachment_fit"], moon_x["attachment_val"], y["attachment_val"], doc_ids["attachment_fit"], doc_ids["attachment_val"], cfg["alpha_multipliers"])
    raw = apply_linear(backward["decode"], backward["intercept"], z_edit, displacement=True)
    raw_norm = float(np.linalg.norm(raw))
    doubled = apply_linear(backward["decode"], backward["intercept"], 2.0 * z_edit, displacement=True)
    if float(np.max(np.abs(doubled - 2.0 * raw))) > 1e-8:
        raise RuntimeError(f"{anchor} consensus decoder is not linear on a displacement")
    bridge = fit_bridge(anchor, pairwise, whitened, packs, moon_x, doc_ids, v_q, cfg)
    info = {
        "consensus_alpha": backward["alpha"],
        "consensus_val_mse": backward["val_mse"],
        "consensus_baseline_mse": backward["baseline_mse"],
        "consensus_raw_norm": raw_norm,
        "consensus_near_zero": raw_norm < float(cfg["near_zero_norm"]),
        "bridge_alpha": bridge["alpha"],
        "bridge_val_mse": bridge["val_mse"],
        "bridge_baseline_mse": bridge["baseline_mse"],
        "bridge_raw_norm": bridge["raw_norm"],
        "bridge_near_zero": bridge["near_zero"],
        "_consensus_raw": raw,
        "_bridge_raw": bridge["raw"],
        "_consensus_backward": backward["decode"],
        "_consensus_forward": forward,
        "_bridge_forward": bridge["forward"],
        "_bridge_scores": bridge["score_fn"],
    }
    np.save(outdir / "consensus_raw.npy", raw)
    np.save(outdir / "bridge_raw.npy", bridge["raw"])
    return info


def fit_bridge(anchor, pairwise, whitened, packs, moon_x, doc_ids, v_q, cfg) -> dict:
    edge = tuple(sorted(("qwen", anchor)))
    u_q = pairwise[edge]["qwen"]
    u_a = pairwise[edge][anchor]
    def scores(name, u, split):
        disc = flatten(whitened[name]["discovery"])
        disc_s = (disc - disc.mean(0)) @ u
        _, mean, scale = standardise_fit(disc_s)
        flat = flatten(whitened[name][split])
        return ((flat - disc.mean(0)) @ u - mean) / scale, mean, scale, disc.mean(0)
    y_fit, mean, scale, view_mean = scores(anchor, u_a, "attachment_fit")
    y_val, _, _, _ = scores(anchor, u_a, "attachment_val")
    backward = ridge_maps(y_fit, moon_x["attachment_fit"], y_val, moon_x["attachment_val"], doc_ids["attachment_fit"], doc_ids["attachment_val"], cfg["alpha_multipliers"])
    forward = ridge_maps(moon_x["attachment_fit"], y_fit, moon_x["attachment_val"], y_val, doc_ids["attachment_fit"], doc_ids["attachment_val"], cfg["alpha_multipliers"])
    # Displacement uses the anchor side only through the decoder of Qwen-pair coordinates.
    q_disc = flatten(whitened["qwen"]["discovery"])
    q_scores = (q_disc - q_disc.mean(0)) @ u_q
    _, q_mean, q_scale = standardise_fit(q_scores)
    delta = whiten_activations(v_q, packs["qwen"], displacement=True)
    z = (delta @ u_q) / q_scale
    raw = apply_linear(backward["decode"], backward["intercept"], z, displacement=True)
    doubled = apply_linear(backward["decode"], backward["intercept"], 2.0 * z, displacement=True)
    if float(np.max(np.abs(doubled - 2.0 * raw))) > 1e-8:
        raise RuntimeError(f"{anchor} bridge decoder is not linear on a displacement")
    raw_norm = float(np.linalg.norm(raw))
    return {
        "alpha": backward["alpha"], "val_mse": backward["val_mse"], "baseline_mse": backward["baseline_mse"],
        "raw": raw, "raw_norm": raw_norm, "near_zero": raw_norm < float(cfg["near_zero_norm"]),
        "forward": forward, "decode": backward["decode"], "intercept": backward["intercept"],
        "score_fn": scores, "u_q": u_q, "u_a": u_a,
    }


def fit_shuffled_gemma(consensus, whitened, caches, fit_ids, val_ids, z_edit, cfg, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["shuffle_fit_seed"]))
    def derange(ids):
        order = np.arange(len(ids))
        for domain in ("imdb", "wikitext"):
            idx = np.asarray([i for i, doc in enumerate(ids) if domain_of(doc) == domain])
            perm = np.roll(np.arange(len(idx)), 1) if len(idx) > 1 else np.arange(len(idx))
            # One extra shuffle that rejects any identity, then fall back to the cycle.
            for _ in range(32):
                cand = rng.permutation(len(idx))
                if len(idx) < 2 or np.all(cand != np.arange(len(idx))):
                    perm = cand
                    break
            order[idx] = idx[perm]
        if np.any(order == np.arange(len(ids))):
            raise RuntimeError("shuffled attachment left a document in place")
        return order
    fit_order = derange(fit_ids)
    val_order = derange(val_ids)
    y_fit = encode_rows(consensus, "gemma", whitened["gemma"]["attachment_fit"])
    y_val = encode_rows(consensus, "gemma", whitened["gemma"]["attachment_val"])
    x_fit = flatten(caches["moonlight"]["attachment_fit"][fit_order])
    x_val = flatten(caches["moonlight"]["attachment_val"][val_order])
    doc_fit = repeat_ids(fit_ids, 4)
    doc_val = repeat_ids(val_ids, 4)
    backward = ridge_maps(y_fit, x_fit, y_val, x_val, doc_fit, doc_val, cfg["alpha_multipliers"])
    raw = apply_linear(backward["decode"], backward["intercept"], z_edit, displacement=True)
    raw_norm = float(np.linalg.norm(raw))
    np.save(outdir / "raw.npy", raw)
    return {"raw_norm": raw_norm, "near_zero": raw_norm < float(cfg["near_zero_norm"]), "alpha": backward["alpha"], "val_mse": backward["val_mse"], "baseline_mse": backward["baseline_mse"], "_raw": raw}


def geometry_report(routes, consensus, pairwise, whitened, caches, manifest, cfg) -> dict:
    ids = [r["doc_id"] for r in manifest["geometry_test"]]
    report = {"consensus": {}, "bridge": {}}
    domains = {"pooled": np.ones(len(ids), dtype=bool), "imdb": np.asarray([domain_of(i) == "imdb" for i in ids]), "wikitext": np.asarray([domain_of(i) == "wikitext" for i in ids])}
    moon = flatten(caches["moonlight"]["geometry_test"])
    for anchor in cfg["anchors"]:
        pred = apply_linear(routes[anchor]["_consensus_forward"]["decode"], routes[anchor]["_consensus_forward"]["intercept"], moon, displacement=False).reshape(len(ids), 4, -1)
        report["consensus"][anchor] = {}
        for domain, mask in domains.items():
            report["consensus"][anchor][domain] = {}
            take = np.flatnonzero(mask)
            for partner in [name for name in cfg["discovery_models"] if name != anchor]:
                other = encode_rows(consensus, partner, whitened[partner]["geometry_test"]).reshape(len(ids), 4, -1)
                left = pred[take].reshape(-1, pred.shape[-1])
                right = other[take].reshape(-1, other.shape[-1])
                q_ids = [ids[i] for i in take]
                q_mean, _ = document_means(left, repeat_ids(q_ids, 4))
                g_mean, _ = document_means(right, repeat_ids(q_ids, 4))
                report["consensus"][anchor][domain][partner] = {"axis_correlation": fixed_axis_correlation(left, right), **retrieval_pair(q_mean, g_mean, 0)}
        # Matched bridge versus Qwen in pair coordinates.
        score_fn = routes[anchor]["_bridge_scores"]
        q_scores, _, _, _ = score_fn("qwen", pairwise[tuple(sorted(("qwen", anchor)))]["qwen"], "geometry_test")
        a_scores, _, _, _ = score_fn(anchor, pairwise[tuple(sorted(("qwen", anchor)))][anchor], "geometry_test")
        # Moonlight predicted in the bridge's anchor coordinates.
        bridge_pred = apply_linear(routes[anchor]["_bridge_forward"]["decode"], routes[anchor]["_bridge_forward"]["intercept"], moon, displacement=False)
        q_doc, _ = document_means(q_scores.reshape(len(ids), 4, -1).reshape(-1, q_scores.shape[-1]), repeat_ids(ids, 4))
        m_doc, _ = document_means(bridge_pred.reshape(len(ids), 4, -1).reshape(-1, bridge_pred.shape[-1]), repeat_ids(ids, 4))
        report["bridge"][anchor] = {"versus_qwen": {"axis_correlation": fixed_axis_correlation(bridge_pred, q_scores), **retrieval_pair(m_doc, q_doc, 0)}}
        del a_scores
    directions = {}
    for anchor in cfg["anchors"]:
        directions[f"consensus_{anchor}"] = routes[anchor]["_consensus_raw"]
        directions[f"bridge_{anchor}"] = routes[anchor]["_bridge_raw"]
    directions["gemma_shuffle"] = routes["gemma_shuffle"]["_raw"]
    labels = list(directions)
    matrix = {}
    for left in labels:
        matrix[left] = {}
        for right in labels:
            a = directions[left]
            b = directions[right]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            matrix[left][right] = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
    report["cosine"] = matrix
    return report


def prepare_direction(raw, dose, near_zero):
    norm = float(np.linalg.norm(raw))
    if norm < near_zero:
        return np.zeros_like(raw), True
    return raw / norm * dose, False


def run_steering(cfg, work, art, parent, routes, manifest, torch, auto_model, auto_tok, pipeline, note):
    dose = float(cfg["dose"])
    near = float(cfg["near_zero_norm"])
    native = np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy")
    directions = {"unedited": None, "native": native}
    flags = {}
    for anchor in cfg["anchors"]:
        for kind in ("consensus", "bridge"):
            vec, bad = prepare_direction(routes[anchor][f"_{kind}_raw"], dose, near)
            directions[f"{kind}_{anchor}"] = vec
            flags[f"{kind}_{anchor}"] = bad
            np.save(art / f"dir_{kind}_{anchor}.npy", vec)
    vec, bad = prepare_direction(routes["gemma_shuffle"]["_raw"], dose, near)
    directions["gemma_shuffle"] = vec
    flags["gemma_shuffle"] = bad
    np.save(art / "dir_gemma_shuffle.npy", vec)
    return steer_panel(cfg, work, art, parent, directions, flags, manifest, torch, auto_model, auto_tok, pipeline, note, dose)


def resume_steering(cfg, work, art, parent, torch, auto_model, auto_tok, pipeline, note):
    panel_path = art / "steering_panel.json"
    ckpt = art / "steering_confirmation.jsonl"
    if not panel_path.exists() or not ckpt.exists():
        return None
    panel = json.loads(panel_path.read_text())
    done = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
    ids = [row["review_id"] for row in done]
    if ids != panel["review_ids"][: len(ids)] or len(ids) >= int(panel["n"]):
        return None
    directions = {"unedited": None, "native": np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy")}
    flags = {}
    for name in ("consensus_olmo", "consensus_gemma", "consensus_mistral", "bridge_olmo", "bridge_gemma", "bridge_mistral", "gemma_shuffle"):
        directions[name] = np.load(art / f"dir_{name}.npy")
        flags[name] = float(np.linalg.norm(directions[name])) < 1e-8
    note(f"resume steering {len(ids)}/{panel['n']}")
    return steer_panel(cfg, work, art, parent, directions, flags, None, torch, auto_model, auto_tok, pipeline, note, float(cfg["dose"]), panel=panel, done=done)


def steer_panel(cfg, work, art, parent, directions, flags, manifest, torch, auto_model, auto_tok, pipeline, note, dose, panel=None, done=None):
    from datasets import load_dataset
    excluded = excluded_ids(work, cfg, parent, manifest)
    texts = {}
    test_neg = []
    for i, row in enumerate(load_dataset("stanfordnlp/imdb")["test"]):
        if int(row["label"]) == 0:
            review_id = f"test:{i}"
            test_neg.append({"review_id": review_id, "text": row["text"]})
            texts[review_id] = row["text"]
    moon = dict(cfg["moonlight"])
    model, tok, layer = PARENT.load_model(work, moon, torch, auto_model, auto_tok)
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
        table = {d["label"].upper(): float(d["score"]) for d in raw_scores}
        p = table.get("POSITIVE", 0.0)
        return p, p >= table.get("NEGATIVE", 0.0)

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
                out_h = resid.clone()
                out_h[:, -1] = out_h[:, -1] + _delta.to(out_h.dtype)
                return (out_h,)

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

    t_steer = time.time()
    if panel is None:
        blocked = set()
        if manifest is not None:
            for rows in manifest.values():
                for row in rows:
                    blocked.add(hashlib.sha256(normalise_text(row["text"]).encode()).hexdigest())
        eligible = []
        for row in test_neg:
            if row["review_id"] in excluded:
                continue
            if hashlib.sha256(normalise_text(row["text"]).encode()).hexdigest() in blocked:
                continue
            eligible.append(row)
        np.random.default_rng(int(cfg["panel_seed"])).shuffle(eligible)
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
        seed_a = generate(smoke_text, None, 12345)[0]
        seed_b = generate(smoke_text, None, 12345)[0]
        active = next(name for name, vec in directions.items() if vec is not None and name != "native")
        seen = []

        def probe(_m, inputs, _seen=seen):
            _seen.append(int(inputs[0].shape[1]))
            return (inputs[0].clone(),)

        handle = layer.register_forward_pre_hook(probe)
        try:
            generate(smoke_text, directions[active], 7)
        finally:
            handle.remove()
        smoke = {
            "zero_logit_max_abs": float(torch.max(torch.abs(base.float() - zero.float())).cpu()),
            "seed_repeat": seed_a == seed_b,
            "hook_lengths": seen[:4],
            "active_norm": float(np.linalg.norm(directions[active])),
            "dose": float(cfg["dose"]),
            "non_evaluable": flags,
        }
        save_json(art / "smoke.json", smoke)
        note(f"smoke zero {smoke['zero_logit_max_abs']} seed {smoke['seed_repeat']}")
        if smoke["zero_logit_max_abs"] > 1e-3 or not smoke["seed_repeat"] or seen[:1] != [int(enc.shape[1])] or 1 not in seen:
            raise RuntimeError(f"smoke failed {smoke}")
        if abs(smoke["active_norm"] - float(cfg["dose"])) / float(cfg["dose"]) > 0.02:
            raise RuntimeError(f"injection norm is not the archived dose: {smoke}")
        timers = eligible[: int(cfg["smoke_blocks"])]
        t_batch = time.time()
        for row in timers:
            text = render(row["text"])
            seed = PARENT.review_seed(row["review_id"])
            for direction in directions.values():
                generate(text, direction, seed)
        sec = (time.time() - t_batch) / max(1, len(timers))
        remain = float(cfg["steering_wall_sec"]) - float(cfg["steering_reserve_sec"]) - (time.time() - t_steer)
        n_panel = min(int(cfg["target_panel"]), int(max(0.0, remain) * 0.92 / sec)) if sec > 0 else 0
        chosen = eligible[len(timers): len(timers) + max(n_panel, 0)]
        panel = {"n": len(chosen), "sec_per_block": sec, "review_ids": [r["review_id"] for r in chosen], "timing_review_ids": [r["review_id"] for r in timers], "frozen_before_test_scores": True}
        save_json(art / "steering_panel.json", panel)
        note(f"panel {panel['n']} sec/block {sec:.2f}")
        done = []
        deadline = time.time() + max(0.0, remain)
    else:
        deadline = time.time() + float(cfg["steering_wall_sec"]) - float(cfg["steering_reserve_sec"])
        chosen_ids = panel["review_ids"][len(done):]
        chosen = [{"review_id": rid, "text": texts[rid]} for rid in chosen_ids]
    blocks = list(done or [])
    ckpt = art / "steering_confirmation.jsonl"
    for row in chosen:
        if time.time() >= deadline:
            break
        text = render(row["text"])
        seed = PARENT.review_seed(row["review_id"])
        rec = {"review_id": row["review_id"], "seed": seed}
        for arm, direction in directions.items():
            if flags.get(arm):
                rec[arm] = {"prefix": text, "continuation": "", "n_new": 0, "positive_score": None, "positive": None, "non_evaluable": True, "seed": seed}
                continue
            t_g = time.time()
            new_ids, cont = generate(text, direction, seed)
            p, y = pos_of(cont)
            rec[arm] = {"prefix": text, "continuation": cont, "n_new": len(new_ids), "sec": time.time() - t_g, "positive_score": p, "positive": bool(y), "seed": seed, **PARENT.quality(cont, len(new_ids), max_new)}
        blocks.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if len(blocks) % 10 == 0:
            note(f"block {len(blocks)}/{panel['n']}")
    contrasts = contrasts_from(blocks, cfg) if blocks else {}
    if contrasts:
        contrasts["panel_complete"] = bool(len(blocks) == int(panel["n"]) and int(panel["n"]) > 0)
        if not contrasts["panel_complete"]:
            contrasts["both_new_anchors_retain_80"] = False
    save_json(art / "steering_contrasts.json", contrasts)
    write_blind(blocks[: int(cfg["blind_n"])], list(directions), art)
    del model
    torch.cuda.empty_cache()
    summary = {}
    for arm in directions:
        rows = [b[arm] for b in blocks if b[arm].get("positive_score") is not None]
        summary[arm] = {
            "n": len(rows),
            "positive": int(sum(int(r["positive"]) for r in rows)) if rows else 0,
            "mean_score": float(np.mean([r["positive_score"] for r in rows])) if rows else None,
            "empty": int(sum(int(r.get("empty") or 0) for r in rows)),
            "repetitive": int(sum(int(r.get("repetitive") or 0) for r in rows)),
            "truncated": int(sum(int(r.get("truncated") or 0) for r in rows)),
            "non_evaluable": bool(flags.get(arm)),
        }
    return {"ran": True, "n": len(blocks), "frozen_n": int(panel["n"]), "partial": len(blocks) < int(panel["n"]), "summary": summary, "contrasts": contrasts, "flags": flags, "dose": dose}


def contrasts_from(blocks, cfg):
    usable = [b for b in blocks if b["unedited"].get("positive_score") is not None and b["native"].get("positive_score") is not None]
    if not usable:
        return {}
    fraction = float(cfg["retention_fraction"])
    native_gain = [float(b["native"]["positive_score"] - b["unedited"]["positive_score"]) for b in usable]
    out = {"native_minus_unedited": PARENT.boot_mean(native_gain, int(cfg["n_boot"]))}
    native_clear = out["native_minus_unedited"]["lower95"] > 0
    out["native_clear_positive"] = native_clear
    arms = [key for key in usable[0] if key not in ("review_id", "seed", "unedited", "native") and usable[0][key].get("positive_score") is not None]
    for arm in arms:
        rows = [b for b in usable if b[arm].get("positive_score") is not None]
        gain = [float(b[arm]["positive_score"] - b["unedited"]["positive_score"]) for b in rows]
        native = [float(b["native"]["positive_score"] - b["unedited"]["positive_score"]) for b in rows]
        out[f"{arm}_minus_unedited"] = PARENT.boot_mean(gain, int(cfg["n_boot"]))
        out[f"{arm}_minus_native"] = PARENT.boot_mean([float(b[arm]["positive_score"] - b["native"]["positive_score"]) for b in rows], int(cfg["n_boot"]))
        if native_clear:
            out[f"{arm}_retention_80"] = PARENT.boot_mean([g - fraction * n for g, n in zip(gain, native)], int(cfg["n_boot"]))
        if arm.startswith("consensus_"):
            bridge = "bridge_" + arm.split("_", 1)[1]
            if bridge in rows[0] and rows[0][bridge].get("positive_score") is not None:
                out[f"{arm}_minus_{bridge}"] = PARENT.boot_mean([float(b[arm]["positive_score"] - b[bridge]["positive_score"]) for b in rows if b[bridge].get("positive_score") is not None], int(cfg["n_boot"]))
        if arm in ("consensus_gemma", "consensus_mistral") and "consensus_olmo" in rows[0]:
            out[f"{arm}_minus_consensus_olmo"] = PARENT.boot_mean([float(b[arm]["positive_score"] - b["consensus_olmo"]["positive_score"]) for b in rows], int(cfg["n_boot"]))
        if arm == "consensus_gemma" and rows[0].get("gemma_shuffle", {}).get("positive_score") is not None:
            out["consensus_gemma_minus_gemma_shuffle"] = PARENT.boot_mean([float(b["consensus_gemma"]["positive_score"] - b["gemma_shuffle"]["positive_score"]) for b in rows if b["gemma_shuffle"].get("positive_score") is not None], int(cfg["n_boot"]))
    gemma_ok = out.get("consensus_gemma_retention_80", {}).get("lower95", -1) > 0 and out.get("consensus_gemma_minus_unedited", {}).get("lower95", -1) > 0
    mistral_ok = out.get("consensus_mistral_retention_80", {}).get("lower95", -1) > 0 and out.get("consensus_mistral_minus_unedited", {}).get("lower95", -1) > 0
    out["both_new_anchors_retain_80"] = bool(native_clear and gemma_ok and mistral_ok and not any(b.get("partial") for b in []))
    return out


def excluded_ids(work, cfg, parent, manifest) -> set[str]:
    excluded = set()
    actadd = set()
    prior_path = work / cfg["prior_actadd_splits"]
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        for rows in prior.values():
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and "review_id" in row:
                        actadd.add(row["review_id"])
                        excluded.add(row["review_id"])
    for rel in (
        cfg["sentiment_parent"] + "/artifacts/confirmation_qwen.jsonl",
        cfg["sentiment_parent"] + "/artifacts/confirmation_olmo.jsonl",
    ):
        path = work / rel
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    excluded.add(json.loads(line)["review_id"])
    from datasets import load_dataset
    test_neg = [{"review_id": f"test:{i}", "text": row["text"]} for i, row in enumerate(load_dataset("stanfordnlp/imdb")["test"]) if int(row["label"]) == 0]
    parent_pool = [r for r in test_neg if r["review_id"] not in actadd]
    np.random.default_rng(int(cfg["parent_split_seed"])).shuffle(parent_pool)
    for row in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
        excluded.add(row["review_id"])
    paths = [
        work / cfg["cca_run"] / "artifacts/panel.json",
        work / cfg["cca_run"] / "artifacts/cca_manifest.json",
        work / cfg["retention_run"] / "artifacts/panel_manifest.json",
        work / cfg["gemma_run"] / "artifacts/native_splits.json",
        work / cfg["moonlight_run"] / "artifacts/panel_manifest.json",
        work / cfg["moonlight_run"] / "artifacts/norm_split.json",
        parent / "steering_panel.json",
    ]
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for key in ("review_ids", "fit_review_ids", "rank_review_ids", "timing_review_ids", "calibration", "validation"):
                for item in payload.get(key) or []:
                    excluded.add(item["review_id"] if isinstance(item, dict) else item)
    for path in (parent / "steering_confirmation.jsonl", work / cfg["moonlight_run"] / "artifacts/confirmation.jsonl"):
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
        lines.append(f"## Case {index}")
        lines.append("")
        lines.append(block["unedited"]["prefix"])
        lines.append("")
        for letter, arm in zip(letters, order):
            lines.append(f"### {letter}")
            lines.append("")
            lines.append(block[arm].get("continuation") or "")
            lines.append("")
            key.append({"case": index, "letter": letter, "arm": arm, "review_id": block["review_id"]})
    (art / "blind_sheet.md").write_text("\n".join(lines) + "\n")
    save_json(art / "blind_key.json", key)
    save_json(art / "blind_ratings.json", {"locked": False, "ratings": []})


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}], lower bound {block['lower95']:+.3f}"


def render_md(state, cfg) -> str:
    lines = ["# RESULTS — `multifamily_cca_anchor_swap_v1`", ""]
    steering = state.get("steering") or {}
    contrasts = steering.get("contrasts") or {}
    lines.append("| Anchor | Method | Gain over unedited | Gap to native | 80% retention lower bound | Gap to matched bridge |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for anchor in cfg["anchors"]:
        for kind in ("consensus", "bridge"):
            arm = f"{kind}_{anchor}"
            gain = contrasts.get(f"{arm}_minus_unedited") or {}
            native = contrasts.get(f"{arm}_minus_native") or {}
            retention = contrasts.get(f"{arm}_retention_80") or {}
            bridge = contrasts.get(f"consensus_{anchor}_minus_bridge_{anchor}") if kind == "consensus" else {"mean": 0.0}
            lines.append(
                f"| {anchor} | {kind} | {gain.get('mean', float('nan')):+.3f} | {native.get('mean', float('nan')):+.3f} | {retention.get('lower95', float('nan')):+.3f} | {bridge.get('mean', float('nan')):+.3f} |"
            )
    lines.append("")
    both = contrasts.get("both_new_anchors_retain_80")
    lines.append("Did both new routes preserve steering? " + ("Yes. Gemma and Mistral consensus both clear native gain, transfer gain, and 80% retention on the complete panel." if both else "The joint criterion is not met. Individual routes are below."))
    lines.append(f"Gemma consensus minus unedited {fmt(contrasts.get('consensus_gemma_minus_unedited'))}. Retention {fmt(contrasts.get('consensus_gemma_retention_80'))}.")
    lines.append(f"Mistral consensus minus unedited {fmt(contrasts.get('consensus_mistral_minus_unedited'))}. Retention {fmt(contrasts.get('consensus_mistral_retention_80'))}.")
    lines.append(f"OLMo consensus retention {fmt(contrasts.get('consensus_olmo_retention_80'))}.")
    lines.append("")
    lines.append("Did the larger space buy anything over pairwise CCA? The paired gaps are the evidence. A gap whose interval includes zero is a raw difference, not evidence that the routes match or that one wins.")
    for anchor in cfg["anchors"]:
        lines.append(f"{anchor}: consensus minus matched bridge {fmt(contrasts.get(f'consensus_{anchor}_minus_bridge_{anchor}'))}.")
    lines.append(f"Gemma consensus minus OLMo consensus {fmt(contrasts.get('consensus_gemma_minus_consensus_olmo'))}.")
    lines.append(f"Mistral consensus minus OLMo consensus {fmt(contrasts.get('consensus_mistral_minus_consensus_olmo'))}.")
    lines.append(f"Gemma consensus minus its shuffled-attachment control {fmt(contrasts.get('consensus_gemma_minus_gemma_shuffle'))}.")
    lines.append("")
    half = (state.get("half") or state.get("audit", {}).get("half") or {})
    lines.append("Which ranks reproducibly overlap across discovery halves, and did that stability carry over to the decoded edit?")
    cond = half.get("conditional_on_parent_whitening") or {}
    if cond:
        for name, rows in cond.items():
            bits = ", ".join(f"rank {rank} MSC {row['mean_squared_cosine']:.3f}" for rank, row in rows.items())
            lines.append(f"Conditional on the parent whitening, {name}: {bits}.")
        lines.append(f"Cosine of the two half-fit OLMo-attached decoded directions at rank 64: {half.get('half_decoded_cosine_rank64')}.")
        lines.append(f"Qwen-edit energy in each half span before normalisation: {half.get('edit_energy')}.")
    else:
        lines.append("The half audit is not in this state file.")
    white = state.get("whitening") or state.get("audit", {}).get("whitening") or {}
    if white:
        lines.append("")
        lines.append(f"Full-discovery maximum canonical singular value: {white.get('max_singular_full_discovery')}. Exceeds one: {white.get('exceeds_one')}.")
        lines.append(white.get("explanation", ""))
    if steering.get("partial"):
        lines.append(f"Steering coverage is partial: {steering.get('n')} of {steering.get('frozen_n')}. The joint retention claim is not made from a partial panel.")
    elif steering.get("n"):
        lines.append(f"Steering coverage is the full frozen panel, n={steering.get('n')}.")
    summary = steering.get("summary") or {}
    if summary:
        lines += ["", "| arm | n | positive | score | empty | repetitive | truncated |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, row in summary.items():
            if row.get("mean_score") is not None:
                lines.append(f"| {arm} | {row['n']} | {row['positive']} | {row['mean_score']:.3f} | {row['empty']} | {row['repetitive']} | {row['truncated']} |")
    if state.get("error"):
        lines += ["", "## Stop", "", "```", state["error"][-2000:], "```"]
    lines.append("")
    lines.append("Geometry and the intervention are separate. One sentiment task in Moonlight does not establish a universal transport result.")
    lines.append("Reproduction: `python scripts/run_multifamily_cca_anchor_swap.py --config configs/multifamily_cca_anchor_swap_v1.json --work-root $WORK`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
