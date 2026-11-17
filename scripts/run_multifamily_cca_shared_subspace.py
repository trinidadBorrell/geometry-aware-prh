#!/usr/bin/env python3
"""Do pairwise CCAs share a subspace, and can that subspace steer a held-out family?"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

from prh_replication.multifamily_shared_subspace import (
    _self_check,
    activation_consensus,
    align_scores,
    apply_linear,
    cca_directions,
    consensus_basis,
    displacement_consensus,
    document_means,
    fit_pca_whitener,
    fixed_axis_correlation,
    maxvar_gcca,
    mean_squared_cosine,
    principal_angles,
    random_orthonormal,
    ridge_maps,
    standardise_fit,
    top1_retrieval,
    whiten_activations,
)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default) + "\n")


def sha256_bytes(values: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def allow_unmarked_safetensors() -> None:
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils, "_unmarked_safetensors_ok", False):
        return
    original = modeling_utils.load_state_dict

    def load_state_dict(checkpoint_file, is_quantized=False):
        if str(checkpoint_file).endswith(".safetensors") and modeling_utils.is_safetensors_available():
            with modeling_utils.safe_open(checkpoint_file, framework="pt") as handle:
                metadata = handle.metadata() or {}
            fmt = metadata.get("format")
            if fmt is not None and fmt not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(f"safetensors archive {checkpoint_file} has metadata {metadata}")
            return modeling_utils.safe_load_file(checkpoint_file)
        return original(checkpoint_file, is_quantized=is_quantized)

    modeling_utils.load_state_dict = load_state_dict
    modeling_utils._unmarked_safetensors_ok = True


def normalise_text(text: str) -> str:
    return " ".join((text or "").split())


def wiki_documents(rows) -> list[dict]:
    docs, buf, index = [], [], 0
    for row in rows:
        line = row["text"]
        stripped = line.strip()
        if stripped.startswith("=") and stripped.endswith("=") and buf:
            text = normalise_text(" ".join(buf))
            if len(text) >= 256:
                docs.append({"doc_id": f"wiki:train:{index}", "domain": "wikitext", "text": text})
                index += 1
            buf = []
        elif stripped:
            buf.append(stripped)
    if buf:
        text = normalise_text(" ".join(buf))
        if len(text) >= 256:
            docs.append({"doc_id": f"wiki:train:{index}", "domain": "wikitext", "text": text})
    return docs


def flatten(acts: np.ndarray) -> np.ndarray:
    return np.asarray(acts, dtype=np.float64).reshape(-1, acts.shape[-1])


def repeat_ids(doc_ids: list[str], n_pos: int) -> list[str]:
    return [doc for doc in doc_ids for _ in range(n_pos)]


def domain_of(doc_id: str) -> str:
    return "wikitext" if doc_id.startswith("wiki:") else "imdb"


def boot_mean(values, n_boot, seed=0):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n_reviews": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = float(v[rng.integers(0, v.size, v.size)].mean())
    return {
        "mean": float(v.mean()),
        "low": float(np.quantile(boots, 0.025)),
        "high": float(np.quantile(boots, 0.975)),
        "lower95": float(np.quantile(boots, 0.05)),
        "n_reviews": int(v.size),
        "n_boot": int(n_boot),
    }


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}], one-sided lower bound {block['lower95']:+.3f}"


def retrieval_pair(query, gallery, seed: int) -> dict:
    true = top1_retrieval(query, gallery)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(gallery))
    # Reassign partner documents. The correct neighbor is no longer in slot i.
    shuffled = top1_retrieval(query, np.asarray(gallery, dtype=np.float64)[perm])
    return {"top1": true, "shuffled_top1": shuffled, "n_docs": int(len(query)), "beats_shuffle": bool(true > shuffled)}


def resume_steering_panel(cfg, work, art, torch, auto_model, auto_tok, pipeline, note):
    panel_path = art / "steering_panel.json"
    ckpt = art / "steering_confirmation.jsonl"
    if not panel_path.exists() or not ckpt.exists() or not (art / "dir_consensus.npy").exists():
        return None
    panel = json.loads(panel_path.read_text())
    done = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
    done_ids = [row["review_id"] for row in done]
    if done_ids != panel["review_ids"][: len(done_ids)] or len(done_ids) >= int(panel["n"]):
        return None
    remaining = panel["review_ids"][len(done_ids):]
    note(f"resume steering {len(done_ids)} done, {len(remaining)} left")
    from datasets import load_dataset

    wanted = set(remaining)
    texts = {}
    for i, row in enumerate(load_dataset("stanfordnlp/imdb")["test"]):
        review_id = f"test:{i}"
        if review_id in wanted:
            texts[review_id] = row["text"]
    moon = json.loads((art / "model_pins.json").read_text())["present"]["moonlight"]
    moon["snapshot"] = cfg["models"]["moonlight"]["snapshot"]
    moon.update({k: cfg["models"]["moonlight"][k] for k in ("trust_remote_code", "attn_implementation", "unmarked_safetensors") if k in cfg["models"]["moonlight"]})
    directions = {
        "unedited": None,
        "native": np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy"),
        "bridge": np.load(work / cfg["moonlight_run"] / "artifacts/dir_transfer.npy"),
        "consensus": np.load(art / "dir_consensus.npy"),
        "maxvar": np.load(art / "dir_maxvar.npy"),
    }
    model, tok, layer = load_model(work, moon, torch, auto_model, auto_tok)
    model.config.use_cache = True
    sample = cfg["sample"]
    max_new = int(sample["max_new_tokens"])
    siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
    scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)
    deadline = time.time() + float(cfg["steering_wall_sec"]) - float(cfg["steering_reserve_sec"])

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
                    nxt = sample_next(fwd.logits[0, -1], ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
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

    blocks = list(done)
    for n_new, review_id in enumerate(remaining, start=1):
        if time.time() >= deadline:
            break
        text = render(texts[review_id])
        seed = review_seed(review_id)
        rec = {"review_id": review_id, "seed": seed}
        for arm, direction in directions.items():
            t_g = time.time()
            new_ids, cont = generate(text, direction, seed)
            p, y = pos_of(cont)
            rec[arm] = {"prefix": text, "continuation": cont, "n_new": len(new_ids), "sec": time.time() - t_g, "positive_score": p, "positive": bool(y), "seed": seed, **quality(cont, len(new_ids), max_new)}
        blocks.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n_new % 10 == 0:
            note(f"steering {len(blocks)}/{panel['n']}")
    summary = {}
    for arm in directions:
        rows = [b[arm] for b in blocks]
        summary[arm] = {
            "n": len(rows),
            "positive": int(sum(int(r["positive"]) for r in rows)),
            "mean_score": float(np.mean([r["positive_score"] for r in rows])) if rows else None,
            "empty": int(sum(int(r["empty"]) for r in rows)),
            "repetitive": int(sum(int(r["repetitive"]) for r in rows)),
            "truncated": int(sum(int(r["truncated"]) for r in rows)),
        }
    contrasts = {}
    if blocks:
        fraction = float(cfg["retention_fraction"])
        native_gain = [float(b["native"]["positive_score"] - b["unedited"]["positive_score"]) for b in blocks]
        contrasts["native_minus_unedited"] = boot_mean(native_gain, int(cfg["n_boot"]))
        native_clear = contrasts["native_minus_unedited"]["lower95"] > 0
        for arm in ("bridge", "consensus", "maxvar"):
            gain = [float(b[arm]["positive_score"] - b["unedited"]["positive_score"]) for b in blocks]
            contrasts[f"{arm}_minus_unedited"] = boot_mean(gain, int(cfg["n_boot"]))
            contrasts[f"{arm}_minus_native"] = boot_mean([float(b[arm]["positive_score"] - b["native"]["positive_score"]) for b in blocks], int(cfg["n_boot"]))
            if arm != "bridge":
                contrasts[f"{arm}_minus_bridge"] = boot_mean([float(b[arm]["positive_score"] - b["bridge"]["positive_score"]) for b in blocks], int(cfg["n_boot"]))
            if native_clear:
                contrasts[f"{arm}_retention_80"] = boot_mean([g - fraction * n for g, n in zip(gain, native_gain)], int(cfg["n_boot"]))
        contrasts["native_clear_positive"] = native_clear
        save_json(art / "steering_contrasts.json", contrasts)
    del model
    torch.cuda.empty_cache()
    return {"ran": True, "n": len(blocks), "frozen_n": int(panel["n"]), "partial": len(blocks) < int(panel["n"]), "summary": summary, "contrasts": contrasts}


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

    _self_check()
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_root)
    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    geo_deadline = t0 + float(cfg["geometry_wall_sec"]) - float(cfg["geometry_reserve_sec"])
    state = {"log": [], "coverage": {}}

    def note(msg):
        state["log"].append({"t": round(time.time() - t0, 1), "msg": msg})
        print(msg, flush=True)

    def late():
        return time.time() >= geo_deadline

    try:
        resumed = resume_steering_panel(cfg, work, art, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, note)
        if resumed is not None:
            pairwise = json.loads((art / "pairwise.json").read_text())
            state["overlap"] = pairwise.get("overlap") or {}
            state["overlap_pass"] = bool(state["overlap"]) and all(row.get("above_twice_random") for row in state["overlap"].values()) and len(state["overlap"]) == 5
            state["gate"] = json.loads((art / "gate.json").read_text())
            state["missing_models"] = []
            state["steering"] = resumed
            raise SystemExit
        present = {}
        missing = []
        fraction = int(cfg["olmo_archived_layer"]) / int(cfg["olmo_archived_layers"])
        for name, spec in cfg["models"].items():
            snap = work / spec["snapshot"]
            if not (snap / "config.json").exists():
                missing.append(name)
                continue
            model_cfg = json.loads((snap / "config.json").read_text())
            n_layers = int(model_cfg["num_hidden_layers"])
            if spec.get("layer") is None:
                layer = int(np.round(fraction * n_layers))
            else:
                layer = int(spec["layer"])
            if not 0 <= layer < n_layers:
                raise RuntimeError(f"{name} layer {layer} is outside 0..{n_layers - 1}")
            present[name] = {**spec, "layer": layer, "n_layers": n_layers, "hidden": int(model_cfg["hidden_size"])}
        state["models"] = {name: {"layer": row["layer"], "n_layers": row["n_layers"], "revision": row["revision"], "hf": row["hf"]} for name, row in present.items()}
        state["missing_models"] = missing
        note(f"models {list(present)} missing {missing}")
        save_json(art / "model_pins.json", {"present": state["models"], "missing": missing, "layer_rule": cfg["layer_rule"]})
        required = ["qwen", "olmo", "gemma", "moonlight"]
        if any(name not in present for name in required):
            raise RuntimeError(f"required snapshot missing: {[name for name in required if name not in present]}")

        excluded = set()
        prior_path = work / cfg["prior_actadd_splits"]
        actadd = set()
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            for rows in prior.values():
                for row in rows:
                    if isinstance(row, dict) and "review_id" in row:
                        actadd.add(row["review_id"])
                        excluded.add(row["review_id"])
        parent_art = work / cfg["sentiment_parent"] / "artifacts"
        for name in ("qwen", "olmo"):
            path = parent_art / f"confirmation_{name}.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        excluded.add(json.loads(line)["review_id"])
        for rel in (
            cfg["cca_run"] + "/artifacts/panel.json",
            cfg["cca_run"] + "/artifacts/cca_manifest.json",
            cfg["retention_run"] + "/artifacts/panel_manifest.json",
            cfg["gemma_run"] + "/artifacts/native_splits.json",
            cfg["moonlight_run"] + "/artifacts/panel_manifest.json",
            cfg["moonlight_run"] + "/artifacts/norm_split.json",
        ):
            path = work / rel
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                for key in ("review_ids", "fit_review_ids", "rank_review_ids", "timing_review_ids", "calibration", "validation"):
                    excluded.update(payload.get(key) or [])

        ds = load_dataset("stanfordnlp/imdb")
        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        test_neg = []
        imdb_pool = []
        for i, row in enumerate(ds["test"]):
            if int(row["label"]) == 0:
                test_neg.append({"review_id": f"test:{i}", "text": row["text"]})
        parent_pool = [r for r in test_neg if r["review_id"] not in actadd]
        np.random.default_rng(int(cfg["parent_split_seed"])).shuffle(parent_pool)
        for row in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
            excluded.add(row["review_id"])
        for i, row in enumerate(ds["train"]):
            text = normalise_text(row["text"])
            if len(text) >= 256:
                imdb_pool.append({"doc_id": f"imdb:train:{i}", "domain": "imdb", "text": text})
        wiki_pool = wiki_documents(wiki["train"])
        note(f"pools imdb {len(imdb_pool)} wiki {len(wiki_pool)} excluded_reviews {len(excluded)}")

        saved_splits = {name: art / f"split_{name}.json" for name in ("discovery", "attachment_fit", "attachment_val", "geometry_test")}
        if all(path.exists() for path in saved_splits.values()):
            manifest = {name: json.loads(path.read_text()) for name, path in saved_splits.items()}
            geometry_ids = {r["doc_id"] for rows in manifest.values() for r in rows}
            note("reuse frozen splits")
        else:
            manifest, geometry_ids = None, None

        if manifest is None:
            allow_unmarked_safetensors()
            moon = present["moonlight"]
            smoke_docs = imdb_pool[:4]
            model, tok, layer = load_model(work, moon, torch, AutoModelForCausalLM, AutoTokenizer)
            t_smoke = time.time()
            _ = extract_acts(model, tok, layer, smoke_docs, cfg["boundaries"], torch, limit_docs=4)
            sec_per_doc = (time.time() - t_smoke) / max(1, len(smoke_docs))
            del model
            torch.cuda.empty_cache()
            n_models = len(present)
            budget = max(60.0, geo_deadline - time.time() - 240.0)
            unit = int(budget / (2 * 7 * n_models * sec_per_doc)) if sec_per_doc > 0 else int(cfg["other_target"])
            unit = max(8, min(int(cfg["other_target"]), unit))
            n_disc = min(int(cfg["discovery_target"]), 4 * unit)
            n_other = min(int(cfg["other_target"]), unit)
            n_disc = min(n_disc, len(imdb_pool) // 4, len(wiki_pool) // 4)
            n_other = min(n_other, (len(imdb_pool) - n_disc) // 3, (len(wiki_pool) - n_disc) // 3)
            if n_disc < 8 or n_other < 8:
                raise RuntimeError(f"not enough documents after sizing disc {n_disc} other {n_other}")
            state["sizing"] = {"sec_per_doc_moonlight": sec_per_doc, "budget_sec": budget, "discovery": n_disc, "other": n_other, "n_models": n_models}
            note(f"smoke {sec_per_doc:.2f}s/doc discovery {n_disc} other {n_other}")

            def draw(pool, seed_offset):
                rows = list(pool)
                np.random.default_rng(int(cfg["split_seed"]) + seed_offset).shuffle(rows)
                return {
                    "discovery": rows[:n_disc],
                    "attachment_fit": rows[n_disc:n_disc + n_other],
                    "attachment_val": rows[n_disc + n_other:n_disc + 2 * n_other],
                    "geometry_test": rows[n_disc + 2 * n_other:n_disc + 3 * n_other],
                }

            imdb_splits = draw(imdb_pool, 0)
            wiki_splits = draw(wiki_pool, 1)
            splits = {}
            for name in ("discovery", "attachment_fit", "attachment_val", "geometry_test"):
                rows = imdb_splits[name] + wiki_splits[name]
                np.random.default_rng(stable_seed(name, int(cfg["split_seed"]) + 100)).shuffle(rows)
                splits[name] = rows
            manifest = {
                name: [{"doc_id": r["doc_id"], "domain": r["domain"], "text": r["text"]} for r in rows]
                for name, rows in splits.items()
            }
            save_json(art / "splits.json", {name: [r["doc_id"] for r in rows] for name, rows in manifest.items()})
            for name, rows in manifest.items():
                save_json(art / f"split_{name}.json", rows)
            geometry_ids = {r["doc_id"] for rows in manifest.values() for r in rows}
            note("splits frozen")

        caches = {}
        for name, spec in present.items():
            cache_path = art / "cache" / f"{name}.npz"
            expected = {split: [r["doc_id"] for r in manifest[split]] for split in manifest}
            if cache_path.exists():
                loaded = np.load(cache_path, allow_pickle=True)
                ok = all(list(loaded[f"ids_{split}"]) == expected[split] for split in manifest)
                if ok:
                    caches[name] = {split: loaded[split] for split in manifest}
                    note(f"reuse cache {name}")
                    continue
            if late():
                note(f"geometry clock stopped before {name}")
                break
            note(f"extract {name} layer {spec['layer']}")
            model, tok, layer = load_model(work, spec, torch, AutoModelForCausalLM, AutoTokenizer)
            arrays, id_arrays = {}, {}
            try:
                for split, rows in manifest.items():
                    if late() and split != "discovery":
                        note(f"clock stopped during {name} {split}")
                        break
                    acts = extract_acts(model, tok, layer, rows, cfg["boundaries"], torch)
                    arrays[split] = acts.astype(np.float32)
                    id_arrays[f"ids_{split}"] = np.asarray([r["doc_id"] for r in rows])
            finally:
                del model
                torch.cuda.empty_cache()
            if set(arrays) != set(manifest):
                state["coverage"][name] = sorted(arrays)
                note(f"incomplete cache {name} {sorted(arrays)}")
                break
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **arrays, **id_arrays)
            caches[name] = arrays
            note(f"cached {name} { {k: list(v.shape) for k, v in arrays.items()} }")

        state["extracted"] = list(caches)
        fingerprints = {}
        for name, arrays in caches.items():
            fingerprints[name] = {split: {"shape": list(values.shape), "sha256": sha256_bytes(values)} for split, values in arrays.items()}
        save_json(art / "activation_fingerprints.json", fingerprints)
        if len(caches) < 4 or "olmo" not in caches or "moonlight" not in caches:
            raise RuntimeError(f"activation coverage {list(caches)} is too small to fit the held-out fold")

        packs = {}
        whitened = {}
        (art / "pca").mkdir(parents=True, exist_ok=True)
        for name, arrays in caches.items():
            disc = flatten(arrays["discovery"])
            packs[name] = fit_pca_whitener(disc, int(cfg["pca_dim"]), float(cfg["eta"]))
            whitened[name] = {}
            for split, acts in arrays.items():
                flat = flatten(acts)
                whitened[name][split] = whiten_activations(flat, packs[name]).reshape(acts.shape[0], acts.shape[1], -1)
            np.savez(
                art / "pca" / f"{name}.npz",
                mean=packs[name]["mean"], components=packs[name]["components"], whitener=packs[name]["whitener"],
            )
            note(f"pca {name} width {packs[name]['width']} var {packs[name]['variance_fraction']:.3f}")
        rank = min(int(cfg["primary_rank"]), min(pack["width"] for pack in packs.values()))
        state["pca_width"] = {name: pack["width"] for name, pack in packs.items()}
        state["rank"] = rank

        names = list(caches)
        edges = list(combinations(sorted(names), 2))
        pairwise = {}
        edge_rows = []
        for left, right in edges:
            fit = cca_directions(flatten(whitened[left]["discovery"]), flatten(whitened[right]["discovery"]), rank)
            pairwise[(left, right)] = {left: fit["U_left"], right: fit["U_right"], "singular": fit["singular"]}
            edge_rows.append(edge_report(left, right, whitened, fit, rank, cfg, [r["doc_id"] for r in manifest["discovery"]]))
        overlap = partner_overlap(names, pairwise, rank, int(cfg["random_frames"]), int(cfg["random_seed"]))
        state["overlap"] = overlap
        state["overlap_pass"] = bool(overlap) and all(row["above_twice_random"] for row in overlap.values()) and len(names) == 5
        save_json(art / "pairwise.json", {"edges": edge_rows, "overlap": overlap, "rank": rank, "models": names})
        note("pairwise overlap " + " ".join(f"{name}:{row['mean_partner_msc']:.3f}/{row['random_msc']:.3f}" for name, row in overlap.items()))

        primary_names = [name for name in ("qwen", "olmo", "gemma", "mistral") if name in caches]
        spaces = {}
        if len(names) == 5:
            spaces["five"] = fit_common_space(names, whitened, pairwise, rank, cfg, art / "space_five")
        if len(primary_names) >= 3 and "olmo" in primary_names:
            # Refit from the four-model edges only. Do not slice a five-model basis.
            primary_edges = {edge: pairwise[edge] for edge in pairwise if set(edge) <= set(primary_names)}
            spaces["primary"] = fit_common_space(primary_names, whitened, primary_edges, rank, cfg, art / "space_primary")
        else:
            raise RuntimeError("primary common space is missing OLMo or two other families")
        state["spaces"] = {key: {"models": value["models"]} for key, value in spaces.items()}

        attach = fit_attachment(spaces["primary"], caches, whitened, manifest, cfg, art)
        state["attachment"] = {method: {k: v for k, v in row.items() if k != "decode"} for method, row in attach.items()}
        geometry = evaluate_geometry(spaces["primary"], attach, caches, whitened, manifest, cfg)
        state["geometry_test"] = geometry
        partners = [name for name in cfg["non_anchor_partners"] if name in primary_names]
        gate = decide_gate(geometry, partners, cfg)
        state["gate"] = gate
        save_json(art / "gate.json", gate)
        save_json(art / "geometry_test.json", geometry)
        note(f"gate advance={gate['advance']} reason={gate['reason']}")

        # Optional extra held-out families only after the primary gate is frozen.
        extras = {}
        for held in ("gemma", "mistral", "qwen"):
            if late() or held not in caches or held not in primary_names:
                continue
            fold_names = [name for name in primary_names if name != held]
            if "olmo" not in fold_names or len(fold_names) < 3:
                continue
            fold_edges = {edge: pairwise[edge] for edge in pairwise if set(edge) <= set(fold_names)}
            space = fit_common_space(fold_names, whitened, fold_edges, rank, cfg, art / f"space_holdout_{held}")
            held_attach = fit_attachment(space, caches, whitened, manifest, cfg, art / f"holdout_{held}", target=held)
            extras[held] = evaluate_geometry(space, held_attach, caches, whitened, manifest, cfg, partners=[n for n in fold_names if n != "olmo"], target=held)
            note(f"extra holdout {held}")
        if extras:
            save_json(art / "extra_holdouts.json", extras)
            state["extra_holdouts"] = list(extras)

        if gate["advance"]:
            note("steering start")
            state["steering"] = run_steering(
                spaces["primary"], attach, present["moonlight"], work, cfg, art, torch,
                AutoModelForCausalLM, AutoTokenizer, pipeline, test_neg, excluded, geometry_ids, note,
            )
        else:
            state["steering"] = {"ran": False, "reason": gate["reason"]}
            note("steering not started")
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


def load_model(work, spec, torch, auto_model, auto_tok):
    snap = str(work / spec["snapshot"])
    kwargs = {"local_files_only": True}
    if spec.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    tok = auto_tok.from_pretrained(snap, **kwargs)
    model_kwargs = dict(kwargs)
    model_kwargs["torch_dtype"] = torch.bfloat16
    if spec.get("attn_implementation"):
        model_kwargs["attn_implementation"] = spec["attn_implementation"]
    if spec.get("unmarked_safetensors"):
        allow_unmarked_safetensors()
    model = auto_model.from_pretrained(snap, **model_kwargs).cuda().eval()
    model.config.use_cache = False
    layer = model.model.layers[int(spec["layer"])]
    return model, tok, layer


def extract_acts(model, tok, layer, rows, boundaries, torch, limit_docs=None):
    acts = []
    use_rows = rows if limit_docs is None else rows[:limit_docs]
    for n, row in enumerate(use_rows, start=1):
        positions = []
        for nchar in boundaries:
            prefix = row["text"][: int(nchar)]
            ids = tok(prefix, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
            if ids.shape[1] < 1:
                raise RuntimeError(f"empty tokenisation for {row['doc_id']}")
            box = {}

            def hook(_m, inputs):
                box["h"] = inputs[0].detach()
                return None

            handle = layer.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    model(ids, attention_mask=torch.ones_like(ids))
            finally:
                handle.remove()
            positions.append(box["h"][0, -1].float().cpu().numpy())
        acts.append(positions)
        if n % 50 == 0:
            print(f"  docs {n}/{len(use_rows)}", flush=True)
    return np.stack(acts)


def edge_report(left, right, whitened, fit, rank, cfg, doc_ids) -> dict:
    diag = {}
    for k in list(cfg["diagnostic_ranks"]) + [rank]:
        width = min(int(k), fit["U_left"].shape[1])
        diag[str(width)] = {
            "mean_singular": float(np.mean(fit["singular"][:width])),
            "last_singular": float(fit["singular"][width - 1]),
            "mean_angle_deg": float(np.degrees(np.mean(principal_angles(fit["U_left"][:, :width], fit["U_right"][:, :width])))),
        }
    n_doc = whitened[left]["discovery"].shape[0]
    rng = np.random.default_rng(stable_seed(f"{left}:{right}", int(cfg["split_half_seed"])))
    order = rng.permutation(n_doc)
    half = n_doc // 2
    first = cca_directions(flatten(whitened[left]["discovery"][order[:half]]), flatten(whitened[right]["discovery"][order[:half]]), rank)
    second = cca_directions(flatten(whitened[left]["discovery"][order[half:]]), flatten(whitened[right]["discovery"][order[half:]]), rank)
    shuf = np.random.default_rng(int(cfg["pairing_shuffle_seed"])).permutation(n_doc)
    shuffled = cca_directions(flatten(whitened[left]["discovery"]), flatten(whitened[right]["discovery"][shuf]), rank)
    domains = {}
    mask_ids = np.asarray(doc_ids)
    for domain in ("imdb", "wikitext"):
        mask = np.asarray([domain_of(doc) == domain for doc in mask_ids])
        if int(mask.sum()) < 8:
            continue
        dom = cca_directions(flatten(whitened[left]["discovery"][mask]), flatten(whitened[right]["discovery"][mask]), rank)
        domains[domain] = {"mean_singular": float(np.mean(dom["singular"]))}
    return {
        "pair": [left, right],
        "diagnostics": diag,
        "split_half_msc": mean_squared_cosine(first["U_left"], second["U_left"]),
        "shuffle_msc": mean_squared_cosine(fit["U_left"], shuffled["U_left"]),
        "shuffle_mean_singular": float(np.mean(shuffled["singular"])),
        "domains": domains,
    }


def stable_seed(text: str, base: int) -> int:
    return int(base) + int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) % 997


def partner_overlap(names, pairwise, rank, n_random, seed) -> dict:
    out = {}
    for name in names:
        bases = []
        for edge, payload in pairwise.items():
            if name in edge:
                bases.append(payload[name][:, :rank])
        if len(bases) < 2:
            continue
        pair_msc = [mean_squared_cosine(a, b) for a, b in combinations(bases, 2)]
        frames = random_orthonormal(bases[0].shape[0], rank, max(len(bases), n_random), stable_seed(name, seed))
        random_msc = [mean_squared_cosine(a, b) for a, b in combinations(frames[: len(bases)], 2)]
        if len(frames) >= n_random:
            random_msc = [mean_squared_cosine(a, b) for a, b in combinations(frames[: max(len(bases), 2)], 2)]
        # A larger random reference: pairwise among the first n_random frames, subsampled to the same count of pairs.
        many = random_orthonormal(bases[0].shape[0], rank, n_random, stable_seed(name, seed + 17))
        many_msc = [mean_squared_cosine(a, b) for a, b in combinations(many, 2)]
        consensus = consensus_basis(bases, rank)
        mean_partner = float(np.mean(pair_msc))
        mean_random = float(np.mean(many_msc))
        out[name] = {
            "n_partners": len(bases),
            "mean_partner_msc": mean_partner,
            "random_msc": mean_random,
            "random_pair_msc": float(np.mean(random_msc)) if random_msc else None,
            "above_twice_random": bool(mean_partner >= 2.0 * mean_random),
            "leading_supports": consensus["supports"][:8],
            "leading_eigenvalues": [float(v) for v in consensus["values"][:16]],
        }
    return out


def fit_common_space(model_names, whitened, pairwise, rank, cfg, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    consensus = {}
    standardised = {}
    for name in model_names:
        partners = [payload[name][:, :rank] for edge, payload in pairwise.items() if name in payload and set(edge) <= set(model_names)]
        basis = consensus_basis(partners, rank)
        flat = flatten(whitened[name]["discovery"])
        view_mean = flat.mean(axis=0)
        scores = (flat - view_mean) @ basis["basis"]
        z, score_mean, score_scale = standardise_fit(scores)
        consensus[name] = {
            "basis": basis["basis"], "view_mean": view_mean, "score_mean": score_mean, "score_scale": score_scale,
            "supports": basis["supports"][:8],
        }
        standardised[name] = z
    rotations = align_scores(standardised, cfg["anchor"], n_iter=int(cfg["procrustes_iters"]))
    for name in model_names:
        consensus[name]["rotation"] = rotations[name]
        pred = activation_consensus(
            flatten(whitened[name]["discovery"]), consensus[name]["basis"], consensus[name]["score_mean"],
            consensus[name]["score_scale"], rotations[name], consensus[name]["view_mean"],
        )
        err = float(np.max(np.abs(pred - standardised[name] @ rotations[name])))
        if err > 1e-5:
            raise RuntimeError(f"consensus reconstruction failed for {name}: {err}")
    views = {name: flatten(whitened[name]["discovery"]) for name in model_names}
    gcca = maxvar_gcca(views, rank, float(cfg["eta"]))
    np.savez(
        outdir / "consensus.npz",
        **{f"basis_{name}": consensus[name]["basis"] for name in model_names},
        **{f"rotation_{name}": consensus[name]["rotation"] for name in model_names},
        **{f"scale_{name}": consensus[name]["score_scale"] for name in model_names},
    )
    np.savez(outdir / "gcca.npz", **{f"loading_{name}": gcca["loadings"][name] for name in model_names}, eigenvalues=gcca["eigenvalues"])
    save_json(outdir / "summary.json", {"models": model_names, "rank": rank, "anchor": cfg["anchor"]})
    return {"models": model_names, "consensus": consensus, "gcca": gcca}


def encode_consensus(space, name, whitened_acts):
    row = space["consensus"][name]
    flat = flatten(whitened_acts)
    encoded = activation_consensus(flat, row["basis"], row["score_mean"], row["score_scale"], row["rotation"], row["view_mean"])
    return encoded.reshape(whitened_acts.shape[0], whitened_acts.shape[1], -1)


def encode_gcca(space, name, whitened_acts):
    gcca = space["gcca"]
    flat = flatten(whitened_acts)
    raw = (flat - gcca["view_mean"][name]) @ gcca["loadings"][name]
    encoded = (raw - gcca["score_mean"]) / gcca["score_scale"]
    return encoded.reshape(whitened_acts.shape[0], whitened_acts.shape[1], -1)


def fit_attachment(space, caches, whitened, manifest, cfg, art: Path, target: str = "moonlight") -> dict:
    art.mkdir(parents=True, exist_ok=True)
    out = {}
    doc_ids = {}
    for split in ("attachment_fit", "attachment_val"):
        doc_ids[split] = repeat_ids([r["doc_id"] for r in manifest[split]], caches[target][split].shape[1])
    x = {split: flatten(caches[target][split]) for split in doc_ids}
    for method, encode in (("consensus", encode_consensus), ("maxvar", encode_gcca)):
        y = {split: flatten(encode(space, "olmo", whitened["olmo"][split])) for split in doc_ids}
        forward = ridge_maps(x["attachment_fit"], y["attachment_fit"], x["attachment_val"], y["attachment_val"], doc_ids["attachment_fit"], doc_ids["attachment_val"], cfg["alpha_multipliers"])
        backward = ridge_maps(y["attachment_fit"], x["attachment_fit"], y["attachment_val"], x["attachment_val"], doc_ids["attachment_fit"], doc_ids["attachment_val"], cfg["alpha_multipliers"])
        np.savez(art / f"{method}_{target}.npz", forward=forward["decode"], forward_intercept=forward["intercept"], backward=backward["decode"], backward_intercept=backward["intercept"])
        out[method] = {
            "forward": forward, "backward": backward,
            "val_mse_coords": forward["val_mse"], "baseline_coords": forward["baseline_mse"],
            "val_mse_residual": backward["val_mse"], "baseline_residual": backward["baseline_mse"],
            "alpha_coords": forward["alpha"], "alpha_residual": backward["alpha"],
        }
    return out


def evaluate_geometry(space, attach, caches, whitened, manifest, cfg, partners=None, target: str = "moonlight") -> dict:
    if partners is None:
        partners = [name for name in cfg["non_anchor_partners"] if name in space["models"]]
    if target not in caches:
        return {}
    # Attachment maps are keyed for the target used in fit_attachment. The primary target is moonlight.
    report = {}
    ids = [r["doc_id"] for r in manifest["geometry_test"]]
    domains = {"pooled": np.ones(len(ids), dtype=bool), "imdb": np.asarray([domain_of(i) == "imdb" for i in ids]), "wikitext": np.asarray([domain_of(i) == "wikitext" for i in ids])}
    for method, encode in (("consensus", encode_consensus), ("maxvar", encode_gcca)):
        report[method] = {}
        moon = apply_linear(attach[method]["forward"]["decode"], attach[method]["forward"]["intercept"], flatten(caches[target]["geometry_test"]), displacement=False)
        moon = moon.reshape(len(ids), -1, moon.shape[-1])
        for domain, mask in domains.items():
            report[method][domain] = {}
            for partner in partners:
                if partner not in whitened:
                    continue
                other = flatten(encode(space, partner, whitened[partner]["geometry_test"])).reshape(len(ids), -1, moon.shape[-1])
                take = np.flatnonzero(mask)
                if take.size < 4:
                    continue
                left = moon[take].reshape(-1, moon.shape[-1])
                right = other[take].reshape(-1, other.shape[-1])
                q_ids = [ids[i] for i in take]
                q_mean, _ = document_means(left, repeat_ids(q_ids, moon.shape[1]))
                g_mean, _ = document_means(right, repeat_ids(q_ids, other.shape[1]))
                retrieved = retrieval_pair(q_mean, g_mean, int(cfg["retrieval_shuffle_seed"]))
                report[method][domain][partner] = {"axis_correlation": fixed_axis_correlation(left, right), **retrieved}
    return report


def decide_gate(geometry, partners, cfg) -> dict:
    expected = list(cfg["non_anchor_partners"])
    if partners != expected:
        return {"advance": False, "reason": f"partners {partners} do not match the prespecified three", "partners": partners, "methods": {}}
    methods = {}
    advance = False
    for method, domains in geometry.items():
        pooled = domains.get("pooled") or {}
        beats = {name: bool((pooled.get(name) or {}).get("beats_shuffle")) for name in expected}
        methods[method] = beats
        if all(beats.values()):
            advance = True
    return {"advance": advance, "reason": "a shared space beat shuffled retrieval for all three partners" if advance else "no shared space beat shuffled retrieval for every non-anchor partner", "partners": partners, "methods": methods, "rule": cfg["gate_rule"]}


def sample_next(logits, seen, *, temperature, top_p, freq_penalty):
    import torch
    final = logits / temperature
    if freq_penalty > 0:
        counts = torch.bincount(seen, minlength=final.shape[-1]).to(final.dtype)
        final = final - freq_penalty * counts
    order = torch.sort(final, descending=True)
    cumulative = order.values.softmax(-1).cumsum(-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    drop = torch.zeros_like(remove).scatter(0, order.indices, remove)
    final = final.masked_fill(drop, float("-inf")).float()
    return torch.distributions.Categorical(logits=final).sample()


def review_seed(review_id: str) -> int:
    return int(hashlib.sha256(review_id.encode()).hexdigest()[:8], 16)


def quality(text, n_new, max_new):
    import re
    from collections import Counter
    toks = re.findall(r"[A-Za-z']+", text or "")
    empty = not (text or "").strip()
    if toks:
        counts = Counter(w.lower() for w in toks)
        repetitive = counts.most_common(1)[0][1] / len(toks) > 0.5 or len(counts) / len(toks) < 0.3
    else:
        repetitive = False
    return {"empty": empty, "repetitive": repetitive, "truncated": n_new >= max_new}


def run_steering(space, attach, moon_spec, work, cfg, art, torch, auto_model, auto_tok, pipeline, test_neg, excluded, geometry_ids, note):
    t0 = time.time()
    deadline = t0 + float(cfg["steering_wall_sec"]) - float(cfg["steering_reserve_sec"])
    dose = float(json.loads((work / cfg["moonlight_run"] / "artifacts/dose.json").read_text())["dose"])
    native = np.load(work / cfg["moonlight_run"] / "artifacts/dir_native.npy")
    bridge = np.load(work / cfg["moonlight_run"] / "artifacts/dir_transfer.npy")
    v_q = np.load(work / cfg["sentiment_parent"] / "artifacts/vec_qwen_averaged.npy").astype(np.float64)
    with np.load(art / "pca" / "qwen.npz") as zpack:
        qwen_pack = {key: np.array(zpack[key]) for key in zpack.files}
    transfers = {}
    for method in ("consensus", "maxvar"):
        w = whiten_activations(v_q, qwen_pack, displacement=True)
        if method == "consensus":
            row = space["consensus"]["qwen"]
            z = displacement_consensus(w, row["basis"], row["score_scale"], row["rotation"])
        else:
            gcca = space["gcca"]
            z = (w @ gcca["loadings"]["qwen"]) / gcca["score_scale"]
        raw = apply_linear(attach[method]["backward"]["decode"], attach[method]["backward"]["intercept"], z, displacement=True)
        norm = float(np.linalg.norm(raw))
        near = norm < float(cfg["near_zero_norm"])
        transfers[method] = np.zeros_like(raw) if near else raw / norm * dose
        np.save(art / f"dir_{method}.npy", transfers[method])
        note(f"{method} raw {norm:.5f} near_zero {near}")
    for name, vec in (("native", native), ("bridge", bridge)):
        if abs(float(np.linalg.norm(vec)) - dose) / dose > 0.02:
            raise RuntimeError(f"archived {name} norm does not match the Moonlight dose")
    eligible = [r for r in test_neg if r["review_id"] not in excluded and r["review_id"] not in geometry_ids]
    np.random.default_rng(int(cfg["panel_seed"])).shuffle(eligible)
    model, tok, layer = load_model(work, moon_spec, torch, auto_model, auto_tok)
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
                    nxt = sample_next(fwd.logits[0, -1], ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
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

    directions = {"unedited": None, "native": native, "bridge": bridge, "consensus": transfers["consensus"], "maxvar": transfers["maxvar"]}
    timers = eligible[: int(cfg["smoke_blocks"])]
    t_batch = time.time()
    for row in timers:
        text = render(row["text"])
        seed = review_seed(row["review_id"])
        for direction in directions.values():
            generate(text, direction, seed)
    sec_block = (time.time() - t_batch) / max(1, len(timers))
    remain = max(0.0, deadline - time.time())
    n_panel = min(int(cfg["target_panel"]), int(remain * 0.92 / sec_block)) if sec_block > 0 else 0
    chosen = eligible[len(timers):len(timers) + max(n_panel, 0)]
    frozen = {"n": len(chosen), "sec_per_block": sec_block, "review_ids": [r["review_id"] for r in chosen], "timing_review_ids": [r["review_id"] for r in timers], "frozen_before_test_scores": True}
    save_json(art / "steering_panel.json", frozen)
    note(f"steering panel {frozen['n']} sec/block {sec_block:.2f}")
    blocks = []
    ckpt = art / "steering_confirmation.jsonl"
    for row in chosen:
        if time.time() >= deadline:
            break
        text = render(row["text"])
        seed = review_seed(row["review_id"])
        rec = {"review_id": row["review_id"], "seed": seed}
        for arm, direction in directions.items():
            t_g = time.time()
            new_ids, cont = generate(text, direction, seed)
            p, y = pos_of(cont)
            rec[arm] = {"prefix": text, "continuation": cont, "n_new": len(new_ids), "sec": time.time() - t_g, "positive_score": p, "positive": bool(y), "seed": seed, **quality(cont, len(new_ids), max_new)}
        blocks.append(rec)
        with ckpt.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    summary = {}
    for arm in directions:
        rows = [b[arm] for b in blocks]
        summary[arm] = {
            "n": len(rows),
            "positive": int(sum(int(r["positive"]) for r in rows)),
            "mean_score": float(np.mean([r["positive_score"] for r in rows])) if rows else None,
            "empty": int(sum(int(r["empty"]) for r in rows)),
            "repetitive": int(sum(int(r["repetitive"]) for r in rows)),
            "truncated": int(sum(int(r["truncated"]) for r in rows)),
        }
    contrasts = {}
    if blocks:
        fraction = float(cfg["retention_fraction"])
        native_gain = [float(b["native"]["positive_score"] - b["unedited"]["positive_score"]) for b in blocks]
        contrasts["native_minus_unedited"] = boot_mean(native_gain, int(cfg["n_boot"]))
        native_clear = contrasts["native_minus_unedited"]["lower95"] > 0
        for arm in ("bridge", "consensus", "maxvar"):
            gain = [float(b[arm]["positive_score"] - b["unedited"]["positive_score"]) for b in blocks]
            gap_native = [float(b[arm]["positive_score"] - b["native"]["positive_score"]) for b in blocks]
            contrasts[f"{arm}_minus_unedited"] = boot_mean(gain, int(cfg["n_boot"]))
            contrasts[f"{arm}_minus_native"] = boot_mean(gap_native, int(cfg["n_boot"]))
            if arm != "bridge":
                gap_bridge = [float(b[arm]["positive_score"] - b["bridge"]["positive_score"]) for b in blocks]
                contrasts[f"{arm}_minus_bridge"] = boot_mean(gap_bridge, int(cfg["n_boot"]))
            if native_clear:
                contrasts[f"{arm}_retention_80"] = boot_mean([g - fraction * n for g, n in zip(gain, native_gain)], int(cfg["n_boot"]))
        contrasts["native_clear_positive"] = native_clear
        save_json(art / "steering_contrasts.json", contrasts)
        blind, key = [], []
        brng = np.random.default_rng(0)
        for b in blocks[: int(cfg["blind_n"])]:
            order = list(directions)
            brng.shuffle(order)
            for arm in order:
                code = hashlib.sha256(f"{b['review_id']}:{arm}".encode()).hexdigest()[:8]
                blind.append({"code": code, "prefix": b[arm]["prefix"], "continuation": b[arm]["continuation"]})
                key.append({"code": code, "arm": arm, "review_id": b["review_id"]})
        save_json(art / "steering_blind.json", blind)
        save_json(art / "steering_blind_key.json", key)
    del model
    torch.cuda.empty_cache()
    return {"ran": True, "n": len(blocks), "frozen_n": frozen["n"], "partial": len(blocks) < frozen["n"], "summary": summary, "contrasts": contrasts, "dose": dose}


def render_md(state, cfg) -> str:
    lines = ["# RESULTS — `multifamily_cca_shared_subspace_v1`", ""]
    missing = state.get("missing_models") or []
    if missing:
        lines.append(f"Coverage is reduced. These checkpoints were not available and were not replaced: {', '.join(missing)}.")
        lines.append("")
    overlap = state.get("overlap") or {}
    if overlap:
        passed = [name for name, row in overlap.items() if row.get("above_twice_random")]
        if state.get("overlap_pass"):
            overlap_answer = "Yes. Every model in the five-model fit clears the twice-random rule."
        elif missing or len(overlap) < 5:
            overlap_answer = "The five-family claim is not met. A prespecified family is missing, so the lines below cover only the models that were fit."
        else:
            overlap_answer = "No. The prespecified twice-random rule does not hold for every included model."
        lines.append("Did different pairwise fits recover overlapping directions? " + overlap_answer)
        for name, row in overlap.items():
            lines.append(
                f"{name}: partner mean-squared-cosine {row['mean_partner_msc']:.3f}, random {row['random_msc']:.3f}, twice-random {'yes' if row['above_twice_random'] else 'no'}."
            )
        lines.append(cfg["overlap_rule"])
    else:
        lines.append("Did different pairwise fits recover overlapping directions? Not estimated.")
    lines.append("")
    gate = state.get("gate") or {}
    if gate:
        lines.append("Did a frozen common space predict alignment with an excluded family beyond its attachment anchor? " + ("Yes." if gate.get("advance") else "No."))
        lines.append(gate.get("reason", ""))
        geo = state.get("geometry_test") or {}
        for method, domains in geo.items():
            pooled = domains.get("pooled") or {}
            for partner, row in pooled.items():
                lines.append(
                    f"{method} vs {partner}: axis correlation {row['axis_correlation']:.3f}, retrieval {row['top1']:.3f}, shuffled {row['shuffled_top1']:.3f}."
                )
    else:
        lines.append("Did a frozen common space predict alignment with an excluded family beyond its attachment anchor? Not estimated.")
    lines.append("")
    steering = state.get("steering") or {}
    if not steering.get("ran"):
        lines.append("Did that space carry the existing sentiment edit? Steering was not run. " + str(steering.get("reason", "")))
    else:
        c = steering.get("contrasts") or {}
        lines.append(f"Steering coverage {steering.get('n')} of {steering.get('frozen_n')} frozen reviews.")
        lines.append("Did the shared spaces carry the existing sentiment edit, and did they keep up with native and the two-model bridge?")
        for arm in ("bridge", "consensus", "maxvar"):
            lines.append(f"{arm} minus unedited {fmt(c.get(arm + '_minus_unedited'))}. Minus native {fmt(c.get(arm + '_minus_native'))}." + ("" if arm == "bridge" else f" Minus bridge {fmt(c.get(arm + '_minus_bridge'))}."))
            if c.get(arm + "_retention_80"):
                lines.append(f"{arm} 80% retention {fmt(c.get(arm + '_retention_80'))}.")
        if c.get("native_clear_positive") is False:
            lines.append("Native gain is not clearly positive, so percentage retention stays unresolved.")
        lines.append(f"Native minus unedited {fmt(c.get('native_minus_unedited'))}.")
        summary = steering.get("summary") or {}
        if summary:
            lines.append("")
            lines.append("| arm | n | positive | score | empty | repetitive | truncated |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for arm, row in summary.items():
                if row.get("n"):
                    lines.append(f"| {arm} | {row['n']} | {row['positive']} | {row['mean_score']:.3f} | {row['empty']} | {row['repetitive']} | {row['truncated']} |")
    lines.append("")
    if state.get("error"):
        lines += ["## Stop", "", "```", state["error"][-1800:], "```", ""]
    lines.append(f"Elapsed {state.get('elapsed_sec', 0):.0f}s. Reproduction: `python scripts/run_multifamily_cca_shared_subspace.py --config configs/multifamily_cca_shared_subspace_v1.json --work-root $WORK`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
