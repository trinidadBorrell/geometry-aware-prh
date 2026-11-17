#!/usr/bin/env python3
"""CCA transport of the archived Qwen sentiment edit into OLMo."""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default))


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v, n


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def words(text):
    return re.findall(r"[A-Za-z']+", text or "")


def quality(text, n_new, max_new):
    toks = words(text)
    empty = not (text or "").strip()
    if toks:
        counts = Counter(w.lower() for w in toks)
        repetitive = counts.most_common(1)[0][1] / len(toks) > 0.5 or len(counts) / len(toks) < 0.3
    else:
        repetitive = False
    return {"empty": empty, "repetitive": repetitive, "truncated": n_new >= max_new}


def summarize(rows):
    if not rows:
        return {"n": 0}
    def mean(k):
        return float(np.mean([float(r[k]) for r in rows]))
    return {
        "n": len(rows),
        "positive": int(sum(int(r["positive"]) for r in rows)),
        "positive_rate": mean("positive"),
        "mean_score": mean("positive_score"),
        "empty": int(sum(int(r["empty"]) for r in rows)),
        "repetitive": int(sum(int(r["repetitive"]) for r in rows)),
        "truncated": int(sum(int(r["truncated"]) for r in rows)),
    }


def paired_interval(base, other, key):
    by = {r["review_id"]: r for r in other}
    diffs, up, down = [], 0, 0
    for r in base:
        o = by.get(r["review_id"])
        if o is None:
            continue
        diffs.append(float(o[key]) - float(r[key]))
        if key == "positive":
            if (not r["positive"]) and o["positive"]:
                up += 1
            if r["positive"] and (not o["positive"]):
                down += 1
    v = np.asarray(diffs, dtype=np.float64)
    if v.size == 0:
        return {"n_reviews": 0}
    rng = np.random.default_rng(0)
    boots = [float(v[rng.integers(0, v.size, v.size)].mean()) for _ in range(1000)]
    out = {"mean": float(v.mean()), "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975)), "n_reviews": int(v.size)}
    if key == "positive":
        out["flips_to_positive"] = up
        out["flips_to_negative"] = down
    return out


def review_seed(review_id: str) -> int:
    return int(hashlib.sha256(review_id.encode()).hexdigest()[:8], 16)


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


def shared_passage(text: str, prompt_chars: int, cont_chars: int):
    body = " ".join(text.split())
    if len(body) < prompt_chars + cont_chars:
        return None
    head = body[:prompt_chars].rstrip()
    prompt = head + " Overall,"
    tail = body[prompt_chars:prompt_chars + cont_chars]
    full = prompt + " " + tail
    offs = [len(prompt), len(prompt) + 1 + 40, len(prompt) + 1 + 80, len(full)]
    if offs[-1] > len(full) or min(np.diff([0] + offs)) <= 0:
        return None
    return full, offs


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
    from prh_replication.benchmark_matched_cca_maps import fit_pca_cca_maps
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_root)
    t0 = time.time()
    report_at = t0 + float(cfg["wall_clock_sec"]) - float(cfg["report_reserve_sec"])
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    state = {"log": []}

    def note(msg):
        state["log"].append({"t": round(time.time() - t0, 1), "msg": msg})
        print(msg, flush=True)

    def late():
        return time.time() >= report_at

    try:
        os.environ.setdefault("HF_HOME", str(work / "hf"))
        parent = work / cfg["parent"]
        parent_art = parent / "artifacts"
        v_q = np.load(parent_art / "vec_qwen_averaged.npy")
        v_o = np.load(parent_art / "vec_olmo_averaged.npy")
        parent_state = json.loads((parent_art / "run_state.json").read_text())
        dose = {
            "qwen": float(parent_state["results"]["qwen"]["dose"]),
            "olmo": float(parent_state["results"]["olmo"]["dose"]),
        }
        parent_check = {"vector_norms": {"qwen": float(np.linalg.norm(v_q)), "olmo": float(np.linalg.norm(v_o))}, "doses": dose, "from_jsonl": {}}
        for name in ("qwen", "olmo"):
            rows = [json.loads(line) for line in (parent_art / f"confirmation_{name}.jsonl").open()]
            parent_check["from_jsonl"][name] = {
                "n": len(rows),
                "unedited": summarize([r["unedited"] for r in rows]),
                "averaged": summarize([r["averaged"] for r in rows]),
                "reversed": summarize([r["reversed"] for r in rows]),
            }
        state["parent_check"] = parent_check
        save_json(art / "parent_check.json", parent_check)
        note(f"parent vectors unit norms {parent_check['vector_norms']} doses {dose}")

        excluded = set()
        prior = json.loads((work / cfg["prior_actadd_splits"]).read_text())
        for rows in prior.values():
            for row in rows:
                if isinstance(row, dict) and "review_id" in row:
                    excluded.add(row["review_id"])
        for name in ("qwen", "olmo"):
            for line in (parent_art / f"confirmation_{name}.jsonl").open():
                excluded.add(json.loads(line)["review_id"])

        ds = load_dataset("stanfordnlp/imdb")
        test_neg, train = [], []
        actadd_excluded_before_parent = set(excluded)
        for split, bucket in (("test", test_neg), ("train", train)):
            for i, row in enumerate(ds[split]):
                rid = f"{split}:{i}"
                bucket.append({"review_id": rid, "label": int(row["label"]), "text": row["text"]})
        parent_pool = [r for r in test_neg if r["label"] == 0 and r["review_id"] not in {x for x in actadd_excluded_before_parent if x.startswith("test:")}]
        # The parent excluded actadd ids, then shuffled negatives. Reconstruct that pool.
        actadd_ids = {row["review_id"] for rows in prior.values() for row in rows if isinstance(row, dict) and "review_id" in row}
        parent_pool = [r for r in test_neg if r["label"] == 0 and r["review_id"] not in actadd_ids]
        np.random.default_rng(int(cfg["parent_split_seed"])).shuffle(parent_pool)
        for r in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
            excluded.add(r["review_id"])
        note(f"excluded {len(excluded)} previously exposed review ids")

        siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
        scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)

        def as_scores(raw):
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                return raw[0]
            return raw

        def pos_of(text):
            table = {d["label"].upper(): float(d["score"]) for d in as_scores(scorer(text[:1500] if text.strip() else " "))}
            p = table.get("POSITIVE", 0.0)
            return p, p >= table.get("NEGATIVE", 0.0)

        kp, ky = pos_of("This movie was wonderful and I loved every minute of it.")
        kn, kny = pos_of("This movie was awful and I hated every minute of it.")
        if not (ky and not kny):
            raise SystemExit("SiEBERT check failed")

        def snap(spec):
            return work / "hf/hub" / ("models--" + spec["hf"].replace("/", "--")) / "snapshots" / spec["revision"]

        def load_model(spec):
            src = str(snap(spec))
            tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(src, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
            return model, tok, model.model.layers[int(spec["layer"])]

        def render_eval(tok, text):
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
            return tok.decode(ids, skip_special_tokens=True).rstrip() + cfg["eval_bridge"]

        def token_for_offset(offsets, pos):
            for i, (a, b) in enumerate(offsets):
                if int(a) < int(pos) <= int(b):
                    return i
            if offsets and int(pos) >= int(offsets[-1][1]):
                return len(offsets) - 1
            return None

        specs = {"qwen": cfg["qwen"], "olmo": cfg["olmo"]}
        vectors = {"qwen": np.asarray(v_q, dtype=np.float64), "olmo": np.asarray(v_o, dtype=np.float64)}
        loaded = {}
        for name, spec in specs.items():
            note(f"load {name}")
            loaded[name] = load_model(spec)

        # CCA corpus
        usable = []
        for row in train:
            if row["review_id"] in excluded:
                continue
            made = shared_passage(row["text"], int(cfg["prompt_chars"]), int(cfg["continuation_chars"]))
            if made is None:
                continue
            usable.append({**row, "full": made[0], "offs": made[1]})
        pos = [r for r in usable if r["label"] == 1]
        neg = [r for r in usable if r["label"] == 0]
        rng = np.random.default_rng(int(cfg["cca_seed"]))
        rng.shuffle(pos)
        rng.shuffle(neg)
        half_f, half_r = int(cfg["fit_n"]) // 2, int(cfg["rank_n"]) // 2
        fit_rows = pos[:half_f] + neg[:half_f]
        rank_rows = pos[half_f:half_f + half_r] + neg[half_f:half_f + half_r]
        rng.shuffle(fit_rows)
        rng.shuffle(rank_rows)
        note(f"CCA reviews fit {len(fit_rows)} rank {len(rank_rows)}")

        def extract_map(name, rows):
            model, tok, block = loaded[name]
            got = {}
            for row in rows:
                if late():
                    break
                enc = tok(row["full"], return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
                ids = enc["input_ids"].to("cuda")
                offs = enc["offset_mapping"][0].tolist()
                idxs = [token_for_offset(offs, c) for c in row["offs"]]
                if any(i is None for i in idxs):
                    continue
                box = {}

                def hook(_m, inputs):
                    box["h"] = inputs[0].detach()
                    return None

                handle = block.register_forward_pre_hook(hook)
                try:
                    with torch.no_grad():
                        model(ids)
                finally:
                    handle.remove()
                h = box["h"][0].float().cpu().numpy()
                got[row["review_id"]] = np.stack([h[i] for i in idxs])
            return got

        def stack_pair(qmap, omap, rows):
            ids = [r["review_id"] for r in rows if r["review_id"] in qmap and r["review_id"] in omap]
            if not ids:
                raise RuntimeError("no paired activation rows")
            Xq = np.concatenate([qmap[i] for i in ids], axis=0)
            Xo = np.concatenate([omap[i] for i in ids], axis=0)
            fam = [i for i in ids for _ in range(qmap[ids[0]].shape[0])]
            return Xq, Xo, ids, fam

        note("extract fit")
        q_fit = extract_map("qwen", fit_rows)
        o_fit = extract_map("olmo", fit_rows)
        Xq_fit, Xo_fit, fit_ids, fit_fam = stack_pair(q_fit, o_fit, fit_rows)
        note("extract rank")
        q_rank = extract_map("qwen", rank_rows)
        o_rank = extract_map("olmo", rank_rows)
        Xq_rank, Xo_rank, rank_ids, rank_fam = stack_pair(q_rank, o_rank, rank_rows)
        n_fit_reviews = len(fit_ids)
        n_rank_reviews = len(rank_ids)
        note(f"rows fit {len(Xq_fit)} reviews {n_fit_reviews}; rank rows {len(Xq_rank)} reviews {n_rank_reviews}")
        save_json(art / "cca_manifest.json", {
            "fit_review_ids": fit_ids,
            "rank_review_ids": rank_ids,
            "n_fit_reviews": n_fit_reviews,
            "n_rank_reviews": n_rank_reviews,
            "boundaries": "prompt end, plus 40, 80, and end of a 120-character teacher-forced continuation",
            "text": "identical literal string for both models",
        })

        maps = fit_pca_cca_maps(
            Xq_fit, Xo_fit, Xq_rank, Xo_rank,
            pca_dim=int(cfg["pca_dim"]), n_pairs=int(cfg["n_pairs"]), eta=float(cfg["eta"]),
            min_pca_dim=int(cfg["min_pca_dim"]),
            family_ids_fit=fit_fam, family_ids_rank=rank_fam,
        )
        Eq = np.asarray(maps["E_A"], dtype=np.float64)
        Eo = np.asarray(maps["E_B"], dtype=np.float64)
        Uq = np.asarray(maps["U_A"], dtype=np.float64)
        Uo = np.asarray(maps["U_B"], dtype=np.float64)
        np.savez(art / "maps.npz", E_qwen=Eq, E_olmo=Eo, U_qwen=Uq, U_olmo=Uo, idx=maps["idx"], rho=maps["rho_rank_selected"])
        save_json(art / "map_meta.json", {**{k: maps[k] for k in ("pca_dim_requested", "pca_dim_used", "pca_dim_reduced", "n_pairs", "n_activation_rows_fit", "n_activation_rows_rank", "n_families_fit", "n_families_rank", "eta", "sha", "rho_rank_selected")}, "U_qwen_width": int(Uq.shape[1]), "U_olmo_width": int(Uo.shape[1]), "E_rows": int(Eq.shape[0])})
        note(f"CCA pca {maps['pca_dim_used']} pairs {maps['n_pairs']}")
        rank_k = int(min(64, Eq.shape[0], maps["pca_dim_used"]))
        rank_small = int(min(16, rank_k))

        def project(v, E):
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            return np.linalg.pinv(E) @ (E @ v)

        def pca_project(v, U):
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            return U @ (U.T @ v)

        def prepare(raw, native_norm, native_vec):
            raw = np.asarray(raw, dtype=np.float64).reshape(-1)
            nrm = float(np.linalg.norm(raw))
            retained = float(np.dot(raw, raw) / (np.dot(native_vec, native_vec) + 1e-30))
            cos = cosine(raw, native_vec) if nrm > 0 else 0.0
            info = {"pre_norm": nrm, "retained_sq": retained, "cosine": cos}
            if nrm < float(cfg["near_zero_norm"]):
                info.update({"near_zero": True, "vector": np.zeros_like(raw), "applied_norm": 0.0})
                return info
            vec, _ = unit(raw)
            info.update({"near_zero": False, "vector": vec * float(native_norm), "applied_norm": float(native_norm)})
            return info

        directions = {}
        geom = {}
        for name, E, U, native in (
            ("qwen", Eq, Uq, vectors["qwen"]),
            ("olmo", Eo, Uo, vectors["olmo"]),
        ):
            geom[name] = {
                "pca": prepare(pca_project(native, U), dose[name], native),
                "rt16": prepare(project(native, E[:rank_small]), dose[name], native),
                "rt64": prepare(project(native, E[:rank_k]), dose[name], native),
            }
            directions[name] = {
                "native": native * dose[name],
                "reversed": -native * dose[name],
                "pca": geom[name]["pca"]["vector"],
                "rt16": geom[name]["rt16"]["vector"],
                "rt64": geom[name]["rt64"]["vector"],
            }
        z64 = Eq[:rank_k] @ vectors["qwen"]
        aligned64 = prepare(np.linalg.pinv(Eo[:rank_k]) @ z64, dose["olmo"], vectors["olmo"])
        aligned16 = prepare(np.linalg.pinv(Eo[:rank_small]) @ (Eq[:rank_small] @ vectors["qwen"]), dose["olmo"], vectors["olmo"])
        rngp = np.random.default_rng(int(cfg["perm_seed"]))
        perms = []
        while len(perms) < 3:
            p = rngp.permutation(rank_k)
            if not np.array_equal(p, np.arange(rank_k)):
                perms.append(p.tolist())
        perm_dirs = []
        for i, p in enumerate(perms):
            raw = np.linalg.pinv(Eo[:rank_k]) @ z64[np.asarray(p)]
            info = prepare(raw, dose["olmo"], vectors["olmo"])
            perm_dirs.append(info)
            directions["olmo"][f"perm{i}"] = info["vector"]
        directions["olmo"]["aligned16"] = aligned16["vector"]
        directions["olmo"]["aligned64"] = aligned64["vector"]
        geom["transfer"] = {"aligned16": {k: v for k, v in aligned16.items() if k != "vector"}, "aligned64": {k: v for k, v in aligned64.items() if k != "vector"}, "perms": [{k: v for k, v in info.items() if k != "vector"} for info in perm_dirs], "perm_indices": perms}
        geom_save = {}
        for name, block in geom.items():
            if name == "transfer":
                geom_save[name] = block
            else:
                geom_save[name] = {k: {kk: vv for kk, vv in info.items() if kk != "vector"} for k, info in block.items()}
        save_json(art / "geometry.json", geom_save)
        for name, dirs in directions.items():
            for arm, vec in dirs.items():
                np.save(art / f"dir_{name}_{arm}.npy", np.asarray(vec, dtype=np.float64))

        # boundary smoke on one shared string
        demo = next(r for r in fit_rows if r["review_id"] == fit_ids[0])
        boundary = {}
        for name in ("qwen", "olmo"):
            tok = loaded[name][1]
            enc = tok(demo["full"], return_offsets_mapping=True, add_special_tokens=False)
            offs = enc["offset_mapping"]
            ids = enc["input_ids"]
            picks = []
            for c in demo["offs"]:
                i = token_for_offset(offs, c)
                piece = tok.decode([ids[i]])
                picks.append({"char": c, "token_index": i, "piece": piece, "span": list(offs[i])})
            boundary[name] = picks
        save_json(art / "boundary_smoke.json", boundary)

        z = np.random.default_rng(0).normal(size=rank_k)
        recon = float(np.linalg.norm(Eq[:rank_k] @ (np.linalg.pinv(Eq[:rank_k]) @ z) - z) / np.linalg.norm(z))
        note(f"coord recon {recon:.3e}")

        sample = cfg["sample"]
        max_new = int(sample["max_new_tokens"])

        def generate(name, text, direction, seed):
            model, tok, block = loaded[name]
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            ids = enc["input_ids"]
            realised = {}
            handle = None
            if direction is not None:
                delta = torch.tensor(np.asarray(direction), device="cuda", dtype=torch.float32)

                def hook(_m, inputs, _delta=delta):
                    resid = inputs[0]
                    before = resid[:, -1].detach()
                    out_h = resid.clone()
                    out_h[:, -1] = out_h[:, -1] + _delta.to(out_h.dtype)
                    key = "prefill" if resid.shape[1] > 1 else "decode"
                    if key not in realised:
                        realised[key] = float(torch.linalg.vector_norm((out_h[:, -1] - before).float()).cpu())
                    return (out_h,) + tuple(inputs[1:])

                handle = block.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    fwd = model(ids, use_cache=True)
                    past = fwd.past_key_values
                    logits = fwd.logits[0, -1]
                    new_ids = []
                    for _ in range(max_new):
                        nxt = sample_next(logits, ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
                        new_ids.append(int(nxt.item()))
                        if tok.eos_token_id is not None and int(nxt.item()) == int(tok.eos_token_id):
                            break
                        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                        fwd = model(nxt.view(1, 1), past_key_values=past, use_cache=True)
                        past = fwd.past_key_values
                        logits = fwd.logits[0, -1]
            finally:
                if handle is not None:
                    handle.remove()
            return new_ids, tok.decode(new_ids, skip_special_tokens=True), realised

        # hook-off and realised-norm smoke, not on the eval panel
        smoke = {"coord_recon_rel": recon, "boundary_same_char_ends": all(a["char"] == b["char"] for a, b in zip(boundary["qwen"], boundary["olmo"]))}
        fresh = [r for r in test_neg if r["label"] == 0 and r["review_id"] not in excluded]
        np.random.default_rng(int(cfg["panel_seed"])).shuffle(fresh)
        timing = fresh[0]
        panel_pool = fresh[1:]
        for name in ("qwen", "olmo"):
            tok = loaded[name][1]
            text = render_eval(tok, timing["text"])
            model, _, block = loaded[name]
            with torch.no_grad():
                base = model(**tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")).logits
            zeros = np.zeros_like(vectors[name])
            # zero add
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

            def zero_hook(_m, inputs):
                return (inputs[0].clone(),) + tuple(inputs[1:])

            h = block.register_forward_pre_hook(zero_hook)
            try:
                with torch.no_grad():
                    zero_logits = model(**enc).logits
            finally:
                h.remove()
            smoke[name] = {"zero_logit_max_abs": float(torch.max(torch.abs(base.float() - zero_logits.float())).cpu())}
            _, _, realised = generate(name, text, directions[name]["native"], 0)
            smoke[name]["native_realised"] = realised
            smoke[name]["target_norm"] = dose[name]
        save_json(art / "smoke.json", smoke)
        note(f"smoke {smoke}")
        for name in ("qwen", "olmo"):
            if smoke[name]["zero_logit_max_abs"] > 1e-3:
                raise RuntimeError(f"{name} hook-off logits differ")
            for key in ("prefill", "decode"):
                got = float(smoke[name]["native_realised"].get(key, 0.0))
                if abs(got - dose[name]) / dose[name] > 0.02:
                    raise RuntimeError(f"{name} realised {key} norm {got} != {dose[name]}")
        if not smoke["boundary_same_char_ends"] or recon > 1e-4:
            raise RuntimeError(f"boundary or coordinate reconstruction failed: {recon}")

        # size panel from one paired block of the cheapest timing: time qwen 6 + olmo 11 on timing text, discard scores
        arms = {
            "qwen": ["unedited", "native", "reversed", "pca", "rt16", "rt64"],
            "olmo": ["unedited", "native", "reversed", "pca", "rt16", "rt64", "aligned16", "aligned64", "perm0", "perm1", "perm2"],
        }
        t1 = time.time()
        for name in ("qwen", "olmo"):
            text = render_eval(loaded[name][1], timing["text"])
            for arm in arms[name]:
                direction = None if arm == "unedited" else directions[name][arm]
                generate(name, text, direction, 0)
        sec_block = time.time() - t1
        remain = max(0.0, report_at - time.time())
        n_panel = int(cfg["target_panel"])
        if sec_block > 0:
            n_panel = min(n_panel, max(4, int(remain * 0.9 / sec_block)))
        panel = panel_pool[:n_panel]
        save_json(art / "panel.json", {"n": len(panel), "sec_per_block": sec_block, "review_ids": [r["review_id"] for r in panel]})
        note(f"panel {len(panel)} sec/block {sec_block:.1f}")

        ckpt = art / "confirmation.jsonl"
        blocks = []
        for row in panel:
            if late():
                break
            rec = {"review_id": row["review_id"]}
            complete = True
            for name in ("qwen", "olmo"):
                text = render_eval(loaded[name][1], row["text"])
                seed = review_seed(row["review_id"])
                for arm in arms[name]:
                    if late():
                        complete = False
                        break
                    direction = None if arm == "unedited" else directions[name][arm]
                    t_g = time.time()
                    new_ids, cont, realised = generate(name, text, direction, seed)
                    p, y = pos_of(cont)
                    rec[f"{name}:{arm}"] = {
                        "review_id": row["review_id"],
                        "prefix": text,
                        "continuation": cont,
                        "n_new": len(new_ids),
                        "sec": time.time() - t_g,
                        "positive_score": p,
                        "positive": bool(y),
                        "realised_norm": realised,
                        **quality(cont, len(new_ids), max_new),
                    }
                if not complete:
                    break
            if complete and all(f"{name}:{arm}" in rec for name in arms for arm in arms[name]):
                blocks.append(rec)
                with ckpt.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                note(f"block {len(blocks)} {row['review_id']}")
        state["n_blocks"] = len(blocks)

        def col(name, arm):
            return [b[f"{name}:{arm}"] for b in blocks]

        summary = {name: {arm: summarize(col(name, arm)) for arm in arms[name]} for name in arms}
        contrasts = {}
        for name in ("qwen", "olmo"):
            native, uned, pca = col(name, "native"), col(name, "unedited"), col(name, "pca")
            contrasts[name] = {
                "native_minus_unedited_score": paired_interval(uned, native, "positive_score"),
                "native_minus_unedited_rate": paired_interval(uned, native, "positive"),
                "pca_minus_native_score": paired_interval(native, pca, "positive_score"),
            }
            for rk in ("rt16", "rt64"):
                contrasts[name][f"{rk}_minus_pca_score"] = paired_interval(pca, col(name, rk), "positive_score")
                contrasts[name][f"{rk}_minus_native_score"] = paired_interval(native, col(name, rk), "positive_score")
        if blocks:
            uned, native = col("olmo", "unedited"), col("olmo", "native")
            for arm in ("aligned16", "aligned64"):
                contrasts["olmo"][f"{arm}_minus_unedited_score"] = paired_interval(uned, col("olmo", arm), "positive_score")
                contrasts["olmo"][f"{arm}_minus_unedited_rate"] = paired_interval(uned, col("olmo", arm), "positive")
                contrasts["olmo"][f"{arm}_minus_native_score"] = paired_interval(native, col("olmo", arm), "positive_score")
            contrasts["olmo"]["aligned64_minus_rt64_score"] = paired_interval(col("olmo", "rt64"), col("olmo", "aligned64"), "positive_score")
            contrasts["olmo"]["aligned16_minus_rt16_score"] = paired_interval(col("olmo", "rt16"), col("olmo", "aligned16"), "positive_score")
            for i in range(3):
                contrasts["olmo"][f"aligned64_minus_perm{i}_score"] = paired_interval(col("olmo", f"perm{i}"), col("olmo", "aligned64"), "positive_score")
            # retention contrast if native interval is clearly positive
            nat = contrasts["olmo"]["native_minus_unedited_score"]
            rate = contrasts["olmo"]["native_minus_unedited_rate"]
            # "Clearly positive" requires both intervals to clear a small margin.
            # A bound that only just exceeds zero is treated as a weak reference.
            if nat.get("low", -1) > 0.02 and rate.get("low", -1) > 0:
                diffs = []
                by_t = {r["review_id"]: r for r in col("olmo", "aligned64")}
                by_n = {r["review_id"]: r for r in native}
                by_u = {r["review_id"]: r for r in uned}
                for r in uned:
                    diffs.append((by_t[r["review_id"]]["positive_score"] - r["positive_score"]) - float(cfg["retention_fraction"]) * (by_n[r["review_id"]]["positive_score"] - r["positive_score"]))
                v = np.asarray(diffs, dtype=np.float64)
                rng_b = np.random.default_rng(0)
                boots = [float(v[rng_b.integers(0, v.size, v.size)].mean()) for _ in range(1000)]
                contrasts["olmo"]["transfer_minus_0.8_native"] = {"mean": float(v.mean()), "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975)), "n_reviews": int(v.size)}
            else:
                contrasts["olmo"]["transfer_minus_0.8_native"] = {"skipped": "native paired interval is not strictly positive"}
        state["summary"] = summary
        state["contrasts"] = contrasts
        state["geometry"] = geom_save
        state["map"] = {k: maps[k] for k in ("pca_dim_used", "n_pairs", "n_families_fit", "n_families_rank", "n_activation_rows_fit", "n_activation_rows_rank")}
        # blind sample
        blind, key = [], []
        brng = np.random.default_rng(0)
        for b in blocks[:8]:
            picks = [("qwen", "unedited"), ("qwen", "native"), ("olmo", "unedited"), ("olmo", "native"), ("olmo", "aligned64")]
            brng.shuffle(picks)
            for name, arm in picks:
                code = hashlib.sha256(f"{b['review_id']}:{name}:{arm}".encode()).hexdigest()[:8]
                cell = b[f"{name}:{arm}"]
                blind.append({"code": code, "prefix": cell["prefix"], "continuation": cell["continuation"]})
                key.append({"code": code, "model": name, "arm": arm, "review_id": b["review_id"]})
        save_json(art / "blind_sample.json", blind)
        save_json(art / "blind_key.json", key)
        loaded.clear()
        torch.cuda.empty_cache()
    except Exception:
        state["error"] = traceback.format_exc()
        note(state["error"])
    state["elapsed_sec"] = time.time() - t0
    save_json(art / "run_state.json", state)
    (dest / "RESULTS.md").write_text(render_md(state))
    print("wrote RESULTS", flush=True)
    return 0


def _iv(block, key):
    d = (block or {}).get(key) or {}
    if "mean" not in d:
        return "not estimated"
    return f"{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] (n={d.get('n_reviews')})"


def render_md(state):
    s = state.get("summary") or {}
    c = state.get("contrasts") or {}
    lines = ["# RESULTS — `imdb_matched_pair_cca_transfer_v1`", ""]
    if not s:
        lines.append("Did the native effects hold on fresh reviews? Incomplete.")
        lines.append("Did PCA or top-16 remove them? Not tested.")
        lines.append("Did wider CCA preserve them? Not tested.")
        lines.append("Did aligned transfer shift OLMo sentiment above unedited and the permutations, and how close did it come to native? Not tested.")
        if state.get("error"):
            lines += ["", "```", state["error"][-2000:], "```"]
        lines.append(f"\nElapsed {state.get('elapsed_sec', 0):.0f}s.")
        return "\n".join(lines) + "\n"

    def cell(model, arm):
        return (s.get(model) or {}).get(arm) or {}

    def brief(model, arm):
        r = cell(model, arm)
        if not r.get("n"):
            return "n/a"
        return f"{r['positive']}/{r['n']} positive, score {r['mean_score']:.3f}"

    cq, co = c.get("qwen") or {}, c.get("olmo") or {}
    lines.append(
        "Did the native effects hold on fresh reviews? "
        f"Qwen native {brief('qwen', 'native')} versus unedited {brief('qwen', 'unedited')}; "
        f"paired score change {_iv(cq, 'native_minus_unedited_score')}. "
        f"OLMo native {brief('olmo', 'native')} versus unedited {brief('olmo', 'unedited')}; "
        f"paired score change {_iv(co, 'native_minus_unedited_score')}."
    )
    lines.append(
        "Did PCA or top-16 remove them? "
        f"PCA minus native, Qwen {_iv(cq, 'pca_minus_native_score')}, OLMo {_iv(co, 'pca_minus_native_score')}. "
        f"Round-trip 16 minus PCA, Qwen {_iv(cq, 'rt16_minus_pca_score')}, OLMo {_iv(co, 'rt16_minus_pca_score')}. "
        f"Scores: Qwen PCA {brief('qwen', 'pca')}, RT16 {brief('qwen', 'rt16')}; "
        f"OLMo PCA {brief('olmo', 'pca')}, RT16 {brief('olmo', 'rt16')}."
    )
    lines.append(
        "Did wider CCA preserve them? "
        f"Round-trip 64 minus PCA, Qwen {_iv(cq, 'rt64_minus_pca_score')}, OLMo {_iv(co, 'rt64_minus_pca_score')}. "
        f"Qwen RT64 {brief('qwen', 'rt64')}. OLMo RT64 {brief('olmo', 'rt64')}."
    )
    lines.append(
        "Did aligned transfer shift OLMo sentiment above unedited and the permutations, and how close did it come to native? "
        f"Aligned-64 minus unedited {_iv(co, 'aligned64_minus_unedited_score')}. "
        f"Aligned-64 minus perm0/1/2 {_iv(co, 'aligned64_minus_perm0_score')}; {_iv(co, 'aligned64_minus_perm1_score')}; {_iv(co, 'aligned64_minus_perm2_score')}. "
        f"Aligned-64 minus OLMo native {_iv(co, 'aligned64_minus_native_score')}. "
        f"Aligned-64 minus OLMo RT64 {_iv(co, 'aligned64_minus_rt64_score')}. "
        f"Arms: aligned-64 {brief('olmo', 'aligned64')}, aligned-16 {brief('olmo', 'aligned16')}, "
        f"native {brief('olmo', 'native')}, unedited {brief('olmo', 'unedited')}."
    )
    ret = co.get("transfer_minus_0.8_native") or {}
    if "mean" in ret:
        lines.append(
            f"The fresh OLMo native gain is clearly positive, so the prespecified retention contrast "
            f"Δaligned64 − 0.8×Δnative is {_iv(co, 'transfer_minus_0.8_native')}."
        )
    elif ret.get("skipped"):
        lines.append(
            "The fresh OLMo native reference is not clearly positive, so percentage retention is not interpreted. "
            "The aligned effect and the permutation comparisons above are the transfer result."
        )
    lines += ["", "Usability is provisional until blinded ratings exist. Empty, repetitive, and truncated counts are separate from sentiment. Word overlap is not a relevance measure.", ""]
    lines.append("| model | arm | n | positive | score | empty | repetitive | truncated |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for model, arms in s.items():
        for arm, r in arms.items():
            if not r.get("n"):
                continue
            lines.append(f"| {model} | {arm} | {r['n']} | {r['positive']} | {r['mean_score']:.3f} | {r['empty']} | {r['repetitive']} | {r['truncated']} |")
    lines += ["", "Parent confirmation files were re-counted from the saved generations. The archived unit vectors were reloaded, and every active arm uses that model's archived edit norm (Qwen 15.47, OLMo 1.45).", ""]
    pc = state.get("parent_check") or {}
    lines.append("Parent recount: `" + json.dumps(pc.get("from_jsonl"), default=str) + "`")
    lines.append("")
    lines.append("Map: `" + json.dumps(state.get("map"), default=str) + "`")
    lines.append("")
    lines.append("Geometry (pre-normalisation norm, retained squared norm, cosine to the native vector) is in `artifacts/geometry.json`.")
    if state.get("error"):
        lines += ["", "## Stop", "", "```", state["error"][-1500:], "```"]
    lines.append("")
    lines.append(f"Elapsed {state.get('elapsed_sec', 0):.0f}s. n_blocks={state.get('n_blocks')}. Reproduction: `python scripts/run_imdb_matched_pair_cca_transfer.py --config configs/imdb_matched_pair_cca_transfer_v1.json --work-root $WORK`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
