#!/usr/bin/env python3
"""Averaged matched-pair sentiment vectors versus three fixed single pairs."""

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


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or "")


def quality(text: str, n_new: int, max_new: int) -> dict:
    toks = words(text)
    empty = not (text or "").strip()
    if toks:
        counts = Counter(w.lower() for w in toks)
        top = counts.most_common(1)[0][1] / len(toks)
        uniq = len(counts) / len(toks)
        repetitive = top > 0.5 or uniq < 0.3
    else:
        repetitive = False
    return {"empty": empty, "repetitive": repetitive, "truncated": n_new >= max_new}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    def mean(k):
        return float(np.mean([float(r[k]) for r in rows]))
    return {
        "n": n,
        "positive": int(sum(int(r["positive"]) for r in rows)),
        "positive_rate": mean("positive"),
        "mean_score": mean("positive_score"),
        "empty": int(sum(int(r["empty"]) for r in rows)),
        "repetitive": int(sum(int(r["repetitive"]) for r in rows)),
        "truncated": int(sum(int(r["truncated"]) for r in rows)),
    }


def paired_interval(base: list[dict], other: list[dict], key: str) -> dict:
    by = {r["review_id"]: r for r in other}
    diffs = []
    up = down = 0
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
    rng = np.random.default_rng(0)
    if v.size == 0:
        return {"n_reviews": 0}
    boots = [float(v[rng.integers(0, v.size, v.size)].mean()) for _ in range(1000)]
    out = {
        "mean": float(v.mean()),
        "low": float(np.quantile(boots, 0.025)),
        "high": float(np.quantile(boots, 0.975)),
        "n_reviews": int(v.size),
    }
    if key == "positive":
        out["flips_to_positive"] = up
        out["flips_to_negative"] = down
    return out


def sample_next(logits, seen, *, temperature, top_p, freq_penalty):
    import torch
    final = logits / temperature
    if freq_penalty > 0:
        counts = torch.bincount(seen, minlength=final.shape[-1]).to(final.dtype)
        final = final - freq_penalty * counts
    sorted_logits, sorted_indices = torch.sort(final, descending=True)
    cumulative = sorted_logits.softmax(-1).cumsum(-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    drop = torch.zeros_like(remove).scatter(0, sorted_indices, remove)
    final = final.masked_fill(drop, float("-inf")).float()
    return torch.distributions.Categorical(logits=final).sample()


def review_seed(review_id: str) -> int:
    return int(hashlib.sha256(review_id.encode()).hexdigest()[:8], 16)


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
    repo = work / "repo"
    t0 = time.time()
    report_at = t0 + float(cfg["wall_clock_sec"]) - float(cfg["report_reserve_sec"])
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    state: dict = {"log": []}

    def note(msg: str) -> None:
        state["log"].append({"t": round(time.time() - t0, 1), "msg": msg})
        print(msg, flush=True)

    def late() -> bool:
        return time.time() >= report_at

    try:
        os.environ.setdefault("HF_HOME", str(work / "hf"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        payload = json.loads((repo / cfg["pairs_file"]).read_text())
        pairs = payload["pairs"]
        note(f"loaded {len(pairs)} frozen pairs")
        rng = np.random.default_rng(int(cfg["baseline_seed"]))
        families = sorted({p["family_id"] for p in pairs})
        chosen_families = list(rng.choice(families, size=3, replace=False))
        baselines = []
        for fam in chosen_families:
            members = [p for p in pairs if p["family_id"] == fam]
            pick = members[int(rng.integers(0, len(members)))]
            baselines.append(pick["pair_id"])
        state["baselines"] = {"families": chosen_families, "pair_ids": baselines, "seed": int(cfg["baseline_seed"])}
        save_json(art / "baselines.json", state["baselines"])
        note(f"baselines fixed {baselines}")

        excluded = set()
        prior = work / cfg["prior_splits"]
        if prior.exists():
            prev = json.loads(prior.read_text())
            for rows in prev.values():
                for row in rows:
                    excluded.add(row["review_id"])
        note(f"excluding {len(excluded)} prior review ids")

        siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
        scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)

        def as_scores(out_s):
            if isinstance(out_s, list) and out_s and isinstance(out_s[0], list):
                return out_s[0]
            return out_s

        def pos_of(text: str) -> tuple[float, bool]:
            table = {d["label"].upper(): float(d["score"]) for d in as_scores(scorer(text[:1500] if text.strip() else " "))}
            p = table.get("POSITIVE", 0.0)
            return p, p >= table.get("NEGATIVE", 0.0)

        known_p, known_y = pos_of("This movie was wonderful and I loved every minute of it.")
        known_n, known_ny = pos_of("This movie was awful and I hated every minute of it.")
        if not (known_y and not known_ny):
            raise SystemExit("SiEBERT known-text check failed")
        state["scorer_smoke"] = {"positive_score": known_p, "negative_score": known_n}

        ds = load_dataset("stanfordnlp/imdb")
        test = []
        for i, row in enumerate(ds["test"]):
            if int(row["label"]) != 0:
                continue
            rid = f"test:{i}"
            if rid in excluded:
                continue
            test.append({"review_id": rid, "text": row["text"]})
        np.random.default_rng(int(cfg["split_seed"])).shuffle(test)
        state["n_test_pool"] = len(test)

        def snap(spec):
            folder = "models--" + spec["hf"].replace("/", "--")
            return work / "hf/hub" / folder / "snapshots" / spec["revision"]

        max_new = int(cfg["sample"]["max_new_tokens"])
        sample = cfg["sample"]

        def load_model(spec):
            src = str(snap(spec))
            tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(src, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return model, tok, model.model.layers[int(spec["layer"])]

        def encode(tok, text):
            return tok(text, return_tensors="pt", add_special_tokens=False)

        def last_id(tok, text) -> int:
            return int(encode(tok, text)["input_ids"][0, -1])

        def render_eval(tok, review: str) -> str:
            ids = tok(review, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
            body = tok.decode(ids, skip_special_tokens=True).rstrip()
            return body + cfg["eval_bridge"]

        def capture(model, tok, block, text):
            enc = encode(tok, text).to("cuda")
            box = {}

            def hook(_m, inputs):
                box["h"] = inputs[0].detach()
                return None

            handle = block.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    out = model(**enc)
            finally:
                handle.remove()
            return box["h"], out.logits.detach(), enc

        def generate(model, tok, block, text, direction, alpha, seed):
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            enc = encode(tok, text).to("cuda")
            ids = enc["input_ids"]
            handle = None
            realised = {}
            if direction is not None:
                delta = torch.tensor(direction, device="cuda", dtype=torch.float32) * float(alpha)

                def hook(_m, inputs, _delta=delta):
                    resid = inputs[0]
                    before = resid[:, -1].detach()
                    out_h = resid.clone()
                    out_h[:, -1] = out_h[:, -1] + _delta.to(out_h.dtype)
                    if "prefill" not in realised and resid.shape[1] > 1:
                        realised["prefill"] = float(torch.linalg.vector_norm((out_h[:, -1] - before).float()).cpu())
                    elif "decode" not in realised and resid.shape[1] == 1:
                        realised["decode"] = float(torch.linalg.vector_norm((out_h[:, -1] - before).float()).cpu())
                    return (out_h,) + tuple(inputs[1:])

                handle = block.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    out = model(ids, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[0, -1]
                    new_ids = []
                    for _ in range(max_new):
                        nxt = sample_next(logits, ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
                        new_ids.append(int(nxt.item()))
                        if tok.eos_token_id is not None and int(nxt.item()) == int(tok.eos_token_id):
                            break
                        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                        out = model(nxt.view(1, 1), past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        logits = out.logits[0, -1]
            finally:
                if handle is not None:
                    handle.remove()
            return new_ids, tok.decode(new_ids, skip_special_tokens=True), realised

        def score_panel(model, tok, block, rows, direction, alpha, arm):
            got = []
            for row in rows:
                if late():
                    break
                text = render_eval(tok, row["text"])
                t1 = time.time()
                new_ids, cont, realised = generate(model, tok, block, text, direction, alpha, review_seed(row["review_id"]))
                p, y = pos_of(cont)
                got.append({
                    "review_id": row["review_id"],
                    "arm": arm,
                    "prefix": text,
                    "continuation": cont,
                    "n_new": len(new_ids),
                    "sec": time.time() - t1,
                    "positive_score": p,
                    "positive": bool(y),
                    "realised_norm": realised,
                    **quality(cont, len(new_ids), max_new),
                })
            return got

        results = {}
        for name, spec in (("qwen", cfg["qwen"]), ("olmo", cfg["olmo"])):
            if late():
                results[name] = {"lead": "time cap before this model"}
                break
            note(f"load {name}")
            model, tok, block = load_model(spec)
            token_rows = []
            ok = True
            for pair in pairs:
                pos_s = pair["positive"] + pair["bridge"]
                neg_s = pair["negative"] + pair["bridge"]
                ip, inn = last_id(tok, pos_s), last_id(tok, neg_s)
                bridge_last = last_id(tok, pair["bridge"])
                token_rows.append({"pair_id": pair["pair_id"], "pos_last": ip, "neg_last": inn, "bridge_last": bridge_last, "match": ip == inn})
                if ip != inn:
                    ok = False
            save_json(art / f"token_boundaries_{name}.json", {"all_pairs_match": ok, "rows": token_rows})
            if not ok:
                results[name] = {"lead": "within-pair final token ids disagreed; stopped before activations"}
                del model
                torch.cuda.empty_cache()
                continue
            # toy subtraction
            toy_pos = "The red door stayed shut. I was glad the lock held." + " Overall,"
            toy_neg = "The red door stayed shut. I was angry the lock held." + " Overall,"
            hp, _, _ = capture(model, tok, block, toy_pos)
            hn, _, _ = capture(model, tok, block, toy_neg)
            toy_d = (hp[0, -1] - hn[0, -1]).float()
            # zero edit logits
            prompt0 = render_eval(tok, test[0]["text"])
            with torch.no_grad():
                base_logits = model(**encode(tok, prompt0).to("cuda")).logits.detach()

            def zero_hook(_m, inputs):
                return (inputs[0].clone(),) + tuple(inputs[1:])

            hz = block.register_forward_pre_hook(zero_hook)
            try:
                with torch.no_grad():
                    zero_logits = model(**encode(tok, prompt0).to("cuda")).logits.detach()
            finally:
                hz.remove()
            # manual prefill and one decode step
            direction = unit(toy_d.cpu().numpy())
            alpha = 3.0
            delta = torch.tensor(direction, device="cuda") * alpha
            clean, _, enc0 = capture(model, tok, block, prompt0)
            manual = clean.clone()
            manual[:, -1] = manual[:, -1] + delta.to(manual.dtype)
            got = {}

            def edit_hook(_m, inputs):
                resid = inputs[0]
                out_h = resid.clone()
                out_h[:, -1] = out_h[:, -1] + delta.to(out_h.dtype)
                got["h"] = out_h.detach()
                got["only_last"] = float(torch.linalg.vector_norm((out_h[:, :-1] - resid[:, :-1]).float()).cpu()) if resid.shape[1] > 1 else 0.0
                return (out_h,) + tuple(inputs[1:])

            he = block.register_forward_pre_hook(edit_hook)
            try:
                with torch.no_grad():
                    out = model(**enc0)
                    past = out.past_key_values
                    nxt = torch.tensor([[int(tok.encode(" The", add_special_tokens=False)[0])]], device="cuda")
                    clean_decode = None

                    def cap_decode(_m, inputs):
                        nonlocal clean_decode
                        clean_decode = inputs[0].detach().clone()
                        return None

                    # remove edit hook for a clean decode capture, then reapply
            finally:
                he.remove()
            prefill_err = float(torch.linalg.vector_norm(got["h"].float() - manual.float()).cpu())
            # decode step with and without the add
            with torch.no_grad():
                out = model(**enc0, use_cache=True)
                past = out.past_key_values
                step_in = torch.tensor([[int(encode(tok, " The")["input_ids"][0, -1])]], device="cuda")
                box = {}

                def cap(_m, inputs):
                    box["clean"] = inputs[0].detach().clone()
                    return None

                hc = block.register_forward_pre_hook(cap)
                try:
                    model(step_in, past_key_values=past, use_cache=True)
                finally:
                    hc.remove()
                # fresh past again
                out = model(**enc0, use_cache=True)
                past = out.past_key_values

                def add_decode(_m, inputs):
                    resid = inputs[0]
                    out_h = resid.clone()
                    out_h[:, -1] = out_h[:, -1] + delta.to(out_h.dtype)
                    box["edited"] = out_h.detach()
                    return (out_h,) + tuple(inputs[1:])

                hd = block.register_forward_pre_hook(add_decode)
                try:
                    model(step_in, past_key_values=past, use_cache=True)
                finally:
                    hd.remove()
            decode_manual = box["clean"].clone()
            decode_manual[:, -1] = decode_manual[:, -1] + delta.to(decode_manual.dtype)
            decode_err = float(torch.linalg.vector_norm(box["edited"].float() - decode_manual.float()).cpu())
            plus = float(torch.linalg.vector_norm(delta).cpu())
            minus = float(torch.linalg.vector_norm(-delta).cpu())
            smoke = {
                "toy_last_ids_match": last_id(tok, toy_pos) == last_id(tok, toy_neg),
                "toy_diff_norm": float(torch.linalg.vector_norm(toy_d).cpu()),
                "zero_logit_max_abs": float(torch.max(torch.abs(base_logits.float() - zero_logits.float())).cpu()),
                "prefill_manual_err": prefill_err,
                "prefill_untouched_earlier_positions": got.get("only_last"),
                "decode_manual_err": decode_err,
                "opposite_sign_norms": [plus, minus],
                "opposite_signs_equal": abs(plus - minus) < 1e-5,
            }
            save_json(art / f"smoke_{name}.json", smoke)
            note(f"{name} smoke {smoke}")
            if smoke["zero_logit_max_abs"] > 1e-3 or prefill_err > 1e-2 or decode_err > 1e-2 or not smoke["toy_last_ids_match"]:
                results[name] = {"lead": "smoke failed", "smoke": smoke}
                del model
                torch.cuda.empty_cache()
                continue

            # vectors
            diffs = []
            raw_norms = []
            for pair in pairs:
                if late():
                    break
                hp, _, _ = capture(model, tok, block, pair["positive"] + pair["bridge"])
                hn, _, _ = capture(model, tok, block, pair["negative"] + pair["bridge"])
                d = (hp[0, -1] - hn[0, -1]).float().cpu().numpy()
                diffs.append(d)
                raw_norms.append(float(np.linalg.norm(d)))
            if len(diffs) < len(pairs):
                results[name] = {"lead": "extraction incomplete at time cap", "n_diff": len(diffs)}
                del model
                torch.cuda.empty_cache()
                break
            diffs_a = np.stack(diffs)
            half = np.random.default_rng(int(cfg["half_seed"])).permutation(len(diffs_a))
            a, b = half[: len(half) // 2], half[len(half) // 2 :]
            ma, mb = unit(diffs_a[a].mean(0)), unit(diffs_a[b].mean(0))
            mean_raw = diffs_a.mean(0)
            averaged = unit(mean_raw)
            single = {}
            for pid in baselines:
                idx = next(i for i, p in enumerate(pairs) if p["pair_id"] == pid)
                single[pid] = unit(diffs_a[idx])
            np.save(art / f"vec_{name}_averaged.npy", averaged)
            for pid, vec in single.items():
                np.save(art / f"vec_{name}_{pid}.npy", vec)
            vector_info = {
                "raw_norm_mean": float(np.mean(raw_norms)),
                "raw_norm_min": float(np.min(raw_norms)),
                "raw_norm_max": float(np.max(raw_norms)),
                "mean_raw_norm": float(np.linalg.norm(mean_raw)),
                "half_cosine": float(np.dot(ma, mb)),
                "n_pairs": len(diffs),
            }
            save_json(art / f"vector_diag_{name}.json", vector_info)
            note(f"{name} half cosine {vector_info['half_cosine']:.3f}")

            # size panels from two timed gens
            t1 = time.time()
            _ = generate(model, tok, block, render_eval(tok, test[0]["text"]), None, 0.0, 0)
            _ = generate(model, tok, block, render_eval(tok, test[1]["text"]), averaged, 1.0, 1)
            sec = (time.time() - t1) / 2
            models_left = 2 if name == "qwen" else 1
            remain = max(0.0, (report_at - time.time()) / models_left)
            n_cal, n_conf = int(cfg["target_calibration"]), int(cfg["target_confirmation"])
            while n_cal >= 8 and (4 * n_cal + 6 * n_conf) * sec > remain * 0.9:
                if n_conf > 16:
                    n_conf = max(16, n_conf // 2)
                elif n_cal > 8:
                    n_cal -= 4
                else:
                    n_conf = max(8, n_conf - 4)
                    if n_conf <= 8 and n_cal <= 8:
                        break
            cal_rows = test[:n_cal]
            conf_rows = test[n_cal:n_cal + n_conf]
            note(f"{name} sec {sec:.2f} panels cal {n_cal} confirm {n_conf}")
            # median residual norm on calibration prefixes
            norms = []
            for row in cal_rows:
                h, _, _ = capture(model, tok, block, render_eval(tok, row["text"]))
                norms.append(float(torch.linalg.vector_norm(h[0, -1].float()).cpu()))
            median_norm = float(np.median(norms))
            fractions = [float(x) for x in cfg["dose_fractions"]]
            doses = [frac * median_norm for frac in fractions]
            save_json(art / f"rendering_{name}.json", {
                "eval_bridge": cfg["eval_bridge"],
                "example": render_eval(tok, cal_rows[0]["text"]),
                "median_residual_norm": median_norm,
                "doses": doses,
                "n_cal": n_cal,
                "n_confirm": n_conf,
                "sec_per_gen": sec,
            })
            unedited = score_panel(model, tok, block, cal_rows, None, 0.0, "unedited")
            curve = []
            chosen = None
            for frac, dose in zip(fractions, doses):
                if late() or len(unedited) < n_cal:
                    break
                edited = score_panel(model, tok, block, cal_rows, averaged, dose, f"avg_{frac}")
                if len(edited) < len(unedited):
                    break
                su, se = summarize(unedited), summarize(edited)
                improves = se["mean_score"] >= su["mean_score"] + float(cfg["score_improve_min"])
                empty_ok = se["empty"] <= su["empty"]
                rep_ok = (se["repetitive"] / se["n"]) <= (su["repetitive"] / su["n"]) + float(cfg["repetition_rate_allowance"])
                rec = {"fraction": frac, "dose": dose, "unedited": su, "edit": se, "improves": improves and empty_ok and rep_ok}
                curve.append(rec)
                note(f"{name} frac {frac} score {se['mean_score']:.3f} vs {su['mean_score']:.3f} ok {rec['improves']}")
                if rec["improves"] and chosen is None:
                    chosen = rec
            flat = chosen is None
            if flat:
                dose = float(cfg["fallback_fraction"]) * median_norm
                conf_use = conf_rows[: int(cfg["flat_confirmation"])]
                dose_note = "calibration flat; fallback dose, small descriptive confirmation"
            else:
                dose = float(chosen["dose"])
                conf_use = conf_rows
                dose_note = "lowest calibration dose with a score gain and no empty/repetition penalty"
            arms = {
                "unedited": (None, 0.0),
                "averaged": (averaged, dose),
                "reversed": (averaged, -dose),
            }
            for pid, vec in single.items():
                arms[pid] = (vec, dose)
            # realised norm check on one prefix
            norm_check = {}
            for arm, (direction, alpha) in arms.items():
                if direction is None:
                    continue
                _, _, realised = generate(model, tok, block, render_eval(tok, cal_rows[0]["text"]), direction, alpha, 0)
                norm_check[arm] = realised
            blocks = []
            ckpt = art / f"confirmation_{name}.jsonl"
            for row in conf_use:
                if late():
                    break
                block_rec = {"review_id": row["review_id"]}
                complete = True
                for arm, (direction, alpha) in arms.items():
                    if late():
                        complete = False
                        break
                    got = score_panel(model, tok, block, [row], direction, alpha, arm)
                    if not got:
                        complete = False
                        break
                    block_rec[arm] = got[0]
                if complete:
                    blocks.append(block_rec)
                    with ckpt.open("a") as fh:
                        fh.write(json.dumps(block_rec) + "\n")
            by_arm = {arm: [b[arm] for b in blocks] for arm in arms}
            single_rows = [by_arm[pid] for pid in single if pid in by_arm and len(by_arm[pid]) == len(blocks)]
            mean_single_score = None
            mean_single_rate = None
            if single_rows and blocks:
                mean_single_score = float(np.mean([[r["positive_score"] for r in rows] for rows in single_rows]))
                mean_single_rate = float(np.mean([[float(r["positive"]) for r in rows] for rows in single_rows]))
            summary = {arm: summarize(rows) for arm, rows in by_arm.items()}
            contrasts = {}
            if blocks and "averaged" in by_arm and "unedited" in by_arm:
                contrasts["averaged_minus_unedited_score"] = paired_interval(by_arm["unedited"], by_arm["averaged"], "positive_score")
                contrasts["averaged_minus_unedited_rate"] = paired_interval(by_arm["unedited"], by_arm["averaged"], "positive")
            if blocks and "reversed" in by_arm:
                contrasts["averaged_minus_reversed_score"] = paired_interval(by_arm["reversed"], by_arm["averaged"], "positive_score")
                contrasts["averaged_minus_reversed_rate"] = paired_interval(by_arm["reversed"], by_arm["averaged"], "positive")
            per_base = {}
            for pid in single:
                if pid in by_arm and blocks:
                    per_base[pid] = {
                        "summary": summary.get(pid),
                        "minus_unedited_rate": paired_interval(by_arm["unedited"], by_arm[pid], "positive"),
                        "averaged_minus_this_score": paired_interval(by_arm[pid], by_arm["averaged"], "positive_score"),
                    }
            su = summary.get("unedited", {})
            sa = summary.get("averaged", {})
            sr = summary.get("reversed", {})
            gate = None
            if su and sa and sr and su.get("n"):
                gate = {
                    "rate_gain": sa["positive_rate"] - su["positive_rate"],
                    "score_shift": sa["mean_score"] - su["mean_score"],
                    "pos_vs_reversed_score": sa["mean_score"] - sr["mean_score"],
                    "passes_rate": (sa["positive_rate"] - su["positive_rate"]) >= float(cfg["gate"]["min_rate_gain"]),
                    "passes_score": sa["mean_score"] > su["mean_score"],
                    "passes_sign": sa["mean_score"] > sr["mean_score"],
                }
                gate["passes"] = bool(gate["passes_rate"] and gate["passes_score"] and gate["passes_sign"])
            # blinded sample
            blind = []
            key = []
            brng = np.random.default_rng(0)
            for b in blocks[: int(cfg["blind_n"])]:
                order = ["unedited", "averaged"]
                brng.shuffle(order)
                for arm in order:
                    code = hashlib.sha256(f"{name}:{b['review_id']}:{arm}".encode()).hexdigest()[:8]
                    blind.append({"code": code, "prefix": b[arm]["prefix"], "continuation": b[arm]["continuation"]})
                    key.append({"code": code, "arm": arm, "review_id": b["review_id"]})
            save_json(art / f"blind_sample_{name}.json", blind)
            save_json(art / f"blind_key_{name}.json", key)
            results[name] = {
                "vector": vector_info,
                "dose_note": dose_note,
                "flat": flat,
                "dose": dose,
                "median_residual_norm": median_norm,
                "calibration_curve": curve,
                "norm_check": norm_check,
                "n_confirmation": len(blocks),
                "summary": summary,
                "contrasts": contrasts,
                "per_baseline": per_base,
                "mean_single_score": mean_single_score,
                "mean_single_rate": mean_single_rate,
                "gate": gate,
                "smoke": smoke,
            }
            note(f"{name} confirm n {len(blocks)} gate {gate}")
            del model
            torch.cuda.empty_cache()
        state["results"] = results
    except Exception:
        state["error"] = traceback.format_exc()
        note(state["error"])
    state["elapsed_sec"] = time.time() - t0
    save_json(art / "run_state.json", state)
    (dest / "RESULTS.md").write_text(render(state, cfg))
    print("wrote RESULTS", flush=True)
    return 0


def render(state: dict, cfg: dict) -> str:
    res = state.get("results") or {}
    lines = ["# RESULTS — `imdb_matched_pair_sentiment_v1`", ""]
    for name in ("qwen", "olmo"):
        r = res.get(name) or {}
        g = r.get("gate")
        if not g:
            lines.append(f"Did the averaged edit steer {name} on fresh reviews? {r.get('lead', 'incomplete')}")
            continue
        lines.append(
            f"Did the averaged edit steer {name} on fresh reviews? "
            f"{'Yes' if g['passes'] else 'No'} on the prespecified gate "
            f"(rate gain {g['rate_gain']:.3f}, score shift {g['score_shift']:.3f}, "
            f"positive minus reversed {g['pos_vs_reversed_score']:.3f})."
        )
        ms = r.get("mean_single_score")
        sa = (r.get("summary") or {}).get("averaged") or {}
        if ms is None or not sa:
            lines.append(f"Did it beat the fixed single-pair controls on {name}? Incomplete.")
        else:
            lines.append(
                f"Did it beat the fixed single-pair controls on {name}? "
                f"Averaged mean score {sa.get('mean_score')} versus the mean of the three fixed pairs {ms:.3f}."
            )
        lines.append(
            f"Did reversing the sign reverse the score shift on {name}? "
            f"Positive minus reversed score difference is {g['pos_vs_reversed_score']:.3f}."
        )
        su = (r.get("summary") or {}).get("unedited") or {}
        lines.append(
            f"Quality on {name}: unedited empty {su.get('empty')} repetitive {su.get('repetitive')} truncated {su.get('truncated')}; "
            f"averaged empty {sa.get('empty')} repetitive {sa.get('repetitive')} truncated {sa.get('truncated')} "
            f"of n {sa.get('n')}. Blinded readability is not rated yet, so any usability claim stays provisional."
        )
    lines += ["", "Sites are the frozen pre-block inputs from the previous run: Qwen layer 17 and OLMo layer 11. They are starting sites, not a layer search.", ""]
    lines.append("Contrasts were written without IMDb review text. Three single-pair baselines were drawn by seed from different wording families before any generation.")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps({k: {kk: vv for kk, vv in (res.get(k) or {}).items() if kk in ("dose_note", "flat", "dose", "n_confirmation", "summary", "contrasts", "gate", "mean_single_score", "mean_single_rate", "vector", "norm_check")} for k in ("qwen", "olmo")}, indent=2, default=str)[:12000])
    lines.append("```")
    if state.get("error"):
        lines += ["", "## Stop", "", "```", state["error"][-2000:], "```"]
    lines.append("")
    lines.append(f"Elapsed {state.get('elapsed_sec', 0):.0f}s. Reproduction: `python scripts/run_imdb_matched_pair_sentiment.py --config configs/imdb_matched_pair_sentiment_v1.json --work-root $WORK`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
