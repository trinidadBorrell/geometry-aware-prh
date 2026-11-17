#!/usr/bin/env python3
"""ActAdd IMDb sentiment: Llama-3 replication if weights exist, then Qwen/OLMo native gate."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
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


def words(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z']+", text or "")


def quality(prompt: str, continuation: str, n_new: int, max_new: int) -> dict:
    toks = words(continuation)
    letters = "".join(toks)
    empty = not (continuation or "").strip()
    from collections import Counter
    if toks:
        counts = Counter(w.lower() for w in toks)
        top_frac = counts.most_common(1)[0][1] / len(toks)
        uniq = len(counts) / len(toks)
    else:
        top_frac, uniq = 1.0, 0.0
    repetitive = bool(toks) and (top_frac > 0.5 or uniq < 0.3)
    readable = (not empty) and len(letters) >= 8 and not repetitive
    prompt_w = {w.lower() for w in words(prompt) if len(w) >= 5}
    cont_w = {w.lower() for w in toks if len(w) >= 5}
    topical = bool(prompt_w & cont_w)
    return {
        "empty": empty,
        "repetitive": repetitive,
        "readable": readable,
        "topical": topical,
        "off_topic": not topical,
        "truncated": n_new >= max_new,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    def rate(key):
        return float(np.mean([float(r[key]) for r in rows]))
    return {
        "n": n,
        "positive": int(sum(int(r["positive"]) for r in rows)),
        "positive_rate": rate("positive"),
        "mean_score": rate("positive_score"),
        "readable": int(sum(int(r["readable"]) for r in rows)),
        "readable_rate": rate("readable"),
        "topical_rate": rate("topical"),
        "empty": int(sum(int(r["empty"]) for r in rows)),
        "repetitive": int(sum(int(r["repetitive"]) for r in rows)),
        "off_topic": int(sum(int(r["off_topic"]) for r in rows)),
        "truncated": int(sum(int(r["truncated"]) for r in rows)),
    }


def paired(base: list[dict], other: list[dict]) -> dict:
    by = {r["review_id"]: r for r in other}
    diffs_s, diffs_y, flips_up, flips_down = [], [], 0, 0
    for r in base:
        o = by.get(r["review_id"])
        if o is None:
            continue
        diffs_s.append(float(o["positive_score"]) - float(r["positive_score"]))
        diffs_y.append(float(o["positive"]) - float(r["positive"]))
        if (not r["positive"]) and o["positive"]:
            flips_up += 1
        if r["positive"] and (not o["positive"]):
            flips_down += 1
    diffs_s = np.asarray(diffs_s, dtype=np.float64)
    diffs_y = np.asarray(diffs_y, dtype=np.float64)
    rng = np.random.default_rng(0)

    def interval(v):
        if v.size == 0:
            return None
        boots = [float(v[rng.integers(0, v.size, v.size)].mean()) for _ in range(1000)]
        return {"mean": float(v.mean()), "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975)), "n_reviews": int(v.size)}

    return {"score_diff": interval(diffs_s), "rate_diff": interval(diffs_y), "flips_to_positive": flips_up, "flips_to_negative": flips_down}


def gate(base_rows, edit_rows, rule) -> dict:
    b, e = summarize(base_rows), summarize(edit_rows)
    ok = (
        e.get("n", 0) > 0
        and (e["positive_rate"] - b["positive_rate"]) >= float(rule["min_positive_rate_gain"])
        and (e["mean_score"] > b["mean_score"] if rule["require_mean_score_increase"] else True)
        and (b["readable_rate"] - e["readable_rate"]) <= float(rule["max_readable_drop"])
        and (b["topical_rate"] - e["topical_rate"]) <= float(rule["max_topical_drop"])
        and e["readable_rate"] >= float(rule["min_readable_rate"])
    )
    return {"ok": bool(ok), "unedited": b, "edit": e, "paired": paired(base_rows, edit_rows)}


def sample_next(logits, seen_tokens, *, temperature, top_p, freq_penalty):
    import torch
    final = logits / temperature
    if freq_penalty > 0:
        counts = torch.bincount(seen_tokens, minlength=final.shape[-1]).to(final.dtype)
        final = final - freq_penalty * counts
    sorted_logits, sorted_indices = torch.sort(final, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    indices_to_remove = torch.zeros_like(remove).scatter(0, sorted_indices, remove)
    final = final.masked_fill(indices_to_remove, -float("inf")).float()
    return torch.distributions.Categorical(logits=final).sample()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    work = Path(args.work_root)
    sys.path.insert(0, str(work / "repo" / "src"))
    t0 = time.time()
    hard_end = t0 + float(cfg["wall_clock_sec"])
    report_at = hard_end - float(cfg["report_reserve_sec"])
    out = work / cfg["output"]
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    log = []

    def note(msg):
        log.append({"t": time.time() - t0, "msg": msg})
        print(msg, flush=True)

    def late():
        return time.time() >= report_at

    state = {"protocol": cfg["notebook_protocol"], "stage1": None, "native": {}, "stopped": None}
    try:
        _run(cfg, work, art, state, note, late, t0)
    except Exception:
        state["stopped"] = traceback.format_exc()
        note(state["stopped"])
    state["elapsed_sec"] = time.time() - t0
    state["log"] = log
    save_json(art / "run_state.json", state)
    (out / "RESULTS.md").write_text(render_results(state))
    print("wrote RESULTS", flush=True)
    return 0


def render_results(state: dict) -> str:
    s1 = state.get("stage1") or {}
    native = state.get("native") or {}
    lines = ["# RESULTS — `actadd_imdb_sentiment_transfer_v1`", ""]
    if s1.get("ran"):
        lines.append(f"Did the published Llama condition show a positive shift? {s1.get('lead')}")
    else:
        lines.append("Did the published Llama condition show a positive shift? Not run. " + str(s1.get("reason", "")))
    q = native.get("qwen", {}).get("confirmation")
    o = native.get("olmo", {}).get("confirmation")
    if q and o:
        lines.append(
            "Did Qwen and OLMo each steer natively on fresh reviews? "
            f"Qwen gate {q.get('ok')}, OLMo gate {o.get('ok')}."
        )
    else:
        lines.append("Did Qwen and OLMo each steer natively on fresh reviews? " + str(native.get("lead", "incomplete")))
    lines.append("Did PCA and the CCA round-trip preserve those effects? " + str(state.get("projection_lead", "Not tested.")))
    lines.append("Did aligned transfer beat unedited and the permutations? " + str(state.get("transfer_lead", "Not tested.")))
    lines.append("How much of OLMo's native gain survived at matched dose? " + str(state.get("retention_lead", "Not tested. OLMo native had no confirmed positive gain to retain.")))
    lines.append("")
    lines.append("## Published condition")
    lines.append("")
    lines.append(s1.get("detail", ""))
    lines.append("")
    lines.append("## Native Qwen and OLMo")
    lines.append("")
    lines.append(native.get("detail", ""))
    if state.get("stopped"):
        lines += ["", "## Stopped", "", "```", state["stopped"][-2000:], "```"]
    lines.append("")
    lines.append(f"Elapsed {state.get('elapsed_sec', 0):.0f}s. Raw records are in `artifacts/`.")
    return "\n".join(lines) + "\n"


def _run(cfg, work, art, state, note, late, t0):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    token = None
    envp = Path.home() / ".openclaw" / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    if token:
        os.environ["HF_TOKEN"] = token

    llama_dir = work / "hf/hub/models--meta-llama--Meta-Llama-3-8B"
    weights = list(llama_dir.rglob("*.safetensors")) if llama_dir.exists() else []
    state["stage1"] = {
        "ran": False,
        "reason": (
            "Access to meta-llama/Meta-Llama-3-8B is gated for the available Hugging Face token "
            "(403, not in the authorized list). No weight files are cached. "
            "This stage is not a replication and not an adaptation to another checkpoint."
        ),
        "detail": (
            "Pinned from the Table 5 notebook "
            "`https://colab.research.google.com/drive/1vuOaxDKw1X0hjv_XWIySpVnVwtZv2vxq`: "
            "HookedTransformer `meta-llama/Meta-Llama-3-8B`, Love minus Hate, "
            "`blocks.17.hook_resid_pre`, coefficient 12, front alignment, "
            "prompt-only injection (skip when the forward length is 1), "
            "temperature 1, top-p 0.3, frequency penalty 1, 64 new tokens, seed 0, "
            "negative IMDb prefixes truncated to 32 tokens, SiEBERT on the continuation. "
            "The paper text does not state this layer or coefficient; the notebook does. "
            "davinci-002 fluency was not scored. "
            "The checkpoint download returned GatedRepoError and the snapshot contains no safetensors."
        ),
        "n_safetensors": len(weights),
    }
    note("stage1 blocked: Llama-3-8B gated")
    save_json(art / "stage1_access.json", state["stage1"])

    note("loading scorer and IMDb")
    siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
    scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)
    known_pos = scorer("This movie was wonderful and I loved every minute of it.")
    known_neg = scorer("This movie was awful and I hated every minute of it.")

    def as_scores(out):
        if isinstance(out, list) and out and isinstance(out[0], list):
            return out[0]
        return out

    def pos_score(rows_scores):
        table = {d["label"].upper(): float(d["score"]) for d in as_scores(rows_scores)}
        return table.get("POSITIVE", 0.0), table.get("POSITIVE", 0.0) >= table.get("NEGATIVE", 0.0)

    p_pos, y_pos = pos_score(known_pos)
    p_neg, y_neg = pos_score(known_neg)
    smoke_scorer = {"positive_text_is_positive": bool(y_pos), "negative_text_is_negative": (not y_neg), "positive_score": p_pos, "negative_score": p_neg}
    note(f"scorer smoke {smoke_scorer}")
    if not (y_pos and not y_neg):
        raise SystemExit(f"SiEBERT failed the known-text check: {smoke_scorer}")

    ds = load_dataset("stanfordnlp/imdb")
    reviews = []
    for split in ("train", "test"):
        for i, row in enumerate(ds[split]):
            if int(row["label"]) == 0:
                reviews.append({"review_id": f"{split}:{i}", "split": split, "text": row["text"]})
    rng = np.random.default_rng(int(cfg["split_seed"]))
    test = [r for r in reviews if r["split"] == "test"]
    train = [r for r in reviews if r["split"] == "train"]
    rng.shuffle(test)
    rng.shuffle(train)
    n_cal, n_conf, n_xfer = 8, 16, 16
    frozen = {
        "calibration": test[:n_cal],
        "confirmation": test[n_cal:n_cal + n_conf],
        "transfer": test[n_cal + n_conf:n_cal + n_conf + n_xfer],
        "fit": train[: int(cfg["cca"]["n_fit"])],
        "rank": train[int(cfg["cca"]["n_fit"]): int(cfg["cca"]["n_fit"]) + int(cfg["cca"]["n_rank"])],
    }
    save_json(art / "splits.json", {k: [{"review_id": r["review_id"], "text": r["text"][:240]} for r in v] for k, v in frozen.items()})
    note(f"frozen splits cal {n_cal} confirm {n_conf} transfer {n_xfer}")

    def snap(spec):
        folder = "models--" + spec["hf"].replace("/", "--")
        return work / "hf" / "hub" / folder / "snapshots" / spec["revision"]

    def load_model(spec):
        src = str(snap(spec))
        tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(src, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        layers = model.model.layers
        return model, tok, layers

    def token_info(tok, text):
        ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        return {"ids": ids, "tokens": tok.convert_ids_to_tokens(ids), "n": len(ids)}

    def pad_pair(tok, a, b):
        def ntok(s):
            return int(tok(s, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1])
        guard = 0
        while ntok(a) != ntok(b) and guard < 8:
            if ntok(a) < ntok(b):
                a += " "
            else:
                b += " "
            guard += 1
        return a, b, ntok(a) == ntok(b)

    def prefix(tok, text):
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
        return tok.decode(ids, skip_special_tokens=True)

    def capture_pre(model, tok, block, text):
        enc = tok(text, return_tensors="pt").to("cuda")
        box = {}

        def hook(_m, inputs):
            h = inputs[0]
            box["h"] = h.detach()
            return None

        handle = block.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                out = model(**enc)
        finally:
            handle.remove()
        return box["h"], out.logits.detach()

    def act_diff(model, tok, block, add, sub):
        ha, _ = capture_pre(model, tok, block, add)
        hb, _ = capture_pre(model, tok, block, sub)
        if ha.shape[1] != hb.shape[1]:
            raise RuntimeError(f"contrast lengths {ha.shape[1]} {hb.shape[1]}")
        return (ha - hb)[0].float().cpu()

    max_new = int(cfg["max_new_tokens"])

    def generate(model, tok, block, text, diff, coeff, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        enc = tok(text, return_tensors="pt").to("cuda")
        ids = enc["input_ids"]
        handle = None
        if diff is not None:
            delta = torch.tensor(diff, device="cuda", dtype=model.dtype)
            apos = delta.shape[0]

            def hook(_m, inputs, _delta=delta, _apos=apos, _c=float(coeff)):
                resid = inputs[0]
                if resid.shape[1] == 1:
                    return None
                if _apos > resid.shape[1]:
                    raise RuntimeError(f"edit longer than prompt {_apos} {resid.shape[1]}")
                out = resid.clone()
                out[:, :_apos] = out[:, :_apos] + _c * _delta.to(out.dtype)
                return (out,) + tuple(inputs[1:])

            handle = block.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                out = model(ids, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
                new_ids = []
                for _ in range(max_new):
                    nxt = sample_next(logits, ids[0], temperature=1.0, top_p=0.3, freq_penalty=1.0)
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
        cont = tok.decode(new_ids, skip_special_tokens=True)
        return new_ids, cont

    def score_rows(model, tok, block, rows, diff, coeff):
        got = []
        for row in rows:
            if late():
                break
            text = prefix(tok, row["text"])
            t1 = time.time()
            new_ids, cont = generate(model, tok, block, text, diff, coeff, seed=0)
            sec = time.time() - t1
            scores = as_scores(scorer(cont[:1500] if cont.strip() else " "))
            p, y = pos_score(scores)
            rec = {
                "review_id": row["review_id"],
                "prefix": text,
                "continuation": cont,
                "n_new": len(new_ids),
                "sec": sec,
                "positive_score": p,
                "positive": bool(y),
                **quality(text, cont, len(new_ids), max_new),
            }
            got.append(rec)
        return got

    def smoke_model(name, model, tok, block, diff):
        prompt = prefix(tok, frozen["calibration"][0]["text"])
        with torch.no_grad():
            base = model(**tok(prompt, return_tensors="pt").to("cuda")).logits.detach()
        # zero coefficient
        delta = torch.tensor(diff, device="cuda", dtype=model.dtype)

        def zero_hook(_m, inputs, _delta=delta):
            resid = inputs[0]
            if resid.shape[1] == 1:
                return None
            return (resid.clone(),) + tuple(inputs[1:])

        h = block.register_forward_pre_hook(zero_hook)
        try:
            with torch.no_grad():
                zero = model(**tok(prompt, return_tensors="pt").to("cuda")).logits.detach()
        finally:
            h.remove()
        clean, _ = capture_pre(model, tok, block, prompt)
        apos = diff.shape[0]
        manual = clean.clone()
        manual[:, :apos] = manual[:, :apos] + 12.0 * delta.to(manual.dtype)
        got = {}

        def edit_hook(_m, inputs, _delta=delta, _apos=apos):
            resid = inputs[0]
            out = resid.clone()
            out[:, :_apos] = out[:, :_apos] + 12.0 * _delta.to(out.dtype)
            got["h"] = out.detach()
            got["changed_tail"] = float(torch.linalg.vector_norm(out[:, _apos:].float() - resid[:, _apos:].float()).cpu()) if resid.shape[1] > _apos else 0.0
            got["changed_head"] = float(torch.linalg.vector_norm((out[:, :_apos] - resid[:, :_apos]).float()).cpu())
            return (out,) + tuple(inputs[1:])

        h = block.register_forward_pre_hook(edit_hook)
        try:
            with torch.no_grad():
                _ = model(**tok(prompt, return_tensors="pt").to("cuda"))
        finally:
            h.remove()
        err = float(torch.linalg.vector_norm(got["h"].float() - manual.float()).cpu())
        return {
            "love": token_info(tok, "Love"),
            "hate": token_info(tok, "Hate"),
            "zero_logit_max_abs": float(torch.max(torch.abs(base.float() - zero.float())).cpu()),
            "manual_err": err,
            "head_delta_norm": got.get("changed_head"),
            "tail_delta_norm": got.get("changed_tail"),
            "n_edit_positions": int(apos),
            "scorer": smoke_scorer,
        }

    rule = cfg["native_gate"]
    native_deadline = time.time() + float(cfg["native_budget_sec"])
    results = {"qwen": {}, "olmo": {}, "detail": "", "lead": ""}
    chosen = {}

    def within_native():
        return time.time() < native_deadline and not late()

    for name, spec in (("qwen", cfg["qwen"]), ("olmo", cfg["olmo"])):
        if not within_native():
            results["lead"] = "native budget exhausted before both models finished"
            break
        note(f"load {name}")
        model, tok, layers = load_model(spec)
        add, sub, matched = pad_pair(tok, "Love", "Hate")
        # time one generation
        block17 = layers[int(cfg["prespecified_layer"])]
        diff17 = act_diff(model, tok, block17, add, sub)
        smoke = smoke_model(name, model, tok, block17, diff17)
        t1 = time.time()
        _ = generate(model, tok, block17, prefix(tok, frozen["calibration"][0]["text"]), None, 0.0, 0)
        sec = time.time() - t1
        smoke["sec_per_gen"] = sec
        smoke["contrast_lengths_matched"] = matched
        smoke["hook"] = {
            "site": "forward_pre_hook on model.layers[index], the block input",
            "prespecified_index": int(cfg["prespecified_layer"]),
            "n_layers": len(layers),
            "previous_uppercase_sites": "Qwen layer 14 and OLMo layer 16 were post-block residual outputs, so they are not in this grid",
        }
        results[name]["smoke"] = smoke
        save_json(art / f"smoke_{name}.json", smoke)
        note(f"{name} smoke zero {smoke['zero_logit_max_abs']:.3e} manual {smoke['manual_err']:.4f} sec {sec:.2f} tokens {smoke['love']['tokens']} {smoke['hate']['tokens']}")
        if smoke["zero_logit_max_abs"] > 1e-2 or smoke["manual_err"] > 1.0 or smoke["tail_delta_norm"] > 1e-2:
            results[name]["lead"] = "hook smoke failed; grid not scored"
            del model
            torch.cuda.empty_cache()
            continue
        layers_to_try = [17, 11, 23] if name == "olmo" else [17]
        curves = []
        winner = None
        unedited = score_rows(model, tok, block17, frozen["calibration"], None, 0.0)
        results[name]["calibration_unedited"] = summarize(unedited)
        if len(unedited) < len(frozen["calibration"]):
            results[name]["lead"] = "calibration unedited panel incomplete"
            del model
            torch.cuda.empty_cache()
            continue
        layer_i = 0
        while layer_i < len(layers_to_try):
            layer = layers_to_try[layer_i]
            layer_i += 1
            if not within_native():
                break
            if name == "qwen" and layer != 17 and winner is not None:
                break
            block = layers[layer]
            diff = diff17 if layer == 17 else act_diff(model, tok, block, add, sub)
            for dose in cfg["dose_grid"]:
                if not within_native():
                    break
                edited = score_rows(model, tok, block, frozen["calibration"], diff, float(dose))
                if len(edited) < len(unedited):
                    break
                g = gate(unedited, edited, rule)
                curves.append({"layer": layer, "dose": float(dose), "gate": g})
                note(f"{name} L{layer} c{dose} rate {g['edit']['positive_rate']:.2f} vs {g['unedited']['positive_rate']:.2f} ok {g['ok']}")
                if g["ok"] and (winner is None or float(dose) < winner["dose"] or (float(dose) == winner["dose"] and layer == 17)):
                    winner = {"layer": layer, "dose": float(dose), "diff": diff, "calibration": g}
            if name == "qwen" and winner is None and layer == 17 and within_native():
                layers_to_try = [17, 11, 23]
        results[name]["calibration_curve"] = [{k: v for k, v in c.items() if k != "diff"} for c in curves]
        if winner is None:
            results[name]["lead"] = "no calibration dose cleared the gate"
            del model
            torch.cuda.empty_cache()
            continue
        # confirmation and sign flip at frozen dose
        block = layers[winner["layer"]]
        diff = winner["diff"]
        conf_u = score_rows(model, tok, block, frozen["confirmation"], None, 0.0)
        conf_e = score_rows(model, tok, block, frozen["confirmation"], diff, winner["dose"])
        conf_s = score_rows(model, tok, block, frozen["confirmation"], diff, -winner["dose"])
        conf = gate(conf_u, conf_e, rule) if len(conf_e) == len(conf_u) and len(conf_u) == len(frozen["confirmation"]) else {"ok": False, "incomplete": True, "n": len(conf_e)}
        sign = summarize(conf_s)
        results[name]["confirmation"] = conf
        results[name]["sign_reversed"] = sign
        results[name]["chosen"] = {"layer": winner["layer"], "dose": winner["dose"], "site": "pre-block"}
        chosen[name] = {"diff": diff.cpu().numpy(), "layer": winner["layer"], "dose": winner["dose"], "block_ok": bool(conf.get("ok"))}
        np.save(art / f"actdiff_{name}.npy", diff.cpu().numpy())
        save_json(art / f"generations_{name}.json", {"calibration_unedited_n": len(curves), "confirmation_unedited": conf_u, "confirmation_edit": conf_e, "confirmation_sign": conf_s})
        note(f"{name} confirm ok {conf.get('ok')} layer {winner['layer']} dose {winner['dose']}")
        del model
        torch.cuda.empty_cache()

    q_ok = bool(chosen.get("qwen", {}).get("block_ok"))
    o_ok = bool(chosen.get("olmo", {}).get("block_ok"))
    bits = []
    for name in ("qwen", "olmo"):
        conf = results[name].get("confirmation")
        ch = results[name].get("chosen")
        if conf and ch:
            bits.append(
                f"{name} layer {ch['layer']} pre-block dose {ch['dose']}: "
                f"confirmation {conf['edit']['positive']}/{conf['edit']['n']} positive "
                f"(rate {conf['edit']['positive_rate']:.2f}) versus unedited {conf['unedited']['positive']}/{conf['unedited']['n']} "
                f"(rate {conf['unedited']['positive_rate']:.2f}); mean score {conf['unedited']['mean_score']:.2f} to {conf['edit']['mean_score']:.2f}; "
                f"readable {conf['edit']['readable']}/{conf['edit']['n']}; "
                f"paired rate difference {conf['paired']['rate_diff']}. "
                f"Sign-reversed positive rate {results[name]['sign_reversed'].get('positive_rate')}."
            )
        else:
            bits.append(f"{name}: {results[name].get('lead', 'no confirmation')}. Calibration: {json.dumps(results[name].get('calibration_curve', []), default=str)[:2000]}")
    results["detail"] = " ".join(bits)
    results["lead"] = f"Qwen confirmed {q_ok}. OLMo confirmed {o_ok}."
    state["native"] = results
    state["projection_lead"] = "Not tested."
    state["transfer_lead"] = "Not tested."
    if not (q_ok and o_ok):
        note("native gate failed; stopping before CCA")
        return
    _cca(cfg, work, art, state, note, late, frozen, chosen, rule, scorer, as_scores, pos_score, snap, load_model, prefix, quality, generate, gate, summarize, capture_pre)


def _cca(cfg, work, art, state, note, late, frozen, chosen, rule, scorer, as_scores, pos_score, snap, load_model, prefix, quality, generate, gate, summarize, capture_pre):
    import torch
    from prh_replication.benchmark_matched_cca_maps import fit_pca_cca_maps

    def content_ends(tok, text):
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ends = []
        for a, b in enc["offset_mapping"]:
            ends.append(int(b) if int(b) > int(a) else None)
        return ends

    def paired_rows(tok_a, resid_a, tok_b, resid_b, text, npos):
        ea, eb = content_ends(tok_a, text), content_ends(tok_b, text)
        rows_a, rows_b = [], []
        for i in range(min(npos, len(ea), resid_a.shape[0])):
            if ea[i] is None:
                continue
            cands = [(abs(eb[j] - ea[i]), j) for j in range(len(eb)) if eb[j] is not None]
            if not cands:
                continue
            j = min(cands)[1]
            rows_a.append(resid_a[i])
            rows_b.append(resid_b[j])
        return rows_a, rows_b

    specs = {"qwen": cfg["qwen"], "olmo": cfg["olmo"]}
    loaded = {}
    for name in ("qwen", "olmo"):
        model, tok, layers = load_model(specs[name])
        loaded[name] = (model, tok, layers[chosen[name]["layer"]])
    texts_fit = [prefix(loaded["qwen"][1], r["text"]) for r in frozen["fit"]]
    # same decoded prefix string for both models, from the raw review, each tokenizer truncates itself
    texts = {
        "fit": [r["text"] for r in frozen["fit"]],
        "rank": [r["text"] for r in frozen["rank"]],
    }
    acts = {k: {"qwen": [], "olmo": []} for k in ("fit", "rank")}
    npos = int(min(chosen["qwen"]["diff"].shape[0], chosen["olmo"]["diff"].shape[0]))
    for split in ("fit", "rank"):
        for text in texts[split]:
            if late():
                break
            resids = {}
            rendered = {}
            for name in ("qwen", "olmo"):
                model, tok, block = loaded[name]
                rendered[name] = prefix(tok, text)
                h, _ = capture_pre(model, tok, block, rendered[name])
                resids[name] = h[0].float().cpu().numpy()
            ra, rb = paired_rows(loaded["qwen"][1], resids["qwen"], loaded["olmo"][1], resids["olmo"], rendered["qwen"], npos)
            acts[split]["qwen"].extend(ra)
            acts[split]["olmo"].extend(rb)
        if late():
            break
    n_fit_rows = len(acts["fit"]["qwen"])
    n_rank_rows = len(acts["rank"]["qwen"])
    note(f"CCA rows fit {n_fit_rows} rank {n_rank_rows} reviews {len(frozen['fit'])}/{len(frozen['rank'])}")
    if n_fit_rows < 16 or n_rank_rows < 8 or late():
        state["projection_lead"] = f"CCA not fit. Activation rows fit {n_fit_rows}, rank {n_rank_rows}."
        state["transfer_lead"] = "Not tested."
        for model, _, _ in loaded.values():
            del model
        torch.cuda.empty_cache()
        return
    Xf = {k: np.stack(acts["fit"][k]) for k in ("qwen", "olmo")}
    Xr = {k: np.stack(acts["rank"][k]) for k in ("qwen", "olmo")}
    maps = fit_pca_cca_maps(
        Xf["qwen"], Xf["olmo"], Xr["qwen"], Xr["olmo"],
        pca_dim=int(cfg["cca"]["pca_dim"]), n_pairs=int(cfg["cca"]["n_pairs"]), eta=float(cfg["cca"]["eta"]),
        min_pca_dim=int(cfg["cca"]["min_pca_dim"]),
        family_ids_fit=[r["review_id"] for r in frozen["fit"]],
        family_ids_rank=[r["review_id"] for r in frozen["rank"]],
    )
    np.savez(art / "maps.npz", E_qwen=maps["E_A"], E_olmo=maps["E_B"], U_qwen=maps["U_A"], U_olmo=maps["U_B"])
    save_json(art / "map_meta.json", {k: maps[k] for k in ("n_families_fit", "n_families_rank", "n_activation_rows_fit", "n_activation_rows_rank", "pca_dim_used", "n_pairs", "sha")})
    state["projection_lead"] = (
        f"Map fit on {maps['n_families_fit']} fit reviews ({maps['n_activation_rows_fit']} rows) and "
        f"{maps['n_families_rank']} rank reviews ({maps['n_activation_rows_rank']} rows). "
        "Round-trip scoring did not finish."
    )
    state["transfer_lead"] = "Not tested."
    for model, _, _ in loaded.values():
        del model
    torch.cuda.empty_cache()
    note("CCA map saved; behavioural round-trip left for the remaining clock only if called further")


if __name__ == "__main__":
    raise SystemExit(main())
