#!/usr/bin/env python3
"""Carry the frozen Qwen sentiment command into Gemma 2 2B through the archived OLMo coordinates."""

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

    path.write_text(json.dumps(obj, indent=2, default=default) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / n if n > 1e-12 else v), n


def shared_passage(text, prompt_chars, cont_chars):
    body = " ".join(text.split())
    if len(body) < prompt_chars + cont_chars:
        return None
    prompt = body[:prompt_chars].rstrip() + " Overall,"
    tail = body[prompt_chars:prompt_chars + cont_chars]
    full = prompt + " " + tail
    offs = [len(prompt), len(prompt) + 41, len(prompt) + 81, len(full)]
    if min(np.diff([0] + offs)) <= 0:
        return None
    return full, offs


def token_for_offset(offsets, pos):
    for i, (a, b) in enumerate(offsets):
        if int(a) < int(pos) <= int(b):
            return i
    if offsets and int(pos) >= int(offsets[-1][1]):
        return len(offsets) - 1
    return None


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}], one-sided lower bound {block['lower95']:+.3f}"


def render_md(state):
    lines = [
        "# RESULTS — `imdb_cca_third_model_transfer_v1`",
        "",
        "Gemma activations were used to fit the ridge decoder. The original Qwen–OLMo CCA space and the Qwen steering command stayed frozen. Gemma's layer and dose were chosen on the native calibration grid only.",
        "",
    ]
    test = state.get("test") or {}
    if test.get("native_gain"):
        ng, tg, gap, ret = test["native_gain"], test["transfer_gain"], test["absolute_gap"], test["retention_80"]
        steered = tg.get("lower95", -1) > 0
        established = bool(ng.get("lower95", -1) > 0 and ret.get("lower95", -1) > 0)
        lines.append(
            "Did the unchanged Qwen command steer Gemma? "
            + ("Yes. " if steered else "No. The transfer gain is not established. ")
            + f"Transfer gain {fmt(tg)}."
        )
        lines.append(f"How close was it to Gemma native? Absolute gap, transfer minus native, {fmt(gap)}.")
        if established:
            lines.append(f"Was 80% retention established? Yes. Native gain {fmt(ng)}. Retention contrast {fmt(ret)}.")
        elif ng.get("lower95", -1) > 0 and ret.get("low", 0) <= 0 <= ret.get("high", 0):
            lines.append(
                f"Was 80% retention established? No. Retention is inconclusive: the interval crosses zero. Native gain {fmt(ng)}. Retention contrast {fmt(ret)}."
            )
        elif ng.get("lower95", -1) > 0:
            lines.append(f"Was 80% retention established? No. Native gain {fmt(ng)}. Retention contrast {fmt(ret)}.")
        else:
            lines.append(
                f"Was 80% retention established? No. Native gain is unresolved, so retention is not expressed as a percentage. Native gain {fmt(ng)}. Transfer gain {fmt(tg)}. Absolute gap {fmt(gap)}."
            )
    elif state.get("stopped"):
        lines.append("Did the unchanged Qwen command steer Gemma? Not tested. No usable Gemma native setting was frozen, so there is no retention claim.")
        lines.append("How close was it to Gemma native? Not tested.")
        lines.append("Was 80% retention established? No.")
    else:
        lines.append("Did the unchanged Qwen command steer Gemma? Incomplete.")
        lines.append("How close was it to Gemma native? Incomplete.")
        lines.append("Was 80% retention established? Incomplete.")
    if state.get("partial"):
        lines.append(f"The test analysis is partial: {state.get('n_blocks', 0)} of {state.get('frozen_n', 0)} frozen reviews.")
    elif state.get("n_blocks"):
        lines.append(f"Coverage is the full frozen panel, n={state.get('n_blocks')}.")
    lines.append("")
    if state.get("chosen"):
        lines.append("Frozen native setting: `" + json.dumps(state["chosen"]) + "`")
        lines.append("")
    if state.get("summary"):
        lines += ["| arm | n | positive | score | empty | repetitive | truncated |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, row in state["summary"].items():
            if row.get("n"):
                lines.append(f"| {arm} | {row['n']} | {row['positive']} | {row['mean_score']:.3f} | {row['empty']} | {row['repetitive']} | {row['truncated']} |")
        lines.append("")
    if test.get("flips"):
        flips = test["flips"]
        lines.append(
            "Secondary flips to positive / to negative: "
            f"native vs unedited {flips['native_vs_unedited'][0]}/{flips['native_vs_unedited'][1]}; "
            f"transfer vs unedited {flips['transfer_vs_unedited'][0]}/{flips['transfer_vs_unedited'][1]}; "
            f"transfer vs native {flips['transfer_vs_native'][0]}/{flips['transfer_vs_native'][1]}."
        )
        lines.append("")
    if state.get("decoder"):
        lines.append("Bridge-validation reconstruction: `" + json.dumps(state["decoder"]) + "`")
        lines.append("")
    if state.get("calibration"):
        lines.append("Native calibration cells are in `artifacts/calibration.json`. Validation is in `artifacts/validation.json`.")
        lines.append("")
    if state.get("error"):
        lines += ["## Stop", "", "```", state["error"][-1800:], "```", ""]
    lines.append(
        f"Elapsed {state.get('elapsed_sec', 0):.0f}s. Reproduction: "
        "`python scripts/run_imdb_cca_third_model_transfer.py --config configs/imdb_cca_third_model_transfer_v1.json --work-root $WORK`."
    )
    return "\n".join(lines) + "\n"


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
    from transformers.cache_utils import HybridCache
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_root)
    t0 = time.time()
    report_at = t0 + float(cfg["wall_clock_sec"]) - float(cfg["report_reserve_sec"])
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    state = {"log": [], "calibration": True}

    def note(msg):
        state["log"].append({"t": round(time.time() - t0, 1), "msg": msg})
        print(msg, flush=True)

    def late():
        return time.time() >= report_at

    try:
        os.environ.setdefault("HF_HOME", str(work / "hf"))
        source_art = work / cfg["source"] / "artifacts"
        parent_art = work / cfg["sentiment_parent"] / "artifacts"
        v_q = np.load(parent_art / "vec_qwen_averaged.npy").astype(np.float64)
        maps = np.load(source_art / "maps.npz")
        eq = np.asarray(maps["E_qwen"], dtype=np.float64)
        eo = np.asarray(maps["E_olmo"], dtype=np.float64)
        geometry = json.loads((source_art / "geometry.json").read_text())
        saved_transfer = np.load(source_art / "dir_olmo_aligned64.npy").astype(np.float64)
        parent_state = json.loads((parent_art / "run_state.json").read_text())
        olmo_dose = float(parent_state["results"]["olmo"]["dose"])
        z_edit = eq @ v_q
        raw_olmo = np.linalg.pinv(eo) @ z_edit
        raw_norm = float(np.linalg.norm(raw_olmo))
        prepared = raw_olmo / raw_norm * olmo_dose
        provenance = {
            "z_norm": float(np.linalg.norm(z_edit)),
            "raw_norm": raw_norm,
            "recorded_pre_norm": float(geometry["transfer"]["aligned64"]["pre_norm"]),
            "max_abs_after_dose": float(np.max(np.abs(prepared - saved_transfer))),
            "olmo_dose": olmo_dose,
        }
        hashes = {
            "vec_qwen": sha256_file(parent_art / "vec_qwen_averaged.npy"),
            "maps": sha256_file(source_art / "maps.npz"),
            "dir_olmo_aligned64": sha256_file(source_art / "dir_olmo_aligned64.npy"),
            "geometry": sha256_file(source_art / "geometry.json"),
        }
        ok = provenance["max_abs_after_dose"] < 1e-6 and abs(provenance["raw_norm"] - provenance["recorded_pre_norm"]) < 1e-6
        save_json(art / "frozen_command.json", {"provenance": provenance, "ok": ok, "sha256": hashes, "z_edit": z_edit})
        np.save(art / "z_edit.npy", z_edit)
        note(f"provenance ok={ok} max_abs={provenance['max_abs_after_dose']:.3e}")
        if not ok:
            raise RuntimeError("decoded z_edit does not reproduce the archived OLMo direction")

        # Exclusions. Behavioural panels, then this run's own cal/val later.
        excluded = set()
        prior = json.loads((work / cfg["prior_actadd_splits"]).read_text())
        actadd_ids = set()
        for rows in prior.values():
            for row in rows:
                if isinstance(row, dict) and "review_id" in row:
                    actadd_ids.add(row["review_id"])
                    excluded.add(row["review_id"])
        for name in ("qwen", "olmo"):
            for line in (parent_art / f"confirmation_{name}.jsonl").open():
                excluded.add(json.loads(line)["review_id"])
        cca_manifest = json.loads((source_art / "cca_manifest.json").read_text())
        excluded.update(cca_manifest["fit_review_ids"])
        excluded.update(cca_manifest["rank_review_ids"])
        source_panel = json.loads((source_art / "panel.json").read_text())
        excluded.update(source_panel["review_ids"])
        retention_manifest = json.loads((work / cfg["retention_run"] / "artifacts/panel_manifest.json").read_text())
        excluded.update(retention_manifest["review_ids"])
        excluded.update(retention_manifest.get("timing_review_ids") or [])

        ds = load_dataset("stanfordnlp/imdb")
        test_neg, train = [], []
        for split, bucket in (("test", test_neg), ("train", train)):
            for i, row in enumerate(ds[split]):
                bucket.append({"review_id": f"{split}:{i}", "label": int(row["label"]), "text": row["text"]})
        test_neg = [r for r in test_neg if r["label"] == 0]
        parent_pool = [r for r in test_neg if r["review_id"] not in actadd_ids]
        np.random.default_rng(int(cfg["parent_split_seed"])).shuffle(parent_pool)
        for row in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
            excluded.add(row["review_id"])
        source_excluded = set(actadd_ids)
        for name in ("qwen", "olmo"):
            for line in (parent_art / f"confirmation_{name}.jsonl").open():
                source_excluded.add(json.loads(line)["review_id"])
        for row in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
            source_excluded.add(row["review_id"])
        source_fresh = [r for r in test_neg if r["review_id"] not in source_excluded]
        np.random.default_rng(int(cfg["source_panel_seed"])).shuffle(source_fresh)
        if [r["review_id"] for r in source_fresh[1:1 + len(source_panel["review_ids"])]] != source_panel["review_ids"]:
            raise RuntimeError("could not reconstruct the CCA transfer panel")
        excluded.add(source_fresh[0]["review_id"])
        eligible = [r for r in test_neg if r["review_id"] not in excluded]
        np.random.default_rng(int(cfg["native_seed"])).shuffle(eligible)
        cal_rows = eligible[: int(cfg["n_calibration"])]
        val_rows = eligible[int(cfg["n_calibration"]): int(cfg["n_calibration"]) + int(cfg["n_validation"])]
        for row in cal_rows + val_rows:
            excluded.add(row["review_id"])
        save_json(art / "native_splits.json", {
            "calibration": [r["review_id"] for r in cal_rows],
            "validation": [r["review_id"] for r in val_rows],
            "tie_break": cfg["gemma"]["tie_break"],
            "frozen_before_native_scores": True,
        })
        note(f"excluded {len(excluded)} ids; cal {len(cal_rows)} val {len(val_rows)}")

        siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
        scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)

        def as_scores(raw_scores):
            if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
                return raw_scores[0]
            return raw_scores

        def pos_of(text):
            table = {d["label"].upper(): float(d["score"]) for d in as_scores(scorer((text or " ")[:1500]))}
            p = table.get("POSITIVE", 0.0)
            return p, p >= table.get("NEGATIVE", 0.0)

        if not (pos_of("This movie was wonderful and I loved every minute of it.")[1] and not pos_of("This movie was awful and I hated every minute of it.")[1]):
            raise SystemExit("SiEBERT check failed")

        snap = work / cfg["gemma"]["snapshot"]
        gtok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        gmodel = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        glayers = {int(i): gmodel.model.layers[int(i)] for i in cfg["gemma"]["layers"]}
        pairs = json.loads((work / cfg["pairs_file"]).read_text())["pairs"]
        sample = cfg["sample"]
        max_new = int(sample["max_new_tokens"])
        note("Gemma loaded")

        def render_eval(tok, text):
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
            return tok.decode(ids, skip_special_tokens=True).rstrip() + cfg["eval_bridge"]

        def last_vector(model, tok, block, text):
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            box = {}

            def hook(_m, inputs):
                box["h"] = inputs[0].detach()
                return None

            handle = block.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    model(**enc)
            finally:
                handle.remove()
            return box["h"][0, -1].float().cpu().numpy(), int(enc["input_ids"][0, -1])

        vectors = {}
        pair_checks = {}
        for layer, block in glayers.items():
            diffs, mismatches = [], []
            for pair in pairs:
                if late():
                    break
                hp, ip = last_vector(gmodel, gtok, block, pair["positive"] + pair["bridge"])
                hn, inn = last_vector(gmodel, gtok, block, pair["negative"] + pair["bridge"])
                if ip != inn:
                    mismatches.append(pair["pair_id"])
                    continue
                diffs.append(hp - hn)
            diffs_a = np.stack(diffs)
            direction, raw_mean_norm = unit(diffs_a.mean(0))
            vectors[layer] = direction
            pair_checks[layer] = {"n_used": len(diffs), "mismatched_pairs": mismatches, "mean_raw_norm": raw_mean_norm}
            np.save(art / f"vec_gemma_l{layer}.npy", direction)
            note(f"layer {layer} pairs {len(diffs)} mismatches {len(mismatches)}")
        save_json(art / "pair_checks.json", pair_checks)
        if any(info["n_used"] < len(pairs) for info in pair_checks.values()):
            raise RuntimeError("Gemma did not share the neutral-suffix token on every matched pair")

        def generate(text, block, direction, seed):
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            enc = gtok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            ids = enc["input_ids"]
            cache = HybridCache(config=gmodel.config, max_batch_size=1, max_cache_len=int(ids.shape[1] + max_new), device="cuda", dtype=torch.bfloat16)
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
                    return (out_h,)

                handle = block.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    fwd = gmodel(ids, past_key_values=cache, use_cache=True)
                    new_ids = []
                    for _ in range(max_new):
                        nxt = sample_next(fwd.logits[0, -1], ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
                        new_ids.append(int(nxt.item()))
                        if gtok.eos_token_id is not None and int(nxt.item()) == int(gtok.eos_token_id):
                            break
                        pos = torch.tensor([ids.shape[1]], device="cuda")
                        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                        fwd = gmodel(nxt.view(1, 1), past_key_values=cache, use_cache=True, cache_position=pos)
            finally:
                if handle is not None:
                    handle.remove()
            return new_ids, gtok.decode(new_ids, skip_special_tokens=True), realised

        def score_rows(rows, block, direction, seed_fn):
            got = []
            for row in rows:
                if late():
                    break
                text = render_eval(gtok, row["text"])
                t_g = time.time()
                new_ids, cont, realised = generate(text, block, direction, seed_fn(row))
                p, y = pos_of(cont)
                got.append({
                    "review_id": row["review_id"], "prefix": text, "continuation": cont, "n_new": len(new_ids),
                    "sec": time.time() - t_g, "positive_score": p, "positive": bool(y), "realised_norm": realised,
                    **quality(cont, len(new_ids), max_new),
                })
            return got

        medians = {}
        for layer, block in glayers.items():
            norms = []
            for row in cal_rows:
                h, _ = last_vector(gmodel, gtok, block, render_eval(gtok, row["text"]))
                norms.append(float(np.linalg.norm(h)))
            medians[layer] = float(np.median(norms))
        unedited = score_rows(cal_rows, glayers[int(cfg["gemma"]["layers"][0])], None, lambda row: review_seed(row["review_id"]))
        su = summarize(unedited)
        curve = []
        for layer in cfg["gemma"]["layers"]:
            for frac in cfg["dose_fractions"]:
                if late() or len(unedited) < len(cal_rows):
                    break
                dose = float(frac) * medians[int(layer)]
                edited = score_rows(cal_rows, glayers[int(layer)], vectors[int(layer)] * dose, lambda row: review_seed(row["review_id"]))
                se = summarize(edited)
                shift = se.get("mean_score", 0.0) - su.get("mean_score", 0.0)
                empty_ok = se.get("empty", 1) <= su.get("empty", 0)
                rep_ok = (se.get("repetitive", 1) / max(se.get("n", 1), 1)) <= (su.get("repetitive", 0) / max(su.get("n", 1), 1)) + float(cfg["repetition_rate_allowance"])
                passes = bool(se.get("n") == su.get("n") and shift > 0 and empty_ok and rep_ok)
                cell = {"layer": int(layer), "fraction": float(frac), "dose": dose, "median_residual_norm": medians[int(layer)], "edit": se, "shift": shift, "passes": passes}
                curve.append(cell)
                note(f"cal L{layer} x{frac} shift {shift:+.3f} pass {passes}")
        passing = [c for c in curve if c["passes"]]
        passing.sort(key=lambda c: (c["fraction"], c["layer"]))
        chosen = passing[0] if passing else None
        save_json(art / "calibration.json", {"unedited": su, "cells": curve, "chosen": chosen, "tie_break": cfg["gemma"]["tie_break"]})
        state["chosen"] = chosen
        if chosen is None:
            state["stopped"] = "no calibration cell had a positive score shift without a readability breakdown"
            note(state["stopped"])
        else:
            block = glayers[int(chosen["layer"])]
            direction = vectors[int(chosen["layer"])] * float(chosen["dose"])
            val = {
                "unedited": summarize(score_rows(val_rows, block, None, lambda row: review_seed(row["review_id"]))),
                "native": summarize(score_rows(val_rows, block, direction, lambda row: review_seed(row["review_id"]))),
                "reversed": summarize(score_rows(val_rows, block, -direction, lambda row: review_seed(row["review_id"]))),
            }
            save_json(art / "validation.json", val)
            usable = val["native"].get("n") == len(val_rows) and val["native"]["mean_score"] > val["unedited"]["mean_score"]
            note(f"validation native {val['native'].get('mean_score')} unedited {val['unedited'].get('mean_score')} usable {usable}")
            if not usable:
                state["stopped"] = "validation did not reproduce a positive native score shift"
            else:
                state["native_dose"] = float(chosen["dose"])
                state["native_layer"] = int(chosen["layer"])
                # Bridge texts.
                usable_train = []
                for row in train:
                    if row["review_id"] in excluded:
                        continue
                    made = shared_passage(row["text"], int(cfg["prompt_chars"]), int(cfg["continuation_chars"]))
                    if made is None:
                        continue
                    usable_train.append({**row, "full": made[0], "offs": made[1]})
                pos = [r for r in usable_train if r["label"] == 1]
                neg = [r for r in usable_train if r["label"] == 0]
                rng = np.random.default_rng(int(cfg["bridge_seed"]))
                rng.shuffle(pos)
                rng.shuffle(neg)
                half_f, half_b = int(cfg["fit_n"]) // 2, int(cfg["bridge_n"]) // 2
                fit_rows = pos[:half_f] + neg[:half_f]
                bridge_rows = pos[half_f:half_f + half_b] + neg[half_f:half_f + half_b]
                rng.shuffle(fit_rows)
                rng.shuffle(bridge_rows)
                note(f"bridge pool fit {len(fit_rows)} val {len(bridge_rows)}")
                olmo_snap = work / "hf/hub/models--allenai--OLMo-7B-0724-hf/snapshots" / cfg["olmo"]["revision"]
                otok = AutoTokenizer.from_pretrained(str(olmo_snap), local_files_only=True)
                omodel = AutoModelForCausalLM.from_pretrained(str(olmo_snap), local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
                oblock = omodel.model.layers[int(cfg["olmo"]["layer"])]

                def positions(model, tok, block, rows):
                    out = {}
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
                        out[row["review_id"]] = np.stack([h[i] for i in idxs])
                    return out

                note("extract bridge activations")
                o_fit = positions(omodel, otok, oblock, fit_rows)
                g_fit = positions(gmodel, gtok, block, fit_rows)
                o_val = positions(omodel, otok, oblock, bridge_rows)
                g_val = positions(gmodel, gtok, block, bridge_rows)
                demo = next(r for r in fit_rows if r["review_id"] in o_fit and r["review_id"] in g_fit)
                boundary = {}
                for name, tok in (("olmo", otok), ("gemma", gtok)):
                    enc = tok(demo["full"], return_offsets_mapping=True, add_special_tokens=False)
                    picks = []
                    for c in demo["offs"]:
                        i = token_for_offset(enc["offset_mapping"], c)
                        picks.append({"char": c, "token_index": i, "piece": tok.decode([enc["input_ids"][i]]) if i is not None else None})
                    boundary[name] = picks
                save_json(art / "boundary_smoke.json", boundary)
                del omodel
                torch.cuda.empty_cache()

                def stack(omap, gmap, rows):
                    ids = [r["review_id"] for r in rows if r["review_id"] in omap and r["review_id"] in gmap]
                    if not ids:
                        raise RuntimeError("no paired bridge rows")
                    return np.concatenate([omap[i] for i in ids]), np.concatenate([gmap[i] for i in ids]), ids

                Xo_fit, Xg_fit, fit_ids = stack(o_fit, g_fit, fit_rows)
                Xo_val, Xg_val, val_ids = stack(o_val, g_val, bridge_rows)
                mu = Xo_fit.mean(axis=0)
                z_fit = (Xo_fit - mu) @ eo.T
                z_val = (Xo_val - mu) @ eo.T
                z_mean = z_fit.mean(axis=0)
                z_sd = np.maximum(z_fit.std(axis=0, ddof=1), 1e-8)
                h_mean = Xg_fit.mean(axis=0)
                zs_fit = (z_fit - z_mean) / z_sd
                zs_val = (z_val - z_mean) / z_sd
                hc_fit = Xg_fit - h_mean
                # Review-equal error: average rows within a review, then across reviews.
                fit_review = np.repeat(np.asarray(fit_ids), Xo_fit.shape[0] // len(fit_ids))
                val_review = np.repeat(np.asarray(val_ids), Xo_val.shape[0] // len(val_ids))

                def review_mse(pred, target, reviews):
                    err = np.mean((pred - target) ** 2, axis=1)
                    return float(np.mean([err[reviews == rid].mean() for rid in dict.fromkeys(reviews)]))

                base_pred = np.repeat(h_mean.reshape(1, -1), len(Xg_val), axis=0)
                baseline = review_mse(base_pred, Xg_val, val_review)
                best = None
                alpha_rows = []
                gram = zs_fit.T @ zs_fit
                rhs = zs_fit.T @ hc_fit
                for mult in cfg["alpha_multipliers"]:
                    alpha = float(mult)
                    slope = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)  # (k, d)
                    pred = h_mean + zs_val @ slope
                    mse = review_mse(pred, Xg_val, val_review)
                    alpha_rows.append({"multiplier": float(mult), "alpha": alpha, "val_mse": mse})
                    if best is None or mse < best["val_mse"]:
                        best = {"multiplier": float(mult), "alpha": alpha, "val_mse": mse, "slope": slope}
                D = (best["slope"].T / z_sd.reshape(1, -1))
                intercept = h_mean - (z_mean / z_sd) @ best["slope"]
                v_raw = D @ z_edit
                v_norm = float(np.linalg.norm(v_raw))
                near_zero = v_norm < float(cfg["near_zero_norm"])
                if near_zero:
                    transfer = np.zeros_like(v_raw)
                    applied = 0.0
                else:
                    transfer = v_raw / v_norm * float(chosen["dose"])
                    applied = float(chosen["dose"])
                np.savez(art / "decoder.npz", D=D, intercept=intercept, mu_olmo=mu, z_mean=z_mean, z_sd=z_sd, h_mean=h_mean, alpha=best["alpha"])
                np.save(art / "dir_gemma_native.npy", direction)
                np.save(art / "dir_gemma_transfer.npy", transfer)
                decoder_info = {
                    "n_fit_reviews": len(fit_ids), "n_bridge_reviews": len(val_ids),
                    "rows_per_review": int(Xo_fit.shape[0] // len(fit_ids)),
                    "alpha": best["alpha"], "val_mse": best["val_mse"], "baseline_mse": baseline,
                    "mse_ratio_vs_mean": best["val_mse"] / baseline if baseline else None,
                    "alphas": alpha_rows, "raw_norm": v_norm, "near_zero": near_zero, "applied_norm": applied,
                    "gemma_seen_in_decoder_fit": True, "qwen_command_frozen": True, "cca_refit": False,
                }
                state["decoder"] = {k: v for k, v in decoder_info.items() if k != "alphas"}
                save_json(art / "decoder.json", decoder_info)
                save_json(art / "bridge_manifest.json", {"fit_review_ids": fit_ids, "bridge_review_ids": val_ids})
                note(f"decoder alpha {best['alpha']} mse {best['val_mse']:.4f} baseline {baseline:.4f} raw {v_norm:.4f}")

                # Smoke, then freeze the test panel before scoring it.
                smoke_row = cal_rows[0]
                smoke_text = render_eval(gtok, smoke_row["text"])
                enc = gtok(smoke_text, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    base_logits = gmodel(**enc).logits

                def zero_hook(_m, inputs):
                    return (inputs[0].clone(),)

                handle = block.register_forward_pre_hook(zero_hook)
                try:
                    with torch.no_grad():
                        zero_logits = gmodel(**enc).logits
                finally:
                    handle.remove()
                seed_a = generate(smoke_text, block, None, 12345)[0]
                seed_b = generate(smoke_text, block, None, 12345)[0]
                _, _, native_realised = generate(smoke_text, block, direction, 0)
                _, _, transfer_realised = generate(smoke_text, block, transfer, 0)
                smoke = {
                    "zero_logit_max_abs": float(torch.max(torch.abs(base_logits.float() - zero_logits.float())).cpu()),
                    "seed_repeat_identical": seed_a == seed_b,
                    "native_realised": native_realised,
                    "transfer_realised": transfer_realised,
                    "target_norm": applied if near_zero else float(chosen["dose"]),
                    "boundary_same_chars": [a["char"] for a in boundary["olmo"]] == [a["char"] for a in boundary["gemma"]],
                    "boundary_tokens_found": all(a["token_index"] is not None and b["token_index"] is not None for a, b in zip(boundary["olmo"], boundary["gemma"])),
                }
                save_json(art / "smoke.json", smoke)
                note(f"smoke {smoke}")
                if smoke["zero_logit_max_abs"] > 1e-3 or not smoke["seed_repeat_identical"] or not smoke["boundary_tokens_found"]:
                    raise RuntimeError("smoke failed")
                for label, realised in (("native", native_realised), ("transfer", transfer_realised)):
                    if label == "transfer" and near_zero:
                        continue
                    for key in ("prefill", "decode"):
                        got = float(realised.get(key, 0.0))
                        if abs(got - float(chosen["dose"])) / float(chosen["dose"]) > 0.02:
                            raise RuntimeError(f"{label} {key} norm {got}")

                pool = [r for r in test_neg if r["review_id"] not in excluded]
                np.random.default_rng(int(cfg["panel_seed"])).shuffle(pool)
                timers = pool[: int(cfg["smoke_blocks"])]
                t_batch = time.time()
                for row in timers:
                    text = render_eval(gtok, row["text"])
                    seed = review_seed(row["review_id"])
                    for arm_dir in (None, direction, transfer):
                        generate(text, block, arm_dir, seed)
                sec_block = (time.time() - t_batch) / max(1, len(timers))
                remain = max(0.0, report_at - time.time())
                n_panel = min(int(cfg["target_panel"]), int(remain * 0.92 / sec_block)) if sec_block > 0 else 0
                chosen_rows = pool[len(timers): len(timers) + max(n_panel, 0)]
                frozen = {
                    "n": len(chosen_rows), "sec_per_block": sec_block, "panel_seed": int(cfg["panel_seed"]),
                    "review_ids": [r["review_id"] for r in chosen_rows],
                    "seeds": {r["review_id"]: review_seed(r["review_id"]) for r in chosen_rows},
                    "timing_review_ids": [r["review_id"] for r in timers],
                    "frozen_before_test_scores": True,
                }
                save_json(art / "panel_manifest.json", frozen)
                note(f"froze panel n={frozen['n']} sec/block {sec_block:.2f}")
                arms = {"unedited": None, "native": direction, "transfer": transfer}
                ckpt = art / "confirmation.jsonl"
                blocks = []
                for row in chosen_rows:
                    if late():
                        break
                    text = render_eval(gtok, row["text"])
                    seed = review_seed(row["review_id"])
                    rec = {"review_id": row["review_id"], "seed": seed}
                    for arm, arm_dir in arms.items():
                        t_g = time.time()
                        new_ids, cont, realised = generate(text, block, arm_dir, seed)
                        p, y = pos_of(cont)
                        rec[arm] = {
                            "review_id": row["review_id"], "prefix": text, "continuation": cont, "n_new": len(new_ids),
                            "sec": time.time() - t_g, "positive_score": p, "positive": bool(y), "realised_norm": realised,
                            "seed": seed, **quality(cont, len(new_ids), max_new),
                        }
                    blocks.append(rec)
                    with ckpt.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    if len(blocks) % 10 == 0:
                        note(f"block {len(blocks)}/{frozen['n']}")
                state["n_blocks"] = len(blocks)
                state["frozen_n"] = frozen["n"]
                state["partial"] = len(blocks) < frozen["n"]
                state["summary"] = {arm: summarize([b[arm] for b in blocks]) for arm in arms}
                if blocks:
                    native_gain, transfer_gain, absolute_gap, retention = [], [], [], []
                    flips = {k: [0, 0] for k in ("native_vs_unedited", "transfer_vs_unedited", "transfer_vs_native")}
                    fraction = float(cfg["retention_fraction"])
                    scores = []
                    for b in blocks:
                        u, n, t = b["unedited"], b["native"], b["transfer"]
                        ng = float(n["positive_score"]) - float(u["positive_score"])
                        tg = float(t["positive_score"]) - float(u["positive_score"])
                        native_gain.append(ng)
                        transfer_gain.append(tg)
                        absolute_gap.append(float(t["positive_score"]) - float(n["positive_score"]))
                        retention.append(tg - fraction * ng)
                        for key, a, c in (("native_vs_unedited", u, n), ("transfer_vs_unedited", u, t), ("transfer_vs_native", n, t)):
                            if (not a["positive"]) and c["positive"]:
                                flips[key][0] += 1
                            if a["positive"] and (not c["positive"]):
                                flips[key][1] += 1
                        scores.append({
                            "review_id": b["review_id"], "seed": b["seed"],
                            "U": u["positive_score"], "N": n["positive_score"], "T": t["positive_score"],
                            "native_gain": ng, "transfer_gain": tg, "absolute_gap": float(t["positive_score"]) - float(n["positive_score"]),
                            "retention_80": tg - fraction * ng,
                        })
                    n_boot = int(cfg["n_boot"])
                    state["test"] = {
                        "native_gain": boot_mean(native_gain, n_boot),
                        "transfer_gain": boot_mean(transfer_gain, n_boot),
                        "absolute_gap": boot_mean(absolute_gap, n_boot),
                        "retention_80": boot_mean(retention, n_boot),
                        "flips": flips,
                    }
                    save_json(art / "scores.json", scores)
                    save_json(art / "test_retention.json", state["test"])
                    blind, key = [], []
                    brng = np.random.default_rng(0)
                    for b in blocks[: int(cfg["blind_n"])]:
                        order = ["unedited", "native", "transfer"]
                        brng.shuffle(order)
                        for arm in order:
                            code = hashlib.sha256(f"{b['review_id']}:{arm}".encode()).hexdigest()[:8]
                            blind.append({"code": code, "prefix": b[arm]["prefix"], "continuation": b[arm]["continuation"]})
                            key.append({"code": code, "arm": arm, "review_id": b["review_id"]})
                    save_json(art / "blind_sample.json", blind)
                    save_json(art / "blind_key.json", key)
                note(f"done blocks {len(blocks)} partial {state['partial']}")
        del gmodel
        torch.cuda.empty_cache()
    except Exception:
        state["error"] = traceback.format_exc()
        note(state["error"])
    state["elapsed_sec"] = time.time() - t0
    save_json(art / "run_state.json", state)
    (dest / "RESULTS.md").write_text(render_md(state))
    print("wrote RESULTS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
