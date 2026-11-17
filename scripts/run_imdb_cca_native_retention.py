#!/usr/bin/env python3
"""Does the frozen top-64 Qwen→OLMo edit retain 80% of OLMo's native gain?"""

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


def bootstrap_values(values, n_boot, seed):
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
        "seed": int(seed),
    }


def retention_rows(records, fraction):
    """records: list of {review_id, unedited, native, transfer} score dicts."""
    native_gain, transfer_gain, absolute_gap, retention = [], [], [], []
    flips = {
        "native_vs_unedited": [0, 0],
        "transfer_vs_unedited": [0, 0],
        "transfer_vs_native": [0, 0],
    }
    rate_native, rate_transfer, rate_gap = [], [], []
    for rec in records:
        u, n, t = rec["unedited"], rec["native"], rec["transfer"]
        ng = float(n["positive_score"]) - float(u["positive_score"])
        tg = float(t["positive_score"]) - float(u["positive_score"])
        native_gain.append(ng)
        transfer_gain.append(tg)
        absolute_gap.append(float(t["positive_score"]) - float(n["positive_score"]))
        retention.append(tg - float(fraction) * ng)
        rate_native.append(float(n["positive"]) - float(u["positive"]))
        rate_transfer.append(float(t["positive"]) - float(u["positive"]))
        rate_gap.append(float(t["positive"]) - float(n["positive"]))
        for key, a, b in (
            ("native_vs_unedited", u, n),
            ("transfer_vs_unedited", u, t),
            ("transfer_vs_native", n, t),
        ):
            if (not a["positive"]) and b["positive"]:
                flips[key][0] += 1
            if a["positive"] and (not b["positive"]):
                flips[key][1] += 1
    return {
        "native_gain": native_gain,
        "transfer_gain": transfer_gain,
        "absolute_gap": absolute_gap,
        "retention_80": retention,
        "rate_native_gain": rate_native,
        "rate_transfer_gain": rate_transfer,
        "rate_absolute_gap": rate_gap,
        "flips": {k: {"to_positive": a, "to_negative": b} for k, (a, b) in flips.items()},
    }


def summarise_retention(raw, n_boot, seed):
    out = {}
    for key in ("native_gain", "transfer_gain", "absolute_gap", "retention_80", "rate_native_gain", "rate_transfer_gain", "rate_absolute_gap"):
        out[key] = bootstrap_values(raw[key], n_boot, seed)
    out["flips"] = raw["flips"]
    ng = out["native_gain"].get("lower95")
    ret = out["retention_80"].get("lower95")
    out["native_established"] = bool(ng is not None and ng > 0)
    out["retention_established"] = bool(out["native_established"] and ret is not None and ret > 0)
    return out


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}], one-sided lower bound {block['lower95']:+.3f}"


def decision_text(stats):
    if not stats or "native_gain" not in stats or "mean" not in stats["native_gain"]:
        return "The new panel was not scored."
    ng, tg, gap, ret = stats["native_gain"], stats["transfer_gain"], stats["absolute_gap"], stats["retention_80"]
    if stats["retention_established"]:
        verdict = "Yes. The one-sided 95% lower bounds for native gain and for the 80% retention contrast are both above zero."
    elif stats["native_established"]:
        shortfall = max(0.0, -float(ret["lower95"]))
        verdict = (
            "No. Native gain is established, and the 80% retention contrast is not. "
            f"Its one-sided lower bound is {ret['lower95']:+.3f}, so the bound still allows the transfer gain to fall short of 80% of the native gain by {shortfall:.3f} score points. "
            f"The absolute gap versus native is {gap['mean']:+.3f} [{gap['low']:+.3f}, {gap['high']:+.3f}]; an interval that includes zero is not evidence that the transfer retained the native edit."
        )
    else:
        verdict = (
            "Native gain is unresolved on this panel, so the transfer effect is not expressed as a percentage of it. "
            f"Native gain {fmt(ng)}. Transfer gain {fmt(tg)}. Absolute gap {fmt(gap)}."
        )
    return (
        f"{verdict} Native gain {fmt(ng)}. Transfer gain {fmt(tg)}. "
        f"Absolute gap {fmt(gap)}. Retention contrast (T−U)−0.8(N−U) {fmt(ret)}."
    )


def render_md(state):
    lines = ["# RESULTS — `imdb_cca_native_retention_v1`", ""]
    new = state.get("test") or {}
    partial = bool(state.get("partial"))
    lines.append("Did transfer keep up with native at the frozen 80% margin? " + decision_text(new))
    if partial:
        lines.append(
            f"This analysis is partial: {state.get('n_blocks', 0)} of {state.get('frozen_n', 0)} frozen reviews were scored before the time cap."
        )
    elif state.get("n_blocks"):
        lines.append(f"Coverage is the full frozen panel, n={state.get('n_blocks')}.")
    lines.append("")
    pilot = state.get("pilot") or {}
    if pilot.get("native_gain"):
        lines.append(
            "Saved 40-review pilot, recomputed separately and not pooled: native gain "
            + fmt(pilot["native_gain"])
            + ". Transfer gain "
            + fmt(pilot["transfer_gain"])
            + ". Absolute gap "
            + fmt(pilot["absolute_gap"])
            + ". Retention contrast "
            + fmt(pilot["retention_80"])
            + "."
        )
        lines.append("")
    lines.append("Usability is provisional until blinded ratings exist. Empty outputs, repetition, and truncation are separate from the sentiment contrast.")
    lines.append("")
    summary = state.get("summary") or {}
    if summary:
        lines.append("| arm | n | positive | score | empty | repetitive | truncated |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in ("unedited", "native", "transfer"):
            r = summary.get(arm) or {}
            if not r.get("n"):
                continue
            lines.append(f"| {arm} | {r['n']} | {r['positive']} | {r['mean_score']:.3f} | {r['empty']} | {r['repetitive']} | {r['truncated']} |")
        lines.append("")
    flips = (new.get("flips") or {})
    if flips:
        lines.append(
            "Secondary flips (to positive / to negative): "
            f"native vs unedited {flips['native_vs_unedited']['to_positive']}/{flips['native_vs_unedited']['to_negative']}; "
            f"transfer vs unedited {flips['transfer_vs_unedited']['to_positive']}/{flips['transfer_vs_unedited']['to_negative']}; "
            f"transfer vs native {flips['transfer_vs_native']['to_positive']}/{flips['transfer_vs_native']['to_negative']}."
        )
        lines.append("")
    if state.get("verification"):
        lines.append("Frozen-vector check: `" + json.dumps(state["verification"].get("match"), default=str) + "`")
        lines.append("")
    if state.get("error"):
        lines += ["## Stop", "", "```", state["error"][-1500:], "```", ""]
    lines.append(
        f"Elapsed {state.get('elapsed_sec', 0):.0f}s. Reproduction: "
        "`python scripts/run_imdb_cca_native_retention.py --config configs/imdb_cca_native_retention_v1.json --work-root $WORK`."
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
        source = work / cfg["source"]
        source_art = source / "artifacts"
        parent_art = work / cfg["sentiment_parent"] / "artifacts"
        fraction = float(cfg["retention_fraction"])
        v_q = np.load(parent_art / "vec_qwen_averaged.npy").astype(np.float64)
        v_o = np.load(parent_art / "vec_olmo_averaged.npy").astype(np.float64)
        saved_native = np.load(source_art / "dir_olmo_native.npy").astype(np.float64)
        saved_transfer = np.load(source_art / "dir_olmo_aligned64.npy").astype(np.float64)
        maps = np.load(source_art / "maps.npz")
        eq = np.asarray(maps["E_qwen"], dtype=np.float64)
        eo = np.asarray(maps["E_olmo"], dtype=np.float64)
        geometry = json.loads((source_art / "geometry.json").read_text())
        parent_state = json.loads((parent_art / "run_state.json").read_text())
        dose = float(parent_state["results"]["olmo"]["dose"])
        raw = np.linalg.pinv(eo[:64]) @ (eq[:64] @ v_q)
        raw_norm = float(np.linalg.norm(raw))
        prepared = raw / raw_norm * dose
        diag = geometry["transfer"]["aligned64"]
        native_expected = v_o * dose
        match = {
            "dose": dose,
            "saved_native_norm": float(np.linalg.norm(saved_native)),
            "saved_transfer_norm": float(np.linalg.norm(saved_transfer)),
            "reconstructed_pre_norm": raw_norm,
            "recorded_pre_norm": float(diag["pre_norm"]),
            "recorded_cosine": float(diag["cosine"]),
            "reconstructed_cosine_to_native_unit": float(np.dot(raw, v_o) / (raw_norm * np.linalg.norm(v_o))),
            "native_max_abs": float(np.max(np.abs(saved_native - native_expected))),
            "transfer_max_abs": float(np.max(np.abs(saved_transfer - prepared))),
            "near_zero_recorded": bool(diag["near_zero"]),
        }
        ok = (
            match["native_max_abs"] < 1e-6
            and match["transfer_max_abs"] < 1e-6
            and abs(match["reconstructed_pre_norm"] - match["recorded_pre_norm"]) < 1e-6
            and abs(match["saved_native_norm"] - dose) < 1e-4
            and abs(match["saved_transfer_norm"] - dose) < 1e-4
            and not match["near_zero_recorded"]
        )
        hashes = {
            name: sha256_file(path)
            for name, path in {
                "vec_qwen_averaged": parent_art / "vec_qwen_averaged.npy",
                "vec_olmo_averaged": parent_art / "vec_olmo_averaged.npy",
                "dir_olmo_native": source_art / "dir_olmo_native.npy",
                "dir_olmo_aligned64": source_art / "dir_olmo_aligned64.npy",
                "maps": source_art / "maps.npz",
                "geometry": source_art / "geometry.json",
            }.items()
        }
        verification = {"match": match, "ok": ok, "sha256": hashes}
        state["verification"] = verification
        save_json(art / "verification.json", verification)
        note(f"verification ok={ok} transfer_max_abs={match['transfer_max_abs']:.3e} pre_norm={raw_norm:.6f}")
        if not ok:
            raise RuntimeError("frozen transfer direction does not match the archived vector")
        np.save(art / "dir_olmo_native.npy", saved_native)
        np.save(art / "dir_olmo_transfer.npy", saved_transfer)

        # Pilot, from the saved generations only.
        pilot_blocks = []
        for line in (source_art / "confirmation.jsonl").open():
            row = json.loads(line)
            pilot_blocks.append({
                "review_id": row["review_id"],
                "unedited": row["olmo:unedited"],
                "native": row["olmo:native"],
                "transfer": row["olmo:aligned64"],
            })
        pilot = summarise_retention(retention_rows(pilot_blocks, fraction), int(cfg["pilot_boot"]), 0)
        archived = json.loads((source_art / "run_state.json").read_text())["contrasts"]["olmo"]["transfer_minus_0.8_native"]
        pilot["archived_contrast"] = archived
        pilot["matches_archived_mean"] = abs(pilot["retention_80"]["mean"] - float(archived["mean"])) < 1e-12
        pilot["matches_archived_interval"] = (
            abs(pilot["retention_80"]["low"] - float(archived["low"])) < 1e-12
            and abs(pilot["retention_80"]["high"] - float(archived["high"])) < 1e-12
        )
        state["pilot"] = pilot
        save_json(art / "pilot_retention.json", pilot)
        note(
            f"pilot n={pilot['native_gain']['n_reviews']} retention {pilot['retention_80']['mean']:+.4f} "
            f"archived_match={pilot['matches_archived_mean'] and pilot['matches_archived_interval']}"
        )

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
        manifest = json.loads((source_art / "cca_manifest.json").read_text())
        excluded.update(manifest["fit_review_ids"])
        excluded.update(manifest["rank_review_ids"])
        source_panel = json.loads((source_art / "panel.json").read_text())
        excluded.update(source_panel["review_ids"])

        ds = load_dataset("stanfordnlp/imdb")
        test_neg = []
        for i, row in enumerate(ds["test"]):
            if int(row["label"]) != 0:
                continue
            test_neg.append({"review_id": f"test:{i}", "text": row["text"]})
        parent_pool = [r for r in test_neg if r["review_id"] not in actadd_ids]
        np.random.default_rng(int(cfg["parent_split_seed"])).shuffle(parent_pool)
        for r in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
            excluded.add(r["review_id"])
        # Reconstruct the source panel from the exclusion set that run used.
        source_excluded = set(actadd_ids)
        for name in ("qwen", "olmo"):
            for line in (parent_art / f"confirmation_{name}.jsonl").open():
                source_excluded.add(json.loads(line)["review_id"])
        for r in parent_pool[: int(cfg["parent_n_cal"]) + int(cfg["parent_n_confirm"])]:
            source_excluded.add(r["review_id"])
        source_fresh = [r for r in test_neg if r["review_id"] not in source_excluded]
        np.random.default_rng(int(cfg["source_panel_seed"])).shuffle(source_fresh)
        if [r["review_id"] for r in source_fresh[1:1 + len(source_panel["review_ids"])]] != source_panel["review_ids"]:
            raise RuntimeError("could not reconstruct the source panel; refusing to guess the timing review")
        excluded.add(source_fresh[0]["review_id"])
        note(f"excluded {len(excluded)} review ids, including source timing {source_fresh[0]['review_id']}")

        siebert = work / "hf/hub/models--siebert--sentiment-roberta-large-english/snapshots/74cea614e245b0832c770ec9aa51bd58df965b9c"
        scorer = pipeline("sentiment-analysis", model=str(siebert), tokenizer=str(siebert), top_k=None, device=-1)

        def as_scores(raw_scores):
            if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
                return raw_scores[0]
            return raw_scores

        def pos_of(text):
            table = {d["label"].upper(): float(d["score"]) for d in as_scores(scorer(text[:1500] if text.strip() else " "))}
            p = table.get("POSITIVE", 0.0)
            return p, p >= table.get("NEGATIVE", 0.0)

        known_p, known_y = pos_of("This movie was wonderful and I loved every minute of it.")
        known_n, known_ny = pos_of("This movie was awful and I hated every minute of it.")
        if not (known_y and not known_ny):
            raise SystemExit("SiEBERT check failed")

        snap = work / "hf/hub/models--allenai--OLMo-7B-0724-hf/snapshots" / cfg["olmo"]["revision"]
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
        block = model.model.layers[int(cfg["olmo"]["layer"])]
        sample = cfg["sample"]
        max_new = int(sample["max_new_tokens"])
        note("OLMo loaded")

        def render_eval(text):
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
            return tok.decode(ids, skip_special_tokens=True).rstrip() + cfg["eval_bridge"]

        def generate(text, direction, seed):
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

        smoke_review = next(r for r in test_neg if r["review_id"] == source_panel["review_ids"][0])
        smoke_text = render_eval(smoke_review["text"])
        enc = tok(smoke_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            base_logits = model(**enc).logits

        def zero_hook(_m, inputs):
            return (inputs[0].clone(),) + tuple(inputs[1:])

        handle = block.register_forward_pre_hook(zero_hook)
        try:
            with torch.no_grad():
                zero_logits = model(**enc).logits
        finally:
            handle.remove()
        seed_a = generate(smoke_text, None, 12345)[0]
        seed_b = generate(smoke_text, None, 12345)[0]
        _, _, native_realised = generate(smoke_text, saved_native, 0)
        _, _, transfer_realised = generate(smoke_text, saved_transfer, 0)
        smoke = {
            "zero_logit_max_abs": float(torch.max(torch.abs(base_logits.float() - zero_logits.float())).cpu()),
            "seed_repeat_identical": seed_a == seed_b,
            "paired_seed_fn": review_seed(smoke_review["review_id"]),
            "native_realised": native_realised,
            "transfer_realised": transfer_realised,
            "target_norm": dose,
        }
        save_json(art / "smoke.json", smoke)
        note(f"smoke {smoke}")
        if smoke["zero_logit_max_abs"] > 1e-3 or not smoke["seed_repeat_identical"]:
            raise RuntimeError("smoke failed: parity or seed repeat")
        for label, realised in (("native", native_realised), ("transfer", transfer_realised)):
            for key in ("prefill", "decode"):
                got = float(realised.get(key, 0.0))
                if abs(got - dose) / dose > 0.02:
                    raise RuntimeError(f"{label} realised {key} norm {got} != {dose}")

        manifest_path = art / "panel_manifest.json"
        if manifest_path.exists():
            frozen = json.loads(manifest_path.read_text())
            note(f"reusing frozen panel n={frozen['n']}")
        else:
            pool = [r for r in test_neg if r["review_id"] not in excluded]
            np.random.default_rng(int(cfg["panel_seed"])).shuffle(pool)
            timers = pool[: int(cfg["smoke_blocks"])]
            # These timing reviews are spent and are not part of the scored panel.
            directions = {"unedited": None, "native": saved_native, "transfer": saved_transfer}
            t_batch = time.time()
            for row in timers:
                text = render_eval(row["text"])
                seed = review_seed(row["review_id"])
                for arm in ("unedited", "native", "transfer"):
                    generate(text, directions[arm], seed)
            sec_block = (time.time() - t_batch) / max(1, len(timers))
            remain = max(0.0, report_at - time.time())
            n_panel = min(int(cfg["target_panel"]), int(remain * 0.92 / sec_block)) if sec_block > 0 else 0
            chosen = pool[len(timers):len(timers) + n_panel]
            frozen = {
                "n": len(chosen),
                "target": int(cfg["target_panel"]),
                "sec_per_block": sec_block,
                "remain_sec_at_freeze": remain,
                "panel_seed": int(cfg["panel_seed"]),
                "review_ids": [r["review_id"] for r in chosen],
                "seeds": {r["review_id"]: review_seed(r["review_id"]) for r in chosen},
                "timing_review_ids": [r["review_id"] for r in timers],
                "frozen_before_fresh_scores": True,
            }
            save_json(manifest_path, frozen)
            note(f"froze panel n={frozen['n']} sec/block {sec_block:.2f}")
        by_id = {r["review_id"]: r for r in test_neg}
        panel = [by_id[i] for i in frozen["review_ids"]]
        state["frozen_n"] = frozen["n"]

        ckpt = art / "confirmation.jsonl"
        done = {}
        if ckpt.exists():
            for line in ckpt.open():
                rec = json.loads(line)
                done[rec["review_id"]] = rec
        blocks = []
        directions = {"unedited": None, "native": saved_native, "transfer": saved_transfer}
        for row in panel:
            if row["review_id"] in done:
                blocks.append(done[row["review_id"]])
                continue
            if late():
                break
            text = render_eval(row["text"])
            seed = review_seed(row["review_id"])
            rec = {"review_id": row["review_id"], "seed": seed}
            for arm in ("unedited", "native", "transfer"):
                t_g = time.time()
                new_ids, cont, realised = generate(text, directions[arm], seed)
                p, y = pos_of(cont)
                rec[arm] = {
                    "review_id": row["review_id"],
                    "prefix": text,
                    "continuation": cont,
                    "n_new": len(new_ids),
                    "sec": time.time() - t_g,
                    "positive_score": p,
                    "positive": bool(y),
                    "realised_norm": realised,
                    "seed": seed,
                    **quality(cont, len(new_ids), max_new),
                }
            blocks.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if len(blocks) % 10 == 0:
                note(f"block {len(blocks)}/{frozen['n']}")
        state["n_blocks"] = len(blocks)
        state["partial"] = len(blocks) < int(frozen["n"])
        state["summary"] = {arm: summarize([b[arm] for b in blocks]) for arm in ("unedited", "native", "transfer")}
        if blocks:
            state["test"] = summarise_retention(retention_rows(blocks, fraction), int(cfg["test_boot"]), 0)
            scores = []
            for b in blocks:
                u, n, t = b["unedited"], b["native"], b["transfer"]
                scores.append({
                    "review_id": b["review_id"],
                    "seed": b["seed"],
                    "U": u["positive_score"],
                    "N": n["positive_score"],
                    "T": t["positive_score"],
                    "positive": {"unedited": u["positive"], "native": n["positive"], "transfer": t["positive"]},
                    "native_gain": n["positive_score"] - u["positive_score"],
                    "transfer_gain": t["positive_score"] - u["positive_score"],
                    "absolute_gap": t["positive_score"] - n["positive_score"],
                    "retention_80": (t["positive_score"] - u["positive_score"]) - fraction * (n["positive_score"] - u["positive_score"]),
                    "empty": {"unedited": u["empty"], "native": n["empty"], "transfer": t["empty"]},
                    "repetitive": {"unedited": u["repetitive"], "native": n["repetitive"], "transfer": t["repetitive"]},
                    "truncated": {"unedited": u["truncated"], "native": n["truncated"], "transfer": t["truncated"]},
                    "sec": {"unedited": u["sec"], "native": n["sec"], "transfer": t["sec"]},
                })
            save_json(art / "scores.json", scores)
            save_json(art / "test_retention.json", state["test"])
            blind, key = [], []
            brng = np.random.default_rng(0)
            for b in blocks[: int(cfg["blind_n"])]:
                arms = ["unedited", "native", "transfer"]
                brng.shuffle(arms)
                for arm in arms:
                    code = hashlib.sha256(f"{b['review_id']}:{arm}".encode()).hexdigest()[:8]
                    blind.append({"code": code, "prefix": b[arm]["prefix"], "continuation": b[arm]["continuation"]})
                    key.append({"code": code, "arm": arm, "review_id": b["review_id"]})
            save_json(art / "blind_sample.json", blind)
            save_json(art / "blind_key.json", key)
        note(f"done blocks {len(blocks)} partial {state['partial']}")
        del model
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
