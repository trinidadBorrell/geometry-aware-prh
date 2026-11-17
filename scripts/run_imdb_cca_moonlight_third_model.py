#!/usr/bin/env python3
"""Decode the frozen Qwen sentiment command into Moonlight and compare it with Moonlight's own vector."""

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


def allow_unmarked_safetensors() -> None:
    """Moonlight's shards have no __metadata__.format. Treat that as a PyTorch archive."""
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


def locate_ids(tok, text):
    """Return input ids and character spans. Prefer tokenizer offsets, else cumulative decode."""
    import torch
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    except (TypeError, ValueError, NotImplementedError):
        enc = None
    if enc is not None and "offset_mapping" in enc:
        offs = enc["offset_mapping"][0].tolist()
        if offs and any(int(b) > int(a) for a, b in offs):
            return enc["input_ids"], offs
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    spans, prev = [], ""
    for i in range(len(ids)):
        cur = tok.decode(ids[: i + 1], skip_special_tokens=True)
        if not (text.startswith(prev) and text.startswith(cur) and len(cur) >= len(prev)):
            return None
        spans.append((len(prev), len(cur)))
        prev = cur
    return enc["input_ids"], spans


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}], one-sided lower bound {block['lower95']:+.3f}"


def render_md(state):
    lines = [
        "# RESULTS — `imdb_cca_moonlight_third_model_v1`",
        "",
        "The Qwen command and the Qwen–OLMo encoders stayed frozen. Moonlight entered only through the new residual decoder and the native reference vector. The injection norm was 0.25 times the median prompt-end residual on a separate calibration set, not a sentiment-tuned dose.",
        "",
    ]
    test = state.get("test") or {}
    if test.get("transfer_minus_unedited"):
        steer = test["transfer_minus_unedited"]
        sign = test["transfer_minus_reversed"]
        rev = test["reversed_minus_unedited"]
        native = test["native_minus_unedited"]
        gap = test["transfer_minus_native"]
        steered = steer.get("lower95", -1) > 0
        sign_ok = sign.get("lower95", -1) > 0
        lines.append(
            "Did the unchanged command steer Moonlight? "
            + ("Yes. " if steered else "No. The transfer gain is not established. ")
            + f"Transfer minus unedited {fmt(steer)}."
        )
        lines.append(
            "Did reversing the sign change the effect? "
            + ("Yes. " if sign_ok else "The sign contrast is not established. ")
            + f"Transfer minus reversed {fmt(sign)}. Reversed minus unedited {fmt(rev)}."
        )
        lines.append(f"How close did it come to native? Transfer minus native {fmt(gap)}. Native minus unedited {fmt(native)}.")
        if native.get("lower95", -1) > 0:
            lines.append(f"Native gain is clearly positive, so the 80% retention contrast is {fmt(test['retention_80'])}.")
        else:
            lines.append("Native gain is not clearly positive, so percentage retention stays unresolved.")
    else:
        lines.append("Did the unchanged command steer Moonlight? Incomplete.")
        lines.append("Did reversing the sign change the effect? Incomplete.")
        lines.append("How close did it come to native? Incomplete.")
    if state.get("resume_note"):
        lines.append(state["resume_note"])
    if state.get("partial"):
        lines.append(f"The test analysis is partial: {state.get('n_blocks', 0)} of {state.get('frozen_n', 0)} frozen reviews.")
    elif state.get("n_blocks"):
        lines.append(f"Coverage is the full frozen panel, n={state.get('n_blocks')}.")
    lines.append("")
    if state.get("dose"):
        lines.append("Frozen norm: `" + json.dumps(state["dose"]) + "`")
        lines.append("")
    if state.get("summary"):
        lines += ["| arm | n | positive | score | empty | repetitive | truncated |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ("unedited", "transfer", "reversed", "native"):
            row = state["summary"].get(arm) or {}
            if row.get("n"):
                lines.append(f"| {arm} | {row['n']} | {row['positive']} | {row['mean_score']:.3f} | {row['empty']} | {row['repetitive']} | {row['truncated']} |")
        lines.append("")
    if test.get("flips"):
        flips = test["flips"]
        lines.append(
            "Flips to positive / to negative: "
            + "; ".join(f"{name} {pair[0]}/{pair[1]}" for name, pair in flips.items())
            + "."
        )
        lines.append("")
    lines.append("Usability is not inferred from SiEBERT. A blinded subset is saved separately from the scores.")
    lines.append("")
    if state.get("decoder"):
        lines.append("Decoder validation: `" + json.dumps(state["decoder"]) + "`")
        lines.append("")
    if state.get("error"):
        lines += ["## Stop", "", "```", state["error"][-1800:], "```", ""]
    lines.append(
        f"Elapsed {state.get('elapsed_sec', 0):.0f}s after the cached-weight launch. "
        "Reproduction: `python scripts/run_imdb_cca_moonlight_third_model.py --config configs/imdb_cca_moonlight_third_model_v1.json --work-root $WORK`."
    )
    return "\n".join(lines) + "\n"


def contrasts_from_blocks(blocks, cfg):
    series = {name: [] for name in ("transfer_minus_unedited", "transfer_minus_reversed", "reversed_minus_unedited", "native_minus_unedited", "transfer_minus_native", "retention_80")}
    flips = {name: [0, 0] for name in ("transfer_vs_unedited", "reversed_vs_unedited", "transfer_vs_reversed", "native_vs_unedited", "transfer_vs_native")}
    fraction = float(cfg["retention_fraction"])
    scores = []
    for b in blocks:
        u, t, r, n = b["unedited"], b["transfer"], b["reversed"], b["native"]
        tg = float(t["positive_score"]) - float(u["positive_score"])
        ng = float(n["positive_score"]) - float(u["positive_score"])
        series["transfer_minus_unedited"].append(tg)
        series["transfer_minus_reversed"].append(float(t["positive_score"]) - float(r["positive_score"]))
        series["reversed_minus_unedited"].append(float(r["positive_score"]) - float(u["positive_score"]))
        series["native_minus_unedited"].append(ng)
        series["transfer_minus_native"].append(float(t["positive_score"]) - float(n["positive_score"]))
        series["retention_80"].append(tg - fraction * ng)
        for name, a, c in (
            ("transfer_vs_unedited", u, t), ("reversed_vs_unedited", u, r), ("transfer_vs_reversed", r, t),
            ("native_vs_unedited", u, n), ("transfer_vs_native", n, t),
        ):
            if (not a["positive"]) and c["positive"]:
                flips[name][0] += 1
            if a["positive"] and (not c["positive"]):
                flips[name][1] += 1
        scores.append({
            "review_id": b["review_id"], "seed": b["seed"],
            "U": u["positive_score"], "T": t["positive_score"], "R": r["positive_score"], "N": n["positive_score"],
            "transfer_gain": tg, "native_gain": ng, "retention_80": tg - fraction * ng,
        })
    n_boot = int(cfg["n_boot"])
    test = {name: boot_mean(values, n_boot) for name, values in series.items()}
    test["flips"] = flips
    test["native_clear_positive"] = test["native_minus_unedited"].get("lower95", -1) > 0
    return test, scores


def resume_remaining(cfg, work) -> int:
    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    t0 = time.time()
    dest = work / cfg["output"]
    art = dest / "artifacts"
    prior_state_path = art / "run_state.json"
    prior_state = json.loads(prior_state_path.read_text()) if prior_state_path.exists() else {}
    manifest = json.loads((art / "panel_manifest.json").read_text())
    ckpt = art / "confirmation.jsonl"
    done = []
    if ckpt.exists():
        done = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
    done_ids = [row["review_id"] for row in done]
    if done_ids != manifest["review_ids"][: len(done_ids)]:
        raise RuntimeError("saved generations are not a prefix of the frozen panel")
    remaining = manifest["review_ids"][len(done_ids):]
    print(f"resume scored {len(done_ids)} remaining {len(remaining)}", flush=True)
    if prior_state_path.exists() and not (art / "run_state_partial_99.json").exists() and len(done_ids) == 99:
        (art / "run_state_partial_99.json").write_text(prior_state_path.read_text())
        results_md = dest / "RESULTS.md"
        if results_md.exists():
            (art / "RESULTS_partial_99.md").write_text(results_md.read_text())

    state = {"log": [], "dose": prior_state.get("dose"), "decoder": prior_state.get("decoder")}
    directions = {
        "unedited": None,
        "transfer": np.load(art / "dir_transfer.npy"),
        "reversed": np.load(art / "dir_reversed.npy"),
        "native": np.load(art / "dir_native.npy"),
    }
    dose = float(json.loads((art / "dose.json").read_text())["dose"])
    if float(np.max(np.abs(directions["transfer"] + directions["reversed"]))) > 1e-8:
        raise RuntimeError("saved transfer and reversed edits are not exact sign flips")
    for name in ("transfer", "reversed", "native"):
        norm = float(np.linalg.norm(directions[name]))
        if abs(norm - dose) / dose > 0.02:
            raise RuntimeError(f"{name} norm {norm} != dose {dose}")

    def finish(blocks):
        state["n_blocks"] = len(blocks)
        state["frozen_n"] = int(manifest["n"])
        state["partial"] = len(blocks) < int(manifest["n"])
        state["summary"] = {arm: summarize([b[arm] for b in blocks]) for arm in directions}
        state["resume_note"] = (
            f"The first {len(done_ids)} reviews were scored in the timed run. "
            f"The remaining {len(blocks) - len(done_ids)} were scored afterwards with the same frozen panel, seeds, and directions."
        )
        if blocks:
            test, scores = contrasts_from_blocks(blocks, cfg)
            state["test"] = test
            save_json(art / "scores.json", scores)
            save_json(art / "test_contrasts.json", test)
        state["elapsed_sec"] = time.time() - t0
        save_json(art / "run_state.json", state)
        (dest / "RESULTS.md").write_text(render_md(state))
        print(f"wrote RESULTS blocks {len(blocks)} partial {state['partial']}", flush=True)

    if not remaining:
        finish(done)
        return 0

    allow_unmarked_safetensors()
    moon_dir = work / "hf/hub/models--moonshotai--Moonlight-16B-A3B/snapshots" / cfg["moonlight"]["revision"]
    mtok = AutoTokenizer.from_pretrained(str(moon_dir), local_files_only=True, trust_remote_code=True)
    mmodel = AutoModelForCausalLM.from_pretrained(
        str(moon_dir), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).cuda().eval()
    block = mmodel.model.layers[int(cfg["moonlight"]["layer"])]
    sample = cfg["sample"]
    max_new = int(sample["max_new_tokens"])
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

    def render_eval(text):
        ids = mtok(text, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
        return mtok.decode(ids, skip_special_tokens=True).rstrip() + cfg["eval_bridge"]

    def generate(text, direction, seed):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        ids = mtok(text, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
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
                fwd = mmodel(ids, attention_mask=torch.ones_like(ids), use_cache=True)
                past = fwd.past_key_values
                new_ids = []
                for _ in range(max_new):
                    nxt = sample_next(fwd.logits[0, -1], ids[0], temperature=sample["temperature"], top_p=sample["top_p"], freq_penalty=sample["freq_penalty"])
                    new_ids.append(int(nxt.item()))
                    if mtok.eos_token_id is not None and int(nxt.item()) == int(mtok.eos_token_id):
                        break
                    ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                    attn = torch.ones((1, ids.shape[1]), device=ids.device, dtype=ids.dtype)
                    fwd = mmodel(nxt.view(1, 1), attention_mask=attn, past_key_values=past, use_cache=True)
                    past = fwd.past_key_values
        finally:
            if handle is not None:
                handle.remove()
        return new_ids, mtok.decode(new_ids, skip_special_tokens=True), realised

    wanted = set(remaining)
    texts = {}
    ds = load_dataset("stanfordnlp/imdb")
    for i, row in enumerate(ds["test"]):
        review_id = f"test:{i}"
        if review_id in wanted:
            texts[review_id] = row["text"]
    missing = [rid for rid in remaining if rid not in texts]
    if missing:
        raise RuntimeError(f"missing review text for {missing[:5]}")

    blocks = list(done)
    try:
        for n_new, review_id in enumerate(remaining, start=1):
            text = render_eval(texts[review_id])
            seed = int(manifest["seeds"][review_id])
            if seed != review_seed(review_id):
                raise RuntimeError(f"frozen seed mismatch for {review_id}")
            rec = {"review_id": review_id, "seed": seed}
            for arm, arm_dir in directions.items():
                t_g = time.time()
                new_ids, cont, realised = generate(text, arm_dir, seed)
                p, y = pos_of(cont)
                rec[arm] = {
                    "review_id": review_id, "prefix": text, "continuation": cont, "n_new": len(new_ids),
                    "sec": time.time() - t_g, "positive_score": p, "positive": bool(y), "realised_norm": realised,
                    "seed": seed, **quality(cont, len(new_ids), max_new),
                }
            blocks.append(rec)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if n_new % 10 == 0 or n_new == len(remaining):
                print(f"resume block {len(blocks)}/{manifest['n']}", flush=True)
    finally:
        finish(blocks)
    return 0


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_root)
    if args.resume:
        return resume_remaining(cfg, work)
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
        source_art = work / cfg["source"] / "artifacts"
        parent_art = work / cfg["sentiment_parent"] / "artifacts"
        v_q = np.load(parent_art / "vec_qwen_averaged.npy").astype(np.float64)
        maps = np.load(source_art / "maps.npz")
        eq = np.asarray(maps["E_qwen"], dtype=np.float64)
        eo = np.asarray(maps["E_olmo"], dtype=np.float64)
        geometry = json.loads((source_art / "geometry.json").read_text())
        saved_transfer = np.load(source_art / "dir_olmo_aligned64.npy").astype(np.float64)
        olmo_dose = float(json.loads((parent_art / "run_state.json").read_text())["results"]["olmo"]["dose"])
        z_edit = eq @ v_q
        raw_olmo = np.linalg.pinv(eo) @ z_edit
        raw_norm = float(np.linalg.norm(raw_olmo))
        prepared = raw_olmo / raw_norm * olmo_dose
        provenance = {
            "z_norm": float(np.linalg.norm(z_edit)),
            "raw_norm": raw_norm,
            "recorded_pre_norm": float(geometry["transfer"]["aligned64"]["pre_norm"]),
            "max_abs_after_dose": float(np.max(np.abs(prepared - saved_transfer))),
            "centering": "archived encode() is E @ h with no stored PCA mean; fit coordinates use E @ (h - fit-split mean), and z_edit stays E @ v_Qwen",
        }
        hashes = {
            "vec_qwen": sha256_file(parent_art / "vec_qwen_averaged.npy"),
            "maps": sha256_file(source_art / "maps.npz"),
            "dir_olmo_aligned64": sha256_file(source_art / "dir_olmo_aligned64.npy"),
            "geometry": sha256_file(source_art / "geometry.json"),
        }
        ok = provenance["max_abs_after_dose"] < 1e-6 and abs(provenance["raw_norm"] - provenance["recorded_pre_norm"]) < 1e-6
        save_json(art / "frozen_command.json", {"provenance": provenance, "ok": ok, "sha256": hashes})
        np.save(art / "z_edit.npy", z_edit)
        note(f"provenance ok={ok} max_abs={provenance['max_abs_after_dose']:.3e}")
        if not ok:
            raise RuntimeError("z_edit does not reproduce the archived OLMo direction")

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
        gemma_splits = work / cfg["gemma_run"] / "artifacts/native_splits.json"
        if gemma_splits.exists():
            gsplit = json.loads(gemma_splits.read_text())
            excluded.update(gsplit.get("calibration") or [])
            excluded.update(gsplit.get("validation") or [])

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
        np.random.default_rng(int(cfg["norm_seed"])).shuffle(eligible)
        norm_rows = eligible[: int(cfg["n_norm_calibration"])]
        for row in norm_rows:
            excluded.add(row["review_id"])
        save_json(art / "norm_split.json", {"review_ids": [r["review_id"] for r in norm_rows], "used_for_scores": False})

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

        allow_unmarked_safetensors()
        moon_dir = work / "hf/hub/models--moonshotai--Moonlight-16B-A3B/snapshots" / cfg["moonlight"]["revision"]
        mtok = AutoTokenizer.from_pretrained(str(moon_dir), local_files_only=True, trust_remote_code=True)
        mmodel = AutoModelForCausalLM.from_pretrained(
            str(moon_dir), local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).cuda().eval()
        block = mmodel.model.layers[int(cfg["moonlight"]["layer"])]
        note(f"Moonlight loaded layer type {type(block).__name__}")
        olmo_snap = work / "hf/hub/models--allenai--OLMo-7B-0724-hf/snapshots" / cfg["olmo"]["revision"]
        otok = AutoTokenizer.from_pretrained(str(olmo_snap), local_files_only=True)
        omodel = AutoModelForCausalLM.from_pretrained(str(olmo_snap), local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
        oblock = omodel.model.layers[int(cfg["olmo"]["layer"])]
        pairs = json.loads((work / cfg["pairs_file"]).read_text())["pairs"]
        sample = cfg["sample"]
        max_new = int(sample["max_new_tokens"])

        def render_eval(tok, text):
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=int(cfg["prefix_tokens"]), return_tensors="pt")["input_ids"][0]
            return tok.decode(ids, skip_special_tokens=True).rstrip() + cfg["eval_bridge"]

        def generate(model, tok, layer, text, direction, seed):
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            ids = enc["input_ids"]
            realised = {}
            positions = {}
            handle = None
            if direction is not None:
                delta = torch.tensor(np.asarray(direction), device="cuda", dtype=torch.float32)

                def hook(_m, inputs, _delta=delta):
                    resid = inputs[0]
                    before = resid[:, -1].detach()
                    out_h = resid.clone()
                    out_h[:, -1] = out_h[:, -1] + _delta.to(out_h.dtype)
                    key = "prefill" if resid.shape[1] > 1 else "decode"
                    positions[key] = int(resid.shape[1])
                    if key not in realised:
                        realised[key] = float(torch.linalg.vector_norm((out_h[:, -1] - before).float()).cpu())
                    return (out_h,)

                handle = layer.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    # A length-1 query makes Transformers drop the causal mask. This model asserts the mask exists.
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
            return new_ids, tok.decode(new_ids, skip_special_tokens=True), realised, positions

        # Cached-generation check with the hook installed, against a full recompute.
        probe_text = "The film was dull."
        full_ids = mtok(probe_text, return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"]
        with torch.no_grad():
            for _ in range(4):
                attn = torch.ones_like(full_ids)
                nxt = torch.argmax(mmodel(full_ids, attention_mask=attn).logits[0, -1])
                full_ids = torch.cat([full_ids, nxt.view(1, 1)], dim=1)
        cached_ids, _, probe_realised, probe_pos = generate(mmodel, mtok, block, probe_text, np.zeros(mmodel.config.hidden_size), 0)
        # The zero hook still changes the path versus an unhooked argmax. Compare hook-off cache to the full recompute.
        torch.manual_seed(0)
        enc = mtok(probe_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        ids = enc["input_ids"]
        with torch.no_grad():
            fwd = mmodel(ids, attention_mask=torch.ones_like(ids), use_cache=True)
            past = fwd.past_key_values
            for _ in range(4):
                nxt = torch.argmax(fwd.logits[0, -1])
                ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
                attn = torch.ones((1, ids.shape[1]), device=ids.device, dtype=ids.dtype)
                fwd = mmodel(nxt.view(1, 1), attention_mask=attn, past_key_values=past, use_cache=True)
                past = fwd.past_key_values
        cache_match = bool(torch.equal(ids, full_ids))
        hook_ok = probe_pos.get("prefill", 0) > 1 and probe_pos.get("decode") == 1 and "prefill" in probe_realised and "decode" in probe_realised
        note(f"cache_match {cache_match} hook {probe_pos} realised {probe_realised}")
        if not cache_match or not hook_ok:
            raise RuntimeError("Moonlight cached generation or pre-block hook check failed")

        diffs, mismatches = [], []
        for pair in pairs:
            located = {}
            for label in ("positive", "negative"):
                text = pair[label] + pair["bridge"]
                enc = mtok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                box = {}

                def hook(_m, inputs):
                    box["h"] = inputs[0].detach()
                    return None

                handle = block.register_forward_pre_hook(hook)
                try:
                    with torch.no_grad():
                        mmodel(**enc)
                finally:
                    handle.remove()
                located[label] = (box["h"][0, -1].float().cpu().numpy(), int(enc["input_ids"][0, -1]))
            if located["positive"][1] != located["negative"][1]:
                mismatches.append(pair["pair_id"])
                continue
            diffs.append(located["positive"][0] - located["negative"][0])
        if not diffs:
            raise RuntimeError("no Moonlight pairs shared the neutral-suffix token")
        native_unit, native_raw_norm = unit(np.stack(diffs).mean(0))
        np.save(art / "vec_moonlight_native_unit.npy", native_unit)
        save_json(art / "pair_check.json", {"n_used": len(diffs), "n_pairs": len(pairs), "mismatched_pairs": mismatches, "mean_raw_norm": native_raw_norm})
        note(f"native pairs {len(diffs)} mismatches {len(mismatches)}")

        norms = []
        for row in norm_rows:
            text = render_eval(mtok, row["text"])
            enc = mtok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            box = {}

            def hook(_m, inputs):
                box["h"] = inputs[0].detach()
                return None

            handle = block.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    mmodel(**enc)
            finally:
                handle.remove()
            norms.append(float(np.linalg.norm(box["h"][0, -1].float().cpu().numpy())))
        median_norm = float(np.median(norms))
        dose = float(cfg["dose_fraction"]) * median_norm
        state["dose"] = {"median_residual_norm": median_norm, "fraction": float(cfg["dose_fraction"]), "dose": dose, "n": len(norms)}
        save_json(art / "dose.json", state["dose"])
        note(f"dose {dose:.4f} from median {median_norm:.4f}")

        usable_train = []
        for row in train:
            if row["review_id"] in excluded:
                continue
            made = shared_passage(row["text"], int(cfg["prompt_chars"]), int(cfg["continuation_chars"]))
            if made is not None:
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

        def collect(model, tok, layer, rows):
            out = {}
            for row in rows:
                if late():
                    break
                located = locate_ids(tok, row["full"])
                if located is None:
                    continue
                ids, spans = located
                idxs = [token_for_offset(spans, c) for c in row["offs"]]
                if any(i is None for i in idxs):
                    continue
                if not torch.is_tensor(ids):
                    ids = torch.tensor(ids, dtype=torch.long)
                if ids.ndim == 1:
                    ids = ids.view(1, -1)
                ids = ids.to("cuda")
                box = {}

                def hook(_m, inputs):
                    box["h"] = inputs[0].detach()
                    return None

                handle = layer.register_forward_pre_hook(hook)
                try:
                    with torch.no_grad():
                        model(ids)
                finally:
                    handle.remove()
                h = box["h"][0].float().cpu().numpy()
                out[row["review_id"]] = np.stack([h[i] for i in idxs])
            return out

        note("extract decoder activations")
        o_fit = collect(omodel, otok, oblock, fit_rows)
        m_fit = collect(mmodel, mtok, block, fit_rows)
        o_val = collect(omodel, otok, oblock, bridge_rows)
        m_val = collect(mmodel, mtok, block, bridge_rows)
        demo = next(r for r in fit_rows if r["review_id"] in o_fit and r["review_id"] in m_fit)
        boundary = {}
        for name, tok in (("olmo", otok), ("moonlight", mtok)):
            located = locate_ids(tok, demo["full"])
            picks = []
            if located is not None:
                ids, spans = located
                id_list = ids[0].tolist()
                for c in demo["offs"]:
                    i = token_for_offset(spans, c)
                    picks.append({"char": c, "token_index": i, "piece": None if i is None else tok.decode([id_list[i]], skip_special_tokens=True)})
            boundary[name] = picks
        save_json(art / "boundary_smoke.json", boundary)
        del omodel
        torch.cuda.empty_cache()

        def stack(a_map, b_map, rows):
            ids = [r["review_id"] for r in rows if r["review_id"] in a_map and r["review_id"] in b_map]
            if not ids:
                raise RuntimeError("no paired decoder rows")
            width = a_map[ids[0]].shape[0]
            if any(a_map[i].shape[0] != width or b_map[i].shape[0] != width for i in ids):
                raise RuntimeError("reviews do not share a row count")
            return np.concatenate([a_map[i] for i in ids]), np.concatenate([b_map[i] for i in ids]), ids, width

        Xo_fit, Xm_fit, fit_ids, width = stack(o_fit, m_fit, fit_rows)
        Xo_val, Xm_val, val_ids, width_val = stack(o_val, m_val, bridge_rows)
        if width != width_val:
            raise RuntimeError("fit and validation row counts differ")
        mu = Xo_fit.mean(axis=0)
        z_fit = (Xo_fit - mu) @ eo.T
        z_val = (Xo_val - mu) @ eo.T
        z_mean, z_sd = z_fit.mean(0), np.maximum(z_fit.std(0, ddof=1), 1e-8)
        h_mean = Xm_fit.mean(0)
        zs_fit, zs_val = (z_fit - z_mean) / z_sd, (z_val - z_mean) / z_sd
        hc_fit = Xm_fit - h_mean
        val_review = np.repeat(np.asarray(val_ids), width)

        def review_mse(pred, target, reviews):
            err = np.mean((pred - target) ** 2, axis=1)
            return float(np.mean([err[reviews == rid].mean() for rid in dict.fromkeys(reviews.tolist())]))

        baseline = review_mse(np.repeat(h_mean.reshape(1, -1), len(Xm_val), 0), Xm_val, val_review)
        gram, rhs = zs_fit.T @ zs_fit, zs_fit.T @ hc_fit
        best, alpha_rows = None, []
        for mult in cfg["alpha_multipliers"]:
            alpha = float(mult)
            slope = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
            mse = review_mse(h_mean + zs_val @ slope, Xm_val, val_review)
            alpha_rows.append({"alpha": alpha, "val_mse": mse})
            if best is None or mse < best["val_mse"]:
                best = {"alpha": alpha, "val_mse": mse, "slope": slope}
        D = best["slope"].T / z_sd.reshape(1, -1)
        intercept = h_mean - (z_mean / z_sd) @ best["slope"]
        v_raw = D @ z_edit
        v_norm = float(np.linalg.norm(v_raw))
        near_zero = v_norm < float(cfg["near_zero_norm"])
        transfer = np.zeros_like(v_raw) if near_zero else v_raw / v_norm * dose
        native = native_unit * dose
        reversed_edit = -transfer
        np.savez(art / "decoder.npz", D=D, intercept=intercept, mu_olmo=mu, z_mean=z_mean, z_sd=z_sd, h_mean=h_mean, alpha=best["alpha"])
        np.save(art / "dir_transfer.npy", transfer)
        np.save(art / "dir_reversed.npy", reversed_edit)
        np.save(art / "dir_native.npy", native)
        decoder_info = {
            "n_fit_reviews": len(fit_ids), "n_bridge_reviews": len(val_ids), "rows_per_review": width,
            "alpha": best["alpha"], "val_mse": best["val_mse"], "baseline_mse": baseline,
            "mse_ratio_vs_mean": best["val_mse"] / baseline if baseline else None,
            "raw_norm": v_norm, "near_zero": near_zero, "applied_norm": 0.0 if near_zero else dose,
            "sentiment_labels_used": False, "native_vector_used": False, "generation_scores_used": False,
        }
        state["decoder"] = decoder_info
        save_json(art / "decoder.json", {**decoder_info, "alphas": alpha_rows})
        save_json(art / "bridge_manifest.json", {"fit_review_ids": fit_ids, "bridge_review_ids": val_ids})
        note(f"decoder mse {best['val_mse']:.5f} baseline {baseline:.5f} raw {v_norm:.5f} near_zero {near_zero}")

        smoke_text = render_eval(mtok, norm_rows[0]["text"])
        enc = mtok(smoke_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            base_logits = mmodel(**enc).logits

        def zero_hook(_m, inputs):
            return (inputs[0].clone(),)

        handle = block.register_forward_pre_hook(zero_hook)
        try:
            with torch.no_grad():
                zero_logits = mmodel(**enc).logits
        finally:
            handle.remove()
        seed_a = generate(mmodel, mtok, block, smoke_text, None, 12345)[0]
        seed_b = generate(mmodel, mtok, block, smoke_text, None, 12345)[0]
        _, _, native_realised, native_pos = generate(mmodel, mtok, block, smoke_text, native, 0)
        _, _, transfer_realised, transfer_pos = generate(mmodel, mtok, block, smoke_text, transfer, 0)
        _, _, reversed_realised, _ = generate(mmodel, mtok, block, smoke_text, reversed_edit, 0)
        target = 0.0 if near_zero else dose
        smoke = {
            "zero_logit_max_abs": float(torch.max(torch.abs(base_logits.float() - zero_logits.float())).cpu()),
            "seed_repeat_identical": seed_a == seed_b,
            "native_positions": native_pos,
            "transfer_positions": transfer_pos,
            "native_realised": native_realised,
            "transfer_realised": transfer_realised,
            "reversed_realised": reversed_realised,
            "vector_sign_max_abs": float(np.max(np.abs(transfer + reversed_edit))),
            "boundary_same_chars": [a["char"] for a in boundary["olmo"]] == [a["char"] for a in boundary["moonlight"]],
            "boundary_tokens_found": bool(boundary["olmo"] and boundary["moonlight"] and all(a["token_index"] is not None and b["token_index"] is not None for a, b in zip(boundary["olmo"], boundary["moonlight"]))),
            "target_norm": target,
            "native_target_norm": dose,
        }
        save_json(art / "smoke.json", smoke)
        note(f"smoke zero {smoke['zero_logit_max_abs']} seeds {smoke['seed_repeat_identical']}")
        if smoke["zero_logit_max_abs"] > 1e-3 or not smoke["seed_repeat_identical"] or smoke["vector_sign_max_abs"] > 1e-8:
            raise RuntimeError("smoke failed")
        if not smoke["boundary_tokens_found"]:
            raise RuntimeError("text-boundary match failed")
        for label, realised, expect in (
            ("native", native_realised, dose),
            ("transfer", transfer_realised, target),
            ("reversed", reversed_realised, target),
        ):
            for key in ("prefill", "decode"):
                if key not in realised:
                    raise RuntimeError(f"{label} missed {key}")
                if expect > 0 and abs(float(realised[key]) - expect) / expect > 0.02:
                    raise RuntimeError(f"{label} {key} norm {realised[key]} != {expect}")

        for row in fit_ids + val_ids:
            excluded.add(row)
        pool = [r for r in test_neg if r["review_id"] not in excluded]
        np.random.default_rng(int(cfg["panel_seed"])).shuffle(pool)
        timers = pool[: int(cfg["smoke_blocks"])]
        directions = {"unedited": None, "transfer": transfer, "reversed": reversed_edit, "native": native}
        t_batch = time.time()
        for row in timers:
            text = render_eval(mtok, row["text"])
            seed = review_seed(row["review_id"])
            for arm_dir in directions.values():
                generate(mmodel, mtok, block, text, arm_dir, seed)
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
        ckpt = art / "confirmation.jsonl"
        blocks = []
        for row in chosen_rows:
            if late():
                break
            text = render_eval(mtok, row["text"])
            seed = review_seed(row["review_id"])
            rec = {"review_id": row["review_id"], "seed": seed}
            for arm, arm_dir in directions.items():
                t_g = time.time()
                new_ids, cont, realised, _positions = generate(mmodel, mtok, block, text, arm_dir, seed)
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
        state["summary"] = {arm: summarize([b[arm] for b in blocks]) for arm in directions}
        if blocks:
            series = {name: [] for name in ("transfer_minus_unedited", "transfer_minus_reversed", "reversed_minus_unedited", "native_minus_unedited", "transfer_minus_native", "retention_80")}
            flips = {name: [0, 0] for name in ("transfer_vs_unedited", "reversed_vs_unedited", "transfer_vs_reversed", "native_vs_unedited", "transfer_vs_native")}
            fraction = float(cfg["retention_fraction"])
            scores = []
            for b in blocks:
                u, t, r, n = b["unedited"], b["transfer"], b["reversed"], b["native"]
                tg = float(t["positive_score"]) - float(u["positive_score"])
                ng = float(n["positive_score"]) - float(u["positive_score"])
                series["transfer_minus_unedited"].append(tg)
                series["transfer_minus_reversed"].append(float(t["positive_score"]) - float(r["positive_score"]))
                series["reversed_minus_unedited"].append(float(r["positive_score"]) - float(u["positive_score"]))
                series["native_minus_unedited"].append(ng)
                series["transfer_minus_native"].append(float(t["positive_score"]) - float(n["positive_score"]))
                series["retention_80"].append(tg - fraction * ng)
                for name, a, c in (
                    ("transfer_vs_unedited", u, t), ("reversed_vs_unedited", u, r), ("transfer_vs_reversed", r, t),
                    ("native_vs_unedited", u, n), ("transfer_vs_native", n, t),
                ):
                    if (not a["positive"]) and c["positive"]:
                        flips[name][0] += 1
                    if a["positive"] and (not c["positive"]):
                        flips[name][1] += 1
                scores.append({
                    "review_id": b["review_id"], "seed": b["seed"],
                    "U": u["positive_score"], "T": t["positive_score"], "R": r["positive_score"], "N": n["positive_score"],
                    "transfer_gain": tg, "native_gain": ng, "retention_80": tg - fraction * ng,
                })
            n_boot = int(cfg["n_boot"])
            state["test"] = {name: boot_mean(values, n_boot) for name, values in series.items()}
            state["test"]["flips"] = flips
            state["test"]["native_clear_positive"] = state["test"]["native_minus_unedited"].get("lower95", -1) > 0
            save_json(art / "scores.json", scores)
            save_json(art / "test_contrasts.json", state["test"])
            blind, key = [], []
            brng = np.random.default_rng(0)
            for b in blocks[: int(cfg["blind_n"])]:
                order = ["unedited", "transfer", "reversed", "native"]
                brng.shuffle(order)
                for arm in order:
                    code = hashlib.sha256(f"{b['review_id']}:{arm}".encode()).hexdigest()[:8]
                    blind.append({"code": code, "prefix": b[arm]["prefix"], "continuation": b[arm]["continuation"]})
                    key.append({"code": code, "arm": arm, "review_id": b["review_id"]})
            save_json(art / "blind_sample.json", blind)
            save_json(art / "blind_key.json", key)
        note(f"done blocks {len(blocks)} partial {state['partial']}")
        del mmodel
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
