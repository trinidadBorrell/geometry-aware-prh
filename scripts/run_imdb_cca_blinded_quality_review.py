#!/usr/bin/env python3
"""Prepare and, once ratings are locked, analyse a blinded OLMo quality review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("unedited", "native", "transfer")
LETTERS = ("A", "B", "C")
SCORES = {0, 1, 2, "unclear"}


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


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def review_seed(review_id: str) -> int:
    return int(hashlib.sha256(review_id.encode()).hexdigest()[:8], 16)


def fence_for(text: str) -> str:
    n = 3
    while "`" * n in text:
        n += 1
    return "`" * n


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
        "n_reviews": int(v.size),
        "n_boot": int(n_boot),
    }


def blank_rating():
    return {"readability": None, "topic_continuity": None, "reason": ""}


def parent_sample_ok(art: Path):
    blind = json.loads((art / "blind_sample.json").read_text())
    key = json.loads((art / "blind_key.json").read_text())
    manifest = json.loads((art / "panel_manifest.json").read_text())
    if len(blind) != len(key) or len(key) % 3 != 0:
        return False, "parent blinded sample is not grouped in triplets"
    cases = []
    for i in range(0, len(key), 3):
        group_k = key[i:i + 3]
        group_b = blind[i:i + 3]
        ids = {row["review_id"] for row in group_k}
        arms = [row["arm"] for row in group_k]
        if len(ids) != 1 or sorted(arms) != sorted(ARMS):
            return False, "a parent case does not contain exactly one of each arm"
        if [row["code"] for row in group_k] != [row["code"] for row in group_b]:
            return False, "parent key and sheet codes disagree"
        cases.append(group_k[0]["review_id"])
    if cases != manifest["review_ids"][: len(cases)]:
        return False, "parent sample is not the pre-score panel prefix"
    return True, {"review_ids": cases, "n": len(cases)}


def build_cases_from_parent(source_art: Path, generations: dict):
    blind = json.loads((source_art / "blind_sample.json").read_text())
    key = json.loads((source_art / "blind_key.json").read_text())
    cases = []
    for i in range(0, len(key), 3):
        letters = {}
        review_id = key[i]["review_id"]
        for letter, row_k, row_b in zip(LETTERS, key[i:i + 3], blind[i:i + 3]):
            arm = row_k["arm"]
            saved = generations[review_id][arm]
            letters[letter] = {
                "letter": letter,
                "code": row_k["code"],
                "arm": arm,
                "prefix": row_b["prefix"],
                "continuation": row_b["continuation"],
                "seed": int(saved["seed"]),
                "truncated": bool(saved["truncated"]),
            }
        cases.append({"case": len(cases) + 1, "review_id": review_id, "letters": letters})
    return cases


def build_cases_fresh(generations: dict, n: int, seed: int):
    ids = sorted(generations)
    rng = np.random.default_rng(seed)
    order = list(ids)
    rng.shuffle(order)
    chosen = order[:n]
    cases = []
    label_rng = np.random.default_rng(seed + 1)
    for review_id in chosen:
        arms = list(ARMS)
        label_rng.shuffle(arms)
        letters = {}
        for letter, arm in zip(LETTERS, arms):
            saved = generations[review_id][arm]
            letters[letter] = {
                "letter": letter,
                "code": hashlib.sha256(f"{review_id}:{arm}".encode()).hexdigest()[:8],
                "arm": arm,
                "prefix": saved["prefix"],
                "continuation": saved["continuation"],
                "seed": int(saved["seed"]),
                "truncated": bool(saved["truncated"]),
            }
        cases.append({"case": len(cases) + 1, "review_id": review_id, "letters": letters})
    return cases


def verify_cases(cases):
    errors = []
    for case in cases:
        letters = case["letters"]
        if set(letters) != set(LETTERS):
            errors.append(f"case {case['case']} letters {sorted(letters)}")
            continue
        arms = [letters[k]["arm"] for k in LETTERS]
        if sorted(arms) != sorted(ARMS):
            errors.append(f"case {case['case']} arms {arms}")
        prefixes = {letters[k]["prefix"] for k in LETTERS}
        seeds = {letters[k]["seed"] for k in LETTERS}
        if len(prefixes) != 1:
            errors.append(f"case {case['case']} prefixes differ")
        if len(seeds) != 1:
            errors.append(f"case {case['case']} seeds differ")
        expected = review_seed(case["review_id"])
        if seeds and expected not in seeds:
            errors.append(f"case {case['case']} seed {seeds} != {expected}")
    return errors


def sheet_text(cases) -> str:
    lines = [
        "# Blinded quality review — `imdb_cca_blinded_quality_review_v1`",
        "",
        "Rate each continuation on its own. A shift from a negative opinion to a positive one is the intended edit: do not penalise it unless the continuation also changes the subject or contradicts a concrete fact.",
        "",
        "Readability: 2 = coherent and easy to read; 1 = understandable with a noticeable defect; 0 = hard to follow or unusable.",
        "",
        "Topic continuity: 2 = naturally continues the same review; 1 = generic but compatible with the prefix; 0 = changes the subject or contradicts a concrete fact in the prefix.",
        "",
        "Use unclear when the prefix or response provides too little evidence. Add a short reason for any 0 or unclear rating. Rate a cut-off ending by how much it impairs the text.",
        "",
        "A clearly usable continuation scores at least 1 on both scales. Record ratings in `artifacts/ratings.json` and set `locked` to true. This sheet does not identify the arms.",
        "",
    ]
    for case in cases:
        prefix = case["letters"]["A"]["prefix"]
        lines.append(f"## Case {case['case']}")
        lines.append("")
        lines.append("### Prefix")
        lines.append("")
        fence = fence_for(prefix)
        lines.append(fence)
        lines.append(prefix)
        lines.append(fence)
        lines.append("")
        for letter in LETTERS:
            text = case["letters"][letter]["continuation"]
            lines.append(f"### {letter}")
            lines.append("")
            fence = fence_for(text)
            lines.append(fence)
            lines.append(text)
            lines.append(fence)
            lines.append("")
    return "\n".join(lines) + "\n"


def blank_ratings(cases):
    return {
        "locked": False,
        "cases": [
            {"case": case["case"], "ratings": {letter: blank_rating() for letter in LETTERS}}
            for case in cases
        ],
    }


def ratings_pending(ratings) -> list[str]:
    if not ratings or not ratings.get("locked"):
        return ["ratings are not locked"]
    missing = []
    for case in ratings.get("cases", []):
        for letter, row in (case.get("ratings") or {}).items():
            if row.get("readability") not in SCORES or row.get("topic_continuity") not in SCORES:
                missing.append(f"case {case.get('case')} {letter}")
            elif (row.get("readability") in (0, "unclear") or row.get("topic_continuity") in (0, "unclear")) and not str(row.get("reason") or "").strip():
                missing.append(f"case {case.get('case')} {letter} missing reason")
    return missing


def numeric(value):
    return isinstance(value, int) and value in (0, 1, 2)


def usable_flag(row, unclear_as_unusable: bool):
    read, topic = row["readability"], row["topic_continuity"]
    if read == "unclear" or topic == "unclear":
        return False if unclear_as_unusable else None
    if not numeric(read) or not numeric(topic):
        return None
    return bool(read >= 1 and topic >= 1)


def analyse(cases, ratings, n_boot):
    by_case = {row["case"]: row for row in ratings["cases"]}
    joined = []
    missing = []
    for case in cases:
        rated = by_case.get(case["case"])
        if rated is None:
            missing.append({"case": case["case"], "letter": None, "issue": "case missing"})
            continue
        rec = {"case": case["case"], "review_id": case["review_id"], "arms": {}}
        for letter in LETTERS:
            row = (rated.get("ratings") or {}).get(letter)
            meta = case["letters"][letter]
            if row is None or row.get("readability") not in SCORES or row.get("topic_continuity") not in SCORES:
                missing.append({"case": case["case"], "letter": letter, "arm": meta["arm"], "issue": "missing"})
                continue
            rec["arms"][meta["arm"]] = {
                "letter": letter,
                "readability": row["readability"],
                "topic_continuity": row["topic_continuity"],
                "reason": row.get("reason") or "",
                "truncated": meta["truncated"],
                "unclear": row["readability"] == "unclear" or row["topic_continuity"] == "unclear",
            }
        joined.append(rec)

    def arm_summary(name):
        rows = [rec["arms"][name] for rec in joined if name in rec["arms"]]
        read = [r["readability"] for r in rows if numeric(r["readability"])]
        topic = [r["topic_continuity"] for r in rows if numeric(r["topic_continuity"])]
        return {
            "n_rated": len(rows),
            "readability_mean": float(np.mean(read)) if read else None,
            "readability_n": len(read),
            "topic_mean": float(np.mean(topic)) if topic else None,
            "topic_n": len(topic),
            "unclear": int(sum(r["unclear"] for r in rows)),
            "truncated": int(sum(r["truncated"] for r in rows)),
        }

    def paired(left, right, unclear_as_unusable):
        diffs = []
        only_left = only_right = both_unclear = 0
        cap_rows = []
        for rec in joined:
            if left not in rec["arms"] or right not in rec["arms"]:
                continue
            a = usable_flag(rec["arms"][left], unclear_as_unusable)
            b = usable_flag(rec["arms"][right], unclear_as_unusable)
            if a is None or b is None:
                if rec["arms"][left]["unclear"] and rec["arms"][right]["unclear"]:
                    both_unclear += 1
                continue
            diffs.append(float(a) - float(b))
            only_left += int(a and not b)
            only_right += int(b and not a)
            if a != b:
                cap_rows.append({
                    "review_id": rec["review_id"],
                    "lower_arm": right if a and not b else left,
                    "lower_truncated": rec["arms"][right if a and not b else left]["truncated"],
                    "higher_truncated": rec["arms"][left if a and not b else right]["truncated"],
                })
        return {
            "usable_difference": boot_mean(diffs, n_boot),
            "only_left": only_left,
            "only_right": only_right,
            "both_unclear_excluded": both_unclear,
            "cap_among_disagreements": {
                "n_disagreements": len(cap_rows),
                "lower_arm_truncated": int(sum(r["lower_truncated"] for r in cap_rows)),
                "higher_arm_truncated": int(sum(r["higher_truncated"] for r in cap_rows)),
            },
        }

    def score_diff(left, right, field):
        diffs = []
        for rec in joined:
            if left not in rec["arms"] or right not in rec["arms"]:
                continue
            a, b = rec["arms"][left][field], rec["arms"][right][field]
            if numeric(a) and numeric(b):
                diffs.append(float(a) - float(b))
        return boot_mean(diffs, n_boot)

    primary = paired("transfer", "native", False)
    sensitivity = paired("transfer", "native", True)
    return {
        "n_cases": len(cases),
        "missing": missing,
        "arms": {name: arm_summary(name) for name in ARMS},
        "transfer_minus_native": primary,
        "transfer_minus_native_unclear_as_unusable": sensitivity,
        "native_minus_unedited": paired("native", "unedited", False),
        "transfer_minus_unedited": paired("transfer", "unedited", False),
        "readability_transfer_minus_native": score_diff("transfer", "native", "readability"),
        "topic_transfer_minus_native": score_diff("transfer", "native", "topic_continuity"),
        "readability_transfer_minus_unedited": score_diff("transfer", "unedited", "readability"),
        "topic_transfer_minus_unedited": score_diff("transfer", "unedited", "topic_continuity"),
        "readability_native_minus_unedited": score_diff("native", "unedited", "readability"),
        "topic_native_minus_unedited": score_diff("native", "unedited", "topic_continuity"),
        "sentiment_retention_unchanged": True,
    }


def fmt(block):
    if not block or "mean" not in block:
        return "not estimated"
    return f"{block['mean']:+.3f} [{block['low']:+.3f}, {block['high']:+.3f}] (n={block['n_reviews']})"


def render_pending(manifest):
    return "\n".join([
        "# RESULTS — `imdb_cca_blinded_quality_review_v1`",
        "",
        "Ratings are pending. The review sheet is blank, and the arms stay blinded.",
        "",
        f"The frozen sample has {manifest['n_cases']} review triplets. Source: {manifest['source_rule']}",
        "",
        "Record readability and topic continuity in `artifacts/ratings.json`, then set `locked` to true.",
        "",
        "Analyse command:",
        "",
        "`python scripts/run_imdb_cca_blinded_quality_review.py --analyse --config configs/imdb_cca_blinded_quality_review_v1.json --work-root $WORK`",
        "",
        "This review does not change the 80% sentiment-retention result.",
        "",
    ])


def render_analysis(result):
    primary = result["transfer_minus_native"]
    sens = result["transfer_minus_native_unclear_as_unusable"]
    lines = [
        "# RESULTS — `imdb_cca_blinded_quality_review_v1`",
        "",
        "Transfer minus native, clearly usable rate, paired by review: " + fmt(primary["usable_difference"]) + ".",
        f"Prompts usable only under transfer: {primary['only_left']}. Prompts usable only under native: {primary['only_right']}.",
        "A small or uncertain difference is not evidence that the edits have the same quality.",
        "",
        "Treating unclear ratings as unusable: " + fmt(sens["usable_difference"]) + ".",
        "",
        "Versus unedited, native usable-rate difference: " + fmt(result["native_minus_unedited"]["usable_difference"]) + ".",
        "Versus unedited, transfer usable-rate difference: " + fmt(result["transfer_minus_unedited"]["usable_difference"]) + ".",
        "",
        "Mean score differences, transfer minus native: readability " + fmt(result["readability_transfer_minus_native"]) + "; topic continuity " + fmt(result["topic_transfer_minus_native"]) + ".",
        "",
        "| arm | rated | readability mean | topic mean | unclear | token-capped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        row = result["arms"][name]
        read = "n/a" if row["readability_mean"] is None else f"{row['readability_mean']:.3f}"
        topic = "n/a" if row["topic_mean"] is None else f"{row['topic_mean']:.3f}"
        lines.append(f"| {name} | {row['n_rated']} | {read} | {topic} | {row['unclear']} | {row['truncated']} |")
    lines += [
        "",
        f"Missing ratings: {len(result['missing'])}.",
    ]
    if result["missing"]:
        lines.append(json.dumps(result["missing"]))
    cap = primary["cap_among_disagreements"]
    lines += [
        "",
        f"Among {cap['n_disagreements']} paired usability disagreements, the lower-scoring arm hit the token cap in {cap['lower_arm_truncated']} and the higher-scoring arm hit it in {cap['higher_arm_truncated']}. This description does not change the sentiment-retention result.",
        "",
        "Analyse command: `python scripts/run_imdb_cca_blinded_quality_review.py --analyse --config configs/imdb_cca_blinded_quality_review_v1.json --work-root $WORK`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    work = Path(args.work_root)
    dest = work / cfg["output"]
    art = dest / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    source_art = work / cfg["source"] / "artifacts"
    generations = {row["review_id"]: row for row in load_jsonl(source_art / "confirmation.jsonl")}

    if args.prepare:
        ok, info = parent_sample_ok(source_art)
        if ok:
            cases = build_cases_from_parent(source_art, generations)
            rule = "parent blinded sample: first complete triplets of the panel frozen before scores, with its original within-case order"
        else:
            complete = {
                rid: row for rid, row in generations.items()
                if all(arm in row and row[arm].get("continuation") is not None for arm in ARMS)
            }
            cases = build_cases_fresh(complete, int(cfg["fallback_n"]), int(cfg["fallback_seed"]))
            rule = f"new sample of {len(cases)} because the parent sample was not usable ({info})"
        errors = verify_cases(cases)
        for case in cases:
            saved = generations[case["review_id"]]
            for letter in LETTERS:
                item = case["letters"][letter]
                got = saved[item["arm"]]
                if got["continuation"] != item["continuation"] or got["prefix"] != item["prefix"]:
                    errors.append(f"case {case['case']} {item['arm']} text does not match the saved generation")
                if int(got["seed"]) != item["seed"] or int(saved["seed"]) != item["seed"]:
                    errors.append(f"case {case['case']} {item['arm']} seed does not match")
        manifest = {
            "n_cases": len(cases),
            "source_rule": rule,
            "parent_sample_reused": bool(ok),
            "review_ids": [case["review_id"] for case in cases],
            "seeds": {case["review_id"]: case["letters"]["A"]["seed"] for case in cases},
            "selected_by_sentiment": False,
            "selected_by_truncation": False,
            "selected_by_winner": False,
            "verification_errors": errors,
            "source_sha256": {
                "blind_sample": sha256_file(source_art / "blind_sample.json"),
                "blind_key": sha256_file(source_art / "blind_key.json"),
                "confirmation": sha256_file(source_art / "confirmation.jsonl"),
                "panel_manifest": sha256_file(source_art / "panel_manifest.json"),
            },
        }
        save_json(art / "sample_manifest.json", manifest)
        if errors:
            raise SystemExit("verification failed: " + "; ".join(errors))
        key = []
        for case in cases:
            for letter in LETTERS:
                item = case["letters"][letter]
                key.append({
                    "case": case["case"],
                    "letter": letter,
                    "arm": item["arm"],
                    "review_id": case["review_id"],
                    "seed": item["seed"],
                    "code": item["code"],
                    "truncated": item["truncated"],
                })
        save_json(art / "arm_key.json", key)
        sheet = dest / "review_sheet.md"
        sheet.write_text(sheet_text(cases))
        ratings_path = art / "ratings.json"
        if not ratings_path.exists():
            save_json(ratings_path, blank_ratings(cases))
        (dest / "RESULTS.md").write_text(render_pending(manifest))
        print(f"cases {len(cases)}")
        print(f"rule {rule}")
        print(f"blank review sheet {sheet}")
        print("ratings pending")
        print("python scripts/run_imdb_cca_blinded_quality_review.py --analyse --config configs/imdb_cca_blinded_quality_review_v1.json --work-root $WORK")
        return 0

    manifest = json.loads((art / "sample_manifest.json").read_text())
    key = json.loads((art / "arm_key.json").read_text())
    ratings_path = art / "ratings.json"
    if not ratings_path.exists():
        (dest / "RESULTS.md").write_text(render_pending(manifest))
        print("ratings pending")
        print("python scripts/run_imdb_cca_blinded_quality_review.py --analyse --config configs/imdb_cca_blinded_quality_review_v1.json --work-root $WORK")
        return 0
    ratings = json.loads(ratings_path.read_text())
    problems = ratings_pending(ratings)
    if problems:
        (dest / "RESULTS.md").write_text(render_pending(manifest))
        print("ratings pending: " + "; ".join(problems))
        print("python scripts/run_imdb_cca_blinded_quality_review.py --analyse --config configs/imdb_cca_blinded_quality_review_v1.json --work-root $WORK")
        return 0
    cases = []
    grouped = {}
    for row in key:
        grouped.setdefault(row["case"], {"case": row["case"], "review_id": row["review_id"], "letters": {}})
        saved = generations[row["review_id"]][row["arm"]]
        grouped[row["case"]]["letters"][row["letter"]] = {
            "arm": row["arm"],
            "prefix": saved["prefix"],
            "continuation": saved["continuation"],
            "seed": row["seed"],
            "truncated": row["truncated"],
            "code": row["code"],
        }
    cases = [grouped[i] for i in sorted(grouped)]
    result = analyse(cases, ratings, int(cfg["n_boot"]))
    save_json(art / "paired_results.json", result)
    (dest / "RESULTS.md").write_text(render_analysis(result))
    print("wrote paired results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
