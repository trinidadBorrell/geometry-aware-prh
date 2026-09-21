#!/usr/bin/env python3
"""Historical parent runner for results/release_anisotropy/.

One-sided fitting in this entry point is **superseded**. Frozen parent outputs
live under results/release_anisotropy/ (tag prh-release-alignment-freeze-20260921).
Authoritative nested-budget fits: scripts/run_release_anisotropy_repair.py.

This command will not silently run a different estimator. Default: skip if
summary.json exists. --force refuses to overwrite parent fits. --extract-only
still extracts caches when the parent run is incomplete.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.extract_final import (  # noqa: E402
    extract_language_final,
    final_feature_path,
    slice_vision_final,
    spec_from_manifest,
)
from prh_replication.io_utils import jsonable, write_json  # noqa: E402
from prh_replication.registry import Paths  # noqa: E402
from prh_replication.release_protocol import VIS, load_splits  # noqa: E402

ARCHIVE_MSG = """\
release_anisotropy fitting is archived.

Frozen parent outputs: results/release_anisotropy/ (do not overwrite).
To inspect the superseded optimiser, check out git tag
  prh-release-alignment-freeze-20260921
Authoritative one-sided fits:
  python scripts/run_release_anisotropy_repair.py --work ...
Default skip if summary.json exists. This command does not run the repaired
estimator into the parent directory.
"""


def parse_args():
    p = argparse.ArgumentParser(description="Archived parent release-alignment runner")
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--only-model", default="")
    p.add_argument("--force", action="store_true", help="refused for parent one-sided overwrite")
    return p.parse_args()


def records_from_captions_file(paths: Paths, ids: list[str]) -> list[dict]:
    cap_json = paths.data / "raw" / "coco" / "annotations" / "captions_val2017.json"
    coco = json.loads(cap_json.read_text())
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    recs = []
    for sid in ids:
        image_id = int(str(sid).split("-")[-1])
        anns_sorted = sorted(by_image[image_id], key=lambda a: a["id"])
        recs.append({"sample_id": sid, "caption": anns_sorted[0]["caption"], "image_id": image_id})
    return recs


def records_for_ids(man, ids, paths: Paths):
    recs = man.get("records")
    if recs:
        by_id = recs if isinstance(recs, dict) else {r["sample_id"]: r for r in recs}
        if all(i in by_id for i in ids) and all("caption" in by_id[i] for i in ids):
            return [by_id[i] for i in ids]
    return records_from_captions_file(paths, ids)


def extract_all(paths, specs, splits, records_by_split, only_model, extract_status):
    blockers = []
    for split, ids in splits.items():
        recs = records_by_split[split]
        for v in VIS:
            try:
                slice_vision_final(paths, v, split, ids)
                extract_status[f"vision:{v}:{split}"] = "ok"
            except Exception as e:
                extract_status[f"vision:{v}:{split}"] = f"FAIL:{e}"
                blockers.append({"model": v, "split": split, "error": str(e), "trace": traceback.format_exc()[-1500:]})
    for key, spec in specs.items():
        if only_model and key != only_model:
            continue
        for split, ids in splits.items():
            tag = f"lang:{key}:{split}"
            if extract_status.get(tag) == "ok":
                p = final_feature_path(paths, spec, "coco_val2017", split, ids)
                if p.exists():
                    print(f"skip cached {tag}", flush=True)
                    continue
            recs = records_by_split[split]
            t0 = time.time()
            print(f"EXTRACT {tag} n={len(recs)}", flush=True)
            try:
                extract_language_final(spec, recs, paths, "coco_val2017", split, batch_size=32)
                extract_status[tag] = "ok"
                extract_status[f"{tag}:sec"] = time.time() - t0
            except Exception as e:
                extract_status[tag] = f"FAIL:{type(e).__name__}:{e}"
                blockers.append({"model": key, "split": split, "error": str(e), "trace": traceback.format_exc()[-2000:]})
                print(f"BLOCKER {tag}: {e}", flush=True)
                break
        write_json(paths.results / "release_anisotropy" / "extract_status.json", extract_status)
    return blockers, extract_status


def main():
    args = parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out = paths.results / "release_anisotropy"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.force and not args.smoke:
        print("release_anisotropy already complete; pass --force to rerun", flush=True)
        print("Parent one-sided fits are superseded; see README.md and run_release_anisotropy_repair.py", flush=True)
        return
    if args.force:
        print(ARCHIVE_MSG, flush=True)
        print("Refusing --force: will not overwrite frozen parent fits.", flush=True)
        sys.exit(2)
    if not args.extract_only:
        print(ARCHIVE_MSG, flush=True)
        sys.exit(2)

    design = json.loads((ROOT / "configs" / "release_anisotropy.json").read_text())
    manifest = json.loads((ROOT / "data" / "manifests" / "release_models.json").read_text())
    man, splits = load_splits(paths, ROOT)
    rows = list(manifest["base_panel"]) + list(manifest["supplementary_qwen3x"])
    specs = {r["key"]: spec_from_manifest(r) for r in rows}
    write_json(out / "design.json", design)
    write_json(out / "model_inventory.json", manifest)

    extract_status = {}
    sp = out / "extract_status.json"
    if sp.exists():
        extract_status = json.loads(sp.read_text())
    blockers = []
    if not args.skip_extract:
        recs = {split: records_for_ids(man, ids, paths) for split, ids in splits.items()}
        b2, extract_status = extract_all(paths, specs, splits, recs, args.only_model, extract_status)
        blockers.extend(b2)
        write_json(out / "extract_status.json", extract_status)
        write_json(out / "blockers.json", jsonable(blockers))
    print("extract-only done", flush=True)


if __name__ == "__main__":
    main()
