#!/usr/bin/env python3
"""Checksum inventory for the release-alignment freeze (work disk)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

WORK = Path("/mnt/sdb1/prh-replication-work")
REPAIR = WORK / "results" / "release_anisotropy_repair"
PARENT = WORK / "results" / "release_anisotropy"
OUT = WORK / "freeze" / "prh-release-alignment-freeze-20260921"
SNAPSHOT_ID = "prh-release-alignment-freeze-20260921"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rec(path: Path, role: str, status: str) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
        "status": status,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv: list[dict] = []
    for p in sorted(REPAIR.rglob("*")):
        if p.is_file():
            inv.append(rec(p, "repair_outputs", "authoritative"))
    superseded = {
        "summary.json",
        "pair_eval.json",
        "headline_rho.json",
        "consistency.json",
        "transfer.json",
        "lopo.json",
        "signatures.json",
        "shuffle_fit.json",
        "stability.json",
        "onesided_fits",
    }
    reused = {"native_matrix.json", "native_anisotropy.json", "two_sided.json", "pca.json"}
    for p in sorted(PARENT.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if name in reused or p.relative_to(PARENT).as_posix() in reused:
            inv.append(rec(p, "parent_native_or_two_sided", "reused"))
        elif any(s in p.as_posix() for s in superseded):
            inv.append(rec(p, "parent_optimisation_dependent", "superseded"))
        else:
            inv.append(rec(p, "parent_historical", "historical_retained"))
    feature_hashes = json.loads((REPAIR / "feature_hashes.json").read_text())
    pca = json.loads((REPAIR / "pca_provenance.json").read_text())
    native = json.loads((REPAIR / "native_reuse.json").read_text())
    payload = {
        "snapshot_id": SNAPSHOT_ID,
        "work_root": str(WORK),
        "authoritative_results": str(REPAIR),
        "superseded_optimisation_results": str(PARENT),
        "feature_caches": feature_hashes,
        "pca_provenance": pca,
        "native_reuse": native,
        "inventory": inv,
        "notes": [
            "Do not delete results/release_anisotropy; optimisation-dependent files there are superseded.",
            "Rerun of completed experiments requires --force (or --resume-after for the repair script).",
            "Direct vs contracted CKA a disagrees at ~2e-5 on sampled pairs; not treated as a new scientific finding.",
        ],
    }
    (OUT / "INVENTORY.json").write_text(json.dumps(payload, indent=2))
    lines = ["# sha256  bytes  status  path"]
    for r in inv:
        lines.append(f"{r['sha256']}  {r['bytes']:12d}  {r['status']:12s}  {r['path']}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} n={len(inv)}")


if __name__ == "__main__":
    main()
