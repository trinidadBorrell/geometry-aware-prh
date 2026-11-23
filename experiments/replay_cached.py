"""Cached geometry replay through the public API.

The shipped fixture is a normal equation, not a Flickr or nocaps score.
Historical audits are run only when their feature arrays are mounted. Missing
arrays are reported as not-run; this script does not invent a passing median.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from rho_cca import (
    Endpoint,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RidgeConfig,
    Split,
    fit_ridge,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments" / "fixtures" / "ridge_normal_equation.npz"

HISTORICAL = (
    ("flickr ridge median excess lift", "+0.1647", "frozen_partner_cka_audit_v1"),
    ("flickr one-sided rho-CCA median excess lift", "+0.1381", "frozen_partner_cka_audit_v1"),
    ("nocaps ridge median excess lift", "+0.1312", "rho_cca_nocaps_frozen_partner_v1"),
    ("nocaps rho-CCA median excess lift", "+0.1207", "rho_cca_nocaps_frozen_partner_v1"),
    ("nocaps weighted minus equal", "+0.1679", "rho_cca_nocaps_frozen_partner_v1"),
)


def _endpoint(payload: dict) -> Endpoint:
    return Endpoint(
        payload["model"],
        payload["revision"],
        payload["layer"],
        payload["site"],
        payload["pooling"],
        int(payload["width"]),
    )


def run_shipped_fixture() -> dict:
    if not FIXTURE.is_file():
        return {"name": "shipped ridge fixture", "status": "not-run", "detail": f"missing {FIXTURE}"}
    with np.load(FIXTURE, allow_pickle=False) as blob:
        meta = json.loads(str(blob["meta"].item()))
        x = np.asarray(blob["source"], dtype=np.float64)
        y = np.asarray(blob["partner"], dtype=np.float64)
        expected = np.asarray(blob["weight"], dtype=np.float64)
    source = _endpoint(meta["source"])
    partner = _endpoint(meta["partner"])
    ids = tuple(meta["sample_ids"])
    groups = tuple(meta["group_ids"])
    bare = PairedDataset(FeatureTable(source, ids, groups, x), FeatureTable(partner, ids, groups, y))
    split = Split(
        train_ids=tuple(meta["train_ids"]),
        validation_ids=tuple(meta["validation_ids"]),
        test_ids=tuple(meta["test_ids"]),
        group_by_sample=tuple((sample_id, group_id) for sample_id, group_id in zip(ids, groups, strict=True)),
        seed=meta["seed"],
        fingerprint=bare.fingerprint(),
        method="published-fixture",
    )
    data = PairedDataset(bare.source, bare.partner, split)
    fit = fit_ridge(ModelPair(source, partner), data, config=RidgeConfig(t=float(meta["t"])))
    gap = float(np.max(np.abs(fit.weight - expected)))
    status = "pass" if gap <= 1e-8 else "fail"
    return {"name": "shipped ridge fixture", "status": status, "max_abs_weight_gap": gap, "t": fit.t}


def historical_status() -> list[dict]:
    root = os.environ.get("RHO_CCA_ARTIFACT_ROOT", "")
    rows = []
    for name, report_check, directory in HISTORICAL:
        mounted = bool(root) and (Path(root) / directory).is_dir()
        rows.append(
            {
                "name": name,
                "status": "not-run",
                "report_check": report_check,
                "detail": (
                    f"mounted {directory}, but this replay has no feature adapter for that tree"
                    if mounted
                    else f"set RHO_CCA_ARTIFACT_ROOT to a directory containing {directory}"
                ),
            }
        )
    return rows


def main() -> int:
    shipped = run_shipped_fixture()
    print(json.dumps(shipped, sort_keys=True))
    failures = shipped["status"] == "fail"
    for row in historical_status():
        print(json.dumps(row, sort_keys=True))
    return 1 if failures or shipped["status"] == "not-run" else 0


if __name__ == "__main__":
    sys.exit(main())
