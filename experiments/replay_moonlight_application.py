"""Application parity for one archived Qwen → OLMo → Moonlight steering run.

This does not fit CCA or a decoder. It loads the matched-pair encoders, the
archived Qwen direction, the Moonlight decoder, and the saved directions from
one artifact root. Review activations, bridge rows, and generation scores are
not read. A missing root is not-run.

The two-row table exists so ``fit_steering`` can read the archived Qwen vector
as a difference of means. That is application of a saved vector, not a refit
from reviews.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from rho_cca import (
    DifferenceInMeans,
    Endpoint,
    FeatureTable,
    ModelPair,
    PairedDataset,
    RhoCCAFit,
    Split,
    SteeringConfig,
    fit_steering,
    load_fit,
    match_norm,
    save_fit,
)
from rho_cca.preprocess import PreparedSpace


PAIR = Path("imdb_matched_pair_cca_transfer_v1") / "artifacts"
MOON = Path("imdb_cca_moonlight_third_model_v1") / "artifacts"
ATOL = 1e-8
RTOL = 1e-8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _space() -> PreparedSpace:
    return PreparedSpace(None, False, None, None, None, None, None)


def _table(endpoint: Endpoint, sample_ids: tuple[str, ...], values: np.ndarray) -> FeatureTable:
    return FeatureTable(endpoint, sample_ids, sample_ids, values)


def _split(dataset: PairedDataset) -> PairedDataset:
    split = Split(
        train_ids=dataset.sample_ids,
        validation_ids=(),
        test_ids=(),
        group_by_sample=tuple((sample_id, sample_id) for sample_id in dataset.sample_ids),
        seed=None,
        fingerprint=dataset.fingerprint(),
        method="archived direction supplied as a two-point difference",
    )
    return PairedDataset(dataset.source, dataset.partner, split)


def _alignment(source: Endpoint, partner: Endpoint, dataset: PairedDataset, e_source: np.ndarray, e_partner: np.ndarray) -> RhoCCAFit:
    width = e_source.shape[0]
    return RhoCCAFit(
        source=source,
        partner=partner,
        source_space=_space(),
        partner_space=_space(),
        l_source=np.asarray(e_source, dtype=np.float64).T,
        l_partner=np.asarray(e_partner, dtype=np.float64).T,
        rho=np.ones(width, dtype=np.float64),
        weights=np.ones(width, dtype=np.float64),
        eta=0.01,
        beta=1.0,
        floor=1e-6,
        tau_source=0.0,
        tau_partner=0.0,
        n_negative_rho=0,
        n_floored=0,
        algebraic_failure=False,
        train_ids=dataset.sample_ids,
        fingerprint=dataset.fingerprint(),
    )


def _gap(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _close(actual, expected) -> bool:
    error = _gap(actual, expected)
    scale = max(1.0, float(np.max(np.abs(expected))))
    return error <= ATOL or error <= RTOL * scale


def run(root: Path) -> dict:
    required = {
        "maps": root / PAIR / "maps.npz",
        "dir_qwen": root / PAIR / "dir_qwen_native.npy",
        "dir_olmo": root / PAIR / "dir_olmo_aligned64.npy",
        "geometry": root / PAIR / "geometry.json",
        "decoder": root / MOON / "decoder.npz",
        "decoder_json": root / MOON / "decoder.json",
        "z_edit": root / MOON / "z_edit.npy",
        "dir_transfer": root / MOON / "dir_transfer.npy",
        "frozen_command": root / MOON / "frozen_command.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        return {"status": "not-run", "missing": missing}
    hashes = {name: _sha(path) for name, path in required.items()}
    maps = np.load(required["maps"])
    e_qwen = np.asarray(maps["E_qwen"], dtype=np.float64)
    e_olmo = np.asarray(maps["E_olmo"], dtype=np.float64)
    source_vector = np.asarray(np.load(required["dir_qwen"]), dtype=np.float64).reshape(-1)
    olmo_direction = np.asarray(np.load(required["dir_olmo"]), dtype=np.float64).reshape(-1)
    z_edit = np.asarray(np.load(required["z_edit"]), dtype=np.float64).reshape(-1)
    decoder = np.load(required["decoder"])
    decoded_weight = np.asarray(decoder["D"], dtype=np.float64)
    intercept = np.asarray(decoder["intercept"], dtype=np.float64)
    moonlight = np.asarray(np.load(required["dir_transfer"]), dtype=np.float64).reshape(-1)
    source = Endpoint("Qwen2.5-7B", "imdb-matched-pair-cca-transfer-v1", None, "encoder-rows", "archived-map", source_vector.shape[0])
    partner = Endpoint("OLMo-3-1025-7B", "imdb-matched-pair-cca-transfer-v1", None, "encoder-rows", "archived-map", olmo_direction.shape[0])
    ids = ("archived-positive", "archived-negative")
    bare = PairedDataset(
        _table(source, ids, np.vstack([source_vector, np.zeros_like(source_vector)])),
        _table(partner, ids, np.zeros((2, partner.width))),
    )
    dataset = _split(bare)
    alignment = _alignment(source, partner, dataset, e_qwen, e_olmo)
    fitted = fit_steering(
        alignment,
        ModelPair(source, partner),
        DifferenceInMeans(("archived-positive",), ("archived-negative",)),
        dataset,
        config=SteeringConfig(decoder="pseudoinverse", weights="equal"),
    )
    raw_reference = source_vector @ e_qwen.T @ np.linalg.pinv(e_olmo.T, rcond=1e-8)
    unit = source_vector / np.linalg.norm(source_vector)
    command_reference = unit @ e_qwen.T
    matched = match_norm(fitted.target_vector, float(np.linalg.norm(olmo_direction)))
    decoder_meta = json.loads(required["decoder_json"].read_text())
    moonlight_raw = z_edit @ decoded_weight.T
    archived_raw = moonlight * (float(decoder_meta["raw_norm"]) / float(np.linalg.norm(moonlight)))
    moonlight_matched = match_norm(moonlight_raw, float(np.linalg.norm(moonlight)))
    with tempfile.TemporaryDirectory(prefix="rho-cca-application-") as temporary:
        restored = load_fit(save_fit(fitted, Path(temporary) / "steering.npz"))
    checks = {
        "olmo_raw_max_abs": _gap(fitted.target_vector, raw_reference),
        "olmo_matched_max_abs": _gap(matched.applied, olmo_direction),
        "frozen_command_max_abs": _gap(command_reference, z_edit),
        "moonlight_raw_max_abs": _gap(moonlight_raw, archived_raw),
        "moonlight_matched_max_abs": _gap(moonlight_matched.applied, moonlight),
        "reload_max_abs": _gap(restored.target_vector, fitted.target_vector),
        "intercept_excluded": _gap(moonlight_raw, moonlight_raw + intercept) > 1e-6,
    }
    passed = (
        _close(fitted.target_vector, raw_reference)
        and _close(matched.applied, olmo_direction)
        and _close(command_reference, z_edit)
        and _close(moonlight_raw, archived_raw)
        and _close(moonlight_matched.applied, moonlight)
        and checks["intercept_excluded"]
        and checks["reload_max_abs"] <= ATOL
    )
    return {
        "status": "pass" if passed else "fail",
        "parity": "application-only",
        "fit_steering": "archived Qwen vector supplied as a two-point difference; encoders were loaded, not fit",
        "attach_steering_target": "not-run",
        "missing_for_fitting": "review activation differences and unlabelled bridge activation rows",
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "checks": checks,
        "sha256": hashes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True, help="Parent of the two result directories. Nothing is downloaded.")
    args = parser.parse_args(argv)
    report = run(args.artifact_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
