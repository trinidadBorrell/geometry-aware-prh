"""Procedure replication of fixed CCA reliability metrics (original embeddings)."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from prh_replication.cca_reliability_replication import (
    CONDITIONS,
    CONTRASTS,
    build_metrics,
    evaluate_pair,
)
from prh_replication.feature_prep import clip_threshold, prepare_with_threshold
from prh_replication.crosscov_weighted_cka import fit_raw_and_cca
from prh_replication.io_utils import load_features, write_json
from prh_replication.registry import MODELS, Paths
from prh_replication.sae_crosscov import fit_pca, project
from prh_replication.sae_crosscov_study import _load_panel

CONFIG_NAME = "cca_reliability_replication_v1.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _prepare(raw: np.ndarray, threshold: float) -> np.ndarray:
    return prepare_with_threshold(torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)), threshold).numpy().astype(np.float64)


def _pair_role(language: str, vision: str, config: dict) -> str:
    new_l = language in config["new_languages"]
    new_v = vision in config["new_visions"]
    if (not new_l) and (not new_v):
        return "anchor"
    if new_l and new_v:
        return "both_new"
    return "mixed"


def _load_coco_lang(paths: Paths, key: str, split: str, sample_ids: list[str]) -> np.ndarray:
    path = paths.features / "coco_val2017_caption_reliability_v1" / split / key / "captions.pt"
    cached = load_features(path)
    ids = list(cached["sample_ids"])
    index = {sid: i for i, sid in enumerate(ids)}
    order = [index[sid] for sid in sample_ids]
    return np.asarray(cached["feats"], dtype=np.float64)[order]


def _load_coco_vis(paths: Paths, key: str, split: str, sample_ids: list[str]) -> np.ndarray:
    # find .pt under final_block
    root = paths.features / "coco_val2017_final_block_pre_norm" / split / key
    pts = sorted(root.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(root)
    cached = load_features(pts[0])
    ids = list(cached["sample_ids"])
    index = {sid: i for i, sid in enumerate(ids)}
    order = [index[sid] for sid in sample_ids]
    feats = np.asarray(cached["feats"], dtype=np.float64)
    if feats.ndim == 3:
        feats = feats[:, -1, :]
    return feats[order]


def _load_flickr_lang(paths: Paths, key: str, split: str, sample_ids: list[str], config: dict) -> np.ndarray:
    root = paths.features / Path(config["flickr_feature_root"]).name / split / key
    pts = list(root.glob("captions.pt")) + list(root.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"missing Flickr language features: {root}")
    cached = load_features(pts[0])
    ids = list(cached["sample_ids"])
    index = {sid: i for i, sid in enumerate(ids)}
    order = [index[sid] for sid in sample_ids]
    return np.asarray(cached["feats"], dtype=np.float64)[order]


def _load_flickr_vis(paths: Paths, key: str, split: str, sample_ids: list[str], config: dict) -> np.ndarray:
    root = paths.features / Path(config["flickr_feature_root"]).name / split / key
    pts = sorted(root.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"missing Flickr vision features: {root}")
    cached = load_features(pts[0])
    ids = list(cached["sample_ids"])
    index = {sid: i for i, sid in enumerate(ids)}
    order = [index[sid] for sid in sample_ids]
    feats = np.asarray(cached["feats"], dtype=np.float64)
    if feats.ndim == 3:
        feats = feats[:, -1, :]
    return feats[order]


def run(work: Path, repo: Path, force: bool = False, sections: list[str] | None = None) -> dict:
    started = time.perf_counter()
    paths = Paths(work=work, repo=repo)
    config = json.loads((repo / "configs" / CONFIG_NAME).read_text())
    out = paths.results / "cca_reliability_replication_v1"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("SPECIFICATION.md", "frozen_specification.json", "split_manifest_coco.json", "split_manifest_flickr30k.json"):
        src = repo / "results" / "cca_reliability_replication_v1" / name
        if src.exists():
            (out / name).write_text(src.read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if (out / "summary.json").exists() and not force:
        print("already complete; pass --force", flush=True)
        return json.loads((out / "summary.json").read_text())

    sections = sections or ["A", "B", "C", "D"]
    coco = json.loads((out / "split_manifest_coco.json").read_text())
    flickr = json.loads((out / "split_manifest_flickr30k.json").read_text())

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    contrast_rows: list[dict] = []
    caption_rows: list[dict] = []
    shuffle_rows: list[dict] = []
    null_rows: list[dict] = []
    subsample_rows: list[dict] = []
    repro_gaps: list[dict] = []

    # archived original-only baselines for COCO anchors
    abl = paths.results / "cca_metric_diagonal_ablation_v1" / "test_metrics.csv"
    parent = paths.results / "crosscov_weighted_cka_v1" / "test_metrics.csv"
    archived = _load_archived(abl, parent)

    panel_pairs = []
    for language in config["anchor_languages"]:
        for vision in config["anchor_visions"]:
            panel_pairs.append((language, vision, "anchor"))
    for language in config["new_languages"]:
        for vision in config["new_visions"]:
            panel_pairs.append((language, vision, "both_new"))

    want_coco = any(s in sections for s in ("A", "B"))
    want_flickr = any(s in sections for s in ("C", "D"))

    if want_coco:
        _run_dataset(
            "coco_val2017",
            coco["train_ids"],
            coco["test_ids"],
            coco.get("test_ids_size_matched_1000"),
            panel_pairs,
            config,
            paths,
            out,
            art,
            archived,
            metric_rows,
            contrast_rows,
            caption_rows,
            shuffle_rows,
            null_rows,
            subsample_rows,
            repro_gaps,
            sections,
            loader="coco",
        )

    if want_flickr:
        try:
            _run_dataset(
                "flickr30k",
                flickr["train_ids"],
                flickr["test_ids"],
                None,
                panel_pairs,
                config,
                paths,
                out,
                art,
                {},
                metric_rows,
                contrast_rows,
                caption_rows,
                shuffle_rows,
                null_rows,
                subsample_rows,
                repro_gaps,
                sections,
                loader="flickr",
            )
        except FileNotFoundError as exc:
            write_json(out / "flickr_blocker.json", {"error": str(exc), "note": "Complete A/B; extract Flickr features then rerun C/D"})
            print(f"Flickr blocked: {exc}", flush=True)

    _write_csv(out / "test_metrics.csv", metric_rows)
    _write_csv(out / "paired_contrasts.csv", contrast_rows)
    _write_csv(out / "single_caption_metrics.csv", caption_rows)
    _write_csv(out / "shuffled_fit_controls.csv", shuffle_rows)
    _write_csv(out / "test_pairing_nulls.csv", null_rows)
    _write_csv(out / "subsample_metrics.csv", subsample_rows)
    write_json(out / "reproduction_checks.json", {"gaps": repro_gaps, "max_cka": max((g.get("cka_gap", 0) for g in repro_gaps), default=0.0)})

    summary = _summarize(config, metric_rows, contrast_rows, caption_rows, shuffle_rows, repro_gaps)
    summary["elapsed_s"] = time.perf_counter() - started
    write_json(out / "summary.json", summary)
    (out / "RESULTS.md").write_text(_results_md(summary, metric_rows, contrast_rows))
    write_json(
        out / "provenance.json",
        {
            "config_sha256": _sha256_file(repo / "configs" / CONFIG_NAME),
            "frozen_specification_sha256": _sha256_file(out / "frozen_specification.json"),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", ""),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "elapsed_s": summary["elapsed_s"],
            "reproduction_command": (
                "PYTHONPATH=src python scripts/run_cca_reliability_replication.py "
                "--work /mnt/sdb1/prh-replication-work --repo /mnt/sdb1/prh-replication-work/repo"
            ),
        },
    )
    _make_figures(fig, metric_rows, contrast_rows)
    print(json.dumps(summary.get("answers", {}), indent=2), flush=True)
    return summary


def _load_archived(abl_path: Path, parent_path: Path) -> dict:
    out = {}
    if abl_path.exists():
        for row in csv.DictReader(abl_path.open()):
            if row["branch"] != "original":
                continue
            key = (row["language"], row["vision"], row["condition"])
            out[key] = {"cka": float(row["cka"]), "mknn": float(row["mknn"])}
    if parent_path.exists():
        for row in csv.DictReader(parent_path.open()):
            if row.get("branch") != "original" or str(row.get("d")) != "64":
                continue
            if row.get("arm") == "B2_unit_cca":
                out[(row["language"], row["vision"], "W")] = {"cka": float(row["cka"]), "mknn": float(row["mnn_k10"])}
    return out


def _run_dataset(
    dataset,
    train_ids,
    test_ids,
    size_matched_ids,
    panel_pairs,
    config,
    paths,
    out,
    art,
    archived,
    metric_rows,
    contrast_rows,
    caption_rows,
    shuffle_rows,
    null_rows,
    subsample_rows,
    repro_gaps,
    sections,
    loader,
):
    k = int(config["knn_k"])
    n_test = len(test_ids)
    n_keep = int(round(float(config["test_subsample_fraction"]) * n_test))
    sub_rng = np.random.Generator(np.random.PCG64(int(config["test_subsample_seed"]) + (0 if dataset.startswith("coco") else 17)))
    subsample_indices = [np.sort(sub_rng.choice(n_test, size=n_keep, replace=False)) for _ in range(int(config["n_test_subsamples"]))]
    np.save(out / f"subsample_indices_{dataset}.npy", np.stack(subsample_indices))

    for language, vision, role in panel_pairs:
        section = "A" if (dataset.startswith("coco") and role == "anchor") else "B" if (dataset.startswith("coco") and role == "both_new") else "C" if role == "anchor" else "D"
        if section not in sections:
            continue
        print(f"{section} {dataset} {language} × {vision}", flush=True)
        try:
            if loader == "coco":
                lang_tr_raw = _load_coco_lang(paths, language, "train", train_ids)
                lang_te_raw = _load_coco_lang(paths, language, "test", test_ids)
                vis_tr_raw = _load_coco_vis(paths, vision, "train", train_ids)
                vis_te_raw = _load_coco_vis(paths, vision, "test", test_ids)
            else:
                lang_tr_raw = _load_flickr_lang(paths, language, "train", train_ids, config)
                lang_te_raw = _load_flickr_lang(paths, language, "test", test_ids, config)
                vis_tr_raw = _load_flickr_vis(paths, vision, "train", train_ids, config)
                vis_te_raw = _load_flickr_vis(paths, vision, "test", test_ids, config)
        except FileNotFoundError as exc:
            print(f"  skip missing features: {exc}", flush=True)
            continue

        # train-fitted clip
        flat_lang = lang_tr_raw.reshape(-1, lang_tr_raw.shape[-1])
        thresh_l = float(clip_threshold(torch.from_numpy(np.ascontiguousarray(flat_lang, dtype=np.float32)), float(config["clip_q"])))
        thresh_v = float(clip_threshold(torch.from_numpy(np.ascontiguousarray(vis_tr_raw, dtype=np.float32)), float(config["clip_q"])))
        lang_tr = _prepare(lang_tr_raw.reshape(-1, lang_tr_raw.shape[-1]), thresh_l).reshape(lang_tr_raw.shape)
        lang_te = _prepare(lang_te_raw.reshape(-1, lang_te_raw.shape[-1]), thresh_l).reshape(lang_te_raw.shape)
        vis_tr = _prepare(vis_tr_raw, thresh_v)
        vis_te = _prepare(vis_te_raw, thresh_v)
        lang_tr_c = lang_tr.mean(axis=1)
        lang_te_c = lang_te.mean(axis=1)

        pca_l = fit_pca(lang_tr_c, int(config["pca_d"]))
        pca_v = fit_pca(vis_tr, int(config["pca_d"]))
        width = int(min(pca_l["width"], pca_v["width"], int(config["pca_d"])))
        if width < int(config["pca_d"]):
            print(f"  unsupported PCA width {width}; marking unavailable", flush=True)
            metric_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "language": language,
                    "vision": vision,
                    "role": role,
                    "condition": "UNAVAILABLE",
                    "cka": float("nan"),
                    "mknn": float("nan"),
                    "note": f"pca_width={width}",
                }
            )
            continue
        if width != pca_l["width"] or width != pca_v["width"]:
            pca_l = fit_pca(lang_tr_c, width)
            pca_v = fit_pca(vis_tr, width)

        a_tr = project(lang_tr_c, pca_l["mean"], pca_l["components"])
        b_tr = project(vis_tr, pca_v["mean"], pca_v["components"])
        a_te = project(lang_te_c, pca_l["mean"], pca_l["components"])
        b_te = project(vis_te, pca_v["mean"], pca_v["components"])
        maps = fit_raw_and_cca(a_tr, b_tr, float(config["eta"]), float(config["support_tol"]))
        mets = build_metrics(
            maps["cca"]["L_A"],
            maps["cca"]["L_B"],
            maps["rho"],
            float(config["rho_floor"]),
            bool(config["normalize_weights_by_mean"]),
        )
        np.savez_compressed(
            art / f"{dataset}_{language}_{vision}_projectors.npz",
            pca_lang_mean=pca_l["mean"],
            pca_lang_components=pca_l["components"],
            pca_vis_mean=pca_v["mean"],
            pca_vis_components=pca_v["components"],
            L_A=maps["cca"]["L_A"],
            L_B=maps["cca"]["L_B"],
            s=maps["cca"]["s"],
            rho=maps["rho"],
            w=mets["w"],
            M_A=mets["M_A"],
            M_B=mets["M_B"],
            D_A=mets["D_A"],
            D_B=mets["D_B"],
            thresh_lang=thresh_l,
            thresh_vis=thresh_v,
        )

        cell = {}
        for kind in CONDITIONS:
            sc = evaluate_pair(
                a_te,
                b_te,
                kind,
                kind,
                maps["cca"]["L_A"],
                maps["cca"]["L_B"],
                mets["w"],
                mets["M_A"],
                mets["M_B"],
                mets["D_A"],
                mets["D_B"],
                knn_k=k,
            )
            cell[kind] = sc
            metric_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "language": language,
                    "vision": vision,
                    "role": role,
                    "condition": kind,
                    "cka": sc["cka"],
                    "mknn": sc["mknn"],
                    "pr_left": sc["pr_left"],
                    "pr_right": sc["pr_right"],
                    "entropy_left": sc["entropy_left"],
                    "entropy_right": sc["entropy_right"],
                    "top_energy_left": sc["top_energy_left"],
                    "top_energy_right": sc["top_energy_right"],
                    "rank_metric_a": sc["rank_metric_a"],
                    "rank_metric_b": sc["rank_metric_b"],
                    "n_rows_with_kth_tie_left": sc["n_rows_with_kth_tie_left"],
                    "n_rows_with_kth_tie_right": sc["n_rows_with_kth_tie_right"],
                    "n_zero_distance_offdiag_counts": sc["n_zero_distance_offdiag_counts"],
                    "gallery_n": n_test,
                }
            )

        for cname, left, right in CONTRASTS:
            contrast_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "language": language,
                    "vision": vision,
                    "role": role,
                    "contrast": cname,
                    "delta_cka": cell[left]["cka"] - cell[right]["cka"],
                    "delta_mknn": cell[left]["mknn"] - cell[right]["mknn"],
                }
            )

        # COCO anchor reproduction
        if dataset.startswith("coco") and role == "anchor":
            mapping = {"I": "identity_identity", "D": "diagonal_diagonal", "F": "full_full", "W": "W"}
            for kind, arch_key in mapping.items():
                ref = archived.get((language, vision, arch_key))
                if ref is None:
                    continue
                g_cka = abs(cell[kind]["cka"] - ref["cka"])
                g_m = abs(cell[kind]["mknn"] - ref["mknn"])
                repro_gaps.append(
                    {
                        "language": language,
                        "vision": vision,
                        "condition": kind,
                        "cka_gap": g_cka,
                        "mknn_gap": g_m,
                    }
                )

        # single caption I and F
        for pos in range(lang_te.shape[1]):
            a_pos = project(lang_te[:, pos, :], pca_l["mean"], pca_l["components"])
            for kind in ("I", "F"):
                sc = evaluate_pair(
                    a_pos,
                    b_te,
                    kind,
                    kind,
                    maps["cca"]["L_A"],
                    maps["cca"]["L_B"],
                    mets["w"],
                    mets["M_A"],
                    mets["M_B"],
                    mets["D_A"],
                    mets["D_B"],
                    knn_k=k,
                )
                caption_rows.append(
                    {
                        "section": section,
                        "dataset": dataset,
                        "language": language,
                        "vision": vision,
                        "role": role,
                        "caption_position": pos,
                        "condition": kind,
                        "cka": sc["cka"],
                        "mknn": sc["mknn"],
                        "pooling": "one_caption",
                    }
                )
        for kind in ("I", "F"):
            caption_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "language": language,
                    "vision": vision,
                    "role": role,
                    "caption_position": -1,
                    "condition": kind,
                    "cka": cell[kind]["cka"],
                    "mknn": cell[kind]["mknn"],
                    "pooling": "five_caption_mean",
                }
            )

        # shuffled-fit controls for F
        for seed in config["shuffle_seeds"]:
            rng = np.random.Generator(np.random.PCG64(int(seed)))
            order = rng.permutation(a_tr.shape[0])
            maps_s = fit_raw_and_cca(a_tr, b_tr[order], float(config["eta"]), float(config["support_tol"]))
            mets_s = build_metrics(
                maps_s["cca"]["L_A"],
                maps_s["cca"]["L_B"],
                maps_s["rho"],
                float(config["rho_floor"]),
                bool(config["normalize_weights_by_mean"]),
            )
            sc_s = evaluate_pair(
                a_te,
                b_te,
                "F",
                "F",
                maps_s["cca"]["L_A"],
                maps_s["cca"]["L_B"],
                mets_s["w"],
                mets_s["M_A"],
                mets_s["M_B"],
                mets_s["D_A"],
                mets_s["D_B"],
                knn_k=k,
            )
            shuffle_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "language": language,
                    "vision": vision,
                    "role": role,
                    "seed": int(seed),
                    "cka_genuine": cell["F"]["cka"],
                    "cka_shuffled_fit": sc_s["cka"],
                    "genuine_minus_shuffled_cka": cell["F"]["cka"] - sc_s["cka"],
                    "mknn_genuine": cell["F"]["mknn"],
                    "mknn_shuffled_fit": sc_s["mknn"],
                    "genuine_minus_shuffled_mknn": cell["F"]["mknn"] - sc_s["mknn"],
                }
            )

        # frozen-transform test pairing null
        for seed in config["shuffle_seeds"]:
            rng = np.random.Generator(np.random.PCG64(int(seed) + 1000))
            order = rng.permutation(a_te.shape[0])
            sc_n = evaluate_pair(
                a_te,
                b_te[order],
                "F",
                "F",
                maps["cca"]["L_A"],
                maps["cca"]["L_B"],
                mets["w"],
                mets["M_A"],
                mets["M_B"],
                mets["D_A"],
                mets["D_B"],
                knn_k=k,
            )
            null_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "language": language,
                    "vision": vision,
                    "role": role,
                    "seed": int(seed),
                    "cka_genuine": cell["F"]["cka"],
                    "cka_shuffled_test": sc_n["cka"],
                    "genuine_minus_null_cka": cell["F"]["cka"] - sc_n["cka"],
                    "mknn_genuine": cell["F"]["mknn"],
                    "mknn_shuffled_test": sc_n["mknn"],
                    "genuine_minus_null_mknn": cell["F"]["mknn"] - sc_n["mknn"],
                }
            )

        for s_i, idx in enumerate(subsample_indices):
            for kind in CONDITIONS:
                sc = evaluate_pair(
                    a_te[idx],
                    b_te[idx],
                    kind,
                    kind,
                    maps["cca"]["L_A"],
                    maps["cca"]["L_B"],
                    mets["w"],
                    mets["M_A"],
                    mets["M_B"],
                    mets["D_A"],
                    mets["D_B"],
                    knn_k=k,
                )
                subsample_rows.append(
                    {
                        "section": section,
                        "dataset": dataset,
                        "language": language,
                        "vision": vision,
                        "role": role,
                        "subsample": int(s_i),
                        "condition": kind,
                        "cka": sc["cka"],
                        "mknn": sc["mknn"],
                    }
                )

        # size-matched COCO companion for absolute mKNN reporting
        if size_matched_ids is not None and loader == "coco":
            id_to_te = {sid: i for i, sid in enumerate(test_ids)}
            idx_m = np.array([id_to_te[sid] for sid in size_matched_ids], dtype=np.int64)
            for kind in CONDITIONS:
                sc = evaluate_pair(
                    a_te[idx_m],
                    b_te[idx_m],
                    kind,
                    kind,
                    maps["cca"]["L_A"],
                    maps["cca"]["L_B"],
                    mets["w"],
                    mets["M_A"],
                    mets["M_B"],
                    mets["D_A"],
                    mets["D_B"],
                    knn_k=k,
                )
                metric_rows.append(
                    {
                        "section": section,
                        "dataset": "coco_val2017_size_matched_1000",
                        "language": language,
                        "vision": vision,
                        "role": role,
                        "condition": kind,
                        "cka": sc["cka"],
                        "mknn": sc["mknn"],
                        "gallery_n": 1000,
                    }
                )


def _summarize(config, metric_rows, contrast_rows, caption_rows, shuffle_rows, repro_gaps) -> dict:
    def mean_contrast(section, role, contrast, metric):
        vals = [
            float(r[metric])
            for r in contrast_rows
            if r["section"] == section and r["role"] == role and r["contrast"] == contrast
        ]
        return float(np.mean(vals)) if vals else float("nan")

    answers = {
        "max_anchor_repro_cka_gap": float(max((g["cka_gap"] for g in repro_gaps), default=0.0)),
        "max_anchor_repro_mknn_gap": float(max((g["mknn_gap"] for g in repro_gaps), default=0.0)),
        "coco_both_new_F_minus_I_cka": mean_contrast("B", "both_new", "F_minus_I", "delta_cka"),
        "coco_both_new_F_minus_I_mknn": mean_contrast("B", "both_new", "F_minus_I", "delta_mknn"),
        "coco_both_new_F_minus_W_cka": mean_contrast("B", "both_new", "F_minus_W", "delta_cka"),
        "coco_both_new_F_minus_D_cka": mean_contrast("B", "both_new", "F_minus_D", "delta_cka"),
        "flickr_anchor_F_minus_I_cka": mean_contrast("C", "anchor", "F_minus_I", "delta_cka"),
        "flickr_both_new_F_minus_I_cka": mean_contrast("D", "both_new", "F_minus_I", "delta_cka"),
        "flickr_both_new_F_minus_I_mknn": mean_contrast("D", "both_new", "F_minus_I", "delta_mknn"),
        "mean_genuine_minus_shuffled_fit_cka": float(np.mean([float(r["genuine_minus_shuffled_cka"]) for r in shuffle_rows]))
        if shuffle_rows
        else float("nan"),
    }
    return {"config_name": config["name"], "answers": answers, "n_metric_rows": len(metric_rows)}


def _results_md(summary, metric_rows, contrast_rows) -> str:
    a = summary["answers"]
    lines = [
        "# CCA reliability procedure replication — RESULTS",
        "",
        "Exploratory. Original embeddings only. Sections A–D separated.",
        "",
        f"Max COCO anchor repro gaps: CKA={a['max_anchor_repro_cka_gap']:.3e}, mKNN={a['max_anchor_repro_mknn_gap']:.3e}.",
        "",
        f"- COCO both-new F−I: ΔCKA={a['coco_both_new_F_minus_I_cka']:.4f}, ΔmKNN={a['coco_both_new_F_minus_I_mknn']:.4f}",
        f"- COCO both-new F−W CKA={a['coco_both_new_F_minus_W_cka']:.4f}, F−D CKA={a['coco_both_new_F_minus_D_cka']:.4f}",
        f"- Flickr anchor F−I CKA={a['flickr_anchor_F_minus_I_cka']:.4f}",
        f"- Flickr both-new F−I: ΔCKA={a['flickr_both_new_F_minus_I_cka']:.4f}, ΔmKNN={a['flickr_both_new_F_minus_I_mknn']:.4f}",
        f"- Mean genuine−shuffled-fit CKA={a['mean_genuine_minus_shuffled_fit_cka']:.4f}",
        "",
        "See `test_metrics.csv` and `paired_contrasts.csv` for per-pair tables.",
        "",
    ]
    return "\n".join(lines)


def _make_figures(fig: Path, metric_rows, contrast_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focus = [r for r in metric_rows if r["condition"] in CONDITIONS and "size_matched" not in r["dataset"]]
    if not focus:
        return
    # I/W/D/F by section mean
    sections = sorted({r["section"] for r in focus})
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    for ax, metric in zip(axes, ("cka", "mknn")):
        width = 0.2
        for i, section in enumerate(sections):
            vals = []
            for cond in CONDITIONS:
                rows = [float(r[metric]) for r in focus if r["section"] == section and r["condition"] == cond]
                vals.append(float(np.mean(rows)) if rows else np.nan)
            ax.bar(x + (i - len(sections) / 2) * width, vals, width, label=section)
        ax.set_xticks(x)
        ax.set_xticklabels(list(CONDITIONS))
        ax.set_title(metric)
        ax.legend(fontsize=7)
    fig1.suptitle("I / W / D / F by section")
    fig1.savefig(fig / "conditions_by_section.png", dpi=140)
    plt.close(fig1)
