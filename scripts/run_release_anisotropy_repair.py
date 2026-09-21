#!/usr/bin/env python3
"""Repair pass: nested incumbents, S-tilde, orientation reference, controls at ρ=0.1.

Writes results/release_anisotropy_repair/ only. Does not extract. Does not
overwrite results/release_anisotropy/.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.anisotropic_kernels import fit_pca, make_pack, scores_numpy
from prh_replication.extract_final import final_feature_path, load_final_prepared, spec_from_manifest
from prh_replication.io_utils import jsonable, sha256_file, write_json
from prh_replication.kernel_experiment import subset_indices
from prh_replication.plots import lines
from prh_replication.registry import MODELS, Paths
from prh_replication.release_protocol import VIS, dump_fit, eval_onesided, load_splits, pack_ab, partner_sets
from prh_replication.release_anisotropy import (
    NEAR_ZERO_S,
    b_from_s,
    decompose_s,
    direction_only_b,
    evaluate_one_sided_vech,
    fit_one_sided,
    fit_one_sided_shared,
    fro_cosine,
    haar_rotate_s,
    projector_agreement,
    response_grams,
    select_rho_by_val,
    select_rho_shared,
    signed_correction,
    split_delta_psd_factors,
    uniform_only_b,
    up_down_projectors,
    vech_from_s,
    response_grams_factor,
)

PARENT = "release_anisotropy"
OUT_NAME = "release_anisotropy_repair"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-perm", type=int, default=0)
    p.add_argument("--resume-after", default="", help="lopo: skip refits; run signatures/shuffle/stability/summary")
    return p.parse_args()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    parent = paths.results / PARENT
    out = paths.results / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "onesided_fits").mkdir(exist_ok=True)
    if (out / "summary.json").exists() and not args.force and not args.smoke and not args.resume_after:
        print(f"{OUT_NAME} already complete; pass --force to rerun", flush=True)
        return

    design = json.loads((ROOT / "configs" / "release_anisotropy_repair.json").read_text())
    orig_design = json.loads((ROOT / "configs" / "release_anisotropy.json").read_text())
    manifest = json.loads((ROOT / "data" / "manifests" / "release_models.json").read_text())
    write_json(out / "design.json", design)
    write_json(out / "parent_design.json", orig_design)

    man, splits = load_splits(paths, ROOT)
    rows = list(manifest["base_panel"]) + list(manifest["supplementary_qwen3x"])
    specs = {r["key"]: spec_from_manifest(r) for r in rows}
    base_qwen = [r["key"] for r in manifest["base_panel"] if r["family"] == "Qwen"]
    base_olmo = [r["key"] for r in manifest["base_panel"] if r["family"] == "OLMo"]
    supp = [r["key"] for r in manifest["supplementary_qwen3x"]]
    q = 4 if args.smoke else int(design["q"])
    rhos = [0.0, 0.1, 0.4] if args.smoke else list(design["budgets_rho"])
    n_steps = 20 if args.smoke else 120
    n_rot = 4 if args.smoke else int(design["orientation_reference"]["n_rotations"])

    langs_all = base_qwen + base_olmo + supp
    cache = {}
    hashes = {}
    pca_parent = json.loads((parent / "pca.json").read_text()) if (parent / "pca.json").exists() else {}
    for key in langs_all + VIS:
        spec = specs.get(key) or MODELS[key]
        cache[key] = {}
        for split, ids in splits.items():
            pth = final_feature_path(paths, spec, "coco_val2017", split, ids)
            if not pth.exists():
                raise FileNotFoundError(pth)
            hashes[f"{key}:{split}"] = {"path": str(pth), "sha256": sha256_file(pth), "bytes": pth.stat().st_size}
            x, obj = load_final_prepared(paths, spec, split, ids)
            cache[key][split] = x.numpy().astype(np.float64)
            if list(obj["sample_ids"]) != list(ids):
                raise ValueError(f"id mismatch {key} {split}")
        print(f"loaded {key} {cache[key]['train'].shape}", flush=True)

    # PCA: recompute from the same train tensors; verify variance against parent
    pca = {}
    pca_check = {}
    for key in cache:
        x = cache[key]["train"]
        mu = x.mean(0)
        rec = fit_pca(x - mu, q=q)
        pca[key] = {"mu": mu, "U": rec["U"], "q": rec["q"]}
        if key in pca_parent and not args.smoke:
            old = pca_parent[key]["variance_fraction"]
            pca_check[key] = {
                "parent_variance_fraction": old,
                "recomputed": rec["variance_fraction"],
                "abs_diff": abs(old - rec["variance_fraction"]),
            }
            if abs(old - rec["variance_fraction"]) > 1e-8:
                raise RuntimeError(f"PCA mismatch {key}: {old} vs {rec['variance_fraction']}")
    write_json(out / "pca_provenance.json", jsonable(pca_check))
    write_json(out / "feature_hashes.json", hashes)

    native_src = parent / "native_matrix.json"
    aniso_src = parent / "native_anisotropy.json"
    two_src = parent / "two_sided.json"
    write_json(out / "native_reuse.json", {
        "native_matrix": str(native_src),
        "native_anisotropy": str(aniso_src),
        "two_sided": str(two_src),
        "two_sided_note": "Two-sided bridge is a single max-budget spectral-clip fit, not a nested rho grid. Original run found non-identity solutions; not recomputed.",
        "native_sha256": sha256_file(native_src) if native_src.exists() else None,
    })
    native_rows = json.loads(native_src.read_text()) if native_src.exists() else []
    aniso = json.loads(aniso_src.read_text()) if aniso_src.exists() else {}
    two_rows = json.loads(two_src.read_text()) if two_src.exists() else []
    if native_src.exists():
        shutil.copy2(native_src, out / "native_matrix.json")
    if aniso_src.exists():
        shutil.copy2(aniso_src, out / "native_anisotropy.json")
    if two_src.exists():
        shutil.copy2(two_src, out / "two_sided.json")

    # Diagnosis: reload one parent rho=0.1 fit from work-disk onesided_fits if present
    diag = {"cause": (
        "Implementation, not nonconvexity of the feasible set. Each rho ran independent Adam "
        "(identity + two perturbations) and scored only the final iterate. Identity starts abort "
        "when torch.linalg.eigh of S=0 (repeated eigenvalues) yields obj.requires_grad False. "
        "At large rho the perturbation last-iterates scored below identity, so identity_explicit "
        "won. Nested feasible sets were never searched: a feasible rho=0.1 S was not a candidate "
        "at larger rho, and no best-iterate/incumbent was kept. params.clamp(-4,4) on vech is not "
        "D<=rho; project_S already enforces spectral clip then Frobenius scale. rho is an upper "
        "bound (scale only if D>rho), not an equality and not a change of objective. Two-sided "
        "bridge uses a single spectral-clip fit_metrics (no nested rho grid) and already retained "
        "non-identity solutions; it is not recomputed."
    )}
    parent_fit_dir = parent / "onesided_fits"
    sample_pair = "qwen2-7b__dinov2-small"
    p01 = parent_fit_dir / f"{sample_pair}__rho0.1.json"
    if p01.exists() and "qwen2-7b" in cache:
        raw = json.loads(p01.read_text())
        s01 = np.array(raw["s_a"], dtype=np.float64)
        pack_tr, _, _ = pack_ab(cache, pca, "qwen2-7b", "dinov2-small", "train")
        vech = vech_from_s(s01)
        e01 = evaluate_one_sided_vech(pack_tr, vech, 0.1)
        diag["sample_pair"] = sample_pair
        diag["parent_train_excess_rho01"] = raw.get("train", {}).get("excess")
        diag["reeval"] = {}
        for rho in rhos:
            ev = evaluate_one_sided_vech(pack_tr, vech, float(rho))
            diag["reeval"][str(rho)] = {
                "train_excess": ev["train_excess"],
                "d": ev["d"],
                "feasible": ev["feasible"],
                "eig_ok": ev["eig_ok"],
                "excess_matches_01": abs(ev["train_excess"] - e01["train_excess"]) < 1e-10,
            }
    write_json(out / "diagnosis.json", jsonable(diag))
    print("diagnosis written", diag.get("sample_pair"), flush=True)

    partners = partner_sets(base_qwen, base_olmo, supp)
    partners = {k: [p for p in v if p in cache] for k, v in partners.items() if k in cache}
    langs = [k for k in (base_qwen + base_olmo + supp) if k in cache]

    def _hydrate_fit(raw: dict) -> dict:
        rec = dict(raw)
        rec["b_a"] = np.array(raw["b_a"], np.float64)
        rec["s_a"] = np.array(raw["s_a"], np.float64)
        rec["eig_a"] = np.array(raw["eig_a"], np.float64)
        return rec

    def _rebuild_fits_from_disk() -> dict:
        rebuilt = {}
        headline = json.loads((out / "headline_rho.json").read_text()) if (out / "headline_rho.json").exists() else {}
        for fpath in sorted((out / "onesided_fits").glob("*.json")):
            stem = fpath.stem
            if "__rho" not in stem:
                continue
            pair, rho_s = stem.rsplit("__rho", 1)
            fit = _hydrate_fit(json.loads(fpath.read_text()))
            rebuilt.setdefault(pair, {"cands": {}, "headline_rho": float(headline.get(pair, 0.0))})
            rebuilt[pair]["cands"][float(rho_s)] = fit
        return rebuilt

    resume_after = (args.resume_after or "").strip().lower()
    if resume_after == "lopo":
        required = ["pair_eval.json", "train_monotonicity.json", "headline_rho.json", "ablations.json", "consistency.json", "transfer.json", "lopo.json"]
        missing = [n for n in required if not (out / n).exists()]
        if missing:
            raise FileNotFoundError(f"--resume-after lopo missing {missing}")
        pair_eval = json.loads((out / "pair_eval.json").read_text())
        train_mono = json.loads((out / "train_monotonicity.json").read_text())
        ablations = json.loads((out / "ablations.json").read_text())
        cons = json.loads((out / "consistency.json").read_text())
        transfer = json.loads((out / "transfer.json").read_text())
        lopo_rows = json.loads((out / "lopo.json").read_text())
        n_mono_fail = sum(1 for r in train_mono if not r["ok"])
        fits = _rebuild_fits_from_disk()
        print(f"resume-after lopo: {len(fits)} pairs, skip refits", flush=True)
    else:
        fits = {}
        pair_eval = []
        train_mono = []
    if resume_after != "lopo":
        for a, plist in partners.items():
            for b in plist:
                incumbents = []
                cands = {}
                prev_train = float("-inf")
                for rho in rhos:
                    fpath = out / "onesided_fits" / f"{a}__{b}__rho{rho}.json"
                    if fpath.exists() and not args.smoke:
                        raw = json.loads(fpath.read_text())
                        raw["b_a"] = np.array(raw["b_a"], np.float64)
                        raw["s_a"] = np.array(raw["s_a"], np.float64)
                        raw["eig_a"] = np.array(raw["eig_a"], np.float64)
                        fit = raw
                    else:
                        pack_tr, _, _ = pack_ab(cache, pca, a, b, "train")
                        fit = fit_one_sided(pack_tr, rho=float(rho), n_steps=n_steps, incumbents=incumbents)
                        write_json(fpath, dump_fit(fit))
                    tex = fit["train"]["excess"]
                    if float(rho) > 0 and tex + 1e-8 < prev_train:
                        train_mono.append({"a": a, "b": b, "rho": float(rho), "train": tex, "prev": prev_train, "ok": False})
                    else:
                        train_mono.append({"a": a, "b": b, "rho": float(rho), "train": tex, "prev": prev_train, "ok": True})
                    prev_train = max(prev_train, tex)
                    incumbents.append(np.array(fit["s_a"], dtype=np.float64))
                    cands[float(rho)] = fit
                    pack_va, _, _ = pack_ab(cache, pca, a, b, "val")
                    pack_te, za_te, zb_te = pack_ab(cache, pca, a, b, "test")
                    ba = fit["b_a"]
                    ev_va = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_va["q_b"], pack_va["q_b"])), pack_va)
                    ev_te = eval_onesided(za_te, zb_te, pca[a]["U"], pca[b]["U"], ba, n_perm=0, seed=0)
                    dec = decompose_s(fit["s_a"])
                    pair_eval.append({
                        "a": a, "b": b, "rho": float(rho), "kind": "one_sided_repaired",
                        "d_attained": fit["d_attained"],
                        "selected_rho_is_not_d": True,
                        "mu": dec["mu"],
                        "d_uniform": dec["d_uniform"],
                        "d_directional": dec["d_directional"],
                        "direction_defined": dec["direction_defined"],
                        "frac_at_bounds": fit.get("diag_a", {}).get("frac_at_bounds"),
                        "train": fit["train"],
                        "val": ev_va,
                        "test": ev_te,
                        "selected_start": fit.get("selected_start"),
                    })
                pack_va, _, _ = pack_ab(cache, pca, a, b, "val")
                rho_h, _ = select_rho_by_val(cands, pack_va)
                fits[f"{a}__{b}"] = {"cands": cands, "headline_rho": float(rho_h)}
                print(f"onesided {a}|{b} headline_rho={rho_h} D={cands[float(rho_h)]['d_attained']:.4f}", flush=True)
        write_json(out / "pair_eval.json", jsonable(pair_eval))
        write_json(out / "headline_rho.json", {k: v["headline_rho"] for k, v in fits.items()})
        write_json(out / "train_monotonicity.json", jsonable(train_mono))
        n_mono_fail = sum(1 for r in train_mono if not r["ok"])
        print(f"train monotonicity failures: {n_mono_fail}/{len(train_mono)}", flush=True)

        # Ablations (unfitted)
        ablations = []
        for rec in pair_eval:
            if rec["rho"] <= 0:
                continue
            fit = fits[f"{rec['a']}__{rec['b']}"]["cands"][float(rec["rho"])]
            s = np.array(fit["s_a"], dtype=np.float64)
            pack_te, _, _ = pack_ab(cache, pca, rec["a"], rec["b"], "test")
            qb = pack_te["q_b"]
            eye_a = np.zeros((pack_te["q_a"], pack_te["q_a"]))
            sc_id = scores_numpy(eye_a, np.zeros((qb, qb)), pack_te)
            sc_full = scores_numpy(fit["b_a"] - np.eye(fit["b_a"].shape[0]), np.zeros((qb, qb)), pack_te)
            bu = uniform_only_b(s)
            sc_u = scores_numpy(bu - np.eye(bu.shape[0]), np.zeros((qb, qb)), pack_te)
            bd, info = direction_only_b(s)
            sc_d = scores_numpy(bd - np.eye(bd.shape[0]), np.zeros((qb, qb)), pack_te)
            ablations.append({
                "a": rec["a"], "b": rec["b"], "rho": rec["rho"],
                "identity_excess": sc_id["excess"],
                "full_excess": sc_full["excess"],
                "uniform_only_excess": sc_u["excess"],
                "direction_only_excess": sc_d["excess"],
                "direction_only_bounds_ok": info["within_spectral_bounds"],
                "direction_only_eig_min": info["eig_min"],
                "direction_only_eig_max": info["eig_max"],
                "mu": rec["mu"],
                "d_uniform": rec["d_uniform"],
                "d_directional": rec["d_directional"],
            })
        write_json(out / "ablations.json", jsonable(ablations))

        # Consistency with S and S_tilde
        cons = []
        for a, plist in partners.items():
            vis_p = [p for p in plist if p in VIS]
            lang_p = [p for p in plist if p not in VIS]
            for rho in rhos:
                for group_name, group in (("all", plist), ("vision", vis_p), ("language", lang_p)):
                    if len(group) < 2:
                        continue
                    for i, b in enumerate(group):
                        for c in group[i + 1 :]:
                            fa = fits[f"{a}__{b}"]["cands"][float(rho)]
                            fb = fits[f"{a}__{c}"]["cands"][float(rho)]
                            sa, sb = np.array(fa["s_a"]), np.array(fb["s_a"])
                            da_, db_ = decompose_s(sa), decompose_s(sb)
                            pu, pd, infa = up_down_projectors(da_["s_tilde"])
                            qu, qd, infb = up_down_projectors(db_["s_tilde"])
                            za = cache[a]["test"] - pca[a]["mu"]
                            ga = response_grams(za, signed_correction(pca[a]["U"], fa["b_a"]))
                            gb = response_grams(za, signed_correction(pca[a]["U"], fb["b_a"]))
                            defd = da_["direction_defined"] and db_["direction_defined"]
                            cons.append({
                                "model": a, "partner_i": b, "partner_j": c, "rho": float(rho), "group": group_name,
                                "s_fro_cosine": fro_cosine(sa, sb) if (np.linalg.norm(sa) > NEAR_ZERO_S and np.linalg.norm(sb) > NEAR_ZERO_S) else None,
                                "stilde_fro_cosine": fro_cosine(da_["s_tilde"], db_["s_tilde"]) if defd else None,
                                "direction_defined": defd,
                                "mu_i": da_["mu"], "mu_j": db_["mu"],
                                "d_uniform_i": da_["d_uniform"], "d_directional_i": da_["d_directional"],
                                "up_agreement_stilde": projector_agreement(pu, qu),
                                "down_agreement_stilde": projector_agreement(pd, qd),
                                "induced_deltaK_cosine": fro_cosine(ga, gb),
                            })
        write_json(out / "consistency.json", jsonable(cons))

        # Transfer
        transfer = []
        for a, plist in partners.items():
            for b in plist:
                for c in plist:
                    if b == c:
                        continue
                    for rho in rhos:
                        fit_b = fits[f"{a}__{b}"]["cands"][float(rho)]
                        fit_c = fits[f"{a}__{c}"]["cands"][float(rho)]
                        pack_te, _, _ = pack_ab(cache, pca, a, c, "test")
                        qb = pack_te["q_b"]
                        z = np.zeros((qb, qb))
                        za = np.zeros((pack_te["q_a"], pack_te["q_a"]))
                        sc_tr = scores_numpy(fit_b["b_a"] - np.eye(fit_b["b_a"].shape[0]), z, pack_te)
                        sc_dir = scores_numpy(fit_c["b_a"] - np.eye(fit_c["b_a"].shape[0]), z, pack_te)
                        sc_id = scores_numpy(za, z, pack_te)
                        sb = np.array(fit_b["s_a"])
                        bu = uniform_only_b(sb)
                        bd, info = direction_only_b(sb)
                        sc_u = scores_numpy(bu - np.eye(bu.shape[0]), z, pack_te)
                        sc_dt = scores_numpy(bd - np.eye(bd.shape[0]), z, pack_te)
                        transfer.append({
                            "model": a, "fit_partner": b, "eval_partner": c, "rho": float(rho),
                            "eval_modality": "vision" if c in VIS else "language",
                            "identity_excess": sc_id["excess"],
                            "transferred_excess": sc_tr["excess"],
                            "direct_excess": sc_dir["excess"],
                            "uniform_only_excess": sc_u["excess"],
                            "direction_only_excess": sc_dt["excess"],
                            "direction_only_bounds_ok": info["within_spectral_bounds"],
                            "delta_vs_identity": sc_tr["excess"] - sc_id["excess"],
                            "gap_vs_direct": sc_tr["excess"] - sc_dir["excess"],
                            "d_transferred": fit_b["d_attained"],
                            "mu": decompose_s(sb)["mu"],
                        })
        write_json(out / "transfer.json", jsonable(transfer))

        # LOPO with nested incumbents
        lopo_rows = []
        groups = {"vision_anchors": list(VIS), "olmo_base": list(base_olmo), "qwen_base": list(base_qwen)}
        for a in langs:
            for gname, gparts in groups.items():
                group = [p for p in gparts if p in cache and p != a]
                if len(group) < 2:
                    continue
                if gname == "olmo_base" and a not in base_qwen + supp:
                    continue
                if gname == "qwen_base" and a not in base_olmo:
                    continue
                for held in group:
                    others = [p for p in group if p != held]
                    incumbents = []
                    cands = {}
                    for rho in rhos:
                        packs_tr = [pack_ab(cache, pca, a, p, "train")[0] for p in others]
                        fit = fit_one_sided_shared(packs_tr, rho=float(rho), n_steps=n_steps, incumbents=incumbents)
                        cands[float(rho)] = fit
                        incumbents.append(fit["s_a"])
                    packs_va = [pack_ab(cache, pca, a, p, "val")[0] for p in others]
                    rho_h, chosen = select_rho_shared(cands, packs_va)
                    pack_te, _, _ = pack_ab(cache, pca, a, held, "test")
                    ba = chosen["b_a"]
                    sc = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                    sc_id = scores_numpy(np.zeros((pack_te["q_a"], pack_te["q_a"])), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                    lopo_rows.append({
                        "model": a, "group": gname, "held_out_partner": held, "fit_partners": others,
                        "selected_rho": float(rho_h), "d_attained": chosen["d_attained"],
                        "test": sc, "identity": sc_id,
                        "held_out_modality": "vision" if held in VIS else "language",
                    })
                    print(f"lopo {a} hold {held} rho={rho_h} D={chosen['d_attained']:.3f} excess={sc['excess']:.4f}", flush=True)
        for a in base_qwen + base_olmo:
            if a not in cache:
                continue
            lang_partners = [p for p in partners.get(a, []) if p not in VIS]
            vis_p = [p for p in VIS if p in cache]
            if len(lang_partners) < 2 or not vis_p:
                continue
            incumbents = []
            cands = {}
            for rho in rhos:
                packs_tr = [pack_ab(cache, pca, a, p, "train")[0] for p in lang_partners]
                fit = fit_one_sided_shared(packs_tr, rho=float(rho), n_steps=n_steps, incumbents=incumbents)
                cands[float(rho)] = fit
                incumbents.append(fit["s_a"])
            packs_va = [pack_ab(cache, pca, a, p, "val")[0] for p in lang_partners]
            rho_h, chosen = select_rho_shared(cands, packs_va)
            for v in vis_p:
                pack_te, _, _ = pack_ab(cache, pca, a, v, "test")
                ba = chosen["b_a"]
                sc = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                sc_id = scores_numpy(np.zeros((pack_te["q_a"], pack_te["q_a"])), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                lopo_rows.append({
                    "model": a, "group": "language_shared_eval_vision", "held_out_partner": v,
                    "fit_partners": lang_partners, "selected_rho": float(rho_h),
                    "d_attained": chosen["d_attained"], "test": sc, "identity": sc_id,
                    "held_out_modality": "vision",
                    "note": "shared language-partner metric; vision never used in fit or selection",
                })
        write_json(out / "lopo.json", jsonable(lopo_rows))

        print("signatures + Haar orientation reference", flush=True)
    sig_rows = []
    rng = np.random.default_rng(int(design["orientation_reference"]["seed"]))
    fixed_partners = [p for p in (["dinov2-small"] + ([base_olmo[-1]] if base_olmo else []) + ([base_qwen[-1]] if base_qwen else [])) if p in cache]
    for family, members in (("Qwen-base", base_qwen), ("OLMo-base", base_olmo), ("Qwen3x-supp", supp)):
        members = [m for m in members if m in cache]
        for partner in fixed_partners:
            if partner in members:
                continue
            for rho in rhos:
                for i, m1 in enumerate(members):
                    for m2 in members[i + 1 :]:
                        k1, k2 = f"{m1}__{partner}", f"{m2}__{partner}"
                        if k1 not in fits or k2 not in fits:
                            continue
                        f1 = fits[k1]["cands"][float(rho)]
                        f2 = fits[k2]["cands"][float(rho)]
                        z1 = cache[m1]["test"] - pca[m1]["mu"]
                        z2 = cache[m2]["test"] - pca[m2]["mu"]
                        p1, n1 = split_delta_psd_factors(f1["b_a"])
                        p2, n2 = split_delta_psd_factors(f2["b_a"])
                        u1, u2 = pca[m1]["U"], pca[m2]["U"]
                        g1p, g1n = response_grams_factor(z1, u1, p1), response_grams_factor(z1, u1, n1)
                        g2p, g2n = response_grams_factor(z2, u2, p2), response_grams_factor(z2, u2, n2)
                        mag1, mag2 = float(np.linalg.norm(g1p)), float(np.linalg.norm(g2p))
                        near = mag1 <= NEAR_ZERO_S or mag2 <= NEAR_ZERO_S
                        obs_p = fro_cosine(g1p, g2p)
                        obs_m = fro_cosine(g1n, g2n)
                        ref_p, ref_m = [], []
                        if not near:
                            s1, s2 = np.array(f1["s_a"]), np.array(f2["s_a"])
                            for _ in range(n_rot):
                                sr1, sr2 = haar_rotate_s(s1, rng), haar_rotate_s(s2, rng)
                                b1, b2 = b_from_s(sr1), b_from_s(sr2)
                                ep1, en1 = split_delta_psd_factors(b1)
                                ep2, en2 = split_delta_psd_factors(b2)
                                ref_p.append(fro_cosine(response_grams_factor(z1, u1, ep1), response_grams_factor(z2, u2, ep2)))
                                ref_m.append(fro_cosine(response_grams_factor(z1, u1, en1), response_grams_factor(z2, u2, en2)))
                        def qtl(xs, p):
                            xs = [x for x in xs if isinstance(x, float) and math.isfinite(x)]
                            if not xs:
                                return None
                            return float(np.quantile(xs, p))
                        sig_rows.append({
                            "family": family, "partner": partner, "rho": float(rho),
                            "model_i": m1, "model_j": m2,
                            "near_zero": near,
                            "gplus_fro_cosine": obs_p,
                            "gminus_fro_cosine": obs_m,
                            "ref_n": len(ref_p),
                            "gplus_ref_q50": qtl(ref_p, 0.5),
                            "gplus_ref_q95": qtl(ref_p, 0.95),
                            "gplus_obs_gt_ref95": (obs_p is not None and qtl(ref_p, 0.95) is not None and obs_p > qtl(ref_p, 0.95)),
                            "gminus_ref_q50": qtl(ref_m, 0.5),
                            "gminus_ref_q95": qtl(ref_m, 0.95),
                            "correction_mag_i": mag1, "correction_mag_j": mag2,
                            "signed_note": "signed cosine is not PSD-kernel CKA; Haar reference preserves eigenvalues only",
                        })
    write_json(out / "signatures.json", jsonable(sig_rows))

    # Shuffle over full budget grid
    shuf = []
    sh_models = [m for m in design["shuffle"]["models"] if m in cache]
    partner = design["shuffle"]["partner"]
    seeds = [0] if args.smoke else design["shuffle"]["seeds"]
    if partner in cache:
        for m in sh_models:
            xa_tr, xb_tr = cache[m]["train"], cache[partner]["train"]
            xa_va, xb_va = cache[m]["val"], cache[partner]["val"]
            xa_te, xb_te = cache[m]["test"], cache[partner]["test"]
            mu_a, mu_b = pca[m]["mu"], pca[partner]["mu"]
            ua, ub = pca[m]["U"], pca[partner]["U"]
            pack_id_te, za_te, zb_te = pack_ab(cache, pca, m, partner, "test")
            id_te = eval_onesided(za_te, zb_te, ua, ub, np.eye(ua.shape[1]), n_perm=0, seed=0)
            for seed in seeds:
                g = np.random.default_rng(int(seed) + 17)
                p_tr = g.permutation(len(xa_tr))
                p_va = g.permutation(len(xa_va))
                p_te = g.permutation(len(xa_te))
                incumbents = []
                cands = {}
                for rho in rhos:
                    pack_tr = make_pack(xa_tr - mu_a, xb_tr[p_tr] - mu_b, ua, ub)
                    fit = fit_one_sided(pack_tr, rho=float(rho), n_steps=n_steps, incumbents=incumbents)
                    cands[float(rho)] = fit
                    incumbents.append(fit["s_a"])
                pack_va = make_pack(xa_va - mu_a, xb_va[p_va] - mu_b, ua, ub)
                rho_h, chosen = select_rho_by_val(cands, pack_va)
                sc_true = eval_onesided(xa_te - mu_a, xb_te - mu_b, ua, ub, chosen["b_a"], n_perm=0, seed=0)
                sc_sh = eval_onesided(xa_te - mu_a, xb_te[p_te] - mu_b, ua, ub, chosen["b_a"], n_perm=0, seed=0)
                shuf.append({
                    "model": m, "partner": partner, "seed": int(seed),
                    "selected_rho": float(rho_h),
                    "d_attained": chosen["d_attained"],
                    "selected_identity": chosen["d_attained"] < 1e-8,
                    "true_test": sc_true,
                    "shuffled_test": sc_sh,
                    "identity_true_test": id_te,
                    "delta_a_true_vs_id": sc_true["a"] - id_te["a"],
                })
                print(f"shuffle {m} seed={seed} rho={rho_h} D={chosen['d_attained']:.3f} dA={sc_true['a']-id_te['a']:.4f}", flush=True)
    write_json(out / "shuffle_fit.json", jsonable(shuf))

    # Stability at 0.1 and at frozen full-data headline if different
    stab = []
    stab_models = [m for m in design["stability"]["models"] if m in cache]
    n_sub = 2 if args.smoke else int(design["stability"]["subsets"])
    sub_size = min(int(design["stability"]["subset_size"]), cache[stab_models[0]]["train"].shape[0] // n_sub) if stab_models else 0
    if stab_models and sub_size >= 16:
        idxs = subset_indices(cache[stab_models[0]]["train"].shape[0], sub_size, n_sub, int(design["stability"]["subset_seed"]))
        for m in stab_models:
            for v in VIS:
                if v not in cache:
                    continue
                rhos_stab = [0.1]
                h = fits.get(f"{m}__{v}", {}).get("headline_rho")
                if h is not None and abs(float(h) - 0.1) > 1e-12 and float(h) > 0:
                    rhos_stab.append(float(h))
                for rho_s in rhos_stab:
                    ss = []
                    for si, idx in enumerate(idxs):
                        ii = idx.numpy()
                        pack = make_pack(cache[m]["train"][ii] - pca[m]["mu"], cache[v]["train"][ii] - pca[v]["mu"], pca[m]["U"], pca[v]["U"])
                        fit = fit_one_sided(pack, rho=float(rho_s), n_steps=n_steps)
                        ss.append(fit)
                        dec = decompose_s(fit["s_a"])
                        stab.append({"model": m, "partner": v, "rho": float(rho_s), "subset": si,
                                     "d": fit["d_attained"], "mu": dec["mu"], "d_directional": dec["d_directional"],
                                     "train_excess": fit["train"]["excess"], "direction_defined": dec["direction_defined"]})
                    pack_te, za_te, zb_te = pack_ab(cache, pca, m, v, "test")
                    id_te = scores_numpy(np.zeros((pack_te["q_a"], pack_te["q_a"])), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                    grams = []
                    for fit in ss:
                        sc_te = scores_numpy(fit["b_a"] - np.eye(fit["b_a"].shape[0]), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                        gp, gn = split_delta_psd_factors(fit["b_a"])
                        grams.append((response_grams_factor(za_te, pca[m]["U"], gp), response_grams_factor(za_te, pca[m]["U"], gn), sc_te))
                    for i in range(len(ss)):
                        dec = decompose_s(ss[i]["s_a"])
                        pu, pd, _ = up_down_projectors(dec["s_tilde"] if dec["direction_defined"] else ss[i]["s_a"])
                        stab.append({
                            "model": m, "partner": v, "rho": float(rho_s), "kind": "subset_heldout",
                            "subset": i, "d": ss[i]["d_attained"], "mu": dec["mu"],
                            "d_directional": dec["d_directional"], "direction_defined": dec["direction_defined"],
                            "heldout_excess": grams[i][2]["excess"],
                            "heldout_a": grams[i][2]["a"],
                            "identity_excess": id_te["excess"],
                            "delta_excess_vs_id": grams[i][2]["excess"] - id_te["excess"],
                            "label": "conditional_stability_frozen_pca",
                        })
                    for i in range(len(ss)):
                        for j in range(i + 1, len(ss)):
                            di, dj = decompose_s(ss[i]["s_a"]), decompose_s(ss[j]["s_a"])
                            defd = di["direction_defined"] and dj["direction_defined"]
                            pu, pd, _ = up_down_projectors(di["s_tilde"])
                            qu, qd, _ = up_down_projectors(dj["s_tilde"])
                            stab.append({
                                "model": m, "partner": v, "rho": float(rho_s), "kind": "within_partner_refit",
                                "subset_i": i, "subset_j": j,
                                "s_fro_cosine": fro_cosine(ss[i]["s_a"], ss[j]["s_a"]) if (np.linalg.norm(ss[i]["s_a"]) > NEAR_ZERO_S and np.linalg.norm(ss[j]["s_a"]) > NEAR_ZERO_S) else None,
                                "stilde_fro_cosine": fro_cosine(di["s_tilde"], dj["s_tilde"]) if defd else None,
                                "direction_defined": defd,
                                "up_agreement_stilde": projector_agreement(pu, qu) if defd else None,
                                "down_agreement_stilde": projector_agreement(pd, qd) if defd else None,
                                "gplus_heldout_cosine": fro_cosine(grams[i][0], grams[j][0]),
                                "gminus_heldout_cosine": fro_cosine(grams[i][1], grams[j][1]),
                                "label": "conditional_stability_frozen_pca",
                            })
    write_json(out / "stability.json", jsonable(stab))

    checks = {
        "train_monotonicity_failures": n_mono_fail,
        "rho_is_upper_bound": True,
        "identity_rho0": all(abs(r["d_attained"]) < 1e-10 for r in pair_eval if r["rho"] == 0.0),
        "all_d_le_rho": all(r["d_attained"] <= r["rho"] + 1e-7 for r in pair_eval),
        "direct_vs_contracted": [],
    }
    sample_keys = [k for k in fits][:3]
    for key in sample_keys:
        a, b = key.split("__", 1)
        rho_h = fits[key]["headline_rho"]
        fit = fits[key]["cands"][float(rho_h)]
        pack_te, za_te, zb_te = pack_ab(cache, pca, a, b, "test")
        ba = fit["b_a"]
        contracted = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
        direct = eval_onesided(za_te, zb_te, pca[a]["U"], pca[b]["U"], ba, n_perm=0, seed=0)
        checks["direct_vs_contracted"].append({
            "pair": key, "rho": float(rho_h),
            "contracted_a": contracted["a"], "direct_a": direct["a"],
            "abs_diff": abs(contracted["a"] - direct["a"]),
            "ok": abs(contracted["a"] - direct["a"]) < 1e-6,
        })
    write_json(out / "checks.json", jsonable(checks))

    # Plots: excess vs rho repaired VL
    series = {}
    for a in langs:
        ys = []
        for rho in rhos:
            vals = [r["test"]["excess"] for r in pair_eval if r["a"] == a and abs(r["rho"] - float(rho)) < 1e-9 and r["b"] in VIS]
            ys.append(float(np.mean(vals)) if vals else float("nan"))
        series[a] = ys
    if series:
        lines([float(r) for r in rhos], series, "Repaired one-sided VL test excess vs rho", "rho", "CKA excess a-b", out / "excess_vs_rho_vl.png")

    # Summary stats
    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    vl0 = [r for r in pair_eval if r["rho"] == 0.0 and r["b"] in VIS]
    vl01 = [r for r in pair_eval if abs(r["rho"] - 0.1) < 1e-9 and r["b"] in VIS]
    vlmax = [r for r in pair_eval if abs(r["rho"] - max(rhos)) < 1e-12 and r["b"] in VIS]
    head_vl = []
    for rec in [r for r in pair_eval if r["b"] in VIS and r["rho"] == 0.0]:
        h = fits[f"{rec['a']}__{rec['b']}"]["headline_rho"]
        sel = next(x for x in pair_eval if x["a"] == rec["a"] and x["b"] == rec["b"] and abs(x["rho"] - float(h)) < 1e-12)
        head_vl.append(sel["test"]["a"] - rec["test"]["a"])
    cons01 = [c for c in cons if abs(c["rho"] - 0.1) < 1e-9 and c["group"] == "vision" and c.get("direction_defined")]
    tr01v = [t for t in transfer if abs(t["rho"] - 0.1) < 1e-9 and t["eval_modality"] == "vision"]
    summary = {
        "evaluation_status": "exploratory",
        "parent": PARENT,
        "n_train_mono_fail": n_mono_fail,
        "vl_identity_mean_a": mean([r["test"]["a"] for r in vl0]),
        "vl_rho01_mean_a": mean([r["test"]["a"] for r in vl01]),
        "vl_maxrho_mean_a": mean([r["test"]["a"] for r in vlmax]),
        "vl_headline_mean_delta_a": mean(head_vl),
        "vl_maxrho_mean_d": mean([r["d_attained"] for r in vlmax]),
        "vision_stilde_cosine_rho01_defined": mean([c["stilde_fro_cosine"] for c in cons01]),
        "n_vision_stilde_defined_rho01": len(cons01),
        "transfer_vl_rho01_delta_excess": mean([t["delta_vs_identity"] for t in tr01v]),
        "transfer_vl_rho01_uniform_delta": mean([t["uniform_only_excess"] - t["identity_excess"] for t in tr01v]),
        "shuffle_mean_delta_a": mean([s["delta_a_true_vs_id"] for s in shuf]),
        "shuffle_n_identity": sum(1 for s in shuf if s["selected_identity"]),
        "two_sided_retained": True,
        "native_reused": True,
    }
    write_json(out / "summary.json", jsonable(summary))

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "git_head": git_head(ROOT),
        "command": "python scripts/run_release_anisotropy_repair.py --work /mnt/sdb1/prh-replication-work",
    }
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:
        pass
    write_json(out / "software_versions.json", versions)
    print("repair done", summary, flush=True)


if __name__ == "__main__":
    main()
