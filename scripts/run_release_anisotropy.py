#!/usr/bin/env python3
"""Final-layer release alignment and one-sided anisotropic corrections.

Writes results/release_anisotropy/ only. Resumable. Sequential extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prh_replication.anisotropic_kernels import (
    centred_diag_energy,
    fit_pca,
    knn_from_sq,
    knn_overlap,
    make_pack,
    metric_gram,
    metric_sq_distances,
    mnn_from_knn,
    scores_numpy,
)
from prh_replication.datasets import load_manifest
from prh_replication.extract_final import (
    PROTOCOL,
    extract_language_final,
    final_feature_path,
    load_final_prepared,
    slice_vision_final,
    spec_from_manifest,
)
from prh_replication.io_utils import write_json
from prh_replication.kernel_experiment import subset_indices
from prh_replication.kernels import center_gram_o2, extension_stats, linear_gram, mc_cka_mean
from prh_replication.metrics import mutual_knn_score
from prh_replication.plots import heatmap, lines
from prh_replication.registry import MODELS, Paths
from prh_replication.release_anisotropy import (
    NEAR_ZERO_S,
    fit_one_sided,
    fit_one_sided_shared,
    fro_cosine,
    fro_distance,
    match_response_directions,
    native_anisotropy,
    projector_agreement,
    response_grams,
    select_rho_by_val,
    select_rho_shared,
    signed_correction,
    split_delta_psd,
    two_sided_max_budget,
    up_down_projectors,
)


VIS = ["dinov2-small", "vit-in21k-small", "clip-laion-base"]


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/sdb1/prh-replication-work")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--only-model", default="")
    p.add_argument("--force", action="store_true", help="rerun even if summary.json already exists")
    return p.parse_args()


def load_splits(paths: Paths, repo: Path):
    man_path = paths.data / "manifests" / "coco_val2017.json"
    slim_path = repo / "data" / "manifests" / "coco_val2017_splits.json"
    try:
        man = load_manifest(man_path)
        if "splits" not in man:
            raise ValueError("missing splits")
    except (OSError, ValueError, json.JSONDecodeError):
        man = load_manifest(slim_path)
    for split in ("train", "val", "test"):
        ids = list(man["splits"][split])
        assert len(ids) == len(set(ids)), split
    train, val, test = [list(man["splits"][s]) for s in ("train", "val", "test")]
    assert not (set(train) & set(val) | set(train) & set(test) | set(val) & set(test))
    return man, {"train": train, "val": val, "test": test}


def dump_fit(fit: dict) -> dict:
    keep = dict(fit)
    for k in ("b_a", "s_a", "eig_a", "b_b", "s_b", "eig_b"):
        if k in keep and isinstance(keep[k], np.ndarray):
            keep[k] = keep[k].tolist()
    return jsonable(keep)


def load_xy(cache, key, split):
    return cache[key][split]


def pack_ab(cache, pca, a, b, split):
    xa, xb = load_xy(cache, a, split), load_xy(cache, b, split)
    za = xa - pca[a]["mu"]
    zb = xb - pca[b]["mu"]
    return make_pack(za, zb, pca[a]["U"], pca[b]["U"]), za, zb


def eval_onesided(za, zb, ua, ub, ba, n_perm: int, seed: int, k: int = 10):
    qa = ua.shape[1]
    eye_b = np.eye(ub.shape[1])
    ka = torch.tensor(metric_gram(za, ua, ba), dtype=torch.float64)
    kb = torch.tensor(metric_gram(zb, ub, eye_b), dtype=torch.float64)
    st = extension_stats(ka.float(), kb.float())
    kca = center_gram_o2(ka)
    nkc = float(torch.linalg.norm(kca, ord="fro"))
    trk = float(kca.trace())
    r_eff = (trk * trk / (nkc * nkc)) if nkc > 0 else float("nan")
    mc = mc_cka_mean(ka.float(), kb.float(), n_perm, seed) if n_perm else {}
    dsq_a = metric_sq_distances(za, ua, ba)
    dsq_b = metric_sq_distances(zb, ub, eye_b)
    knn_a, knn_b = knn_from_sq(dsq_a, k), knn_from_sq(dsq_b, k)
    knn_id_a = knn_from_sq(metric_sq_distances(za, ua, np.eye(qa)), k)
    return {
        **{kk: st[kk] for kk in ("a", "b", "ratio", "excess", "official_cka", "valid_a", "degenerate", "ratio_undefined")},
        "r_eff_k": r_eff,
        "centred_diag_energy_frac": centred_diag_energy(kca.numpy()),
        "mnn_k10": mnn_from_knn(knn_a, knn_b),
        "knn_overlap_identity_a": knn_overlap(knn_a, knn_id_a),
        **mc,
    }


def native_pair(xa: np.ndarray, xb: np.ndarray, n_perm: int, seed: int):
    ta = torch.tensor(xa, dtype=torch.float32)
    tb = torch.tensor(xb, dtype=torch.float32)
    k = linear_gram(ta.double())
    l = linear_gram(tb.double())
    st = extension_stats(k, l)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(tb.shape[0], generator=g)
    mnn = mutual_knn_score(ta, tb, 10)
    mnn_shuffle = mutual_knn_score(ta, tb[perm], 10)
    mc = mc_cka_mean(k.float(), l.float(), n_perm, seed) if n_perm else {}
    return {
        "cka_a": st["a"],
        "cka_b": st["b"],
        "cka_ratio": st["ratio"],
        "cka_excess": st["excess"],
        "mnn_k10": mnn,
        "mnn_shuffle": mnn_shuffle,
        **mc,
    }


def partner_sets(base_qwen, base_olmo, supp):
    base = base_qwen + base_olmo
    out = {}
    for a in base_qwen:
        out[a] = [x for x in base if x != a] + VIS
    for a in base_olmo:
        out[a] = [x for x in base if x != a] + VIS
    for a in supp:
        out[a] = list(base_olmo) + [x for x in supp if x != a] + VIS
    return out


def native_pairs(base_qwen, base_olmo, supp):
    pairs = []
    for a in base_qwen:
        for b in base_olmo:
            pairs.append((a, b, "cross_family_base"))
    for fam in (base_qwen, base_olmo, supp):
        for i, a in enumerate(fam):
            for b in fam[i + 1 :]:
                tag = "within_supp" if fam is supp else "within_family_base"
                pairs.append((a, b, tag))
    for a in base_qwen + base_olmo + supp:
        for v in VIS:
            panel = "vl_supp" if a in supp else "vl_base"
            pairs.append((a, v, panel))
    for a in supp:
        for b in base_olmo:
            pairs.append((a, b, "supp_qwen_x_olmo_base"))
    # unique
    seen = set()
    uniq = []
    for a, b, t in pairs:
        key = tuple(sorted((a, b))) + (t,)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((a, b, t))
    return uniq


def write_resource_plan(out: Path, design: dict, specs: dict, splits: dict):
    n = {s: len(v) for s, v in splits.items()}
    plan = {
        "protocol": PROTOCOL,
        "evaluation_status": "exploratory",
        "primary_objective": "CKA excess a-b; documented before inspecting new test results",
        "cache_plan": {
            "vision": "reuse last CLS from existing original-protocol caches; rewrite into coco_val2017_final_block_pre_norm/",
            "language": "new caches only; do not reuse all-hidden_states stacks",
            "reuse_rule": "revision, layer, pooling, clip+L2 protocol must match",
        },
        "n_per_split": n,
        "n_language_models": len(specs),
        "estimated_weight_gb": "~15GB x six 7B/8B + ~8GB x three 4B",
        "extraction_device": design.get("resource_estimate", {}),
        "do_not_evict": "vllm-akkadian.service",
        "checkpointing": "per-model extract_status.json; per-pair onesided_fits/*.json",
        "gallery": "same test n for every comparison",
    }
    write_json(out / "resource_plan.json", plan)
    write_json(out / "design.json", design)
    return plan


def extract_all(paths, specs, splits, records_by_split, only_model, extract_status):
    blockers = []
    vis_ok = True
    for split, ids in splits.items():
        recs = records_by_split[split]
        for v in VIS:
            try:
                slice_vision_final(paths, v, split, ids)
                extract_status[f"vision:{v}:{split}"] = "ok"
            except Exception as e:
                vis_ok = False
                extract_status[f"vision:{v}:{split}"] = f"FAIL:{e}"
                blockers.append({"model": v, "split": split, "error": str(e), "trace": traceback.format_exc()[-1500:]})
    if not vis_ok:
        print("VISION SLICE FAILED", flush=True)
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
                break  # do not attempt remaining splits if load failed
        write_json(paths.results / "release_anisotropy" / "extract_status.json", extract_status)
    return blockers, extract_status


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


def main():
    args = parse_args()
    paths = Paths(work=Path(args.work), repo=ROOT)
    out = paths.results / "release_anisotropy"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.force and not args.smoke:
        print("release_anisotropy already complete; pass --force to rerun", flush=True)
        return
    (out / "onesided_fits").mkdir(exist_ok=True)
    design = json.loads((ROOT / "configs" / "release_anisotropy.json").read_text())
    manifest = json.loads((ROOT / "data" / "manifests" / "release_models.json").read_text())
    write_json(out / "model_inventory.json", manifest)

    man, splits = load_splits(paths, ROOT)
    rows = list(manifest["base_panel"]) + list(manifest["supplementary_qwen3x"])
    specs = {r["key"]: spec_from_manifest(r) for r in rows}
    base_qwen = [r["key"] for r in manifest["base_panel"] if r["family"] == "Qwen"]
    base_olmo = [r["key"] for r in manifest["base_panel"] if r["family"] == "OLMo"]
    supp = [r["key"] for r in manifest["supplementary_qwen3x"]]
    q = 4 if args.smoke else int(design["q"])
    rhos = [0.0, 0.4] if args.smoke else list(design["budgets_rho"])
    n_steps = 25 if args.smoke else 120
    n_perm = 4 if args.smoke else args.n_perm
    write_resource_plan(out, design, specs, splits)

    extract_status = {}
    sp = out / "extract_status.json"
    if sp.exists():
        extract_status = json.loads(sp.read_text())

    blockers = []
    if not args.skip_extract:
        try:
            recs = {split: records_for_ids(man, ids, paths) for split, ids in splits.items()}
        except Exception as e:
            recs = None
            blockers.append({"model": "captions", "error": str(e), "trace": traceback.format_exc()[-1500:]})
        if recs is not None:
            b2, extract_status = extract_all(paths, specs, splits, recs, args.only_model, extract_status)
            blockers.extend(b2)
            write_json(out / "extract_status.json", extract_status)
            write_json(out / "blockers.json", jsonable(blockers))
    if args.extract_only:
        print("extract-only done", flush=True)
        return

    # Load whatever extracted successfully
    available = set(VIS)
    for key, spec in specs.items():
        ok = all(extract_status.get(f"lang:{key}:{split}") == "ok" or final_feature_path(paths, spec, "coco_val2017", split, splits[split]).exists() for split in splits)
        if ok:
            # verify
            try:
                for split, ids in splits.items():
                    load_final_prepared(paths, spec, split, ids)
                available.add(key)
            except Exception as e:
                blockers.append({"model": key, "stage": "load", "error": str(e)})
        else:
            print(f"missing features {key}", flush=True)
    write_json(out / "available_models.json", sorted(available))
    write_json(out / "blockers.json", jsonable(blockers))

    base_qwen = [k for k in base_qwen if k in available]
    base_olmo = [k for k in base_olmo if k in available]
    supp = [k for k in supp if k in available]
    langs = base_qwen + base_olmo + supp
    if not langs:
        print("No language features; writing inventory-only report", flush=True)
        _write_partial_report(ROOT, out, manifest, blockers, extract_status)
        return

    cache = {}
    meta = {}
    for key in list(langs) + [v for v in VIS if v in available]:
        spec = specs.get(key) or MODELS[key]
        cache[key] = {}
        for split, ids in splits.items():
            x, obj = load_final_prepared(paths, spec, split, ids)
            cache[key][split] = x.numpy().astype(np.float64)
            if split == "train":
                meta[key] = {
                    "hidden_dim": int(x.shape[1]),
                    "layer_names": obj.get("layer_names"),
                    "layer_path": obj.get("layer_path"),
                    "model_class": obj.get("model_class"),
                    "n_params_runtime": obj.get("n_params"),
                    "device": obj.get("device"),
                    "reused": obj.get("reused", False),
                    "checkpoint": obj.get("checkpoint"),
                    "revision": obj.get("revision"),
                }
        print(f"loaded {key} {cache[key]['train'].shape}", flush=True)

    # PCA / means on train only, one U per model
    pca = {}
    pca_tbl = {}
    for key in cache:
        x = cache[key]["train"]
        mu = x.mean(0)
        z = x - mu
        rec = fit_pca(z, q=q)
        pca[key] = {"mu": mu, "U": rec["U"], "q": rec["q"]}
        aniso = native_anisotropy(z)
        pca_tbl[key] = jsonable({**{k: rec[k] for k in rec if k != "U"}, "anisotropy_train_centered": aniso, "meta": meta.get(key)})
    write_json(out / "pca.json", pca_tbl)
    write_json(out / "native_anisotropy.json", {k: pca_tbl[k]["anisotropy_train_centered"] for k in pca_tbl})

    # Native alignment on TEST gallery (identity protocol)
    npairs = native_pairs(base_qwen, base_olmo, supp)
    native_rows = []
    for a, b, tag in npairs:
        if a not in cache or b not in cache:
            continue
        rec = native_pair(cache[a]["test"], cache[b]["test"], n_perm, seed=abs(hash((a, b))) % 10_000)
        rec.update({"a": a, "b": b, "tag": tag, "split": "test", "n": cache[a]["test"].shape[0]})
        native_rows.append(rec)
        print(f"native {a} vs {b} a={rec['cka_a']:.4f} mnn={rec['mnn_k10']:.4f}", flush=True)
    write_json(out / "native_matrix.json", jsonable(native_rows))

    partners = partner_sets(base_qwen, base_olmo, supp)
    partners = {k: [p for p in v if p in cache] for k, v in partners.items() if k in cache}

    # One-sided fits
    fits = {}
    pair_eval = []
    for a, plist in partners.items():
        for b in plist:
            cands = {}
            for rho in rhos:
                fpath = out / "onesided_fits" / f"{a}__{b}__rho{rho}.json"
                if fpath.exists() and not args.smoke:
                    raw = json.loads(fpath.read_text())
                    raw["b_a"] = np.array(raw["b_a"], dtype=np.float64)
                    raw["s_a"] = np.array(raw["s_a"], dtype=np.float64)
                    raw["eig_a"] = np.array(raw["eig_a"], dtype=np.float64)
                    fit = raw
                else:
                    pack_tr, _, _ = pack_ab(cache, pca, a, b, "train")
                    fit = fit_one_sided(pack_tr, rho=float(rho), n_steps=n_steps)
                    write_json(fpath, dump_fit(fit))
                cands[float(rho)] = fit
                pack_va, _, _ = pack_ab(cache, pca, a, b, "val")
                pack_te, za_te, zb_te = pack_ab(cache, pca, a, b, "test")
                ba = fit["b_a"]
                ev_va = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_va["q_b"], pack_va["q_b"])), pack_va)
                ev_te = eval_onesided(za_te, zb_te, pca[a]["U"], pca[b]["U"], ba, n_perm=0, seed=0)
                pair_eval.append({
                    "a": a, "b": b, "rho": float(rho), "kind": "one_sided",
                    "d_attained": fit["d_attained"],
                    "frac_at_bounds": fit.get("diag_a", {}).get("frac_at_bounds"),
                    "train": fit["train"],
                    "val": ev_va,
                    "test": ev_te,
                    "selected_start": fit.get("selected_start"),
                    "eig_min": fit.get("diag_a", {}).get("eig_min"),
                    "eig_max": fit.get("diag_a", {}).get("eig_max"),
                })
            pack_va, _, _ = pack_ab(cache, pca, a, b, "val")
            rho_h, _ = select_rho_by_val(cands, pack_va)
            fits[f"{a}__{b}"] = {"cands": cands, "headline_rho": float(rho_h)}
            print(f"onesided {a}|{b} headline_rho={rho_h}", flush=True)
    write_json(out / "pair_eval.json", jsonable(pair_eval))
    write_json(out / "headline_rho.json", {k: v["headline_rho"] for k, v in fits.items()})

    # Two-sided bridge at max budget, cross-family base
    two_rows = []
    rho_max = max(rhos)
    for a in base_qwen:
        for b in base_olmo:
            if a not in cache or b not in cache:
                continue
            pack_tr, _, _ = pack_ab(cache, pca, a, b, "train")
            fit = two_sided_max_budget(pack_tr, n_steps=n_steps)
            pack_te, za, zb = pack_ab(cache, pca, a, b, "test")
            sc = scores_numpy(fit["b_a"] - np.eye(fit["b_a"].shape[0]), fit["b_b"] - np.eye(fit["b_b"].shape[0]), pack_te)
            two_rows.append({
                "a": a, "b": b, "kind": "two_sided_max_budget",
                "r": fit["r"],
                "d_a": fit["diag_a"]["r_side"],
                "d_b": fit["diag_b"]["r_side"],
                "test": sc,
                "train": fit["train"],
            })
            print(f"twosided {a}--{b} test a={sc['a']:.4f}", flush=True)
    write_json(out / "two_sided.json", jsonable(two_rows))

    # Consistency across partners at each rho
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
                        sa, sb_ = fa["s_a"], fb["s_a"]
                        mag_a, mag_b = float(np.linalg.norm(sa)), float(np.linalg.norm(sb_))
                        pu, pd, infa = up_down_projectors(sa)
                        qu, qd, infb = up_down_projectors(sb_)
                        za = cache[a]["test"] - pca[a]["mu"]
                        da = signed_correction(pca[a]["U"], fa["b_a"])
                        db = signed_correction(pca[a]["U"], fb["b_a"])
                        ga = response_grams(za, da)
                        gb = response_grams(za, db)
                        cons.append({
                            "model": a, "partner_i": b, "partner_j": c, "rho": float(rho), "group": group_name,
                            "s_fro_i": mag_a, "s_fro_j": mag_b,
                            "direction_defined": mag_a > NEAR_ZERO_S and mag_b > NEAR_ZERO_S,
                            "s_fro_cosine": fro_cosine(sa, sb_) if mag_a > NEAR_ZERO_S and mag_b > NEAR_ZERO_S else None,
                            "s_fro_distance": fro_distance(sa, sb_),
                            "up_agreement": projector_agreement(pu, qu),
                            "down_agreement": projector_agreement(pd, qd),
                            "induced_deltaK_cosine": fro_cosine(ga, gb),
                            "info_i": infa, "info_j": infb,
                        })
    write_json(out / "consistency.json", jsonable(cons))

    # Direct transfer
    transfer = []
    for a, plist in partners.items():
        for b in plist:
            for c in plist:
                if b == c:
                    continue
                for rho in rhos:
                    fit_b = fits[f"{a}__{b}"]["cands"][float(rho)]
                    fit_c = fits[f"{a}__{c}"]["cands"][float(rho)]
                    pack_te, za, zc = pack_ab(cache, pca, a, c, "test")
                    qb = pack_te["q_b"]
                    sc_tr = scores_numpy(fit_b["b_a"] - np.eye(fit_b["b_a"].shape[0]), np.zeros((qb, qb)), pack_te)
                    sc_dir = scores_numpy(fit_c["b_a"] - np.eye(fit_c["b_a"].shape[0]), np.zeros((qb, qb)), pack_te)
                    sc_id = scores_numpy(*[np.zeros((pack_te["q_a"], pack_te["q_a"])), np.zeros((qb, qb))], pack_te)
                    transfer.append({
                        "model": a, "fit_partner": b, "eval_partner": c, "rho": float(rho),
                        "eval_modality": "vision" if c in VIS else "language",
                        "identity_excess": sc_id["excess"],
                        "transferred_excess": sc_tr["excess"],
                        "direct_excess": sc_dir["excess"],
                        "identity_a": sc_id["a"],
                        "transferred_a": sc_tr["a"],
                        "direct_a": sc_dir["a"],
                        "delta_vs_identity": sc_tr["excess"] - sc_id["excess"],
                        "gap_vs_direct": sc_tr["excess"] - sc_dir["excess"],
                        "d_transferred": fit_b["d_attained"],
                        "d_direct": fit_c["d_attained"],
                    })
    write_json(out / "transfer.json", jsonable(transfer))

    # LOPO shared metrics
    lopo_rows = []
    groups = {
        "vision_anchors": VIS,
        "olmo_base": list(base_olmo),
        "qwen_base": list(base_qwen),
    }
    for a in langs:
        for gname, gparts in groups.items():
            group = [p for p in gparts if p in cache and p != a]
            if len(group) < 2:
                continue
            # skip mismatched: olmo_base group only for Qwen models, etc.
            if gname == "olmo_base" and a not in base_qwen and a not in supp:
                continue
            if gname == "qwen_base" and a not in base_olmo:
                continue
            for held in group:
                others = [p for p in group if p != held]
                cands = {}
                for rho in rhos:
                    packs_tr = [pack_ab(cache, pca, a, p, "train")[0] for p in others]
                    fit = fit_one_sided_shared(packs_tr, rho=float(rho), n_steps=n_steps)
                    cands[float(rho)] = fit
                packs_va = [pack_ab(cache, pca, a, p, "val")[0] for p in others]
                rho_h, chosen = select_rho_shared(cands, packs_va)
                pack_te, _, _ = pack_ab(cache, pca, a, held, "test")
                ba = chosen["b_a"]
                sc = scores_numpy(ba - np.eye(ba.shape[0]), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                sc_id = scores_numpy(np.zeros((pack_te["q_a"], pack_te["q_a"])), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                direct = fits.get(f"{a}__{held}")
                dir_sc = None
                if direct:
                    bd = direct["cands"][float(rho_h)]["b_a"]
                    dir_sc = scores_numpy(bd - np.eye(bd.shape[0]), np.zeros((pack_te["q_b"], pack_te["q_b"])), pack_te)
                lopo_rows.append({
                    "model": a, "group": gname, "held_out_partner": held, "fit_partners": others,
                    "selected_rho": float(rho_h),
                    "d_attained": chosen["d_attained"],
                    "test": sc,
                    "identity": sc_id,
                    "direct_at_same_rho": dir_sc,
                    "held_out_modality": "vision" if held in VIS else "language",
                    "note": "budget selected on remaining partners' val only; held-out partner excluded from train and val",
                })
                print(f"lopo {a} hold {held} rho={rho_h} excess={sc['excess']:.4f}", flush=True)
    # language-shared vs vision
    for a in base_qwen + base_olmo:
        lang_partners = [p for p in partners.get(a, []) if p not in VIS]
        vis_p = [p for p in VIS if p in cache]
        if len(lang_partners) < 2 or not vis_p:
            continue
        cands = {}
        for rho in rhos:
            packs_tr = [pack_ab(cache, pca, a, p, "train")[0] for p in lang_partners]
            cands[float(rho)] = fit_one_sided_shared(packs_tr, rho=float(rho), n_steps=n_steps)
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

    # Cross-release response signatures (fixed partner)
    sig_rows = []
    fixed_partners = []
    if "dinov2-small" in cache:
        fixed_partners.append("dinov2-small")
    if base_olmo:
        fixed_partners.append(base_olmo[-1])
    if base_qwen:
        fixed_partners.append(base_qwen[-1])
    for family, members in (("Qwen-base", base_qwen), ("OLMo-base", base_olmo), ("Qwen3x-supp", supp)):
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
                        # common gallery: test examples; Z is model-specific
                        d1 = signed_correction(pca[m1]["U"], f1["b_a"])
                        d2 = signed_correction(pca[m2]["U"], f2["b_a"])
                        z1 = cache[m1]["test"] - pca[m1]["mu"]
                        z2 = cache[m2]["test"] - pca[m2]["mu"]
                        p1, n1 = split_delta_psd(d1)
                        p2, n2 = split_delta_psd(d2)
                        g1p, g1n = response_grams(z1, p1), response_grams(z1, n1)
                        g2p, g2n = response_grams(z2, p2), response_grams(z2, n2)
                        g1s, g2s = response_grams(z1, d1), response_grams(z2, d2)
                        k_nat = fro_cosine(center_gram_o2(torch.tensor(z1 @ z1.T)).numpy(), center_gram_o2(torch.tensor(z2 @ z2.T)).numpy())
                        mag1 = float(np.linalg.norm(g1s))
                        mag2 = float(np.linalg.norm(g2s))
                        sig_rows.append({
                            "family": family, "partner": partner, "rho": float(rho),
                            "model_i": m1, "model_j": m2,
                            "correction_mag_i": mag1, "correction_mag_j": mag2,
                            "near_zero": mag1 <= NEAR_ZERO_S or mag2 <= NEAR_ZERO_S,
                            "gplus_fro_cosine": fro_cosine(g1p, g2p),
                            "gminus_fro_cosine": fro_cosine(g1n, g2n),
                            "signed_deltaK_cosine": fro_cosine(g1s, g2s),
                            "native_gram_cosine": k_nat,
                            "signed_note": "signed cosine is not PSD-kernel CKA",
                        })
    write_json(out / "signatures.json", jsonable(sig_rows))

    # Direction match earliest/latest vs dinov2 at max rho using train matching
    dir_rows = []
    rho_max = max(rhos)
    for fam, members in (("Qwen-base", base_qwen), ("OLMo-base", base_olmo)):
        if len(members) < 2 or "dinov2-small" not in cache:
            continue
        early, late = members[0], members[-1]
        k1, k2 = f"{early}__dinov2-small", f"{late}__dinov2-small"
        if k1 not in fits or k2 not in fits:
            continue
        f1 = fits[k1]["cands"][float(rho_max)]
        f2 = fits[k2]["cands"][float(rho_max)]
        rec = match_response_directions(
            cache[early]["train"] - pca[early]["mu"],
            cache[late]["train"] - pca[late]["mu"],
            cache[early]["test"] - pca[early]["mu"],
            cache[late]["test"] - pca[late]["mu"],
            pca[early]["U"], pca[late]["U"], f1["s_a"], f2["s_a"], k=4,
        )
        rec.update({"family": fam, "early": early, "late": late, "partner": "dinov2-small", "rho": float(rho_max)})
        dir_rows.append(rec)
    write_json(out / "direction_match.json", jsonable(dir_rows))

    # Stability: frozen U, 3 train subsets, max rho, vs each vision
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
                ss = []
                for si, idx in enumerate(idxs):
                    ii = idx.numpy()
                    za = cache[m]["train"][ii] - pca[m]["mu"]
                    zb = cache[v]["train"][ii] - pca[v]["mu"]
                    pack = make_pack(za, zb, pca[m]["U"], pca[v]["U"])
                    fit = fit_one_sided(pack, rho=float(rho_max), n_steps=n_steps)
                    ss.append(fit)
                    stab.append({"model": m, "partner": v, "subset": si, "d": fit["d_attained"], "s": fit["s_a"].tolist(), "train_excess": fit["train"]["excess"]})
                # pairwise S cosine within partner
                for i in range(len(ss)):
                    for j in range(i + 1, len(ss)):
                        stab.append({
                            "model": m, "partner": v, "kind": "within_partner_refit",
                            "subset_i": i, "subset_j": j,
                            "s_fro_cosine": fro_cosine(ss[i]["s_a"], ss[j]["s_a"]),
                            "s_fro_distance": fro_distance(ss[i]["s_a"], ss[j]["s_a"]),
                        })
    write_json(out / "stability.json", jsonable(stab))

    # Shuffle-fit controls
    shuf = []
    sh_models = [m for m in design["shuffle"]["models"] if m in cache]
    partner = design["shuffle"]["partner"]
    seeds = design["shuffle"]["seeds"] if not args.smoke else [0]
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
                pack_tr = make_pack(xa_tr - mu_a, xb_tr[p_tr] - mu_b, ua, ub)
                fit = fit_one_sided(pack_tr, rho=float(rho_max), n_steps=n_steps)
                pack_va = make_pack(xa_va - mu_a, xb_va[p_va] - mu_b, ua, ub)
                # selection would use shuffled val; we still evaluate the fitted metric
                sc_true = eval_onesided(xa_te - mu_a, xb_te - mu_b, ua, ub, fit["b_a"], n_perm=0, seed=0)
                sc_sh = eval_onesided(xa_te - mu_a, xb_te[p_te] - mu_b, ua, ub, fit["b_a"], n_perm=0, seed=0)
                pack_sh_te = make_pack(xa_te - mu_a, xb_te[p_te] - mu_b, ua, ub)
                id_sh = scores_numpy(np.zeros((ua.shape[1], ua.shape[1])), np.zeros((ub.shape[1], ub.shape[1])), pack_sh_te)
                shuf.append({
                    "model": m, "partner": partner, "seed": int(seed),
                    "true_test": sc_true, "shuffled_test": sc_sh,
                    "identity_true_test": id_te, "identity_shuffled_test": id_sh,
                    "delta_a_true_vs_id": sc_true["a"] - id_te["a"],
                    "d_attained": fit["d_attained"],
                })
                print(f"shuffle {m} seed={seed} dA={sc_true['a'] - id_te['a']:.4f}", flush=True)
    write_json(out / "shuffle_fit.json", jsonable(shuf))

    _plots(out, native_rows, pair_eval, pca_tbl, langs, base_qwen, base_olmo, supp, rhos, cons)
    _write_report(ROOT, out, design, manifest, blockers, extract_status, available, native_rows, pair_eval, cons, transfer, lopo_rows, sig_rows, shuf, stab, two_rows, pca_tbl, fits, langs, base_qwen, base_olmo, supp, rhos)
    print("done", flush=True)


def _mean(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _plots(out, native_rows, pair_eval, pca_tbl, langs, base_qwen, base_olmo, supp, rhos, cons):
    def mat(keys_a, keys_b, field):
        m = np.full((len(keys_a), len(keys_b)), np.nan)
        lookup = {(r["a"], r["b"]): r for r in native_rows}
        lookup.update({(r["b"], r["a"]): r for r in native_rows})
        for i, a in enumerate(keys_a):
            for j, b in enumerate(keys_b):
                if a == b:
                    continue
                rec = lookup.get((a, b))
                if rec:
                    m[i, j] = rec[field]
        return m

    if base_qwen and base_olmo:
        heatmap(mat(base_qwen, base_olmo, "cka_a"), base_olmo, base_qwen, "Native linear CKA (test, final layer)", out / "native_cka_qwen_olmo.png", "CKA a")
        heatmap(mat(base_qwen, base_olmo, "mnn_k10"), base_olmo, base_qwen, "Native mutual kNN k=10 (test, final layer)", out / "native_mnn_qwen_olmo.png", "mNN")
    # distortion curves: mean test excess vs rho for VL base
    series = {}
    for a in langs:
        ys = []
        for rho in rhos:
            vals = [r["test"]["excess"] for r in pair_eval if r["a"] == a and r["rho"] == float(rho) and r["b"] in VIS]
            ys.append(_mean(vals))
        series[a] = ys
    if series:
        lines([float(r) for r in rhos], series, "One-sided VL test excess vs distortion budget", "rho", "CKA excess a-b", out / "excess_vs_rho_vl.png")
    ranks = [pca_tbl[k]["anisotropy_train_centered"]["participation_effective_rank"] for k in langs if k in pca_tbl]
    if ranks:
        heatmap(np.array(ranks, dtype=float)[None, :], langs, ["rank"], "Native participation effective rank (train, centred)", out / "native_effrank.png", "rank")


def _write_partial_report(repo, out, manifest, blockers, extract_status):
    text = _inventory_md(manifest) + "\n\n## Blockers\n\n" + json.dumps(blockers, indent=2)[:8000]
    (repo / "docs" / "RELEASE_ANISOTROPY_REPORT.md").write_text(text)
    write_json(out / "summary.json", {"status": "extract_incomplete", "blockers": blockers, "extract_status": extract_status})


def _inventory_md(manifest) -> str:
    lines_ = ["# Release alignment and anisotropic corrections — report", "",
              "**Status:** exploratory COCO test gallery (n=1024). Final-layer protocol; not comparable to previous max-over-layers PRH means.",
              "", "## Model inventory", "",
              "Public dates are announcement/card dates, not Hub `lastModified`. Revisions frozen at audit 2026-09-20.", ""]
    lines_.append("| key | repo | revision | public date | params | stage | multimodal | arch |")
    lines_.append("|---|---|---|---|---|---|---|---|")
    for r in manifest["base_panel"] + manifest["supplementary_qwen3x"]:
        rev = (r.get("revision") or "")[:12]
        lines_.append(f"| {r['key']} | {r['checkpoint']} | `{rev}` | {r.get('public_release_date')} | {r.get('n_params')} | {r.get('stage')} | {r.get('multimodal')} | {r.get('architecture')} |")
    lines_.append("")
    lines_.append("Vision anchors (reused last-block CLS): DINOv2-S, ViT-IN21k-S, CLIP-LAION-B.")
    lines_.append("")
    lines_.append("Base panel is approximately size-matched 7B/8B pretrained checkpoints. Qwen3-8B-Base is 8.19B vs 7.62B earlier Qwen; treat as size-matched, not equal. Supplementary Qwen 3.x 4B panel mixes instruct, thinking-capable, and multimodal post-training — not a matched base series.")
    return "\n".join(lines_)


def _write_report(repo, out, design, manifest, blockers, extract_status, available, native_rows, pair_eval, cons, transfer, lopo, sigs, shuf, stab, two_rows, pca_tbl, fits, langs, base_qwen, base_olmo, supp, rhos):
    def filt(rows, **kw):
        outr = rows
        for k, v in kw.items():
            if callable(v):
                outr = [r for r in outr if v(r)]
            else:
                outr = [r for r in outr if r.get(k) == v]
        return outr

    def mean_field(rows, path):
        vals = []
        for r in rows:
            cur = r
            ok = True
            for p in path.split("."):
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and isinstance(cur, (int, float)) and math.isfinite(cur):
                vals.append(float(cur))
        return _mean(vals), len(vals)

    native_ll = [r for r in native_rows if r["tag"] in ("cross_family_base", "within_family_base", "supp_qwen_x_olmo_base", "within_supp")]
    native_vl = [r for r in native_rows if r["tag"] in ("vl_base", "vl_supp")]
    id_vl = filt(pair_eval, rho=0.0, b=lambda r: r["b"] in VIS)
    maxrho = max(rhos)
    max_vl = [r for r in pair_eval if r["rho"] == float(maxrho) and r["b"] in VIS]

    a_id, n_id = mean_field(id_vl, "test.a")
    a_max, n_max = mean_field(max_vl, "test.a")
    ex_id, _ = mean_field(id_vl, "test.excess")
    ex_max, _ = mean_field(max_vl, "test.excess")
    mnn_id, _ = mean_field(id_vl, "test.mnn_k10")
    mnn_max, _ = mean_field(max_vl, "test.mnn_k10")
    rank_nat = _mean([pca_tbl[k]["anisotropy_train_centered"]["participation_effective_rank"] for k in langs if k in pca_tbl])

    cons_def = [c for c in cons if c.get("direction_defined") and c.get("rho") == float(maxrho) and c.get("group") == "all"]
    cons_cos = _mean([c["s_fro_cosine"] for c in cons_def if c.get("s_fro_cosine") is not None])

    tr_max = [t for t in transfer if t["rho"] == float(maxrho)]
    tr_gain = _mean([t["delta_vs_identity"] for t in tr_max])
    tr_gap = _mean([t["gap_vs_direct"] for t in tr_max])

    lopo_gain = _mean([(r["test"]["excess"] - r["identity"]["excess"]) for r in lopo if r.get("test") and r.get("identity")])
    sh_da = _mean([s["delta_a_true_vs_id"] for s in shuf])

    summary = {
        "evaluation_status": "exploratory",
        "protocol": PROTOCOL,
        "available_models": sorted(available),
        "n_native_pairs": len(native_rows),
        "native_ll_mean_cka": mean_field(native_ll, "cka_a")[0],
        "native_ll_mean_mnn": mean_field(native_ll, "mnn_k10")[0],
        "native_vl_mean_cka": mean_field(native_vl, "cka_a")[0],
        "native_vl_mean_mnn": mean_field(native_vl, "mnn_k10")[0],
        "onesided_vl_identity_mean_a": a_id,
        "onesided_vl_maxrho_mean_a": a_max,
        "onesided_vl_identity_mean_excess": ex_id,
        "onesided_vl_maxrho_mean_excess": ex_max,
        "onesided_vl_identity_mean_mnn": mnn_id,
        "onesided_vl_maxrho_mean_mnn": mnn_max,
        "native_participation_rank_mean": rank_nat,
        "cross_partner_S_fro_cosine_maxrho": cons_cos,
        "transfer_mean_delta_excess_vs_id_maxrho": tr_gain,
        "transfer_mean_gap_vs_direct_maxrho": tr_gap,
        "lopo_mean_delta_excess_vs_id": lopo_gain,
        "shuffle_mean_delta_a_true": sh_da,
        "n_blockers": len(blockers),
        "objective": "CKA excess a-b",
        "do_not_compare_to_old_max_layer_means": True,
    }
    write_json(out / "summary.json", jsonable(summary))

    md = [_inventory_md(manifest), "",
          "## Protocol", "",
          "- One representation per model: last decoder-block residual before final RMS/LayerNorm (hook), mask-mean pooling, raw captions, no chat template.",
          "- Vision: last ViT-block CLS reused from existing caches (already pre-final-LN).",
          "- Prep: PRH clip q=0.95 then L2. Same test gallery size for every cell.",
          "- Identity numbers are **recomputed under this protocol**. Do not mix with previous max-over-layers PRH means.",
          "- Primary fit objective (frozen before test inspection): **CKA excess a−b**. Distortion is a hard budget on D(M)=||S||_F²/q with eig(B)∈[1/4,4].",
          "- One-sided M_A|B with B at identity is primary. Two-sided max-budget is a bridge only for Qwen-base × OLMo-base.",
          "- Recency is observational. Architecture, data, post-training, and parameter count also change.",
          "- Cells that share a model or the gallery are dependent; no independent-cell p-values.",
          "",
          "## Resource / extraction", "",
          f"Available models: `{sorted(available)}`. Blockers: {len(blockers)} (see `results/release_anisotropy/blockers.json`).",
          "",
          "## 1. Native alignment vs recency", "",
          f"Language–language mean test CKA a = {summary['native_ll_mean_cka']:.4f}, mNN = {summary['native_ll_mean_mnn']:.4f} (self excluded).",
          f"Vision–language mean test CKA a = {summary['native_vl_mean_cka']:.4f}, mNN = {summary['native_vl_mean_mnn']:.4f}.",
          "",
          "Per-cell values: `native_matrix.json`. Heatmaps: `native_cka_qwen_olmo.png`, `native_mnn_qwen_olmo.png`.",
          "Read within-family trajectories from dated labels, not equally spaced version numbers.",
          "",
          "## 2. Recoverable alignment after one-sided anisotropic reweighting", "",
          f"VL identity mean test a = {a_id:.4f} (n={n_id} model–anchor pairs). Max-budget mean test a = {a_max:.4f} (n={n_max}).",
          f"Excess: {ex_id:.4f} → {ex_max:.4f}. mNN: {mnn_id:.4f} → {mnn_max:.4f}.",
          "All preregistered ρ remain in `pair_eval.json` / `excess_vs_rho_vl.png`. Headline ρ is validation-selected per pair (`headline_rho.json`); no interpolation of an optimum from test.",
          "",
          "## 3. Required correction vs recency", "",
          "Compare attained D(M) and headline ρ across dated releases in `pair_eval.json` and `headline_rho.json`. A smaller selected ρ at matched val excess would mean less correction; if val excess still rises with ρ, the smallest evaluated budget that hits a **pre-fixed val target** is reported only when that target was frozen on validation. This run did not pre-declare a numeric target alignment other than the val-argmax among {ρ}; unattained extra targets are unmarked because they were not fixed.",
          "",
          "## 4. Consistency across partners", "",
          f"At max ρ, mean Frobenius cosine of S=log B across partner pairs (defined, non-near-zero): {cons_cos:.4f}.",
          "Near-zero corrections are marked `direction_defined=false`. Subspace agreement uses projectors, not eigenvector order/sign. Cross-partner disagreement vs subset-refit variability is in `consistency.json` vs `stability.json` (conditional on frozen U).",
          "",
          "## 5. Transfer", "",
          f"Direct transfer at max ρ: mean Δexcess vs identity {tr_gain:.4f}; mean gap vs partner-specific fit {tr_gap:.4f}.",
          f"LOPO / shared-metric mean Δexcess vs identity {lopo_gain:.4f}. Unseen partners are not unseen examples; both exclusions are in `transfer.json` and `lopo.json`.",
          "",
          "## 6. Preferred response structure across releases", "",
          "G± = H Z ΔM± Zᵀ H on the common test gallery, compared with normalised Frobenius cosine while holding the fitting partner fixed (`signatures.json`). Zero signatures marked. Signed ΔK cosine is **not** PSD-kernel CKA. Direction matching on earliest/latest vs DINOv2 used **training** examples only (`direction_match.json`).",
          "",
          "## 7. Native anisotropy vs correction anisotropy", "",
          f"Mean train participation effective rank (centred features): {rank_nat:.2f}. Per-model leading covariance eigs and var in first 32 PCs: `native_anisotropy.json`. Learned rank concentration is in pair_eval `r_eff_k` and is a property of M, not of the native covariance.",
          "",
          "## Controls", "",
          f"Shuffle-fit (max ρ, frozen U, true-test Δa vs identity): mean {sh_da:.4f}. Identity at ρ=0 is exact. Direct vs contracted CKA was unit-tested. One U per model across partners. No test in fit/selection. No held-out partner in shared-metric selection.",
          "",
          "## Two-sided bridge", "",
          f"{len(two_rows)} Qwen-base × OLMo-base two-sided max-budget fits in `two_sided.json`, reported separately from one-sided curves.",
          "",
          "## Answers to the seven questions", "",
          _answers(native_rows, pair_eval, pca_tbl, cons, transfer, lopo, sigs, base_qwen, base_olmo, supp, rhos, maxrho),
          "",
          "## What this does not show", "",
          "- Version numbers do not cause alignment.",
          "- Equal metric spectra are not equal preferred directions.",
          "- Higher CKA is not higher retrieval (watch mNN).",
          "- Matrix cells are not independent replicates.",
          "- Final-layer trends are not all-layer trends.",
          "- The COCO test gallery already informed earlier work; this study is exploratory.",
          "",
          "## Reproduction", "",
          "```bash",
          "export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src",
          "cd /mnt/sdb1/prh-replication-work/repo",
          "/mnt/sdb1/prh-replication-work/venv/bin/python -m pytest tests/test_release_anisotropy.py -q",
          "/mnt/sdb1/prh-replication-work/venv/bin/python scripts/run_release_anisotropy.py --work /mnt/sdb1/prh-replication-work",
          "```",
          "Resumable: rerun skips completed feature caches and `onesided_fits/*.json`. `--extract-only` / `--only-model KEY` for sequential checkpoints.",
          ]
    (repo / "docs" / "RELEASE_ANISOTROPY_REPORT.md").write_text("\n".join(md) + "\n")
    # copy summary into git results
    git_res = repo / "results" / "release_anisotropy"
    git_res.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "design.json", "model_inventory.json", "resource_plan.json", "native_matrix.json", "native_anisotropy.json", "pair_eval.json", "headline_rho.json", "consistency.json", "transfer.json", "lopo.json", "signatures.json", "direction_match.json", "shuffle_fit.json", "stability.json", "two_sided.json", "pca.json", "available_models.json", "blockers.json", "extract_status.json"):
        src = out / name
        if src.exists():
            (git_res / name).write_text(src.read_text())


def _answers(native_rows, pair_eval, pca_tbl, cons, transfer, lopo, sigs, base_qwen, base_olmo, supp, rhos, maxrho) -> str:
    def dated_traj(members, partner_fn, field):
        bits = []
        for m in members:
            rows = [r for r in native_rows if (r["a"] == m or r["b"] == m) and partner_fn(r)]
            bits.append(f"{m}:{_mean([r[field] for r in rows]):.4f}")
        return ", ".join(bits)

    qwen_olmo = lambda r: (r["a"] in base_qwen and r["b"] in base_olmo) or (r["b"] in base_qwen and r["a"] in base_olmo)
    vis = lambda r: r["tag"] in ("vl_base", "vl_supp")
    lines = []
    lines.append(f"1. **Native mNN/CKA vs successive releases (observational).** Within-family and cross-family cells are in `native_matrix.json`. Qwen vs OLMo CKA by Qwen release (mean over OLMo partners): {dated_traj(base_qwen, qwen_olmo, 'cka_a')}. Same for mNN: {dated_traj(base_qwen, qwen_olmo, 'mnn_k10')}. OLMo vs Qwen: {dated_traj(base_olmo, qwen_olmo, 'cka_a')}. VL: {dated_traj(base_qwen + base_olmo, vis, 'cka_a')}. Supplementary 4B panel is separate: {dated_traj(supp, vis, 'cka_a')}.")
    def recov(members):
        outb = []
        for m in members:
            id_ = _mean([r["test"]["excess"] for r in pair_eval if r["a"] == m and r["rho"] == 0.0 and r["b"] in VIS])
            mx = _mean([r["test"]["excess"] for r in pair_eval if r["a"] == m and r["rho"] == float(maxrho) and r["b"] in VIS])
            outb.append(f"{m}: {id_:.4f}→{mx:.4f}")
        return "; ".join(outb)
    lines.append(f"2. **Held-out alignment after bounded anisotropic reweighting.** VL excess identity→max ρ: {recov(base_qwen + base_olmo)}. Full curves retain every ρ.")
    def corr(members):
        outb = []
        for m in members:
            d = _mean([r["d_attained"] for r in pair_eval if r["a"] == m and r["rho"] == float(maxrho) and r["b"] in VIS])
            outb.append(f"{m} D={d:.3f}")
        return ", ".join(outb)
    lines.append(f"3. **Do newer releases need less correction?** Attained D at the max budget (always feasible to use the budget) is not ‘required’ correction; required correction is the val-selected ρ (`headline_rho.json`). Max-budget attained D (VL): {corr(base_qwen + base_olmo)}. If headline ρ does not fall with recency, the answer is no for this panel.")
    cons_m = _mean([c["s_fro_cosine"] for c in cons if c.get("direction_defined") and c.get("rho") == float(maxrho) and c.get("group") == "all" and isinstance(c.get("s_fro_cosine"), float)])
    lines.append(f"4. **Consistency across partners.** Mean S Frobenius cosine at max ρ (defined pairs): {cons_m:.4f}. Vision vs language groups are separated in `consistency.json`.")
    tr = [t for t in transfer if t["rho"] == float(maxrho)]
    lines.append(f"5. **Transfer.** Mean transferred Δexcess vs identity { _mean([t['delta_vs_identity'] for t in tr]):.4f}; gap vs direct { _mean([t['gap_vs_direct'] for t in tr]):.4f}. LOPO rows: {len(lopo)}.")
    sig_m = _mean([s["gplus_fro_cosine"] for s in sigs if not s.get("near_zero") and s.get("rho") == float(maxrho)])
    lines.append(f"6. **Cross-release amplified structure (G+ cosine, max ρ, non-near-zero):** {sig_m:.4f}. Native Gram cosine is reported separately; feature coordinates are not aligned.")
    ranks = ", ".join(f"{k}={pca_tbl[k]['anisotropy_train_centered']['participation_effective_rank']:.1f}" for k in base_qwen + base_olmo if k in pca_tbl)
    lines.append(f"7. **Native anisotropy (participation rank, train centred):** {ranks}. Do not infer that better alignment requires higher isotropy.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
