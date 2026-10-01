"""Post-process frozen one-sided fits: diag(B) on stored training-PCA axes.

Does not fit, extract, or overwrite freeze artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from prh_replication.anisotropic_kernels import fit_pca
from prh_replication.extract_final import final_feature_path, load_final_prepared, spec_from_manifest
from prh_replication.io_utils import jsonable, write_json
from prh_replication.plots import heatmap, mean_sd_lines
from prh_replication.registry import MODELS, Paths
from prh_replication.release_anisotropy import ambient_metric_diag
from prh_replication.release_protocol import VIS, load_splits, partner_sets

FIT_NAME = re.compile(r"^(?P<a>.+)__(?P<b>.+)__rho(?P<rho>.+)$")
PRIMARY_RHO = 0.1
OUT_NAME = "learned_anisotropic_weights"
REPAIR_NAME = "release_anisotropy_repair"
PC_NOTE = (
    "PC j is this model's training-PCA column j (descending train variance). "
    "It is not an identified common semantic direction across models. "
    "These plots show within-model partner consistency; cross-release "
    "directional correspondence is the existing response-signature analysis."
)


def parse_fit_stem(stem: str) -> tuple[str, str, float]:
    m = FIT_NAME.match(stem)
    if not m:
        raise ValueError(f"unrecognised fit name {stem}")
    return m.group("a"), m.group("b"), float(m.group("rho"))


def sample_mean_sd(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, bool]:
    """Arithmetic mean and sample SD in the same space as ``stack``."""
    n = stack.shape[axis]
    mean = stack.mean(axis=axis)
    if n < 2:
        sd = np.full(mean.shape, np.nan, dtype=np.float64)
        return mean, sd, False
    sd = stack.std(axis=axis, ddof=1)
    return mean, sd, True


def u_fingerprint(u: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).tobytes()).hexdigest()


def load_fit_b_s(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    b = np.asarray(raw["b_a"], dtype=np.float64)
    s = np.asarray(raw["s_a"], dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(f"B is not square in {path}")
    if s.shape != b.shape:
        raise ValueError(f"S shape mismatch in {path}")
    rho = float(raw.get("rho", parse_fit_stem(path.stem)[2]))
    return {
        "path": str(path),
        "b": b,
        "s": s,
        "diag_b": np.diag(b).copy(),
        "q": int(b.shape[0]),
        "rho": rho,
        "d_attained": raw.get("d_attained"),
        "ok": raw.get("ok"),
    }


def catalog_fits(fit_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not fit_dir.is_dir():
        return rows
    for path in sorted(fit_dir.glob("*.json")):
        a, b, rho = parse_fit_stem(path.stem)
        rec = load_fit_b_s(path)
        rec.update({"model": a, "partner": b, "rho_from_name": rho})
        rows.append(rec)
    return rows


def rho_tag(rho: float) -> str:
    text = f"{rho:.12g}"
    return "rho" + text.replace("-", "m").replace(".", "p")


def partner_group(partner: str) -> str:
    return "vision" if partner in VIS else "language"


def model_meta(manifest: dict, key: str) -> dict[str, Any]:
    for block in ("base_panel", "supplementary_qwen3x", "vision_anchors"):
        for row in manifest.get(block, []):
            if row["key"] == key:
                return {
                    "key": key,
                    "checkpoint": row.get("checkpoint"),
                    "family": row.get("family"),
                    "panel": row.get("panel"),
                    "layer": row.get("layer") or row.get("extraction_hook"),
                    "hidden_size": row.get("hidden_size"),
                    "n_layers": row.get("n_layers"),
                    "kind": row.get("kind", "language" if block != "vision_anchors" else "vision"),
                    "source_block": block,
                }
    if key in MODELS:
        spec = MODELS[key]
        return {
            "key": key,
            "checkpoint": spec.checkpoint,
            "family": spec.family,
            "panel": ",".join(spec.panels),
            "layer": "last ViT block CLS (pre final LN)" if spec.kind == "vision" else None,
            "hidden_size": spec.hidden_size,
            "n_layers": spec.n_layers,
            "kind": spec.kind,
            "source_block": "MODELS",
        }
    return {"key": key, "unavailable_metadata": True}


def expected_partners(manifest: dict) -> dict[str, list[str]]:
    base_qwen = [r["key"] for r in manifest["base_panel"] if r["family"] == "Qwen"]
    base_olmo = [r["key"] for r in manifest["base_panel"] if r["family"] == "OLMo"]
    supp = [r["key"] for r in manifest["supplementary_qwen3x"]]
    return partner_sets(base_qwen, base_olmo, supp)


def reconstruct_train_pca(paths: Paths, spec, sample_ids: list[str], q: int) -> dict[str, Any]:
    x, obj = load_final_prepared(paths, spec, "train", sample_ids)
    z = x.numpy().astype(np.float64)
    mu = z.mean(0)
    rec = fit_pca(z - mu, q=q)
    return {
        "U": rec["U"],
        "q": rec["q"],
        "d": rec["d"],
        "variance_fraction": rec["variance_fraction"],
        "singular_values": rec["singular_values"][: rec["q"]],
        "feature_path": str(final_feature_path(paths, spec, "coco_val2017", "train", sample_ids)),
        "u_sha256": u_fingerprint(rec["U"]),
        "protocol": obj.get("protocol"),
        "layer_names": obj.get("layer_names"),
        "basis": "training PCA columns; descending training variance; not resorted",
    }


def collect_group(
    rows: list[dict[str, Any]],
    model: str,
    rho: float,
    group: str,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    picked = [
        r
        for r in rows
        if r["model"] == model and abs(r["rho"] - rho) < 1e-12 and partner_group(r["partner"]) == group
    ]
    picked.sort(key=lambda r: (0, VIS.index(r["partner"])) if r["partner"] in VIS else (1, r["partner"]))
    notes: list[str] = []
    if not picked:
        return [], np.zeros((0, 0)), np.zeros((0, 0, 0)), notes
    qs = {r["q"] for r in picked}
    if len(qs) != 1:
        notes.append(f"q mismatch among partners: {qs}")
        return [], np.zeros((0, 0)), np.zeros((0, 0, 0)), notes
    names = [r["partner"] for r in picked]
    diag = np.stack([r["diag_b"] for r in picked], axis=0)
    sstack = np.stack([r["s"] for r in picked], axis=0)
    return names, diag, sstack, notes


def _finite_abs_max(arr: np.ndarray) -> float:
    v = np.abs(arr[np.isfinite(arr)])
    if v.size == 0:
        return 1.0
    m = float(v.max())
    return m if m > 0 else 1.0


def _weight_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    cat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values if v.size])
    cat = cat[np.isfinite(cat) & (cat > 0)]
    if cat.size == 0:
        return (0.2, 5.0)
    lo, hi = float(cat.min()), float(cat.max())
    return (max(lo / 1.15, 1e-3), hi * 1.15)


def write_group_tables(
    out: Path,
    model: str,
    group: str,
    rho: float,
    partners: list[str],
    diag: np.ndarray,
    sstack: np.ndarray,
    mean_w: np.ndarray,
    sd_w: np.ndarray,
    sd_ok: bool,
    mean_s: np.ndarray,
    sd_s: np.ndarray,
    meta: dict[str, Any],
    pca: dict[str, Any] | None,
    source_paths: list[str],
) -> None:
    q = mean_w.shape[0]
    pcs = np.arange(1, q + 1)
    tbl = out / "tables"
    tbl.mkdir(parents=True, exist_ok=True)
    tag = f"{model}__{group}__{rho_tag(rho)}"
    header = ["pc_index_1based", "stored_pca_column"] + partners + ["mean_diag_B_weight_space", "sample_sd_diag_B_weight_space"]
    lines = [",".join(header)]
    sd_col = sd_w if sd_ok else np.full(q, np.nan)
    for j in range(q):
        row = [str(pcs[j]), str(j)] + [f"{diag[i, j]:.12g}" for i in range(len(partners))]
        row += [f"{mean_w[j]:.12g}", "nan" if not sd_ok else f"{sd_col[j]:.12g}"]
        lines.append(",".join(row))
    (tbl / f"{tag}__diagB.csv").write_text("\n".join(lines) + "\n")
    np.savez(
        tbl / f"{tag}__arrays.npz",
        partners=np.array(partners),
        diag_B=diag,
        S_stack=sstack,
        mean_diag_B=mean_w,
        sample_sd_diag_B=sd_col,
        mean_S=mean_s,
        sample_sd_S=sd_s,
        pc_index_0based=np.arange(q),
    )
    payload = {
        "model": model,
        "partner_group": group,
        "rho": rho,
        "budget_note": "fixed distortion cap; not a validation-selected operating point",
        "n_partners": len(partners),
        "partners": partners,
        "q": q,
        "basis_ordering": "stored training-PCA columns, descending training variance; not independently sorted",
        "weight_definition": "diag(B) on those axes; not sorted eigenvalues; not exp(diag(S))",
        "mean_sd_space": "arithmetic mean and sample SD (ddof=1) in weight space",
        "sample_sd_available": sd_ok,
        "sample_sd_meaning": "partner variation, not a CI or training-seed uncertainty",
        "source_fits": source_paths,
        "model_metadata": meta,
        "pca": None
        if pca is None
        else {k: pca[k] for k in pca if k != "U" and k != "singular_values"} | {
            "singular_values": np.asarray(pca["singular_values"]).tolist(),
            "U": "omitted_from_json",
        },
        "pc_identification_note": PC_NOTE,
    }
    write_json(tbl / f"{tag}__meta.json", jsonable(payload))


def plot_group(
    fig_dir: Path,
    model: str,
    group: str,
    rho: float,
    partners: list[str],
    diag: np.ndarray,
    mean_w: np.ndarray,
    sd_w: np.ndarray,
    sd_ok: bool,
    mean_s: np.ndarray,
    sd_s: np.ndarray,
    scales: dict[str, Any],
    ambient: dict[str, Any] | None,
) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model}__{group}__{rho_tag(rho)}"
    q = mean_w.shape[0]
    xs = np.arange(1, q + 1)
    pc_ticks = [str(i) for i in xs]
    wrote: list[str] = []
    series = {p: diag[i] for i, p in enumerate(partners)}
    mean_sd_lines(
        xs,
        series,
        mean_w,
        sd_w,
        title=f"{model}  {group} partners  ρ={rho:g}\ndiag(B) on stored training-PCA axes",
        xlabel="training PC index (stored order)",
        ylabel="diag(B) (arithmetic mean in weight space)",
        path=fig_dir / f"{tag}__diagB_mean.png",
        logy=True,
        ylim=tuple(scales["weight_ylim"]),
        identity=1.0,
        sd_available=sd_ok,
        note="Faint lines: individual partners. SD is partner-sample SD, not a CI.",
    )
    wrote.append(str(fig_dir / f"{tag}__diagB_mean.png"))
    log2 = np.log2(np.clip(diag, 1e-12, None))
    heatmap(
        log2,
        pc_ticks,
        partners,
        f"{model}  {group}  ρ={rho:g}  log2(diag(B))",
        fig_dir / f"{tag}__diagB_heatmap.png",
        "log2(weight); 0 = identity",
        cmap="RdBu_r",
        vmin=-scales["log2_abs"],
        vmax=scales["log2_abs"],
    )
    wrote.append(str(fig_dir / f"{tag}__diagB_heatmap.png"))
    heatmap(
        mean_s,
        pc_ticks,
        pc_ticks,
        f"{model}  {group}  ρ={rho:g}  mean S=log(B)",
        fig_dir / f"{tag}__S_mean.png",
        "mean S (weight-partner arithmetic mean)",
        cmap="RdBu_r",
        vmin=-scales["s_abs"],
        vmax=scales["s_abs"],
    )
    wrote.append(str(fig_dir / f"{tag}__S_mean.png"))
    sd_plot = np.array(sd_s, copy=True)
    if not sd_ok:
        sd_plot[:] = np.nan
    heatmap(
        sd_plot,
        pc_ticks,
        pc_ticks,
        f"{model}  {group}  ρ={rho:g}  entrywise sample SD of S"
        + ("" if sd_ok else " (unavailable, n<2)"),
        fig_dir / f"{tag}__S_sd.png",
        "sample SD of S (partner variation)",
        cmap="viridis",
        vmin=0.0,
        vmax=scales["s_sd_max"],
    )
    wrote.append(str(fig_dir / f"{tag}__S_sd.png"))
    if ambient is not None:
        d = ambient["mean"].shape[0]
        xs_a = np.arange(d)
        series_a = {p: ambient["partners"][p] for p in partners}
        mean_sd_lines(
            xs_a,
            series_a,
            ambient["mean"],
            ambient["sd"],
            title=f"{model}  {group}  ρ={rho:g}\ndiag(M)=1+diag(U(B−I)Uᵀ)  original hidden coordinates",
            xlabel="original hidden coordinate index (not PCA)",
            ylabel="diag(M) (arithmetic mean in weight space)",
            path=fig_dir / f"{tag}__ambient_diagM_mean.png",
            logy=True,
            ylim=tuple(scales["ambient_ylim"]),
            identity=1.0,
            sd_available=sd_ok,
            note="Basis: original feature coordinates. U from frozen train PCA. " + PC_NOTE[:80],
        )
        wrote.append(str(fig_dir / f"{tag}__ambient_diagM_mean.png"))
    return wrote


def run_weight_plots(*, work: Path, repo: Path, out: Path | None = None) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    paths = Paths(work=work, repo=repo)
    repair = paths.results / REPAIR_NAME
    fit_dir = repair / "onesided_fits"
    out = out or (paths.results / OUT_NAME)
    out.mkdir(parents=True, exist_ok=True)

    unavailable: list[dict[str, Any]] = []
    design = json.loads((repo / "configs" / "release_anisotropy_repair.json").read_text())
    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    provenance = {}
    prov_path = repair / "pca_provenance.json"
    if prov_path.exists():
        provenance = json.loads(prov_path.read_text())
    else:
        unavailable.append({"item": "pca_provenance.json", "reason": "missing from repair results"})

    rows = catalog_fits(fit_dir)
    if not rows:
        unavailable.append({"item": str(fit_dir), "reason": "no repaired onesided_fits"})
        write_json(out / "unavailable.json", jsonable(unavailable))
        return {"out": str(out), "unavailable": unavailable, "n_plots": 0}

    expect = expected_partners(manifest)
    models = sorted({r["model"] for r in rows})
    rhos_found = sorted({r["rho"] for r in rows})
    design_rhos = [float(x) for x in design["budgets_rho"]]

    for model, plist in expect.items():
        if model not in models:
            unavailable.append({"model": model, "reason": "no repaired fits as reweighted side"})
            continue
        for partner in plist:
            for rho in design_rhos:
                hit = [
                    r
                    for r in rows
                    if r["model"] == model and r["partner"] == partner and abs(r["rho"] - rho) < 1e-12
                ]
                if not hit:
                    unavailable.append(
                        {"model": model, "partner": partner, "rho": rho, "reason": "fit file missing"}
                    )

    extra = sorted({r["model"] for r in rows if r["model"] not in expect})
    for m in extra:
        unavailable.append({"model": m, "reason": "fitted as reweighted side but not in partner_sets"})

    _, splits = load_splits(paths, repo)
    train_ids = splits["train"]
    pca_by_model: dict[str, dict[str, Any] | None] = {}
    q_design = int(design["q"])
    (out / "pca_bases").mkdir(parents=True, exist_ok=True)
    for model in models:
        row = next((r for block in ("base_panel", "supplementary_qwen3x") for r in manifest[block] if r["key"] == model), None)
        if row is None:
            unavailable.append({"model": model, "reason": "missing from release_models.json; ambient omitted"})
            pca_by_model[model] = None
            continue
        spec = spec_from_manifest(row)
        feat = final_feature_path(paths, spec, "coco_val2017", "train", train_ids)
        if not feat.exists():
            unavailable.append({"model": model, "reason": "frozen train features missing; ambient diag(M) omitted", "path": str(feat)})
            pca_by_model[model] = None
            continue
        rec = reconstruct_train_pca(paths, spec, train_ids, q_design)
        qs = {r["q"] for r in rows if r["model"] == model}
        if qs != {rec["q"]}:
            unavailable.append({"model": model, "reason": f"PCA q={rec['q']} vs fit q={qs}"})
        if model in provenance:
            old = provenance[model]["recomputed"]
            if abs(old - rec["variance_fraction"]) > 1e-8:
                raise RuntimeError(f"PCA provenance mismatch {model}: {old} vs {rec['variance_fraction']}")
        pca_by_model[model] = rec
        np.savez(
            out / "pca_bases" / f"{model}.npz",
            U=rec["U"],
            q=rec["q"],
            d=rec["d"],
            variance_fraction=rec["variance_fraction"],
            u_sha256=np.array(rec["u_sha256"]),
        )
        if len(qs) != 1:
            raise RuntimeError(f"non-identical fit q for {model}: {qs}")

    groups = ("vision", "language")
    scale_pool: dict[tuple[str, str], list[np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for model in models:
        for rho in rhos_found:
            for group in groups:
                names, diag, sstack, notes = collect_group(rows, model, rho, group)
                if notes:
                    unavailable.append({"model": model, "rho": rho, "group": group, "reason": "; ".join(notes)})
                if diag.size == 0:
                    unavailable.append({"model": model, "rho": rho, "group": group, "reason": "no partners"})
                    continue
                mean_w, sd_w, sd_ok = sample_mean_sd(diag, axis=0)
                mean_s, sd_s, sd_ok_s = sample_mean_sd(sstack, axis=0)
                assert sd_ok == sd_ok_s
                src = [
                    r["path"]
                    for r in rows
                    if r["model"] == model and abs(r["rho"] - rho) < 1e-12 and partner_group(r["partner"]) == group
                ]
                ambient = None
                pca = pca_by_model.get(model)
                if pca is not None:
                    u = pca["U"]
                    if u.shape[1] != diag.shape[1]:
                        unavailable.append({"model": model, "reason": "U columns != q; ambient omitted"})
                    else:
                        amb_stack = []
                        amb_map = {}
                        picked = [
                            r
                            for r in rows
                            if r["model"] == model
                            and abs(r["rho"] - rho) < 1e-12
                            and partner_group(r["partner"]) == group
                        ]
                        picked.sort(key=lambda r: names.index(r["partner"]))
                        for r in picked:
                            dvec = ambient_metric_diag(u, r["b"])
                            amb_stack.append(dvec)
                            amb_map[r["partner"]] = dvec
                        amb_arr = np.stack(amb_stack, axis=0)
                        am, asd, aok = sample_mean_sd(amb_arr, axis=0)
                        ambient = {"mean": am, "sd": asd, "sd_ok": aok, "partners": amb_map}
                rec = {
                    "model": model,
                    "rho": rho,
                    "group": group,
                    "partners": names,
                    "diag": diag,
                    "sstack": sstack,
                    "mean_w": mean_w,
                    "sd_w": sd_w,
                    "sd_ok": sd_ok,
                    "mean_s": mean_s,
                    "sd_s": sd_s,
                    "source_paths": src,
                    "meta": model_meta(manifest, model),
                    "pca": pca,
                    "ambient": ambient,
                }
                records.append(rec)
                scale_pool.setdefault((group, "w"), []).append(diag)
                scale_pool.setdefault((group, "log2"), []).append(np.log2(np.clip(diag, 1e-12, None)))
                scale_pool.setdefault((group, "s"), []).append(sstack)
                scale_pool.setdefault((group, "ssd"), []).append(sd_s if sd_ok else np.zeros_like(mean_s))
                if ambient is not None:
                    scale_pool.setdefault((group, "amb"), []).append(amb_arr)

    scales = {}
    for group in groups:
        wvals = scale_pool.get((group, "w"), [])
        scales[group] = {
            "weight_ylim": _weight_ylim(wvals) if wvals else (0.2, 5.0),
            "log2_abs": _finite_abs_max(np.concatenate([a.ravel() for a in scale_pool.get((group, "log2"), [])])) if scale_pool.get((group, "log2")) else 1.0,
            "s_abs": _finite_abs_max(np.concatenate([a.ravel() for a in scale_pool.get((group, "s"), [])])) if scale_pool.get((group, "s")) else 1.0,
            "s_sd_max": float(max((_finite_abs_max(a) for a in scale_pool.get((group, "ssd"), [])), default=1.0)),
            "ambient_ylim": _weight_ylim(scale_pool.get((group, "amb"), [])) if scale_pool.get((group, "amb")) else (0.2, 5.0),
        }

    n_plots = 0
    for rec in records:
        primary = abs(rec["rho"] - PRIMARY_RHO) < 1e-12
        fig_root = out / "figures" / ("primary" if primary else "supplementary") / rho_tag(rec["rho"])
        write_group_tables(
            out,
            rec["model"],
            rec["group"],
            rec["rho"],
            rec["partners"],
            rec["diag"],
            rec["sstack"],
            rec["mean_w"],
            rec["sd_w"],
            rec["sd_ok"],
            rec["mean_s"],
            rec["sd_s"],
            rec["meta"],
            rec["pca"],
            rec["source_paths"],
        )
        # pass actual sstack to tables — fix the ugly hack above
        n_plots += len(
            plot_group(
                fig_root,
                rec["model"],
                rec["group"],
                rec["rho"],
                rec["partners"],
                rec["diag"],
                rec["mean_w"],
                rec["sd_w"],
                rec["sd_ok"],
                rec["mean_s"],
                rec["sd_s"],
                scales[rec["group"]],
                rec["ambient"],
            )
        )

    write_readme(out, repair, models, rhos_found, scales, unavailable, design_rhos)
    summary = {
        "source_repair": str(repair),
        "n_fits_loaded": len(rows),
        "models": models,
        "rhos": rhos_found,
        "primary_rho": PRIMARY_RHO,
        "partner_groups": list(groups),
        "basis_ordering": "stored training PCA columns, descending training variance",
        "weight_definition": "diag(B); not sorted eig(B); not exp(diag(S))",
        "common_scales": jsonable(scales),
        "pc_identification_note": PC_NOTE,
        "n_plot_files": n_plots,
        "unavailable": unavailable,
        "did_not": ["fit", "extract", "select_rho", "overwrite_repair_or_parent"],
    }
    write_json(out / "run_manifest.json", jsonable(summary))
    write_json(out / "unavailable.json", jsonable(unavailable))
    return summary


def write_readme(out: Path, repair: Path, models: list[str], rhos: list[float], scales: dict, unavailable: list, design_rhos: list[float]) -> None:
    lines = [
        "# Learned anisotropic weights",
        "",
        "Post-processing of frozen repaired one-sided fits. No refitting.",
        "",
        f"- Source: `{repair}`",
        f"- Freeze tag: `prh-release-alignment-freeze-20260921` (do not move)",
        f"- Primary budget: fixed ρ={PRIMARY_RHO} (not validation-selected ρ)",
        f"- Supplementary budgets: {', '.join(f'{r:g}' for r in rhos if abs(r - PRIMARY_RHO) > 1e-12)}",
        f"- Models (reweighted side): {', '.join(models)}",
        f"- Design ρ list: {design_rhos}",
        "",
        "## Weights",
        "",
        "On each model's stored training-PCA axes, the plotted weights are `diag(B)`.",
        "These are not sorted eigenvalues of B, and not `exp(diag(S))`.",
        "Mean and sample SD are computed in weight space first.",
        "The SD is partner variation (ddof=1), not a confidence interval.",
        "If a group has fewer than two partners, SD is marked unavailable.",
        "",
        "## Basis",
        "",
        PC_NOTE,
        "",
        "Ambient plots (when frozen train features exist) use original hidden coordinates:",
        "`diag(M) = 1 + diag(U(B−I)Uᵀ)`, computed without forming the d×d matrix.",
        "",
        "## Layout",
        "",
        "- `figures/primary/rho0p1/` — primary fixed cap",
        "- `figures/supplementary/` — other saved fixed budgets",
        "- `tables/` — CSV/NPZ/JSON for each model × partner-group × ρ",
        "- `pca_bases/` — reconstructed U from frozen train features (not a new fit of B)",
        "",
        "Vision and language partners are never averaged together.",
        "",
        "## Unavailable",
        "",
    ]
    if not unavailable:
        lines.append("None.")
    else:
        for u in unavailable:
            lines.append(f"- `{json.dumps(u)}`")
    lines += ["", "## Shared scales", "", "```json", json.dumps(jsonable(scales), indent=2), "```", ""]
    (out / "README.md").write_text("\n".join(lines))
