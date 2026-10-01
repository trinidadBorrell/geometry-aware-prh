"""Feature-selection frequency and ΔM heatmaps in original coordinates.

Post-process frozen repaired one-sided fits only. Does not fit or extract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prh_replication.extract_final import final_feature_path, spec_from_manifest
from prh_replication.io_utils import jsonable, write_json
from prh_replication.plots import _save, raster_matrix
from prh_replication.registry import Paths
from prh_replication.release_anisotropy import ambient_metric_diag, signed_correction
from prh_replication.release_protocol import VIS
from prh_replication.weight_plots import (
    PRIMARY_RHO,
    REPAIR_NAME,
    catalog_fits,
    expected_partners,
    model_meta,
    partner_group,
    reconstruct_train_pca,
    sample_mean_sd,
)

OUT_NAME = "feature_selection_frequency"
WEIGHT_ATOL = 1e-10
WEIGHT_RTOL = 1e-8
FRAC_K = 0.2
CROP_N = 64
PCTL = 99.5
COORD_NOTE = (
    "Original hidden-feature coordinates of this model only. "
    "Index j is not identified across models or releases. "
    "Top-20% membership is coordinate-axis weighting, not a semantic concept. "
    "Off-diagonal ΔM entries are cross-coordinate metric terms, not correlations."
)


def weight_tolerance(w: np.ndarray) -> float:
    """Abs+rel floor for float64 reconstruction / JSON B round-trip."""
    scale = float(np.max(np.abs(w))) if w.size else 1.0
    return WEIGHT_ATOL + WEIGHT_RTOL * max(1.0, scale)


def selection_k(d: int) -> int:
    return int(math.ceil(FRAC_K * d))


def tie_seed(model: str, partner: str, rho: float) -> int:
    blob = f"{model}\0{partner}\0{rho:.12g}\0top20-original-features".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") % (2**32)


def select_top_k(w: np.ndarray, k: int, *, rng: np.random.Generator) -> dict[str, Any]:
    """Largest-w selection with explicit flat/tie handling.

    Fully flat diagonals are uninformative: no features are manufactured.
    Boundary ties use ``rng``, not feature index.
    """
    w = np.asarray(w, dtype=np.float64)
    d = int(w.size)
    k = int(min(max(k, 0), d))
    tol = weight_tolerance(w)
    selected = np.zeros(d, dtype=bool)
    rank = np.full(d, -1, dtype=np.int32)
    span = float(w.max() - w.min()) if d else 0.0
    if d == 0 or k == 0:
        return {
            "selected": selected,
            "rank": rank,
            "uninformative": True,
            "tie_break": False,
            "n_tie_pool": 0,
            "tol": tol,
            "span": span,
        }
    if span <= tol:
        return {
            "selected": selected,
            "rank": rank,
            "uninformative": True,
            "tie_break": False,
            "n_tie_pool": int(d),
            "tol": tol,
            "span": span,
        }
    order = np.argsort(-w, kind="mergesort")
    tau = float(w[order[k - 1]])
    above = w > tau + tol
    on = np.abs(w - tau) <= tol
    n_above = int(above.sum())
    n_need = k - n_above
    tie_break = False
    n_pool = int(on.sum())
    chosen = np.where(above)[0]
    if n_need > 0:
        pool = np.where(on)[0]
        if n_need < pool.size:
            pick = rng.choice(pool, size=n_need, replace=False)
            tie_break = True
            n_pool = int(pool.size)
            chosen = np.concatenate([chosen, pick])
        else:
            chosen = np.concatenate([chosen, pool])
            tie_break = n_need < pool.size or pool.size > n_need
    selected[chosen] = True
    # ranks among selected by decreasing w, ties among selected use rng order already
    sel_idx = np.where(selected)[0]
    sel_order = sel_idx[np.argsort(-w[sel_idx], kind="mergesort")]
    for r, j in enumerate(sel_order, start=1):
        rank[j] = r
    return {
        "selected": selected,
        "rank": rank,
        "uninformative": False,
        "tie_break": bool(tie_break),
        "n_tie_pool": n_pool if tie_break else 0,
        "tol": tol,
        "span": span,
        "tau": tau,
        "n_selected": int(selected.sum()),
    }


def permute_freq_then_index(frequency: np.ndarray) -> np.ndarray:
    """One permutation for all rows: decreasing frequency, then original index."""
    d = frequency.size
    idx = np.arange(d)
    # lex: -freq, +index
    return np.lexsort((idx, -frequency))


def crop_ids(frequency: np.ndarray, n: int = CROP_N) -> np.ndarray:
    perm = permute_freq_then_index(frequency)
    return perm[: min(n, perm.size)]


def _delta_from_b(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    return signed_correction(u, np.asarray(b, dtype=np.float64))


def _mean_delta(u: np.ndarray, bs: list[np.ndarray]) -> np.ndarray:
    q = u.shape[1]
    eye = np.eye(q, dtype=np.float64)
    c = np.mean([0.5 * (b + b.T) - eye for b in bs], axis=0)
    return u @ c @ u.T


def _entrywise_sd(u: np.ndarray, bs: list[np.ndarray], mean_delta: np.ndarray) -> tuple[np.ndarray, bool]:
    n = len(bs)
    d = u.shape[0]
    if n < 2:
        return np.full((d, d), np.nan, dtype=np.float64), False
    m2 = np.zeros((d, d), dtype=np.float64)
    for b in bs:
        delta = _delta_from_b(u, b)
        diff = delta - mean_delta
        m2 += diff * diff
        del delta, diff
    return np.sqrt(m2 / (n - 1)), True


def _p995_abs(mat: np.ndarray) -> float:
    a = np.abs(mat[np.isfinite(mat)])
    if a.size == 0:
        return 1.0
    return float(np.quantile(a, PCTL / 100.0))


def _caption(*, model: str, group: str, n: int, d: int, k: int, rho: float, extra: str = "") -> str:
    bits = [
        f"One-sided fit (reweighted {model}; partners at identity).",
        f"Fixed ρ={rho:g}. Original hidden coordinates. Group={group}, n={n}, d={d}, k=⌈0.2d⌉={k}.",
        COORD_NOTE,
    ]
    if extra:
        bits.append(extra)
    return " ".join(bits)


def plot_count_hist(
    counts: np.ndarray,
    *,
    n_partners: int,
    d: int,
    k: int,
    model: str,
    group: str,
    rho: float,
    path: Path,
    n_uninform: int,
    n_tie_fits: int,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.arange(0, n_partners + 1)
    frac = np.array([(counts == c).mean() if counts.size else 0.0 for c in xs], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(xs, frac, width=0.85, color="C0", align="center")
    ax.set_xticks(xs)
    ax.set_xlim(-0.6, n_partners + 0.6)
    ax.set_xlabel("selected in this many partners")
    ax.set_ylabel("fraction of original features")
    ax.set_title(f"{model}  {group}  ρ={rho:g}\nfeature-selection count distribution")
    for c, f in zip(xs, frac):
        if f > 0:
            ax.text(c, f, f"{c}/{n_partners}", ha="center", va="bottom", fontsize=7)
    extra = (
        f"Each informative partner selects k={k} features (~20%). "
        f"Mean frequency ≈ 0.2 by construction. "
        f"Mass at 0 and {n_partners} is recurring selection, not independence. "
        f"Uninformative fits excluded: {n_uninform}. Tie-broken fits: {n_tie_fits}."
    )
    fig.text(0.01, 0.01, _caption(model=model, group=group, n=n_partners, d=d, k=k, rho=rho, extra=extra), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, path)


def plot_overview_hists(
    items: list[dict[str, Any]],
    *,
    group: str,
    rho: float,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n = len(items)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    for i, rec in enumerate(items):
        ax = axes[i // cols][i % cols]
        n_p = rec["n_valid"]
        xs = np.arange(0, n_p + 1)
        frac = rec["count_frac"]
        ax.bar(xs, frac, width=0.85, color="C0")
        ax.set_xticks(xs)
        ax.set_title(rec["model"], fontsize=9)
        ax.set_ylim(0, max(0.05, float(np.max(frac)) * 1.15) if frac.size else 1)
        if i // cols == rows - 1:
            ax.set_xlabel("count")
        if i % cols == 0:
            ax.set_ylabel("feature fraction")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(
        f"Selection-count histograms  {group} partners  ρ={rho:g}\n"
        "Discrete counts; no KDE. Mean frequency ≈ 0.2 by construction.",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_freq_vs_index(frequency: np.ndarray, *, title: str, path: Path, caption: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(np.arange(frequency.size), frequency, lw=0.6, color="C0")
    ax.set_xlabel("original feature ID")
    ax.set_ylabel("empirical frequency (count / n_valid)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    fig.text(0.01, 0.01, caption, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path)


def plot_binary_selection(
    selected: np.ndarray,
    partners: list[str],
    col_order: np.ndarray,
    *,
    title: str,
    path: Path,
    caption: str,
    xlabel: str,
) -> None:
    mat = selected[:, col_order].astype(np.float64)
    fig_w = 10.0
    fig_h = max(2.8, 0.35 * len(partners) + 1.8)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(
        mat,
        cmap="Greys",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_yticks(range(len(partners)), partners, fontsize=8)
    d = mat.shape[1]
    ax.set_xticks([0, d // 2, d - 1], [str(col_order[0]), str(col_order[d // 2]), str(col_order[d - 1])])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    fig.text(0.01, 0.01, caption, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, path)


def _group_partners(rows: list[dict], model: str, rho: float, group: str) -> list[dict]:
    picked = [r for r in rows if r["model"] == model and abs(r["rho"] - rho) < 1e-12]
    if group == "vision":
        picked = [r for r in picked if partner_group(r["partner"]) == "vision"]
        picked.sort(key=lambda r: VIS.index(r["partner"]) if r["partner"] in VIS else 99)
    elif group == "language":
        picked = [r for r in picked if partner_group(r["partner"]) == "language"]
        picked.sort(key=lambda r: r["partner"])
    else:
        picked.sort(key=lambda r: (0, VIS.index(r["partner"])) if r["partner"] in VIS else (1, r["partner"]))
    return picked


def run_feature_selection(*, work: Path, repo: Path, out: Path | None = None) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    paths = Paths(work=work, repo=repo)
    repair = paths.results / REPAIR_NAME
    out = out or (paths.results / OUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    rho = PRIMARY_RHO
    design = json.loads((repo / "configs" / "release_anisotropy_repair.json").read_text())
    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    provenance = json.loads((repair / "pca_provenance.json").read_text()) if (repair / "pca_provenance.json").exists() else {}
    from prh_replication.release_protocol import load_splits

    _, splits = load_splits(paths, repo)
    rows = [r for r in catalog_fits(repair / "onesided_fits") if abs(r["rho"] - rho) < 1e-12]
    q_design = int(design["q"])
    unavailable: list[dict[str, Any]] = []
    models = sorted({r["model"] for r in rows})
    pca_by: dict[str, dict[str, Any] | None] = {}
    for model in models:
        row = next((r for block in ("base_panel", "supplementary_qwen3x") for r in manifest[block] if r["key"] == model), None)
        if row is None:
            pca_by[model] = None
            unavailable.append({"model": model, "reason": "not in release_models.json"})
            continue
        spec = spec_from_manifest(row)
        feat = final_feature_path(paths, spec, "coco_val2017", "train", splits["train"])
        if not feat.exists():
            pca_by[model] = None
            unavailable.append({"model": model, "reason": "missing frozen train features", "path": str(feat)})
            continue
        rec = reconstruct_train_pca(paths, spec, splits["train"], q_design)
        if model in provenance:
            old = provenance[model]["recomputed"]
            if abs(old - rec["variance_fraction"]) > 1e-8:
                raise RuntimeError(f"PCA provenance mismatch {model}")
        qs = {r["q"] for r in rows if r["model"] == model}
        if qs != {rec["q"]}:
            unavailable.append({"model": model, "reason": f"q mismatch fits {qs} vs PCA {rec['q']}"})
        if rec["d"] != rec["U"].shape[0]:
            raise RuntimeError(model)
        pca_by[model] = rec

    groups_primary = ("vision", "language")
    groups_all = ("vision", "language", "all")
    # Pass 1: selection + colour limits
    packed: list[dict[str, Any]] = []
    p995: dict[str, list[float]] = {g: [] for g in groups_all}
    p995_sd: dict[str, list[float]] = {g: [] for g in groups_all}

    for model in models:
        pca = pca_by.get(model)
        if pca is None:
            continue
        u = pca["U"]
        d = int(u.shape[0])
        meta = model_meta(manifest, model)
        hid = meta.get("hidden_size")
        if hid is not None and int(hid) != d:
            unavailable.append({"model": model, "reason": f"hidden_size {hid} vs PCA d {d} (using PCA d)"})
        for group in groups_all:
            fits = _group_partners(rows, model, rho, group)
            if not fits:
                unavailable.append({"model": model, "group": group, "reason": "no rho=0.1 fits"})
                continue
            if len({r["q"] for r in fits}) != 1 or fits[0]["q"] != u.shape[1]:
                unavailable.append({"model": model, "group": group, "reason": "PCA/B q mismatch"})
                continue
            partners = [r["partner"] for r in fits]
            bs = [r["b"] for r in fits]
            src = [r["path"] for r in fits]
            k = selection_k(d)
            w_rows = []
            sel_rows = []
            valid_idx = []
            tie_flags = []
            uninform = []
            per_partner = []
            for r in fits:
                w = ambient_metric_diag(u, r["b"])
                rng = np.random.default_rng(tie_seed(model, r["partner"], rho))
                sel = select_top_k(w, k, rng=rng)
                recp = {
                    "partner": r["partner"],
                    "path": r["path"],
                    "uninformative": sel["uninformative"],
                    "tie_break": sel["tie_break"],
                    "n_tie_pool": sel["n_tie_pool"],
                    "span": sel["span"],
                    "tol": sel["tol"],
                    "n_selected": int(sel["selected"].sum()),
                    "d_attained": r.get("d_attained"),
                }
                per_partner.append(recp)
                w_rows.append(w)
                sel_rows.append(sel["selected"])
                tie_flags.append(sel["tie_break"])
                if sel["uninformative"]:
                    uninform.append(r["partner"])
                else:
                    valid_idx.append(len(w_rows) - 1)
            w_mat = np.stack(w_rows, axis=0)
            sel_mat = np.stack(sel_rows, axis=0)
            if valid_idx:
                sel_valid = sel_mat[np.array(valid_idx)]
                n_valid = int(sel_valid.shape[0])
                count = sel_valid.sum(axis=0).astype(np.int32)
                freq = count / n_valid
                mean_w, sd_w, sd_ok = sample_mean_sd(w_mat[np.array(valid_idx)], axis=0)
            else:
                n_valid = 0
                count = np.zeros(d, dtype=np.int32)
                freq = np.zeros(d, dtype=np.float64)
                mean_w, sd_w, sd_ok = sample_mean_sd(w_mat[:1], axis=0) if w_mat.shape[0] else (np.zeros(d), np.zeros(d), False)
                sd_ok = False
            bs_valid = [bs[i] for i in valid_idx] if valid_idx else []
            if bs_valid:
                mean_d = _mean_delta(u, bs_valid)
                sd_d, sd_ok_m = _entrywise_sd(u, bs_valid, mean_d)
                p995[group].append(_p995_abs(mean_d))
                for b in bs_valid:
                    p995[group].append(_p995_abs(_delta_from_b(u, b)))
                if sd_ok_m:
                    p995_sd[group].append(_p995_abs(sd_d))
                del mean_d, sd_d
            packed.append(
                {
                    "model": model,
                    "group": group,
                    "meta": meta,
                    "pca_meta": {k: pca[k] for k in pca if k != "U"},
                    "d": d,
                    "k": k,
                    "partners": partners,
                    "valid_idx": valid_idx,
                    "src": src,
                    "count": count,
                    "freq": freq,
                    "mean_w": mean_w,
                    "sd_w": sd_w,
                    "sd_ok_w": sd_ok,
                    "n_valid": n_valid,
                    "uninform": uninform,
                    "n_tie_fits": int(sum(tie_flags)),
                    "per_partner": per_partner,
                    "layer": meta.get("layer"),
                    "w_mat": w_mat,
                    "sel_mat": sel_mat,
                }
            )

    vmax = {g: (max(v) if v else 1.0) for g, v in p995.items()}
    vmax_sd = {g: (max(v) if v else 1.0) for g, v in p995_sd.items()}
    clip_log: list[dict[str, Any]] = []
    overview: dict[str, list] = {g: [] for g in groups_all}
    concentration: list[dict[str, Any]] = []

    fig_root = out / "figures"
    tbl_root = out / "tables"
    tbl_root.mkdir(parents=True, exist_ok=True)

    for rec in packed:
        model, group = rec["model"], rec["group"]
        primary = group in groups_primary
        dest = fig_root / ("primary" if primary else "supplementary") / group
        dest.mkdir(parents=True, exist_ok=True)
        d, k, n_valid = rec["d"], rec["k"], rec["n_valid"]
        cap = _caption(
            model=model,
            group=group,
            n=n_valid,
            d=d,
            k=k,
            rho=rho,
            extra=f"Uninformative excluded={len(rec['uninform'])}. Tie-broken fits={rec['n_tie_fits']}.",
        )
        if n_valid:
            plot_count_hist(
                rec["count"],
                n_partners=n_valid,
                d=d,
                k=k,
                model=model,
                group=group,
                rho=rho,
                path=dest / f"{model}__count_hist.png",
                n_uninform=len(rec["uninform"]),
                n_tie_fits=rec["n_tie_fits"],
            )
            xs = np.arange(0, n_valid + 1)
            frac = np.array([(rec["count"] == c).mean() for c in xs])
            overview[group].append({"model": model, "n_valid": n_valid, "count_frac": frac})
            mass_ext = float(((rec["count"] == 0) | (rec["count"] == n_valid)).mean())
            concentration.append(
                {
                    "model": model,
                    "group": group,
                    "n_valid": n_valid,
                    "mean_frequency": float(rec["freq"].mean()),
                    "frac_never_selected": float((rec["count"] == 0).mean()),
                    "frac_always_selected": float((rec["count"] == n_valid).mean()),
                    "frac_at_0_or_n": mass_ext,
                    "n_tie_fits": rec["n_tie_fits"],
                    "n_uninformative": len(rec["uninform"]),
                }
            )
        plot_freq_vs_index(
            rec["freq"],
            title=f"{model}  {group}  ρ={rho:g}  selection frequency vs original index",
            path=dest / f"{model}__freq_vs_index.png",
            caption=cap,
        )
        if n_valid:
            sel_valid = rec["sel_mat"][np.array(rec["valid_idx"])]
            p_valid = [rec["partners"][i] for i in rec["valid_idx"]]
            idx_order = np.arange(d)
            freq_order = permute_freq_then_index(rec["freq"])
            plot_binary_selection(
                sel_valid,
                p_valid,
                idx_order,
                title=f"{model}  {group}  top-20% membership  original index order",
                path=dest / f"{model}__select_heatmap_index.png",
                caption=cap + " Same column order on every row.",
                xlabel="original feature ID (native order)",
            )
            plot_binary_selection(
                sel_valid,
                p_valid,
                freq_order,
                title=f"{model}  {group}  top-20% membership  sorted by group frequency",
                path=dest / f"{model}__select_heatmap_freq.png",
                caption=cap + " One frequency sort for all partners; ties broken by original index.",
                xlabel="features sorted by decreasing selection frequency",
            )
            np.savez(
                tbl_root / f"{model}__{group}__column_order.npz",
                original_index=idx_order,
                freq_then_index=freq_order,
                crop_ids=crop_ids(rec["freq"]),
                frequency=rec["freq"],
                count=rec["count"],
            )
            u = pca_by[model]["U"]
            fits = _group_partners(rows, model, rho, group)
            bs_valid = [fits[i]["b"] for i in rec["valid_idx"]]
            mean_d = _mean_delta(u, bs_valid)
            sd_d, sd_ok_m = _entrywise_sd(u, bs_valid, mean_d)
            vm = vmax[group]
            vs = vmax_sd[group] if vmax_sd[group] > 0 else 1.0
            info = raster_matrix(
                mean_d,
                title=f"{model}  {group}  mean ΔM=M−I  original coordinates",
                path=dest / f"{model}__deltaM_mean.png",
                cbar="mean ΔM (partner arithmetic mean)",
                cmap="RdBu_r",
                vmin=-vm,
                vmax=vm,
                caption=cap + f" Shared |ΔM| clip at {PCTL}th pct rule, vmax={vm:.4g}.",
            )
            clip_log.append({"model": model, "group": group, "panel": "mean", **info})
            if sd_ok_m:
                info = raster_matrix(
                    sd_d,
                    title=f"{model}  {group}  entrywise sample SD of ΔM",
                    path=dest / f"{model}__deltaM_sd.png",
                    cbar="sample SD across partners (not a CI)",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=vs,
                    caption=cap,
                )
                clip_log.append({"model": model, "group": group, "panel": "sd", **info})
            else:
                unavailable.append({"model": model, "group": group, "reason": "ΔM SD unavailable (n_valid<2)"})
            pdir = dest / "partners"
            pdir.mkdir(exist_ok=True)
            for i, b in zip(rec["valid_idx"], bs_valid):
                delta = _delta_from_b(u, b)
                pname = rec["partners"][i]
                info = raster_matrix(
                    delta,
                    title=f"{model} vs {pname}  ΔM=M−I  ρ={rho:g}",
                    path=pdir / f"{model}__{pname}__deltaM.png",
                    cbar="ΔM (correction; 0 = native metric)",
                    cmap="RdBu_r",
                    vmin=-vm,
                    vmax=vm,
                    caption=cap,
                )
                clip_log.append({"model": model, "group": group, "partner": pname, "panel": "individual", **info})
                del delta
            crop = crop_ids(rec["freq"])
            labels = [str(int(j)) for j in crop]
            mean_c = mean_d[np.ix_(crop, crop)]
            raster_matrix(
                mean_c,
                title=f"{model}  {group}  mean ΔM  64-feature frequency crop",
                path=dest / f"{model}__deltaM_detail_mean.png",
                cbar="mean ΔM",
                cmap="RdBu_r",
                vmin=-vm,
                vmax=vm,
                xticks=labels,
                yticks=labels,
                xlabel="original feature ID (crop)",
                ylabel="original feature ID (crop)",
                caption=cap + " Descriptive crop of highest-frequency coordinates, not the full metric.",
                square=True,
            )
            if sd_ok_m:
                raster_matrix(
                    sd_d[np.ix_(crop, crop)],
                    title=f"{model}  {group}  SD ΔM  64-feature frequency crop",
                    path=dest / f"{model}__deltaM_detail_sd.png",
                    cbar="sample SD of ΔM",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=vs,
                    xticks=labels,
                    yticks=labels,
                    xlabel="original feature ID (crop)",
                    ylabel="original feature ID (crop)",
                    caption=cap,
                    square=True,
                )
            ddir = dest / "partners_detail"
            ddir.mkdir(exist_ok=True)
            for i, b in zip(rec["valid_idx"], bs_valid):
                delta = _delta_from_b(u, b)
                raster_matrix(
                    delta[np.ix_(crop, crop)],
                    title=f"{model} vs {rec['partners'][i]}  ΔM crop  ρ={rho:g}",
                    path=ddir / f"{model}__{rec['partners'][i]}__deltaM_detail.png",
                    cbar="ΔM",
                    cmap="RdBu_r",
                    vmin=-vm,
                    vmax=vm,
                    xticks=labels,
                    yticks=labels,
                    xlabel="original feature ID (crop)",
                    ylabel="original feature ID (crop)",
                    caption=cap + " Same 64 IDs/order as mean/SD detail.",
                    square=True,
                )
                del delta
        # tables
        header = ["feature_id", "count", "frequency", "mean_w", "sample_sd_w"] + [
            f"sel_{rec['partners'][i]}" for i in rec["valid_idx"]
        ]
        lines = [",".join(header)]
        for j in range(d):
            row = [
                str(j),
                str(int(rec["count"][j])),
                f"{rec['freq'][j]:.12g}",
                f"{rec['mean_w'][j]:.12g}",
                "nan" if not rec["sd_ok_w"] else f"{rec['sd_w'][j]:.12g}",
            ]
            for i in rec["valid_idx"]:
                row.append("1" if rec["sel_mat"][i, j] else "0")
            lines.append(",".join(row))
        (tbl_root / f"{model}__{group}__features.csv").write_text("\n".join(lines) + "\n")
        np.savez(
            tbl_root / f"{model}__{group}__weights.npz",
            w=rec["w_mat"],
            selected=rec["sel_mat"],
            partners=np.array(rec["partners"]),
            valid_idx=np.array(rec["valid_idx"]),
            count=rec["count"],
            frequency=rec["freq"],
            mean_w=rec["mean_w"],
            sample_sd_w=rec["sd_w"],
        )
        write_json(
            tbl_root / f"{model}__{group}__partners.json",
            jsonable(
                {
                    "model": model,
                    "layer": rec["layer"],
                    "group": group,
                    "rho": rho,
                    "d": d,
                    "q": int(pca_by[model]["U"].shape[1]),
                    "k": k,
                    "pca": rec["pca_meta"],
                    "source_fits": rec["src"],
                    "per_partner": rec["per_partner"],
                    "n_valid": n_valid,
                    "uninformative_partners": rec["uninform"],
                    "weight_atol": WEIGHT_ATOL,
                    "weight_rtol": WEIGHT_RTOL,
                    "colour_vmax_deltaM": vmax[group],
                    "colour_vmax_sd": vmax_sd[group],
                    "colour_rule": f"shared {PCTL}th percentile of |ΔM| within partner group (mean and individuals)",
                }
            ),
        )
        rec.pop("w_mat", None)
        rec.pop("sel_mat", None)

    for group in groups_all:
        if not overview[group]:
            continue
        dest = fig_root / ("primary" if group in groups_primary else "supplementary") / group
        plot_overview_hists(overview[group], group=group, rho=rho, path=dest / "overview_count_hist.png")

    write_readme(out, repair, rho, vmax, vmax_sd, concentration, unavailable, clip_log)
    summary = {
        "source_repair": str(repair),
        "rho": rho,
        "n_fits_rho": len(rows),
        "models": models,
        "vmax": vmax,
        "vmax_sd": vmax_sd,
        "concentration": concentration,
        "unavailable": unavailable,
        "clip": clip_log,
        "did_not": ["fit", "extract", "select_rho", "overwrite_repair"],
        "definitions": {
            "w_j": "M_jj = 1 + [U(B-I)U^T]_jj",
            "DeltaM": "M-I = U(B-I)U^T",
            "frequency": "count / n_valid partners; empirical, not a probability",
        },
        "coord_note": COORD_NOTE,
    }
    write_json(out / "run_manifest.json", jsonable(summary))
    write_json(out / "unavailable.json", jsonable(unavailable))
    write_json(out / "concentration.json", jsonable(concentration))
    return summary


def write_readme(out, repair, rho, vmax, vmax_sd, concentration, unavailable, clip_log) -> None:
    vis = [c for c in concentration if c["group"] == "vision"]
    lang = [c for c in concentration if c["group"] == "language"]

    def _avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    lines = [
        "# Feature selection frequency and ΔM",
        "",
        "Plotting-only analysis of frozen repaired one-sided fits. No refitting.",
        "",
        f"- Source: `{repair}`",
        "- Freeze tag: `prh-release-alignment-freeze-20260921` (do not move)",
        f"- Budget: fixed ρ={rho} (not validation-selected)",
        "- Coordinates: original hidden features. Weights w_j = M_jj.",
        "- B is q×q in PCA; M and ΔM=M−I are d×d in original coordinates.",
        "- Frequency = selection count / n_valid partners (empirical).",
        "",
        "## Reproduction",
        "",
        "```bash",
        "export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src",
        "/mnt/sdb1/prh-replication-work/venv/bin/python scripts/plot_feature_selection_frequency.py \\",
        "  --work /mnt/sdb1/prh-replication-work",
        "```",
        "",
        "## Colour limits",
        "",
        f"Diverging ΔM: shared ±vmax per partner group from the {PCTL}th percentile of |ΔM|",
        "(mean and individual matrices). Sequential SD: 0–vmax_sd from the same percentile of SD entries.",
        f"vmax={json.dumps(jsonable(vmax))} vmax_sd={json.dumps(jsonable(vmax_sd))}",
        "",
        "## Visual summary (descriptive, not new freeze claims)",
        "",
        f"Vision: mean freq={_avg(vis,'mean_frequency'):.3f} (construction ≈0.2); "
        f"mean mass at 0 or n={_avg(vis,'frac_at_0_or_n'):.3f}; "
        f"always-selected={_avg(vis,'frac_always_selected'):.3f}; never={_avg(vis,'frac_never_selected'):.3f}.",
        f"Language: mean freq={_avg(lang,'mean_frequency'):.3f}; "
        f"mass at 0 or n={_avg(lang,'frac_at_0_or_n'):.3f}; "
        f"always={_avg(lang,'frac_always_selected'):.3f}; never={_avg(lang,'frac_never_selected'):.3f}.",
        "Whether that mass is concentrated vs diffuse is in `concentration.json` and the histograms.",
        "Vision vs language patterns should be compared only within a model; feature IDs do not match across models.",
        "Off-diagonal structure is in the ΔM mean/SD figures; it is not feature importance.",
        "",
        "## Unavailable / ties",
        "",
    ]
    if not unavailable:
        lines.append("No missing models/fits.")
    else:
        for u in unavailable:
            lines.append(f"- `{json.dumps(u)}`")
    lines += ["", "See `tables/` for partners.json (per-fit uninformative/tie flags).", ""]
    (out / "README.md").write_text("\n".join(lines))
