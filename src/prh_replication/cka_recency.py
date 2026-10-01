"""Native CKA versus approximate version ordinality of each pair.

Ranks are 1-based within a documented family, ordered by public_release_date
then model key. They are not calendar time and not Hub lastModified.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from prh_replication.io_utils import jsonable, write_json
from prh_replication.plots import _save, heatmap
from prh_replication.registry import Paths
from prh_replication.release_protocol import VIS
from prh_replication.weight_plots import PRIMARY_RHO, REPAIR_NAME, model_meta

OUT_NAME = "cka_vs_recency"
NOTE = (
    "Version ordinal is 1-based order within a family (public_release_date, then key). "
    "For a language–language pair, pair ordinal = mean of the two ranks. "
    "For a language–vision pair, pair ordinal = the language rank (vision is an unranked anchor). "
    "Ordinals are approximate (size/stage differences; month-level dates). "
    "Recency is observational. Pairwise cells are dependent. Final-layer COCO gallery only."
)


def parse_release_date(text: str | None):
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def native_lookup(rows: list[dict], a: str, b: str) -> dict | None:
    for r in rows:
        if r["a"] == a and r["b"] == b:
            return r
        if r["a"] == b and r["b"] == a:
            return r
    return None


def family_series(manifest: dict) -> dict[str, list[str]]:
    qwen = [r["key"] for r in manifest["base_panel"] if r["family"] == "Qwen"]
    olmo = [r["key"] for r in manifest["base_panel"] if r["family"] == "OLMo"]
    supp = [r["key"] for r in manifest["supplementary_qwen3x"]]
    return {"Qwen-base": qwen, "OLMo-base": olmo, "Qwen3x-supplementary": supp}


def rank_family(manifest: dict, keys: list[str]) -> dict[str, int]:
    """Approximate version order: older → 1."""
    decorated = []
    for k in keys:
        row = next(
            r
            for block in ("base_panel", "supplementary_qwen3x")
            for r in manifest[block]
            if r["key"] == k
        )
        dt = parse_release_date(row.get("public_release_date"))
        decorated.append((dt is None, dt or datetime.max.date(), k))
    decorated.sort()
    return {k: i + 1 for i, (*_, k) in enumerate(decorated)}


def pair_ordinal(rank_a: int | None, rank_b: int | None) -> float | None:
    """Mean rank when both are ranked; otherwise the ranked member (VL)."""
    if rank_a is not None and rank_b is not None:
        return 0.5 * (rank_a + rank_b)
    if rank_a is not None:
        return float(rank_a)
    if rank_b is not None:
        return float(rank_b)
    return None


def _label(manifest: dict, key: str) -> str:
    meta = model_meta(manifest, key)
    return meta.get("key") or key


def _ylim(vals: list[float]) -> tuple[float, float]:
    xs = [v for v in vals if np.isfinite(v)]
    if not xs:
        return (0.0, 1.0)
    lo, hi = float(min(xs)), float(max(xs))
    pad = 0.06 * (hi - lo if hi > lo else 0.1)
    return (lo - pad, hi + pad)


def plot_vl_ordinal(
    *,
    ranks: list[int],
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    path: Path,
    ylim: tuple[float, float],
    caption: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    x = np.asarray(ranks, dtype=np.float64)
    stack = [np.asarray(series[p], dtype=np.float64) for p in VIS if p in series]
    mean = np.nanmean(np.stack(stack, axis=0), axis=0) if stack else x * np.nan
    for p in VIS:
        if p not in series:
            continue
        ax.plot(x, series[p], marker="o", lw=1.4, label=p)
    ax.plot(x, mean, marker="s", lw=2.2, color="k", label="mean of 3 vision partners")
    ax.set_xticks(ranks, [f"{r}\n{lab}" for r, lab in zip(ranks, labels)], fontsize=8)
    ax.set_xlim(min(ranks) - 0.3, max(ranks) + 0.3)
    ax.set_ylim(*ylim)
    ax.set_xlabel("language version ordinal (1 = earliest in family)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, caption, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, path)


def plot_pair_scatter(
    *,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    ylim: tuple[float, float],
    caption: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.scatter(xs, ys, s=42, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    fig.text(0.01, 0.01, caption, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, path)


def run_cka_recency(*, work: Path, repo: Path, out: Path | None = None) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    paths = Paths(work=work, repo=repo)
    repair = paths.results / REPAIR_NAME
    out = out or (paths.results / OUT_NAME)
    figdir = out / "figures"
    tbldir = out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tbldir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((repo / "data" / "manifests" / "release_models.json").read_text())
    native = json.loads((repair / "native_matrix.json").read_text())
    pair_eval = json.loads((repair / "pair_eval.json").read_text())
    series = family_series(manifest)
    ranks: dict[str, dict[str, int]] = {name: rank_family(manifest, keys) for name, keys in series.items()}

    unavailable: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    native_vl: list[float] = []
    os_vl: list[float] = []

    cap = "Native linear CKA a on COCO test. " + NOTE

    vl_plots: list[dict[str, Any]] = []
    for fname, keys in series.items():
        mapping = ranks[fname]
        ordered = sorted(keys, key=lambda k: mapping[k])
        labels = [_label(manifest, k) for k in ordered]
        rnk = [mapping[k] for k in ordered]
        ser_n = {p: [] for p in VIS}
        ser_os = {p: [] for p in VIS}
        for k in ordered:
            for p in VIS:
                rec = native_lookup(native, k, p)
                if rec is None:
                    unavailable.append({"model": k, "partner": p, "metric": "native"})
                    ser_n[p].append(float("nan"))
                    yn = float("nan")
                else:
                    yn = float(rec["cka_a"])
                    ser_n[p].append(yn)
                    native_vl.append(yn)
                hits = [
                    r
                    for r in pair_eval
                    if r["a"] == k and r["b"] == p and abs(float(r["rho"]) - PRIMARY_RHO) < 1e-12
                ]
                if not hits:
                    unavailable.append({"model": k, "partner": p, "metric": "onesided_rho0.1"})
                    ser_os[p].append(float("nan"))
                    yo = float("nan")
                else:
                    yo = float(hits[0]["test"]["a"])
                    ser_os[p].append(yo)
                    os_vl.append(yo)
                table_rows.append(
                    {
                        "kind": "language_vision",
                        "family": fname,
                        "a": k,
                        "b": p,
                        "rank_a": mapping[k],
                        "rank_b": None,
                        "pair_ordinal": pair_ordinal(mapping[k], None),
                        "native_cka_a": yn,
                        "onesided_rho0p1_test_a": yo,
                    }
                )
        vl_plots.append(
            {"name": fname, "ranks": rnk, "labels": labels, "ser_n": ser_n, "ser_os": ser_os}
        )

    ylim_n = _ylim(native_vl)
    ylim_os = _ylim(os_vl)
    for fam in vl_plots:
        slug = fam["name"].lower().replace(" ", "_")
        plot_vl_ordinal(
            ranks=fam["ranks"],
            labels=fam["labels"],
            series=fam["ser_n"],
            title=f"{fam['name']}  native CKA vs language version ordinal",
            ylabel="native CKA a (test)",
            path=figdir / f"{slug}__vl_native_cka_vs_ordinal.png",
            ylim=ylim_n,
            caption="Vision partners are unranked anchors. Pair ordinal = language rank. " + cap,
        )
        plot_vl_ordinal(
            ranks=fam["ranks"],
            labels=fam["labels"],
            series=fam["ser_os"],
            title=f"{fam['name']}  one-sided CKA a vs language version ordinal (ρ=0.1)",
            ylabel="one-sided test CKA a (ρ=0.1)",
            path=figdir / f"{slug}__vl_onesided_rho0p1_cka_vs_ordinal.png",
            ylim=ylim_os,
            caption="Supplementary. Language reweighted; vision identity. Pair ordinal = language rank. " + NOTE,
        )

    # Within-family language pairs: pair ordinal = mean rank
    for fname, keys in series.items():
        mapping = ranks[fname]
        n = len(keys)
        mat = np.full((n, n), np.nan)
        names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
        xs, ys, labs = [], [], []
        for i, a in enumerate(names):
            mat[i, i] = 1.0
            for b in names[i + 1 :]:
                rec = native_lookup(native, a, b)
                if rec is None:
                    unavailable.append({"a": a, "b": b, "metric": "native", "kind": "within_family"})
                    continue
                val = float(rec["cka_a"])
                ia, ib = mapping[a] - 1, mapping[b] - 1
                mat[ia, ib] = mat[ib, ia] = val
                po = pair_ordinal(mapping[a], mapping[b])
                xs.append(float(po))
                ys.append(val)
                labs.append(f"{a}\n× {b}")
                table_rows.append(
                    {
                        "kind": "within_family",
                        "family": fname,
                        "a": a,
                        "b": b,
                        "rank_a": mapping[a],
                        "rank_b": mapping[b],
                        "pair_ordinal": po,
                        "native_cka_a": val,
                        "onesided_rho0p1_test_a": None,
                    }
                )
        ticks = [f"{mapping[k]}:{k}" for k in names]
        slug = fname.lower().replace(" ", "_")
        heatmap(
            mat,
            ticks,
            ticks,
            f"{fname}  native CKA  (rows/cols = version ordinal)",
            figdir / f"{slug}__within_family_cka_heatmap.png",
            "native CKA a",
            cmap="viridis",
        )
        if xs:
            plot_pair_scatter(
                xs=xs,
                ys=ys,
                labels=labs,
                title=f"{fname}  native CKA vs pair version ordinal",
                xlabel="pair ordinal = mean of the two version ranks",
                ylabel="native CKA a (test)",
                path=figdir / f"{slug}__within_family_cka_vs_pair_ordinal.png",
                ylim=_ylim(ys),
                caption=cap,
            )

    # Cross-family base Qwen × OLMo: pair ordinal = mean of each family's rank
    q_map, o_map = ranks["Qwen-base"], ranks["OLMo-base"]
    xs, ys, labs = [], [], []
    q_names = [k for k, _ in sorted(q_map.items(), key=lambda kv: kv[1])]
    o_names = [k for k, _ in sorted(o_map.items(), key=lambda kv: kv[1])]
    mat = np.full((len(q_names), len(o_names)), np.nan)
    for a in q_names:
        for b in o_names:
            rec = native_lookup(native, a, b)
            if rec is None:
                unavailable.append({"a": a, "b": b, "metric": "native", "kind": "cross_family"})
                continue
            val = float(rec["cka_a"])
            mat[q_map[a] - 1, o_map[b] - 1] = val
            po = pair_ordinal(q_map[a], o_map[b])
            xs.append(float(po))
            ys.append(val)
            labs.append(f"{a} × {b}")
            table_rows.append(
                {
                    "kind": "cross_family_base",
                    "family": "Qwen-base×OLMo-base",
                    "a": a,
                    "b": b,
                    "rank_a": q_map[a],
                    "rank_b": o_map[b],
                    "pair_ordinal": po,
                    "native_cka_a": val,
                    "onesided_rho0p1_test_a": None,
                }
            )
    heatmap(
        mat,
        [f"{o_map[k]}:{k}" for k in o_names],
        [f"{q_map[k]}:{k}" for k in q_names],
        "Qwen-base × OLMo-base  native CKA  (ordinal × ordinal)",
        figdir / "cross_family__cka_heatmap.png",
        "native CKA a",
        cmap="viridis",
    )
    if xs:
        plot_pair_scatter(
            xs=xs,
            ys=ys,
            labels=labs,
            title="Qwen-base × OLMo-base  native CKA vs pair version ordinal",
            xlabel="pair ordinal = mean of Qwen rank and OLMo rank",
            ylabel="native CKA a (test)",
            path=figdir / "cross_family__cka_vs_pair_ordinal.png",
            ylim=_ylim(ys),
            caption="Same-generation pairs sit near integers (1,2,3); mixed generations at half-integers. " + cap,
        )

    write_json(tbldir / "cka_vs_pair_ordinal.json", jsonable(table_rows))
    hdr = ["kind", "family", "a", "b", "rank_a", "rank_b", "pair_ordinal", "native_cka_a", "onesided_rho0p1_test_a"]
    lines = [",".join(hdr)]
    for r in table_rows:
        def fmt(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.12g}"
            return str(v)

        lines.append(",".join(fmt(r[h]) for h in hdr))
    (tbldir / "cka_vs_pair_ordinal.csv").write_text("\n".join(lines) + "\n")
    write_json(out / "ranks.json", jsonable({"ranks": ranks, "rule": "1-based; public_release_date then key"}))
    write_json(out / "unavailable.json", jsonable(unavailable))
    summary = {
        "source_native": str(repair / "native_matrix.json"),
        "source_pair_eval": str(repair / "pair_eval.json"),
        "ranks": ranks,
        "n_rows": len(table_rows),
        "unavailable": unavailable,
        "did_not": ["fit", "extract", "overwrite_repair"],
        "note": NOTE,
    }
    write_json(out / "run_manifest.json", jsonable(summary))
    (out / "README.md").write_text(
        "\n".join(
            [
                "# CKA vs pair version ordinal",
                "",
                "Plotting-only. Native CKA from frozen `native_matrix.json`.",
                "",
                NOTE,
                "",
                "```bash",
                "export PYTHONPATH=/mnt/sdb1/prh-replication-work/repo/src",
                "/mnt/sdb1/prh-replication-work/venv/bin/python scripts/plot_cka_vs_recency.py \\",
                "  --work /mnt/sdb1/prh-replication-work",
                "```",
                "",
            ]
        )
    )
    return summary
