"""Figures for exp-005 from cknna_k_sweep.py output.

    uv run python experiments/exp-005-cknna-large-k/plot.py --root results/exp-005-cknna-large-k

Reads <root>/diagnostics (--diagnostics run) and, when present, <root>/grid; writes PNG + PDF to
<root>/figures:

* fig1_convergence      mKNN(k) and CKNNA(k) with linear CKA and unbiased CKA overlaid, one
                        panel per ViT, mean over LLMs (band = min..max)
* fig2_why_above_1      CKNNA vs its centred / mask-only variants and its permutation null
* fig3_anisotropy       CKNNA at k=800 per layer pair vs the layers' mean cosine similarity
* fig4_families         CKNNA(k) for one LLM against every ViT, coloured by vision family
* fig5_vision_vision    last-block ViT x ViT (PRH Fig. 12 setting): CKNNA and its centred
                        variant for MAE-MAE, MAE-other and other pairs (<root>/vision_vision)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
# categorical slots in fixed order (reference palette): blue, orange, aqua, yellow, magenta, green
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METRIC_COLOR = {
    "mutual_knn": SERIES[0],
    "cknna": SERIES[1],
    "cka": SERIES[2],
    "unbiased_cka": SERIES[3],
    "centered": SERIES[4],
    "mask_only": SERIES[5],
}
FAMILIES = [
    ("ImageNet21K", lambda n: "augreg" in n),
    ("MAE", lambda n: ".mae" in n),
    ("DINOv2", lambda n: "dinov2" in n),
    ("CLIP", lambda n: "clip" in n and "ft_in12k" not in n),
    ("CLIP (I12K ft)", lambda n: "ft_in12k" in n),
]
DEGENERATE_STD = 0.01  # a layer whose cosine similarities have std below this is ~constant

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 2,
    }
)


def family(lvm: str) -> str:
    return next(name for name, match in FAMILIES if match(lvm))


def short(name: str) -> str:
    return (
        name.split("/")[-1]
        .replace("_patch16_224", "")
        .replace("_patch14_224", "")
        .replace("_patch14", "")
        .replace("_patch16", "")
        .replace("_clip_224", "_clip")
        .replace(".augreg_in21k", " IN21k")
        .replace(".lvd142m", "")
        .replace(".laion2b", " laion2b")
        .replace(".mae", " MAE")
    )


def load(run: Path) -> list[dict]:
    path = run / "sweep_pairs.jsonl"
    if not path.exists():
        return []
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    for r in recs:
        stem = f"{r['llm'].replace('/', '__')}__x__{r['lvm']}"
        r["mats"] = dict(np.load(run / "pairs" / f"{stem}.npz"))
    return recs


def series(rec: dict, key: str) -> np.ndarray:
    return np.array([p[key] for p in rec["per_k"]])


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}.png")


def fig1_convergence(recs: list[dict], out: Path) -> None:
    lvms = list(dict.fromkeys(r["lvm"] for r in recs))
    llms = list(dict.fromkeys(r["llm"] for r in recs))
    n = recs[0]["n"]
    fig, axes = plt.subplots(1, len(lvms), figsize=(2.6 * len(lvms), 2.9), sharey=True)
    for ax, lvm in zip(np.atleast_1d(axes), lvms, strict=True):
        rs = [r for r in recs if r["lvm"] == lvm]
        ks = series(rs[0], "k")
        for key, label in (
            ("mutual_knn", "mutual kNN(k)"),
            ("cknna", "CKNNA(k)"),
            ("centered", "CKNNA(k) on centred Grams"),
        ):
            if key not in rs[0]["per_k"][0]:
                continue
            y = np.stack([series(r, key) for r in rs])
            ax.fill_between(ks, y.min(0), y.max(0), color=METRIC_COLOR[key], alpha=0.15, lw=0)
            ax.plot(ks, y.mean(0), color=METRIC_COLOR[key], label=label)
        cka = np.mean([r["cka"] for r in rs])
        ax.axhline(cka, color=METRIC_COLOR["cka"], ls="--", lw=1.5, label="linear CKA")
        # CKNNA at k = n - 1 is unbiased CKA exactly
        ucka = np.mean([series(r, "cknna")[-1] for r in rs])
        ax.plot(
            [ks[-1]],
            [ucka],
            "o",
            ms=8,
            color=METRIC_COLOR["unbiased_cka"],
            mec=SURFACE,
            mew=2,
            label="unbiased CKA (= CKNNA at k=n-1)",
            zorder=5,
        )
        ax.plot(ks, ks / (n - 1), color=INK_MUTED, ls=":", lw=1, label="mKNN chance k/(n-1)")
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_title(short(lvm), color=INK)
        ax.set_xlabel("k (n = 1024)")
        ax.set_xticks([10, 200, 400, 600, 800, 1023])
        ax.set_xticklabels(["10", "200", "400", "600", "800", "1024"], rotation=45)
    np.atleast_1d(axes)[0].set_ylabel("score (max over layer pairs)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16)
    )
    fig.suptitle(
        f"Raw alignment vs neighbourhood size, mean over {len(llms)} LLMs (band = min..max)",
        y=1.22,
        color=INK,
    )
    save(fig, out, "fig1_convergence")


def fig2_why_above_1(recs: list[dict], out: Path, llm: str) -> None:
    rs = [r for r in recs if r["llm"] == llm]
    fig, axes = plt.subplots(1, len(rs), figsize=(2.6 * len(rs), 2.9), sharey=True)
    curves = [
        ("cknna", "CKNNA (upstream)", METRIC_COLOR["cknna"], "-"),
        ("centered", "CKNNA on centred Grams", METRIC_COLOR["centered"], "-"),
        ("mask_only", "mask-only CKNNA (K = L = 1)", METRIC_COLOR["mask_only"], "-"),
        ("cknna_null_mean", "CKNNA, permuted pairing (null mean)", INK_MUTED, "--"),
    ]
    for ax, r in zip(np.atleast_1d(axes), rs, strict=True):
        ks, p = series(r, "k"), series(r, "p")
        for key, label, color, ls in curves:
            ax.plot(ks, series(r, key), color=color, ls=ls, label=label)
        ax.plot(ks[:-1], (p * (1 + p))[:-1], color=INK, ls=":", lw=1, label="random masks p(1+p)")
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_title(short(r["lvm"]), color=INK)
        ax.set_xlabel("k")
        ax.set_xticks([10, 200, 400, 600, 800, 1023])
        ax.set_xticklabels(["10", "200", "400", "600", "800", "1024"], rotation=45)
    np.atleast_1d(axes)[0].set_ylabel("score (max over layer pairs)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.1)
    )
    fig.suptitle(f"Why CKNNA exceeds 1: {short(llm)} vs each ViT", y=1.16, color=INK)
    save(fig, out, "fig2_why_above_1")


def fig3_anisotropy(recs: list[dict], out: Path, k: int = 800) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for fam_idx, (fam, _) in enumerate(FAMILIES):
        xs, ys = [], []
        for r in recs:
            if family(r["lvm"]) != fam:
                continue
            t = int(np.argmin(np.abs(r["mats"]["ks"] - k)))
            a = r["anisotropy"]
            vm, lm = np.array(a["vision_mean_cos"]), np.array(a["language_mean_cos"])
            ok_v = np.array(a["vision_std_cos"]) > DEGENERATE_STD
            ok_l = np.array(a["language_std_cos"]) > DEGENERATE_STD
            mat = r["mats"]["cknna"][t][np.ix_(ok_v, ok_l)]
            xs.append(np.sqrt(np.outer(vm[ok_v], lm[ok_l])).ravel())
            ys.append(mat.ravel())
        if xs:
            ax.scatter(
                np.concatenate(xs),
                np.concatenate(ys),
                s=6,
                alpha=0.35,
                color=SERIES[fam_idx],
                lw=0,
                label=fam,
            )
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xlabel("anisotropy: sqrt(mean cos(vision layer) x mean cos(LLM layer))")
    ax.set_ylabel(f"CKNNA at k={k} (every layer pair)")
    ax.legend(frameon=False, markerscale=3, loc="lower right")
    ax.set_title("Large-k CKNNA tracks anisotropy, not alignment", color=INK)
    save(fig, out, "fig3_anisotropy")


def fig4_families(recs: list[dict], out: Path, llm: str) -> None:
    rs = [r for r in recs if r["llm"] == llm]
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, (key, title) in zip(
        axes, (("cknna", "CKNNA (upstream)"), ("mutual_knn", "mutual kNN")), strict=True
    ):
        seen = set()
        for r in sorted(rs, key=lambda r: [f for f, _ in FAMILIES].index(family(r["lvm"]))):
            fam = family(r["lvm"])
            color = SERIES[[f for f, _ in FAMILIES].index(fam)]
            ax.plot(
                series(r, "k"),
                series(r, key),
                color=color,
                lw=1.2,
                alpha=0.9,
                label=None if fam in seen else fam,
            )
            seen.add(fam)
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_title(title, color=INK)
        ax.set_xlabel("k")
        ax.set_xticks([10, 200, 400, 600, 800, 1023])
        ax.set_xticklabels(["10", "200", "400", "600", "800", "1024"])
    axes[0].set_ylabel("score (max over layer pairs)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"{short(llm)} vs all {len(rs)} ViTs", color=INK)
    save(fig, out, "fig4_families")


def fig5_vision_vision(path: Path, out: Path) -> None:
    d = np.load(path)
    ks, lvms = d["ks"], [str(m) for m in d["lvms"]]
    mae = np.array([".mae" in m for m in lvms])
    iu = np.triu_indices(len(lvms), 1)
    groups = {
        "MAE x MAE": mae[iu[0]] & mae[iu[1]],
        "MAE x other": mae[iu[0]] ^ mae[iu[1]],
        "other x other": ~mae[iu[0]] & ~mae[iu[1]],
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    for ax, (name, sel) in zip(axes, groups.items(), strict=True):
        for key, label in (
            ("mutual_knn", "mutual kNN(k)"),
            ("cknna", "CKNNA(k)"),
            ("centered", "CKNNA(k) on centred Grams"),
            ("mask_only", "mask-only CKNNA"),
        ):
            y = d[key][:, iu[0], iu[1]][:, sel]
            ax.fill_between(ks, y.min(1), y.max(1), color=METRIC_COLOR[key], alpha=0.15, lw=0)
            ax.plot(ks, y.mean(1), color=METRIC_COLOR[key], label=label)
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_title(f"{name} ({int(sel.sum())} pairs)", color=INK)
        ax.set_xlabel("k")
        ax.set_xticks([10, 200, 400, 600, 800, 1023])
        ax.set_xticklabels(["10", "200", "400", "600", "800", "1024"], rotation=45)
    axes[0].set_ylabel("score, last-block CLS")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1)
    )
    cos_mae, cos_other = d["mean_cos"][mae].mean(), d["mean_cos"][~mae].mean()
    fig.suptitle(
        f"Vision-vision (PRH Fig. 12 setting). Mean cosine of last block: MAE {cos_mae:.2f}, "
        f"others {cos_other:.2f}",
        y=1.17,
        color=INK,
    )
    save(fig, out, "fig5_vision_vision")


def summary(recs: list[dict]) -> None:
    print(f"\n{'pair':<60} {'CKA':>5} {'uCKA':>5}  first k with CKNNA>1   peak")
    for r in recs:
        c = series(r, "cknna")
        ks = series(r, "k")
        above = ks[c > 1]
        first = int(above[0]) if len(above) else "-"
        print(
            f"{short(r['llm']) + ' x ' + short(r['lvm']):<60} {r['cka']:.2f}  {c[-1]:.2f}  "
            f"{first!s:>8}              {c.max():.2f} @ k={int(ks[c.argmax()])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--llm", default="huggyllama/llama-13b", help="LLM for fig 2")
    parser.add_argument("--family-llm", default="bigscience/bloomz-1b1", help="LLM for fig 4")
    args = parser.parse_args()
    out = args.root / "figures"
    diag = load(args.root / "diagnostics")
    grid = load(args.root / "grid")
    if diag:
        summary(diag)
        fig1_convergence(diag, out)
        fig2_why_above_1(diag, out, args.llm)
        fig3_anisotropy(diag, out)
    if grid:
        fig4_families(grid, out, args.family_llm)
    vv = args.root / "vision_vision" / "vision_vision.npz"
    if vv.exists():
        fig5_vision_vision(vv, out)


if __name__ == "__main__":
    main()
