#!/usr/bin/env python
"""Build the load-bearing main-text figures from verified artifacts only.

Six figures, each carrying one finding the manuscript makes. No decorative plots.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
PAPER = REPO / "papers" / "xgboost_hpo_vrfnbi"
A = PAPER / "analysis"
FIG = PAPER / "manuscript" / "figures"
ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402

DISP = {"magic": "MAGIC", "adult": "Adult", "bank_marketing": "Bank Marketing",
        "spambase": "Spambase (boundary)"}
PRIMARY = ("magic", "adult", "bank_marketing")
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def hv(ds, method, ref="core"):
    return np.array([json.loads((ROOT/ds/f"rep_{r:02d}"/"metrics_by_method.json").read_text())
                     ["methods"][method][ref]["hv_ratio"] for r in range(N_REPLICATIONS)])


def fig_arm_lattice():
    """The causal decomposition: what each contrast holds fixed and what it varies."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.axis("off")
    nodes = [("HISTORICAL-WS\n-asrun", 0.03), ("HISTORICAL-WS", 0.27),
             ("WS-S", 0.51), ("NBI-S", 0.72), ("NBI-R", 0.93)]
    for name, x in nodes:
        ax.add_patch(plt.Rectangle((x-0.085, 0.45), 0.17, 0.22, fc="#eef2f7",
                                   ec="#3b4a5a", lw=1.0, zorder=2))
        ax.text(x, 0.56, name, ha="center", va="center", fontsize=8, zorder=3)
    labels = [("reproduction\n→ shared spec", 0.15), ("normalization", 0.39),
              ("front geometry", 0.615), ("anchor provenance", 0.825)]
    for txt, x in labels:
        ax.annotate("", xy=(x+0.075, 0.56), xytext=(x-0.075, 0.56),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#3b4a5a"))
        ax.text(x, 0.30, txt, ha="center", va="center", fontsize=7.5, style="italic")
    ax.text(0.615, 0.13, "primary contrasts", ha="center", fontsize=7.5, color="#8a1c1c")
    ax.plot([0.44, 0.99], [0.20, 0.20], color="#8a1c1c", lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(0.05, 0.75)
    fig.savefig(FIG/"fig1_arm_lattice.png"); plt.close(fig)


def fig_primary_effects():
    """Paired geometry effect per primary dataset, and the boundary dataset beside it."""
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.5), sharey=False)
    for ax, ds in zip(axes, PRIMARY + ("spambase",)):
        d = hv(ds, "nbi_s") - hv(ds, "ws_s")
        ax.axhline(0, color="#999", lw=0.8, zorder=1)
        ax.scatter(np.arange(len(d)), np.sort(d), s=11,
                   c=["#1b6ca8" if v > 0 else "#b03030" for v in np.sort(d)], zorder=3)
        ax.axhline(np.median(d), color="#1b6ca8", ls="--", lw=1.0, zorder=2)
        ax.set_title(f"{DISP[ds]}\nmedian {np.median(d):+.3f}, {int((d>0).sum())}/30",
                     fontsize=8)
        ax.set_xlabel("replication (sorted)", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("WS-S → NBI-S\nΔ core-relative HV", fontsize=8)
    fig.savefig(FIG/"fig2_geometry_effect.png"); plt.close(fig)


def fig_boundary_dispersion():
    """Why the boundary dataset did not resolve: dispersion, not absence."""
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    data = [hv(ds, "nbi_s") - hv(ds, "ws_s") for ds in PRIMARY + ("spambase",)]
    bp = ax.boxplot(data, vert=True, widths=0.55, showfliers=True,
                    patch_artist=True, medianprops=dict(color="#1b1b1b"))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("#cfe0ef" if i < 3 else "#f2d9d9")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticklabels([DISP[d].replace(" (boundary)", "\n(boundary)") for d in
                        PRIMARY + ("spambase",)], fontsize=7)
    ax.set_ylabel("WS-S → NBI-S\nΔ core-relative HV", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.savefig(FIG/"fig3_boundary_dispersion.png"); plt.close(fig)


def fig_anchor_and_chim():
    """The negative anchor result, and the CHIM contraction associated with it."""
    chim = json.loads((A/"nbi_r_chim_collapse.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ds_all = PRIMARY + ("spambase",)
    med = [np.median(hv(d, "nbi_r") - hv(d, "nbi_s")) for d in ds_all]
    inj = [json.loads((A/"secondary_analysis.json").read_text())
           ["controls"][d]["anchor_injection"]["injection_effect_median"] for d in ds_all]
    x = np.arange(len(ds_all))
    a1.bar(x-0.19, med, 0.38, label="NBI-S → NBI-R (full)", color="#b03030")
    a1.bar(x+0.19, inj, 0.38, label="anchor injection only", color="#7fa8c9")
    a1.axhline(0, color="#999", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels([DISP[d].split(" (")[0] for d in ds_all],
                                         fontsize=7, rotation=20, ha="right")
    a1.set_ylabel("Δ core-relative HV", fontsize=8); a1.legend(fontsize=6.5, frameon=False)
    a1.tick_params(labelsize=7)
    r = [chim[d]["extent_ratio_median"] for d in ds_all]
    a2.bar(x, r, 0.5, color="#4f7ea8")
    a2.axhline(1.0, color="#999", lw=0.8, ls="--")
    a2.set_xticks(x); a2.set_xticklabels([DISP[d].split(" (")[0] for d in ds_all],
                                         fontsize=7, rotation=20, ha="right")
    a2.set_ylabel("CHIM extent, NBI-R / NBI-S", fontsize=8); a2.set_ylim(0, 1.15)
    a2.tick_params(labelsize=7)
    fig.savefig(FIG/"fig4_anchor_provenance.png"); plt.close(fig)


def fig_reference_sensitivity():
    """CORE primary against AUGMENTED sensitivity, all 12 cells."""
    prim = json.loads((A/"primary_analysis.json").read_text())
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for blk in ("primary_family", "boundary_control"):
        for ds in prim[blk]["core"]:
            for rc, ra in zip(prim[blk]["core"][ds], prim[blk]["augmented"][ds]):
                sig = rc["holm_significant_at_05"], ra["holm_significant_at_05"]
                col = "#b03030" if sig[0] != sig[1] else "#1b6ca8"
                ax.scatter(rc["median_diff"], ra["median_diff"], s=34, c=col,
                           edgecolors="k", linewidths=0.4, zorder=3)
    lim = 0.55
    ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
    ax.set_xlabel("median Δ, CORE reference (primary)", fontsize=8)
    ax.set_ylabel("median Δ, AUGMENTED (sensitivity)", fontsize=8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.tick_params(labelsize=7)
    ax.text(-0.5, 0.45, "red = the one cell whose\nsignificance differs", fontsize=6.5,
            color="#b03030", va="top")
    fig.savefig(FIG/"fig5_reference_sensitivity.png"); plt.close(fig)


def fig_gate_and_baselines():
    """Surrogate-gate regime, and the frozen-budget grid baseline."""
    sec = json.loads((A/"secondary_analysis.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ds_all = PRIMARY + ("spambase",); x = np.arange(len(ds_all))
    q = [sec["gate_regimes"][d]["quality_pass_rate"] for d in ds_all]
    c = [sec["gate_regimes"][d]["cost_pass_rate"] for d in ds_all]
    a1.bar(x-0.19, q, 0.38, label="composite quality", color="#b03030")
    a1.bar(x+0.19, c, 0.38, label="cost", color="#4f7ea8")
    a1.set_xticks(x); a1.set_xticklabels([DISP[d].split(" (")[0] for d in ds_all],
                                         fontsize=7, rotation=20, ha="right")
    a1.set_ylabel("gate pass rate over R = 30", fontsize=8); a1.set_ylim(0, 1.05)
    a1.legend(fontsize=6.5, frameon=False); a1.tick_params(labelsize=7)
    ents = ("WS-S", "NBI-S", "NBI-R", "GRID", "NSGA2-MATCHED")
    w = 0.16
    for i, e in enumerate(ents):
        vals = [sec["secondary_indicators"][d].get(e, {}).get("hv_ratio", {}).get("median")
                or sec["baselines"][d]["evaluation_matched"].get(e, {}).get("median_hv_ratio")
                for d in ds_all]
        a2.bar(x + (i-2)*w, vals, w, label=e.replace("-MATCHED", ""), )
    a2.axhline(1.0, color="#999", lw=0.8, ls="--")
    a2.set_xticks(x); a2.set_xticklabels([DISP[d].split(" (")[0] for d in ds_all],
                                         fontsize=7, rotation=20, ha="right")
    a2.set_ylabel("median core-relative HV", fontsize=8)
    a2.legend(fontsize=6, frameon=False, ncol=2); a2.tick_params(labelsize=7)
    fig.savefig(FIG/"fig6_gate_and_baselines.png"); plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_arm_lattice, fig_primary_effects, fig_boundary_dispersion,
              fig_anchor_and_chim, fig_reference_sensitivity, fig_gate_and_baselines):
        f(); print(f"  built {f.__name__}")
    print(f"\nwrote {len(list(FIG.glob('*.png')))} figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
