#!/usr/bin/env python
"""Build the manuscript figures and LaTeX tables from the frozen R = 30 artifacts.

Reads only:
  reports/pco213_postwork_benchmark/{tables,statistics}/*.csv
  experiments/pco213_postwork_benchmark/<dataset>/rep_XX/*   (unversioned raw artifacts)
Writes:
  papers/surrogate_nbi_ensemble/figures/fig0X_*.{pdf,png}
  papers/surrogate_nbi_ensemble/tables/tab0X_*.tex
Nothing under reports/ or experiments/ is modified (frozen at tag pco213-postwork-r30).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from mixens.scheffe import model_from_coefficients  # noqa: E402

REP = REPO / "reports" / "pco213_postwork_benchmark"
EXP = REPO / "experiments" / "pco213_postwork_benchmark"
OUT = REPO / "papers" / "surrogate_nbi_ensemble"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

DATASETS = ["santander", "bnp", "porto", "uci_credit"]
DLABEL = {"santander": "Santander", "bnp": "BNP Paribas", "porto": "Porto Seguro", "uci_credit": "UCI credit"}
MODELS = ["lr", "gnb", "knn", "rf", "xgb"]
MLABEL = {"lr": "LR", "gnb": "GNB", "knn": "kNN", "rf": "RF", "xgb": "XGB"}
DSHORT = {"santander": "Santander", "bnp": "BNP", "porto": "Porto", "uci_credit": "UCI"}
SETS = {"nbi_A": "NBI-A", "nbi_B": "NBI-B", "nbi_C": "NBI-C", "ws_random_scalarization": "Scalarization",
        "random_dirichlet_budget": "Random Dirichlet", "design_runs": "Design runs"}
COL = {"nbi_A": "#d62728", "nbi_B": "#1f77b4", "nbi_C": "#2ca02c", "ws_random_scalarization": "#9467bd",
       "random_dirichlet_budget": "#8c564b", "design_runs": "#7f7f7f"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def nondominated(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of F (all objectives minimized)."""
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            mask[i] = False
    return mask


# --------------------------------------------------------------------------------------
# Figure 1: lineage and pipeline schematic
# --------------------------------------------------------------------------------------
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    colors = {"inh": "#dfe8f3", "ada": "#fbe8c8", "new": "#dff2df"}
    edge = {"inh": "#4a6fa5", "ada": "#c8862a", "new": "#3a8f3a"}

    def box(x, y, w, h, text, kind, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=colors[kind], ec=edge[kind], lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color="#333333"))

    # top lane: predecessor
    ax.text(1, 95, "Predecessor framework (engineering optimization)", fontsize=8, weight="bold", va="center")
    top = [("Process factors\n(box domain)", "inh"), ("Factorial / RSM\ndesign", "inh"),
           ("Polynomial\nmetamodels", "inh"), ("NBI on the\nmetamodels", "inh"),
           ("Mixture-design\npost-optimization", "inh")]
    xs = np.linspace(2, 80, len(top))
    for i, (t, k) in enumerate(top):
        box(xs[i], 78, 16, 12, t, k)
        if i:
            arrow(xs[i - 1] + 16.3, 84, xs[i] - 0.3, 84)
    # transfer arrow
    ax.annotate("", xy=(50, 62), xytext=(50, 76), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333333"))
    ax.text(52, 69, "transfer to compositional decision variables\n(ensemble weights on the simplex)",
            fontsize=7, va="center", style="italic")
    # bottom lane: present study
    ax.text(1, 58, "Present study (classifier ensemble weighting)", fontsize=8, weight="bold", va="center")
    bot = [("Out-of-fold\nprobabilities\n(5 models)", "new"),
           ("66-run mixture\ndesign on the\nweight simplex", "ada"),
           ("Scheffé surfaces\n+ external gate\n(100 unseen pts)", "ada"),
           ("Anchors:\nsurrogate (A)\nor real (B)", "new"),
           ("NBI-A, NBI-B\n(surrogate);\nNBI-C (real)", "ada"),
           ("Real-OOF\nrevalidation of\nall candidates", "new"),
           ("Empirical ref.;\nIGD$^+$, HV;\nholdout check", "new")]
    xs = 1 + 14.2 * np.arange(len(bot))
    for i, (t, k) in enumerate(bot):
        box(xs[i], 30, 12.2, 18, t, k, fs=5.8)
        if i:
            arrow(xs[i - 1] + 12.5, 39, xs[i] - 0.3, 39)
    # replication bracket
    ax.plot([1, 1, 98.5, 98.5], [26, 23, 23, 26], color="#333333", lw=0.8)
    ax.text(50, 19, "repeated over 4 datasets × 30 outer stratified 80/20 partitions (5-fold inner OOF); "
                    "comparisons paired by partition", ha="center", fontsize=7)
    # legend
    for i, (k, lab) in enumerate([("inh", "inherited"), ("ada", "adapted"), ("new", "new in this study")]):
        ax.add_patch(FancyBboxPatch((2 + 30 * i, 4), 4, 5, boxstyle="round,pad=0.2", fc=colors[k], ec=edge[k], lw=0.8))
        ax.text(7.5 + 30 * i, 6.5, lab, va="center", fontsize=7)
    save(fig, "fig01_lineage_pipeline")


# --------------------------------------------------------------------------------------
# Figure 2: surrogate success vs failure on unseen compositions
# --------------------------------------------------------------------------------------
def fig02_surrogate_validation():
    panels = [("uci_credit", 0, "roc_auc", "ROC-AUC"), ("santander", 0, "roc_auc", "ROC-AUC"),
              ("santander", 0, "log_loss", "log-loss"), ("bnp", 7, "roc_auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    fig.subplots_adjust(wspace=0.5)
    for ax, (ds, rep, resp, rl) in zip(axes, panels):
        d = EXP / ds / f"rep_{rep:02d}"
        s = json.load(open(d / "scheffe.json"))[resp]
        order = s["selected_order"]
        o = s["orders"][order]
        m = model_from_coefficients([f"w_{k}" for k in MODELS], o["terms"], o["coefficients"])
        val = pd.read_csv(d / "validation_eval.csv")
        W = val[[f"w_{k}" for k in MODELS]].to_numpy()
        pred = m.predict_weights(W)
        obs = val[resp].to_numpy()
        ax.scatter(obs, pred, s=7, color="#1f77b4", alpha=0.75, lw=0)
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.7, ls="--")
        r2 = o["external"]["r2_external"]
        rho = o["spearman_external"]
        gate = "pass" if s["reliable"] else "fail"
        ax.set_title(f"{DSHORT[ds]}, {rl}, partition {rep}\n{order}: $R^2_{{ext}}$ = {r2:.2f}, ρ = {rho:.2f}\ngate {gate}", fontsize=6.8)
        ax.set_xlabel(f"observed {rl}")
        if ax is axes[0]:
            ax.set_ylabel("Scheffé prediction")
        ax.tick_params(labelsize=6.5)
    save(fig, "fig02_surrogate_validation")


# --------------------------------------------------------------------------------------
# Figure 3: NBI-A / NBI-B / NBI-C fronts against the empirical reference
# --------------------------------------------------------------------------------------
def fig03_fronts():
    cases = [("santander", 0), ("bnp", 7)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.32)
    for r, (ds, rep) in enumerate(cases):
        d = EXP / ds / f"rep_{rep:02d}"
        ref = pd.read_csv(d / "empirical_reference_front_weighted.csv")
        for c, (xk, yk, xl, yl, logy) in enumerate([("roc_auc", "log_loss", "ROC-AUC", "log-loss", False),
                                                     ("roc_auc", "cost_weighted", "ROC-AUC", "weighted cost (ms / 1k rows)", True)]):
            ax = axes[r, c]
            ax.scatter(ref[xk], ref[yk], s=5, color="#bbbbbb", lw=0, label="empirical reference front")
            for key in ["nbi_C", "nbi_B", "nbi_A"]:
                cand = pd.read_csv(d / f"{key}_candidates.csv")
                cand = cand[cand["success"].astype(bool)] if "success" in cand else cand
                F = np.column_stack([-cand["roc_auc"], cand["log_loss"], cand["cost_weighted"]])
                nd = nondominated(F)
                sub = cand[nd]
                ax.scatter(sub[xk], sub[yk], s=13, color=COL[key], alpha=0.85, lw=0.3, edgecolor="white",
                           label=f"{SETS[key]} real front (n = {nd.sum()})")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{DLABEL[ds]}, partition {rep}", loc="left", fontsize=8)
            if r == 0 and c == 0:
                handles, labels = ax.get_legend_handles_labels()
    labels = [l.split(" (n")[0] for l in labels]
    order = [0, 3, 2, 1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig03_nbi_fronts")


# --------------------------------------------------------------------------------------
# Figure 4: paired differences (weighted cost)
# --------------------------------------------------------------------------------------
def fig04_paired():
    q = pd.read_csv(REP / "tables" / "pareto_quality.csv")
    q = q[q.cost == "weighted"]
    comps = [("nbi_B", "nbi_A", "NBI-B − NBI-A"), ("nbi_C", "nbi_B", "NBI-C − NBI-B"),
             ("nbi_C", "ws_random_scalarization", "NBI-C − Scal.")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharey="row", sharex="col")
    fig.subplots_adjust(hspace=0.12, wspace=0.12)
    rng = np.random.default_rng(0)
    for j, ds in enumerate(DATASETS):
        sub = q[q.dataset == ds]
        for i, (metric, sign, lab) in enumerate([("hv_ratio", 1, "Δ hypervolume ratio"), ("igd_plus", -1, "Δ IGD$^+$ (ref − new)")]):
            ax = axes[i, j]
            data = []
            for a, b, _ in comps:
                pa = sub[sub.set == a].set_index("rep")[metric]
                pb = sub[sub.set == b].set_index("rep")[metric]
                delta = sign * (pa - pb.reindex(pa.index))
                data.append(delta.dropna().to_numpy())
            for k, dlt in enumerate(data):
                x = k + rng.uniform(-0.18, 0.18, len(dlt))
                ax.scatter(x, dlt, s=6, color="#1f77b4", alpha=0.55, lw=0)
                ax.hlines(np.median(dlt), k - 0.3, k + 0.3, color="#d62728", lw=1.4)
            ax.axhline(0, color="#777777", lw=0.6, ls="--")
            ax.set_xticks(range(len(comps)))
            if i == 1:
                ax.set_xticklabels([c[2] for c in comps], rotation=25, ha="right", fontsize=6.5)
            else:
                ax.set_title(DLABEL[ds])
            if j == 0:
                ax.set_ylabel(lab)
            if metric == "igd_plus":
                ax.set_yscale("symlog", linthresh=0.01)
    save(fig, "fig04_paired_deltas")


# --------------------------------------------------------------------------------------
# Figure 5: reliability gate frequencies and external R²
# --------------------------------------------------------------------------------------
def fig05_gate():
    g = pd.read_csv(REP / "statistics" / "reliability_gate_r30.csv")
    so = pd.read_csv(REP / "tables" / "scheffe_orders.csv")
    so = so[(so.selected.astype(bool)) & (so.response.isin(["roc_auc", "log_loss"]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.subplots_adjust(wspace=0.38)
    ax = axes[0]
    w = 0.36
    for k, (resp, lab, col) in enumerate([("roc_auc", "ROC-AUC", "#1f77b4"), ("log_loss", "log-loss", "#ff7f0e")]):
        sub = g[g.response == resp].set_index("dataset").loc[DATASETS]
        x = np.arange(len(DATASETS)) + (k - 0.5) * w
        err = np.vstack([np.maximum(sub.p_pass - sub.wilson_lo, 0), np.maximum(sub.wilson_hi - sub.p_pass, 0)])
        ax.bar(x, sub.p_pass, w, color=col, label=lab, yerr=err, capsize=2, error_kw={"lw": 0.7})
        for xi, (p, kk) in zip(x, zip(sub.p_pass, sub["pass"])):
            if p > 0.85:
                ax.text(xi, 0.5, f"{int(kk)}/30", ha="center", va="center", fontsize=6, color="white", rotation=90)
            else:
                ax.text(xi, p + 0.03 + (0.04 if p < 0.05 else 0), f"{int(kk)}/30", ha="center", fontsize=6)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DSHORT[d] for d in DATASETS], fontsize=7)
    ax.set_ylabel("P(gate pass), Wilson 95%")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax = axes[1]
    rng = np.random.default_rng(1)
    for k, (resp, col) in enumerate([("roc_auc", "#1f77b4"), ("log_loss", "#ff7f0e")]):
        for j, ds in enumerate(DATASETS):
            v = so[(so.dataset == ds) & (so.response == resp)].r2_external.to_numpy()
            x = j + (k - 0.5) * 0.36 + rng.uniform(-0.1, 0.1, len(v))
            ax.scatter(x, np.clip(v, -1.05, 1.05), s=6, color=col, alpha=0.6, lw=0)
    ax.axhline(0.5, color="#d62728", lw=0.8, ls="--")
    ax.text(3.45, 0.53, "gate: $R^2_{ext}$ ≥ 0.5", fontsize=6.5, color="#d62728", ha="right")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DSHORT[d] for d in DATASETS], fontsize=7)
    ax.set_ylabel("external $R^2$, selected order\n(clipped at −1)")
    ax.set_ylim(-1.1, 1.1)
    save(fig, "fig05_reliability_gate")


# --------------------------------------------------------------------------------------
# Figure 6: R = 10 versus R = 30
# --------------------------------------------------------------------------------------
def fig06_r10_r30():
    r = pd.read_csv(REP / "statistics" / "r10_vs_r30_stability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.45)
    mk = {"nbi_A": "o", "nbi_B": "s", "nbi_C": "^"}
    dcol = {"santander": "#1f77b4", "bnp": "#ff7f0e", "porto": "#2ca02c", "uci_credit": "#9467bd"}
    for ax, (metric, lab) in zip(axes[:2], [("hv_ratio", "hypervolume ratio"), ("igd_plus", "IGD$^+$")]):
        sub = r[(r.metric == metric) & (r.cost == "weighted") & (r.set.isin(mk))]
        for _, row in sub.iterrows():
            ax.errorbar(row.r10_estimate, row.r30_estimate, yerr=[[row.r30_estimate - row.r30_ci95_lo], [row.r30_ci95_hi - row.r30_estimate]],
                        fmt=mk[row.set], ms=4, color=dcol[row.dataset], capsize=1.5, lw=0.7, mew=0.5, alpha=0.9)
        lo = min(sub.r10_estimate.min(), sub.r30_ci95_lo.min())
        hi = max(sub.r10_estimate.max(), sub.r30_ci95_hi.max())
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.7, ls="--")
        ax.set_xlabel(f"R = 10 mean {lab}")
        ax.set_ylabel(f"R = 30 mean {lab}\n[95% bootstrap]")
        if metric == "igd_plus":
            ax.set_xscale("log")
            ax.set_yscale("log")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk[s], color="k", ls="", ms=4, label=SETS[s]) for s in mk]
    h += [Line2D([], [], marker="o", color=dcol[d], ls="", ms=4, label=DLABEL[d]) for d in DATASETS]
    fig.legend(handles=h, frameon=False, fontsize=6.5, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.16))
    ax = axes[2]
    sub = r[r.metric == "reliability_gate_pass"]
    for k, (resp, col) in enumerate([("roc_auc", "#1f77b4"), ("log_loss", "#ff7f0e")]):
        s2 = sub[sub.response == resp].set_index("dataset").loc[DATASETS]
        x = np.arange(4) + (k - 0.5) * 0.36
        ax.bar(x - 0.09, s2.r10_estimate, 0.17, color=col, alpha=0.45, label=f"{'AUC' if resp=='roc_auc' else 'log-loss'} R=10")
        ax.bar(x + 0.09, s2.r30_estimate, 0.17, color=col, label=f"{'AUC' if resp=='roc_auc' else 'log-loss'} R=30")
    ax.set_xticks(range(4))
    ax.set_xticklabels([DSHORT[d] for d in DATASETS], fontsize=6)
    ax.set_ylabel("gate pass fraction")
    ax.set_ylim(0, 1.32)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    save(fig, "fig06_r10_vs_r30")


# --------------------------------------------------------------------------------------
# Figure 7: beta_ij versus real 50/50 blend gain
# --------------------------------------------------------------------------------------
def fig07_beta_synergy():
    c = pd.read_csv(REP / "statistics" / "coefficient_stability_r30.csv")
    c = c[(c.kind == "interaction") & (c.response == "roc_auc")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    fig.subplots_adjust(wspace=0.45)
    for ax, ds in zip(axes, DATASETS):
        sub = c[c.dataset == ds].sort_values("mean", ascending=False)
        col = np.where(sub.blend50_beats_best_member_freq >= 0.8, "#2ca02c", "#d62728")
        ax.scatter(sub["mean"], sub.blend50_minus_best_member_mean, s=12 + 250 * sub.vertex_gap_mean,
                   c=col, alpha=0.75, lw=0.4, edgecolor="k")
        top = sub.iloc[0]
        ax.annotate(top.term.replace("*", "·").upper().replace("KNN", "kNN"),
                    (top["mean"], top.blend50_minus_best_member_mean), textcoords="offset points",
                    xytext=(-6, 4), fontsize=6, ha="right")
        ax.set_xlim(0, sub["mean"].max() * 1.15)
        ax.axhline(0, color="#777777", lw=0.6, ls="--")
        ax.set_title(DLABEL[ds])
        ax.set_xlabel("mean $\\hat\\beta_{ij}$ (ROC-AUC)")
        if ax is axes[0]:
            ax.set_ylabel("real 50/50 blend − better member\n(OOF ROC-AUC, mean of 30)")
    save(fig, "fig07_beta_vs_synergy")


# --------------------------------------------------------------------------------------
# Figure 8: weighted versus support cost
# --------------------------------------------------------------------------------------
def fig08_cost_definition():
    q = pd.read_csv(REP / "tables" / "pareto_quality.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fig.subplots_adjust(wspace=0.42)
    # (a) winner frequencies
    ax = axes[0]
    sets = ["nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization", "random_dirichlet_budget", "design_runs"]
    for j, ds in enumerate(DATASETS):
        for k, cost in enumerate(["weighted", "support"]):
            sub = q[(q.dataset == ds) & (q.cost == cost) & (q.set.isin(sets))]
            best = sub.loc[sub.groupby("rep")["hv_ratio"].idxmax()]
            counts = best.set.value_counts()
            bottom = 0
            x = j + (k - 0.5) * 0.38
            for s in sets:
                n = counts.get(s, 0)
                if n:
                    ax.bar(x, n, 0.34, bottom=bottom, color=COL[s], label=SETS[s])
                    bottom += n
            ax.text(x, 30.6, "W" if cost == "weighted" else "S", ha="center", fontsize=6)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([DSHORT[d] for d in DATASETS], fontsize=6.5)
    ax.set_ylabel("partitions where the set has\nthe best hypervolume (of 30)")
    ax.set_ylim(0, 33)
    # (b),(c) example: BNP rep 0 NBI-C candidates under both costs
    for ax, (ds, rep) in zip(axes[1:], [("bnp", 0), ("uci_credit", 0)]):
        d = EXP / ds / f"rep_{rep:02d}"
        cand = pd.read_csv(d / "nbi_C_candidates.csv")
        cand = cand[cand["success"].astype(bool)] if "success" in cand else cand
        design = pd.read_csv(d / "design_eval.csv")
        ax.scatter(cand.roc_auc, cand.cost_weighted, s=9, color=COL["nbi_C"], label="NBI-C, weighted cost", lw=0)
        ax.scatter(cand.roc_auc, cand.cost_support, s=9, marker="^", color="#145214", label="NBI-C, support cost", lw=0)
        ax.scatter(design.roc_auc, design.cost_support, s=9, marker="x", color="#7f7f7f", label="design runs, support cost", lw=0.6)
        ax.set_yscale("log")
        ax.set_xlim(cand.roc_auc.min() - 0.006, cand.roc_auc.max() + 0.004)
        ax.set_xlabel("OOF ROC-AUC")
        ax.set_ylabel("cost (ms / 1k rows)")
        ax.set_title(f"{DLABEL[ds]}, partition {rep}", fontsize=7.5)
        if ds == "bnp":
            ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3)
    save(fig, "fig08_cost_definition")


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def tab01_datasets():
    rows = []
    meta = {"santander": ("Santander Customer Transaction Prediction (Kaggle, 2019)", 200000, 200, 0.100),
            "bnp": ("BNP Paribas Cardif Claims Management (Kaggle, 2016)", 114321, 131, 0.761),
            "porto": ("Porto Seguro Safe Driver Prediction (Kaggle, 2017); 200,000-row stratified subsample", 200000, 57, 0.036),
            "uci_credit": ("Default of credit card clients (UCI 350; Yeh and Lien, 2009)", 30000, 23, 0.221)}
    for ds in DATASETS:
        z = np.load(EXP / ds / "rep_00" / "oof.npz")
        n_tr, n_te = len(z["y_train"]), len(z["y_test"])
        name, n, p, prev = meta[ds]
        rows.append(f"{DLABEL[ds]} & {name} & {n:,} & {p} & {prev:.3f} & {n_tr:,} / {n_te:,} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
\centering
\caption{Datasets. Rows used, number of raw features, positive-class prevalence and training/holdout sizes of each outer partition (stratified 80/20). Kaggle test labels are never used; all selection is done on 5-fold out-of-fold predictions within the training part.}
\label{tab:datasets}
\small
\begin{tabular}{llrrrl}
\toprule
Dataset & Source & Rows & Features & Prevalence & Train / holdout rows \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "tab01_datasets.tex").write_text(tex)
    print("wrote tab01")


def tab02_gate():
    g = pd.read_csv(REP / "statistics" / "reliability_gate_r30.csv")
    rows = []
    for ds in DATASETS:
        for resp, rl in [("roc_auc", "ROC-AUC"), ("log_loss", "log-loss")]:
            r = g[(g.dataset == ds) & (g.response == resp)].iloc[0]
            orders = eval(r.orders)
            SPC = "sp.\\,cubic"
            od = ", ".join(f"{k.replace('special_cubic', SPC)} {v}" for k, v in sorted(orders.items(), key=lambda kv: -kv[1]))
            rows.append(f"{DLABEL[ds]} & {rl} & {od} & {int(r['pass'])}/30 & [{r.wilson_lo:.2f}, {r.wilson_hi:.2f}] & "
                        f"{r.r2_external_median:.3f} [{r.r2_external_ci95_lo:.3f}, {r.r2_external_ci95_hi:.3f}] & {r.spearman_median:.3f} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
\centering
\caption{Surrogate reliability over 30 partitions. Selected Scheffé order (parsimony rule: lowest order within 10\% of the best external RMSE), reliability-gate passes ($R^2_{\mathrm{ext}} \ge 0.5$ and Spearman $\rho \ge 0.9$ on 100 unseen compositions) with Wilson 95\% intervals, and the median external $R^2$ with its bootstrap interval. Source: \texttt{statistics/reliability\_gate\_r30.csv}.}
\label{tab:gate}
\small
\begin{tabular}{llllll r}
\toprule
Dataset & Response & Selected order (count) & Pass & Wilson 95\% & Median $R^2_{\mathrm{ext}}$ [95\% CI] & Median $\rho$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "tab02_gate.tex").write_text(tex)
    print("wrote tab02")


def tab03_paired(cost="weighted", name="tab03_paired_primary", label="tab:paired"):
    e = pd.read_csv(REP / "statistics" / "paired_primary_effects.csv")
    t = pd.read_csv(REP / "statistics" / "paired_primary_tests.csv")
    e = e[e.cost == cost]
    t = t[t.cost == cost]
    comps = [("nbi_B vs nbi_A", "NBI-B vs NBI-A"), ("nbi_C vs nbi_B", "NBI-C vs NBI-B"),
             ("nbi_C vs ws_random_scalarization", "NBI-C vs scalarization"),
             ("nbi_C vs random_dirichlet_budget", "NBI-C vs random Dirichlet")]
    rows = []
    for ds in DATASETS:
        first = True
        for ck, cl in comps:
            cells = []
            for ep in ["igd_plus", "hv_ratio"]:
                r = e[(e.dataset == ds) & (e.comparison == ck) & (e.endpoint == ep)].iloc[0]
                p = t[(t.dataset == ds) & (t.comparison == ck) & (t.endpoint == ep)].iloc[0]
                nd = 4 if abs(r["mean"]) < 0.05 else 3
                ptxt = "$<$0.001" if p.p_nb_holm_family_dataset < 0.001 else f"{p.p_nb_holm_family_dataset:.3f}"
                cells.append(f"{r['mean']:+.{nd}f} [{r.ci95_mean_lo:+.{nd}f}, {r.ci95_mean_hi:+.{nd}f}] & {r['median']:+.{nd}f} & "
                             f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {r.rank_biserial:+.2f} & {ptxt}")
            rows.append(f"{DLABEL[ds] if first else ''} & {cl} & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    cap = ("Primary paired comparisons under the weighted cost (30 partitions per dataset; $\\Delta > 0$ favours the second-named set: "
           "$\\Delta\\mathrm{IGD}^+ = \\mathrm{IGD}^+_{\\mathrm{ref}} - \\mathrm{IGD}^+_{\\mathrm{new}}$, $\\Delta\\mathrm{HV} = \\mathrm{HV}_{\\mathrm{new}} - \\mathrm{HV}_{\\mathrm{ref}}$). "
           "Mean with percentile-bootstrap 95\\% interval, median, wins/ties/losses, matched-pairs rank-biserial $r$, and the Holm-corrected "
           "Nadeau--Bengio $p$ ($\\rho = 0.25$, family of eight tests per dataset). Source: \\texttt{statistics/paired\\_primary\\_effects.csv}, "
           "\\texttt{paired\\_primary\\_tests.csv}."
           if cost == "weighted" else
           "Primary paired comparisons re-scored under the support cost (same conventions as Table~\\ref{tab:paired}; all NBI sets were optimized "
           "under the weighted cost and only re-scored). Source: \\texttt{statistics/paired\\_primary\\_effects.csv}.")
    tex = r"""\begin{table*}[t]
\centering
\caption{""" + cap + r"""}
\label{""" + label + r"""}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll lrlrr lrlrr}
\toprule
& & \multicolumn{5}{c}{$\Delta \mathrm{IGD}^+$} & \multicolumn{5}{c}{$\Delta$ hypervolume ratio} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
Dataset & Comparison & mean [95\% CI] & median & W/T/L & $r_{rb}$ & $p_{\mathrm{NB,Holm}}$ & mean [95\% CI] & median & W/T/L & $r_{rb}$ & $p_{\mathrm{NB,Holm}}$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (TAB / f"{name}.tex").write_text(tex)
    print("wrote", name)


def tab04_r10_r30():
    r = pd.read_csv(REP / "statistics" / "r10_vs_r30_stability.csv")
    rows = []
    for ds in DATASETS:
        first = True
        for s in ["nbi_A", "nbi_B", "nbi_C"]:
            cells = []
            for metric in ["hv_ratio", "igd_plus"]:
                x = r[(r.metric == metric) & (r.dataset == ds) & (r.set == s) & (r.cost == "weighted")].iloc[0]
                inside = x.r30_ci95_lo <= x.r10_estimate <= x.r30_ci95_hi
                nd = 3 if metric == "hv_ratio" else 4
                cells.append(f"{x.r10_estimate:.{nd}f} & {x.r30_estimate:.{nd}f} [{x.r30_ci95_lo:.{nd}f}, {x.r30_ci95_hi:.{nd}f}] & {'yes' if inside else 'no'}")
            rows.append(f"{DLABEL[ds] if first else ''} & {SETS[s]} & " + " & ".join(cells) + " \\\\")
            first = False
    body = "\n".join(rows)
    tex = r"""\begin{table*}[t]
\centering
\caption{Robustness from $R = 10$ to $R = 30$: mean hypervolume ratio and IGD$^+$ (weighted cost) of each NBI variant estimated from partitions 0--9 (the frozen $R = 10$ study, tag \texttt{pco213-postwork-r10}) and from all 30 partitions, with the $R = 30$ bootstrap interval and whether the $R = 10$ estimate lies inside it. Source: \texttt{statistics/r10\_vs\_r30\_stability.csv}.}
\label{tab:r10r30}
\small
\begin{tabular}{ll rll rll}
\toprule
& & \multicolumn{3}{c}{Hypervolume ratio} & \multicolumn{3}{c}{IGD$^+$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Dataset & Set & $R=10$ & $R=30$ [95\% CI] & inside & $R=10$ & $R=30$ [95\% CI] & inside \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (TAB / "tab04_r10_vs_r30.tex").write_text(tex)
    print("wrote tab04")


def tab05_compute():
    st = pd.read_csv(REP / "tables" / "stage_times.csv")
    nr = pd.read_csv(REP / "tables" / "nbi_runs.csv")
    rows = []
    for ds in DATASETS:
        s = st[st.dataset == ds].groupby("stage")["seconds"].mean()
        n = nr[nr.dataset == ds].groupby("variant")["n_real_objective_evals"].mean()
        succ = nr[nr.dataset == ds].groupby("variant").apply(lambda d: (d.n_success / d.n_subproblems).mean())
        rows.append(f"{DLABEL[ds]} & {s['oof']:.0f} & {s['refs']:.0f} & {s['reference']:.0f} & {s['nbi_A']:.0f} & {s['nbi_B']:.0f} & {s['nbi_C']:.0f} & "
                    f"{s['nbi_C']/s['nbi_B']:.0f}$\\times$ & {n['C']/1e3:.0f}k & {succ['A']:.2f} / {succ['B']:.2f} / {succ['C']:.2f} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table*}[t]
\centering
\caption{Mean wall-clock seconds per partition of the main stages (Apple M4 Max, 8 worker threads), the NBI-C/NBI-B time ratio, the mean number of real out-of-fold objective evaluations consumed by NBI-C (NBI-A and NBI-B consume none: they optimize the surrogates), and the mean fraction of the 66 NBI subproblems that terminated successfully (A/B: SLSQP-certified; C: equality-feasible under the lenient rule). Source: \texttt{tables/stage\_times.csv}, \texttt{tables/nbi\_runs.csv}.}
\label{tab:compute}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l rrr rrr r r l}
\toprule
Dataset & OOF fits & Single-obj. refs & Reference & NBI-A & NBI-B & NBI-C & C/B & Real evals (C) & Success A / B / C \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (TAB / "tab05_compute.tex").write_text(tex)
    print("wrote tab05")


def tab06_holdout_cost():
    """Supplementary-style compact table: holdout ranking agreement and cost-definition sensitivity."""
    h = pd.read_csv(REP / "statistics" / "holdout_transfer_r30.csv")
    c = pd.read_csv(REP / "statistics" / "cost_definition_sensitivity_r30.csv")

    def wilson(k, n, z=1.959964):
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return centre - half, centre + half

    rows = []
    for ds in DATASETS:
        cc = c[(c.dataset == ds) & (c.endpoint == "hv_ratio")].iloc[0]
        hh = h[(h.cost == "weighted") & (h.dataset == ds) & (h.metric == "ranking_agreement_auc")].iloc[0]
        lo, hi = wilson(hh.wins, hh.n)
        htxt = f"{int(hh.wins)}/30 [{lo:.2f}, {hi:.2f}]"
        rows.append(f"{DLABEL[ds]} & {htxt} & {int(cc.best_set_differs_count)}/30 [{cc.wilson_lo:.2f}, {cc.wilson_hi:.2f}] & {cc.median_rank_spearman_weighted_vs_support:.2f} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
\centering
\caption{Holdout ranking agreement (partitions in which the OOF-best knee pick among NBI-A/B/C and scalarization is also holdout-best, weighted cost) and cost-definition sensitivity (partitions in which the hypervolume-best set differs between the weighted and the support cost; median Spearman correlation of the set rankings). Wilson 95\% intervals. Sources: \texttt{statistics/proportion\_intervals.csv}, \texttt{statistics/cost\_definition\_sensitivity\_r30.csv}.}
\label{tab:holdout_cost}
\small
\begin{tabular}{l l l r}
\toprule
Dataset & Holdout ranking agreement & Best set differs (W vs S) & Rank $\rho$ (W vs S) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "tab06_holdout_cost.tex").write_text(tex)
    print("wrote tab06")


if __name__ == "__main__":
    fig01_pipeline()
    fig02_surrogate_validation()
    fig03_fronts()
    fig04_paired()
    fig05_gate()
    fig06_r10_r30()
    fig07_beta_synergy()
    fig08_cost_definition()
    tab01_datasets()
    tab02_gate()
    tab03_paired()
    tab03_paired(cost="support", name="tabS01_paired_support", label="tab:paired_support")
    tab04_r10_r30()
    tab05_compute()
    tab06_holdout_cost()


# --------------------------------------------------------------------------------------
# Table 7 / analysis: the classical edge condition beta_ij > |beta_i - beta_j|
#
# On the binary edge i-j of a quadratic Scheffe model,
#   yhat(t) = beta_i t + beta_j (1-t) + beta_ij t(1-t),
# the stationary point t* = (d + b)/(2b), with d = beta_i - beta_j and b = beta_ij,
# lies strictly inside (0,1) iff b > |d|, and there yhat(t*) = beta_j + (d+b)^2/(4b),
# which strictly exceeds max(beta_i, beta_j) whenever b > |d|.
# So the FITTED surface predicts a blend beating both pure components iff beta_ij > |beta_i - beta_j|.
# This is the classical synergism criterion (departure from linear blending; Scheffe 1958, Cornell 2002),
# stated against the chord rather than against the better vertex.
# --------------------------------------------------------------------------------------
def tab07_edge_condition():
    rows_pair = []
    for ds in DATASETS:
        for rep in range(30):
            s = json.load(open(EXP / ds / f"rep_{rep:02d}" / "scheffe.json"))["roc_auc"]
            o = s["orders"]["quadratic"]
            co = dict(zip(o["terms"], o["coefficients"]))
            for a in range(5):
                for b in range(a + 1, 5):
                    mi, mj = MODELS[a], MODELS[b]
                    k = f"w_{mi}*w_{mj}" if f"w_{mi}*w_{mj}" in co else f"w_{mj}*w_{mi}"
                    bi, bj, bij = co[f"w_{mi}"], co[f"w_{mj}"], co[k]
                    rows_pair.append({"dataset": ds, "rep": rep, "term": f"{mi}*{mj}",
                                      "beta_ij": bij, "vertex_gap_fitted": abs(bi - bj),
                                      "predicts_interior_optimum": bij > abs(bi - bj)})
    pf = pd.DataFrame(rows_pair)
    pf.to_csv(TAB / "edge_condition_per_replication.csv", index=False)

    cs = pd.read_csv(REP / "statistics" / "coefficient_stability_r30.csv")
    cs = cs[(cs.kind == "interaction") & (cs.response == "roc_auc")]

    lines, summary = [], []
    for ds in DATASETS:
        sub = cs[cs.dataset == ds]
        top = sub.sort_values("mean", ascending=False).iloc[0]
        t = pf[(pf.dataset == ds) & (pf.term == top.term)]
        agree = fp = fn = 0
        for _, r in sub.iterrows():
            p = pf[(pf.dataset == ds) & (pf.term == r.term)].predicts_interior_optimum.mean() > 0.5
            real = r.blend50_beats_best_member_freq > 0.5
            agree += (p == real); fp += (p and not real); fn += (real and not p)
        summary.append({"dataset": ds, "top_pair": top.term, "beta": top["mean"],
                        "gap": t.vertex_gap_fitted.mean(),
                        "pred_k": int(t.predicts_interior_optimum.sum()),
                        "real_freq": top.blend50_beats_best_member_freq,
                        "agree": agree, "fp": fp, "fn": fn})
        name = top.term.replace("*", "$\\cdot$").replace("gnb", "GNB").replace("knn", "kNN") \
                       .replace("xgb", "XGB").replace("rf", "RF").replace("lr", "LR")
        lines.append(f"{DLABEL[ds]} & {name} & {top['mean']:.3f} & {t.vertex_gap_fitted.mean():.3f} & "
                     f"{int(t.predicts_interior_optimum.sum())}/30 & "
                     f"{int(round(top.blend50_beats_best_member_freq * 30))}/30 & "
                     f"{agree}/10 & {fp} & {fn} \\\\")
    pd.DataFrame(summary).to_csv(TAB / "edge_condition_summary.csv", index=False)

    tex = r"""\begin{table}[t]
\centering
\caption{The classical synergism criterion applied correctly, and what the real objectives say. On the binary edge
$i$--$j$ of a quadratic Scheff\'e model the fitted surface has an interior optimum exceeding both pure components if
and only if $\hat\beta_{ij} > |\hat\beta_i - \hat\beta_j|$. Columns 3--5 evaluate that criterion for the
largest-$\hat\beta$ pair of each dataset; column 6 gives the partitions in which the \emph{real} out-of-fold 50/50
blend of the same pair beats its better member. The last three columns compare criterion and reality across all ten
pairs: agreements, cases where the surface predicts a superior blend that does not exist, and the reverse. The error
is systematic and one-directional. Source: \texttt{tables/edge\_condition\_summary.csv}.}
\label{tab:edge}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll rr cc ccc}
\toprule
& & \multicolumn{3}{c}{Largest-$\hat\beta$ pair, fitted surface} & Real blend & \multicolumn{3}{c}{All 10 pairs} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}
Dataset & Pair & $\hat\beta_{ij}$ & $|\hat\beta_i - \hat\beta_j|$ & criterion met & beats better member & agree & surface over-predicts & under-predicts \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "tab07_edge_condition.tex").write_text(tex)
    print("wrote tab07")


if __name__ == "__main__":
    tab07_edge_condition()


# --------------------------------------------------------------------------------------
# Post-processing: keep wide tables inside the text block.
# Several tables are wider than \textwidth in the single-column layout and would be
# clipped. Wrapping the tabular in \resizebox is the least invasive fix and keeps the
# generated files self-contained.
# --------------------------------------------------------------------------------------
WIDE = ["tab01_datasets", "tab02_gate", "tab03_paired_primary", "tabS01_paired_support",
        "tab04_r10_vs_r30", "tab05_compute", "tab07_edge_condition"]


def fit_wide_tables():
    for name in WIDE:
        f = TAB / f"{name}.tex"
        t = f.read_text()
        if "resizebox" in t:
            continue
        t = t.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        i = t.rfind("\\end{tabular}")
        t = t[:i] + "\\end{tabular}}" + t[i + len("\\end{tabular}"):]
        f.write_text(t)
        print("fitted", name)


def shorten_tab01():
    """The Source column of Table 1 overflows; shorten it and move the Porto note to the caption."""
    f = TAB / "tab01_datasets.tex"
    t = f.read_text()
    for a, b in [
        ("Santander Customer Transaction Prediction (Kaggle, 2019)", "Santander Customer Transaction (Kaggle 2019)"),
        ("BNP Paribas Cardif Claims Management (Kaggle, 2016)", "BNP Paribas Cardif Claims (Kaggle 2016)"),
        ("Porto Seguro Safe Driver Prediction (Kaggle, 2017); 200,000-row stratified subsample",
         "Porto Seguro Safe Driver (Kaggle 2017)$^{\\dagger}$"),
        ("Default of credit card clients (UCI 350; Yeh and Lien, 2009)", "Default of credit card clients (UCI 350)"),
        ("all selection is done on 5-fold out-of-fold predictions within the training part.",
         "all selection uses 5-fold out-of-fold predictions within the training part. "
         "$^{\\dagger}$200,000-row stratified subsample of the 595,212 available rows, drawn once with a recorded seed."),
    ]:
        t = t.replace(a, b)
    f.write_text(t)
    print("shortened tab01")


if __name__ == "__main__":
    shorten_tab01()
    fit_wide_tables()


# --------------------------------------------------------------------------------------
# Table 8: the evaluation-matched NSGA-II baseline (one compact main-text block)
# --------------------------------------------------------------------------------------
NSGA = REP / "nsga2"


def tab08_nsga2():
    e = pd.read_csv(NSGA / "nsga2_paired_effects.csv")
    t = pd.read_csv(NSGA / "nsga2_paired_tests.csv")
    lv = pd.read_csv(NSGA / "nsga2_indicator_levels.csv")
    br = pd.read_csv(NSGA / "nsga2_budget_runtime.csv").set_index("dataset")
    rows = []
    for ds in DATASETS:
        cells = []
        for ep in ["igd_plus", "hv_ratio"]:
            r = e[(e.reference == "augmented_union") & (e.cost == "weighted") &
                  (e.comparison == "nsga2 vs nbi_C") & (e.dataset == ds) & (e.endpoint == ep)].iloc[0]
            tt = t[(t.reference == "augmented_union") & (t.cost == "weighted") &
                   (t.comparison == "nsga2 vs nbi_C") & (t.dataset == ds) & (t.endpoint == ep)]
            p = tt.iloc[0].p_nb_holm_family_dataset if len(tt) else float("nan")
            ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
            cells.append(f"{r['median']:+.4f} [{r.ci95_median_lo:+.4f}, {r.ci95_median_hi:+.4f}] & "
                         f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {ptxt}")
        hv_c = lv[(lv.reference == "augmented_union") & (lv.cost == "weighted") &
                  (lv.dataset == ds) & (lv.set == "nbi_C")].hv_ratio.iloc[0]
        hv_n = lv[(lv.reference == "augmented_union") & (lv.cost == "weighted") &
                  (lv.dataset == ds) & (lv.set == "nsga2")].hv_ratio.iloc[0]
        b = br.loc[ds]
        rows.append(f"{DLABEL[ds]} & {hv_c:.3f} & {hv_n:.3f} & " + " & ".join(cells) +
                    f" & {b.time_ratio_median:.1f}$\\times$ \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table*}[t]
\centering
\caption{Evaluation-matched NSGA-II against metamodel-free NBI (NBI-C), weighted cost, scored against the common
augmented reference. Both optimize the same three real out-of-fold objectives and consume a matched number of real
objective evaluations per replication (realized ratio 0.999996 overall, 0.99992--1.00010 per run). $\Delta$ is the
paired difference signed so that $\Delta > 0$ favours NSGA-II, with its percentile-bootstrap 95\% interval, the
win/tie/loss count over 30 partitions and the Holm-corrected Nadeau--Bengio $p$. The last column is the wall-clock
ratio, which is \emph{not} matched. The direction is the same under the sample-core reference
(supplementary Table~S2). Source: \texttt{reports/pco213\_postwork\_benchmark/nsga2/}.}
\label{tab:nsga2}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l rr lcc lcc c}
\toprule
& \multicolumn{2}{c}{Median HV ratio} & \multicolumn{3}{c}{$\Delta \mathrm{IGD}^+$} & \multicolumn{3}{c}{$\Delta$ HV ratio} & Time \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Dataset & NBI-C & NSGA-II & median [95\% CI] & W/T/L & $p$ & median [95\% CI] & W/T/L & $p$ & ratio \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    (TAB / "tab08_nsga2.tex").write_text(tex)

    # supplementary: the sample-core repetition
    rows = []
    for ds in DATASETS:
        cells = []
        for ep in ["igd_plus", "hv_ratio"]:
            r = e[(e.reference == "sample_core") & (e.cost == "weighted") &
                  (e.comparison == "nsga2 vs nbi_C") & (e.dataset == ds) & (e.endpoint == ep)].iloc[0]
            cells.append(f"{r['median']:+.4f} [{r.ci95_median_lo:+.4f}, {r.ci95_median_hi:+.4f}] & "
                         f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}")
        sup = []
        for ep in ["igd_plus", "hv_ratio"]:
            r = e[(e.reference == "augmented_union") & (e.cost == "support") &
                  (e.comparison == "nsga2 vs nbi_C") & (e.dataset == ds) & (e.endpoint == ep)].iloc[0]
            sup.append(f"{r['median']:+.4f} & {int(r.wins)}/{int(r.ties)}/{int(r.losses)}")
        rows.append(f"{DLABEL[ds]} & " + " & ".join(cells) + " & " + " & ".join(sup) + r" \\")
    tex2 = r"""\begin{table*}[t]
\centering
\caption{NSGA-II versus NBI-C under the sample-core reference (left), which contains the search output of no optimizer,
and under the support cost scored post hoc (right). The direction matches Table~\ref{tab:nsga2} in every cell except
Santander IGD$^+$ under the support cost. Source: \texttt{nsga2\_paired\_effects.csv}.}
\label{tab:nsga2_supp}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{l lc lc lc lc}
\toprule
& \multicolumn{4}{c}{Sample-core reference, weighted cost} & \multicolumn{4}{c}{Augmented reference, support cost} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Dataset & $\Delta \mathrm{IGD}^+$ [CI] & W/T/L & $\Delta$ HV [CI] & W/T/L & $\Delta \mathrm{IGD}^+$ & W/T/L & $\Delta$ HV & W/T/L \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    (TAB / "tabS02_nsga2_sensitivity.tex").write_text(tex2)
    print("wrote tab08 + tabS02")


if __name__ == "__main__":
    tab08_nsga2()


# --------------------------------------------------------------------------------------
# Table 3, main-text version: load-bearing quantities only.
# Mean, rank-biserial, Wilcoxon and the two floor-check comparisons move to the supplement.
# --------------------------------------------------------------------------------------
def tab03_main_compact():
    e = pd.read_csv(REP / "statistics" / "paired_primary_effects.csv")
    t = pd.read_csv(REP / "statistics" / "paired_primary_tests.csv")
    e = e[e.cost == "weighted"]
    t = t[t.cost == "weighted"]
    comps = [("nbi_B vs nbi_A", "NBI-B vs NBI-A"), ("nbi_C vs nbi_B", "NBI-C vs NBI-B")]
    rows = []
    for ds in DATASETS:
        first = True
        for ck, cl in comps:
            cells = []
            for ep in ["igd_plus", "hv_ratio"]:
                r = e[(e.dataset == ds) & (e.comparison == ck) & (e.endpoint == ep)].iloc[0]
                p = t[(t.dataset == ds) & (t.comparison == ck) & (t.endpoint == ep)].iloc[0]
                nd = 4 if abs(r["median"]) < 0.05 else 3
                ptxt = "$<$0.01" if p.p_nb_holm_family_dataset < 0.01 else (
                    "$<$0.05" if p.p_nb_holm_family_dataset < 0.05 else "n.s.")
                cells.append(f"{r['median']:+.{nd}f} [{r.ci95_median_lo:+.{nd}f}, {r.ci95_median_hi:+.{nd}f}] & "
                             f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {ptxt}")
            rows.append(f"{DLABEL[ds] if first else ''} & {cl} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\addlinespace")
    body = "\n".join(rows[:-1])
    tex = r"""\begin{table*}[t]
\centering
\caption{Primary paired comparisons under the weighted cost, 30 partitions per dataset. $\Delta > 0$ favours the
second-named set ($\Delta\mathrm{IGD}^+ = \mathrm{IGD}^+_{\mathrm{ref}} - \mathrm{IGD}^+_{\mathrm{new}}$,
$\Delta\mathrm{HV} = \mathrm{HV}_{\mathrm{new}} - \mathrm{HV}_{\mathrm{ref}}$). Median with its percentile-bootstrap
95\% interval, wins/ties/losses, and the Holm-corrected Nadeau--Bengio significance band ($\rho = 0.25$, family of
eight tests per dataset; \emph{n.s.} means the corrected test does not reach 0.05, which for a bimodal or heavy-tailed
difference is expected and is why the median and the win count are reported). Means, rank-biserial effect sizes,
exact $p$-values, the two floor-check comparisons against random scalarization and random Dirichlet search, and the
full support-cost repetition are in supplementary Tables~S1 and~S5. Source:
\texttt{statistics/paired\_primary\_effects.csv}, \texttt{paired\_primary\_tests.csv}.}
\label{tab:paired}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll lcc lcc}
\toprule
& & \multicolumn{3}{c}{$\Delta \mathrm{IGD}^+$} & \multicolumn{3}{c}{$\Delta$ hypervolume ratio} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Dataset & Comparison & median [95\% CI] & W/T/L & Holm $p$ & median [95\% CI] & W/T/L & Holm $p$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    (TAB / "tab03_paired_primary.tex").write_text(tex)
    print("wrote compact tab03 (main); full version remains in statistics CSVs")


if __name__ == "__main__":
    tab03_main_compact()
