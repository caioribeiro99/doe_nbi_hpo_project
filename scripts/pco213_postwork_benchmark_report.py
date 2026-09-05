#!/usr/bin/env python3
"""Aggregate the PCO213 post-work benchmark (all datasets x replications) into
machine-readable tables, figures and a numeric basis for the cross-dataset
questions Q1-Q12.

Reads  <root>/<dataset>/rep_XX/*  (see pco213_run_postwork_benchmark.py)
Writes <reports>/tables/*.csv, <reports>/summary.json, <reports>/README.md
       <figures>/*.png
Only completed replications (all stages 'done') enter the aggregates.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

STAGES = ["oof", "design", "scheffe", "refs", "reference", "nbi_A", "nbi_B", "nbi_C", "comparators", "quality"]
SETS = ["nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization", "random_dirichlet_budget", "design_runs", "single_objective_refs"]
SET_LABELS = {"nbi_A": "NBI-A (surrogate anchors)", "nbi_B": "NBI-B (real anchors)", "nbi_C": "NBI-C (metamodel-free)",
              "ws_random_scalarization": "random scalarization", "random_dirichlet_budget": "random Dirichlet (66)",
              "design_runs": "DoE runs (66)", "single_objective_refs": "single-objective refs"}
MODELS = ["lr", "gnb", "knn", "rf", "xgb"]
RESPONSES = ["roc_auc", "log_loss", "brier", "pr_auc"]


def rj(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def completed_reps(root: Path, ds: str) -> list[int]:
    out = []
    for d in sorted((root / ds).glob("rep_*")):
        st = rj(d / "stage_status.json", {})
        if all(st.get(s, {}).get("status") == "done" for s in STAGES):
            out.append(int(d.name.split("_")[1]))
    return out


def agg_table(df: pd.DataFrame, group: str, cols: list[str]) -> dict:
    """{group value: {col__mean, col__std}} with flat string keys (JSON-safe)."""
    if df.empty:
        return {}
    g = df.groupby(group)[cols].agg(["mean", "std"])
    g.columns = [f"{c}__{s}" for c, s in g.columns]
    return {str(k): {kk: (None if pd.isna(vv) else round(float(vv), 6)) for kk, vv in v.items()} for k, v in g.to_dict("index").items()}


def stats(x) -> dict:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "experiments" / "pco213_postwork_benchmark"))
    ap.add_argument("--figures", default=str(REPO / "figures" / "pco213_postwork_benchmark"))
    ap.add_argument("--reports", default=str(REPO / "reports" / "pco213_postwork_benchmark"))
    args = ap.parse_args()
    root, figdir, repdir = Path(args.root), Path(args.figures), Path(args.reports)
    figdir.mkdir(parents=True, exist_ok=True); (repdir / "tables").mkdir(parents=True, exist_ok=True)
    manifest = rj(root / "benchmark_manifest.json", {})
    datasets = [d for d in manifest.get("datasets", []) if (root / d).exists()]
    reps = {d: completed_reps(root, d) for d in datasets}
    print("completed replications:", reps)

    # ------------------------------------------------------------------ long tables
    rows_model, rows_cost, rows_sch, rows_beta, rows_refs, rows_refdiag, rows_nbi, rows_q, rows_face, rows_div, rows_pick = ([] for _ in range(11))
    rows_time = []
    for ds in datasets:
        for r in reps[ds]:
            d = root / ds / f"rep_{r:02d}"
            st = rj(d / "stage_status.json"); meta = rj(d / "oof_meta.json"); sch = rj(d / "scheffe.json")
            refs = rj(d / "references.json"); rdiag = rj(d / "reference_diagnostics.json"); q = rj(d / "quality.json")
            for s in STAGES:
                rows_time.append({"dataset": ds, "rep": r, "stage": s, "seconds": st[s]["seconds"], "attempts": st[s]["attempts"]})
            for m, perf in meta["model_performance"].items():
                rows_model.append({"dataset": ds, "rep": r, "model": m, **perf,
                                   "fit_seconds": meta["fit_seconds"][m], "cost_ms_per_1k": meta["cost_ms_per_1k"][m]})
            for m in MODELS:
                rows_cost.append({"dataset": ds, "rep": r, "model": m, "cost_ms_per_1k": meta["cost_ms_per_1k"][m]})
            for resp in RESPONSES:
                c = sch[resp]
                for order, fit in c["orders"].items():
                    if not fit.get("estimable"):
                        continue
                    rows_sch.append({"dataset": ds, "rep": r, "response": resp, "order": order,
                                     "selected": order == c["selected_order"], "reliable": c["reliable"] if order == c["selected_order"] else None,
                                     "r2_train": fit["r2_train"], "r2_external": fit["external"]["r2_external"],
                                     "rmse_external": fit["external"]["rmse"], "rmse_rel_range": fit["external"]["rmse_relative_to_range"],
                                     "spearman_external": fit["spearman_external"], "condition_number": fit["condition_number"],
                                     "rank": fit["rank"], "n_terms": fit["n_terms"], "extrapolation_excess": fit["extrapolation_excess"]})
                if "quadratic_beta" in c:
                    for term, val in c["quadratic_beta"].items():
                        rows_beta.append({"dataset": ds, "rep": r, "response": resp, "term": term.replace("w_", ""),
                                          "kind": "interaction" if "*" in term else "linear", "beta": val})
            for k, v in refs["references"].items():
                rows_refs.append({"dataset": ds, "rep": r, "method": k,
                                  **{kk: vv for kk, vv in v.items() if kk not in ("detail", "w")},
                                  **({f"w_{MODELS[j]}": v["w"][j] for j in range(5)} if "w" in v else {})})
            rows_refdiag.append({"dataset": ds, "rep": r, "n_points": rdiag["n_points"], "rounds": rdiag["rounds"],
                                 "front_weighted_size": rdiag["front_weighted_size"], "front_support_size": rdiag["front_support_size"],
                                 "displaced_fraction": rdiag["independent_check"]["front_displaced_fraction"],
                                 "prefix_displaced_last": rdiag["prefix_convergence_random_part"][-1]["prev_front_displaced_fraction"],
                                 "n_eps_constraint": rdiag["n_eps_constraint_solutions"], "seconds": rdiag["seconds"],
                                 **{f"src_{k}": v for k, v in rdiag["front_weighted_sources"].items()}})
            for v in ("A", "B", "C"):
                s = rj(d / f"nbi_{v}_summary.json")
                rows_nbi.append({"dataset": ds, "rep": r, "variant": v, "n_subproblems": s["n_subproblems"], "n_success": s["n_success"],
                                 "n_front_real_weighted": s["n_front_real_weighted"], "n_front_real_support": s["n_front_real_support"],
                                 "total_nfev": s["total_nfev"], "n_real_objective_evals": s["n_real_objective_evals"], "seconds": s["seconds"],
                                 "anchor_auc_real_auc": s["anchors_real_metrics"][0]["roc_auc"], "anchor_auc_cost": s["anchors_real_metrics"][0]["cost_weighted"],
                                 "anchor_ll_real_ll": s["anchors_real_metrics"][1]["log_loss"], "anchor_ll_cost": s["anchors_real_metrics"][1]["cost_weighted"],
                                 "surrogate_auc_reliable": s["surrogate_reliable"]["roc_auc"], "surrogate_ll_reliable": s["surrogate_reliable"]["log_loss"]})
            for tag in ("weighted", "support"):
                blk = q[tag]
                for key, qq in blk["sets"].items():
                    rows_q.append({"dataset": ds, "rep": r, "cost": tag, "set": key,
                                   **{k: v for k, v in qq.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                                   "extreme_gap_auc": qq.get("extreme_gap", [np.nan] * 3)[0], "extreme_gap_ll": qq.get("extreme_gap", [np.nan] * 3)[1],
                                   "extreme_gap_cost": qq.get("extreme_gap", [np.nan] * 3)[2],
                                   **{f"mw_{m}": qq.get("front_mean_weights", {}).get(m, np.nan) for m in MODELS}})
                    for rule, pk in qq.get("mcdm_picks", {}).items():
                        rows_pick.append({"dataset": ds, "rep": r, "cost": tag, "set": key, "rule": rule,
                                          **{k: v for k, v in pk.items() if k != "w"}, **{f"w_{MODELS[j]}": pk["w"][j] for j in range(5)}})
                for sup, cnt in blk["front_support_distribution"].items():
                    rows_face.append({"dataset": ds, "rep": r, "cost": tag, "set": "empirical_reference", "support": sup, "count": cnt,
                                      "n_front": blk["n_front"]})
                for key, qq in blk["sets"].items():
                    for sup, cnt in qq.get("front_support_distribution", {}).items():
                        rows_face.append({"dataset": ds, "rep": r, "cost": tag, "set": key, "support": sup, "count": cnt, "n_front": qq.get("n_front")})
            # diversity vs beta_ij (Q11)
            ec = np.array(meta["error_correlation"]); dis = np.array(meta["disagreement_rate"])
            for resp in ("roc_auc", "log_loss"):
                qb = sch[resp].get("quadratic_beta", {})
                for i, j in combinations(range(5), 2):
                    term = f"w_{MODELS[i]}*w_{MODELS[j]}"
                    if term in qb:
                        rows_div.append({"dataset": ds, "rep": r, "response": resp, "pair": f"{MODELS[i]}-{MODELS[j]}",
                                         "beta_ij": qb[term], "error_correlation": ec[i, j], "disagreement": dis[i, j]})

    T = {"stage_times": pd.DataFrame(rows_time), "model_performance": pd.DataFrame(rows_model), "inference_costs": pd.DataFrame(rows_cost),
         "scheffe_orders": pd.DataFrame(rows_sch), "scheffe_quadratic_beta": pd.DataFrame(rows_beta), "single_objective_refs": pd.DataFrame(rows_refs),
         "reference_diagnostics": pd.DataFrame(rows_refdiag), "nbi_runs": pd.DataFrame(rows_nbi), "pareto_quality": pd.DataFrame(rows_q),
         "front_supports": pd.DataFrame(rows_face), "diversity_vs_beta": pd.DataFrame(rows_div), "mcdm_picks_holdout": pd.DataFrame(rows_pick)}
    for k, df in T.items():
        df.to_csv(repdir / "tables" / f"{k}.csv", index=False)

    # ------------------------------------------------------------------ summaries
    S: dict = {"completed_replications": reps, "git_commit": manifest.get("git_commit"), "effective_reps": manifest.get("effective_reps"),
               "decisions": manifest.get("decisions", []), "environment": manifest.get("environment")}
    st = T["stage_times"]
    S["runtime"] = {"total_hours": float(st["seconds"].sum() / 3600), "per_dataset_hours": st.groupby("dataset")["seconds"].sum().div(3600).round(3).to_dict(),
                    "per_stage_hours": st.groupby("stage")["seconds"].sum().div(3600).round(3).to_dict(),
                    "retried_stages": int((st["attempts"] > 1).sum())}
    S["counts"] = {"model_fits": int(sum(rj(root / ds / f"rep_{r:02d}" / "oof_meta.json")["n_fits"] for ds in datasets for r in reps[ds])),
                   "doe_evaluations": int(sum(66 + 100 for ds in datasets for r in reps[ds])),
                   "nbi_subproblems": int(T["nbi_runs"]["n_subproblems"].sum()),
                   "nbi_real_objective_evals_variant_C": int(T["nbi_runs"].query("variant=='C'")["n_real_objective_evals"].sum()),
                   "reference_points": int(T["reference_diagnostics"]["n_points"].sum()),
                   "direct_auc_search_evals": int(sum(rj(root / ds / f"rep_{r:02d}" / "references.json")["counts"]["direct_auc_evals"] for ds in datasets for r in reps[ds]))}
    mp = T["model_performance"]
    S["model_performance"] = {ds: agg_table(mp[mp.dataset == ds], "model", ["oof_roc_auc", "oof_log_loss", "holdout_roc_auc", "holdout_log_loss", "cost_ms_per_1k"])
                              for ds in datasets}
    so = T["scheffe_orders"]
    S["scheffe"] = {}
    for ds in datasets:
        S["scheffe"][ds] = {}
        for resp in RESPONSES:
            sub = so[(so.dataset == ds) & (so.response == resp)]
            sel = sub[sub.selected]
            S["scheffe"][ds][resp] = {"selected_order_freq": sel["order"].value_counts().to_dict(),
                                      "reliable_fraction": float(sel["reliable"].mean()) if len(sel) else np.nan,
                                      "r2_external_selected": stats(sel["r2_external"]), "rmse_rel_range_selected": stats(sel["rmse_rel_range"]),
                                      "spearman_selected": stats(sel["spearman_external"]),
                                      "by_order": {o: {"r2_external": stats(sub[sub.order == o]["r2_external"]), "rmse_external": stats(sub[sub.order == o]["rmse_external"])}
                                                   for o in sub["order"].unique()}}
    sb = T["scheffe_quadratic_beta"]
    S["beta_stability"] = {}
    for ds in datasets:
        S["beta_stability"][ds] = {}
        for resp in ("roc_auc", "log_loss"):
            sub = sb[(sb.dataset == ds) & (sb.response == resp)]
            g = sub.groupby("term")["beta"]
            S["beta_stability"][ds][resp] = {t: {**stats(v), "sign_positive_freq": float((v > 0).mean())} for t, v in g}
    rq = T["pareto_quality"]
    S["pareto_quality"] = {}
    for tag in ("weighted", "support"):
        S["pareto_quality"][tag] = {}
        for ds in datasets:
            S["pareto_quality"][tag][ds] = {}
            for key in SETS:
                sub = rq[(rq.dataset == ds) & (rq.cost == tag) & (rq.set == key)]
                S["pareto_quality"][tag][ds][key] = {m: stats(sub[m]) for m in
                                                     ("gd_front", "gd_all_valid", "igd", "igd_plus", "spacing", "spacing_cv", "hv_ratio",
                                                      "joint_nondominated_fraction_all_valid", "coverage_ref_over_set", "coverage_set_over_ref",
                                                      "spacing_size_matched_percentile", "n_front", "n_valid") if m in sub}
    nb = T["nbi_runs"]
    S["nbi"] = {ds: {v: {"success_rate": stats(sub["n_success"] / sub["n_subproblems"]), "n_front_weighted": stats(sub["n_front_real_weighted"]),
                        "seconds": stats(sub["seconds"]), "anchor_auc_cost": stats(sub["anchor_auc_cost"]), "anchor_ll_cost": stats(sub["anchor_ll_cost"])}
                     for v, sub in nb[nb.dataset == ds].groupby("variant")} for ds in datasets}
    rd = T["reference_diagnostics"]
    S["reference"] = {ds: {"n_points": stats(rd[rd.dataset == ds]["n_points"]), "displaced_fraction": stats(rd[rd.dataset == ds]["displaced_fraction"]),
                           "front_weighted_size": stats(rd[rd.dataset == ds]["front_weighted_size"]), "front_support_size": stats(rd[rd.dataset == ds]["front_support_size"]),
                           "rounds": stats(rd[rd.dataset == ds]["rounds"])} for ds in datasets}
    rf = T["single_objective_refs"]
    S["references"] = {ds: agg_table(rf[rf.dataset == ds], "method", ["oof_roc_auc", "oof_log_loss", "holdout_roc_auc", "holdout_log_loss", "cost_weighted", "cost_support", "n_eff"])
                       for ds in datasets}
    # weights stability of single-objective optima
    S["weight_stability"] = {}
    for ds in datasets:
        S["weight_stability"][ds] = {}
        for meth in ("slsqp_direct_logloss", "direct_auc_search", "scheffe_optimum_logloss", "scheffe_optimum_auc"):
            sub = rf[(rf.dataset == ds) & (rf.method == meth)]
            S["weight_stability"][ds][meth] = {m: stats(sub[f"w_{m}"]) for m in MODELS}
    # face / support frequencies on the empirical reference front
    fs = T["front_supports"]
    S["support_frequency"] = {}
    for tag in ("weighted", "support"):
        S["support_frequency"][tag] = {}
        for ds in datasets:
            sub = fs[(fs.dataset == ds) & (fs.cost == tag) & (fs.set == "empirical_reference")]
            tot = sub.groupby("support")["count"].sum().sort_values(ascending=False)
            S["support_frequency"][tag][ds] = (tot / tot.sum()).round(4).head(12).to_dict()
            # model activity: fraction of front points where model is active
            act = {}
            for m in MODELS:
                act[m] = float(sub[sub.support.str.split("+").apply(lambda s: m in s)]["count"].sum() / max(sub["count"].sum(), 1))
            S["support_frequency"][tag][ds + "__model_active_fraction"] = act
    # diversity vs beta (Q11)
    dv = T["diversity_vs_beta"]
    S["diversity_vs_beta"] = {}
    for resp in ("roc_auc", "log_loss"):
        sub = dv[dv.response == resp]
        S["diversity_vs_beta"][resp] = {"spearman_beta_vs_error_corr_all": float(spearmanr(sub["beta_ij"], sub["error_correlation"]).correlation) if len(sub) > 3 else np.nan,
                                        "spearman_beta_vs_disagreement_all": float(spearmanr(sub["beta_ij"], sub["disagreement"]).correlation) if len(sub) > 3 else np.nan,
                                        "per_dataset": {ds: float(spearmanr(sub[sub.dataset == ds]["beta_ij"], sub[sub.dataset == ds]["error_correlation"]).correlation)
                                                        if len(sub[sub.dataset == ds]) > 3 else np.nan for ds in datasets}}
    # Q4: AUC vs log-loss practical conflict: gap between direct AUC optimum and SLSQP optimum
    S["auc_vs_logloss_conflict"] = {}
    for ds in datasets:
        a = rf[(rf.dataset == ds) & (rf.method == "direct_auc_search")].set_index("rep")
        b = rf[(rf.dataset == ds) & (rf.method == "slsqp_direct_logloss")].set_index("rep")
        fold = mp[(mp.dataset == ds)].groupby("rep")[["oof_roc_auc"]].std()
        S["auc_vs_logloss_conflict"][ds] = {"delta_auc_directauc_minus_slsqp": stats(a["oof_roc_auc"] - b["oof_roc_auc"]),
                                             "delta_logloss_directauc_minus_slsqp": stats(a["oof_log_loss"] - b["oof_log_loss"]),
                                             "delta_cost_weighted": stats(a["cost_weighted"] - b["cost_weighted"]),
                                             "holdout_delta_auc": stats(a["holdout_roc_auc"] - b["holdout_roc_auc"]),
                                             "holdout_delta_logloss": stats(a["holdout_log_loss"] - b["holdout_log_loss"])}
    # Q8/Q9: paired differences B-A, C-B on IGD+ and HV ratio, per dataset (weighted cost)
    S["paired_variant_differences"] = {}
    for tag in ("weighted", "support"):
        S["paired_variant_differences"][tag] = {}
        for ds in datasets:
            piv = {k: rq[(rq.dataset == ds) & (rq.cost == tag) & (rq.set == k)].set_index("rep") for k in ("nbi_A", "nbi_B", "nbi_C", "random_dirichlet_budget", "ws_random_scalarization")}
            out = {}
            for a_, b_ in (("nbi_B", "nbi_A"), ("nbi_C", "nbi_B"), ("nbi_C", "nbi_A"), ("nbi_B", "random_dirichlet_budget"), ("nbi_B", "ws_random_scalarization")):
                for metric in ("igd_plus", "hv_ratio", "joint_nondominated_fraction_all_valid", "spacing_cv"):
                    if metric in piv[a_] and metric in piv[b_]:
                        dd = (piv[a_][metric] - piv[b_][metric]).dropna()
                        out[f"{a_}_minus_{b_}__{metric}"] = {**stats(dd), "frac_positive": float((dd > 0).mean()) if len(dd) else np.nan}
            S["paired_variant_differences"][tag][ds] = out
    S["support_vs_weighted_agreement"] = {}
    for ds in datasets:
        w = rq[(rq.dataset == ds) & (rq.cost == "weighted") & (rq.set == "nbi_B")].set_index("rep")
        s_ = rq[(rq.dataset == ds) & (rq.cost == "support") & (rq.set == "nbi_B")].set_index("rep")
        S["support_vs_weighted_agreement"][ds] = {"nbi_B_igd_plus_weighted": stats(w["igd_plus"]), "nbi_B_igd_plus_support": stats(s_["igd_plus"]),
                                                  "nbi_B_joint_nd_weighted": stats(w["joint_nondominated_fraction_all_valid"]),
                                                  "nbi_B_joint_nd_support": stats(s_["joint_nondominated_fraction_all_valid"])}
    (repdir / "summary.json").write_text(json.dumps(S, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))

    # ------------------------------------------------------------------ figures
    make_figures(root, datasets, reps, T, S, figdir)
    write_readme(repdir, datasets, reps, T, S)
    print("report written to", repdir, "figures to", figdir)


def make_figures(root: Path, datasets, reps, T, S, figdir: Path) -> None:
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    # 1. methodology diagram
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
    steps = ["Outer replication\n(R x stratified 80/20)", "5-fold OOF\n5-model zoo", "66-run mixture DoE\n+100 Dirichlet validation",
             "Scheffé RSM\nlin/quad/sp.cubic\n+ reliability gate", "Single-objective refs\nSLSQP LL, direct AUC", "NBI A / B / C\n(66 betas)",
             "Real-OOF revalidation\n+ Pareto filter", "Empirical Pareto reference\n(>=100k, convergence)", "Quality + stability\nGD/IGD/IGD+/HV, faces"]
    for i, s in enumerate(steps):
        ax.text(i / (len(steps) - 1), 0.5, s, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eef3fb", ec="#4a6fa5"), transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=((i + 0.62) / (len(steps) - 1), 0.5), xytext=((i + 0.38) / (len(steps) - 1), 0.5),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#4a6fa5"))
    fig.savefig(figdir / "fig01_methodology.png", bbox_inches="tight"); plt.close(fig)

    # 2. per-dataset empirical fronts + NBI A/B/C (rep 0), weighted cost; 3 projections
    for ds in datasets:
        if not reps[ds]:
            continue
        d = root / ds / f"rep_{reps[ds][0]:02d}"
        ref = pd.read_csv(d / "empirical_reference_front_weighted.csv")
        sets = {v: pd.read_csv(d / f"nbi_{v}_candidates.csv") for v in ("A", "B", "C")}
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        pairs = [("roc_auc", "log_loss"), ("roc_auc", "cost_weighted"), ("log_loss", "cost_weighted")]
        for ax, (xa, ya) in zip(axes, pairs):
            ax.scatter(ref[xa], ref[ya], s=6, c="lightgray", label="empirical reference front")
            for v, col in zip(("A", "B", "C"), ("tab:red", "tab:blue", "tab:green")):
                df = sets[v]; m = df["nd_real_weighted"].astype(bool)
                ax.scatter(df.loc[m, xa], df.loc[m, ya], s=18, c=col, label=f"NBI-{v} front ({int(m.sum())})", alpha=0.8)
            ax.set_xlabel(xa); ax.set_ylabel(ya)
            if ya.startswith("cost"):
                ax.set_yscale("log")
        axes[0].legend(fontsize=7); fig.suptitle(f"{ds}: empirical Pareto reference vs NBI variants (rep {reps[ds][0]}, weighted cost)")
        fig.tight_layout(); fig.savefig(figdir / f"fig02_fronts_{ds}.png", bbox_inches="tight"); plt.close(fig)
        # weighted vs support cost fronts
        refs_ = pd.read_csv(d / "empirical_reference_front_support.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].scatter(ref["cost_weighted"], ref["log_loss"], s=6, c="tab:gray"); axes[0].set_xscale("log"); axes[0].set_title("weighted (linear) cost")
        axes[0].set_xlabel("cost ms/1k"); axes[0].set_ylabel("log-loss")
        axes[1].scatter(refs_["cost_support"], refs_["log_loss"], s=10, c="tab:purple"); axes[1].set_xscale("log"); axes[1].set_title("support (deployment) cost")
        axes[1].set_xlabel("cost ms/1k")
        fig.suptitle(f"{ds}: empirical reference front under the two cost definitions"); fig.tight_layout()
        fig.savefig(figdir / f"fig03_cost_definitions_{ds}.png", bbox_inches="tight"); plt.close(fig)

    # 4. quality indicator distributions (IGD+, HV ratio, joint-ND, GD) per dataset x set
    rq = T["pareto_quality"]
    for metric, fname in (("igd_plus", "fig04_igd_plus"), ("hv_ratio", "fig05_hv_ratio"), ("joint_nondominated_fraction_all_valid", "fig06_joint_nd"),
                          ("gd_front", "fig07_gd"), ("igd", "fig08_igd"), ("spacing_cv", "fig09_spacing_cv")):
        fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            sub = rq[(rq.dataset == ds) & (rq.cost == "weighted")]
            data = [sub[sub.set == k][metric].dropna().to_numpy() for k in SETS[:6]]
            ax.boxplot(data, tick_labels=[k.replace("_", "\n") for k in SETS[:6]]); ax.set_title(ds); ax.tick_params(axis="x", labelsize=6)
        axes[0].set_ylabel(metric); fig.suptitle(f"{metric} across replications (weighted cost)"); fig.tight_layout()
        fig.savefig(figdir / f"{fname}.png", bbox_inches="tight"); plt.close(fig)

    # 10. beta_ij heatmaps by dataset (quadratic, AUC and log-loss), mean over reps
    sb = T["scheffe_quadratic_beta"]
    for resp in ("roc_auc", "log_loss"):
        fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.2))
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            sub = sb[(sb.dataset == ds) & (sb.response == resp) & (sb.kind == "interaction")]
            mat = np.full((5, 5), np.nan)
            for term, val in sub.groupby("term")["beta"].mean().items():
                a, b = term.split("*"); i, j = MODELS.index(a), MODELS.index(b); mat[i, j] = mat[j, i] = val
            vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xticklabels(MODELS); ax.set_yticklabels(MODELS); ax.set_title(f"{ds}")
            for i in range(5):
                for j in range(5):
                    if i != j and np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.suptitle(f"Scheffé quadratic interactions beta_ij ({resp}), mean over replications"); fig.tight_layout()
        fig.savefig(figdir / f"fig10_beta_ij_{resp}.png", bbox_inches="tight"); plt.close(fig)

    # 11. coefficient stability (box per term) for log-loss and AUC
    for resp in ("roc_auc", "log_loss"):
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            sub = sb[(sb.dataset == ds) & (sb.response == resp)]
            terms = list(sub["term"].unique())
            ax.boxplot([sub[sub.term == t]["beta"].to_numpy() for t in terms], tick_labels=terms); ax.tick_params(axis="x", rotation=90, labelsize=6)
            ax.axhline(0, color="k", lw=0.5); ax.set_title(ds)
        fig.suptitle(f"Scheffé quadratic coefficient stability across replications ({resp})"); fig.tight_layout()
        fig.savefig(figdir / f"fig11_coef_stability_{resp}.png", bbox_inches="tight"); plt.close(fig)

    # 12. Pareto weight composition + active-support frequency + N_eff
    fs = T["front_supports"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.4))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        act = S["support_frequency"]["weighted"].get(ds + "__model_active_fraction", {})
        ax.bar(list(act.keys()), list(act.values()), color="tab:blue"); ax.set_ylim(0, 1); ax.set_title(ds)
    axes[0].set_ylabel("fraction of reference-front points\nwith model active (w > 1e-3)")
    fig.suptitle("Active-support frequency on the empirical reference front (weighted cost)"); fig.tight_layout()
    fig.savefig(figdir / "fig12_active_support.png", bbox_inches="tight"); plt.close(fig)
    rq = T["pareto_quality"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = rq[(rq.dataset == ds) & (rq.cost == "weighted") & (rq.set.isin(["nbi_A", "nbi_B", "nbi_C"]))]
        mw = sub.groupby("set")[[f"mw_{m}" for m in MODELS]].mean()
        bottom = np.zeros(len(mw))
        for m in MODELS:
            ax.bar(mw.index, mw[f"mw_{m}"], bottom=bottom, label=m); bottom += mw[f"mw_{m}"].to_numpy()
        ax.set_title(ds); ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(fontsize=7); fig.suptitle("Mean weight composition of validated NBI fronts"); fig.tight_layout()
    fig.savefig(figdir / "fig13_weight_composition.png", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = rq[(rq.dataset == ds) & (rq.cost == "weighted")]
        ax.boxplot([sub[sub.set == k]["front_n_eff_median"].dropna().to_numpy() for k in SETS[:6]], tick_labels=[k[:8] for k in SETS[:6]])
        ax.set_title(ds); ax.tick_params(axis="x", labelsize=6)
    axes[0].set_ylabel("median N_eff of validated front"); fig.suptitle("Effective ensemble size on validated fronts"); fig.tight_layout()
    fig.savefig(figdir / "fig14_neff.png", bbox_inches="tight"); plt.close(fig)

    # 15. diversity vs beta_ij
    dv = T["diversity_vs_beta"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, resp in zip(axes, ("roc_auc", "log_loss")):
        sub = dv[dv.response == resp]
        for ds in datasets:
            s2 = sub[sub.dataset == ds]; ax.scatter(s2["error_correlation"], s2["beta_ij"], s=12, label=ds, alpha=0.7)
        ax.set_xlabel("pairwise error correlation (OOF)"); ax.set_ylabel(f"beta_ij ({resp})"); ax.axhline(0, color="k", lw=0.5)
        rho = S["diversity_vs_beta"][resp]["spearman_beta_vs_error_corr_all"]; ax.set_title(f"Spearman = {rho:.2f}")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(figdir / "fig15_diversity_vs_beta.png", bbox_inches="tight"); plt.close(fig)

    # 16. predicted vs real surrogate validation (rep 0, selected order)
    from mixens.scheffe import model_from_coefficients
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.4 * len(datasets), 6.4))
    axes = np.atleast_2d(axes)
    for c, ds in enumerate(datasets):
        if not reps[ds]:
            continue
        d = root / ds / f"rep_{reps[ds][0]:02d}"
        sch = json.loads((d / "scheffe.json").read_text()); z = np.load(d / "design_points.npz"); ve = pd.read_csv(d / "validation_eval.csv")
        for r_, resp in enumerate(("roc_auc", "log_loss")):
            sel = sch[resp]["selected_order"]; fit = sch[resp]["orders"][sel]
            m = model_from_coefficients(sch["component_names"], fit["terms"], fit["coefficients"])
            pred = m.predict_weights(z["W_val"]); ax = axes[r_, c]
            ax.scatter(ve[resp], pred, s=10); lo, hi = min(ve[resp].min(), pred.min()), max(ve[resp].max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.7); ax.set_xlabel(f"observed {resp}"); ax.set_ylabel("predicted")
            ax.set_title(f"{ds} {resp} [{sel}] R2ext={fit['external']['r2_external']:.2f}", fontsize=8)
    fig.suptitle("Scheffé surrogate: predicted vs observed on unseen Dirichlet points (rep 0)"); fig.tight_layout()
    fig.savefig(figdir / "fig16_surrogate_validation.png", bbox_inches="tight"); plt.close(fig)

    # 17. NBI vs dense reference: IGD+ A/B/C paired lines per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = rq[(rq.dataset == ds) & (rq.cost == "weighted") & (rq.set.isin(["nbi_A", "nbi_B", "nbi_C"]))]
        piv = sub.pivot(index="rep", columns="set", values="igd_plus")[["nbi_A", "nbi_B", "nbi_C"]]
        for _, row in piv.iterrows():
            ax.plot(["A", "B", "C"], row.to_numpy(), "-o", color="gray", alpha=0.6, ms=3)
        ax.plot(["A", "B", "C"], piv.mean().to_numpy(), "-o", color="tab:red", lw=2, label="mean"); ax.set_title(ds)
    axes[0].set_ylabel("IGD+ vs empirical reference"); axes[0].legend(); fig.suptitle("NBI variants: surrogate anchors (A) -> real anchors (B) -> metamodel-free (C)")
    fig.tight_layout(); fig.savefig(figdir / "fig17_nbi_variants_paired.png", bbox_inches="tight"); plt.close(fig)


def write_readme(repdir: Path, datasets, reps, T, S) -> None:
    lines = ["# PCO213 post-work benchmark — aggregated results", "",
             f"git commit: `{S['git_commit']}` · completed replications: {S['completed_replications']} · effective R: {S['effective_reps']}",
             f"total runtime (sum of stage times): {S['runtime']['total_hours']:.2f} h · per dataset (h): {S['runtime']['per_dataset_hours']}",
             f"counts: {S['counts']}", ""]
    for d in S["decisions"]:
        lines.append(f"- decision: {d}")
    lines += ["", "## Scheffé selected orders / reliability (selected surface, unseen Dirichlet points)", "",
              "| dataset | response | order freq | reliable frac | R2ext median | rel-RMSE median | Spearman median |", "|---|---|---|---|---|---|---|"]
    for ds in datasets:
        for resp in RESPONSES:
            s = S["scheffe"][ds][resp]
            lines.append(f"| {ds} | {resp} | {s['selected_order_freq']} | {s['reliable_fraction']:.2f} | {s['r2_external_selected'].get('median', float('nan')):.3f} | "
                         f"{s['rmse_rel_range_selected'].get('median', float('nan')):.3f} | {s['spearman_selected'].get('median', float('nan')):.3f} |")
    lines += ["", "## Pareto quality vs empirical reference (weighted cost; median over replications)", "",
              "| dataset | set | n_front | GD | IGD | IGD+ | HV ratio | joint-ND frac | spacing CV | size-matched spacing pct |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in datasets:
        for key in SETS:
            q = S["pareto_quality"]["weighted"][ds].get(key, {})
            g = lambda m: q.get(m, {}).get("median", float("nan"))
            lines.append(f"| {ds} | {key} | {g('n_front'):.0f} | {g('gd_front'):.4f} | {g('igd'):.4f} | {g('igd_plus'):.4f} | {g('hv_ratio'):.3f} | "
                         f"{g('joint_nondominated_fraction_all_valid'):.3f} | {g('spacing_cv'):.3f} | {g('spacing_size_matched_percentile'):.2f} |")
    lines += ["", "## Pareto quality vs empirical reference (support cost; median over replications)", "",
              "| dataset | set | n_front | IGD+ | HV ratio | joint-ND frac |", "|---|---|---|---|---|---|"]
    for ds in datasets:
        for key in SETS:
            q = S["pareto_quality"]["support"][ds].get(key, {})
            g = lambda m: q.get(m, {}).get("median", float("nan"))
            lines.append(f"| {ds} | {key} | {g('n_front'):.0f} | {g('igd_plus'):.4f} | {g('hv_ratio'):.3f} | {g('joint_nondominated_fraction_all_valid'):.3f} |")
    lines += ["", "## Empirical reference convergence", "", "| dataset | points (median) | rounds | displaced by independent check (median) | front size weighted | front size support |", "|---|---|---|---|---|---|"]
    for ds in datasets:
        r = S["reference"][ds]
        lines.append(f"| {ds} | {r['n_points'].get('median', 0):.0f} | {r['rounds'].get('median', 0):.0f} | {r['displaced_fraction'].get('median', float('nan')):.3f} | "
                     f"{r['front_weighted_size'].get('median', 0):.0f} | {r['front_support_size'].get('median', 0):.0f} |")
    lines += ["", "## Active-support frequency on the empirical reference front (weighted cost)", ""]
    for ds in datasets:
        lines.append(f"- {ds}: {S['support_frequency']['weighted'].get(ds + '__model_active_fraction')}")
    lines += ["", "## AUC vs log-loss conflict (direct-AUC optimum minus SLSQP optimum, OOF)", ""]
    for ds in datasets:
        c = S["auc_vs_logloss_conflict"][ds]
        lines.append(f"- {ds}: ΔAUC mean {c['delta_auc_directauc_minus_slsqp'].get('mean', float('nan')):+.5f} (sd {c['delta_auc_directauc_minus_slsqp'].get('std', float('nan')):.5f}); "
                     f"Δlog-loss mean {c['delta_logloss_directauc_minus_slsqp'].get('mean', float('nan')):+.5f}; Δcost {c['delta_cost_weighted'].get('mean', float('nan')):+.3f} ms/1k")
    lines += ["", "## Diversity vs beta_ij", "", f"- {S['diversity_vs_beta']}", "", "Tables: `tables/*.csv`; full numbers: `summary.json`."]
    (repdir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
