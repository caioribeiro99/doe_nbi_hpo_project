#!/usr/bin/env python
"""Generate the supplementary LaTeX tables and copy the supplementary figures.

Reads only the frozen artifacts. Emits papers/surrogate_nbi_ensemble/supplementary/.
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REP = REPO / "reports" / "pco213_postwork_benchmark"
FIGSRC = REPO / "figures" / "pco213_postwork_benchmark"
OUT = HERE / "supplementary"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)

DL = {"santander": "Santander", "bnp": "BNP Paribas", "porto": "Porto Seguro", "uci_credit": "UCI credit"}


def longtab(df, name, caption, label, maxrows=None, round_to=4, cols=None, landscape=False):
    """Emit a longtable. ``cols`` selects and orders the columns actually worth showing;
    dumping every CSV column produces unreadable 20-inch tables, so each call curates."""
    d = df.copy()
    if cols:
        keep = [c for c in cols if c in d.columns]
        d = d[keep]
    if maxrows and len(d) > maxrows:
        d = d.head(maxrows)
        caption += f" First {maxrows} of {len(df)} rows; the complete table is the source CSV."
    if cols and len(cols) > len([c for c in cols if c in df.columns]):
        pass
    if len(df.columns) > len(d.columns):
        caption += f" Showing {len(d.columns)} of {len(df.columns)} columns; the rest are in the source CSV."
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(round_to)
    d.columns = [str(c).replace("_", " ").replace("%", r"\%") for c in d.columns]
    body = d.to_latex(index=False, escape=True, longtable=True, na_rep="--")
    size = r"\tiny" if (landscape or len(d.columns) > 9) else r"\scriptsize"
    body = body.replace(r"\begin{longtable}",
                        f"\\begingroup{size}\\setlength{{\\tabcolsep}}{{2.5pt}}\n" + r"\begin{longtable}")
    # longtable steps the table counter on its own, even with no \caption inside it.
    # The caption here is emitted by \captionof BEFORE the environment, so without this
    # the numbering advances by two per table (S1, S3, S5, ...).
    body = body.replace(r"\end{longtable}",
                        r"\end{longtable}\endgroup\addtocounter{table}{-1}")
    body = body.replace(r"\caption{}", "")
    head = f"\\begin{{center}}\\captionof{{table}}{{{caption}}}\\label{{{label}}}\\end{{center}}\n"
    tex = head + body
    if landscape:
        tex = "\\begin{landscape}\n" + tex + "\n\\end{landscape}\n"
    (OUT / "tables" / f"{name}.tex").write_text(tex)
    return name


def main() -> int:
    T, S, N = REP / "tables", REP / "statistics", REP / "nsga2"
    made = []

    # S2 base models and references
    made.append(longtab(pd.read_csv(T / "model_performance.csv").groupby(["dataset", "model"]).mean(numeric_only=True).round(4).reset_index(),
        "s2_model_performance", "Base-model performance averaged over the 30 outer partitions of each dataset.", "tab:s2model", cols=["dataset","model","oof_roc_auc","holdout_roc_auc","oof_log_loss","oof_brier","cost_ms_per_1k"]))
    made.append(longtab(pd.read_csv(T / "single_objective_refs.csv").groupby(["dataset", "method"]).mean(numeric_only=True).round(4).reset_index(),
        "s2_single_objective_refs", "Single-objective reference solutions, averaged over 30 partitions.", "tab:s2refs", cols=["dataset","method","oof_roc_auc","oof_log_loss","holdout_roc_auc","holdout_log_loss","cost_weighted","cost_support"]))
    made.append(longtab(pd.read_csv(T / "inference_costs.csv"), "s2_inference_costs",
        "Measured inference cost per base model and dataset, in milliseconds per 1{,}000 rows.", "tab:s2cost", maxrows=60, cols=["dataset","model","cost_ms_per_1k"]))

    # S3 Scheffe
    so = pd.read_csv(T / "scheffe_orders.csv")
    made.append(longtab(so.groupby(["dataset", "response", "order"]).agg(
        n=("rep", "count"), selected=("selected", "sum"), reliable=("reliable", "sum"),
        r2_train=("r2_train", "mean"), r2_external=("r2_external", "mean"),
        rmse_external=("rmse_external", "mean"), spearman=("spearman_external", "mean"),
        extrapolation=("extrapolation_excess", "mean")).round(4).reset_index(),
        "s3_scheffe_orders", "Scheff\\'e order comparison: how often each order is selected and passes the gate, with mean fit diagnostics over 30 partitions.", "tab:s3orders", cols=["dataset","response","order","n","selected","reliable","r2_train","r2_external","rmse_external","spearman","extrapolation"]))
    made.append(longtab(pd.read_csv(S / "coefficient_stability_r30.csv"), "s3_coefficients",
        "Full Scheff\\'e coefficient stability: mean, dispersion, bootstrap interval, sign frequency, rank statistics, vertex gap, real 50/50 blend outcome and Pareto participation for every term.", "tab:s3coef", maxrows=70, cols=["dataset","response","term","mean","sd","cv","ci95_lo","ci95_hi","sign_pos_freq","top1_freq","vertex_gap_mean","blend50_minus_best_member_mean","blend50_beats_best_member_freq"], landscape=True))

    # S4 indicators
    pq = pd.read_csv(T / "pareto_quality.csv")
    made.append(longtab(pq.groupby(["cost", "dataset", "set"]).median(numeric_only=True).round(4).reset_index().drop(columns=["rep"], errors="ignore"),
        "s4_pareto_indicators", "All Pareto quality indicators, median over 30 partitions, for every candidate set under both cost definitions.", "tab:s4ind", maxrows=60, cols=["cost","dataset","set","n_front","gd_front","igd","igd_plus","spacing","spacing_cv","hv_ratio","joint_nondominated_fraction_front","coverage_set_over_ref"], landscape=True))

    # S5 paired statistics
    made.append(longtab(pd.read_csv(S / "paired_primary_effects.csv"), "s5_paired_effects",
        "Complete paired primary effects: both cost definitions, all four comparisons, both endpoints, with means, medians, dispersion, both bootstrap intervals, win/tie/loss, Wilson interval, rank-biserial correlation and both tests.", "tab:s5eff", maxrows=64, cols=["cost","dataset","comparison","endpoint","n","mean","median","ci95_median_lo","ci95_median_hi","wins","ties","losses","rank_biserial","p_nb","p_wilcoxon"], landscape=True))
    made.append(longtab(pd.read_csv(S / "paired_primary_tests.csv"), "s5_paired_tests",
        "Holm-corrected test families, both cost definitions.", "tab:s5tests", maxrows=64, cols=["cost","dataset","comparison","endpoint","mean_delta","t_nadeau_bengio","p_nadeau_bengio","p_nb_holm_family_dataset","p_wilcoxon_holm","rank_biserial","win_frac"], landscape=True))
    made.append(longtab(pd.read_csv(S / "proportion_intervals.csv"), "s5_proportions",
        "Wilson and Jeffreys intervals for every reported frequency.", "tab:s5prop", maxrows=70, cols=["quantity","dataset","level","k","n","p","wilson_lo","wilson_hi","jeffreys_lo","jeffreys_hi"]))

    # S6 gate and regimes
    made.append(longtab(pd.read_csv(S / "reliability_gate_r30.csv"), "s6_gate",
        "Reliability gate for all four responses, including the Brier and PR-AUC responses omitted from the main text.", "tab:s6gate", cols=["dataset","response","n","pass","p_pass","wilson_lo","wilson_hi","r2_external_median","r2_external_ci95_lo","r2_external_ci95_hi","spearman_median"]))
    made.append(longtab(pd.read_csv(S / "reliability_gate_conditional_gain_r30.csv"), "s6_gate_conditional",
        "Anchor gain split by gate outcome. On BNP Paribas this split restates NBI-A's collapse and must not be read as evidence that real anchors add more when the surface is reliable.", "tab:s6cond", maxrows=40, cols=["dataset","gate_response","gate_passed","cost","endpoint","comparison","n","mean","median","wins","ties","losses"]))
    made.append(longtab(pd.read_csv(S / "bimodality_regimes_r30.csv"), "s6_regimes",
        "Exploratory regime diagnostics: Sarle's bimodality coefficient and a one- versus two-component Gaussian-mixture BIC difference. No formal multimodality test is claimed.", "tab:s6reg", maxrows=50, cols=["dataset","set","cost","metric","n","mean","median","sd","bimodality_coefficient","gmm_bic_delta"], landscape=True))

    # S7 solver diagnostics
    made.append(longtab(pd.read_csv(T / "nbi_runs.csv").groupby(["dataset", "variant"]).mean(numeric_only=True).round(4).reset_index(),
        "s7_nbi_runs", "NBI solver diagnostics per dataset and variant, averaged over 30 partitions. Success semantics differ between arms: A and B are SLSQP-certified, C is equality-feasible under the lenient rule.", "tab:s7nbi", cols=["dataset","variant","n_subproblems","n_success","n_front_real_weighted","n_front_real_support","total_nfev","n_real_objective_evals","seconds"], landscape=True))

    # S8 reference convergence
    made.append(longtab(pd.read_csv(T / "reference_diagnostics.csv").groupby("dataset").mean(numeric_only=True).round(4).reset_index(),
        "s8_reference", "Empirical reference construction diagnostics, averaged over 30 partitions.", "tab:s8ref", cols=["dataset","n_points","rounds","displaced_fraction","front_size_weighted","front_size_support"]))

    # S9 cost and conflict
    made.append(longtab(pd.read_csv(S / "cost_definition_sensitivity_r30.csv"), "s9_cost_sensitivity",
        "Cost-definition sensitivity in full.", "tab:s9cost", cols=["dataset","endpoint","n","best_set_differs_count","best_set_differs_frac","wilson_lo","wilson_hi","median_rank_spearman_weighted_vs_support"], landscape=True))
    made.append(longtab(pd.read_csv(S / "auc_logloss_conflict_r30.csv"), "s9_conflict",
        "AUC-versus-log-loss conflict in full, including weight-space distances, support Jaccard and both cost differences.", "tab:s9conf", maxrows=40, cols=["dataset","quantity","n","mean","median","sd","ci95_lo","ci95_hi","min","max"], landscape=True))

    # S10 holdout
    made.append(longtab(pd.read_csv(S / "holdout_transfer_r30.csv"), "s10_holdout",
        "Holdout transfer for every set under both cost definitions.", "tab:s10hold", maxrows=80, cols=["cost","dataset","set","metric","n","oof_mean","holdout_mean","delta_mean","delta_ci95_lo","delta_ci95_hi","frac_abs_delta_gt_0.005","wins"], landscape=True))

    # S11 R10 vs R30
    made.append(longtab(pd.read_csv(S / "r10_vs_r30_stability.csv"), "s11_r10_r30",
        "Complete $R = 10$ versus $R = 30$ comparison across all sixteen metric families.", "tab:s11", maxrows=90, cols=["metric","dataset","response","set","cost","term","r10_estimate","r30_estimate","abs_change","rel_change","r30_ci95_lo","r30_ci95_hi"], landscape=True))

    # S12 anchor injection control
    ai = pd.read_csv(REP / "anchor_injection_control.csv")
    made.append(longtab(ai.groupby("dataset").agg(
        n=("rep", "count"), hv_A=("hv_A", "median"), hv_B=("hv_B", "median"),
        hv_A_plus_anchors=("hv_A_plus_anchors", "median"),
        fraction_closed=("fraction_closed", "median"),
        residual_chim=("residual_chim_effect", "median")).round(4).reset_index(),
        "s12_anchor_injection", "Anchor-injection control: median over partitions of the fraction of the NBI-A to NBI-B hypervolume gap closed by adding the three real anchors to NBI-A's own candidate set, and the residual attributable to the relocated CHIM.", "tab:s12"))

    # S13 NSGA-II
    made.append(longtab(pd.read_csv(N / "nsga2_paired_effects.csv"), "s13_nsga2_effects",
        "NSGA-II paired effects against NBI-C, NBI-B and random scalarization, under both reference definitions and both cost definitions.", "tab:s13eff", maxrows=70, cols=["reference","cost","dataset","comparison","endpoint","n","mean","median","ci95_median_lo","ci95_median_hi","wins","ties","losses","rank_biserial","p_nb"], landscape=True))
    made.append(longtab(pd.read_csv(N / "nsga2_indicator_levels.csv"), "s13_nsga2_levels",
        "Median indicator levels for every set under both reference definitions.", "tab:s13lev", maxrows=70, cols=["reference","cost","dataset","set","igd_plus","hv_ratio","gd_front","spacing","spacing_cv","n_front"], landscape=True))
    made.append(longtab(pd.read_csv(N / "nsga2_budget_runtime.csv"), "s13_nsga2_budget",
        "NSGA-II realized evaluation and wall-clock ratios against NBI-C. Only evaluations are matched.", "tab:s13bud", cols=["dataset","n","nsga2_evals_total","nbi_c_evals_total","eval_ratio_median","nsga2_seconds_mean","nbi_c_seconds_mean","time_ratio_median","n_nondominated_median"], landscape=True))

    # figures: everything not already in the main paper
    main_figs = {"fig01_methodology"}
    copied = []
    for f in sorted(FIGSRC.glob("*.png")):
        if f.stem in main_figs:
            continue
        shutil.copy(f, OUT / "figures" / f.name)
        copied.append(f.stem)
    print(f"tables: {len(made)} | figures copied: {len(copied)}")
    (OUT / "figure_list.txt").write_text("\n".join(copied))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
