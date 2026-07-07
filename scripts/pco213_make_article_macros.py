#!/usr/bin/env python3
"""Generate the article's numeric macros and tables from experiment artifacts.

Reads experiments/pco213/santander/*.{json,csv} and figures/pco213/*.png and
writes, under reports/pco213/article/:
  results_macros.tex      \\newcommand macros used by Article_PCO213.tex
  table_base_models.tex   Table I body (booktabs)
  table_holdout.tex       Table II body (booktabs)
  Figures/*.png           article-named copies of the study figures

Never edit those generated files by hand — re-run this script instead.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "experiments" / "pco213" / "santander"
FIG = REPO / "figures" / "pco213"
ART = REPO / "reports" / "pco213" / "article"

MODEL_LABELS = {
    "lr": "Reg.\\ Logística",
    "gnb": "Gaussian NB",
    "knn": "$k$-NN",
    "et": "ExtraTrees",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
METHOD_LABELS = {
    "best_single": "Melhor modelo individual",
    "uniform_voting": "Voto uniforme ($w_j{=}1/5$)",
    "stacking_lr": "Stacking (reg.\\ logística)",
    "slsqp_direct_logloss": "SLSQP direto (log-loss)",
    "scheffe_optimum_logloss": "Mistura Scheffé (log-loss)",
    "scheffe_optimum_auc": "Mistura Scheffé (AUC)",
    "dirichlet_scan_auc": "Varredura Dirichlet (AUC)",
}


def br(x: float, nd: int = 4) -> str:
    """pt-BR decimal comma, LaTeX-safe."""
    return f"{x:.{nd}f}".replace(".", "{,}")


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}".replace(".", "{,}") + "\\%"


def main() -> None:
    data_meta = json.loads((OUT / "data_meta.json").read_text())
    scheffe = json.loads((OUT / "scheffe_report.json").read_text())
    opt_rep = json.loads((OUT / "optimization_report.json").read_text())
    stats = json.loads((OUT / "statistics_corrected_ttest.json").read_text())
    oof_t = json.loads((OUT / "oof_timings.json").read_text())
    design_meta = json.loads((OUT / "design_meta.json").read_text())
    fm = pd.read_csv(OUT / "fold_metrics_models.csv")
    res = pd.read_csv(OUT / "holdout_results.csv").set_index("method")
    eda = json.loads((OUT / "eda_summary.json").read_text())

    names = list(fm["model"].unique())
    weights = opt_rep["weights"]

    # ---------------- Table I: base models ----------------
    g = fm.groupby("model")[["roc_auc", "log_loss", "brier"]].agg(["mean", "std"])
    order = g[("roc_auc", "mean")].sort_values(ascending=False).index
    lines = [
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Modelo & ROC-AUC & log-loss & Brier \\\\", "\\midrule",
    ]
    for m in order:
        lines.append(
            f"{MODEL_LABELS.get(m, m)} & "
            f"{br(g.loc[m, ('roc_auc', 'mean')])} $\\pm$ {br(g.loc[m, ('roc_auc', 'std')])} & "
            f"{br(g.loc[m, ('log_loss', 'mean')])} $\\pm$ {br(g.loc[m, ('log_loss', 'std')])} & "
            f"{br(g.loc[m, ('brier', 'mean')])} $\\pm$ {br(g.loc[m, ('brier', 'std')])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ART / "table_base_models.tex").write_text("\n".join(lines))

    # ---------------- Table II: holdout ----------------
    show = ["best_single", "uniform_voting", "stacking_lr", "slsqp_direct_logloss",
            "scheffe_optimum_logloss", "scheffe_optimum_auc"]
    lines = [
        "\\scriptsize\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "Método & ROC-AUC & log-loss & Brier & F1 & Ac.\\ bal. \\\\", "\\midrule",
    ]
    for m in show:
        r = res.loc[m]
        lines.append(
            f"{METHOD_LABELS[m]} & {br(r['roc_auc'])} & {br(r['log_loss'])} & "
            f"{br(r['brier'])} & {br(r['f1'], 3)} & {br(r['balanced_accuracy'], 3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ART / "table_holdout.tex").write_text("\n".join(lines))

    # ---------------- narrative fragments ----------------
    auc_mean = g[("roc_auc", "mean")]
    rank = auc_mean.sort_values(ascending=False)
    first, second, worst = rank.index[0], rank.index[1], rank.index[-1]
    base_summary = (
        f"O melhor modelo individual em OOF foi o {MODEL_LABELS[first]} "
        f"(AUC média {br(rank.iloc[0], 3)}), seguido de perto pelo {MODEL_LABELS[second]} "
        f"({br(rank.iloc[1], 3)}); o {MODEL_LABELS[worst]} ficou substancialmente abaixo "
        f"({br(rank.iloc[-1], 3)}), como esperado em 200 dimensões."
    )

    holdout_summary = (
        "Em perda logarítmica no holdout, as misturas otimizadas "
        f"(direta: {br(res.loc['slsqp_direct_logloss', 'log_loss'])}; Scheffé: "
        f"{br(res.loc['scheffe_optimum_logloss', 'log_loss'])}) superaram o melhor modelo "
        f"individual ({br(res.loc['best_single', 'log_loss'])}), o stacking "
        f"({br(res.loc['stacking_lr', 'log_loss'])}) e, com folga, o voto uniforme "
        f"({br(res.loc['uniform_voting', 'log_loss'])}). Em ROC-AUC as diferenças são "
        f"pequenas: a melhor mistura ({br(res.loc['dirichlet_scan_auc', 'roc_auc'])}, "
        "varredura direta) e o voto uniforme "
        f"({br(res.loc['uniform_voting', 'roc_auc'])}) praticamente empatam, ambos acima "
        f"do melhor modelo individual ({br(res.loc['best_single', 'roc_auc'])})."
    )

    def fmt_p(p: float) -> str:
        return "p<0{,}001" if p < 1e-3 else f"p={br(p, 3)}"

    def verdict(entry: dict, lower_is_better: bool) -> str:
        """Data-driven wording: melhor/pior/indistinguível (diff = scheffé - outro)."""
        if entry["p_value"] >= 0.05:
            return f"estatisticamente indistinguível ({fmt_p(entry['p_value'])})"
        favorable = (entry["mean_diff"] < 0) if lower_is_better else (entry["mean_diff"] > 0)
        word = "significativamente melhor" if favorable else "significativamente pior"
        return f"{word} ($t={br(abs(entry['t']), 1)}$; {fmt_p(entry['p_value'])})"

    sll, sa = stats["log_loss"], stats["roc_auc"]
    stats_summary = (
        "em perda logarítmica, a mistura de Scheffé foi "
        f"{verdict(sll['scheffe_vs_uniform_voting'], True)} que o voto uniforme e "
        f"{verdict(sll['scheffe_vs_stacking_lr'], True)} que o stacking; em ROC-AUC, foi "
        f"{verdict(sa['scheffe_vs_uniform_voting'], False)} do voto uniforme e "
        f"{verdict(sa['scheffe_vs_best_single'], False)} que o melhor modelo individual — "
        "coerente com uma região quase ótima ampla que contém o centroide do simplex."
    )

    coef = pd.read_csv(OUT / "scheffe_coefficients_roc_auc.csv")
    cross = coef[coef["term"].str.contains("\\*")].copy()
    cross["absmean"] = cross["mean"].abs()
    top = cross.sort_values("absmean", ascending=False).iloc[0]
    top_pair = top["term"].replace("w_", "").replace("*", "--")
    coef_reading = (
        f"O termo cruzado de maior magnitude foi {top_pair} "
        f"($\\beta={br(top['mean'], 3)}$), identificando o par de famílias com maior "
        f"complementaridade sob AUC."
    )

    w_auc = np.array(weights["scheffe_optimum_auc"])
    w_ll = np.array(weights["scheffe_optimum_logloss"])
    j_gnb = names.index("gnb")

    macros = {
        "DesignRuns": str(design_meta["design_runs"]),
        "ValPoints": str(design_meta["validation_points"]),
        "PooledRtwoAUC": br(scheffe["roc_auc"]["pooled_r2"], 3),
        "PooledRtwoLL": br(scheffe["log_loss"]["pooled_r2"], 3),
        "CondNumber": br(scheffe["roc_auc"]["condition_number"], 1),
        "RelRmseAUC": pct(scheffe["roc_auc"]["external_validation"]["rmse_relative_to_range"]),
        "RelRmseLL": pct(scheffe["log_loss"]["external_validation"]["rmse_relative_to_range"]),
        "GapMetaDirect": br(opt_rep["gap_scheffe_vs_direct_logloss_oof"], 5),
        "NBWeightAUC": br(float(w_auc[j_gnb]), 2),
        "NBWeightLL": br(float(w_ll[j_gnb]), 2),
        "TotalWallMin": br(oof_t["wall_seconds"] / 60.0, 1),
        "MaxUnivarAUC": br(eda["max_univariate_auc"], 3),
        "MedUnivarAUC": br(eda["median_univariate_auc"], 3),
        "MaxAbsCorr": br(eda["max_abs_offdiag_corr"], 3),
        "BaseModelsSummary": base_summary,
        "HoldoutSummary": holdout_summary,
        "StatsSummary": stats_summary,
        "CoefReading": coef_reading,
    }
    out = ["% Auto-generated by scripts/pco213_make_article_macros.py — do not edit."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (ART / "results_macros.tex").write_text("\n".join(out) + "\n")

    # ---------------- figures ----------------
    (ART / "Figures").mkdir(exist_ok=True)
    copies = {
        "fig1b_eda_correlation.png": "fig2_eda_correlation.png",
        "fig3_ternary_auc.png": "fig3_ternary_auc.png",
        "fig5_scheffe_coefficients_auc.png": "fig4_scheffe_coefficients.png",
        "fig6_holdout_comparison.png": "fig5_holdout_comparison.png",
    }
    for src, dst in copies.items():
        shutil.copy(FIG / src, ART / "Figures" / dst)
    print("macros/tables/figures written to", ART)


if __name__ == "__main__":
    main()
