#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from doe_xgb.factor_analysis import run_factor_analysis, save_fa_outputs
from doe_xgb.rsm import fit_rsm_backward, save_rsm_coefficients
from doe_xgb.config import PARAM_NAMES
from doe_xgb.io_utils import save_csv_ptbr


def main():
    p = argparse.ArgumentParser(description="Run Factor Analysis (ML+Varimax) and fit RSM models (backward elimination)")
    p.add_argument("--doe_results", required=True, help="Path to doe_results.csv from run_doe")
    args = p.parse_args()

    doe_path = Path(args.doe_results)
    if not doe_path.exists():
        raise FileNotFoundError(doe_path)

    out_dir = doe_path.parent

    df = pd.read_csv(doe_path, sep=';', decimal=',')

    # FA expects Time_MeanFold
    fa_res = run_factor_analysis(df)

    # Save loadings + scores
    loadings_path = out_dir / "factor_loadings.csv"
    scores_path = out_dir / "factor_scores.csv"
    save_fa_outputs(fa_res, str(loadings_path), str(scores_path))

    # Merge scores into DOE results
    df_scored = df.join(fa_res.scores)
    scored_path = out_dir / "doe_results_with_scores.csv"
    save_csv_ptbr(df_scored, scored_path)

    # Fit RSM for oriented scores
    factors_df = df_scored[PARAM_NAMES].copy()
    model_q = fit_rsm_backward(factors_df, df_scored["Score_Quality"], response_name="Score_Quality", alpha=0.05)
    model_c = fit_rsm_backward(factors_df, df_scored["Score_Cost"], response_name="Score_Cost", alpha=0.05)

    coef_q_path = out_dir / "rsm_coefficients_quality.csv"
    coef_c_path = out_dir / "rsm_coefficients_cost.csv"
    save_rsm_coefficients(model_q, str(coef_q_path))
    save_rsm_coefficients(model_c, str(coef_c_path))

    # Save model info
    info = {
        "quality": {"r2": model_q.r2, "r2_adj": model_q.r2_adj, "alpha": model_q.alpha, "n_terms": len(model_q.terms)},
        "cost": {"r2": model_c.r2, "r2_adj": model_c.r2_adj, "alpha": model_c.alpha, "n_terms": len(model_c.terms)},
        "fa": {"quality_factor": fa_res.quality_factor, "cost_factor": fa_res.cost_factor},
    }
    (out_dir / "fa_rsm_info.json").write_text(json.dumps(info, indent=2), encoding='utf-8')

    print("✅ FA + RSM completed")
    print(f"- {loadings_path}")
    print(f"- {scores_path}")
    print(f"- {scored_path}")
    print(f"- {coef_q_path}")
    print(f"- {coef_c_path}")


if __name__ == "__main__":
    main()
