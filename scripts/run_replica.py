#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Ensure local src/ is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from doe_xgb.io_utils import load_dataset, load_design, save_csv_ptbr
from doe_xgb.doe_runner import run_doe
from doe_xgb.factor_analysis import run_factor_analysis, save_fa_outputs
from doe_xgb.rsm import fit_rsm_backward, save_rsm_coefficients
from doe_xgb.nbi import load_coefficients_csv, run_nbi_weighted_sum, save_nbi_candidates
import doe_xgb.benchmarks as bm
from doe_xgb.tracking import build_replica_dir, write_manifest
from doe_xgb.config import PARAM_NAMES


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full pipeline for one replica")
    p.add_argument("--dataset", required=True, help="Path to dataset (xlsx/csv/parquet)")
    p.add_argument("--design", required=True, help="Path to DOE design CSV")
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--seed", type=int, default=None, help="Replica seed (if omitted, uses --replica as seed)")
    p.add_argument("--out_root", default="experiments")
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--target", default="y")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)
    seed = args.seed if args.seed is not None else int(args.replica)

    out_dir = build_replica_dir(out_root, dataset_path, design_path, args.replica)
    write_manifest(out_dir, replica=args.replica, seed=seed, dataset_path=dataset_path, design_path=design_path)

    stage_times: dict[str, float] = {}

    # Load inputs
    X, y = load_dataset(dataset_path, target_col=args.target, target_map={"g": 0, "h": 1})
    design_df = load_design(design_path)

    # 1) DOE
    t0 = time.perf_counter()
    doe_df = run_doe(design_df, X, y, seed=seed, n_splits=5, n_jobs=-1, tree_method="hist")
    stage_times["doe_seconds"] = float(time.perf_counter() - t0)
    save_csv_ptbr(doe_df, out_dir / "doe_results.csv")

    # 2) FA
    t0 = time.perf_counter()
    fa_res = run_factor_analysis(doe_df)
    stage_times["fa_seconds"] = float(time.perf_counter() - t0)

    loadings_path = out_dir / "factor_loadings.csv"
    scores_path = out_dir / "factor_scores.csv"
    save_fa_outputs(fa_res, str(loadings_path), str(scores_path))

    doe_scored = doe_df.join(fa_res.scores)
    save_csv_ptbr(doe_scored, out_dir / "doe_results_with_scores.csv")

    # 3) RSM
    t0 = time.perf_counter()

    # Force DataFrame selection (avoids Pylance inferring Series[Any])
    factors_df = doe_scored.loc[:, PARAM_NAMES].copy()

    model_q = fit_rsm_backward(
        factors_df,
        doe_scored["Score_Quality"],
        response_name="Score_Quality",
        alpha=0.05,
    )
    model_c = fit_rsm_backward(
        factors_df,
        doe_scored["Score_Cost"],
        response_name="Score_Cost",
        alpha=0.05,
    )
    stage_times["rsm_seconds"] = float(time.perf_counter() - t0)

    coef_q_path = out_dir / "rsm_coefficients_quality.csv"
    coef_c_path = out_dir / "rsm_coefficients_cost.csv"
    save_rsm_coefficients(model_q, str(coef_q_path))
    save_rsm_coefficients(model_c, str(coef_c_path))

    # 4) NBI candidates
    t0 = time.perf_counter()
    utopia = (float(doe_scored["Score_Quality"].max()), float(doe_scored["Score_Cost"].max()))
    nadir = (float(doe_scored["Score_Quality"].min()), float(doe_scored["Score_Cost"].min()))
    m1 = load_coefficients_csv(str(coef_q_path))
    m2 = load_coefficients_csv(str(coef_c_path))

    candidates = run_nbi_weighted_sum(
        m1,
        m2,
        observed_utopia=utopia,
        observed_nadir=nadir,
        beta_step=0.05,
        seed=seed,
        n_starts=10,
        constrain_pred_range=True,
    )
    nbi_path = out_dir / "nbi_candidates.csv"
    save_nbi_candidates(candidates, str(nbi_path))
    stage_times["nbi_seconds"] = float(time.perf_counter() - t0)

    # 5) Confirmation + benchmarks
    t0_confirm = time.perf_counter()
    cands = bm.load_nbi_candidates(str(nbi_path))
    best_nbi, all_nbi = bm.evaluate_candidate_list(cands, X, y, seed=seed, n_splits=5, n_jobs=-1, tree_method="hist")
    stage_times["nbi_candidate_evaluation_seconds"] = float(time.perf_counter() - t0_confirm)

    save_csv_ptbr(pd.DataFrame(all_nbi), out_dir / "nbi_candidate_evaluations.csv")

    summary_rows: list[dict] = []
    best_nbi_row = dict(best_nbi)
    best_nbi_row["method"] = "doe_nbi"
    best_nbi_row["budget"] = len(cands)

    # Optimization time: DOE + FA + RSM + NBI + candidate evaluation
    best_nbi_row["Optimization_Time_Seconds"] = float(
        stage_times.get("doe_seconds", 0.0)
        + stage_times.get("fa_seconds", 0.0)
        + stage_times.get("rsm_seconds", 0.0)
        + stage_times.get("nbi_seconds", 0.0)
        + stage_times.get("nbi_candidate_evaluation_seconds", 0.0)
    )

    # Total time: optimization + final model evaluation time (mean per fold)
    best_nbi_row["Total_Time_Seconds"] = float(
        best_nbi_row["Optimization_Time_Seconds"] + float(best_nbi_row.get("Time_MeanFold", 0.0))
    )

    summary_rows.append(best_nbi_row)

    # Benchmarks (each returns Optimization_Time_Seconds and Total_Time_Seconds)
    summary_rows.append(bm.coarse_grid_search(X, y, seed=seed, budget=args.budget))
    summary_rows.append(bm.random_search(X, y, seed=seed, budget=args.budget))

    try:
        summary_rows.append(bm.bayes_search(X, y, seed=seed, budget=args.budget))
    except Exception as e:
        summary_rows.append({"method": "bayes_search", "error": str(e)})

    try:
        summary_rows.append(bm.hyperopt_tpe(X, y, seed=seed, budget=args.budget))
    except Exception as e:
        summary_rows.append({"method": "hyperopt_tpe", "error": str(e)})

    bm.save_benchmark_summary(summary_rows, str(out_dir / "confirmation_summary.csv"))

    # Update manifest with timing summary
    write_manifest(
        out_dir,
        replica=args.replica,
        seed=seed,
        dataset_path=dataset_path,
        design_path=design_path,
        extra={"stage_times": stage_times},
    )

    print("✅ Replica finished")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
