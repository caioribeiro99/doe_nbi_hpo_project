#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure local src/ is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Optional .env support (nice when running scripts directly)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

import numpy as np
import pandas as pd

from doe_xgb.io_utils import load_dataset, load_design, save_csv_ptbr
from doe_xgb.doe_runner import run_doe
from doe_xgb.factor_analysis import run_factor_analysis, save_fa_outputs
from doe_xgb.rsm import fit_rsm_backward, save_rsm_coefficients
from doe_xgb.nbi import load_coefficients_csv, run_nbi_weighted_sum, save_nbi_candidates, NBICandidate
import doe_xgb.benchmarks as bm
from doe_xgb.tracking import build_replica_dir, write_manifest
from doe_xgb.config import PARAM_NAMES, INT_PARAMS


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None or v == "" else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None or v == "" else float(v)


def _normalize_params(params: Dict[str, Any], ndigits: int = 6) -> Tuple[Tuple[str, Any], ...]:
    """Normalize a hyperparameter dict to a hashable key (helps deduplicate near-identical solutions)."""
    norm: Dict[str, Any] = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            norm[k] = int(round(float(v)))
        else:
            if isinstance(v, (float, int)):
                norm[k] = round(float(v), ndigits)
            else:
                norm[k] = v
    return tuple(sorted(norm.items(), key=lambda kv: kv[0]))


def _dedup_candidates(cands: List[NBICandidate]) -> List[NBICandidate]:
    seen: set[Tuple[Tuple[str, Any], ...]] = set()
    uniq: List[NBICandidate] = []
    for c in cands:
        key = _normalize_params(c.params)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _select_evenly_by_beta(cands: List[NBICandidate], k: int) -> List[NBICandidate]:
    """Select k candidates evenly spaced by beta_2 (cost weight)."""
    if k >= len(cands):
        return list(cands)

    # Sort by beta_2 (0..1), then take evenly spaced indices
    c_sorted = sorted(cands, key=lambda c: float(c.betas[1]))
    idxs = np.linspace(0, len(c_sorted) - 1, k)
    idxs = np.unique(np.round(idxs).astype(int))

    chosen: List[NBICandidate] = [c_sorted[i] for i in idxs]

    # Ensure exact k (can happen if rounding collapsed indices)
    if len(chosen) < k:
        used = set(int(i) for i in idxs)
        for i in range(len(c_sorted)):
            if i in used:
                continue
            chosen.append(c_sorted[i])
            if len(chosen) >= k:
                break

    return chosen[:k]


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full pipeline for one replica")
    p.add_argument("--dataset", required=True, help="Path to dataset (xlsx/csv/parquet)")
    p.add_argument("--design", required=True, help="Path to DOE design CSV")
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--seed", type=int, default=None, help="Replica seed (if omitted, uses --replica as seed)")
    p.add_argument("--out_root", default="experiments")
    p.add_argument(
        "--budget",
        type=int,
        default=_env_int("BUDGET", 0),
        help="Benchmark budget (# evaluations). 0 = auto-match DOE+NBI total evals. CLI > .env:BUDGET",
    )
    p.add_argument("--target", default=os.getenv("TARGET_COL", "y"))
    p.add_argument(
        "--beta-step",
        type=float,
        default=_env_float("NBI_BETA_STEP", 0.05),
        help="Beta step for NBI weight grid (e.g., 0.05 or 0.02). CLI > .env:NBI_BETA_STEP",
    )
    p.add_argument(
        "--nbi-eval-k",
        type=int,
        default=_env_int("NBI_EVAL_K", 20),
        help="Max # of NBI candidates to evaluate (subsampled evenly by beta). CLI > .env:NBI_EVAL_K",
    )
    p.add_argument(
        "--nbi-n-starts",
        type=int,
        default=_env_int("NBI_N_STARTS", 10),
        help="Multi-starts for each NBI optimization (cheap). CLI > .env:NBI_N_STARTS",
    )

    args = p.parse_args()

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)
    seed = args.seed if args.seed is not None else int(args.replica)

    out_dir = build_replica_dir(out_root, dataset_path, design_path, args.replica)
    write_manifest(out_dir, replica=args.replica, seed=seed, dataset_path=dataset_path, design_path=design_path)

    stage_times: Dict[str, float] = {}

    # Load inputs
    X, y = load_dataset(dataset_path, target_col=str(args.target), target_map={"g": 0, "h": 1})
    design_df = load_design(design_path)
    doe_runs = int(len(design_df))

    # 1) DOE
    t0 = time.perf_counter()
    doe_df = run_doe(design_df, X, y, seed=seed, n_splits=5, n_jobs=-1, tree_method="hist")
    stage_times["doe_seconds"] = float(time.perf_counter() - t0)
    doe_path = out_dir / "doe_results.csv"
    save_csv_ptbr(doe_df, doe_path)

    # 2) FA
    t0 = time.perf_counter()
    fa_res = run_factor_analysis(doe_df)
    stage_times["fa_seconds"] = float(time.perf_counter() - t0)

    loadings_path = out_dir / "factor_loadings.csv"
    scores_path = out_dir / "factor_scores.csv"
    save_fa_outputs(fa_res, str(loadings_path), str(scores_path))

    doe_scored = doe_df.join(fa_res.scores)
    doe_scored_path = out_dir / "doe_results_with_scores.csv"
    save_csv_ptbr(doe_scored, doe_scored_path)

    # 3) RSM
    t0 = time.perf_counter()
    factors_df = doe_scored[PARAM_NAMES].copy()
    model_q = fit_rsm_backward(factors_df, doe_scored["Score_Quality"], response_name="Score_Quality", alpha=0.05)
    model_c = fit_rsm_backward(factors_df, doe_scored["Score_Cost"], response_name="Score_Cost", alpha=0.05)
    stage_times["rsm_seconds"] = float(time.perf_counter() - t0)

    coef_q_path = out_dir / "rsm_coefficients_quality.csv"
    coef_c_path = out_dir / "rsm_coefficients_cost.csv"
    save_rsm_coefficients(model_q, str(coef_q_path))
    save_rsm_coefficients(model_c, str(coef_c_path))

    # 4) NBI candidates (cheap)
    t0 = time.perf_counter()
    utopia = (float(doe_scored["Score_Quality"].max()), float(doe_scored["Score_Cost"].max()))
    nadir = (float(doe_scored["Score_Quality"].min()), float(doe_scored["Score_Cost"].min()))
    m1 = load_coefficients_csv(str(coef_q_path))
    m2 = load_coefficients_csv(str(coef_c_path))

    nbi_candidates_all = run_nbi_weighted_sum(
        m1,
        m2,
        observed_utopia=utopia,
        observed_nadir=nadir,
        beta_step=float(args.beta_step),
        seed=seed,
        n_starts=int(args.nbi_n_starts),
        constrain_pred_range=True,
    )
    stage_times["nbi_seconds"] = float(time.perf_counter() - t0)

    # Deduplicate (important when optimizer converges to same point for multiple betas)
    nbi_candidates_unique = _dedup_candidates(nbi_candidates_all)

    # Save full candidate list (unique) for auditability
    nbi_path = out_dir / "nbi_candidates.csv"
    save_nbi_candidates(nbi_candidates_unique, str(nbi_path))

    # 5) Confirmation: evaluate a capped subset of candidates (expensive)
    nbi_eval_k = max(1, int(args.nbi_eval_k))
    nbi_candidates_eval = _select_evenly_by_beta(nbi_candidates_unique, nbi_eval_k)
    cand_params: List[Dict[str, Any]] = [c.params for c in nbi_candidates_eval]

    t0_confirm = time.perf_counter()
    best_nbi, all_nbi = bm.evaluate_candidate_list(
        cand_params, X, y, seed=seed, n_splits=5, n_jobs=-1, tree_method="hist"
    )
    stage_times["nbi_candidate_evaluation_seconds"] = float(time.perf_counter() - t0_confirm)

    save_csv_ptbr(pd.DataFrame(all_nbi), out_dir / "nbi_candidate_evaluations.csv")

    # ----------------------------
    # Summary row (DOE+NBI method)
    # ----------------------------
    summary_rows: List[Dict[str, Any]] = []

    best_nbi_row = dict(best_nbi)
    best_nbi_row["method"] = "doe_nbi"
    best_nbi_row["DOE_Evals"] = int(doe_runs)
    best_nbi_row["NBI_Candidates_Generated"] = int(len(nbi_candidates_all))
    best_nbi_row["NBI_Candidates_Unique"] = int(len(nbi_candidates_unique))
    best_nbi_row["NBI_Candidates_Evaluated"] = int(len(cand_params))

    total_evals = int(doe_runs + len(cand_params))
    best_nbi_row["budget"] = int(total_evals)

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

    # ----------------------------
    # Benchmarks (fair budgets)
    # ----------------------------
    benchmark_budget = int(args.budget) if int(args.budget) > 0 else int(total_evals)

    summary_rows.append(bm.coarse_grid_search(X, y, seed=seed, budget=benchmark_budget))
    summary_rows.append(bm.random_search(X, y, seed=seed, budget=benchmark_budget))

    try:
        summary_rows.append(bm.bayes_search(X, y, seed=seed, budget=benchmark_budget))
    except Exception as e:
        summary_rows.append({"method": "bayes_search", "budget": int(benchmark_budget), "error": str(e)})

    try:
        summary_rows.append(bm.hyperopt_tpe(X, y, seed=seed, budget=benchmark_budget))
    except Exception as e:
        summary_rows.append({"method": "hyperopt_tpe", "budget": int(benchmark_budget), "error": str(e)})

    bm.save_benchmark_summary(summary_rows, str(out_dir / "confirmation_summary.csv"))

    # Update manifest with timing summary + counts
    write_manifest(
        out_dir,
        replica=args.replica,
        seed=seed,
        dataset_path=dataset_path,
        design_path=design_path,
        extra={
            "stage_times": stage_times,
            "doe_runs": doe_runs,
            "nbi_candidates_generated": int(len(nbi_candidates_all)),
            "nbi_candidates_unique": int(len(nbi_candidates_unique)),
            "nbi_candidates_evaluated": int(len(cand_params)),
            "total_evaluations_doe_nbi": int(total_evals),
            "benchmark_budget": int(benchmark_budget),
            "beta_step": float(args.beta_step),
            "nbi_eval_k": int(nbi_eval_k),
        },
    )

    print("✅ Replica finished")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
