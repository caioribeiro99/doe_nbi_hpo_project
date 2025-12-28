#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import contextlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, IO, List, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Allow running scripts directly without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from doe_xgb.io_utils import load_dataset, load_design, save_csv_ptbr
from doe_xgb.doe_runner import run_doe
from doe_xgb.factor_analysis import run_factor_analysis, save_fa_outputs
from doe_xgb.rsm import fit_rsm_backward, save_rsm_coefficients
from doe_xgb.nbi import (
    load_coefficients_csv,
    run_nbi_weighted_sum,
    save_nbi_candidates,
    nbi_candidates_to_df,
)
from doe_xgb.benchmarks import (
    evaluate_candidate_list,
    coarse_grid_search,
    random_search,
    bayes_search,
    hyperopt_tpe,
    save_benchmark_summary,
)
from doe_xgb.tracking import build_replica_dir, write_manifest
from doe_xgb.config import PARAM_NAMES, DEFAULT_BOUNDS


def _parse_hyperparameters_cell(cell: object) -> Dict[str, object]:
    if isinstance(cell, dict):
        return cell
    if cell is None:
        return {}
    try:
        if isinstance(cell, float) and math.isnan(cell):
            return {}
    except Exception:
        pass
    if isinstance(cell, str):
        try:
            parsed = ast.literal_eval(cell)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_float_series(s: pd.Series) -> pd.Series:
    return pd.Series(pd.to_numeric(s, errors="coerce"), index=s.index, dtype="float64")


def _ensure_param_columns(df: pd.DataFrame, param_names: List[str]) -> pd.DataFrame:
    missing = [p for p in param_names if p not in df.columns]
    if not missing:
        return df

    if "hyperparameters" in df.columns:
        hp_series = df["hyperparameters"].apply(_parse_hyperparameters_cell)
        for p in missing:
            df[p] = hp_series.apply(lambda d: d.get(p, np.nan))
    else:
        for p in missing:
            df[p] = np.nan
    return df


class Tee(IO[str]):
    def __init__(self, *streams: IO[str]):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)

    p = argparse.ArgumentParser(description="Run full pipeline for one replica")
    p.add_argument("--dataset", default=os.getenv("DATASET_PATH"))
    p.add_argument("--design", default=os.getenv("DESIGN_PATH"))
    p.add_argument("--replica", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_root", default="experiments")
    p.add_argument("--budget", type=int, default=0)
    p.add_argument("--target", default="y")
    p.add_argument("--beta-step", type=float, default=0.02)
    p.add_argument("--nbi-eval-k", type=int, default=0)
    p.add_argument("--nbi-n-starts", type=int, default=10)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--tree-method", default="hist")
    p.add_argument("--nbi-constrain-pred-range", action="store_true")

    args = p.parse_args()

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)
    seed = args.seed if args.seed is not None else args.replica

    out_dir = build_replica_dir(out_root, dataset_path, design_path, args.replica)

    # Manifest (file path, not dir)
    write_manifest(
        out_dir / "manifest.json",
        replica=args.replica,
        seed=seed,
        dataset_path=dataset_path,
        design_path=design_path,
        extra={
            "args": {
                "budget": int(args.budget),
                "target": str(args.target),
                "beta_step": float(args.beta_step),
                "nbi_eval_k": int(args.nbi_eval_k),
                "nbi_n_starts": int(args.nbi_n_starts),
                "n_splits": int(args.n_splits),
                "n_jobs": int(args.n_jobs),
                "tree_method": str(args.tree_method),
                "nbi_constrain_pred_range": bool(args.nbi_constrain_pred_range),
            }
        },
    )

    log_path = out_dir / "run_replica.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            stage_times: Dict[str, float] = {}

            X, y = load_dataset(
                dataset_path,
                target_col=args.target,
                target_map={"g": 0, "h": 1},
            )
            design_df = load_design(design_path)

            # 1) DOE
            t0 = time.perf_counter()
            doe_df = run_doe(
                design_df,
                X,
                y,
                seed=seed,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
                tree_method=args.tree_method,
            )
            stage_times["doe_seconds"] = float(time.perf_counter() - t0)
            save_csv_ptbr(doe_df, out_dir / "doe_results.csv")

            # 2) FA
            t0 = time.perf_counter()
            fa_res = run_factor_analysis(doe_df)
            stage_times["fa_seconds"] = float(time.perf_counter() - t0)
            save_fa_outputs(
                fa_res,
                str(out_dir / "factor_loadings.csv"),
                str(out_dir / "factor_scores.csv"),
            )

            doe_scored = doe_df.join(fa_res.scores)
            save_csv_ptbr(doe_scored, out_dir / "doe_results_with_scores.csv")

            # 3) RSM
            t0 = time.perf_counter()
            model_q = fit_rsm_backward(
                doe_scored[PARAM_NAMES],
                doe_scored["Score_Quality"],
                response_name="Score_Quality",
            )
            model_c = fit_rsm_backward(
                doe_scored[PARAM_NAMES],
                doe_scored["Score_Cost"],
                response_name="Score_Cost",
            )
            stage_times["rsm_seconds"] = float(time.perf_counter() - t0)

            save_rsm_coefficients(model_q, str(out_dir / "rsm_coefficients_quality.csv"))
            save_rsm_coefficients(model_c, str(out_dir / "rsm_coefficients_cost.csv"))

            # 4) NBI
            utopia = (float(doe_scored["Score_Quality"].max()), float(doe_scored["Score_Cost"].max()))
            nadir = (float(doe_scored["Score_Quality"].min()), float(doe_scored["Score_Cost"].min()))

            t0 = time.perf_counter()
            candidates = run_nbi_weighted_sum(
                load_coefficients_csv(str(out_dir / "rsm_coefficients_quality.csv")),
                load_coefficients_csv(str(out_dir / "rsm_coefficients_cost.csv")),
                observed_utopia=utopia,
                observed_nadir=nadir,
                beta_step=args.beta_step,
                seed=seed,
                n_starts=args.nbi_n_starts,
                constrain_pred_range=args.nbi_constrain_pred_range,
            )
            stage_times["nbi_seconds"] = float(time.perf_counter() - t0)

            save_nbi_candidates(candidates, str(out_dir / "nbi_candidates.csv"))

            # Prepare unique candidate list
            nbi_df = nbi_candidates_to_df(candidates)
            nbi_df = _ensure_param_columns(nbi_df, PARAM_NAMES)

            for p_name in PARAM_NAMES:
                nbi_df[p_name] = _coerce_float_series(nbi_df[p_name])

            for p_name in PARAM_NAMES:
                if nbi_df[p_name].isna().any():
                    lo, hi = DEFAULT_BOUNDS[p_name]
                    nbi_df[p_name] = nbi_df[p_name].fillna((float(lo) + float(hi)) / 2.0)

            nbi_unique = nbi_df.drop_duplicates(subset=PARAM_NAMES).reset_index(drop=True)

            # Select top-K NBI candidates to evaluate (0 = all)
            requested_k = int(args.nbi_eval_k)
            if requested_k <= 0 or requested_k >= len(nbi_unique):
                nbi_eval = nbi_unique
            else:
                # Use predicted distance to utopia if predicted cols exist; otherwise just take head
                if "Pred_Score_Quality" in nbi_unique.columns and "Pred_Score_Cost" in nbi_unique.columns:
                    q_hi, c_hi = utopia
                    q_lo, c_lo = nadir

                    def _norm(v: float, lo: float, hi: float) -> float:
                        den = hi - lo
                        return 0.0 if abs(den) < 1e-12 else (v - lo) / den

                    tmp = nbi_unique.copy()

                    # ✅ Pylance-safe: avoid `.map` on inferred scalars
                    tmp["_q_norm"] = (
                        pd.Series(pd.to_numeric(tmp["Pred_Score_Quality"], errors="coerce"), index=tmp.index)
                        .apply(lambda v: _norm(float(v), q_lo, q_hi))
                        .clip(0.0, 1.0)
                    )
                    tmp["_c_norm"] = (
                        pd.Series(pd.to_numeric(tmp["Pred_Score_Cost"], errors="coerce"), index=tmp.index)
                        .apply(lambda v: _norm(float(v), c_lo, c_hi))
                        .clip(0.0, 1.0)
                    )

                    tmp["_dist2_utopia"] = (1.0 - tmp["_q_norm"]) ** 2 + (1.0 - tmp["_c_norm"]) ** 2
                    nbi_eval = tmp.nsmallest(requested_k, "_dist2_utopia").drop(
                        columns=["_q_norm", "_c_norm", "_dist2_utopia"]
                    )
                else:
                    nbi_eval = nbi_unique.head(requested_k)

            cand_params = [cast(Dict[str, float], row[PARAM_NAMES].to_dict()) for _, row in nbi_eval.iterrows()]

            # 5) Evaluate NBI candidates (real CV)
            t0 = time.perf_counter()
            best_nbi, all_nbi = evaluate_candidate_list(
                cast(List[Dict[str, float]], cand_params),
                X,
                y,
                seed=seed,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
                tree_method=args.tree_method,
                desc=f"Evaluating candidates ({len(cand_params)})",
            )
            stage_times["nbi_candidate_eval_seconds"] = float(time.perf_counter() - t0)

            save_csv_ptbr(pd.DataFrame(all_nbi), out_dir / "nbi_candidate_evaluations.csv")

            # Fairness budget for benchmarks
            doe_runs = int(len(doe_df))
            evaluated_candidates = int(len(cand_params))
            total_evals = int(doe_runs + evaluated_candidates)
            benchmark_budget = int(args.budget) if int(args.budget) > 0 else total_evals

            # ✅ Fill DOE+NBI bookkeeping columns
            opt_seconds = float(
                stage_times.get("doe_seconds", 0.0)
                + stage_times.get("fa_seconds", 0.0)
                + stage_times.get("rsm_seconds", 0.0)
                + stage_times.get("nbi_seconds", 0.0)
                + stage_times.get("nbi_candidate_eval_seconds", 0.0)
            )

            summary_rows: List[Dict[str, Any]] = []
            best_row = dict(best_nbi)
            best_row["method"] = "doe_nbi"
            best_row["budget"] = float(total_evals)
            best_row["Optimization_Time_Seconds"] = float(opt_seconds)
            best_row["Total_Time_Seconds"] = float(opt_seconds + float(best_row.get("Time_MeanFold", 0.0)))
            summary_rows.append(best_row)

            # Benchmarks
            summary_rows.append(coarse_grid_search(X, y, seed=seed, budget=benchmark_budget))
            summary_rows.append(random_search(X, y, seed=seed, budget=benchmark_budget))
            summary_rows.append(bayes_search(X, y, seed=seed, budget=benchmark_budget))
            summary_rows.append(hyperopt_tpe(X, y, seed=seed, budget=benchmark_budget))

            save_benchmark_summary(summary_rows, str(out_dir / "confirmation_summary.csv"))

            # Update manifest with stage times / counts
            write_manifest(
                out_dir / "manifest.json",
                replica=args.replica,
                seed=seed,
                dataset_path=dataset_path,
                design_path=design_path,
                extra={
                    "stage_times": stage_times,
                    "counts": {
                        "doe_runs": doe_runs,
                        "nbi_candidates_generated": int(len(nbi_df)),
                        "nbi_candidates_unique": int(len(nbi_unique)),
                        "nbi_candidates_evaluated": evaluated_candidates,
                        "total_evaluations_doe_plus_nbi": total_evals,
                        "benchmark_budget": benchmark_budget,
                    },
                },
            )


if __name__ == "__main__":
    main()
