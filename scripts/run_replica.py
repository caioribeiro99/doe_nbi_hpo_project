#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import contextlib
import math
import os
import sys
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

    # ✅ FIX: write manifest to FILE, not directory
    write_manifest(
        out_dir / "manifest.json",
        replica=args.replica,
        seed=seed,
        dataset_path=dataset_path,
        design_path=design_path,
    )

    log_path = out_dir / "run_replica.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            X, y = load_dataset(
                dataset_path,
                target_col=args.target,
                target_map={"g": 0, "h": 1},
            )

            design_df = load_design(design_path)

            doe_df = run_doe(
                design_df,
                X,
                y,
                seed=seed,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
                tree_method=args.tree_method,
            )
            save_csv_ptbr(doe_df, out_dir / "doe_results.csv")

            fa_res = run_factor_analysis(doe_df)
            save_fa_outputs(
                fa_res,
                str(out_dir / "factor_loadings.csv"),
                str(out_dir / "factor_scores.csv"),
            )

            doe_scored = doe_df.join(fa_res.scores)

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

            save_rsm_coefficients(model_q, str(out_dir / "rsm_coefficients_quality.csv"))
            save_rsm_coefficients(model_c, str(out_dir / "rsm_coefficients_cost.csv"))

            utopia = (float(doe_scored["Score_Quality"].max()), float(doe_scored["Score_Cost"].max()))
            nadir = (float(doe_scored["Score_Quality"].min()), float(doe_scored["Score_Cost"].min()))

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
            save_nbi_candidates(candidates, str(out_dir / "nbi_candidates.csv"))

            nbi_df = nbi_candidates_to_df(candidates)
            nbi_df = _ensure_param_columns(nbi_df, PARAM_NAMES)

            for p_name in PARAM_NAMES:
                nbi_df[p_name] = _coerce_float_series(nbi_df[p_name])

            for p_name in PARAM_NAMES:
                if nbi_df[p_name].isna().any():
                    lo, hi = DEFAULT_BOUNDS[p_name]
                    nbi_df[p_name] = nbi_df[p_name].fillna((float(lo) + float(hi)) / 2.0)

            nbi_unique = nbi_df.drop_duplicates(subset=PARAM_NAMES)
            cand_params = [cast(Dict[str, float], row[PARAM_NAMES].to_dict()) for _, row in nbi_unique.iterrows()]

            best_nbi, _all_nbi = evaluate_candidate_list(
                cast(List[Dict[str, float]], cand_params),
                X,
                y,
                seed=seed,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
                tree_method=args.tree_method,
            )

            summary_rows: List[Dict[str, Any]] = []
            best_row = dict(best_nbi)
            best_row["method"] = "doe_nbi"
            summary_rows.append(best_row)

            budget = args.budget if args.budget > 0 else len(doe_df) + len(cand_params)
            summary_rows.append(coarse_grid_search(X, y, seed=seed, budget=budget))
            summary_rows.append(random_search(X, y, seed=seed, budget=budget))
            summary_rows.append(bayes_search(X, y, seed=seed, budget=budget))
            summary_rows.append(hyperopt_tpe(X, y, seed=seed, budget=budget))

            save_benchmark_summary(summary_rows, str(out_dir / "confirmation_summary.csv"))


if __name__ == "__main__":
    main()
