#!/usr/bin/env python
from __future__ import annotations

"""Single-replica fairness pipeline (BankDataset).

Design goals for this script:
  - Keep the original repo methodology: DOE -> RSM -> NBI (optionally DOE-refine -> RSM -> NBI).
  - Avoid touching src/doe_xgb/nbi.py (to keep merge with main safe).
  - Provide progress bars (DOE, NBI evaluations, baselines).
  - Keep Pylance happy (explicit casts, correct nbi.py signature).
"""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from doe_xgb.config import DEFAULT_BOUNDS, INT_PARAMS, PARAM_NAMES
from doe_xgb.doe_runner_fairness import run_doe_fairness
from doe_xgb.io_utils import load_design, save_csv_ptbr
from doe_xgb.nbi import load_coefficients_csv, nbi_candidates_to_df, run_nbi_weighted_sum
from doe_xgb.rsm import fit_rsm_backward, save_rsm_coefficients
from doe_xgb.tracking import build_replica_dir, write_manifest


class Tee(IO[str]):
    """Tee stdout/stderr to both terminal and file."""

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


def _print_warning(msg: str) -> None:
    print(f"WARNING:root:{msg}")


def _drop_unknown_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows where any object column equals 'unknown' (case-insensitive)."""
    obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not obj_cols:
        return df, 0

    unknown_mask = pd.Series(False, index=df.index)
    for c in obj_cols:
        unknown_mask = unknown_mask | (df[c].astype(str).str.lower() == "unknown")

    removed = int(unknown_mask.sum())
    if removed > 0:
        df = df.loc[~unknown_mask].copy()
    return df, removed


def load_bank_dataset_from_repo(path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load UCI Bank Marketing and mimic the common AIF360 BankDataset preprocessing.

    Expected input file:
      - bank-additional-full.csv (separator=';')

    Preprocessing:
      - drop column 'duration' (leakage)
      - drop rows containing 'unknown' in any categorical column
      - target: y == 'yes' -> 1 else 0
      - protected attribute: age >= 25  (privileged=1)
      - one-hot encode categoricals
    """

    df = pd.read_csv(path, sep=";")
    if "y" not in df.columns:
        raise ValueError("Expected column 'y' in bank dataset.")
    if "age" not in df.columns:
        raise ValueError("Expected column 'age' in bank dataset.")

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])  # leakage feature

    df, removed = _drop_unknown_rows(df)
    if removed > 0:
        _print_warning(f"Missing Data: {removed} rows removed from BankDataset.")

    y = (df["y"].astype(str).str.lower() == "yes").astype(int)

    # privileged group (binary)
    protected = (df["age"].astype(int) >= 25).astype(int)

    X = df.drop(columns=["y"]).copy()
    X = pd.get_dummies(X, drop_first=False)
    return X, y, protected


def _minmax_norm_from_ref(values: pd.Series, ref: pd.Series) -> pd.Series:
    mn, mx = float(ref.min()), float(ref.max())
    if mx - mn < 1e-12:
        return pd.Series(np.ones(len(values)), index=values.index, dtype="float64")
    out = (values - mn) / (mx - mn)
    return out.clip(lower=-10.0, upper=10.0)


def select_best_pareto_utopia(
    df: pd.DataFrame,
    *,
    obj_cols: Tuple[str, str] = ("BalancedAccuracy_Mean", "FairnessScore_1_minus_Bias"),
    quality_floor: Optional[float] = None,
    quality_col: str = "BalancedAccuracy_Mean",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Pareto filter (maximize both objs) + closest to utopia.

    Returns:
      - best_row: pd.Series
      - diag_df: original dataframe plus diagnostics columns:
          Pass_QualityFloor, Is_Pareto, A_norm, B_norm, Utopia_Distance
    """

    a, b = obj_cols
    d = df.copy()

    if quality_floor is None:
        d["Pass_QualityFloor"] = True
        work = d.copy()
    else:
        d["Pass_QualityFloor"] = d[quality_col].astype(float) >= float(quality_floor)
        work = d[d["Pass_QualityFloor"]].copy()
        if work.empty:
            # Never crash: if floor too high, fallback to all.
            work = d.copy()

    # norms computed with reference = work (to avoid degenerate low-quality points dominating min-max)
    d["A_norm"] = _minmax_norm_from_ref(d[a].astype(float), work[a].astype(float))
    d["B_norm"] = _minmax_norm_from_ref(d[b].astype(float), work[b].astype(float))

    # Pareto on work subset
    vals = work[[a, b]].to_numpy(dtype=float)
    n = vals.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (vals[j, 0] >= vals[i, 0] and vals[j, 1] >= vals[i, 1]) and (
                vals[j, 0] > vals[i, 0] or vals[j, 1] > vals[i, 1]
            ):
                is_nd[i] = False
                break

    d["Is_Pareto"] = False
    pareto_idx = work.index[is_nd]
    d.loc[pareto_idx, "Is_Pareto"] = True

    d["Utopia_Distance"] = np.sqrt((1.0 - d["A_norm"]) ** 2 + (1.0 - d["B_norm"]) ** 2)

    # Choose best among Pareto points that pass quality floor (if any)
    cand = d[d["Is_Pareto"] & d["Pass_QualityFloor"]].copy()
    if cand.empty:
        cand = d[d["Is_Pareto"]].copy()
    if cand.empty:
        cand = d.copy()

    best_idx = cast(int, cand["Utopia_Distance"].astype(float).idxmin())
    best_any = d.loc[best_idx]
    best = cast(pd.Series, best_any)  # Pylance: .loc can be Series|DataFrame
    return best, d


def _flatten_nbi_params(df_nbi: pd.DataFrame) -> pd.DataFrame:
    """Expand nbi_candidates_to_df(hyperparameters=dict) into PARAM_NAMES columns."""
    d = df_nbi.copy()
    if "hyperparameters" not in d.columns:
        raise KeyError("Expected column 'hyperparameters' in NBI candidates dataframe")

    hps = d["hyperparameters"].apply(lambda x: x if isinstance(x, dict) else {})
    hp_df = pd.DataFrame(list(hps))
    for p in PARAM_NAMES:
        if p not in hp_df.columns:
            hp_df[p] = np.nan

    out = pd.concat([d.drop(columns=["hyperparameters"]), hp_df[PARAM_NAMES]], axis=1)
    return out


def _pick_evenly(df: pd.DataFrame, k: int, *, sort_col: str) -> pd.DataFrame:
    if k <= 0 or k >= len(df):
        return df.copy()
    d = df.sort_values(sort_col).reset_index(drop=True)
    idx = np.linspace(0, len(d) - 1, k)
    idx_i = np.unique(np.round(idx).astype(int))
    return d.iloc[idx_i].copy()


def _sample_random_configs(
    n: int,
    *,
    seed: int,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    for _ in range(int(n)):
        cfg: Dict[str, Any] = {}
        for p in PARAM_NAMES:
            lo, hi = bounds[p]
            v = float(rng.uniform(lo, hi))
            if p in INT_PARAMS:
                v = int(round(v))
            cfg[p] = v
        rows.append(cfg)
    return pd.DataFrame(rows)


def _sample_refine_configs(
    anchor_points: pd.DataFrame,
    *,
    n_samples: int,
    seed: int,
    sigma_frac: float = 0.10,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
) -> pd.DataFrame:
    """Gaussian local sampling around top points."""
    rng = np.random.default_rng(int(seed))
    anchors = anchor_points.copy()

    # Ensure anchor points have all params
    for p in PARAM_NAMES:
        if p not in anchors.columns:
            raise KeyError(f"Anchor points missing param column: {p}")

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    # Precompute sigmas
    sigmas = {p: sigma_frac * (bounds[p][1] - bounds[p][0]) for p in PARAM_NAMES}

    # Round-robin across anchors
    anchor_list = anchors.reset_index(drop=True)
    if anchor_list.empty:
        return pd.DataFrame(columns=PARAM_NAMES)

    i = 0
    attempts = 0
    max_attempts = max(10_000, n_samples * 50)
    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        base = anchor_list.iloc[i % len(anchor_list)]
        i += 1

        cfg: Dict[str, Any] = {}
        for p in PARAM_NAMES:
            mu = float(base[p])
            lo, hi = bounds[p]
            v = float(rng.normal(mu, sigmas[p]))
            v = float(np.clip(v, lo, hi))
            if p in INT_PARAMS:
                v = int(round(v))
            cfg[p] = v

        key = tuple(cfg[p] for p in PARAM_NAMES)
        if key in seen:
            continue
        seen.add(key)
        rows.append(cfg)

    return pd.DataFrame(rows)


def _range_for_nbi(
    df: pd.DataFrame,
    *,
    quality_floor: float,
    q_col: str = "BalancedAccuracy_Mean",
    f_col: str = "FairnessScore_1_minus_Bias",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Pick observed (utopia, nadir) used for NBI normalization."""
    work = df[df[q_col].astype(float) >= float(quality_floor)].copy()
    if work.empty:
        work = df.copy()

    utopia = (float(work[q_col].max()), float(work[f_col].max()))
    nadir = (float(work[q_col].min()), float(work[f_col].min()))
    return utopia, nadir


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)

    p = argparse.ArgumentParser(description="Run fairness pipeline for one replica (BankDataset)")
    p.add_argument("--dataset", default=os.getenv("FAIRNESS_DATASET_PATH", "data/source/bank/bank-additional-full.csv"))
    p.add_argument("--design", default=os.getenv("DESIGN_PATH", "data/design/hyperparameter_design.csv"))
    p.add_argument("--replica", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_root", default="experiments")

    # NBI
    p.add_argument("--beta-step", type=float, default=0.01)
    p.add_argument("--nbi-eval-k", type=int, default=60)
    p.add_argument("--nbi-n-starts", type=int, default=50)
    p.add_argument("--nbi-constrain-pred-range", action="store_true")

    # CV / model
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--tree-method", default="hist")
    p.add_argument("--auto-scale-pos-weight", action="store_true")
    p.add_argument("--no-auto-scale-pos-weight", action="store_true")
    p.add_argument("--stratify-by-group", action="store_true")

    # Selection safeguards
    p.add_argument("--quality-floor", type=float, default=0.55)
    p.add_argument("--nbi-range-quality-floor", type=float, default=0.55)

    # Stage2 refinement (optional)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--refine-top-m", type=int, default=25)
    p.add_argument("--refine-n-samples", type=int, default=40)
    p.add_argument("--refine-sigma-frac", type=float, default=0.10)

    # Baselines
    p.add_argument("--run-baselines", action="store_true")

    args = p.parse_args()

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)
    seed = int(args.seed if args.seed is not None else args.replica)

    out_dir = build_replica_dir(out_root, dataset_path, design_path, int(args.replica))

    auto_spw = True
    if args.no_auto_scale_pos_weight:
        auto_spw = False
    if args.auto_scale_pos_weight:
        auto_spw = True

    write_manifest(
        out_dir / "manifest.json",
        replica=int(args.replica),
        seed=seed,
        dataset_path=dataset_path,
        design_path=design_path,
        extra={
            "args": {
                "beta_step": float(args.beta_step),
                "nbi_eval_k": int(args.nbi_eval_k),
                "nbi_n_starts": int(args.nbi_n_starts),
                "nbi_constrain_pred_range": bool(args.nbi_constrain_pred_range),
                "n_splits": int(args.n_splits),
                "n_jobs": int(args.n_jobs),
                "tree_method": str(args.tree_method),
                "auto_scale_pos_weight": bool(auto_spw),
                "stratify_by_group": bool(args.stratify_by_group),
                "quality_floor": float(args.quality_floor),
                "nbi_range_quality_floor": float(args.nbi_range_quality_floor),
                "refine": bool(args.refine),
                "refine_top_m": int(args.refine_top_m),
                "refine_n_samples": int(args.refine_n_samples),
                "refine_sigma_frac": float(args.refine_sigma_frac),
                "run_baselines": bool(args.run_baselines),
            }
        },
    )

    log_path = out_dir / "run_fairness_replica.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            stage_times: Dict[str, float] = {}

            # ------------------------------------------------------------
            # Data
            # ------------------------------------------------------------
            X, y, protected = load_bank_dataset_from_repo(dataset_path)
            design_df = load_design(design_path)

            # ------------------------------------------------------------
            # Stage 1: DOE -> RSM -> NBI -> evaluate NBI
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            doe_df = run_doe_fairness(
                design_df=design_df,
                X=X,
                y=y,
                protected=protected,
                seed=seed,
                n_splits=int(args.n_splits),
                n_jobs=int(args.n_jobs),
                tree_method=str(args.tree_method),
                auto_scale_pos_weight=bool(auto_spw),
                stratify_by_group=bool(args.stratify_by_group),
                privileged_value=1,
                desc="DOE runs (fairness)",
            )
            stage_times["doe_seconds"] = float(time.perf_counter() - t0)
            save_csv_ptbr(doe_df, out_dir / "doe_results_fairness.csv")

            # Fit RSMs
            t0 = time.perf_counter()
            model_q = fit_rsm_backward(doe_df[PARAM_NAMES], doe_df["BalancedAccuracy_Mean"], response_name="BalancedAccuracy_Mean")
            model_f = fit_rsm_backward(
                doe_df[PARAM_NAMES],
                doe_df["FairnessScore_1_minus_Bias"],
                response_name="FairnessScore_1_minus_Bias",
            )
            stage_times["rsm_seconds"] = float(time.perf_counter() - t0)

            save_rsm_coefficients(model_q, str(out_dir / "rsm_coefficients_balanced_accuracy.csv"))
            save_rsm_coefficients(model_f, str(out_dir / "rsm_coefficients_fairness_score.csv"))

            # NBI normalization range (filter by quality floor)
            utopia, nadir = _range_for_nbi(
                doe_df,
                quality_floor=float(args.nbi_range_quality_floor),
            )

            t0 = time.perf_counter()
            nbi_candidates_1 = run_nbi_weighted_sum(
                load_coefficients_csv(str(out_dir / "rsm_coefficients_balanced_accuracy.csv")),
                load_coefficients_csv(str(out_dir / "rsm_coefficients_fairness_score.csv")),
                observed_utopia=utopia,
                observed_nadir=nadir,
                beta_step=float(args.beta_step),
                seed=int(seed),
                n_starts=int(args.nbi_n_starts),
                constrain_pred_range=bool(args.nbi_constrain_pred_range),
            )
            stage_times["nbi_stage1_seconds"] = float(time.perf_counter() - t0)

            df_nbi_1_raw = nbi_candidates_to_df(nbi_candidates_1)
            df_nbi_1 = _flatten_nbi_params(df_nbi_1_raw)
            df_nbi_1 = df_nbi_1.drop_duplicates(subset=PARAM_NAMES).reset_index(drop=True)
            save_csv_ptbr(df_nbi_1, out_dir / "nbi_candidates_stage1_fairness.csv")

            # Pick K to evaluate (evenly across beta_2)
            nbi_evalset_1 = _pick_evenly(df_nbi_1, int(args.nbi_eval_k), sort_col="beta_2")
            nbi_evalset_1_params = cast(pd.DataFrame, nbi_evalset_1[PARAM_NAMES].copy())

            t0 = time.perf_counter()
            nbi_eval_1 = run_doe_fairness(
                design_df=nbi_evalset_1_params,
                X=X,
                y=y,
                protected=protected,
                seed=seed + 10_000,
                n_splits=int(args.n_splits),
                n_jobs=int(args.n_jobs),
                tree_method=str(args.tree_method),
                auto_scale_pos_weight=bool(auto_spw),
                stratify_by_group=bool(args.stratify_by_group),
                privileged_value=1,
                desc=f"Evaluating NBI stage1 candidates ({len(nbi_evalset_1_params)})",
            )
            stage_times["nbi_stage1_eval_seconds"] = float(time.perf_counter() - t0)
            save_csv_ptbr(nbi_eval_1, out_dir / "nbi_evaluated_stage1_fairness.csv")

            # Stage1 combined selection
            stage1_all = pd.concat(
                [
                    doe_df.assign(Method="DOE+RSM"),
                    nbi_eval_1.assign(Method="DOE+RSM+NBI_stage1"),
                ],
                ignore_index=True,
            )
            best_stage1, stage1_diag = select_best_pareto_utopia(
                stage1_all,
                quality_floor=float(args.quality_floor),
            )
            save_csv_ptbr(stage1_diag, out_dir / "all_evaluated_stage1_fairness.csv")
            save_csv_ptbr(pd.DataFrame([best_stage1.to_dict()]), out_dir / "best_solution_stage1_fairness.csv")

            # ------------------------------------------------------------
            # Stage 2 (optional): local refinement DOE -> RSM -> NBI
            # ------------------------------------------------------------
            refine_eval = pd.DataFrame()
            nbi_eval_2 = pd.DataFrame()

            if bool(args.refine):
                pareto = stage1_diag[stage1_diag["Is_Pareto"] & stage1_diag["Pass_QualityFloor"]].copy()
                if pareto.empty:
                    pareto = stage1_diag[stage1_diag["Pass_QualityFloor"]].copy()
                if pareto.empty:
                    pareto = stage1_diag.copy()

                pareto = pareto.sort_values("Utopia_Distance", ascending=True)
                anchors = pareto.head(int(args.refine_top_m)).copy()

                refine_design = _sample_refine_configs(
                    anchors,
                    n_samples=int(args.refine_n_samples),
                    seed=seed + 20_000,
                    sigma_frac=float(args.refine_sigma_frac),
                )

                if not refine_design.empty:
                    t0 = time.perf_counter()
                    refine_eval = run_doe_fairness(
                        design_df=refine_design,
                        X=X,
                        y=y,
                        protected=protected,
                        seed=seed + 30_000,
                        n_splits=int(args.n_splits),
                        n_jobs=int(args.n_jobs),
                        tree_method=str(args.tree_method),
                        auto_scale_pos_weight=bool(auto_spw),
                        stratify_by_group=bool(args.stratify_by_group),
                        privileged_value=1,
                        desc=f"Evaluating DOE stage2 refinement ({len(refine_design)})",
                    )
                    stage_times["refine_eval_seconds"] = float(time.perf_counter() - t0)
                    save_csv_ptbr(refine_eval, out_dir / "doe_refine_stage2_fairness.csv")

                stage2_doe = pd.concat([doe_df, refine_eval], ignore_index=True) if not refine_eval.empty else doe_df.copy()

                # Refit RSM
                t0 = time.perf_counter()
                model_q2 = fit_rsm_backward(stage2_doe[PARAM_NAMES], stage2_doe["BalancedAccuracy_Mean"], response_name="BalancedAccuracy_Mean")
                model_f2 = fit_rsm_backward(
                    stage2_doe[PARAM_NAMES],
                    stage2_doe["FairnessScore_1_minus_Bias"],
                    response_name="FairnessScore_1_minus_Bias",
                )
                stage_times["rsm_stage2_seconds"] = float(time.perf_counter() - t0)

                save_rsm_coefficients(model_q2, str(out_dir / "rsm_coefficients_balanced_accuracy_stage2.csv"))
                save_rsm_coefficients(model_f2, str(out_dir / "rsm_coefficients_fairness_score_stage2.csv"))

                utopia2, nadir2 = _range_for_nbi(stage2_doe, quality_floor=float(args.nbi_range_quality_floor))

                t0 = time.perf_counter()
                nbi_candidates_2 = run_nbi_weighted_sum(
                    load_coefficients_csv(str(out_dir / "rsm_coefficients_balanced_accuracy_stage2.csv")),
                    load_coefficients_csv(str(out_dir / "rsm_coefficients_fairness_score_stage2.csv")),
                    observed_utopia=utopia2,
                    observed_nadir=nadir2,
                    beta_step=float(args.beta_step),
                    seed=int(seed + 40_000),
                    n_starts=int(args.nbi_n_starts),
                    constrain_pred_range=bool(args.nbi_constrain_pred_range),
                )
                stage_times["nbi_stage2_seconds"] = float(time.perf_counter() - t0)

                df_nbi_2_raw = nbi_candidates_to_df(nbi_candidates_2)
                df_nbi_2 = _flatten_nbi_params(df_nbi_2_raw)
                df_nbi_2 = df_nbi_2.drop_duplicates(subset=PARAM_NAMES).reset_index(drop=True)
                save_csv_ptbr(df_nbi_2, out_dir / "nbi_candidates_stage2_fairness.csv")

                nbi_evalset_2 = _pick_evenly(df_nbi_2, int(args.nbi_eval_k), sort_col="beta_2")
                nbi_evalset_2_params = cast(pd.DataFrame, nbi_evalset_2[PARAM_NAMES].copy())

                t0 = time.perf_counter()
                nbi_eval_2 = run_doe_fairness(
                    design_df=nbi_evalset_2_params,
                    X=X,
                    y=y,
                    protected=protected,
                    seed=seed + 50_000,
                    n_splits=int(args.n_splits),
                    n_jobs=int(args.n_jobs),
                    tree_method=str(args.tree_method),
                    auto_scale_pos_weight=bool(auto_spw),
                    stratify_by_group=bool(args.stratify_by_group),
                    privileged_value=1,
                    desc=f"Evaluating NBI stage2 candidates ({len(nbi_evalset_2_params)})",
                )
                stage_times["nbi_stage2_eval_seconds"] = float(time.perf_counter() - t0)
                save_csv_ptbr(nbi_eval_2, out_dir / "nbi_evaluated_stage2_fairness.csv")

            # ------------------------------------------------------------
            # Final selection (across all evaluated points)
            # ------------------------------------------------------------
            parts = [doe_df.assign(Method="DOE+RSM")]
            if not nbi_eval_1.empty:
                parts.append(nbi_eval_1.assign(Method="DOE+RSM+NBI_stage1"))
            if not refine_eval.empty:
                parts.append(refine_eval.assign(Method="DOE_refine_stage2"))
            if not nbi_eval_2.empty:
                parts.append(nbi_eval_2.assign(Method="DOE+RSM+NBI_stage2"))

            all_eval = pd.concat(parts, ignore_index=True)
            best_overall, all_diag = select_best_pareto_utopia(all_eval, quality_floor=float(args.quality_floor))
            save_csv_ptbr(all_diag, out_dir / "all_evaluated_candidates_fairness.csv")
            save_csv_ptbr(pd.DataFrame([best_overall.to_dict()]), out_dir / "best_solution_fairness.csv")

            # ------------------------------------------------------------
            # Baselines (optional)
            # ------------------------------------------------------------
            comparison_rows: List[Dict[str, Any]] = []

            def _row_from_best(tag: str, best_series: pd.Series) -> Dict[str, Any]:
                dct = best_series.to_dict()
                return {
                    "Method": tag,
                    "BalancedAccuracy_Mean": float(dct.get("BalancedAccuracy_Mean", 0.0)),
                    "BiasMean_Mean": float(dct.get("BiasMean_Mean", 0.0)),
                    "FairnessScore_1_minus_Bias": float(dct.get("FairnessScore_1_minus_Bias", 0.0)),
                }

            comparison_rows.append(_row_from_best("DOE+RSM+NBI" + ("_2stage" if args.refine else ""), best_overall))

            if bool(args.run_baselines):
                # Budget parity with our method
                budget = int(len(all_eval))

                # XGB default
                xgb_default_df = pd.DataFrame([{}])
                xgb_default_eval = run_doe_fairness(
                    design_df=xgb_default_df,
                    X=X,
                    y=y,
                    protected=protected,
                    seed=seed + 60_000,
                    n_splits=int(args.n_splits),
                    n_jobs=int(args.n_jobs),
                    tree_method=str(args.tree_method),
                    auto_scale_pos_weight=bool(auto_spw),
                    stratify_by_group=bool(args.stratify_by_group),
                    privileged_value=1,
                    desc="Baseline: XGB default",
                )
                best_xgb = cast(pd.Series, xgb_default_eval.iloc[0])
                comparison_rows.append(_row_from_best("XGB_Default", best_xgb))

                # Random Search baseline (same budget)
                rs_design = _sample_random_configs(budget, seed=seed + 70_000)
                rs_eval = run_doe_fairness(
                    design_df=rs_design,
                    X=X,
                    y=y,
                    protected=protected,
                    seed=seed + 80_000,
                    n_splits=int(args.n_splits),
                    n_jobs=int(args.n_jobs),
                    tree_method=str(args.tree_method),
                    auto_scale_pos_weight=bool(auto_spw),
                    stratify_by_group=bool(args.stratify_by_group),
                    privileged_value=1,
                    desc=f"Baseline: Random Search (N={budget})",
                )
                best_rs, rs_diag = select_best_pareto_utopia(rs_eval, quality_floor=float(args.quality_floor))
                save_csv_ptbr(rs_diag, out_dir / "baseline_random_search_fairness.csv")
                comparison_rows.append(_row_from_best(f"RandomSearch_N={budget}", best_rs))

            comp = pd.DataFrame(comparison_rows)
            save_csv_ptbr(comp, out_dir / "comparison_summary_fairness.csv")

            print("\n=== Comparison summary (best points) ===")
            cols_show = ["Method", "BalancedAccuracy_Mean", "BiasMean_Mean", "FairnessScore_1_minus_Bias"]
            print(comp[cols_show].to_string(index=False))

            # ------------------------------------------------------------
            # Finish
            # ------------------------------------------------------------
            save_csv_ptbr(pd.DataFrame([stage_times]), out_dir / "stage_times_fairness.csv")

            print("\n✅ Fairness replica pipeline finished")
            print(f"- out_dir: {out_dir}")
            print(f"- log: {log_path}")


if __name__ == "__main__":
    main()
