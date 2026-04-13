#!/usr/bin/env python
from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from doe_xgb.config import FAIRNESS_DEFAULT_BOUNDS, FAIRNESS_PARAM_NAMES, INT_PARAMS  # noqa: E402
from doe_xgb.doe_runner_fairness import run_doe_fairness  # noqa: E402
from doe_xgb.io_utils import load_design, save_csv_ptbr  # noqa: E402
from doe_xgb.nbi import load_coefficients_csv, nbi_candidates_to_df  # noqa: E402
from doe_xgb.nbi_fairness import run_nbi_weighted_sum_fairness  # noqa: E402
from doe_xgb.rsm import fit_rsm_backward, save_rsm_coefficients  # noqa: E402
from doe_xgb.tracking import build_replica_dir, write_manifest  # noqa: E402
from doe_xgb.fairness_dataset_utils import (  # noqa: E402
    load_bank_dataset,
    load_credit_card_default_dataset,
    load_generic_fairness_dataset,
)

QUALITY_OBJ_COL = "BalancedAccuracy_Mean"
FAIRNESS_OBJ_COL = "FairnessScore_DI_Only"
FAIRNESS_BIAS_COL = "Bias_DI_Mean"
LEGACY_FAIRNESS_OBJ_COL = "FairnessScore_1_minus_Bias"
LEGACY_FAIRNESS_BIAS_COL = "BiasMean_Mean"


class Tee(IO[str]):
    """Tee stdout/stderr to both terminal and a file."""

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
    """Load UCI Bank Marketing and mimic common fairness preprocessing.

    Expected input:
      - bank-additional-full.csv (separator=';')

    Preprocessing:
      - drop column 'duration' (leakage)
      - drop rows containing 'unknown' in any categorical column
      - target: y == 'yes' -> 1 else 0
      - protected attribute: age >= 25 (privileged=1)
      - one-hot encode categoricals
    """
    df = pd.read_csv(path, sep=";")
    if "y" not in df.columns:
        raise ValueError("Expected column 'y' in bank dataset.")
    if "age" not in df.columns:
        raise ValueError("Expected column 'age' in bank dataset.")

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    df, removed = _drop_unknown_rows(df)
    if removed > 0:
        _print_warning(f"Missing Data: {removed} rows removed from BankDataset.")

    y = (df["y"].astype(str).str.lower() == "yes").astype(int)
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
    obj_cols: Tuple[str, str] = (QUALITY_OBJ_COL, FAIRNESS_OBJ_COL),
    quality_floor: Optional[float] = None,
    quality_col: str = QUALITY_OBJ_COL,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Pareto filter (maximize both objs) + closest to utopia."""
    a, b = obj_cols
    d = df.copy()

    # 1) Apply quality floor for selection (never crash: fallback to full set if empty)
    if quality_floor is None:
        d["Pass_QualityFloor"] = True
        work = d.copy()
    else:
        d["Pass_QualityFloor"] = d[quality_col].astype(float) >= float(quality_floor)
        work = d[d["Pass_QualityFloor"]].copy()
        if work.empty:
            work = d.copy()

    # 2) Normalize objectives (min-max) using reference = work
    d["A_norm"] = _minmax_norm_from_ref(d[a].astype(float), work[a].astype(float))
    d["B_norm"] = _minmax_norm_from_ref(d[b].astype(float), work[b].astype(float))

    # 3) Pareto filter on 'work'
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

    # 4) Utopia distance on normalized objectives
    d["Utopia_Distance"] = np.sqrt((1.0 - d["A_norm"]) ** 2 + (1.0 - d["B_norm"]) ** 2)

    cand = d[d["Is_Pareto"] & d["Pass_QualityFloor"]].copy()
    if cand.empty:
        cand = d[d["Is_Pareto"]].copy()
    if cand.empty:
        cand = d.copy()

    best_idx = cast(int, cand["Utopia_Distance"].astype(float).idxmin())
    best = cast(pd.Series, d.loc[best_idx])
    return best, d


def select_best_fairness_subject_to_ba(
    diag_df: pd.DataFrame,
    *,
    ba_floor: float,
    pareto_only: bool = True,
    ba_col: str = "BalancedAccuracy_Mean",
    bias_col: str = FAIRNESS_BIAS_COL,
) -> pd.Series:
    """Pick minimum bias subject to BA >= floor."""
    d = diag_df.copy()
    d[ba_col] = d[ba_col].astype(float)
    d[bias_col] = d[bias_col].astype(float)

    base = d[d["Is_Pareto"]].copy() if pareto_only and "Is_Pareto" in d.columns else d.copy()
    cand = base[base[ba_col] >= float(ba_floor)].copy()

    if cand.empty:
        cand = d[d[ba_col] >= float(ba_floor)].copy()

    if cand.empty:
        cand = base.copy() if not base.empty else d.copy()

    idx = cast(int, cand[bias_col].idxmin())
    return cast(pd.Series, d.loc[idx])


def select_refine_anchors(
    stage1_diag: pd.DataFrame,
    *,
    top_m: int,
    strategy: str = "mixed",
    mix: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    quality_col: str = QUALITY_OBJ_COL,
    fairness_col: str = FAIRNESS_OBJ_COL,
) -> pd.DataFrame:
    """Select anchor points for stage2 refinement.

    strategy:
      - 'utopia': legacy behavior (closest-to-utopia on Pareto)
      - 'mixed': combine:
          * closest-to-utopia
          * highest fairness
          * highest BA

    mix:
      Fractions for (utopia, fairness, ba). They will be normalized to sum=1.
    """
    if top_m <= 0:
        return stage1_diag.head(0).copy()

    d = stage1_diag.copy()

    # Candidate pool: Pareto + pass quality floor (fallbacks)
    pool = d
    if "Is_Pareto" in d.columns and "Pass_QualityFloor" in d.columns:
        pool = d[d["Is_Pareto"] & d["Pass_QualityFloor"]].copy()
        if pool.empty:
            pool = d[d["Pass_QualityFloor"]].copy()
        if pool.empty:
            pool = d[d["Is_Pareto"]].copy()
    elif "Is_Pareto" in d.columns:
        pool = d[d["Is_Pareto"]].copy()

    if pool.empty:
        pool = d.copy()

    # Ensure numeric
    for c in ["Utopia_Distance", quality_col, fairness_col]:
        if c in pool.columns:
            pool[c] = pool[c].astype(float)

    pool = pool.dropna(subset=[quality_col, fairness_col]).copy()

    if pool.empty:
        return d.head(min(int(top_m), len(d))).copy()

    top_m_eff = min(int(top_m), int(len(pool)))

    if strategy == "utopia":
        anchors = pool.sort_values("Utopia_Distance", ascending=True).head(top_m_eff).copy()
        anchors["AnchorGroup"] = "utopia"
        return anchors

    # Normalize mix
    fu, ff, fq = (float(mix[0]), float(mix[1]), float(mix[2]))
    s = fu + ff + fq
    if s <= 1e-12:
        fu, ff, fq = 0.4, 0.4, 0.2
        s = fu + ff + fq
    fu, ff, fq = fu / s, ff / s, fq / s

    # Quotas
    k_fair = max(1, int(round(top_m_eff * ff)))
    k_ba = max(1, int(round(top_m_eff * fq)))
    k_utopia = top_m_eff - k_fair - k_ba
    if k_utopia < 1:
        # Borrow from the largest bucket
        if k_fair >= k_ba and k_fair > 1:
            k_fair -= 1
        elif k_ba > 1:
            k_ba -= 1
        k_utopia = top_m_eff - k_fair - k_ba
        if k_utopia < 1:
            k_utopia = 1
            k_fair = max(1, top_m_eff - 1 - k_ba)

    # Orders (tie-breakers help stability)
    utopia_order = pool.sort_values(
        ["Utopia_Distance", fairness_col, quality_col],
        ascending=[True, False, False],
    ).index.tolist()
    fairness_order = pool.sort_values(
        [fairness_col, quality_col, "Utopia_Distance"],
        ascending=[False, False, True],
    ).index.tolist()
    ba_order = pool.sort_values(
        [quality_col, fairness_col, "Utopia_Distance"],
        ascending=[False, False, True],
    ).index.tolist()

    chosen_set: set[Any] = set()
    group_map: Dict[Any, str] = {}
    chosen: List[Any] = []

    def take(order: List[Any], k: int, label: str) -> None:
        if k <= 0:
            return
        taken = 0
        for idx in order:
            if idx in chosen_set:
                continue
            chosen_set.add(idx)
            group_map[idx] = label
            chosen.append(idx)
            taken += 1
            if taken >= k:
                break

    take(utopia_order, k_utopia, "utopia")
    take(fairness_order, k_fair, "fairness")
    take(ba_order, k_ba, "quality")

    # Fill remainder (prefer utopia -> fairness -> quality)
    while len(chosen) < top_m_eff:
        added = False
        for order, label in [
            (utopia_order, "fill_utopia"),
            (fairness_order, "fill_fairness"),
            (ba_order, "fill_quality"),
        ]:
            for idx in order:
                if idx in chosen_set:
                    continue
                chosen_set.add(idx)
                group_map[idx] = label
                chosen.append(idx)
                added = True
                break
            if len(chosen) >= top_m_eff:
                break
        if not added:
            break

    anchors = pool.loc[chosen].copy()
    anchors["AnchorGroup"] = anchors.index.map(lambda idx: group_map.get(idx, "fill"))
    return anchors


def _flatten_nbi_params(df_nbi: pd.DataFrame) -> pd.DataFrame:
    d = df_nbi.copy()
    if "hyperparameters" not in d.columns:
        raise KeyError("Expected column 'hyperparameters' in NBI candidates dataframe")

    hps = d["hyperparameters"].apply(lambda x: x if isinstance(x, dict) else {})
    hp_df = pd.DataFrame(list(hps))
    for p in FAIRNESS_PARAM_NAMES:
        if p not in hp_df.columns:
            hp_df[p] = np.nan

    out = pd.concat([d.drop(columns=["hyperparameters"]), hp_df[FAIRNESS_PARAM_NAMES]], axis=1)
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
    bounds: Dict[str, Tuple[float, float]] = FAIRNESS_DEFAULT_BOUNDS,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    for _ in range(int(n)):
        cfg: Dict[str, Any] = {}
        for p in FAIRNESS_PARAM_NAMES:
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
    bounds: Dict[str, Tuple[float, float]] = FAIRNESS_DEFAULT_BOUNDS,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    anchors = anchor_points.copy()

    for p in FAIRNESS_PARAM_NAMES:
        if p not in anchors.columns:
            raise KeyError(f"Anchor points missing param column: {p}")

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    sigmas = {p: sigma_frac * (bounds[p][1] - bounds[p][0]) for p in FAIRNESS_PARAM_NAMES}

    anchor_list = anchors.reset_index(drop=True)
    if anchor_list.empty:
        return pd.DataFrame(columns=FAIRNESS_PARAM_NAMES)

    i = 0
    attempts = 0
    max_attempts = max(10_000, n_samples * 50)
    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        base = anchor_list.iloc[i % len(anchor_list)]
        i += 1

        cfg: Dict[str, Any] = {}
        for p in FAIRNESS_PARAM_NAMES:
            mu = float(base[p])
            lo, hi = bounds[p]
            v = float(rng.normal(mu, sigmas[p]))
            v = float(np.clip(v, lo, hi))
            if p in INT_PARAMS:
                v = int(round(v))
            cfg[p] = v

        key = tuple(cfg[p] for p in FAIRNESS_PARAM_NAMES)
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
    f_col: str = FAIRNESS_OBJ_COL,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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

    # Dataset loader controls
    p.add_argument("--dataset-kind", choices=["bank", "credit_card_default", "generic"], default="bank")
    p.add_argument("--target-col", default="y")
    p.add_argument("--target-positive", default=None)
    p.add_argument("--protected-col", default=None)
    p.add_argument("--protected-attr-mode", default="age_ge_25")
    p.add_argument("--drop-unknown-rows", action="store_true")

    # NBI
    p.add_argument("--beta-step", type=float, default=0.01)
    p.add_argument("--nbi-eval-k", type=int, default=60)
    p.add_argument("--nbi-eval-k-stage1", type=int, default=None)
    p.add_argument("--nbi-eval-k-stage2", type=int, default=None)
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

    # NEW: anchor selection strategy for refine
    p.add_argument(
        "--refine-anchor-strategy",
        choices=["utopia", "mixed"],
        default="mixed",
        help="Anchor selection for refine stage2. 'utopia' reproduces legacy; 'mixed' adds fairness+BA extremes.",
    )
    p.add_argument(
        "--refine-anchor-mix",
        type=float,
        nargs=3,
        default=[0.4, 0.4, 0.2],
        metavar=("UTOPIA", "FAIR", "BA"),
        help="Fractions used when --refine-anchor-strategy=mixed. Will be normalized to sum=1.",
    )

    # Baselines
    p.add_argument("--run-baselines", action="store_true")

    # Fairness best subject to BA floor
    p.add_argument(
        "--fairness-best-floor",
        choices=["quality", "absolute", "best", "rs"],
        default="rs",
        help=(
            "How to set BA floor for the secondary 'best fairness' report: "
            "'quality' uses --quality-floor; "
            "'absolute' uses --fairness-best-ba-floor; "
            "'best' uses (BestUtopia_BA - --fairness-best-delta); "
            "'rs' uses (BestRandomSearch_BA - --fairness-best-delta) if baselines are enabled."
        ),
    )
    p.add_argument(
        "--fairness-best-ba-floor",
        type=float,
        default=0.68,
        help="Absolute BA floor when --fairness-best-floor=absolute (default: 0.68)",
    )
    p.add_argument(
        "--fairness-best-delta",
        type=float,
        default=0.005,
        help="Delta used when --fairness-best-floor is 'best' or 'rs' (default: 0.005)",
    )

    args = p.parse_args()

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)
    seed = int(args.seed if args.seed is not None else args.replica)

    out_dir = build_replica_dir(out_root, dataset_path, design_path, int(args.replica))

    auto_spw = True
    if bool(args.no_auto_scale_pos_weight):
        auto_spw = False
    if bool(args.auto_scale_pos_weight):
        auto_spw = True

    nbi_eval_k_stage1 = int(args.nbi_eval_k_stage1 or args.nbi_eval_k)
    nbi_eval_k_stage2 = int(args.nbi_eval_k_stage2 or args.nbi_eval_k)

    write_manifest(
        out_dir / "manifest.json",
        replica=int(args.replica),
        seed=seed,
        dataset_path=dataset_path,
        design_path=design_path,
        extra={
            "args": vars(args),
            "resolved": {
                "auto_scale_pos_weight": bool(auto_spw),
                "nbi_eval_k_stage1": int(nbi_eval_k_stage1),
                "nbi_eval_k_stage2": int(nbi_eval_k_stage2),
            },
        },
    )

    log_path = out_dir / "run_fairness_replica.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            stage_times: Dict[str, float] = {}

            # Data + design
            dataset_kind = str(args.dataset_kind)
            if dataset_kind == "bank":
                X, y, protected = load_bank_dataset(dataset_path)
            elif dataset_kind == "credit_card_default":
                X, y, protected = load_credit_card_default_dataset(
                    dataset_path,
                    protected_attr_mode=str(args.protected_attr_mode),
                    target_positive=str(args.target_positive) if args.target_positive is not None else "1",
                )
            elif dataset_kind == "generic":
                X, y, protected = load_generic_fairness_dataset(
                    dataset_path,
                    target_col=str(args.target_col),
                    target_positive=str(args.target_positive) if args.target_positive is not None else None,
                    protected_col=str(args.protected_col) if args.protected_col is not None else None,
                    protected_attr_mode=str(args.protected_attr_mode),
                    drop_unknown_rows=bool(args.drop_unknown_rows),
                )
            else:
                raise ValueError(f"Unsupported dataset kind: {dataset_kind}")

            design_df = load_design(design_path)

            # Stage 1: DOE -> RSM -> NBI -> eval NBI
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
            factors_df = doe_df[FAIRNESS_PARAM_NAMES].copy()

            model_q = fit_rsm_backward(
                factors_df,
                doe_df["BalancedAccuracy_Mean"],
                response_name="BalancedAccuracy_Mean",
                param_names=FAIRNESS_PARAM_NAMES,
            )
            model_f = fit_rsm_backward(
                factors_df,
                doe_df[FAIRNESS_OBJ_COL],
                response_name=FAIRNESS_OBJ_COL,
                param_names=FAIRNESS_PARAM_NAMES,
            )
            stage_times["rsm_seconds"] = float(time.perf_counter() - t0)

            coef_q_path = out_dir / "rsm_coefficients_balanced_accuracy.csv"
            coef_f_path = out_dir / "rsm_coefficients_fairness_di_only.csv"
            save_rsm_coefficients(model_q, str(coef_q_path))
            save_rsm_coefficients(model_f, str(coef_f_path))

            # NBI range
            utopia, nadir = _range_for_nbi(doe_df, quality_floor=float(args.nbi_range_quality_floor))

            t0 = time.perf_counter()
            nbi_candidates_1 = run_nbi_weighted_sum_fairness(
                load_coefficients_csv(str(coef_q_path)),
                load_coefficients_csv(str(coef_f_path)),
                observed_utopia=utopia,
                observed_nadir=nadir,
                beta_step=float(args.beta_step),
                seed=int(seed),
                n_starts=int(args.nbi_n_starts),
                clip_pred_range=bool(args.nbi_constrain_pred_range),
            )
            stage_times["nbi_stage1_seconds"] = float(time.perf_counter() - t0)

            df_nbi_1_raw = nbi_candidates_to_df(nbi_candidates_1)
            df_nbi_1 = _flatten_nbi_params(df_nbi_1_raw)
            df_nbi_1 = df_nbi_1.drop_duplicates(subset=FAIRNESS_PARAM_NAMES).reset_index(drop=True)
            save_csv_ptbr(df_nbi_1, out_dir / "nbi_candidates_stage1_fairness.csv")

            # Evaluate NBI stage1
            nbi_evalset_1 = _pick_evenly(df_nbi_1, int(nbi_eval_k_stage1), sort_col="beta_2")
            nbi_evalset_1_params = cast(pd.DataFrame, nbi_evalset_1[FAIRNESS_PARAM_NAMES].copy())

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

            # Stage 2: refine DOE -> refit RSM -> NBI -> eval
            refine_eval = pd.DataFrame()
            nbi_eval_2 = pd.DataFrame()

            if bool(args.refine):
                stage1_all = pd.concat(
                    [
                        doe_df.assign(Method="DOE+RSM"),
                        nbi_eval_1.assign(Method="DOE+RSM+NBI_stage1"),
                    ],
                    ignore_index=True,
                )
                best_stage1, stage1_diag = select_best_pareto_utopia(stage1_all, quality_floor=float(args.quality_floor))
                save_csv_ptbr(stage1_diag, out_dir / "all_evaluated_stage1_fairness.csv")
                save_csv_ptbr(pd.DataFrame([best_stage1.to_dict()]), out_dir / "best_solution_stage1_fairness.csv")

                anchors = select_refine_anchors(
                    stage1_diag,
                    top_m=int(args.refine_top_m),
                    strategy=str(args.refine_anchor_strategy),
                    mix=tuple(float(x) for x in args.refine_anchor_mix),
                )
                save_csv_ptbr(anchors, out_dir / "refine_anchors_stage2.csv")
                try:
                    counts = anchors["AnchorGroup"].value_counts().to_dict()
                except Exception:
                    counts = {}
                print(
                    f"Refine anchors: strategy={args.refine_anchor_strategy}, requested={args.refine_top_m}, "
                    f"selected={len(anchors)}, mix={args.refine_anchor_mix}, counts={counts}",
                    flush=True,
                )

                refine_design = _sample_refine_configs(
                    anchors[FAIRNESS_PARAM_NAMES].copy(),
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
                stage2_factors = stage2_doe[FAIRNESS_PARAM_NAMES].copy()

                model_q2 = fit_rsm_backward(
                    stage2_factors,
                    stage2_doe["BalancedAccuracy_Mean"],
                    response_name="BalancedAccuracy_Mean",
                    param_names=FAIRNESS_PARAM_NAMES,
                )
                model_f2 = fit_rsm_backward(
                    stage2_factors,
                    stage2_doe[FAIRNESS_OBJ_COL],
                    response_name=FAIRNESS_OBJ_COL,
                    param_names=FAIRNESS_PARAM_NAMES,
                )
                stage_times["rsm_stage2_seconds"] = float(time.perf_counter() - t0)

                coef_q2_path = out_dir / "rsm_coefficients_balanced_accuracy_stage2.csv"
                coef_f2_path = out_dir / "rsm_coefficients_fairness_di_only_stage2.csv"
                save_rsm_coefficients(model_q2, str(coef_q2_path))
                save_rsm_coefficients(model_f2, str(coef_f2_path))

                utopia2, nadir2 = _range_for_nbi(stage2_doe, quality_floor=float(args.nbi_range_quality_floor))

                t0 = time.perf_counter()
                nbi_candidates_2 = run_nbi_weighted_sum_fairness(
                    load_coefficients_csv(str(coef_q2_path)),
                    load_coefficients_csv(str(coef_f2_path)),
                    observed_utopia=utopia2,
                    observed_nadir=nadir2,
                    beta_step=float(args.beta_step),
                    seed=int(seed + 40_000),
                    n_starts=int(args.nbi_n_starts),
                    clip_pred_range=bool(args.nbi_constrain_pred_range),
                )
                stage_times["nbi_stage2_seconds"] = float(time.perf_counter() - t0)

                df_nbi_2_raw = nbi_candidates_to_df(nbi_candidates_2)
                df_nbi_2 = _flatten_nbi_params(df_nbi_2_raw)
                df_nbi_2 = df_nbi_2.drop_duplicates(subset=FAIRNESS_PARAM_NAMES).reset_index(drop=True)
                save_csv_ptbr(df_nbi_2, out_dir / "nbi_candidates_stage2_fairness.csv")

                nbi_evalset_2 = _pick_evenly(df_nbi_2, int(nbi_eval_k_stage2), sort_col="beta_2")
                nbi_evalset_2_params = cast(pd.DataFrame, nbi_evalset_2[FAIRNESS_PARAM_NAMES].copy())

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

            # Final selection (across all evaluated points)
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

            # Baselines + Comparison summary
            comparison_rows: List[Dict[str, Any]] = []

            def _row_from_best(tag: str, best_series: pd.Series) -> Dict[str, Any]:
                dct = best_series.to_dict()
                return {
                    "Method": tag,
                    "BalancedAccuracy_Mean": float(dct.get("BalancedAccuracy_Mean", 0.0)),
                    "Bias_DI_Mean": float(dct.get(FAIRNESS_BIAS_COL, 0.0)),
                    "FairnessScore_DI_Only": float(dct.get(FAIRNESS_OBJ_COL, 0.0)),
                    "BiasMean_Mean": float(dct.get(LEGACY_FAIRNESS_BIAS_COL, 0.0)),
                    "FairnessScore_1_minus_Bias": float(dct.get(LEGACY_FAIRNESS_OBJ_COL, 0.0)),
                    "Time_MeanFold": float(dct.get("Time_MeanFold", np.nan)),
                    "Time_TotalCV": float(dct.get("Time_TotalCV", np.nan)),
                    "scale_pos_weight": float(dct.get("scale_pos_weight", np.nan)),
                    "threshold": float(dct.get("threshold", np.nan)),
                    "BA_Floor": float(dct.get("FairnessBest_BA_Floor", np.nan)),
                }

            comparison_rows.append(_row_from_best("Proposed_Utopia", best_overall))

            rs_best_ba: Optional[float] = None
            rs_diag: Optional[pd.DataFrame] = None

            if bool(args.run_baselines):
                budget = int(len(all_eval))

                # XGB default baseline
                xgb_default_df = pd.DataFrame([{"scale_pos_weight": 1.0, "threshold": 0.5}])
                xgb_default_eval = run_doe_fairness(
                    design_df=xgb_default_df,
                    X=X,
                    y=y,
                    protected=protected,
                    seed=seed + 60_000,
                    n_splits=int(args.n_splits),
                    n_jobs=int(args.n_jobs),
                    tree_method=str(args.tree_method),
                    auto_scale_pos_weight=False,
                    stratify_by_group=bool(args.stratify_by_group),
                    privileged_value=1,
                    desc="Baseline: XGB default",
                )
                best_xgb = cast(pd.Series, xgb_default_eval.iloc[0])
                save_csv_ptbr(xgb_default_eval, out_dir / "baseline_xgb_default_fairness.csv")
                comparison_rows.append(_row_from_best("XGB_Default", best_xgb))

                # Random Search baseline
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
                    auto_scale_pos_weight=False,
                    stratify_by_group=bool(args.stratify_by_group),
                    privileged_value=1,
                    desc=f"Baseline: Random Search (N={budget})",
                )
                best_rs, rs_diag = select_best_pareto_utopia(rs_eval, quality_floor=float(args.quality_floor))
                save_csv_ptbr(rs_diag, out_dir / "baseline_random_search_fairness.csv")
                save_csv_ptbr(pd.DataFrame([best_rs.to_dict()]), out_dir / "best_solution_random_search_fairness.csv")
                comparison_rows.append(_row_from_best("RandomSearch_Utopia", best_rs))

                rs_best_ba = float(best_rs.get("BalancedAccuracy_Mean", np.nan))
                if np.isnan(rs_best_ba):
                    rs_best_ba = None

            # Secondary report: best fairness subject to BA floor
            floor_mode = str(args.fairness_best_floor)
            delta = float(args.fairness_best_delta)

            if floor_mode == "quality":
                ba_floor = float(args.quality_floor)
            elif floor_mode == "absolute":
                ba_floor = float(args.fairness_best_ba_floor)
            elif floor_mode == "best":
                ba_floor = float(best_overall.get("BalancedAccuracy_Mean", 0.0)) - float(delta)
            elif floor_mode == "rs":
                if rs_best_ba is not None:
                    ba_floor = float(rs_best_ba) - float(delta)
                else:
                    ba_floor = float(best_overall.get("BalancedAccuracy_Mean", 0.0)) - float(delta)
            else:
                ba_floor = float(args.quality_floor)

            ba_floor = max(float(args.quality_floor), float(ba_floor))

            best_fair = select_best_fairness_subject_to_ba(all_diag, ba_floor=ba_floor, pareto_only=True)
            best_fair = best_fair.copy()
            best_fair["FairnessBest_BA_Floor"] = float(ba_floor)
            save_csv_ptbr(pd.DataFrame([best_fair.to_dict()]), out_dir / "best_solution_fairness_constrained.csv")
            comparison_rows.append(_row_from_best("Proposed_Constrained", best_fair))

            if rs_diag is not None:
                best_rs_fair = select_best_fairness_subject_to_ba(rs_diag, ba_floor=ba_floor, pareto_only=True)
                best_rs_fair = best_rs_fair.copy()
                best_rs_fair["FairnessBest_BA_Floor"] = float(ba_floor)
                save_csv_ptbr(pd.DataFrame([best_rs_fair.to_dict()]), out_dir / "best_solution_random_search_constrained.csv")
                comparison_rows.append(_row_from_best("RandomSearch_Constrained", best_rs_fair))

            comp = pd.DataFrame(comparison_rows)
            save_csv_ptbr(comp, out_dir / "comparison_summary_fairness.csv")

            print("\n=== Comparison summary (best points) ===")
            cols_show = [
                "Method",
                "BalancedAccuracy_Mean",
                "Bias_DI_Mean",
                "FairnessScore_DI_Only",
                "BiasMean_Mean",
                "FairnessScore_1_minus_Bias",
                "Time_MeanFold",
                "scale_pos_weight",
                "threshold",
                "BA_Floor",
            ]
            print(comp[cols_show].to_string(index=False))

            save_csv_ptbr(pd.DataFrame([stage_times]), out_dir / "stage_times_fairness.csv")

            print("\n✅ Fairness replica pipeline finished")
            print(f"- out_dir: {out_dir}")
            print(f"- log: {log_path}")


if __name__ == "__main__":
    main()
