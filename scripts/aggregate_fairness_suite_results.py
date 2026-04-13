#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
from doe_xgb.io_utils import _read_csv_flexible  # noqa: E402

QUALITY_COL = "BalancedAccuracy_Mean"
FAIRNESS_OBJ_COL = "FairnessScore_DI_Only"
FAIRNESS_BIAS_COL = "Bias_DI_Mean"
LEGACY_FAIRNESS_OBJ_COL = "FairnessScore_1_minus_Bias"
LEGACY_FAIRNESS_BIAS_COL = "BiasMean_Mean"


def _load_cfg(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_replica_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.rglob("replica_*") if p.is_dir()])


def _read_csv(path: Path) -> pd.DataFrame:
    return _read_csv_flexible(path)


def _read_manifest_seed(replica_dir: Path) -> Optional[int]:
    p = replica_dir / "manifest.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("seed"))
    except Exception:
        return None


def _summary_row(comp: pd.DataFrame, method: str) -> Optional[pd.Series]:
    if "Method" not in comp.columns:
        return None
    sub = comp.loc[comp["Method"].astype(str) == method]
    if sub.empty:
        return None
    return sub.iloc[0]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _stats_diff(a: Iterable[float], b: Iterable[float], *, greater_better: bool = True) -> Dict[str, Any]:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    diffs = a_arr[mask] - b_arr[mask]
    n = int(diffs.size)
    if n == 0:
        return {
            "n": 0,
            "mean_diff": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "median_diff": np.nan,
            "win_rate_a_gt_b": np.nan,
            "win_rate_a_lt_b": np.nan,
            "sd_diff": np.nan,
        }
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean_diff": mean,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
        "median_diff": float(np.median(diffs)),
        "win_rate_a_gt_b": float(np.mean(diffs > 0)),
        "win_rate_a_lt_b": float(np.mean(diffs < 0)),
        "sd_diff": sd,
    }


def _select_best_fairness_subject_to_ba(diag_df: pd.DataFrame, ba_floor: float, *, pareto_only: bool = True) -> pd.Series:
    d = diag_df.copy()
    d[QUALITY_COL] = d[QUALITY_COL].astype(float)
    d[FAIRNESS_BIAS_COL] = d[FAIRNESS_BIAS_COL].astype(float)
    base = d[d["Is_Pareto"]].copy() if pareto_only and "Is_Pareto" in d.columns else d.copy()
    cand = base[base[QUALITY_COL] >= float(ba_floor)].copy()
    if cand.empty:
        cand = d[d[QUALITY_COL] >= float(ba_floor)].copy()
    if cand.empty:
        cand = base.copy() if not base.empty else d.copy()
    return cand.loc[cand[FAIRNESS_BIAS_COL].astype(float).idxmin()]


def _compute_tradeoff_curves(replica_rows: pd.DataFrame, replica_dirs: Dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replica_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    start = math.ceil(float(replica_rows[["ours_utopia_BA", "rs_utopia_BA", "xgb_default_BA"]].min().min()) * 100) / 100.0
    max_ba = -np.inf
    all_data: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for _, row in replica_rows.iterrows():
        rep = str(row["replica"])
        rep_dir = replica_dirs[rep]
        ours = _read_csv(rep_dir / "all_evaluated_candidates_fairness.csv")
        rs = _read_csv(rep_dir / "baseline_random_search_fairness.csv")
        all_data[rep] = (ours, rs)
        max_ba = max(max_ba, float(ours["BalancedAccuracy_Mean"].max()), float(rs["BalancedAccuracy_Mean"].max()))

    end = math.ceil(max_ba * 100) / 100.0
    thresholds = np.arange(start, end + 0.0001, 0.005)

    avg_rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []

    for thr in thresholds:
        ours_vals = []
        rs_vals = []
        diffs = []
        for rep, (ours, rs) in all_data.items():
            ours_c = ours.loc[ours["BalancedAccuracy_Mean"].astype(float) >= float(thr)]
            rs_c = rs.loc[rs["BalancedAccuracy_Mean"].astype(float) >= float(thr)]
            if not ours_c.empty:
                ours_bias = float(ours_c[FAIRNESS_BIAS_COL].astype(float).min())
                ours_vals.append(ours_bias)
            if not rs_c.empty:
                rs_bias = float(rs_c[FAIRNESS_BIAS_COL].astype(float).min())
                rs_vals.append(rs_bias)
            if not ours_c.empty and not rs_c.empty:
                diffs.append(ours_bias - rs_bias)
        if not ours_vals or not rs_vals:
            continue
        avg_rows.append(
            {
                "ba_threshold": round(float(thr), 3),
                "ours_mean_min_bias": float(np.mean(ours_vals)),
                "rs_mean_min_bias": float(np.mean(rs_vals)),
                "ours_median_min_bias": float(np.median(ours_vals)),
                "rs_median_min_bias": float(np.median(rs_vals)),
                "n_ours": int(len(ours_vals)),
                "n_rs": int(len(rs_vals)),
                "diff_mean": float(np.mean(diffs)) if diffs else np.nan,
            }
        )
        if diffs:
            d = np.asarray(diffs, dtype=float)
            n = int(d.size)
            sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
            half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
            diff_rows.append(
                {
                    "ba_threshold": round(float(thr), 3),
                    "n": n,
                    "mean": float(np.mean(d)),
                    "sd": sd,
                    "ci95_low": float(np.mean(d) - half),
                    "ci95_high": float(np.mean(d) + half),
                    "win_rate": float(np.mean(d < 0)),
                    "median": float(np.median(d)),
                }
            )

    return pd.DataFrame(avg_rows), pd.DataFrame(diff_rows)


def _compute_delta_sensitivity(replica_rows: pd.DataFrame, replica_dirs: Dict[str, Path], quality_floor: float) -> pd.DataFrame:
    if replica_rows.empty:
        return pd.DataFrame()
    deltas = [0.01, 0.02, 0.03, 0.04]
    out_rows: List[Dict[str, Any]] = []
    for delta in deltas:
        bias_diffs = []
        ba_diffs = []
        for _, row in replica_rows.iterrows():
            rep = str(row["replica"])
            rep_dir = replica_dirs[rep]
            ours = _read_csv(rep_dir / "all_evaluated_candidates_fairness.csv")
            rs = _read_csv(rep_dir / "baseline_random_search_fairness.csv")
            rs_utopia_ba = float(row["rs_utopia_BA"])
            ba_floor = max(float(quality_floor), rs_utopia_ba - float(delta))
            ours_sel = _select_best_fairness_subject_to_ba(ours, ba_floor=ba_floor, pareto_only=True)
            rs_sel = _select_best_fairness_subject_to_ba(rs, ba_floor=ba_floor, pareto_only=True)
            bias_diffs.append(float(ours_sel[FAIRNESS_BIAS_COL]) - float(rs_sel[FAIRNESS_BIAS_COL]))
            ba_diffs.append(float(ours_sel["BalancedAccuracy_Mean"]) - float(rs_sel["BalancedAccuracy_Mean"]))
        d = np.asarray(bias_diffs, dtype=float)
        n = int(d.size)
        sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
        half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
        out_rows.append(
            {
                "delta": float(delta),
                "mean_bias_diff": float(np.mean(d)),
                "median_bias_diff": float(np.median(d)),
                "win_rate_bias": float(np.mean(d < 0)),
                "sd": sd,
                "ci95_low": float(np.mean(d) - half),
                "ci95_high": float(np.mean(d) + half),
                "mean_ba_diff": float(np.mean(ba_diffs)),
                "median_ba_diff": float(np.median(ba_diffs)),
            }
        )
    return pd.DataFrame(out_rows)


def aggregate_dataset(dataset_root: Path, *, quality_floor: float) -> Dict[str, str]:
    replica_dirs = _find_replica_dirs(dataset_root)
    if not replica_dirs:
        raise FileNotFoundError(f"No replica_* directories found under {dataset_root}")

    per_rep_rows: List[Dict[str, Any]] = []
    replica_dir_map: Dict[str, Path] = {}

    for rep_dir in replica_dirs:
        rep = rep_dir.name
        replica_dir_map[rep] = rep_dir
        comp = _read_csv(rep_dir / "comparison_summary_fairness.csv")
        rows = {m: _summary_row(comp, m) for m in [
            "Proposed_Utopia",
            "Proposed_Constrained",
            "RandomSearch_Utopia",
            "RandomSearch_Constrained",
            "XGB_Default",
        ]}
        seed = _read_manifest_seed(rep_dir)
        per_rep_rows.append(
            {
                "replica": rep,
                "seed": seed,
                "ba_floor": _safe_float(rows["Proposed_Constrained"].get("BA_Floor")) if rows["Proposed_Constrained"] is not None else np.nan,
                "ours_utopia_BA": _safe_float(rows["Proposed_Utopia"].get("BalancedAccuracy_Mean")) if rows["Proposed_Utopia"] is not None else np.nan,
                "ours_utopia_Bias": _safe_float(rows["Proposed_Utopia"].get(FAIRNESS_BIAS_COL)) if rows["Proposed_Utopia"] is not None else np.nan,
                "ours_utopia_BiasMeanComposite": _safe_float(rows["Proposed_Utopia"].get(LEGACY_FAIRNESS_BIAS_COL)) if rows["Proposed_Utopia"] is not None else np.nan,
                "ours_utopia_Fair": _safe_float(rows["Proposed_Utopia"].get(FAIRNESS_OBJ_COL)) if rows["Proposed_Utopia"] is not None else np.nan,
                "ours_utopia_FairComposite": _safe_float(rows["Proposed_Utopia"].get(LEGACY_FAIRNESS_OBJ_COL)) if rows["Proposed_Utopia"] is not None else np.nan,
                "ours_utopia_TimeMeanFold": _safe_float(rows["Proposed_Utopia"].get("Time_MeanFold")) if rows["Proposed_Utopia"] is not None else np.nan,
                "ours_constrained_BA": _safe_float(rows["Proposed_Constrained"].get("BalancedAccuracy_Mean")) if rows["Proposed_Constrained"] is not None else np.nan,
                "ours_constrained_Bias": _safe_float(rows["Proposed_Constrained"].get(FAIRNESS_BIAS_COL)) if rows["Proposed_Constrained"] is not None else np.nan,
                "ours_constrained_BiasMeanComposite": _safe_float(rows["Proposed_Constrained"].get(LEGACY_FAIRNESS_BIAS_COL)) if rows["Proposed_Constrained"] is not None else np.nan,
                "ours_constrained_Fair": _safe_float(rows["Proposed_Constrained"].get(FAIRNESS_OBJ_COL)) if rows["Proposed_Constrained"] is not None else np.nan,
                "ours_constrained_FairComposite": _safe_float(rows["Proposed_Constrained"].get(LEGACY_FAIRNESS_OBJ_COL)) if rows["Proposed_Constrained"] is not None else np.nan,
                "ours_constrained_TimeMeanFold": _safe_float(rows["Proposed_Constrained"].get("Time_MeanFold")) if rows["Proposed_Constrained"] is not None else np.nan,
                "rs_utopia_BA": _safe_float(rows["RandomSearch_Utopia"].get("BalancedAccuracy_Mean")) if rows["RandomSearch_Utopia"] is not None else np.nan,
                "rs_utopia_Bias": _safe_float(rows["RandomSearch_Utopia"].get(FAIRNESS_BIAS_COL)) if rows["RandomSearch_Utopia"] is not None else np.nan,
                "rs_utopia_BiasMeanComposite": _safe_float(rows["RandomSearch_Utopia"].get(LEGACY_FAIRNESS_BIAS_COL)) if rows["RandomSearch_Utopia"] is not None else np.nan,
                "rs_utopia_Fair": _safe_float(rows["RandomSearch_Utopia"].get(FAIRNESS_OBJ_COL)) if rows["RandomSearch_Utopia"] is not None else np.nan,
                "rs_utopia_FairComposite": _safe_float(rows["RandomSearch_Utopia"].get(LEGACY_FAIRNESS_OBJ_COL)) if rows["RandomSearch_Utopia"] is not None else np.nan,
                "rs_utopia_TimeMeanFold": _safe_float(rows["RandomSearch_Utopia"].get("Time_MeanFold")) if rows["RandomSearch_Utopia"] is not None else np.nan,
                "xgb_default_BA": _safe_float(rows["XGB_Default"].get("BalancedAccuracy_Mean")) if rows["XGB_Default"] is not None else np.nan,
                "xgb_default_Bias": _safe_float(rows["XGB_Default"].get(FAIRNESS_BIAS_COL)) if rows["XGB_Default"] is not None else np.nan,
                "xgb_default_BiasMeanComposite": _safe_float(rows["XGB_Default"].get(LEGACY_FAIRNESS_BIAS_COL)) if rows["XGB_Default"] is not None else np.nan,
                "xgb_default_Fair": _safe_float(rows["XGB_Default"].get(FAIRNESS_OBJ_COL)) if rows["XGB_Default"] is not None else np.nan,
                "xgb_default_FairComposite": _safe_float(rows["XGB_Default"].get(LEGACY_FAIRNESS_OBJ_COL)) if rows["XGB_Default"] is not None else np.nan,
                "xgb_default_TimeMeanFold": _safe_float(rows["XGB_Default"].get("Time_MeanFold")) if rows["XGB_Default"] is not None else np.nan,
                "rs_best_at_floor_BA": _safe_float(rows["RandomSearch_Constrained"].get("BalancedAccuracy_Mean")) if rows["RandomSearch_Constrained"] is not None else np.nan,
                "rs_best_at_floor_Bias": _safe_float(rows["RandomSearch_Constrained"].get(FAIRNESS_BIAS_COL)) if rows["RandomSearch_Constrained"] is not None else np.nan,
                "rs_best_at_floor_BiasMeanComposite": _safe_float(rows["RandomSearch_Constrained"].get(LEGACY_FAIRNESS_BIAS_COL)) if rows["RandomSearch_Constrained"] is not None else np.nan,
                "rs_best_at_floor_Fair": _safe_float(rows["RandomSearch_Constrained"].get(FAIRNESS_OBJ_COL)) if rows["RandomSearch_Constrained"] is not None else np.nan,
                "rs_best_at_floor_FairComposite": _safe_float(rows["RandomSearch_Constrained"].get(LEGACY_FAIRNESS_OBJ_COL)) if rows["RandomSearch_Constrained"] is not None else np.nan,
                "rs_best_at_floor_TimeMeanFold": _safe_float(rows["RandomSearch_Constrained"].get("Time_MeanFold")) if rows["RandomSearch_Constrained"] is not None else np.nan,
            }
        )

    per_rep_df = pd.DataFrame(per_rep_rows).sort_values("replica")
    per_rep_path = dataset_root / "R30_per_replica_summary.csv"
    per_rep_df.to_csv(per_rep_path, index=False, sep=";", decimal=",")

    comps: List[Dict[str, Any]] = []
    comparison_specs = [
        ("Utopia: BA ours - RS", "ours_utopia_BA", "rs_utopia_BA"),
        ("Utopia: DI-bias ours - RS", "ours_utopia_Bias", "rs_utopia_Bias"),
        ("Constrained: BA ours - RS_utopia", "ours_constrained_BA", "rs_utopia_BA"),
        ("Constrained: DI-bias ours - RS_utopia", "ours_constrained_Bias", "rs_utopia_Bias"),
        ("Constrained: BA ours - RS_best_at_floor", "ours_constrained_BA", "rs_best_at_floor_BA"),
        ("Constrained: DI-bias ours - RS_best_at_floor", "ours_constrained_Bias", "rs_best_at_floor_Bias"),
        ("Utopia: BA ours - XGB_default", "ours_utopia_BA", "xgb_default_BA"),
        ("Utopia: DI-bias ours - XGB_default", "ours_utopia_Bias", "xgb_default_Bias"),
        ("Constrained: BA ours - XGB_default", "ours_constrained_BA", "xgb_default_BA"),
        ("Constrained: DI-bias ours - XGB_default", "ours_constrained_Bias", "xgb_default_Bias"),
    ]
    for label, a_col, b_col in comparison_specs:
        stats = _stats_diff(per_rep_df[a_col], per_rep_df[b_col], greater_better=("Bias" not in label))
        comps.append({"comparison": label, **stats})
    agg_comp_df = pd.DataFrame(comps)
    agg_comp_path = dataset_root / "R30_aggregated_comparisons.csv"
    agg_comp_df.to_csv(agg_comp_path, index=False, sep=";", decimal=",")

    trade_avg_df, trade_diff_df = _compute_tradeoff_curves(per_rep_df, replica_dir_map)
    trade_avg_path = dataset_root / "R30_bias_tradeoff_curve_avg.csv"
    trade_diff_path = dataset_root / "R30_bias_tradeoff_curve_diff_stats.csv"
    trade_avg_df.to_csv(trade_avg_path, index=False, sep=";", decimal=",")
    trade_diff_df.to_csv(trade_diff_path, index=False, sep=";", decimal=",")

    delta_df = _compute_delta_sensitivity(per_rep_df, replica_dir_map, quality_floor=quality_floor)
    delta_path = dataset_root / "R30_delta_sensitivity.csv"
    delta_df.to_csv(delta_path, index=False, sep=";", decimal=",")

    return {
        "dataset_root": str(dataset_root),
        "per_replica_summary": str(per_rep_path),
        "aggregated_comparisons": str(agg_comp_path),
        "bias_tradeoff_avg": str(trade_avg_path),
        "bias_tradeoff_diff_stats": str(trade_diff_path),
        "delta_sensitivity": str(delta_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate fairness suite outputs into R30_* CSV artifacts.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "fairness_suite_3bases_finance_r30.json"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "experiments" / "fairness_suite_3bases_finance_r30"))
    p.add_argument("--quality-floor", type=float, default=0.55)
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config).resolve())
    out_root = Path(args.out_root).resolve()
    summary_rows: List[Dict[str, str]] = []
    for dcfg in cfg.get("datasets", []):
        dataset_root = out_root / str(dcfg["name"])
        summary_rows.append(aggregate_dataset(dataset_root, quality_floor=float(args.quality_floor)))
    pd.DataFrame(summary_rows).to_csv(out_root / "suite_artifact_index.csv", index=False)
    print(f"saved suite artifact index -> {out_root / 'suite_artifact_index.csv'}")
    print("✅ Aggregation finished.")


if __name__ == "__main__":
    main()
