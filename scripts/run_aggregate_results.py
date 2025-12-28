#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, *, acc_col: str, time_col: str) -> pd.Series:
    """
    Compute Pareto membership for a set of points where we maximize acc_col and minimize time_col.
    Returns a boolean Series aligned to df.
    """
    acc = df[acc_col].to_numpy(dtype=float)
    t = df[time_col].to_numpy(dtype=float)

    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        # j dominates i if: acc_j >= acc_i and t_j <= t_i, with at least one strict
        dominates = (acc >= acc[i]) & (t <= t[i]) & ((acc > acc[i]) | (t < t[i]))
        dominates[i] = False
        if dominates.any():
            is_pareto[i] = False
    return pd.Series(is_pareto, index=df.index)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate confirmation_summary across replicas (union + stats).")
    p.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment directory (e.g., experiments/telescope2/hyperparameter_design)",
    )
    p.add_argument(
        "--union",
        type=str,
        default="confirmation_summary_all_replicas.csv",
        help="Union CSV filename inside exp-dir (default: confirmation_summary_all_replicas.csv)",
    )
    p.add_argument(
        "--out-stats",
        type=str,
        default="confirmation_summary_stats.csv",
        help="Output stats CSV filename inside exp-dir",
    )
    p.add_argument(
        "--out-wins",
        type=str,
        default="confirmation_summary_winrate.csv",
        help="Output win-rate CSV filename inside exp-dir",
    )
    p.add_argument(
        "--out-pareto",
        type=str,
        default="confirmation_summary_pareto.csv",
        help="Output Pareto frequency CSV filename inside exp-dir",
    )
    args = p.parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    union_path = exp_dir / args.union
    if not union_path.exists():
        raise FileNotFoundError(f"Union file not found: {union_path}")

    df = pd.read_csv(union_path, sep=";", decimal=",")
    required_cols = {"replica", "seed", "method", "Accuracy_Mean", "Time_MeanFold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Union CSV missing required columns: {sorted(missing)}")

    # --- basic stats per method ---
    num_cols: List[str] = []
    for c in [
        "Accuracy_Mean",
        "Precision_Mean",
        "Recall_Mean",
        "Specificity_Mean",
        "Time_MeanFold",
        "Optimization_Time_Seconds",
        "Total_Time_Seconds",
        "budget",
    ]:
        if c in df.columns:
            num_cols.append(c)

    # force numeric where possible
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    stats = (
        df.groupby("method", dropna=False)[num_cols]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    stats.columns = ["method"] + [f"{a}_{b}" for a, b in stats.columns[1:]]
    stats_path = exp_dir / args.out_stats
    stats.to_csv(stats_path, sep=";", decimal=",", index=False, encoding="utf-8")

    # --- win-rate on Accuracy (per replica) ---
    # for each replica, identify the method(s) with best Accuracy_Mean (ties count as win)
    best_acc = df.groupby("replica")["Accuracy_Mean"].transform("max")
    df["_is_best_acc"] = df["Accuracy_Mean"] >= best_acc - 1e-12

    win = (
        df.groupby("method")["_is_best_acc"]
        .agg(wins="sum", n="count")
        .reset_index()
    )
    win["win_rate"] = win["wins"] / df["replica"].nunique()
    win_path = exp_dir / args.out_wins
    win.to_csv(win_path, sep=";", decimal=",", index=False, encoding="utf-8")

    # --- Pareto frequency per replica (Accuracy max, Time_MeanFold min) ---
    pareto_flags = []
    for rep, g in df.groupby("replica", sort=True):
        pf = pareto_front(g, acc_col="Accuracy_Mean", time_col="Time_MeanFold")
        tmp = g.loc[pf, ["replica", "seed", "method", "Accuracy_Mean", "Time_MeanFold"]].copy()
        tmp["is_pareto"] = True
        pareto_flags.append(tmp)

    pareto_df = pd.concat(pareto_flags, ignore_index=True) if pareto_flags else pd.DataFrame()
    pareto_path = exp_dir / args.out_pareto
    pareto_df.to_csv(pareto_path, sep=";", decimal=",", index=False, encoding="utf-8")

    # Also produce Pareto frequency per method
    pareto_freq = (
        pareto_df.groupby("method")["is_pareto"]
        .agg(pareto_count="sum")
        .reset_index()
    )
    pareto_freq["pareto_rate"] = pareto_freq["pareto_count"] / df["replica"].nunique()
    pareto_freq_path = exp_dir / "confirmation_summary_pareto_rate.csv"
    pareto_freq.to_csv(pareto_freq_path, sep=";", decimal=",", index=False, encoding="utf-8")

    print(f"✅ Wrote:\n- {stats_path}\n- {win_path}\n- {pareto_path}\n- {pareto_freq_path}")


if __name__ == "__main__":
    main()
