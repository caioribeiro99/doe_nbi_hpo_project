#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from doe_xgb.io_utils import load_dataset, save_csv_ptbr
from doe_xgb.benchmarks import (
    load_nbi_candidates,
    evaluate_candidate_list,
    coarse_grid_search,
    random_search,
    bayes_search,
    hyperopt_tpe,
    save_benchmark_summary,
)


def main():
    p = argparse.ArgumentParser(description="Run confirmation (best NBI candidate) + benchmark optimizers")
    p.add_argument("--dataset", required=True, help="Path to dataset (xlsx/csv/parquet)")
    p.add_argument("--replica_dir", required=True, help="Replica output directory (contains nbi_candidates.csv)")
    p.add_argument("--seed", type=int, required=True, help="Replica seed")
    p.add_argument("--budget", type=int, default=40, help="Evaluation budget for benchmark methods")
    p.add_argument("--target", default="y", help="Target column name")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    rep_dir = Path(args.replica_dir)

    cand_path = rep_dir / "nbi_candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(cand_path)

    X, y = load_dataset(dataset_path, target_col=args.target, target_map={"g": 0, "h": 1})

    # 1) DOE+NBI candidates
    cands = load_nbi_candidates(str(cand_path))
    best_nbi, all_nbi = evaluate_candidate_list(cands, X, y, seed=args.seed, n_splits=5, n_jobs=-1, tree_method="hist")

    df_all = pd.DataFrame(all_nbi)
    save_csv_ptbr(df_all, rep_dir / "nbi_candidate_evaluations.csv")

    summary_rows = []
    best_nbi_row = dict(best_nbi)
    best_nbi_row["method"] = "doe_nbi"
    best_nbi_row["budget"] = len(cands)
    summary_rows.append(best_nbi_row)

    # 2) Benchmarks
    summary_rows.append(coarse_grid_search(X, y, seed=args.seed, budget=args.budget))
    summary_rows.append(random_search(X, y, seed=args.seed, budget=args.budget))

    # Optional methods (may require extra deps)
    try:
        summary_rows.append(bayes_search(X, y, seed=args.seed, budget=args.budget))
    except Exception as e:
        summary_rows.append({"method": "bayes_search", "error": str(e)})

    try:
        summary_rows.append(hyperopt_tpe(X, y, seed=args.seed, budget=args.budget))
    except Exception as e:
        summary_rows.append({"method": "hyperopt_tpe", "error": str(e)})

    out_path = rep_dir / "confirmation_summary.csv"
    save_benchmark_summary(summary_rows, str(out_path))
    print(f"✅ Confirmation summary saved: {out_path}")


if __name__ == "__main__":
    main()
