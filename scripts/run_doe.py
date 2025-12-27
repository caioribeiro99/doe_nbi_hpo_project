#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# allow running without installation
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from doe_xgb.io_utils import load_dataset, load_design
from doe_xgb.doe_runner import run_doe, save_doe_results
from doe_xgb.tracking import build_replica_dir, write_manifest
from doe_xgb.config import DEFAULT_BOUNDS


def main():
    p = argparse.ArgumentParser(description="Run DOE evaluation (XGBoost + CV)")
    p.add_argument("--dataset", required=True, help="Path to dataset (xlsx/csv/parquet)")
    p.add_argument("--design", required=True, help="Path to DOE design CSV (Minitab export)")
    p.add_argument("--replica", type=int, required=True, help="Replica number (1..N)")
    p.add_argument("--seed", type=int, required=True, help="Replica seed")
    p.add_argument("--out_root", default="experiments", help="Experiments output root folder")
    p.add_argument("--target", default="y", help="Target column name")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)

    out_dir = build_replica_dir(out_root, dataset_path, design_path, args.replica)
    write_manifest(out_dir, replica=args.replica, seed=args.seed, dataset_path=dataset_path, design_path=design_path)

    X, y = load_dataset(dataset_path, target_col=args.target, target_map={"g": 0, "h": 1})
    design_df = load_design(design_path)

    df = run_doe(design_df, X, y, seed=args.seed, n_splits=5, n_jobs=-1, tree_method="hist")

    out_file = out_dir / "doe_results.csv"
    save_doe_results(df, out_file)
    print(f"✅ DOE results saved: {out_file}")


if __name__ == "__main__":
    main()
