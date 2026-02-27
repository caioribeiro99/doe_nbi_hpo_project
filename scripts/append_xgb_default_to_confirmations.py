#!/usr/bin/env python
from __future__ import annotations

"""Append an XGBoost *default* baseline row to existing confirmation_summary.csv files.

Why this script exists
----------------------
Your thesis already contains DOE+NBI and benchmark results stored under:

  experiments/<dataset>/<design>/replica_XX/confirmation_summary.csv

The committee requested an additional comparison against **XGBoost default (no tuning)**.
Re-running the whole pipeline is unnecessary. This script:

1) reads each replica seed from manifest.json
2) evaluates a single XGBClassifier with *default hyperparameters* under the same CV protocol
3) appends a new row (method = 'xgb_default') to confirmation_summary.csv
4) optionally builds a union CSV across replicas (confirmation_summary_all_replicas.csv)

It is intentionally minimal and does not touch DOE/NBI outputs.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Allow running scripts directly without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from doe_xgb.config import INT_PARAMS, PARAM_NAMES  # noqa: E402
from doe_xgb.evaluation import evaluate_xgb_cv  # noqa: E402
from doe_xgb.io_utils import load_dataset, save_csv_ptbr  # noqa: E402


def _resolve_dataset_from_manifest(replica_dir: Path) -> Path:
    man_path = replica_dir / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {replica_dir}")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    ds = man.get("dataset", {})
    ds_path = ds.get("path")
    if not ds_path:
        raise KeyError(f"manifest.json missing dataset.path in {replica_dir}")

    p = Path(ds_path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _read_seed(replica_dir: Path) -> int:
    man_path = replica_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    seed = man.get("seed")
    if seed is None:
        raise KeyError(f"manifest.json missing seed in {replica_dir}")
    return int(seed)


# NOTE:
# In XGBoost 3.x, the scikit-learn wrapper may expose many defaults as `None`
# (meaning: "use library defaults"). To make the baseline explicit, we
# hard-code the canonical XGBClassifier defaults for the hyperparameters
# used in this dissertation.
XGB_EXPLICIT_DEFAULTS: Dict[str, float | int] = {
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0,
    "learning_rate": 0.3,
    "max_depth": 6,
    "gamma": 0.0,
    "n_estimators": 100,
}


def _evaluate_default(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_splits: int,
    n_jobs: int,
    tree_method: str,
    eval_metric: str,
) -> Dict[str, Any]:
    defaults = dict(XGB_EXPLICIT_DEFAULTS)

    X_np = X.to_numpy()
    y_np = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Measure the evaluation wall time (optional, useful for sanity checks)
    t0 = time.perf_counter()
    ev = evaluate_xgb_cv(
        defaults,
        X_np,
        y_np,
        kfold,
        seed=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric=eval_metric,
    )
    _ = time.perf_counter() - t0

    row: Dict[str, Any] = dict(ev.as_dict())
    row["hyperparameters"] = dict(defaults)
    row["method"] = "xgb_default"

    # No tuning/search happened.
    row["budget"] = 0.0
    row["Optimization_Time_Seconds"] = 0.0
    row["Total_Time_Seconds"] = float(row.get("Time_MeanFold", 0.0))
    return row


def _append_row_if_missing(csv_path: Path, row: Dict[str, Any]) -> bool:
    """Append row to confirmation_summary.csv if method not present. Returns True if written."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, sep=";", decimal=",")
    if "method" in df.columns and (df["method"].astype(str) == "xgb_default").any():
        return False

    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Keep column order stable (append any new cols at the end)
    preferred = [
        "Accuracy_Mean",
        "Precision_Mean",
        "Recall_Mean",
        "Specificity_Mean",
        "Time_MeanFold",
        "hyperparameters",
        "method",
        "budget",
        "Optimization_Time_Seconds",
        "Total_Time_Seconds",
    ]
    cols = [c for c in preferred if c in df2.columns] + [c for c in df2.columns if c not in preferred]
    df2 = df2[cols]

    save_csv_ptbr(df2, csv_path)
    return True


def _build_union(exp_dir: Path, out_name: str = "confirmation_summary_all_replicas.csv") -> Path:
    """Create a union CSV across replicas adding replica+seed columns (no aggregation)."""
    frames: List[pd.DataFrame] = []
    for rep_dir in sorted(exp_dir.glob("replica_*")):
        csv_path = rep_dir / "confirmation_summary.csv"
        if not csv_path.exists():
            continue
        seed = _read_seed(rep_dir)
        replica = int(rep_dir.name.split("_")[1])
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df.insert(0, "replica", replica)
        df.insert(1, "seed", seed)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No confirmation_summary.csv found under {exp_dir}")

    union_df = pd.concat(frames, ignore_index=True)
    out_path = exp_dir / out_name
    union_df.to_csv(out_path, sep=";", decimal=",", index=False, encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Append XGBoost default baseline to existing confirmation_summary.csv files",
    )
    p.add_argument(
        "--exp-dir",
        required=True,
        help="Experiment directory containing replica_XX folders (e.g., experiments/telescope2/hyperparameter_design)",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset path. If omitted, inferred from replica_01/manifest.json.",
    )
    p.add_argument("--target", default="y", help="Target column name (default: y)")
    p.add_argument("--n-splits", type=int, default=5, help="CV folds (default: 5)")
    p.add_argument("--n-jobs", type=int, default=-1, help="XGBoost n_jobs (default: -1)")
    p.add_argument("--tree-method", default="hist", help="XGBoost tree_method (default: hist)")
    p.add_argument("--eval-metric", default="logloss", help="XGBoost eval_metric (default: logloss)")
    p.add_argument(
        "--write-union",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write confirmation_summary_all_replicas.csv after patching (default: true)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of replicas processed (0 = all). Useful for quick tests.",
    )
    args = p.parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    replica_dirs = sorted([d for d in exp_dir.glob("replica_*") if d.is_dir()])
    if not replica_dirs:
        raise FileNotFoundError(f"No replica_* dirs found under {exp_dir}")

    if int(args.limit) > 0:
        replica_dirs = replica_dirs[: int(args.limit)]

    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else _resolve_dataset_from_manifest(replica_dirs[0])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    # Load once (same dataset for all replicas)
    X, y = load_dataset(dataset_path, target_col=args.target, target_map={"g": 0, "h": 1})

    written = 0
    skipped = 0
    total = len(replica_dirs)
    t_all = time.perf_counter()
    for idx, rep_dir in enumerate(replica_dirs, start=1):
        seed = _read_seed(rep_dir)
        rep_num = rep_dir.name.replace("replica_", "")
        print(f"[{idx:02d}/{total:02d}] replica_{rep_num} (seed={seed}) ...", flush=True)
        row = _evaluate_default(
            X,
            y,
            seed=seed,
            n_splits=int(args.n_splits),
            n_jobs=int(args.n_jobs),
            tree_method=str(args.tree_method),
            eval_metric=str(args.eval_metric),
        )

        csv_path = rep_dir / "confirmation_summary.csv"
        did_write = _append_row_if_missing(csv_path, row)
        if did_write:
            written += 1
        else:
            skipped += 1

    elapsed = time.perf_counter() - t_all
    print(f"✅ xgb_default appended to {written} replicas (skipped {skipped} already-present).")
    print(f"⏱️  Elapsed: {elapsed:.1f}s")

    if bool(args.write_union):
        out_union = _build_union(exp_dir)
        print(f"✅ Union written: {out_union}")


if __name__ == "__main__":
    main()
