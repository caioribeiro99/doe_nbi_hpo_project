#!/usr/bin/env python
"""Tiny binary smoke for the article-track v1 panel.

Loads three small fetched datasets (and optionally Breast Cancer
offline), runs ``evaluate_xgb_cv`` once each at a single safe
hyperparameter point with 2-fold CV, asserts that the
dissertation-era binary metric keys are populated, and writes
``experiments/_v1_smoke/binary_smoke_output.json``.

Does NOT run DOE / RSM / NBI / MBPA. Does NOT run multiclass datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.datasets import (  # noqa: E402
    DatasetUnavailableError,
    load_breast_cancer,
    load_german_credit,
    load_pima_diabetes,
    load_spambase,
)
from doe_xgb.evaluation import evaluate_xgb_cv  # noqa: E402


SAFE_PARAMS: dict[str, float | int] = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "gamma": 0.1,
}

EXPECTED_BINARY_KEYS = (
    "Accuracy_Mean",
    "Precision_Mean",
    "Recall_Mean",
    "Specificity_Mean",
)


def _stratified_subsample(
    X: pd.DataFrame, y: pd.Series, *, max_rows: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    fractions = counts / counts.sum()
    take = np.maximum(1, np.round(fractions * max_rows).astype(int)).astype(int)
    idx_per_class: list[np.ndarray] = []
    for c, k in zip(classes, take, strict=True):
        cand = np.flatnonzero(y.to_numpy() == c)
        chosen = rng.choice(cand, size=min(int(k), len(cand)), replace=False)
        idx_per_class.append(chosen)
    idx = np.concatenate(idx_per_class)
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _smoke_one(
    *, dataset_id: str, loader, max_rows: int, seed: int, n_splits: int
) -> dict:
    t0 = time.perf_counter()
    try:
        ds = loader()
    except DatasetUnavailableError as e:
        return {
            "dataset_id": dataset_id,
            "ok": False,
            "reason": f"unavailable: {e}",
        }
    X, y = ds.X, ds.y

    # Convert any non-numeric column to integer category codes for
    # XGBoost. The article smoke does not exercise CatBoost; one-hot
    # is not required for this validation.
    def _encode(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        return pd.Series(pd.Categorical(s).codes, index=s.index, dtype="int64")

    X = X.apply(_encode)

    X_sub, y_sub = _stratified_subsample(X, y, max_rows=max_rows, seed=seed)
    X_np = X_sub.to_numpy()
    y_np = y_sub.to_numpy()

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ev = evaluate_xgb_cv(
        SAFE_PARAMS,
        X_np,
        y_np,
        kfold,
        seed=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        task_type="binary",
    )

    runtime = time.perf_counter() - t0
    metric_keys_present = {k: (k in ev.metrics) for k in EXPECTED_BINARY_KEYS}
    counts = y_sub.value_counts().to_dict()
    return {
        "dataset_id": dataset_id,
        "ok": True,
        "task_resolved": ev.task,
        "n_rows": int(len(X_sub)),
        "n_features": int(X_sub.shape[1]),
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "metrics": {k: float(v) for k, v in ev.metrics.items()},
        "time_mean_fold": float(ev.time_mean_fold),
        "runtime_seconds": float(runtime),
        "metric_keys_present": metric_keys_present,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1500,
        help="Per-dataset stratified subsample cap (0 = no subsample).",
    )
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-breast-cancer",
        action="store_true",
        help="Also smoke Breast Cancer (sklearn-bundled; offline).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "experiments" / "_v1_smoke" / "binary_smoke_output.json",
    )
    args = parser.parse_args(argv)

    targets = [
        ("german_credit", load_german_credit),
        ("pima_diabetes", load_pima_diabetes),
        ("spambase", load_spambase),
    ]
    if args.include_breast_cancer:
        targets.append(("breast_cancer", load_breast_cancer))

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    results: list[dict] = []
    for did, loader in targets:
        print(f">>> smoking {did} ...")
        rec = _smoke_one(
            dataset_id=did,
            loader=loader,
            max_rows=args.max_rows,
            seed=args.seed,
            n_splits=args.n_splits,
        )
        results.append(rec)

    failures: list[str] = []
    for rec in results:
        if not rec.get("ok"):
            failures.append(f"{rec['dataset_id']}: {rec.get('reason')}")
            continue
        if rec["task_resolved"] != "binary":
            failures.append(f"{rec['dataset_id']}: expected binary, got {rec['task_resolved']}")
        missing = [k for k, present in rec["metric_keys_present"].items() if not present]
        if missing:
            failures.append(f"{rec['dataset_id']}: missing keys {missing}")

    payload = {
        "smoke": "v1_binary",
        "started_at": started_at,
        "params": SAFE_PARAMS,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "results": results,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1

    print("\nAll three binary datasets returned the dissertation-era metric keys.")
    for rec in results:
        if rec.get("ok"):
            m = rec["metrics"]
            print(
                f"  {rec['dataset_id']:18s} "
                f"acc={m.get('Accuracy_Mean', float('nan')):.3f} "
                f"prec={m.get('Precision_Mean', float('nan')):.3f} "
                f"rec={m.get('Recall_Mean', float('nan')):.3f} "
                f"spec={m.get('Specificity_Mean', float('nan')):.3f} "
                f"t/fold={rec['time_mean_fold']:.3f}s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
