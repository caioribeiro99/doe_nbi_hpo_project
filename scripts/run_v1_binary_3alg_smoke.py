#!/usr/bin/env python
"""Three-algorithm binary smoke for the article-track v1 panel.

Loads the three small fetched binary datasets and evaluates each with
XGBoost / LightGBM / CatBoost at a single safe hyperparameter point
under 2-fold stratified CV. Asserts the dissertation-era binary metric
keys are populated, accuracy is at least 0.50, and runtime is finite.
Writes ``experiments/_v1_smoke/binary_3alg_smoke_output.json``.

Does NOT run DOE / RSM / NBI / MBPA. Does NOT load Dry Bean.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import warnings
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
    load_german_credit,
    load_pima_diabetes,
    load_spambase,
)
from doe_xgb.metrics import (  # noqa: E402
    aggregate_metric_dicts,
    compute_classification_metrics,
)


# ---------------------------------------------------------------------------
# Hyperparameters (one safe point per algorithm)
# ---------------------------------------------------------------------------


XGB_PARAMS: dict[str, float | int] = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "gamma": 0.1,
    "tree_method": "hist",
}

LGBM_PARAMS: dict[str, float | int] = {
    "n_estimators": 50,
    "num_leaves": 31,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 5,
    "verbose": -1,
}

CATBOOST_PARAMS: dict[str, float | int] = {
    "iterations": 50,
    "depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "rsm": 0.8,
    "min_data_in_leaf": 5,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
    "bootstrap_type": "Bernoulli",
}


EXPECTED_BINARY_KEYS = (
    "Accuracy_Mean",
    "Precision_Mean",
    "Recall_Mean",
    "Specificity_Mean",
)
ACCURACY_FLOOR = 0.50


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _categorical_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, col in enumerate(X.columns) if _is_categorical(X[col])]


def _encode_to_int_codes(X: pd.DataFrame) -> pd.DataFrame:
    """Encode any non-numeric column to deterministic integer category codes.

    Used by the XGBoost / LightGBM paths in this smoke (CatBoost does
    its own native categorical handling).
    """

    def _enc(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        return pd.Series(pd.Categorical(s).codes, index=s.index, dtype="int64")

    return X.apply(_enc)


def _stratified_subsample(
    X: pd.DataFrame, y: pd.Series, *, max_rows: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True)
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


# ---------------------------------------------------------------------------
# Per-(dataset, algorithm) runner
# ---------------------------------------------------------------------------


def _evaluate_xgboost(X: pd.DataFrame, y: pd.Series, *, seed: int, n_splits: int) -> dict:
    from xgboost import XGBClassifier

    Xn = _encode_to_int_codes(X).to_numpy()
    yn = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, fold_times = [], []
    for tr, te in kfold.split(Xn, yn):
        clf = XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k != "tree_method"},
            tree_method=XGB_PARAMS["tree_method"],
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        )
        t0 = time.perf_counter()
        clf.fit(Xn[tr], yn[tr])
        y_pred = clf.predict(Xn[te])
        fold_times.append(time.perf_counter() - t0)
        folds.append(compute_classification_metrics(yn[te], y_pred, task_type="binary"))
    metrics = aggregate_metric_dicts(folds)
    metrics["Time_MeanFold"] = float(np.mean(fold_times))
    return {"metrics": metrics, "preprocessing_mode": "encoded_int_codes"}


def _evaluate_lightgbm(X: pd.DataFrame, y: pd.Series, *, seed: int, n_splits: int) -> dict:
    from lightgbm import LGBMClassifier

    Xn = _encode_to_int_codes(X).to_numpy()
    yn = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, fold_times = [], []
    for tr, te in kfold.split(Xn, yn):
        clf = LGBMClassifier(**LGBM_PARAMS, random_state=seed, n_jobs=1)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xn[tr], yn[tr])
        y_pred = clf.predict(Xn[te])
        fold_times.append(time.perf_counter() - t0)
        folds.append(compute_classification_metrics(yn[te], y_pred, task_type="binary"))
    metrics = aggregate_metric_dicts(folds)
    metrics["Time_MeanFold"] = float(np.mean(fold_times))
    return {"metrics": metrics, "preprocessing_mode": "encoded_int_codes"}


def _evaluate_catboost(
    X: pd.DataFrame, y: pd.Series, *, seed: int, n_splits: int
) -> dict:
    from catboost import CatBoostClassifier

    cat_idx = _categorical_indices(X)
    yn = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _try_native() -> tuple[list[dict], list[float]] | None:
        """Attempt CatBoost native categorical handling. Returns None
        on failure so the caller can fall back to encoded ints."""
        Xc = X.copy()
        for col in Xc.columns:
            if _is_categorical(Xc[col]):
                Xc[col] = Xc[col].astype(str)
        try:
            f, ft = [], []
            for tr, te in kfold.split(Xc, yn):
                clf = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
                t0 = time.perf_counter()
                clf.fit(Xc.iloc[tr], yn[tr], cat_features=cat_idx)
                y_pred = clf.predict(Xc.iloc[te]).ravel().astype(int)
                ft.append(time.perf_counter() - t0)
                f.append(compute_classification_metrics(yn[te], y_pred, task_type="binary"))
            return f, ft
        except Exception:  # pragma: no cover - exotic CatBoost paths
            return None

    out = _try_native() if cat_idx else None
    if out is not None:
        folds, fold_times = out
        mode = "catboost_native_categorical"
    else:
        Xn = _encode_to_int_codes(X).to_numpy()
        folds, fold_times = [], []
        for tr, te in kfold.split(Xn, yn):
            clf = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
            t0 = time.perf_counter()
            clf.fit(Xn[tr], yn[tr])
            y_pred = clf.predict(Xn[te]).ravel().astype(int)
            fold_times.append(time.perf_counter() - t0)
            folds.append(compute_classification_metrics(yn[te], y_pred, task_type="binary"))
        mode = "catboost_fallback_encoded_int_codes"

    metrics = aggregate_metric_dicts(folds)
    metrics["Time_MeanFold"] = float(np.mean(fold_times))
    return {"metrics": metrics, "preprocessing_mode": mode}


_EVALUATORS = {
    "xgboost": _evaluate_xgboost,
    "lightgbm": _evaluate_lightgbm,
    "catboost": _evaluate_catboost,
}


def _smoke_one(*, dataset_id, loader, max_rows, seed, n_splits, algorithm) -> dict:
    out_record: dict = {"dataset_id": dataset_id, "algorithm": algorithm, "ok": False, "warnings": []}
    started = time.perf_counter()
    try:
        ds = loader()
    except DatasetUnavailableError as e:
        out_record["reason"] = f"unavailable: {e}"
        return out_record
    X, y = ds.X, ds.y
    X_sub, y_sub = _stratified_subsample(X, y, max_rows=max_rows, seed=seed)
    cat_count = sum(1 for c in X_sub.columns if _is_categorical(X_sub[c]))
    out_record.update(
        {
            "n_rows": int(len(X_sub)),
            "n_features": int(X_sub.shape[1]),
            "class_distribution": {str(k): int(v) for k, v in y_sub.value_counts().to_dict().items()},
            "categorical_columns_count": int(cat_count),
        }
    )
    try:
        run = _EVALUATORS[algorithm](X_sub, y_sub, seed=seed, n_splits=n_splits)
    except Exception as e:  # pragma: no cover - defensive
        out_record["reason"] = f"evaluator_failed: {e}"
        return out_record

    metrics = run["metrics"]
    runtime = time.perf_counter() - started
    out_record.update(
        {
            "preprocessing_mode": run["preprocessing_mode"],
            "metrics": {k: float(v) for k, v in metrics.items()},
            "fit_time_mean_fold": float(metrics.get("Time_MeanFold", float("nan"))),
            "runtime_seconds": float(runtime),
            "metric_keys_present": {k: (k in metrics) for k in EXPECTED_BINARY_KEYS},
            "ok": True,
        }
    )
    return out_record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "experiments" / "_v1_smoke" / "binary_3alg_smoke_output.json",
    )
    args = parser.parse_args(argv)

    pkg_versions = {
        name: importlib.import_module(name).__version__
        for name in ("xgboost", "lightgbm", "catboost", "sklearn")
    }

    targets = [
        ("german_credit", load_german_credit),
        ("pima_diabetes", load_pima_diabetes),
        ("spambase", load_spambase),
    ]
    algorithms = ("xgboost", "lightgbm", "catboost")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    results: list[dict] = []
    for did, loader in targets:
        for algo in algorithms:
            print(f">>> {did:14s}  {algo}")
            results.append(
                _smoke_one(
                    dataset_id=did,
                    loader=loader,
                    max_rows=args.max_rows,
                    seed=args.seed,
                    n_splits=args.n_splits,
                    algorithm=algo,
                )
            )

    failures: list[str] = []
    for rec in results:
        tag = f"{rec['dataset_id']}/{rec['algorithm']}"
        if not rec.get("ok"):
            failures.append(f"{tag}: {rec.get('reason', 'unknown error')}")
            continue
        missing = [k for k, present in rec["metric_keys_present"].items() if not present]
        if missing:
            failures.append(f"{tag}: missing keys {missing}")
        acc = rec["metrics"].get("Accuracy_Mean", 0.0)
        if not (acc >= ACCURACY_FLOOR):  # NaN-safe
            failures.append(f"{tag}: accuracy {acc:.3f} below floor {ACCURACY_FLOOR}")
        rt = rec.get("runtime_seconds", float("inf"))
        if not np.isfinite(rt):
            failures.append(f"{tag}: non-finite runtime")

    payload = {
        "smoke": "v1_binary_3alg",
        "started_at": started_at,
        "package_versions": pkg_versions,
        "params": {
            "xgboost": XGB_PARAMS,
            "lightgbm": LGBM_PARAMS,
            "catboost": CATBOOST_PARAMS,
        },
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "datasets": [d for d, _ in targets],
        "algorithms": list(algorithms),
        "results": results,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1

    print("\nAll (dataset, algorithm) pairs passed.")
    print(f"  package versions: {pkg_versions}")
    for rec in results:
        if not rec.get("ok"):
            continue
        m = rec["metrics"]
        print(
            f"  {rec['dataset_id']:14s} {rec['algorithm']:9s} "
            f"acc={m.get('Accuracy_Mean', float('nan')):.3f} "
            f"prec={m.get('Precision_Mean', float('nan')):.3f} "
            f"rec={m.get('Recall_Mean', float('nan')):.3f} "
            f"spec={m.get('Specificity_Mean', float('nan')):.3f} "
            f"t/fold={m.get('Time_MeanFold', float('nan')):.3f}s "
            f"prep={rec['preprocessing_mode']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
