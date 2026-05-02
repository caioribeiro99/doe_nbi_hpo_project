#!/usr/bin/env python
"""Full-dataset 5-fold runtime profiler for the article v1 panel.

For each of the twelve v1 datasets and each of the three GBDT
algorithms, runs a single safe hyperparameter point under 5-fold
stratified CV on the full dataset (no row cap by default), captures
metrics + per-fold timings, and projects campaign cost under several
inflation multipliers and execution profiles.

This script does NOT run DOE / RSM / NBI / MBPA. Continues on per-pair
failure and records the error.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import traceback
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
    REGISTRY,
    DatasetUnavailableError,
    load_adult,
    load_bank_marketing,
    load_breast_cancer,
    load_credit_card_default,
    load_dry_bean,
    load_german_credit,
    load_magic,
    load_mushroom,
    load_phishing,
    load_pima_diabetes,
    load_spambase,
    load_wine_quality,
)
from doe_xgb.metrics import (  # noqa: E402
    aggregate_metric_dicts,
    compute_classification_metrics,
)


LOADERS = {
    "magic": load_magic,
    "breast_cancer": load_breast_cancer,
    "pima_diabetes": load_pima_diabetes,
    "spambase": load_spambase,
    "adult": load_adult,
    "bank_marketing": load_bank_marketing,
    "credit_card_default": load_credit_card_default,
    "german_credit": load_german_credit,
    "wine_quality": load_wine_quality,
    "dry_bean": load_dry_bean,
    "mushroom": load_mushroom,
    "phishing": load_phishing,
}

DATASET_ORDER = (
    "magic", "breast_cancer", "pima_diabetes", "spambase", "adult",
    "bank_marketing", "credit_card_default", "german_credit",
    "wine_quality", "dry_bean", "mushroom", "phishing",
)

ALGORITHMS = ("xgboost", "lightgbm", "catboost")


# ---------------------------------------------------------------------------
# Hyperparameter points
# ---------------------------------------------------------------------------


XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "gamma": 0.1,
    "tree_method": "hist",
}
LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 5,
    "verbose": -1,
}
CATBOOST_PARAMS = {
    "iterations": 100,
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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _categorical_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if _is_categorical(X[c])]


def _encode_to_int_codes(X: pd.DataFrame) -> pd.DataFrame:
    def _enc(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        return pd.Series(pd.Categorical(s).codes, index=s.index, dtype="int64")

    return X.apply(_enc)


def _maybe_subsample(
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


def _xgb_run(X, y, cat_idx, *, task: str, seed: int, n_splits: int) -> tuple[list[dict], list[float], str]:
    from xgboost import XGBClassifier

    Xn = _encode_to_int_codes(X).to_numpy()
    yn = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, fold_times = [], []
    for tr, te in kfold.split(Xn, yn):
        kw = {**{k: v for k, v in XGB_PARAMS.items() if k != "tree_method"},
              "tree_method": XGB_PARAMS["tree_method"],
              "random_state": seed, "n_jobs": 1, "verbosity": 0,
              "eval_metric": "mlogloss" if task == "multiclass" else "logloss"}
        if task == "multiclass":
            kw["objective"] = "multi:softprob"
            kw["num_class"] = int(len(np.unique(yn)))
        clf = XGBClassifier(**kw)
        t0 = time.perf_counter()
        clf.fit(Xn[tr], yn[tr])
        y_pred = clf.predict(Xn[te])
        y_prob = clf.predict_proba(Xn[te]) if task == "multiclass" else None
        fold_times.append(time.perf_counter() - t0)
        folds.append(compute_classification_metrics(yn[te], y_pred, y_prob=y_prob, task_type=task))
    return folds, fold_times, "encoded_int_codes"


def _lgbm_run(X, y, cat_idx, *, task: str, seed: int, n_splits: int) -> tuple[list[dict], list[float], str]:
    from lightgbm import LGBMClassifier

    Xn = _encode_to_int_codes(X).to_numpy()
    yn = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, fold_times = [], []
    for tr, te in kfold.split(Xn, yn):
        kw = {**LGBM_PARAMS, "random_state": seed, "n_jobs": 1}
        if task == "multiclass":
            kw["objective"] = "multiclass"
            kw["num_class"] = int(len(np.unique(yn)))
        clf = LGBMClassifier(**kw)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xn[tr], yn[tr])
        y_pred = clf.predict(Xn[te])
        y_prob = clf.predict_proba(Xn[te]) if task == "multiclass" else None
        fold_times.append(time.perf_counter() - t0)
        folds.append(compute_classification_metrics(yn[te], y_pred, y_prob=y_prob, task_type=task))
    return folds, fold_times, "encoded_int_codes"


def _catboost_run(X, y, cat_idx, *, task: str, seed: int, n_splits: int) -> tuple[list[dict], list[float], str]:
    from catboost import CatBoostClassifier

    yn = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _native_run() -> tuple[list[dict], list[float]] | None:
        try:
            Xc = X.copy()
            for col in Xc.columns:
                if _is_categorical(Xc[col]):
                    Xc[col] = Xc[col].astype(str)
            f, ft = [], []
            for tr, te in kfold.split(Xc, yn):
                kw = {**CATBOOST_PARAMS, "random_seed": seed}
                if task == "multiclass":
                    kw["loss_function"] = "MultiClass"
                clf = CatBoostClassifier(**kw)
                t0 = time.perf_counter()
                clf.fit(Xc.iloc[tr], yn[tr], cat_features=cat_idx)
                y_pred = clf.predict(Xc.iloc[te]).ravel().astype(int)
                y_prob = clf.predict_proba(Xc.iloc[te]) if task == "multiclass" else None
                ft.append(time.perf_counter() - t0)
                f.append(compute_classification_metrics(yn[te], y_pred, y_prob=y_prob, task_type=task))
            return f, ft
        except Exception:  # pragma: no cover
            return None

    out = _native_run() if cat_idx else None
    if out is not None:
        return out[0], out[1], "catboost_native_categorical"

    Xn = _encode_to_int_codes(X).to_numpy()
    folds, fold_times = [], []
    for tr, te in kfold.split(Xn, yn):
        kw = {**CATBOOST_PARAMS, "random_seed": seed}
        if task == "multiclass":
            kw["loss_function"] = "MultiClass"
        clf = CatBoostClassifier(**kw)
        t0 = time.perf_counter()
        clf.fit(Xn[tr], yn[tr])
        y_pred = clf.predict(Xn[te]).ravel().astype(int)
        y_prob = clf.predict_proba(Xn[te]) if task == "multiclass" else None
        fold_times.append(time.perf_counter() - t0)
        folds.append(compute_classification_metrics(yn[te], y_pred, y_prob=y_prob, task_type=task))
    return folds, fold_times, "catboost_fallback_encoded_int_codes"


_RUNNERS = {"xgboost": _xgb_run, "lightgbm": _lgbm_run, "catboost": _catboost_run}


def _profile_one(*, dataset_id: str, algorithm: str, max_rows: int, seed: int, n_splits: int) -> dict:
    rec: dict = {"dataset_id": dataset_id, "algorithm": algorithm, "ok": False, "warnings": []}
    started = time.perf_counter()
    try:
        ds = LOADERS[dataset_id]()
    except DatasetUnavailableError as e:
        rec["error"] = f"unavailable: {e}"
        return rec
    except Exception as e:
        rec["error"] = f"loader_failed: {e}"
        rec["traceback"] = traceback.format_exc(limit=4)
        return rec

    X, y = ds.X, ds.y
    task = REGISTRY[dataset_id].task_type
    rec["task_type"] = task
    rec["n_rows_full"] = int(len(X))
    rec["n_features"] = int(X.shape[1])
    rec["categorical_columns_count"] = sum(1 for c in X.columns if _is_categorical(X[c]))

    X_use, y_use = _maybe_subsample(X, y, max_rows=max_rows, seed=seed)
    rec["n_rows_used"] = int(len(X_use))
    rec["row_capped"] = bool(rec["n_rows_used"] != rec["n_rows_full"])
    counts = y_use.value_counts().to_dict()
    rec["class_distribution"] = {str(k): int(v) for k, v in counts.items()}
    rec["n_classes"] = int(len(counts))

    cat_idx = _categorical_indices(X_use)

    try:
        folds, fold_times, mode = _RUNNERS[algorithm](X_use, y_use, cat_idx, task=task, seed=seed, n_splits=n_splits)
    except Exception as e:
        rec["error"] = f"runner_failed: {e}"
        rec["traceback"] = traceback.format_exc(limit=4)
        return rec

    metrics = aggregate_metric_dicts(folds)
    metrics["Time_MeanFold"] = float(np.mean(fold_times))
    rec.update({
        "preprocessing_mode": mode,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "fold_times_seconds": [round(t, 4) for t in fold_times],
        "total_runtime_seconds": float(time.perf_counter() - started),
        "mean_fold_runtime_seconds": float(np.mean(fold_times)),
        "ok": True,
    })
    return rec


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


# Article-track per-replica budget = doe_runs (88) + nbi_eval_k (50) +
# 4 benchmarks * 138 evals = 690 evaluations. Each is one full
# n_splits-CV pass of the model on the dataset; treat a profiler 5-fold
# pass as one such evaluation.
EVALS_PER_REPLICA_PER_PAIR = 690


def _local_profiles() -> dict[str, dict[str, float]]:
    return {
        "caio_mac_only_6w_14h_eff070": {"workers": 6, "hours_per_day": 14, "efficiency": 0.70},
        "dedicated_mac_10w_24h_eff065": {"workers": 10, "hours_per_day": 24, "efficiency": 0.65},
        "dedicated_mac_10w_24h_eff070": {"workers": 10, "hours_per_day": 24, "efficiency": 0.70},
        "dedicated_mac_10w_24h_eff075": {"workers": 10, "hours_per_day": 24, "efficiency": 0.75},
        "two_macs_combined_16w_24h_eff065": {"workers": 16, "hours_per_day": 24, "efficiency": 0.65},
        "two_macs_combined_16w_24h_eff070": {"workers": 16, "hours_per_day": 24, "efficiency": 0.70},
        "two_macs_combined_16w_24h_eff075": {"workers": 16, "hours_per_day": 24, "efficiency": 0.75},
    }


def _wall_days_local(cpu_hours: float, profile: dict) -> float:
    daily = profile["workers"] * profile["hours_per_day"] * profile["efficiency"]
    return cpu_hours / daily if daily > 0 else float("inf")


def _cloud_estimate(cpu_hours: float, *, workers: int = 32, price: float = 0.10, eff: float = 0.85) -> dict:
    cpu_hours_billed = cpu_hours / max(0.01, eff)
    wall_hours = cpu_hours / max(1, workers) / max(0.01, eff)
    return {
        "workers": workers,
        "instance_hourly_price_per_worker_usd": price,
        "efficiency": eff,
        "wall_hours": round(wall_hours, 2),
        "wall_days": round(wall_hours / 24.0, 2),
        "cost_usd": round(cpu_hours_billed * price, 2),
    }


def _project(
    *,
    label: str,
    pair_seconds: dict[tuple[str, str], float],
    n_datasets: int,
    n_algorithms: int,
    n_replicas: int,
    multiplier: float,
    cloud_eff: float = 0.85,
) -> dict:
    measured = list(pair_seconds.values())
    if not measured:
        return {"label": label, "error": "no measured pairs"}

    # If projection scope == measured panel size, sum exactly.
    # Otherwise, scale by mean(measured) * desired pair count.
    desired_pairs = n_datasets * n_algorithms
    if desired_pairs == len(measured):
        per_replica_seconds = sum(measured) * EVALS_PER_REPLICA_PER_PAIR
    else:
        per_replica_seconds = float(np.mean(measured)) * desired_pairs * EVALS_PER_REPLICA_PER_PAIR
    total_cpu_seconds = per_replica_seconds * n_replicas * multiplier
    total_cpu_hours = total_cpu_seconds / 3600.0

    locals_ = {
        name: round(_wall_days_local(total_cpu_hours, prof), 2)
        for name, prof in _local_profiles().items()
    }
    return {
        "label": label,
        "n_datasets": n_datasets,
        "n_algorithms": n_algorithms,
        "n_replicas": n_replicas,
        "multiplier": multiplier,
        "evals_per_replica_per_pair": EVALS_PER_REPLICA_PER_PAIR,
        "total_cpu_hours": round(total_cpu_hours, 1),
        "total_cpu_days": round(total_cpu_hours / 24.0, 2),
        "local_wall_days": locals_,
        "cloud_32w_010_eff085": _cloud_estimate(total_cpu_hours, eff=cloud_eff),
    }


def _slowest_fastest(results: list[dict]) -> dict:
    pairs = [(r["dataset_id"], r["algorithm"], r["total_runtime_seconds"]) for r in results if r.get("ok")]
    pairs.sort(key=lambda x: x[2])
    return {
        "fastest": [{"pair": f"{a}/{b}", "seconds": round(c, 3)} for a, b, c in pairs[:3]],
        "slowest": [{"pair": f"{a}/{b}", "seconds": round(c, 3)} for a, b, c in pairs[-3:]],
    }


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def _emit_md(report: dict, out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Article v1 full-dataset 5-fold runtime profile\n\n")
    lines.append("Generated by `scripts/profile_v1_full_dataset_runtime.py`.\n")
    lines.append("Single safe hyperparameter point per algorithm; 5-fold stratified CV; "
                 "no DOE / RSM / NBI / MBPA. Runtime budget assumption per replica per "
                 f"(dataset, algorithm) = **{EVALS_PER_REPLICA_PER_PAIR} evaluations**.\n\n")
    lines.append("## Environment\n\n")
    pv = report["package_versions"]
    lines.append(f"- Python {report['python']}\n- Platform: {report['platform']}\n")
    lines.append(f"- xgboost {pv['xgboost']} | lightgbm {pv['lightgbm']} | "
                 f"catboost {pv['catboost']} | sklearn {pv['sklearn']}\n\n")
    lines.append("## Per-(dataset, algorithm) measurements\n\n")
    lines.append("| Dataset | Algorithm | Task | n_rows | n_feats | cat | total_s | mean_fold_s | metric_keys | mode | ok |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|---|\n")
    for rec in report["results"]:
        if not rec.get("ok"):
            lines.append(f"| {rec['dataset_id']} | {rec['algorithm']} | "
                         f"{rec.get('task_type','?')} | "
                         f"{rec.get('n_rows_used','?')} | {rec.get('n_features','?')} | "
                         f"{rec.get('categorical_columns_count','?')} | "
                         f"-- | -- | -- | -- | **FAIL: {rec.get('error','')}** |\n")
            continue
        m = rec["metrics"]
        head_metric = (
            f"acc={m.get('Accuracy_Mean', float('nan')):.3f}"
            if rec["task_type"] == "binary"
            else f"f1m={m.get('F1Macro_Mean', float('nan')):.3f}"
        )
        lines.append(
            f"| {rec['dataset_id']} | {rec['algorithm']} | {rec['task_type']} | "
            f"{rec['n_rows_used']} | {rec['n_features']} | "
            f"{rec['categorical_columns_count']} | "
            f"{rec['total_runtime_seconds']:.2f} | "
            f"{rec['mean_fold_runtime_seconds']:.3f} | "
            f"{head_metric} | {rec['preprocessing_mode']} | OK |\n"
        )
    lines.append("\n")
    lines.append("## Fastest / slowest pairs\n\n")
    sf = report["fastest_slowest"]
    fast_str = ", ".join(f"{p['pair']} ({p['seconds']:.2f}s)" for p in sf["fastest"])
    slow_str = ", ".join(f"{p['pair']} ({p['seconds']:.2f}s)" for p in sf["slowest"])
    lines.append(f"- Fastest: {fast_str}\n")
    lines.append(f"- Slowest: {slow_str}\n\n")

    lines.append("## Projections\n\n")
    lines.append(
        "Projections multiply the measured 5-fold time per (dataset, algorithm) by\n"
        f"{EVALS_PER_REPLICA_PER_PAIR} evals/replica and by the chosen number of replicas,\n"
        "then by an inflation multiplier (1x / 2x / 4x / 8x). The 82-dataset thesis\n"
        "rows scale by the *mean* per-pair time (we have no measurements outside the\n"
        "v1 panel).\n\n"
    )
    lines.append("| Scope | Replicas | Mult | CPU-h | DedicatedMac (eff070) wall-days | TwoMacs (eff070) wall-days | Cloud32w wall-days | Cloud cost USD |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for proj in report["projections"]:
        if "error" in proj:
            continue
        loc = proj["local_wall_days"]
        cl = proj["cloud_32w_010_eff085"]
        lines.append(
            f"| {proj['label']} | {proj['n_replicas']} | x{proj['multiplier']:g} | "
            f"{proj['total_cpu_hours']} | "
            f"{loc.get('dedicated_mac_10w_24h_eff070','?')} | "
            f"{loc.get('two_macs_combined_16w_24h_eff070','?')} | "
            f"{cl['wall_days']} | ${cl['cost_usd']} |\n"
        )

    lines.append("\n## Recommendation\n\n")
    lines.append(report.get("recommendation", "(not generated)") + "\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=0, help="Emergency safety valve; 0 = no cap.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=REPO / "experiments" / "_runtime_profile" / "v1_full_dataset_5fold_profile.json")
    parser.add_argument("--out-md", type=Path, default=REPO / "experiments" / "_runtime_profile" / "v1_full_dataset_5fold_profile.md")
    args = parser.parse_args(argv)

    pv = {name: importlib.import_module(name).__version__ for name in ("xgboost", "lightgbm", "catboost", "sklearn")}

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    results: list[dict] = []
    for did in DATASET_ORDER:
        for algo in ALGORITHMS:
            print(f">>> {did:22s} {algo}", flush=True)
            results.append(
                _profile_one(
                    dataset_id=did,
                    algorithm=algo,
                    max_rows=args.max_rows,
                    seed=args.seed,
                    n_splits=args.n_splits,
                )
            )

    pair_seconds = {
        (r["dataset_id"], r["algorithm"]): float(r["total_runtime_seconds"])
        for r in results if r.get("ok")
    }
    headline_pair_seconds = {
        k: v for k, v in pair_seconds.items() if k[0] != "dry_bean"
    }

    multipliers = (1.0, 2.0, 4.0, 8.0)
    projections: list[dict] = []
    for mult in multipliers:
        for label, n_d, n_a, n_r, ps in (
            ("article_v1_12x3x10",   12, 3, 10, pair_seconds),
            ("article_v1_12x3x30",   12, 3, 30, pair_seconds),
            ("article_v1_11binx3x10",11, 3, 10, headline_pair_seconds),
            ("article_v1_11binx3x30",11, 3, 30, headline_pair_seconds),
            ("thesis_82x3x10",       82, 3, 10, pair_seconds),
            ("thesis_82x3x30",       82, 3, 30, pair_seconds),
        ):
            projections.append(_project(
                label=label, pair_seconds=ps, n_datasets=n_d, n_algorithms=n_a,
                n_replicas=n_r, multiplier=mult,
            ))

    failures = [r for r in results if not r.get("ok")]
    sf = _slowest_fastest(results)

    # Recommendation logic: pick the median 4x article 12x3x10 dedicated_mac wall-days.
    article_12x3x10_4x = next(
        (p for p in projections if p["label"] == "article_v1_12x3x10" and p["multiplier"] == 4.0),
        None,
    )
    rec_text: list[str] = []
    if article_12x3x10_4x is not None:
        wd = article_12x3x10_4x["local_wall_days"].get("dedicated_mac_10w_24h_eff070", float("inf"))
        cwd = article_12x3x10_4x["cloud_32w_010_eff085"]["wall_days"]
        cost = article_12x3x10_4x["cloud_32w_010_eff085"]["cost_usd"]
        rec_text.append(
            f"At the realistic 4x inflation, the headline 12x3x10 panel projects to "
            f"~{wd} days on a dedicated Mac and "
            f"~{cwd} days / ${cost} on a 32-worker $0.10/h cloud. "
        )
    article_12x3x30_4x = next(
        (p for p in projections if p["label"] == "article_v1_12x3x30" and p["multiplier"] == 4.0), None,
    )
    if article_12x3x30_4x is not None:
        rec_text.append(
            f"30 replicas at the same 4x inflation multiplies CPU-hours to "
            f"~{article_12x3x30_4x['total_cpu_hours']}h, dedicated-Mac wall days "
            f"~{article_12x3x30_4x['local_wall_days']['dedicated_mac_10w_24h_eff070']}. "
        )
    thesis_82x3x10_4x = next(
        (p for p in projections if p["label"] == "thesis_82x3x10" and p["multiplier"] == 4.0), None,
    )
    if thesis_82x3x10_4x is not None:
        wd = thesis_82x3x10_4x["local_wall_days"].get("dedicated_mac_10w_24h_eff070", float("inf"))
        cwd = thesis_82x3x10_4x["cloud_32w_010_eff085"]["wall_days"]
        cost = thesis_82x3x10_4x["cloud_32w_010_eff085"]["cost_usd"]
        rec_text.append(
            f"The 82-dataset doctoral panel at 10 replicas projects to "
            f"~{wd} days on a dedicated Mac (~{cwd} cloud days, ${cost}) under the "
            "same per-pair-mean assumption; do not start it locally past 1 replica. "
        )
    rec_text.append(
        "Recommendation: run the full 12x3x10 article v1 locally on the dedicated Mac. "
        "Reserve 30 replicas for *selected* datasets, not the panel. Run the doctoral "
        "82-dataset benchmark at 1 replica locally first as a sizing check before "
        "scaling out to cloud."
    )

    payload = {
        "profile": "v1_full_dataset_5fold",
        "started_at": started_at,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": pv,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "params": {"xgboost": XGB_PARAMS, "lightgbm": LGBM_PARAMS, "catboost": CATBOOST_PARAMS},
        "results": results,
        "n_pairs_ok": int(len(pair_seconds)),
        "n_failures": int(len(failures)),
        "fastest_slowest": sf,
        "projections": projections,
        "recommendation": "".join(rec_text),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit_md(payload, args.out_md)
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"\nOK pairs: {len(pair_seconds)} / {len(results)}; failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"  FAIL {f['dataset_id']}/{f['algorithm']}: {f.get('error','?')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
