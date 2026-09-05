"""Heterogeneous five-model zoo with fold-safe preprocessing, per-replication
OOF generation and direct inference-cost measurement (benchmark version).

Zoo (fixed, documented configurations; no hyperparameter search):
  lr   StandardScaler → LogisticRegression(C=1, max_iter=2000)
  gnb  GaussianNB
  knn  StandardScaler → KNeighborsClassifier(k=100, brute, n_jobs=-1)
  rf   RandomForestClassifier(200 trees, min_samples_leaf=20, n_jobs=-1)
  xgb  XGBClassifier(hist, 400 trees, lr 0.1, depth 6, subsample 0.8, colsample 0.8)

Preprocessing is a ColumnTransformer fitted INSIDE each training fold:
  numeric     → median imputation (+ missing indicator) [→ scaler for lr/knn]
  categorical → OneHotEncoder(min_frequency=0.005) for lr/gnb/knn,
                OrdinalEncoder(unknown → -1) for rf/xgb; missing = own level
  binary      → passthrough
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from mixens.datasets import DatasetSpec

MODEL_NAMES = ["lr", "gnb", "knn", "rf", "xgb"]
CLIP_EPS = 1e-3


def _preprocessor(spec: DatasetSpec, *, scale: bool, onehot: bool) -> ColumnTransformer:
    num = spec.numeric_cols
    cats = spec.categorical_cols
    bins = spec.binary_cols
    transformers = []
    if num:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=spec.missing_fraction > 0))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num))
    if cats:
        if onehot:
            enc = Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                         sparse_output=False, dtype=np.float32)),
            ])
        else:
            enc = Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                           dtype=np.float32)),
            ])
        transformers.append(("cat", enc, cats))
    if bins:
        transformers.append(("bin", "passthrough", bins))
    return ColumnTransformer(transformers, sparse_threshold=0.0, n_jobs=None)


def make_model(name: str, spec: DatasetSpec, random_state: int = 42) -> Pipeline:
    if name == "lr":
        pre = _preprocessor(spec, scale=True, onehot=True)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    elif name == "gnb":
        pre = _preprocessor(spec, scale=False, onehot=True)
        clf = GaussianNB()
    elif name == "knn":
        pre = _preprocessor(spec, scale=True, onehot=True)
        clf = KNeighborsClassifier(n_neighbors=100, algorithm="brute", n_jobs=-1)
    elif name == "rf":
        pre = _preprocessor(spec, scale=False, onehot=False)
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1,
                                     random_state=random_state)
    elif name == "xgb":
        pre = _preprocessor(spec, scale=False, onehot=False)
        clf = XGBClassifier(tree_method="hist", n_estimators=400, learning_rate=0.1, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                            random_state=random_state)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("pre", pre), ("clf", clf)])


def model_params(spec: DatasetSpec) -> dict:
    return {n: {k: v for k, v in make_model(n, spec).named_steps["clf"].get_params().items()
                if isinstance(v, (int, float, str, bool)) or v is None} for n in MODEL_NAMES}


def clip_probs(p: np.ndarray, eps: float = CLIP_EPS) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


@dataclass
class ReplicationOOF:
    rep: int
    seed: int
    model_names: list[str]
    P: np.ndarray            # (n_train, M) clipped OOF probabilities
    fold_ids: np.ndarray     # (n_train,)
    Q: np.ndarray            # (n_test, M) clipped holdout probabilities (single refit)
    y_train: np.ndarray
    y_test: np.ndarray
    fit_seconds: dict[str, float]
    predict_seconds: dict[str, float]
    refit_seconds: dict[str, float]
    cost_ms_per_1k: dict[str, float]
    cost_measurement: dict = field(default_factory=dict)
    n_fits: int = 0


def measure_inference_cost(model: Pipeline, X_batch: pd.DataFrame, *, repeats: int = 5) -> dict:
    """Median wall-clock of ``repeats`` full predict_proba calls (preprocessing
    included) on a fixed batch, in ms per 1k predictions."""
    n = len(X_batch)
    times = []
    model.predict_proba(X_batch)  # warm-up (JIT/caches)
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(X_batch)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return {"ms_per_1k": float(1000.0 * np.median(times) / n * 1000.0), "n_batch": int(n),
            "repeats": int(repeats), "all_ms_per_1k": (1000.0 * times / n * 1000.0).round(4).tolist()}


def run_replication(
    X: pd.DataFrame,
    y: pd.Series,
    spec: DatasetSpec,
    *,
    rep: int,
    base_seed: int = 20260904,
    n_splits: int = 5,
    holdout_size: float = 0.2,
    cost_batch_rows: int = 10_000,
    cost_repeats: int = 5,
    model_names: list[str] | None = None,
    log=print,
) -> ReplicationOOF:
    """One outer replication: fresh stratified holdout (seed = base_seed + rep),
    5-fold stratified OOF on the training side, single refit for the holdout,
    and direct inference-cost measurement on a fixed holdout batch."""
    names = model_names or MODEL_NAMES
    seed = base_seed + rep
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=holdout_size, stratify=y, random_state=seed)
    X_tr = X_tr.reset_index(drop=True); X_te = X_te.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True).to_numpy(); y_te = y_te.reset_index(drop=True).to_numpy()
    n = len(y_tr)
    M = len(names)
    P = np.full((n, M), np.nan, dtype=np.float32)
    fold_ids = np.full(n, -1, dtype=np.int16)
    fit_s = dict.fromkeys(names, 0.0); pred_s = dict.fromkeys(names, 0.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_fits = 0
    for f, (tr, te) in enumerate(skf.split(X_tr, y_tr)):
        fold_ids[te] = f
        for j, name in enumerate(names):
            m = make_model(name, spec, random_state=seed)
            t0 = time.perf_counter(); m.fit(X_tr.iloc[tr], y_tr[tr]); t1 = time.perf_counter()
            P[te, j] = clip_probs(m.predict_proba(X_tr.iloc[te])[:, 1]); t2 = time.perf_counter()
            fit_s[name] += t1 - t0; pred_s[name] += t2 - t1; n_fits += 1
        log(f"    fold {f + 1}/{n_splits} done")
    if np.isnan(P).any():
        raise RuntimeError("OOF matrix has unfilled entries")
    Q = np.empty((len(y_te), M), dtype=np.float32)
    refit_s = {}; costs = {}; cost_meas = {}
    rng = np.random.default_rng(seed)
    batch_idx = rng.choice(len(X_te), size=min(cost_batch_rows, len(X_te)), replace=False)
    X_batch = X_te.iloc[np.sort(batch_idx)]
    for j, name in enumerate(names):
        m = make_model(name, spec, random_state=seed)
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); refit_s[name] = time.perf_counter() - t0
        Q[:, j] = clip_probs(m.predict_proba(X_te)[:, 1]); n_fits += 1
        meas = measure_inference_cost(m, X_batch, repeats=cost_repeats)
        costs[name] = meas["ms_per_1k"]; cost_meas[name] = meas
        log(f"    refit {name}: {refit_s[name]:.1f}s, cost {costs[name]:.3f} ms/1k")
    return ReplicationOOF(rep=rep, seed=seed, model_names=list(names), P=P, fold_ids=fold_ids, Q=Q,
                          y_train=y_tr, y_test=y_te, fit_seconds=fit_s, predict_seconds=pred_s,
                          refit_seconds=refit_s, cost_ms_per_1k=costs, cost_measurement=cost_meas,
                          n_fits=n_fits)


__all__ = ["CLIP_EPS", "MODEL_NAMES", "ReplicationOOF", "clip_probs", "make_model", "measure_inference_cost",
           "model_params", "run_replication"]
