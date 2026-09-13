#!/usr/bin/env python
"""Pilot Stage A: screen the candidate dataset panel.

`protocol/dataset_selection.md` sets four measurements that a dataset must pass to
stay in the panel, and `protocol/EXPERIMENT_PROTOCOL.md` §12 makes Stage A of the
pilot responsible for taking them on *every* candidate. This script is Stage A.

Per dataset, on one partition:

  * evaluate the 88 design rows on the real objectives;
  * evaluate 100 held-out Latin-hypercube points, which are both the surrogate
    gate's external set and the screening's adequacy measurement;
  * run the decided factor stage of protocol §6.3;
  * take the four screening measurements and the measured per-evaluation cost.

Nothing here runs an arm. Stage A exists to decide the panel and to replace the
budget projection with a measurement.

Usage:
    python pilot_stage_a_screening.py [--datasets magic spambase adult bank_marketing]
                                      [--seed 20260913] [--n-valid 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
DESIGN = REPO / "data" / "design" / "hyperparameter_design.csv"

PARAMS = ["subsample", "colsample_bytree", "colsample_bylevel",
          "learning_rate", "max_depth", "gamma", "n_estimators"]
INTS = {"max_depth", "n_estimators"}
BOUNDS = {"subsample": (0.05, 1.0), "colsample_bytree": (0.05, 1.0),
          "colsample_bylevel": (0.05, 1.0), "learning_rate": (0.01, 0.30),
          "max_depth": (3, 18), "gamma": (0.05, 5.0), "n_estimators": (50, 700)}

# Protocol section 6.2. "minimize" says which direction the canonicalization flips.
RESPONSES = {
    "Accuracy_Mean":    {"minimize": False, "role": "quality"},
    "Precision_Mean":   {"minimize": False, "role": "quality"},
    "Recall_Mean":      {"minimize": False, "role": "quality"},
    "Specificity_Mean": {"minimize": False, "role": "quality"},
    "RocAuc_Mean":      {"minimize": False, "role": "quality"},
    "LogLoss_Mean":     {"minimize": True,  "role": "quality"},
    "Leaves_Mean":      {"minimize": True,  "role": "cost"},
}


# --------------------------------------------------------------------------- data

def prepare(dataset_id: str):
    from doe_xgb.datasets.loaders import load
    d = load(dataset_id)
    X = d.X.copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False)
    return X.astype(float).to_numpy(), np.asarray(d.y, dtype=int), len(cat)


# ------------------------------------------------------------------- evaluation

def evaluate(p: dict, X, y, kf, seed: int) -> dict:
    """One configuration, five folds, every response of protocol section 6.2."""
    acc, pre, rec, spe, auc, ll, leaves, times = ([] for _ in range(8))
    for tr, va in kf.split(X, y):
        m = XGBClassifier(**p, eval_metric="logloss", verbosity=0,
                          tree_method="hist", n_jobs=8, random_state=seed)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[va])[:, 1]
        times.append(time.perf_counter() - t0)
        pred = (prob >= 0.5).astype(int)
        tn = int(((pred == 0) & (y[va] == 0)).sum())
        fp = int(((pred == 1) & (y[va] == 0)).sum())
        acc.append(accuracy_score(y[va], pred))
        pre.append(precision_score(y[va], pred, zero_division=0))
        rec.append(recall_score(y[va], pred, zero_division=0))
        spe.append(tn / (tn + fp) if (tn + fp) else 0.0)
        auc.append(roc_auc_score(y[va], prob))
        ll.append(log_loss(y[va], np.clip(prob, 1e-7, 1 - 1e-7)))
        leaves.append(sum(s.count("leaf=") for s in m.get_booster().get_dump()))
    return {"Accuracy_Mean": float(np.mean(acc)), "Precision_Mean": float(np.mean(pre)),
            "Recall_Mean": float(np.mean(rec)), "Specificity_Mean": float(np.mean(spe)),
            "RocAuc_Mean": float(np.mean(auc)), "LogLoss_Mean": float(np.mean(ll)),
            "Leaves_Mean": float(np.mean(leaves)), "Time_MeanFold": float(np.mean(times))}


def cast(row) -> dict:
    return {k: (int(round(float(row[k]))) if k in INTS else float(row[k])) for k in PARAMS}


def lhs_points(n: int, seed: int) -> pd.DataFrame:
    """Held-out compositions: protocol section 7's external set, one design, one seed."""
    s = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n)
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])
    return pd.DataFrame(qmc.scale(s, lo, hi), columns=PARAMS)


# ---------------------------------------------------------------- factor stage

def varimax(L: np.ndarray, tol=1e-7, it=200):
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(it):
        d_old = d
        Lam = L @ R
        u, s, vh = np.linalg.svd(L.T @ (Lam**3 - Lam @ np.diag((Lam**2).sum(0)) / p))
        R = u @ vh
        d = float(s.sum())
        if d_old and d / d_old < 1 + tol:
            break
    return L @ R, R


def factor_stage(df: pd.DataFrame, k: int = 3) -> dict:
    """Protocol section 6.3: canonicalize to minimization, PCA, Varimax on SCALED
    loadings, quality composite weighted by explained-variance share."""
    cols = list(RESPONSES)
    M = np.column_stack([df[c].to_numpy() * (1.0 if RESPONSES[c]["minimize"] else -1.0)
                         for c in cols])
    Z = (M - M.mean(0)) / np.where(M.std(0, ddof=1) == 0, 1.0, M.std(0, ddof=1))
    pca = PCA(n_components=k, random_state=0).fit(Z)
    lam = pca.explained_variance_
    loadings = pca.components_.T * np.sqrt(lam)          # scaled, not eigenvectors
    rot, R = varimax(loadings)
    scores = pca.transform(Z) @ R
    cost_idx = int(np.argmax(np.abs(rot[cols.index("Leaves_Mean")])))
    q_idx = [j for j in range(k) if j != cost_idx]
    share = lam / lam.sum()
    w = share[q_idx] / share[q_idx].sum()                 # variance weighting
    zs = (scores - scores.mean(0)) / scores.std(0, ddof=1)
    return {"quality": zs[:, q_idx] @ w,
            "quality_equal": zs[:, q_idx].mean(1),        # the pre-registered sensitivity
            "cost": zs[:, cost_idx],
            "loadings": pd.DataFrame(rot, index=cols,
                                     columns=[f"F{j+1}" for j in range(k)]),
            "explained_variance_share": share.tolist(),
            "cost_factor": cost_idx + 1,
            "quality_weights": w.tolist()}


# -------------------------------------------------------------------- screening

def nondominated(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Indices minimizing both canonicalized objectives."""
    pts = np.column_stack([q, c])
    keep = []
    for i, a in enumerate(pts):
        if not np.any(np.all(pts <= a, axis=1) & np.any(pts < a, axis=1)):
            keep.append(i)
    return np.array(keep)


def front_curvature(q: np.ndarray, c: np.ndarray) -> float:
    """How far the front departs from the chord between its extremes, normalized.

    Zero means a straight front, where weighted sum and NBI agree by construction.
    """
    idx = nondominated(q, c)
    if len(idx) < 3:
        return 0.0
    P = np.column_stack([q[idx], c[idx]])
    rng_ = np.ptp(P, axis=0)     # numpy 2 removed ndarray.ptp
    P = (P - P.min(0)) / np.where(rng_ == 0, 1.0, rng_)
    P = P[np.argsort(P[:, 0])]
    a, b = P[0], P[-1]
    d = b - a
    n = np.linalg.norm(d)
    if n == 0:
        return 0.0
    # perpendicular distance from the chord; written out because numpy 2 deprecated
    # the 2-D form of np.cross
    rel = P - a
    dev = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / n
    return float(dev.max())


def external_r2(fit_df, fit_y, val_df, val_y) -> tuple[float, float]:
    """Quadratic surface in coded units, fitted on the design, scored on the held-out set."""
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])

    def code(d):
        return 2 * (d[PARAMS].to_numpy(dtype=float) - lo) / (hi - lo) - 1

    def basis(Xc):
        cols = [np.ones(len(Xc))] + [Xc[:, i] for i in range(Xc.shape[1])]
        cols += [Xc[:, i] ** 2 for i in range(Xc.shape[1])]
        cols += [Xc[:, i] * Xc[:, j]
                 for i in range(Xc.shape[1]) for j in range(i + 1, Xc.shape[1])]
        return np.column_stack(cols)

    A, b = basis(code(fit_df)), np.asarray(fit_y, float)
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = basis(code(val_df)) @ beta
    yv = np.asarray(val_y, float)
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return r2, float(spearmanr(pred, yv).statistic)


# -------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["magic", "spambase", "adult", "bank_marketing"])
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--n-valid", type=int, default=100)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    design = pd.read_csv(DESIGN, sep=";", decimal=",", encoding="utf-8-sig")
    design.columns = [str(c).strip().strip('"') for c in design.columns]
    valid = lhs_points(args.n_valid, args.seed)

    report: dict = {"seed": args.seed, "n_design": int(len(design)),
                    "n_valid": args.n_valid, "datasets": {}}

    for ds in args.datasets:
        t0 = time.perf_counter()
        X, y, n_cat = prepare(ds)
        # one outer partition; the holdout is untouched here and reserved for the campaign
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=args.seed)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        rows = [evaluate(cast(design.iloc[i]), Xtr, ytr, kf, args.seed)
                for i in range(len(design))]
        d_df = pd.concat([design[PARAMS].reset_index(drop=True),
                          pd.DataFrame(rows)], axis=1)
        v_rows = [evaluate(cast(valid.iloc[i]), Xtr, ytr, kf, args.seed)
                  for i in range(len(valid))]
        v_df = pd.concat([valid.reset_index(drop=True), pd.DataFrame(v_rows)], axis=1)
        elapsed = time.perf_counter() - t0
        n_eval = len(d_df) + len(v_df)

        d_df.to_csv(OUT / f"{ds}_design.csv", index=False)
        v_df.to_csv(OUT / f"{ds}_validation.csv", index=False)

        fs = factor_stage(d_df)
        fs_v = factor_stage(v_df)
        fs["loadings"].to_csv(OUT / f"{ds}_loadings.csv")

        rho_conf = float(spearmanr(fs["quality"], fs["cost"]).statistic)
        curv = front_curvature(fs["quality"], fs["cost"])
        r2q, sq = external_r2(d_df, fs["quality"], v_df, fs_v["quality"])
        r2c, sc = external_r2(d_df, fs["cost"], v_df, fs_v["cost"])
        cost_ratio = float(d_df.Leaves_Mean.max() / max(d_df.Leaves_Mean.min(), 1.0))
        rho_weighting = float(spearmanr(fs["quality"], fs["quality_equal"]).statistic)

        report["datasets"][ds] = {
            "rows": int(X.shape[0]), "columns_after_encoding": int(X.shape[1]),
            "categorical_columns_one_hot_encoded": n_cat,
            "prevalence": float(y.mean()),
            "screening": {
                "objective_conflict_spearman": round(rho_conf, 4),
                "front_curvature": round(curv, 4),
                "external_r2_quality": round(r2q, 4),
                "external_spearman_quality": round(sq, 4),
                "external_r2_cost": round(r2c, 4),
                "external_spearman_cost": round(sc, 4),
                "gate_pass_quality": bool(r2q >= 0.5 and sq >= 0.9),
                "gate_pass_cost": bool(r2c >= 0.5 and sc >= 0.9),
                "cost_range_ratio": round(cost_ratio, 1),
            },
            "factor_stage": {
                "explained_variance_share": [round(v, 4) for v in fs["explained_variance_share"]],
                "cost_factor": fs["cost_factor"],
                "quality_weights_variance": [round(v, 4) for v in fs["quality_weights"]],
                "spearman_variance_vs_equal_weighting": round(rho_weighting, 4),
            },
            "cost": {
                "evaluations": n_eval,
                "seconds_total": round(elapsed, 1),
                "seconds_per_evaluation": round(elapsed / n_eval, 3),
            },
        }
        s = report["datasets"][ds]
        print(f"{ds:16s} {n_eval} evals in {elapsed/60:5.1f} min "
              f"({s['cost']['seconds_per_evaluation']:.2f} s/eval) | "
              f"conflict {rho_conf:+.3f} curv {curv:.3f} "
              f"R2q {r2q:+.3f} R2c {r2c:+.3f} costratio {cost_ratio:.0f}")

    (OUT / "stage_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'stage_a_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
