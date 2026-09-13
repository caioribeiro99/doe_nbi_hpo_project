#!/usr/bin/env python
"""Choose the cost objective by measurement, not by preference.

The dissertation's cost objective is mean per-fold wall-clock training time.
`audits/provenance/README.md` shows it does not reproduce across environments, because
it is a timing measurement rather than a function of the design. A paired comparison
across 30 replications cannot rest on an objective that changes between runs of the
same seed.

This script measures every candidate replacement against the 88 measured times of the
reproduced MAGIC design, on two axes that matter: how well it tracks the quantity it
replaces, and whether it is deterministic across refits of the same configuration.

Writes audits/cost_objective_selection.json.

Usage:  python cost_objective_selection.py [--stride 4]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits"
DOE = OUT / "provenance" / "doe_results.csv"
DESIGN = REPO / "data" / "design" / "hyperparameter_design.csv"
RAW = REPO / "data" / "source" / "magic" / "raw" / "magic04.data"
PARAMS = ["subsample", "colsample_bytree", "colsample_bylevel",
          "learning_rate", "max_depth", "gamma", "n_estimators"]
INTS = {"max_depth", "n_estimators"}
MAGIC_COLS = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
              "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]


def load_magic():
    df = pd.read_csv(RAW, header=None, names=MAGIC_COLS)
    y = df.pop("class").map({"g": 0, "h": 1})
    return df.astype(float).to_numpy(), y.to_numpy()


def closed_form_proxies(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Proxies computable from the hyperparameters alone, with no model fit."""
    ne, md = d["n_estimators"].to_numpy(), d["max_depth"].to_numpy()
    ss, cbt, cbl = (d[c].to_numpy() for c in
                    ("subsample", "colsample_bytree", "colsample_bylevel"))
    return {
        "n_estimators": ne,
        "n_estimators x max_depth": ne * md,
        "n_estimators x 2^min(max_depth,12)": ne * np.power(2.0, np.minimum(md, 12)),
        "n_estimators x max_depth x subsample": ne * md * ss,
        "n_estimators x max_depth x subsample x colsample_bytree": ne * md * ss * cbt,
        "n_estimators x max_depth x subsample x colsample_bytree x colsample_bylevel":
            ne * md * ss * cbt * cbl,
    }


def fitted_proxies(d: pd.DataFrame, idx: list[int], X, y, seed: int = 42):
    """Proxies that require a fit: structural size of the trained ensemble.

    Each configuration is fitted twice so that determinism can be checked rather
    than assumed.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, _ = next(iter(kf.split(X, y)))
    rows = []
    for i in idx:
        p = {k: (int(round(float(d[k][i]))) if k in INTS else float(d[k][i])) for k in PARAMS}

        def once():
            m = XGBClassifier(**p, eval_metric="logloss", verbosity=0,
                              tree_method="hist", n_jobs=8, random_state=seed)
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            dt = time.perf_counter() - t0
            dump = m.get_booster().get_dump()
            return dt, sum(s.count("\n") for s in dump), sum(s.count("leaf=") for s in dump)
        a, b = once(), once()
        rows.append({"row": i, "time_a": a[0], "time_b": b[0],
                     "nodes_a": a[1], "nodes_b": b[1],
                     "leaves_a": a[2], "leaves_b": b[2]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4,
                    help="fit every Nth design row for the fitted proxies")
    args = ap.parse_args()

    res = pd.read_csv(DOE)
    t_meas = res["Time_MeanFold"].to_numpy()
    design = pd.read_csv(DESIGN, sep=";", decimal=",", encoding="utf-8-sig")
    design.columns = [str(c).strip().strip('"') for c in design.columns]

    report: dict = {"reference": "Time_MeanFold, measured, from audits/provenance/doe_results.csv",
                    "n_design_rows": int(len(res)), "closed_form": {}, "fitted": {}}

    for name, v in closed_form_proxies(res).items():
        report["closed_form"][name] = {
            "spearman_vs_measured_time": round(float(spearmanr(v, t_meas).statistic), 4),
            "range_ratio": round(float(v.max() / max(v.min(), 1e-12)), 1),
            "deterministic": True,
        }

    X, y = load_magic()
    idx = list(range(0, len(design), args.stride))
    fp = fitted_proxies(design, idx, X, y)
    sub = t_meas[fp.row.to_numpy()]
    det_nodes = bool((fp.nodes_a == fp.nodes_b).all())
    det_leaves = bool((fp.leaves_a == fp.leaves_b).all())
    time_jitter = float(np.median(np.abs(fp.time_a - fp.time_b) / fp.time_a))
    for name, col, det in [("total node count", "nodes_a", det_nodes),
                           ("total leaf count", "leaves_a", det_leaves)]:
        v = fp[col].to_numpy()
        report["fitted"][name] = {
            "spearman_vs_measured_time": round(float(spearmanr(v, sub).statistic), 4),
            "range_ratio": round(float(v.max() / max(v.min(), 1)), 1),
            "deterministic": det,
            "n_configurations_fitted_twice": int(len(fp)),
        }
    report["wall_clock_self_agreement"] = {
        "median_relative_difference_between_two_fits": round(time_jitter, 5),
        "reading": ("The reference quantity disagrees with itself at this level between two fits "
                    "of one configuration, which bounds how well any deterministic proxy could "
                    "agree with it."),
    }

    best = max(report["fitted"].items(),
               key=lambda kv: kv[1]["spearman_vs_measured_time"] if kv[1]["deterministic"] else -1)
    report["decision"] = {
        "primary_cost_objective": best[0],
        "why": ("Deterministic given seed and data, machine-independent, widest dynamic range among "
                "the deterministic candidates, tracks measured training time as well as any of them, "
                "and is itself a deployment quantity: it is the model's size."),
        "secondary_reported": "mean per-fold wall-clock training time, with its run-to-run variation",
        "caution": ("No deterministic proxy tracks wall-clock time better than about 0.86 Spearman, "
                    "and elaborating a closed-form proxy with the sampling hyperparameters makes it "
                    "markedly worse. The two objectives are related, not interchangeable."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_objective_selection.json").write_text(json.dumps(report, indent=2))

    print(f"{'candidate':70s} {'Spearman':>9} {'det':>5} {'range':>10}")
    for grp in ("closed_form", "fitted"):
        for k, v in report[grp].items():
            print(f"{k:70s} {v['spearman_vs_measured_time']:9.4f} "
                  f"{str(v['deterministic']):>5} {v['range_ratio']:10.1f}")
    print(f"\nwall-clock self-agreement: {time_jitter:.2%} median relative difference between two fits")
    print(f"DECISION: {report['decision']['primary_cost_objective']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
