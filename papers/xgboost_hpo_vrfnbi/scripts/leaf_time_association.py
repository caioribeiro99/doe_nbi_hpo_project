#!/usr/bin/env python
"""Leaf count, the cost objective and measured time over the 120 units (Supplement S4.6).

protocol/OBJECTIVE_DEFINITIONS.md section 9 and protocol/COST_OBJECTIVE_CLAIM_BOUNDARY.md
declare that the campaign reports the association between the leaf-count objective and
measured time (Time_MeanFold, a secondary audit variable persisted on every evaluation)
over all 120 dataset-by-replication units. No analysis computed it. This script does,
from each unit's SQLite evaluation cache, which holds one row per unique physical fit:

  * per unit, the Spearman correlation of Time_MeanFold with raw Leaves_Mean and with
    f_cost, the frozen cost objective (frozen reference factor model, applied);
  * over every cross-validated evaluation of the unit (fold_id "all": design, external
    audit, anchor search, revalidated candidates and all direct baselines; the holdout
    fits are excluded), and over the unit's design rows alone (the 79 unique design
    configurations; the 10 replicated centre points of the 88-row design were fitted
    and timed once);
  * per dataset, the median, minimum and maximum of each per-unit correlation.

The unmatched NSGA-II run is outside the unit (runner budget registry; Supplement S7),
but its fits are stored in the replication-0 caches. They are excluded, so that all 120
units contribute the same stages; requests show that no other method asked for any of
them. With them, replication 0 sets the range end on three datasets; that variant is
recorded under "diagnostics".

Provenance label:
retrospective (declared in the frozen protocol: protocol/OBJECTIVE_DEFINITIONS.md section 9 and protocol/COST_OBJECTIVE_CLAIM_BOUNDARY.md; computed after the campaign from stored outputs)

The design-row subset (fits requested by the design stage) is checked against the
configurations of the unit's design.json. Campaign timings were taken with several
worker processes in parallel, so they carry load-dependent noise.

Inputs (read only), per dataset and replication rep_00 to rep_29 under
experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/: evaluations.sqlite (opened
read-only and immutable) and design.json; and
papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json.
Output: papers/xgboost_hpo_vrfnbi/analysis/leaf_time_association.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:  python papers/xgboost_hpo_vrfnbi/scripts/leaf_time_association.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.evaluation_cache import canonical_config  # noqa: E402
from doe_xgb.campaign.evaluator import PARAMS  # noqa: E402
from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402
from doe_xgb.campaign.runner import (DATASETS, N_REPLICATIONS,  # noqa: E402
                                     NSGA2_UNMATCHED_REPLICATION)

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis" / "leaf_time_association.json"
LABEL = ("retrospective (declared in the frozen protocol: protocol/OBJECTIVE_DEFINITIONS.md "
         "section 9 and protocol/COST_OBJECTIVE_CLAIM_BOUNDARY.md; computed after the "
         "campaign from stored outputs)")
CV_FOLD = "all"                      # the inner-resampling evaluation; "holdout" is not
DESIGN_METHOD = "design"             # the method key of the design stage
UNMATCHED_METHOD = "nsga2_unmatched"  # the method key of the run outside the unit


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def cfg_id(cfg: dict) -> str:
    return json.dumps(canonical_config({p: cfg[p] for p in PARAMS}), sort_keys=True)


def read_cache(path: Path) -> list[tuple[str, dict, dict, frozenset]]:
    """(fold_id, config, result, methods that requested it), one per physical fit."""
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        recs = con.execute("SELECT key, fold_id, config, result "
                           "FROM evaluations ORDER BY key").fetchall()
        by: dict[str, set] = {}
        for method, key in con.execute("SELECT method, key FROM requests"):
            by.setdefault(key, set()).add(method)
    finally:
        con.close()
    if any(k not in by for k, *_ in recs):
        raise SystemExit(f"{rel(path)}: a stored fit has no request in the ledger")
    return [(f, json.loads(c), json.loads(r), frozenset(by[k])) for k, f, c, r in recs]


def rho(a: np.ndarray, b: np.ndarray) -> float:
    v = float(spearmanr(a, b).statistic)
    if not np.isfinite(v):
        raise SystemExit("undefined Spearman correlation")
    return v


def span(x: list[float]) -> dict:
    return {"median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def main() -> int:
    datasets, diagnostics = {}, {}
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        per = {k: [] for k in ("all_L", "all_F", "des_L", "des_F")}
        n_cv, n_design, n_holdout, n_unmatched = [], [], 0, 0
        with_unmatched = {}
        for rep in range(N_REPLICATIONS):
            u = ROOT / ds / f"rep_{rep:02d}"
            recs = read_cache(u / "evaluations.sqlite")
            cv = [(c, r, ms) for f, c, r, ms in recs if f == CV_FOLD]
            n_holdout += len(recs) - len(cv)
            if any(UNMATCHED_METHOD in ms and len(ms) > 1 for _, _, ms in cv):
                raise SystemExit(f"{ds} rep {rep}: an unmatched-run fit was also requested "
                                 f"inside the unit, so excluding it would drop a unit fit")
            df = pd.DataFrame([r for _, r, _ in cv])
            T = df["Time_MeanFold"].to_numpy(float)
            L = df["Leaves_Mean"].to_numpy(float)
            Fc = fm.objectives(df)[:, 1]
            if not (np.isfinite(T).all() and np.isfinite(L).all() and np.isfinite(Fc).all()):
                raise SystemExit(f"{ds} rep {rep}: non-finite time, leaf count or f_cost")
            own = np.array([UNMATCHED_METHOD not in ms for _, _, ms in cv])
            if rep != NSGA2_UNMATCHED_REPLICATION and not own.all():
                raise SystemExit(f"{ds} rep {rep}: unmatched-run fits outside replication "
                                 f"{NSGA2_UNMATCHED_REPLICATION}")
            n_unmatched += int((~own).sum())
            des = np.array([DESIGN_METHOD in ms for _, _, ms in cv])
            stored = {cfg_id(r) for r in json.loads((u / "design.json").read_text())["rows"]}
            if {cfg_id(c) for (c, _, ms) in cv if DESIGN_METHOD in ms} != stored:
                raise SystemExit(f"{ds} rep {rep}: design rows in the cache differ from "
                                 f"design.json")
            per["all_L"].append(rho(L[own], T[own]))
            per["all_F"].append(rho(Fc[own], T[own]))
            per["des_L"].append(rho(L[des], T[des]))
            per["des_F"].append(rho(Fc[des], T[des]))
            n_cv.append(int(own.sum()))
            n_design.append(int(des.sum()))
            if rep == NSGA2_UNMATCHED_REPLICATION:
                with_unmatched = {"leaves_vs_time": rho(L, T), "f_cost_vs_time": rho(Fc, T)}
        datasets[ds] = {
            "n_units": len(n_cv),
            "n_evaluations": int(sum(n_cv)),
            "all_evaluations": {"leaves_vs_time": span(per["all_L"]),
                                "f_cost_vs_time": span(per["all_F"])},
            "design_rows": {"leaves_vs_time": span(per["des_L"]),
                            "f_cost_vs_time": span(per["des_F"])}}
        r0 = NSGA2_UNMATCHED_REPLICATION
        sens = {k: span([with_unmatched[k] if i == r0 else v for i, v in enumerate(per[p])])
                for k, p in (("leaves_vs_time", "all_L"), ("f_cost_vs_time", "all_F"))}
        diagnostics[ds] = {
            "evaluations_per_unit_min": int(min(n_cv)),
            "evaluations_per_unit_max": int(max(n_cv)),
            "design_evaluations_per_unit": sorted(set(n_design)),
            "n_design_evaluations": int(sum(n_design)),
            "holdout_evaluations_excluded": int(n_holdout),
            "unmatched_run_evaluations_excluded": int(n_unmatched),
            "variant_with_unmatched_run_fits_in_replication_0": {
                "replication_0": with_unmatched, "all_evaluations": sens}}

    unit = "experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/"
    out = {
        "script": rel(Path(__file__).resolve()),
        "label": LABEL,
        "inputs": [unit + "evaluations.sqlite", unit + "design.json",
                   "papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json"],
        "definitions": {
            "per-unit statistic": ("Spearman correlation (average ranks for ties) of "
                                   "Time_MeanFold with raw Leaves_Mean (leaves_vs_time) "
                                   "and with f_cost (f_cost_vs_time)"),
            "f_cost": "the frozen cost objective: frozen reference factor model, applied",
            "all_evaluations": ("every unique cross-validated evaluation in the unit's "
                                "cache (fold_id 'all'), one row per physical fit; "
                                "holdout fits excluded, and the unmatched NSGA-II run's "
                                "fits (outside the unit; replication 0 only) excluded"),
            "design_rows": ("the unit's unique design configurations, requested by the "
                            "design stage (79: the 10 replicated centre points of the "
                            "88-row design are one fit)"),
            "median, min, max": "over the dataset's units",
            "n_evaluations": "all_evaluations rows summed over the dataset's units"},
        "diagnostics": diagnostics,
        "datasets": datasets,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    s = lambda d: f"{d['median']:.3f} ({d['min']:.3f} to {d['max']:.3f})"  # noqa: E731
    print("PER-UNIT SPEARMAN WITH Time_MeanFold: median (min to max)\n")
    print(f"  {'dataset':15s} {'units':>5s} {'all: leaves':>24s} {'all: f_cost':>24s} "
          f"{'design: leaves':>24s} {'design: f_cost':>24s}")
    for ds in DATASETS:
        v = datasets[ds]
        print(f"  {ds:15s} {v['n_units']:5d} {s(v['all_evaluations']['leaves_vs_time']):>24s} "
              f"{s(v['all_evaluations']['f_cost_vs_time']):>24s} "
              f"{s(v['design_rows']['leaves_vs_time']):>24s} "
              f"{s(v['design_rows']['f_cost_vs_time']):>24s}")
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
