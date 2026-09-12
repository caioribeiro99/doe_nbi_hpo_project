#!/usr/bin/env python
"""Score the NSGA-II baseline against two reference definitions and run paired statistics.

Reference A (sample-core): non-dominated set of the independently sampled real-objective
reference only — Dirichlet samples, lattice, vertices, edges, the epsilon-constraint sweep,
the design runs and the single-objective references. It contains the search output of NO
optimizer under comparison, so it cannot favour any of them. This is the clean sensitivity
reference.

Reference B (common augmented union): non-dominated set of the sample core together with
EVERY candidate set, NSGA-II included. Indicators for all algorithms are recomputed against
this same union, so NSGA-II is neither uniquely advantaged nor uniquely disadvantaged.

Scoring reuses ``_quality_for_set`` from the frozen benchmark runner verbatim, so the
NSGA-II numbers are produced by exactly the code that produced the R = 30 numbers. The
frozen tables are not overwritten; everything is written under
``reports/pco213_postwork_benchmark/nsga2/``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mixens.pareto_tools as pt  # noqa: E402

# import the frozen runner as a module so the scoring helper is the *same* code
_spec = importlib.util.spec_from_file_location(
    "bench_runner", REPO / "scripts" / "pco213_run_postwork_benchmark.py")
_bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_runner"] = _bench
_spec.loader.exec_module(_bench)
_quality_for_set = _bench._quality_for_set
objectives = _bench.objectives

BENCH = REPO / "experiments" / "pco213_postwork_benchmark"
NSGA = REPO / "experiments" / "pco213_postwork_nsga2"
OUT = REPO / "reports" / "pco213_postwork_benchmark" / "nsga2"
DATASETS = ["santander", "bnp", "porto", "uci_credit"]
NAMES = ["lr", "gnb", "knn", "rf", "xgb"]
WCOLS = [f"w_{n}" for n in NAMES]
BASE_SEED = 20260904
OLD_SETS = ["nbi_A", "nbi_B", "nbi_C", "ws_random_scalarization",
            "random_dirichlet_budget", "design_runs", "single_objective_refs"]


def load_candidate_sets(d: Path) -> dict[str, pd.DataFrame]:
    """Rebuild the frozen candidate sets exactly as the benchmark's quality stage does."""
    sets: dict[str, pd.DataFrame] = {}
    for v in ("A", "B", "C"):
        sets[f"nbi_{v}"] = pd.read_csv(d / f"nbi_{v}_candidates.csv")
    for key in ("ws_random_scalarization", "random_dirichlet_budget", "design_runs"):
        sets[key] = pd.read_csv(d / f"comparator_{key}.csv")
    refs = json.loads((d / "references.json").read_text())
    sets["single_objective_refs"] = pd.DataFrame(
        [{**{c: v["w"][j] for j, c in enumerate(WCOLS)}, "roc_auc": v["oof_roc_auc"],
          "log_loss": v["oof_log_loss"], "cost_weighted": v["cost_weighted"],
          "cost_support": v["cost_support"], "method": k}
         for k, v in refs["references"].items() if "w" in v])
    return sets


def load_nsga2(ds: str, rep: int) -> pd.DataFrame | None:
    """NSGA-II final non-dominated set as a candidate-set frame with real objectives."""
    f = NSGA / ds / f"rep_{rep:02d}" / "nsga2_population.npz"
    if not f.exists():
        return None
    z = np.load(f)
    W, F = z["W_nd"], z["F_nd"]
    costs = np.array([float(json.loads((BENCH / ds / f"rep_{rep:02d}" / "oof_meta.json")
                                       .read_text())["cost_ms_per_1k"][n]) for n in NAMES])
    df = pd.DataFrame(W, columns=WCOLS)
    df["roc_auc"] = -F[:, 0]
    df["log_loss"] = F[:, 1]
    df["cost_weighted"] = F[:, 2]
    # support cost is scored POST HOC only; NSGA-II never optimized it
    df["cost_support"] = [(costs * (w > 1e-3)).sum() for w in W]
    df["success"] = True
    return df


def score_replication(ds: str, rep: int) -> list[dict]:
    d = BENCH / ds / f"rep_{rep:02d}"
    ns = load_nsga2(ds, rep)
    if ns is None:
        return []
    sets = load_candidate_sets(d)
    sets["nsga2"] = ns
    ref = np.load(d / "reference_sample.npz", allow_pickle=False)
    rows = []
    rng = np.random.default_rng(BASE_SEED + rep + 13000)
    for cost_col, tag in (("cost_weighted", "weighted"), ("cost_support", "support")):
        F_sample = np.column_stack([-ref["roc_auc"].astype(float),
                                    ref["log_loss"].astype(float), ref[cost_col].astype(float)])
        # ---- Reference A: sample core only, no optimizer output -------------------
        mA = pt.fast_pareto_mask(F_sample)
        FA = F_sample[mA]
        # ---- Reference B: sample core union every candidate set, NSGA-II included --
        F_all = np.vstack([F_sample] + [objectives(df, cost_col) for df in sets.values()])
        mB = pt.fast_pareto_mask(F_all)
        FB = F_all[mB]
        for ref_name, F_front, F_pool in (("sample_core", FA, F_sample), ("augmented_union", FB, F_all)):
            lo, hi = F_front.min(axis=0), F_front.max(axis=0)
            for key, df in sets.items():
                F = objectives(df, cost_col)
                ok = df["success"].to_numpy(bool) if "success" in df.columns else np.ones(len(df), bool)
                q = _quality_for_set(F, ok, F_front, lo, hi, F_pool, rng)
                rows.append({"dataset": ds, "rep": rep, "cost": tag, "reference": ref_name,
                             "set": key, "n_reference_front": int(len(F_front)), **q})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for ds in DATASETS:
        for rep in range(30):
            s = NSGA / ds / f"rep_{rep:02d}" / "nsga2_summary.json"
            if s.exists():
                runs.append(json.loads(s.read_text()))
            rows.extend(score_replication(ds, rep))
    if not rows:
        print("no NSGA-II results found yet")
        return 1
    q = pd.DataFrame(rows)
    q.to_csv(OUT / "nsga2_pareto_quality.csv", index=False)
    r = pd.DataFrame(runs)
    r.to_csv(OUT / "nsga2_runs.csv", index=False)
    print(f"scored {q.dataset.nunique()} datasets x {q.rep.nunique()} replications "
          f"-> {len(q)} rows; {len(r)} runs")
    print("\nbudget matching (actual / target):")
    print(f"  min {r.budget_ratio_actual_over_target.min():.5f}  "
          f"median {r.budget_ratio_actual_over_target.median():.5f}  "
          f"max {r.budget_ratio_actual_over_target.max():.5f}")
    print(f"  total NSGA-II real evaluations: {r.actual_evals.sum():,} "
          f"(target {r.target_evals.sum():,})")
    print(f"  total NSGA-II wall clock: {r.seconds.sum()/3600:.2f} h (sum over runs, run in parallel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
