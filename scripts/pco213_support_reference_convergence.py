#!/usr/bin/env python
"""Convergence of the empirical reference on the support-cost front, from frozen samples.

Provenance label: post hoc (coauthor review R2)

The benchmark checked the convergence of its sampled reference on the weighted-cost
front only: an independent sample of 2 x 10^4 compositions had to displace at most 5%
of that front, and a prefix diagnostic tracked how much of a smaller prefix's front a
larger one displaces. The support-cost front was never checked, and the independent
check sample was not stored. This script computes the corresponding diagnostics for
the support-cost front from the stored reference samples alone, with no new
composition evaluated, and the same diagnostics for the weighted-cost front under the
same definitions, so that the two can be read side by side.

Definition (fixed before the first run):
  * Data: reference_sample.npz of every dataset x replication. The first 564 rows are
    the structured blocks; the next 100,000 are the first-round random part in its
    stored (shuffled) arrival order; rows after that are the extension rounds, which
    the weighted check triggered and which are not shuffled. Objectives as in the
    runner: (-ROC-AUC, log-loss, cost), with cost_weighted or cost_support, from the
    stored float32 values cast to float64.
  * Prefix diagnostic: the frozen pareto_tools.front_convergence on the first-round
    random part with fractions (0.1, 0.25, 0.5, 0.75, 1.0), as the runner computed it
    for the weighted front; reported is the displaced fraction at the last step.
  * Held-out-block check (the stored substitute for the unstored independent sample):
    reference = the structured blocks plus the first 80,000 first-round random rows;
    check = the remaining 20,000 first-round random rows, independent draws from the
    same shuffled mixture. Reported is the fraction of the reference's front that the
    check block displaces. The reference here is smaller than the one actually used
    (it omits 20,000 random rows and the extension rounds), so this reading is, if
    anything, pessimistic.
  * Reading: met when the displaced fraction is at most 0.05, the frozen tolerance of
    the benchmark's weighted check. No new reference is built and nothing is rescored.

Inputs (frozen, read-only):
  experiments/pco213_postwork_benchmark/<dataset>/rep_XX/reference_sample.npz
  experiments/pco213_postwork_benchmark/<dataset>/rep_XX/reference_diagnostics.json
Outputs:
  reports/pco213_postwork_benchmark/statistics/support_reference_convergence_r30.csv
  reports/pco213_postwork_benchmark/statistics/support_reference_convergence_r30.json
  reports/pco213_postwork_benchmark/statistics/support_reference_convergence_r30_summary.csv

Reads stored outputs only; runs no learner, optimizer, surrogate fit or composition
evaluation.

Usage:
    python scripts/pco213_support_reference_convergence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mixens import pareto_tools as pt  # noqa: E402

RAW = REPO / "experiments" / "pco213_postwork_benchmark"
OUT = REPO / "reports" / "pco213_postwork_benchmark" / "statistics"
DATASETS = ("santander", "bnp", "porto", "uci_credit")
REPS = 30
N_STRUCT, N_FIRST, N_CHECK = 564, 100_000, 20_000
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
TOL = 0.05


def objectives(z, cost: str) -> np.ndarray:
    return np.column_stack([-z["roc_auc"].astype(np.float64), z["log_loss"].astype(np.float64),
                            z[cost].astype(np.float64)])


def displaced_by(front: np.ndarray, check: np.ndarray) -> float:
    m = pt.fast_pareto_mask(np.vstack([front, check]))
    return float(1.0 - m[: front.shape[0]].mean())


def one(ds: str, rep: int) -> dict:
    d = RAW / ds / f"rep_{rep:02d}"
    z = np.load(d / "reference_sample.npz", allow_pickle=False)
    src = z["source"]
    if (np.isin(src[:N_STRUCT], ["dir1", "dir03"]).any()
            or not np.isin(src[N_STRUCT:N_STRUCT + N_FIRST], ["dir1", "dir03"]).all()):
        raise SystemExit(f"{ds} rep {rep}: unexpected sample layout")
    diag = json.loads((d / "reference_diagnostics.json").read_text())
    out = {"dataset": ds, "replication": rep, "n_points": int(len(src)),
           "extension_rounds": int(diag["rounds"]) - 1,
           "recorded_weighted_independent_check": diag["independent_check"]["front_displaced_fraction"]}
    for space, cost in (("weighted", "cost_weighted"), ("support", "cost_support")):
        F = objectives(z, cost)
        rand = F[N_STRUCT:N_STRUCT + N_FIRST]
        conv = pt.front_convergence(rand, fractions=FRACTIONS)
        ref = np.vstack([F[:N_STRUCT], rand[: N_FIRST - N_CHECK]])
        front = ref[pt.fast_pareto_mask(ref)]
        out[f"{space}_prefix_last_displaced"] = conv[-1]["prev_front_displaced_fraction"]
        out[f"{space}_heldout_displaced"] = displaced_by(front, rand[N_FIRST - N_CHECK:])
        out[f"{space}_heldout_reference_front_size"] = int(front.shape[0])
        out[f"{space}_stored_front_size"] = int(np.asarray(z[f"nd_{space}"]).sum())
    out["recorded_weighted_prefix_last_displaced"] = (
        diag["prefix_convergence_random_part"][-1]["prev_front_displaced_fraction"])
    return out


def main() -> None:
    rows = [one(ds, r) for ds in DATASETS for r in range(REPS)]
    df = pd.DataFrame(rows)
    summary = []
    for ds, g in df.groupby("dataset", sort=False):
        summary.append({
            "dataset": ds,
            **{f"{s}_heldout_met": int((g[f"{s}_heldout_displaced"] <= TOL).sum())
               for s in ("weighted", "support")},
            **{f"{s}_heldout_median": round(float(g[f"{s}_heldout_displaced"].median()), 4)
               for s in ("weighted", "support")},
            **{f"{s}_heldout_max": round(float(g[f"{s}_heldout_displaced"].max()), 4)
               for s in ("weighted", "support")},
            **{f"{s}_prefix_median": round(float(g[f"{s}_prefix_last_displaced"].median()), 4)
               for s in ("weighted", "support")},
            **{f"{s}_prefix_max": round(float(g[f"{s}_prefix_last_displaced"].max()), 4)
               for s in ("weighted", "support")},
            "support_stored_front_size_median": float(g["support_stored_front_size"].median()),
            "weighted_stored_front_size_median": float(g["weighted_stored_front_size"].median()),
            "recorded_weighted_independent_check_met": int(
                (g["recorded_weighted_independent_check"] <= TOL).sum()),
            "weighted_prefix_vs_recorded_max_abs_diff": round(float(
                (g["weighted_prefix_last_displaced"] - g["recorded_weighted_prefix_last_displaced"])
                .abs().max()), 4),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "support_reference_convergence_r30.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "support_reference_convergence_r30_summary.csv", index=False)
    (OUT / "support_reference_convergence_r30.json").write_text(json.dumps({
        "provenance": "post hoc (coauthor review R2); frozen reference samples only",
        "definition": {"structured_rows": N_STRUCT, "first_round_random_rows": N_FIRST,
                       "heldout_check_rows": N_CHECK, "prefix_fractions": FRACTIONS,
                       "tolerance": TOL},
        "summary": summary}, indent=1) + "\n")
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
