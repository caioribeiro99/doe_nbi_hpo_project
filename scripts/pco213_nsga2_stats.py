#!/usr/bin/env python
"""Paired statistics for the evaluation-matched NSGA-II baseline.

Primary comparison: NSGA-II versus NBI-C. Both optimize the same three real cached
out-of-fold objectives, both consume a matched number of real objective evaluations, so
the objective source is not confounded and the budget is equal.

The statistical philosophy is the frozen study's, and the helper functions are imported
from the frozen statistics script rather than reimplemented: paired by partition, the
dataset as the unit of generalization, nothing pooled across the 120 replications,
Nadeau-Bengio correction for the overlap between partitions (rho = 0.25), Holm within
each dataset's family, and effect consistency preferred over p-values.

Every comparison is reported twice, once against the sample-core reference and once
against the common augmented union, and under both cost definitions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "bench_stats", REPO / "scripts" / "pco213_postwork_benchmark_stats.py")
_st = importlib.util.module_from_spec(_spec)
sys.modules["bench_stats"] = _st
_spec.loader.exec_module(_st)

OUT = REPO / "reports" / "pco213_postwork_benchmark" / "nsga2"
DATASETS = ["santander", "bnp", "porto", "uci_credit"]
ENDPOINTS = {"igd_plus": "lower", "hv_ratio": "higher"}
# NSGA-II is the "new" method in every pair, so delta > 0 always favours NSGA-II
COMPARISONS = [("nsga2", "nbi_C"), ("nsga2", "nbi_B"), ("nsga2", "ws_random_scalarization")]
SECONDARY = ["gd_front", "spacing", "spacing_cv", "n_front", "joint_nondominated_fraction_front",
             "coverage_set_over_ref", "coverage_ref_over_set"]


def paired(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eff, tests = [], []
    for ref in sorted(q.reference.unique()):
        for cost in sorted(q.cost.unique()):
            sub = q[(q.reference == ref) & (q.cost == cost)]
            for ds in DATASETS:
                fam = []
                for new, old in COMPARISONS:
                    for ep, direction in ENDPOINTS.items():
                        a = sub[(sub.dataset == ds) & (sub.set == new)].set_index("rep")[ep]
                        b = sub[(sub.dataset == ds) & (sub.set == old)].set_index("rep")[ep]
                        common = a.index.intersection(b.index)
                        if len(common) == 0:
                            continue
                        a, b = a.loc[common], b.loc[common]
                        d = (a - b) if direction == "higher" else (b - a)
                        d = d.to_numpy(float)
                        d = d[np.isfinite(d)]
                        if len(d) == 0:
                            continue
                        row = {"reference": ref, "cost": cost, "dataset": ds,
                               "comparison": f"{new} vs {old}", "new": new, "ref_method": old,
                               "endpoint": ep, "n": len(d),
                               "sign_convention": "delta > 0 = NSGA-II better",
                               **_st.summarize_delta(d)}
                        eff.append(row)
                        fam.append(row)
                for row, p in zip(fam, _st.holm([r["p_nb"] for r in fam])):
                    tests.append({k: row[k] for k in ("reference", "cost", "dataset", "comparison",
                                                      "endpoint", "n", "mean", "median", "wins",
                                                      "ties", "losses", "rank_biserial",
                                                      "t_nb", "p_nb", "p_wilcoxon")}
                                 | {"p_nb_holm_family_dataset": p, "family_size": len(fam)})
    return pd.DataFrame(eff), pd.DataFrame(tests)


def levels(q: pd.DataFrame) -> pd.DataFrame:
    keep = ["igd_plus", "hv_ratio"] + SECONDARY
    g = (q.groupby(["reference", "cost", "dataset", "set"])[keep]
           .median().round(6).reset_index())
    return g


def budget_and_runtime() -> pd.DataFrame:
    r = pd.read_csv(OUT / "nsga2_runs.csv")
    nb = pd.read_csv(REPO / "reports" / "pco213_postwork_benchmark" / "tables" / "nbi_runs.csv")
    c = nb[nb.variant == "C"][["dataset", "rep", "n_real_objective_evals", "seconds"]]
    c = c.rename(columns={"n_real_objective_evals": "nbi_c_evals", "seconds": "nbi_c_seconds"})
    m = r.merge(c, on=["dataset", "rep"], how="left")
    m["eval_ratio_nsga2_over_nbi_c"] = m.actual_evals / m.nbi_c_evals
    m["time_ratio_nsga2_over_nbi_c"] = m.seconds / m.nbi_c_seconds
    out = m.groupby("dataset").agg(
        n=("rep", "count"),
        nsga2_evals_total=("actual_evals", "sum"), nbi_c_evals_total=("nbi_c_evals", "sum"),
        eval_ratio_min=("eval_ratio_nsga2_over_nbi_c", "min"),
        eval_ratio_median=("eval_ratio_nsga2_over_nbi_c", "median"),
        eval_ratio_max=("eval_ratio_nsga2_over_nbi_c", "max"),
        nsga2_seconds_mean=("seconds", "mean"), nbi_c_seconds_mean=("nbi_c_seconds", "mean"),
        time_ratio_median=("time_ratio_nsga2_over_nbi_c", "median"),
        n_nondominated_median=("n_nondominated", "median"),
    ).round(6).reset_index()
    return out


def main() -> int:
    q = pd.read_csv(OUT / "nsga2_pareto_quality.csv")
    eff, tests = paired(q)
    eff.to_csv(OUT / "nsga2_paired_effects.csv", index=False)
    tests.to_csv(OUT / "nsga2_paired_tests.csv", index=False)
    levels(q).to_csv(OUT / "nsga2_indicator_levels.csv", index=False)
    budget_and_runtime().to_csv(OUT / "nsga2_budget_runtime.csv", index=False)

    print("=== PRIMARY: NSGA-II vs NBI-C, weighted cost (delta > 0 = NSGA-II better)\n")
    for ref in ["sample_core", "augmented_union"]:
        print(f"--- reference: {ref}")
        s = eff[(eff.reference == ref) & (eff.cost == "weighted") &
                (eff.comparison == "nsga2 vs nbi_C")]
        for ds in DATASETS:
            for ep in ["igd_plus", "hv_ratio"]:
                r = s[(s.dataset == ds) & (s.endpoint == ep)]
                if r.empty:
                    continue
                r = r.iloc[0]
                t = tests[(tests.reference == ref) & (tests.cost == "weighted") &
                          (tests.dataset == ds) & (tests.endpoint == ep) &
                          (tests.comparison == "nsga2 vs nbi_C")]
                ph = t.iloc[0].p_nb_holm_family_dataset if not t.empty else float("nan")
                print(f"  {ds:11s} {ep:9s} mean {r['mean']:+.4f} [{r.ci95_mean_lo:+.4f}, "
                      f"{r.ci95_mean_hi:+.4f}] median {r['median']:+.4f} "
                      f"W/T/L {int(r.wins)}/{int(r.ties)}/{int(r.losses)} "
                      f"r_rb {r.rank_biserial:+.2f} Holm p {ph:.4f}")
        print()
    print("=== budget and runtime")
    print(budget_and_runtime().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
