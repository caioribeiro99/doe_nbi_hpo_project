#!/usr/bin/env python
"""RQ9 batch analysis: does the R = 10 sample (partitions 0-9) agree with the twenty
partitions added by the extension (10-29) and with the full R = 30 sample (0-29)?

Provenance label: post hoc

Only the nested level comparison (R = 10 estimate inside the R = 30 bootstrap interval)
was specified by the statistical analysis plan; everything computed here is a post hoc,
exploratory reading of the committed R = 30 tables:

  * per primary cell (4 datasets x 4 primary comparisons x 2 endpoints, weighted cost,
    sign convention of the frozen analysis: delta > 0 means the new, first-named method is
    better): median and mean paired effect and win/tie/loss counts in partitions 0-9,
    10-29 and 0-29; agreement of the sign of the median paired effect in the disjoint
    form (0-9 vs 10-29) and the nested form (0-9 vs 0-29 = R = 10 vs R = 30), with
    |median| <= tie_tol counted as sign 0; Holm significance at R = 30 taken from the
    committed tests; a two-sided Fisher exact test of the win counts in 0-9 against
    10-29 (wins vs non-wins, ties counted as non-wins; uncorrected);
  * per level cell (4 datasets x NBI-A/B/C x 2 endpoints, weighted cost): batch means,
    and whether the R = 10 estimate lies inside the percentile-bootstrap 95% interval of
    the mean of partitions 10-29 (the "new-20" level check) and inside the committed
    R = 30 interval (the nested check of tab:r10r30, reproduced as a consistency check).

Tie tolerance, R = 10 boundary, bootstrap size and seed, primary comparisons and
endpoint directions are read from the frozen statistics/analysis_config.json; the
bootstrap is the frozen boot_ci of scripts/pco213_postwork_benchmark_stats.py.
Statistical significance means Holm-adjusted Nadeau-Bengio p < 0.05, as in the paper.

Inputs (committed, read-only):
  reports/pco213_postwork_benchmark/tables/pareto_quality.csv
  reports/pco213_postwork_benchmark/statistics/paired_primary_tests.csv
  reports/pco213_postwork_benchmark/statistics/r10_vs_r30_stability.csv
  reports/pco213_postwork_benchmark/statistics/analysis_config.json
Outputs:
  reports/pco213_postwork_benchmark/statistics/rq9_sign_agreement_r30.csv
      one row per primary cell (block "paired_effect", 32 rows) and per level cell
      (block "level", 24 rows); agree_disjoint / agree_nested apply the row's
      agreement_rule (median sign, or R = 10 mean inside the bootstrap interval);
      statistically established at R = 30 means p_nb_holm_r30 < 0.05
  reports/pco213_postwork_benchmark/statistics/rq9_sign_agreement_r30_summary.json
      counts, the exceptions, Fisher tests and batch means quoted by the manuscript

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:
    python scripts/pco213_rq9_sign_agreement.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "bench_stats", REPO / "scripts" / "pco213_postwork_benchmark_stats.py")
_st = importlib.util.module_from_spec(_spec)
sys.modules["bench_stats"] = _st
_spec.loader.exec_module(_st)

SCRIPT = "scripts/pco213_rq9_sign_agreement.py"
LABEL = "post hoc"
REPORTS = Path("reports") / "pco213_postwork_benchmark"
IN_PQ = REPORTS / "tables" / "pareto_quality.csv"
IN_TESTS = REPORTS / "statistics" / "paired_primary_tests.csv"
IN_STAB = REPORTS / "statistics" / "r10_vs_r30_stability.csv"
IN_CFG = REPORTS / "statistics" / "analysis_config.json"
OUT_CSV = REPORTS / "statistics" / "rq9_sign_agreement_r30.csv"
OUT_JSON = REPORTS / "statistics" / "rq9_sign_agreement_r30_summary.json"

COST = "weighted"                          # primary cost of the frozen analysis
ALPHA = 0.05                               # "Holm-corrected p < 0.05" (sections/05_results.tex)
LEVEL_SETS = ["nbi_A", "nbi_B", "nbi_C"]   # the 12 cells per endpoint of tab:r10r30
BATCHES = ("reps0_9", "reps10_29", "reps0_29")

RULE_SIGN = "same sign of the median paired effect (|median| <= tie_tol counts as 0)"
RULE_LEVEL = "R = 10 mean inside the percentile-bootstrap 95% interval of the comparison-sample mean"
# agree_disjoint compares partitions 0-9 with 10-29, agree_nested 0-9 with 0-29, under the
# row's agreement_rule; both are filled in every row so that they load as booleans.
# Statistical significance at R = 30 is p_nb_holm_r30 < 0.05 (paired_effect rows only).
COLUMNS = [
    "block", "dataset", "comparison", "set", "endpoint", "cost", "agreement_rule",
    "n_reps0_9", "n_reps10_29", "n_reps0_29",
    "median_reps0_9", "median_reps10_29", "median_reps0_29",
    "mean_reps0_9", "mean_reps10_29", "mean_reps0_29",
    "shift_mean_reps10_29_minus_reps0_9",
    "agree_disjoint", "agree_nested",
    "sign_median_reps0_9", "sign_median_reps10_29", "sign_median_reps0_29",
    "wins_reps0_9", "ties_reps0_9", "losses_reps0_9",
    "wins_reps10_29", "ties_reps10_29", "losses_reps10_29",
    "wins_reps0_29", "ties_reps0_29", "losses_reps0_29",
    "fisher_p_wins_reps0_9_vs_reps10_29",
    "p_nb_holm_r30",
    "ci95_mean_reps10_29_lo", "ci95_mean_reps10_29_hi",
    "ci95_mean_reps0_29_lo", "ci95_mean_reps0_29_hi",
]
INT_COLS = [c for c in COLUMNS if c.startswith(("n_", "sign_", "wins_", "ties_", "losses_"))]
BOOL_COLS = ["agree_disjoint", "agree_nested"]


def sign_tol(x: float, tol: float) -> int:
    """Sign of a median paired effect; |x| <= tol is a tie (0), as in the frozen W/T/L rule."""
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def wtl(d: np.ndarray, tol: float) -> tuple[int, int, int]:
    w = int((d > tol).sum()); lo = int((d < -tol).sum())
    return w, d.size - w - lo, lo


def split(s: pd.Series, r10_max: int) -> dict[str, pd.Series]:
    """Partitions 0-9, 10-29 and 0-29 of a rep-indexed series (table order kept, so the
    bootstrap draws match the frozen r10_vs_r30_stability.csv)."""
    return {"reps0_9": s[s.index <= r10_max], "reps10_29": s[s.index > r10_max], "reps0_29": s}


def cell_id(r: dict) -> str:
    return f"{r['dataset']} | {r['comparison']} | {r['endpoint']}"


def main() -> None:
    cfg = json.loads((REPO / IN_CFG).read_text())
    tol = float(cfg["tie_tol"]); r10_max = int(cfg["r10_max_rep"])
    n_boot = int(cfg["n_boot"]); seed = int(cfg["seed"])
    endpoints = dict(cfg["endpoints"])
    primary = [tuple(c.split(" vs ")) for c in cfg["primary_comparisons"]]
    # the frozen config must be the one the frozen statistics module was run with
    assert tol == _st.TIE_TOL and n_boot == _st.N_BOOT and seed == _st.SEED, "config differs from frozen module"
    assert primary == [tuple(p) for p in _st.PRIMARY] and endpoints == _st.ENDPOINTS
    datasets = list(_st.DATASETS)

    pq = pd.read_csv(REPO / IN_PQ)
    pq = pq[pq.cost == COST]
    tests = pd.read_csv(REPO / IN_TESTS)
    tests = tests[tests.cost == COST].set_index(["dataset", "comparison", "endpoint"])
    stab = pd.read_csv(REPO / IN_STAB)
    stab = stab[(stab.cost == COST) & stab.metric.isin(list(endpoints))].set_index(["metric", "dataset", "set"])

    rows: list[dict] = []
    # ------------------------------------------------------------- paired primary cells
    for ds in datasets:
        for new, ref in primary:
            comp = f"{new} vs {ref}"
            for ep, direction in endpoints.items():
                a = pq[(pq.dataset == ds) & (pq.set == new)].set_index("rep")[ep]
                b = pq[(pq.dataset == ds) & (pq.set == ref)].set_index("rep")[ep]
                j = a.index.intersection(b.index)
                d = (a.loc[j] - b.loc[j]) if direction == "higher" else (b.loc[j] - a.loc[j])
                d = d[np.isfinite(d.to_numpy(float))]
                parts = {k: v.to_numpy(float) for k, v in split(d, r10_max).items()}
                t = tests.loc[(ds, comp, ep)]
                # consistency with the committed R = 30 tests (same pairing and sign convention)
                assert int(t["n"]) == parts["reps0_29"].size
                assert abs(float(t["mean_delta"]) - parts["reps0_29"].mean()) < 1e-12
                r = {"block": "paired_effect", "dataset": ds, "comparison": comp, "set": "", "endpoint": ep, "cost": COST,
                     "agreement_rule": RULE_SIGN}
                for k, x in parts.items():
                    w, ti, lo = wtl(x, tol)
                    r.update({f"n_{k}": x.size, f"median_{k}": float(np.median(x)), f"mean_{k}": float(x.mean()),
                              f"sign_median_{k}": sign_tol(float(np.median(x)), tol),
                              f"wins_{k}": w, f"ties_{k}": ti, f"losses_{k}": lo})
                r["shift_mean_reps10_29_minus_reps0_9"] = r["mean_reps10_29"] - r["mean_reps0_9"]
                r["agree_disjoint"] = r["sign_median_reps0_9"] == r["sign_median_reps10_29"]
                r["agree_nested"] = r["sign_median_reps0_9"] == r["sign_median_reps0_29"]
                table = [[r["wins_reps0_9"], r["n_reps0_9"] - r["wins_reps0_9"]],
                         [r["wins_reps10_29"], r["n_reps10_29"] - r["wins_reps10_29"]]]
                r["fisher_p_wins_reps0_9_vs_reps10_29"] = float(stats.fisher_exact(table, alternative="two-sided")[1])
                r["p_nb_holm_r30"] = float(t["p_nb_holm_family_dataset"])
                r["_holm_sig"] = bool(r["p_nb_holm_r30"] < ALPHA)
                r["_sign_mean_r30"] = sign_tol(float(t["mean_delta"]), tol)
                r["_raw_signs"] = {k: int(np.sign(np.median(x))) for k, x in parts.items()}
                rows.append(r)

    # ------------------------------------------------------------- level cells (tab:r10r30)
    for ds in datasets:
        for st in LEVEL_SETS:
            for ep in endpoints:
                s = pq[(pq.dataset == ds) & (pq.set == st)].set_index("rep")[ep].dropna()
                parts = split(s, r10_max)
                c = stab.loc[(ep, ds, st)]
                x10, x20, x30 = (parts[k].to_numpy(float) for k in BATCHES)
                lo20, hi20 = _st.boot_ci(x20, n_boot=n_boot, seed=seed)
                lo30, hi30 = _st.boot_ci(x30, n_boot=n_boot, seed=seed)
                # the committed nested check is reproduced exactly (same bootstrap draws)
                assert abs(float(c["r10_estimate"]) - x10.mean()) < 1e-12 and abs(float(c["r30_estimate"]) - x30.mean()) < 1e-12
                assert abs(float(c["r30_ci95_lo"]) - lo30) < 1e-12 and abs(float(c["r30_ci95_hi"]) - hi30) < 1e-12
                r10 = float(c["r10_estimate"])
                r = {"block": "level", "dataset": ds, "comparison": "", "set": st, "endpoint": ep, "cost": COST,
                     "agreement_rule": RULE_LEVEL}
                for k, x in zip(BATCHES, (x10, x20, x30)):
                    r.update({f"n_{k}": x.size, f"median_{k}": float(np.median(x)), f"mean_{k}": float(x.mean())})
                r.update({"ci95_mean_reps10_29_lo": lo20, "ci95_mean_reps10_29_hi": hi20,
                          "ci95_mean_reps0_29_lo": float(c["r30_ci95_lo"]), "ci95_mean_reps0_29_hi": float(c["r30_ci95_hi"]),
                          "agree_disjoint": bool(lo20 <= r10 <= hi20),
                          "agree_nested": bool(float(c["r30_ci95_lo"]) <= r10 <= float(c["r30_ci95_hi"])),
                          "shift_mean_reps10_29_minus_reps0_9": float(x20.mean() - x10.mean())})
                rows.append(r)

    # ------------------------------------------------------------- CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).reindex(columns=COLUMNS)
    for col in INT_COLS:
        df[col] = df[col].astype("Int64")
    for col in BOOL_COLS:
        df[col] = df[col].astype("boolean")
    df = df.sort_values(["block", "dataset", "comparison", "set", "endpoint"], kind="mergesort").reset_index(drop=True)
    (REPO / OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPO / OUT_CSV, index=False, lineterminator="\n")

    # ------------------------------------------------------------- summary JSON
    P = [r for r in rows if r["block"] == "paired_effect"]
    L = [r for r in rows if r["block"] == "level"]
    exc_dis = sorted(cell_id(r) for r in P if not r["agree_disjoint"])
    exc_nes = sorted(cell_id(r) for r in P if not r["agree_nested"])
    both = sum(r["agree_disjoint"] and r["agree_nested"] for r in P)

    def frac(r: dict, k: str) -> str:
        return f"{r[f'wins_{k}']}/{r[f'n_{k}']}"

    exceptions = []
    for r in sorted(P, key=cell_id):
        if r["agree_disjoint"] and r["agree_nested"]:
            continue
        exceptions.append({
            "cell": cell_id(r), "dataset": r["dataset"], "comparison": r["comparison"], "endpoint": r["endpoint"],
            "agree_disjoint": r["agree_disjoint"], "agree_nested": r["agree_nested"],
            **{f"median_{k}": r[f"median_{k}"] for k in BATCHES},
            **{f"wins_{k}": frac(r, k) for k in BATCHES},
            **{f"wtl_{k}": f"{r[f'wins_{k}']}/{r[f'ties_{k}']}/{r[f'losses_{k}']}" for k in BATCHES},
            "max_abs_median": max(abs(r[f"median_{k}"]) for k in BATCHES),
            "p_nb_holm_r30": r["p_nb_holm_r30"], "holm_significant_r30": r["_holm_sig"]})
    groups: dict[str, list[dict]] = {}
    for e in exceptions:
        groups.setdefault(f"{e['dataset']} | {e['comparison']}", []).append(e)
    exc_groups = {g: {"endpoints": sorted(e["endpoint"] for e in v), "max_abs_median": max(e["max_abs_median"] for e in v)}
                  for g, v in sorted(groups.items())}

    sig = [r for r in P if r["_holm_sig"]]
    raw_dis = sum(r["_raw_signs"]["reps0_9"] == r["_raw_signs"]["reps10_29"] for r in P)
    raw_nes = sum(r["_raw_signs"]["reps0_9"] == r["_raw_signs"]["reps0_29"] for r in P)

    level = {ep: {"n_cells": sum(r["endpoint"] == ep for r in L),
                  "r10_inside_ci_reps10_29": sum(r["endpoint"] == ep and r["agree_disjoint"] for r in L),
                  "r10_inside_ci_reps0_29": sum(r["endpoint"] == ep and r["agree_nested"] for r in L)}
             for ep in endpoints}
    batch_means: dict = {}
    for r in sorted(L, key=lambda r: (r["dataset"], r["endpoint"], r["set"])):
        batch_means.setdefault(r["dataset"], {}).setdefault(r["endpoint"], {})[r["set"]] = {
            "mean_reps0_9": r["mean_reps0_9"], "mean_reps10_29": r["mean_reps10_29"],
            "higher_in_reps10_29": bool(r["mean_reps10_29"] > r["mean_reps0_9"])}
    higher = {ds: {ep: sorted(s for s, v in eps.items() if v["higher_in_reps10_29"]) for ep, eps in d.items()}
              for ds, d in batch_means.items()}
    anchor = {}
    for r in sorted(P, key=cell_id):
        if r["comparison"] == f"{primary[0][0]} vs {primary[0][1]}" and r["endpoint"] == "hv_ratio":
            anchor[r["dataset"]] = {"wins_reps0_9": frac(r, "reps0_9"), "wins_reps10_29": frac(r, "reps10_29"),
                                    "wins_reps0_29": frac(r, "reps0_29"),
                                    **{f"wtl_{k}": f"{r[f'wins_{k}']}/{r[f'ties_{k}']}/{r[f'losses_{k}']}" for k in BATCHES},
                                    "fisher_p_two_sided_uncorrected": r["fisher_p_wins_reps0_9_vs_reps10_29"]}

    summary = {
        "script": SCRIPT, "label": LABEL,
        "inputs": [str(p) for p in (IN_PQ, IN_TESTS, IN_STAB, IN_CFG)],
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON)},
        "config": {"cost": COST, "tie_tol": tol, "r10_max_rep": r10_max, "n_boot": n_boot, "seed": seed,
                   "alpha_holm": ALPHA, "primary_comparisons": [f"{a} vs {b}" for a, b in primary],
                   "endpoints": endpoints, "level_sets": LEVEL_SETS,
                   "partitions": {"reps0_9": [0, r10_max], "reps10_29": [r10_max + 1, int(pq.rep.max())],
                                  "reps0_29": [0, int(pq.rep.max())]}},
        "operationalization": {
            "paired_effect": "delta > 0 means the first-named (new) method is better: dHV = HV_new - HV_ref, dIGD+ = IGD+_ref - IGD+_new",
            "median_sign": "sign of the median paired effect, 0 when |median| <= tie_tol",
            "disjoint_form": "sign in partitions 0-9 vs sign in partitions 10-29",
            "nested_form": "sign in partitions 0-9 (R = 10) vs sign in partitions 0-29 (R = 30)",
            "holm_significant_r30": "committed p_nb_holm_family_dataset < alpha_holm (Nadeau-Bengio, Holm within dataset)",
            "fisher": "two-sided Fisher exact test on wins vs non-wins (ties count as non-wins), partitions 0-9 vs 10-29, uncorrected",
            "level_check_new20": "R = 10 mean inside the percentile-bootstrap 95% interval of the mean of partitions 10-29 (frozen boot_ci, n_boot and seed)",
            "level_check_nested": "R = 10 mean inside the committed R = 30 bootstrap interval (r10_vs_r30_stability.csv)"},
        "median_sign_agreement": {
            "n_cells": len(P),
            "agree_disjoint": len(P) - len(exc_dis), "agree_nested": len(P) - len(exc_nes), "agree_both_forms": both,
            "exceptions_disjoint": exc_dis, "exceptions_nested": exc_nes,
            "same_exceptions_in_both_forms": exc_dis == exc_nes,
            "sensitivity_no_tie_zone": {"agree_disjoint": raw_dis, "agree_nested": raw_nes}},
        "exceptions": exceptions,
        "exception_groups": exc_groups,
        "holm_significant_r30": {
            "n": len(sig),
            "cells": sorted(cell_id(r) for r in sig),
            "keep_sign_reps0_9_vs_reps10_29": sum(r["agree_disjoint"] for r in sig),
            "keep_sign_reps0_9_vs_reps0_29": sum(r["agree_nested"] for r in sig),
            "keep_sign_reps0_9_vs_mean_r30": sum(r["sign_median_reps0_9"] == r["_sign_mean_r30"] for r in sig),
            "among_exceptions": sum(r["_holm_sig"] and not (r["agree_disjoint"] and r["agree_nested"]) for r in P)},
        "level_check": level,
        "batch_means": batch_means,
        "batch_mean_higher_in_reps10_29": higher,
        "anchor_hv_batch_wins": anchor,
    }
    (REPO / OUT_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    ms = summary["median_sign_agreement"]; hs = summary["holm_significant_r30"]
    print(f"median sign: disjoint {ms['agree_disjoint']}/{ms['n_cells']}, nested {ms['agree_nested']}/{ms['n_cells']}, "
          f"both {ms['agree_both_forms']}, same exceptions {ms['same_exceptions_in_both_forms']}")
    for e in exceptions:
        print(f"  exception {e['cell']}: wins {e['wins_reps0_9']} vs {e['wins_reps10_29']}, "
              f"medians {e['median_reps0_9']:+.6f} / {e['median_reps10_29']:+.6f} / {e['median_reps0_29']:+.6f}")
    print(f"Holm-significant at R = 30: {hs['n']}; keep sign disjoint {hs['keep_sign_reps0_9_vs_reps10_29']}, "
          f"nested {hs['keep_sign_reps0_9_vs_reps0_29']}, vs R30 mean {hs['keep_sign_reps0_9_vs_mean_r30']}")
    for ep, v in level.items():
        print(f"level {ep}: new-20 {v['r10_inside_ci_reps10_29']}/{v['n_cells']}, nested {v['r10_inside_ci_reps0_29']}/{v['n_cells']}")
    for ds, v in anchor.items():
        print(f"anchor HV {ds}: {v['wins_reps0_9']} vs {v['wins_reps10_29']}, Fisher p = {v['fisher_p_two_sided_uncorrected']:.4f}")
    print("wrote", OUT_CSV, "and", OUT_JSON)


if __name__ == "__main__":
    main()
