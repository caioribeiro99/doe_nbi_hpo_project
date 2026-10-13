#!/usr/bin/env python
"""The coarse grid at the arms' budget: its first 186 stored evaluations (R2).

Question (coauthor review R2). The grid led NBI-S at the frozen comparator budget of
386 real evaluations, while NBI-S costs 186 standalone. How does the grid compare when
it is stopped at 186?

No grid point is evaluated. The grid's stored rows are in evaluation order, and that
order is fixed by the frozen code (``baselines.coarse_grid``): the two-level corner
mesh of 128 points first, then seeded uniform padding. The script checks, for every
unit, that the first 186 stored rows are exactly the configurations the frozen code
builds at a budget of 186 with the same seed, so the prefix is the grid the protocol
would have run at that budget, measured by the campaign's own evaluations.

Definition (fixed before the first run):
  * Scoring: the frozen indicator code (``scoring.indicators``) against each unit's
    stored CORE front, with the frozen per-dataset factor model, exactly as the
    campaign scored GRID from its whole stored archive. The script first rescores the
    full 386 rows and requires the stored GRID values in indicators_long.csv, and
    reproduces Table 5 (baseline_vs_arms.json) from them; it stops on any difference.
  * Reported per dataset, at 186 and at 386: the median CORE-relative hypervolume
    ratio of GRID, the median paired difference NBI-S - GRID, NBI-S wins of 30, the
    grid's wins of 30 against the frozen 21-of-30 direction rule, and the percentile
    bootstrap interval of the median paired difference (the frozen
    ``primary_analysis.boot_median_ci``, its seed and resample count).
  * At 186 the grid is 128 corners and 58 padding points. The number of its points
    that re-measure one of the unit's 88 design configurations, which sit inside CORE,
    is counted at both budgets and reported, not corrected.

Provenance label: retrospective, post hoc; computed after the campaign from stored
outputs. It is not a budget-matched comparator arm and replaces nothing.

Inputs (repo-relative):
    experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/
        direct_baselines.json, reference_core.json
    papers/xgboost_hpo_vrfnbi/analysis/indicators_long.csv, baseline_vs_arms.json
    papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json
Output:
    papers/xgboost_hpo_vrfnbi/analysis/grid_budget_prefix.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage, from the repository root:
    python -B papers/xgboost_hpo_vrfnbi/scripts/grid_budget_prefix.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.arms import Realizer  # noqa: E402
from doe_xgb.campaign.evaluator import BOUNDS, INT_PARAMS, PARAMS  # noqa: E402
from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402
from doe_xgb.campaign.runner import DATASETS  # noqa: E402
from doe_xgb.campaign.scoring import canonical_key, indicators  # noqa: E402
from doe_xgb.campaign.seeding import derive_seed  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
ANALYSIS = PAPER / "analysis"
OUT = ANALYSIS / "grid_budget_prefix.json"
N_REPS, FULL, ARM_BUDGET = 30, 386, 186
TOL = 1e-12

_spec = importlib.util.spec_from_file_location("primary_analysis", HERE / "primary_analysis.py")
_pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pa)


def frozen_grid_configs(budget: int, seed: int) -> list[dict]:
    """The configurations ``baselines.coarse_grid`` builds, without evaluating any."""
    k = len(PARAMS)
    levels = max(2, int(np.floor(budget ** (1.0 / k))))
    mesh = np.array(np.meshgrid(*[np.linspace(-1.0, 1.0, levels)] * k)).reshape(k, -1).T
    rng = np.random.default_rng(seed)
    if len(mesh) < budget:
        mesh = np.vstack([mesh, rng.uniform(-1.0, 1.0, size=(budget - len(mesh), k))])
    rz = Realizer(PARAMS, BOUNDS, list(INT_PARAMS))
    return [rz.realize(x)[0] for x in mesh]


def main() -> int:
    ind = pd.read_csv(ANALYSIS / "indicators_long.csv")
    bva = json.loads((ANALYSIS / "baseline_vs_arms.json").read_text())
    out = {"provenance": ("retrospective, post hoc (coauthor review R2): the first 186 of "
                         "the grid's stored evaluations, scored with the frozen code"),
           "datasets": {}}
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        hv = {FULL: [], ARM_BUDGET: []}
        overlap = {FULL: [], ARM_BUDGET: []}
        nbi_s = []
        for rep in range(N_REPS):
            u = ROOT / ds / f"rep_{rep:02d}"
            rows = json.loads((u / "direct_baselines.json").read_text())["methods"]["grid"]["rows"]
            if len(rows) != FULL:
                raise SystemExit(f"{ds} rep {rep}: {len(rows)} grid rows")
            seed = derive_seed(ds, rep, "grid")
            want = [canonical_key(c) for c in frozen_grid_configs(ARM_BUDGET, seed)]
            have = [canonical_key({p: r[p] for p in PARAMS}) for r in rows[:ARM_BUDGET]]
            if want != have:
                raise SystemExit(f"{ds} rep {rep}: stored prefix is not the frozen 186-point grid")
            dkeys = {canonical_key({p: r[p] for p in PARAMS})
                     for r in json.loads((u / "design.json").read_text())["rows"]}
            overlap[FULL].append(sum(canonical_key({p: r[p] for p in PARAMS}) in dkeys for r in rows))
            overlap[ARM_BUDGET].append(sum(k in dkeys for k in have))
            core = np.asarray(json.loads((u / "reference_core.json").read_text())["front"], float)
            sel = ind[(ind.dataset == ds) & (ind.replication == rep)]
            stored_grid = float(sel[sel.entity == "GRID"]["core__hv_ratio"].iloc[0])
            full = indicators(fm.objectives(pd.DataFrame(rows)), core)["hv_ratio"]
            if abs(full - stored_grid) > TOL:
                raise SystemExit(f"{ds} rep {rep}: full grid rescored {full} vs stored {stored_grid}")
            hv[FULL].append(full)
            hv[ARM_BUDGET].append(indicators(fm.objectives(pd.DataFrame(rows[:ARM_BUDGET])),
                                             core)["hv_ratio"])
            nbi_s.append(float(sel[sel.entity == "NBI-S"]["core__hv_ratio"].iloc[0]))
        nbi_s = np.asarray(nbi_s)
        res = {}
        for b in (FULL, ARM_BUDGET):
            g = np.asarray(hv[b], float)
            d = nbi_s - g
            rng = np.random.default_rng(_pa.SEED)
            lo, hi = _pa.boot_median_ci(d, rng)
            res[str(b)] = {"grid_median_hv_ratio": float(np.median(g)),
                           "median_diff_nbi_s_minus_grid": float(np.median(d)),
                           "median_diff_ci95": [float(lo), float(hi)],
                           "nbi_s_wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                           "grid_wins": int((d < 0).sum()),
                           "grid_meets_21_of_30_rule": bool((d < 0).sum() >= 21),
                           "wilcoxon_p_raw_post_hoc": float(stats.wilcoxon(d).pvalue)}
        ref = bva[ds]
        r = res[str(FULL)]
        if (abs(r["grid_median_hv_ratio"] - ref["grid_median"]) > TOL
                or abs(r["median_diff_nbi_s_minus_grid"] - ref["median_diff_nbi_s_minus_grid"]) > TOL
                or r["nbi_s_wins"] != ref["nbi_s_wins"]
                or abs(r["wilcoxon_p_raw_post_hoc"] - ref["wilcoxon_p_raw"]) > 1e-9):
            raise SystemExit(f"{ds}: Table 5 not reproduced from the rescored grid")
        out["datasets"][ds] = {"nbi_s_median_hv_ratio": float(np.median(nbi_s)),
                               "prefix_is_frozen_186_point_grid": True,
                               "full_grid_rescored_equals_stored": True,
                               "design_rows_reevaluated": {str(b): sorted(set(overlap[b]))
                                                           for b in (FULL, ARM_BUDGET)},
                               "by_budget": res}
        print(ds, json.dumps(res))
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("->", OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
