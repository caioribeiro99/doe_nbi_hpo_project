#!/usr/bin/env python
"""The frozen quality composite beyond its fitting sample (Supplement S4.3).

Evaluates the frozen orientation check of amendment 19, ``composite_alignment``: the
Spearman correlation of f_quality with the equal-weight mean of the six standardized,
canonicalized quality responses, which must be positive. Per dataset it is evaluated on
the 88 Stage A fitting rows, on each replication's own 88 design rows (and checked against
the value the runner recorded there), and on the configurations the random-search
baseline drew uniformly from the coded box in each replication. It also counts
reversals: ordered pairs of configurations in which the first is at least as good as the
second on all six quality responses and strictly better on one, yet receives the larger
(worse) f_quality. Random-search pairs are formed within a replication and pooled.
Finally it ranks, among the 88 fitting rows, the f_quality values of the rows that predict
no positive case in any fold (Recall_Mean 0 and Specificity_Mean 1 at the 0.5 threshold).

Provenance label: post hoc.

Inputs (read-only):
  papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json, through the
      frozen load_reference_factor_model, transform and composite_alignment
  papers/xgboost_hpo_vrfnbi/audits/pilot_stage_a/<dataset>_design.csv (the fitting rows)
  experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/design.json,
      factor_model.json and direct_baselines.json (methods.random.rows)

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage: python papers/xgboost_hpo_vrfnbi/scripts/composite_beyond_fitting_rows.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.factor_model import (RESPONSES, apply_transforms,  # noqa: E402
                                           composite_alignment,
                                           load_reference_factor_model)
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402

PAPER = REPO / "papers" / "xgboost_hpo_vrfnbi"
ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = PAPER / "analysis" / "composite_beyond_fitting_rows.json"
QUALITY = [i for i, c in enumerate(RESPONSES) if RESPONSES[c]["role"] == "quality"]
TOL = 1e-12          # recomputed against recorded alignment: same frozen code, same rows


def reversals(fm, df: pd.DataFrame) -> list[int]:
    """[reversed pairs, quality-dominance pairs] among the rows of ``df``.

    A pair (i, j) counts when i is at least as good as j on every canonicalized
    (minimization-oriented) quality response and strictly better on one; it is reversed
    when f_quality(i) > f_quality(j).
    """
    M = apply_transforms(df)[:, QUALITY]
    q = np.asarray(fm.transform(df)["quality"], dtype=float)
    dom = ((M[:, None, :] <= M[None, :, :]).all(axis=2)
           & (M[:, None, :] < M[None, :, :]).any(axis=2))
    return [int((dom & (q[:, None] > q[None, :])).sum()), int(dom.sum())]


def no_positive_ranks(fm, fit: pd.DataFrame) -> dict:
    """Where the fitting rows that predict no positive case fall among the 88 composites."""
    none_pos = ((fit["Recall_Mean"].to_numpy(float) == 0)
                & (fit["Specificity_Mean"].to_numpy(float) == 1))
    q = np.asarray(fm.transform(fit)["quality"], dtype=float)
    rank = np.array([1 + int((q > x).sum()) for x in q])     # 1 = largest (worst) f_quality
    ranks = sorted(int(r) for r in rank[none_pos])
    n = len(ranks)
    return {"n_rows": n, "n_fitting_rows": int(len(fit)), "ranks_from_worst": ranks,
            "are_the_n_worst": (ranks == list(range(1, n + 1))) if n else None}


def main() -> int:
    out = {"script": "papers/xgboost_hpo_vrfnbi/scripts/composite_beyond_fitting_rows.py",
           "label": "post hoc", "post_hoc": True,
           "note": ("Computed after the campaign from stored evaluations with the frozen "
                    "reference factor models; it changes no frozen number and fed no arm. "
                    "Alignment is composite_alignment (Spearman of f_quality with the "
                    "equal-weight mean of the six standardized canonicalized quality "
                    "responses, standardized within each sample). Random-search rows are "
                    "the configurations baselines.random_search drew uniformly in the coded "
                    "box per replication. A reversal is an ordered pair at least as good on "
                    "all six quality responses and strictly better on one that receives the "
                    "larger f_quality; random-search pairs are formed within replications "
                    "and pooled as [reversed, pairs]. no_positive_fitting_rows ranks the "
                    "fitting rows with Recall_Mean 0 and Specificity_Mean 1 among the "
                    "f_quality values of all fitting rows, rank 1 being the largest "
                    "(worst)."),
           "inputs": ["papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json",
                      "papers/xgboost_hpo_vrfnbi/audits/pilot_stage_a/<dataset>_design.csv",
                      "experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/design.json",
                      "experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/"
                      "factor_model.json",
                      "experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/"
                      "direct_baselines.json"],
           "datasets": {}}
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        m = json.loads((PAPER / "audits" / "reference_factor_models" / f"{ds}.json").read_text())
        fit = pd.read_csv(PAPER / "audits" / "pilot_stage_a" / f"{ds}_design.csv")
        a_fit = composite_alignment(fm, fit)
        a_fit_committed = float(
            m["diagnostics_on_reference_set"]["composite_alignment_with_raw_quality"])
        assert abs(a_fit - a_fit_committed) < TOL, (ds, a_fit, a_fit_committed)

        a_design, a_random, n_random, pooled = [], [], set(), [0, 0]
        for rep in range(N_REPLICATIONS):
            u = ROOT / ds / f"rep_{rep:02d}"
            design = pd.DataFrame(json.loads((u / "design.json").read_text())["rows"])
            a_d = composite_alignment(fm, design)
            recorded = json.loads((u / "factor_model.json").read_text())[
                "composite_alignment_with_raw_quality"]
            assert abs(a_d - recorded) < TOL, (ds, rep, a_d, recorded)
            a_design.append(a_d)

            rnd = pd.DataFrame(json.loads((u / "direct_baselines.json").read_text())
                               ["methods"]["random"]["rows"])
            n_random.add(int(len(rnd)))
            a_random.append(composite_alignment(fm, rnd))
            rv = reversals(fm, rnd)
            pooled = [pooled[0] + rv[0], pooled[1] + rv[1]]

        a = np.asarray(a_random, dtype=float)
        out["datasets"][ds] = {
            "fitting_rows": float(a_fit),
            "fitting_rows_committed": a_fit_committed,
            "design_rows_min": float(min(a_design)),
            "design_rows_max": float(max(a_design)),
            "random_median": float(np.median(a)),
            "random_min": float(a.min()),
            "random_max": float(a.max()),
            "random_negative": int((a < 0).sum()),
            "n_replications": int(len(a)),
            "random_rows_per_replication": sorted(n_random),
            "reversals_fitting": reversals(fm, fit),
            "reversals_random_pooled": pooled,
            "no_positive_fitting_rows": no_positive_ranks(fm, fit),
        }

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for ds, r in out["datasets"].items():
        rate = r["reversals_random_pooled"][0] / r["reversals_random_pooled"][1]
        print(f"{ds:15s} fitting {r['fitting_rows']:+.3f}  design {r['design_rows_min']:+.3f}"
              f" to {r['design_rows_max']:+.3f}  random median {r['random_median']:+.3f}"
              f" ({r['random_min']:+.3f} to {r['random_max']:+.3f}), negative"
              f" {r['random_negative']}/{r['n_replications']}  reversal rate {rate:.4f}"
              f"  no-positive ranks {r['no_positive_fitting_rows']['ranks_from_worst']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
