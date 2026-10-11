#!/usr/bin/env python
"""Descriptive diagnostics of the frozen factor models (Supplement S4.1 and S4.2).

Reads only committed pre-campaign artifacts: the frozen reference models
(audits/reference_factor_models/<dataset>.json) and the pilot design rows they were
fitted on (audits/pilot_stage_a/<dataset>_design.csv). Writes
analysis/factor_dimensionality.json with, per dataset:

- the eigenvalues of the correlation matrix of the seven standardized responses Z on
  the 88 fitting rows, the cumulative share of their total (which is 7), the Kaiser
  count and lambda3/lambda4;
- the share of the transformed leaf count's variance reproduced by the first one, two
  and three components;
- the coefficients of the two frozen objectives on Z (f_quality and f_cost are affine
  in Z), checked against the frozen transform of src/doe_xgb/campaign/factor_model.py;
- the Spearman agreement of the variance-weighted and equal-weight quality composites
  on the fitting rows, which reproduces analysis/factor_diagnostics_88.json.

These values are reported and never acted on: k = 3 is fixed at protocol level
(protocol/OBJECTIVE_DEFINITIONS.md section 8). They are computed after the campaign
from pre-campaign artifacts; no learner, optimizer or campaign stage is run, and no
committed artifact other than the output file is written.

Provenance label: retrospective.

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage: python papers/xgboost_hpo_vrfnbi/scripts/factor_dimensionality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parents[1]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.factor_model import RESPONSES, load_reference_factor_model  # noqa: E402

DATASETS = ("magic", "spambase", "adult", "bank_marketing")
OUT = PAPER / "analysis" / "factor_dimensionality.json"
TOL = 1e-9


def objective_coefficients(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of f_quality and f_cost on the standardized responses Z.

    The frozen map is S_z = (Z V diag(1/sqrt(lambda)) R - mu_S) / sigma_S, so each
    objective is affine in Z; the intercept does not enter the coefficients.
    """
    B = ((np.asarray(m["pca_components"], dtype=float).T
          / np.sqrt(np.asarray(m["eigenvalues"], dtype=float)))
         @ np.asarray(m["varimax_rotation"], dtype=float)
         / np.asarray(m["score_standardization_sd"], dtype=float))
    cq = B[:, list(m["quality_factor_indices"])] @ np.asarray(m["quality_weights"], dtype=float)
    cc = B[:, int(m["cost_factor_index"])]
    return cq, cc


def raw_rows_for(m: dict, Z: np.ndarray) -> pd.DataFrame:
    """Raw responses whose standardized, canonicalized values are the rows of Z."""
    M = (np.asarray(m["standardization_mean"], dtype=float)
         + np.asarray(m["standardization_sd"], dtype=float) * Z)
    cols = {}
    for j, (name, spec) in enumerate(RESPONSES.items()):
        v = M[:, j] * (1.0 if spec["minimize"] else -1.0)
        if spec["transform"] == "log1p":
            v = np.expm1(v)
        cols[name] = v
    return pd.DataFrame(cols)


def main() -> int:
    fd88 = json.loads((PAPER / "analysis" / "factor_diagnostics_88.json").read_text())
    out = {"script": "papers/xgboost_hpo_vrfnbi/scripts/factor_dimensionality.py",
           "label": "retrospective",
           "computed": "after the campaign, from committed pre-campaign artifacts",
           "descriptive_only": True,
           "datasets": {}}
    for ds in DATASETS:
        m = json.loads((PAPER / "audits" / "reference_factor_models" / f"{ds}.json").read_text())
        names = list(m["response_names"])
        assert names == list(RESPONSES), ds
        ev = np.asarray(m["eigenvalues_full"], dtype=float)
        li = names.index("Leaves_Mean")
        lv = (np.asarray(m["pca_components"], dtype=float)[:, li] ** 2
              * np.asarray(m["eigenvalues"], dtype=float))
        cq, cc = objective_coefficients(m)

        # The closed form must agree with the frozen transform itself.
        fm = load_reference_factor_model(ds)
        obj = fm.objectives(raw_rows_for(m, np.vstack([np.zeros(len(names)), np.eye(len(names))])))
        assert np.allclose(obj[1:, 0] - obj[0, 0], cq, atol=TOL), ds
        assert np.allclose(obj[1:, 1] - obj[0, 1], cc, atol=TOL), ds

        # Agreement of the two weightings on the fitting rows, the figure of
        # analysis/factor_diagnostics_88.json, which has no other committed generator.
        t = fm.transform(pd.read_csv(PAPER / "audits" / "pilot_stage_a" / f"{ds}_design.csv"))
        rho = float(spearmanr(t["quality"], t["quality_equal"]).statistic)
        assert abs(rho - fd88[ds]["spearman_weighted_vs_equal"]) < TOL, ds
        assert np.allclose(ev, fd88[ds]["eigenvalues_88"], atol=TOL), ds
        assert int((ev > 1.0).sum()) == int(fd88[ds]["kaiser"]) == int(m["kaiser_retained"]), ds

        out["datasets"][ds] = {
            "n_fitting_rows": int(m["reference_set"]["n_total"]),
            "n_components_fixed": int(m["n_components_fixed"]),
            "eigenvalues": ev.tolist(),
            "cumulative_variance_share": (np.cumsum(ev) / ev.sum()).tolist(),
            "kaiser_retained": int((ev > 1.0).sum()),
            "lambda3_over_lambda4": float(ev[2] / ev[3]),
            "leaf_count_variance_reproduced": np.cumsum(lv).tolist(),
            "objective_coefficients_on_Z": {"f_quality": dict(zip(names, cq.tolist())),
                                            "f_cost": dict(zip(names, cc.tolist()))},
            "spearman_weighted_vs_equal_fitting_rows": rho,
        }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
