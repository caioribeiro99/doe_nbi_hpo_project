#!/usr/bin/env python
"""Re-run ALL FOUR Stage-A screening criteria on the corrected factor model.

`audits/PILOT_STAGE_A_FINDINGS.md` recorded "all four datasets pass all four
criteria" on the factor algebra the protocol has since withdrawn. Only the
objective-count evidence was revalidated afterwards; the screening itself was not.
This recomputes every criterion in `protocol/dataset_selection.md` against the
frozen per-dataset reference models, so the panel decision rests on the algebra the
campaign will actually use.

This is PRE-CONFIRMATORY SCREENING EVIDENCE. It measures one partition per dataset
and is not a study result. It is computed before any arm has run.

Usage:
    python rescreen_panel_corrected.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.design import external_scores  # noqa: E402
from doe_xgb.campaign.factor_model import (load_reference_factor_model,  # noqa: E402
                                           raw_conflict)
from doe_xgb.reporting import pareto_front  # noqa: E402

PILOT = PAPER / "audits" / "pilot_stage_a"
OUT = PAPER / "audits" / "panel_rescreen_corrected.json"
DATASETS = ("magic", "spambase", "adult", "bank_marketing")


def front_curvature(F: np.ndarray) -> dict:
    """Deviation of the non-dominated set from the chord joining its endpoints.

    Normalized by the chord length, so it is scale free. Zero means the front is
    linear, which is the case in which weighted-sum scalarization reaches every
    point of it and the geometry contrast has nothing to separate.
    """
    mask = pareto_front(F)
    P = F[mask]
    P = P[np.argsort(P[:, 0])]
    if len(P) < 3:
        return {"n_front": int(mask.sum()), "curvature": None,
                "note": "fewer than three non-dominated points; curvature undefined"}
    a, b = P[0], P[-1]
    v = b - a
    L = float(np.linalg.norm(v))
    signed = (v[0] * (P[:, 1] - a[1]) - v[1] * (P[:, 0] - a[0])) / L
    return {"n_front": int(mask.sum()),
            "front_share_of_points": round(float(mask.mean()), 4),
            "curvature": round(float(np.abs(signed).max() / L), 4),
            "max_deviation_toward_utopia": round(float(-signed.min() / L), 4),
            "max_deviation_away_from_utopia": round(float(signed.max() / L), 4)}


def screen(ds: str) -> dict:
    design = pd.read_csv(PILOT / f"{ds}_design.csv")
    complement = pd.read_csv(PILOT / f"{ds}_validation_complement.csv")
    both = pd.concat([design, complement], ignore_index=True)
    fm = load_reference_factor_model(ds)

    t = fm.transform(design)
    q, c = t["quality"], t["cost"]

    # --- criterion 1: latent conflict, AND the raw-response conflict beside it ---
    # raw_conflict() is the canonical measurement and correlates the equally weighted
    # quality badness against the RAW cost response, log leaf count. An earlier
    # version of this script reimplemented it and correlated the raw quality
    # composite against the LATENT cost factor instead, which is a third quantity
    # that is neither criterion: it put a different "raw conflict" number in this
    # artifact from the one in audits/reference_factor_models/.
    c1 = {
        "latent_spearman": round(float(spearmanr(q, c).statistic), 4),
        "latent_pearson": float(np.corrcoef(q, c)[0, 1]),
        "raw_response_spearman": round(float(raw_conflict(design)), 4),
        "raw_response_definition": ("equally weighted quality badness against the RAW "
                                    "cost response (log leaf count); factor stage not "
                                    "involved on either side"),
        "criterion_text": "clearly negative; a value near zero means no conflict",
        "latent_is_zero_by_construction": bool(abs(np.corrcoef(q, c)[0, 1]) < 1e-10),
    }

    # --- criterion 2: curvature of the non-dominated set of design rows ---------
    c2 = front_curvature(np.column_stack([q, c]))

    # --- criterion 3: external R2 of each surrogate, strictly between -----------
    Yd, Ye = fm.objectives(design), fm.objectives(complement)
    c3 = {}
    for j, obj in enumerate(("quality", "cost")):
        sc = external_scores(design, Yd[:, j], complement, Ye[:, j])
        r2 = float(sc["external_r2"])
        c3[obj] = {"external_r2": round(r2, 4),
                   "external_spearman": round(float(sc["external_spearman"]), 4),
                   "surface_terms": int(sc["terms"]),
                   "spread_ratio_design_over_external":
                       round(float(sc["spread_ratio_design_over_external"]), 3),
                   "strictly_between_trivial_and_perfect": bool(0.0 < r2 < 1.0),
                   # the frozen reliability gate, reported beside the criterion
                   "meets_frozen_gate":
                       bool(r2 >= 0.5 and float(sc["external_spearman"]) >= 0.9)}

    # --- criterion 4: cost dynamic range ---------------------------------------
    leaves = both["Leaves_Mean"].to_numpy(float)
    c4 = {"max_over_min_leaf_count": round(float(leaves.max() / max(leaves.min(), 1e-9)), 1),
          "min": float(leaves.min()), "max": float(leaves.max())}

    return {"dataset": ds,
            "criterion_1_objective_conflict": c1,
            "criterion_2_front_curvature": c2,
            "criterion_3_surrogate_usefulness": c3,
            "criterion_4_cost_dynamic_range": c4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    payload = {
        "purpose": "all four Stage-A screening criteria recomputed on the corrected "
                   "factor model and the frozen per-dataset reference models",
        "status": "PRE-CONFIRMATORY SCREENING EVIDENCE; not a study result",
        "no_arm_executed": True,
        "datasets": {ds: screen(ds) for ds in DATASETS},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("MISMATCH: committed rescreen differs from a rebuild")
            return 1
    else:
        OUT.write_text(text)

    print(f"{'dataset':16} {'C1 latent':>10} {'C1 raw':>8} {'C1 pearson':>11} "
          f"{'C2 |front|':>11} {'C2 curv':>8} {'C3 q R2':>8} {'C3 c R2':>8} {'C4 range':>9}")
    for ds, d in payload["datasets"].items():
        c1, c2, c3, c4 = (d["criterion_1_objective_conflict"], d["criterion_2_front_curvature"],
                          d["criterion_3_surrogate_usefulness"], d["criterion_4_cost_dynamic_range"])
        cv = "n/a" if c2["curvature"] is None else f"{c2['curvature']:.4f}"
        print(f"{ds:16} {c1['latent_spearman']:10.3f} {c1['raw_response_spearman']:8.3f} "
              f"{c1['latent_pearson']:11.1e} {c2['n_front']:11} {cv:>8} "
              f"{c3['quality']['external_r2']:8.3f} {c3['cost']['external_r2']:8.3f} "
              f"{c4['max_over_min_leaf_count']:9.1f}")
    print(f"\n{'checked' if a.check else 'wrote ' + str(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
