#!/usr/bin/env python
"""Build the ONE factor model per dataset that the campaign applies everywhere.

`protocol/EXPERIMENT_PROTOCOL.md` 7.2 freezes this, and states the reason:

    if the model is refit per replication, the objective is not the same variable
    in every pair, so 30 paired indicator values do not live in one objective
    space, and no normalized indicator is invariant to that.

The reference set is declared there too: the Stage A 88 design rows plus their
78-point complement, 166 points, already evaluated and committed. The runner had
been calling `fit_factor_model(design_df)` on every replication instead, which is
the confound that decision exists to remove. This script produces the artifact the
runner reads, so the objective definition is a committed, diffable, version-
controlled file rather than something recomputed 120 times.

The model is a MEASUREMENT DEFINITION, not a predictor of any campaign outcome. It
fixes what "quality" and "cost" mean so that 30 replications can be compared; it is
built before any arm runs and no arm result is in its input.

Usage:
    python build_reference_factor_models.py [--check]

`--check` rebuilds and compares against the committed artifacts without writing,
exiting non-zero on any difference. That is the form used in the test suite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.factor_model import (composite_alignment,  # noqa: E402
                                           fit_factor_model, raw_conflict)

PILOT = PAPER / "audits" / "pilot_stage_a"
OUT = PAPER / "audits" / "reference_factor_models"
DATASETS = ("magic", "spambase", "adult", "bank_marketing")

# The declared reference set, by file. Both halves are committed Stage A output.
DESIGN = "{ds}_design.csv"
COMPLEMENT = "{ds}_validation_complement.csv"


def reference_frame(ds: str) -> pd.DataFrame:
    design = pd.read_csv(PILOT / DESIGN.format(ds=ds))
    complement = pd.read_csv(PILOT / COMPLEMENT.format(ds=ds))
    frame = pd.concat([design, complement], ignore_index=True)
    if len(frame) != len(design) + len(complement):
        raise SystemExit(f"{ds}: concatenation lost rows")
    return frame, len(design), len(complement)


def build(ds: str) -> dict:
    frame, n_design, n_complement = reference_frame(ds)
    fm = fit_factor_model(frame)
    from scipy.stats import spearmanr
    t = fm.transform(frame)
    payload = {
        "dataset": ds,
        "protocol_clause": "EXPERIMENT_PROTOCOL.md 7.2",
        "reference_set": {
            "description": ("the Stage A design rows plus their complementary half "
                            "fraction, already evaluated and committed"),
            "files": [DESIGN.format(ds=ds), COMPLEMENT.format(ds=ds)],
            "n_design": int(n_design),
            "n_complement": int(n_complement),
            "n_total": int(len(frame)),
        },
        "fitted_on": "the reference set above, once; APPLIED to every replication",
        "no_arm_result_in_input": True,
        **fm.as_dict(),
        "diagnostics_on_reference_set": {
            "composite_alignment_with_raw_quality":
                float(composite_alignment(fm, frame)),
            "objective_conflict_latent_spearman":
                float(spearmanr(t["quality"], t["cost"]).statistic),
            "objective_conflict_latent_pearson":
                float(np.corrcoef(t["quality"], t["cost"])[0, 1]),
            "objective_conflict_raw_responses": float(raw_conflict(frame)),
            "max_abs_offdiagonal_factor_correlation":
                float(np.abs(np.corrcoef(t["factor_scores"], rowvar=False)
                             - np.eye(t["factor_scores"].shape[1])).max()),
        },
    }
    return payload


def canonical(d: dict) -> str:
    return json.dumps(d, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed artifacts; write nothing")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bad = []
    for ds in DATASETS:
        payload = build(ds)
        path = OUT / f"{ds}.json"
        text = canonical(payload)
        if a.check:
            if not path.exists():
                bad.append(f"{ds}: no committed reference model at {path}")
            elif path.read_text() != text:
                bad.append(f"{ds}: committed reference model differs from a rebuild")
        else:
            path.write_text(text)
        diag = payload["diagnostics_on_reference_set"]
        print(f"{ds:16} n={payload['reference_set']['n_total']:4}  "
              f"alignment {diag['composite_alignment_with_raw_quality']:+.4f}  "
              f"latent rho {diag['objective_conflict_latent_spearman']:+.4f}  "
              f"latent r {diag['objective_conflict_latent_pearson']:+.2e}  "
              f"raw rho {diag['objective_conflict_raw_responses']:+.4f}")
    if bad:
        for b in bad:
            print("MISMATCH:", b)
        return 1
    print("\n" + ("checked" if a.check else f"written to {OUT}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
