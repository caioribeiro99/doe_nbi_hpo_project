#!/usr/bin/env python
"""What screening criterion 1 actually measures, quantified rather than asserted.

Criterion 1 requires the Spearman between the quality composite and the cost
objective to be "clearly negative". Three claims about that statistic have been made
in this workspace and TWO OF THEM WERE WRONG, both by over-generalizing from a
handful of numbers:

    "Pearson is zero by construction on every dataset"      -- exact only on the
                                                               FITTING sample
    "|r| <= 0.04 on any subsample"                          -- false; at n = 30,
                                                               80% of draws exceed it
    "the latent Spearman's sign is not stable"              -- too broad; it is
                                                               stable per dataset and
                                                               differs BETWEEN them

This script measures the thing instead of describing it, so the next claim is
checkable. It resamples at n = 88, the size that actually occurs: every replication
of the campaign evaluates exactly the 88 design rows.

PRE-CONFIRMATORY SCREENING EVIDENCE. No arm is involved.
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

from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402

PILOT = PAPER / "audits" / "pilot_stage_a"
OUT = PAPER / "audits" / "latent_conflict_stability.json"
DATASETS = ("magic", "spambase", "adult", "bank_marketing")
DRAWS = 1500
N = 88          # the design size every replication measures


def measure(ds: str, rng) -> dict:
    # The fitting sample IS the 88 design rows: the frozen model is fitted on them
    # and on nothing else, so the exact-orthogonality identity is a property of this
    # frame. Resampling is a bootstrap of those same rows, which is what a
    # replication's own design measurement is a draw from.
    design = pd.read_csv(PILOT / f"{ds}_design.csv")
    fm = load_reference_factor_model(ds)

    F_fit = fm.objectives(design)
    r_fit = float(np.corrcoef(F_fit[:, 0], F_fit[:, 1])[0, 1])

    pear, spear = [], []
    for _ in range(DRAWS):
        idx = rng.integers(0, len(design), len(design))     # bootstrap, n = 88
        F = fm.objectives(design.iloc[idx])
        pear.append(abs(float(np.corrcoef(F[:, 0], F[:, 1])[0, 1])))
        spear.append(float(spearmanr(F[:, 0], F[:, 1]).statistic))
    pear, spear = np.asarray(pear), np.asarray(spear)
    return {
        "pearson_on_the_fitting_sample": r_fit,
        "pearson_is_exactly_zero_where_fitted": bool(abs(r_fit) < 1e-10),
        "abs_pearson_at_n88": {
            "median": round(float(np.percentile(pear, 50)), 4),
            "p95": round(float(np.percentile(pear, 95)), 4),
            "max": round(float(pear.max()), 4)},
        "latent_spearman_at_n88": {
            "p05": round(float(np.percentile(spear, 5)), 4),
            "median": round(float(np.percentile(spear, 50)), 4),
            "p95": round(float(np.percentile(spear, 95)), 4),
            "fraction_positive": round(float((spear > 0).mean()), 4),
            "sign_is_a_coin_flip": bool(0.25 < (spear > 0).mean() < 0.75)},
        "criterion_1_would_pass": bool(np.percentile(spear, 95) < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    payload = {
        "purpose": ("stability of screening criterion 1's statistic, resampled at the "
                    "design size the campaign actually measures"),
        "status": "PRE-CONFIRMATORY SCREENING EVIDENCE; no arm involved",
        "draws": DRAWS, "resample_size": N, "resampling": "bootstrap of the 88 design rows",
        "what_this_settles": (
            "the LINEAR association between the two frozen objectives is zero by "
            "construction on the sample the factor model is fitted to, and small but "
            "not exactly zero elsewhere. The RANK association that criterion 1 tests "
            "is not noise: it is stable within a dataset. What it is not is a "
            "consistent DIRECTION -- it is negative on MAGIC, positive on Spambase and "
            "Adult, and a coin flip on Bank Marketing. A criterion requiring it to be "
            "'clearly negative' therefore partitions the panel by the sign of a "
            "monotone-nonlinearity residual around a structural zero."),
        "datasets": {ds: measure(ds, rng) for ds in DATASETS},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("MISMATCH: committed artifact differs from a rebuild")
            return 1
    else:
        OUT.write_text(text)
    print(f"{'dataset':16} {'r on fit set':>13} {'|r| n88 p95':>12} "
          f"{'rho p05':>9} {'rho p50':>9} {'rho p95':>9} {'P(rho>0)':>9} {'C1 passes':>10}")
    for ds, d in payload["datasets"].items():
        s = d["latent_spearman_at_n88"]
        print(f"{ds:16} {d['pearson_on_the_fitting_sample']:13.1e} "
              f"{d['abs_pearson_at_n88']['p95']:12.4f} {s['p05']:9.3f} {s['median']:9.3f} "
              f"{s['p95']:9.3f} {s['fraction_positive']:9.1%} "
              f"{str(d['criterion_1_would_pass']):>10}")
    print(f"\n{'checked' if a.check else 'wrote ' + str(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
