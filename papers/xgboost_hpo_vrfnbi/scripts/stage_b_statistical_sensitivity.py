#!/usr/bin/env python
"""What R = 30 can and cannot resolve, computed from pre-campaign information only.

This is **not** a power analysis used to choose a replication count. R = 30 is
fixed, inherited from the dissertation and from Paper 1. The question here is the
other one: given that R, what is the smallest paired effect the design can
distinguish, and how wide will the intervals be?

Nothing in this script may change a primary endpoint, the comparison family, the
gate thresholds, an effect direction, or the panel. It reads only:

  * the committed Stage A design and external-set evaluations;
  * the dissertation's own replication variability where it is recorded;
  * analytical results that do not depend on any arm outcome.

No arm has been run and none is run here. The paired standard deviations are
obtained by a resampling surrogate: repeatedly split the 166 evaluated
configurations of a dataset into two disjoint halves, score each half's
non-dominated set against the other's as a reference, and take the spread of the
resulting paired indicator differences. That is a *proxy* for between-replication
variability, and it is labelled as one throughout. It is deliberately conservative:
half-sized sets are noisier than the campaign's.

Writes STAGE_B_STATISTICAL_SENSITIVITY.md and audits/stage_b_sensitivity.json.

Usage:  python stage_b_statistical_sensitivity.py [--draws 200]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
PILOT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi"
DATASETS = ["magic", "spambase", "adult", "bank_marketing"]
R = 30

_spec = importlib.util.spec_from_file_location("pilot", HERE / "pilot_stage_a_screening.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

from doe_xgb.reporting import hypervolume, igd_plus, pareto_front   # noqa: E402


def normalized(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (F - lo) / span


def indicators(sub: np.ndarray, ref: np.ndarray, lo, hi) -> dict[str, float]:
    S = normalized(sub[pareto_front(sub)], lo, hi)
    Rf = normalized(ref[pareto_front(ref)], lo, hi)
    ref_pt = np.full(S.shape[1], 1.1)
    hv_ref = hypervolume(Rf, ref_pt)
    return {"hv_ratio": float(hypervolume(S, ref_pt) / hv_ref) if hv_ref > 0 else float("nan"),
            "igd_plus": float(igd_plus(S, Rf))}


def paired_spread(ds: str, draws: int, rng) -> dict:
    """Between-set variability of paired indicator differences, as a proxy."""
    d = pd.read_csv(PILOT / f"{ds}_design.csv")
    v = pd.read_csv(PILOT / f"{ds}_validation_complement.csv")
    both = pd.concat([d, v], ignore_index=True)
    fm = pilot.FactorModel().fit(d)          # fitted on the design, as the protocol requires
    fs = fm.transform(both)
    F = np.column_stack([fs["quality"], fs["cost"]])
    lo, hi = F.min(axis=0), F.max(axis=0)
    n = len(F)
    diffs = {"hv_ratio": [], "igd_plus": []}
    for _ in range(draws):
        perm = rng.permutation(n)
        a, b = F[perm[: n // 2]], F[perm[n // 2:]]
        ref = F                                  # a common reference, as the campaign uses
        ia, ib = indicators(a, ref, lo, hi), indicators(b, ref, lo, hi)
        for k in diffs:
            if np.isfinite(ia[k]) and np.isfinite(ib[k]):
                diffs[k].append(ia[k] - ib[k])
    return {k: float(np.std(np.asarray(vv), ddof=1)) for k, vv in diffs.items() if vv}


def resolution(sd_paired: float) -> dict:
    """What R = 30 resolves for a paired difference with this standard deviation."""
    # two-sided paired t at alpha = 0.05, 80% power: delta = (t_{a/2} + t_{beta}) * sd / sqrt(R)
    dof = R - 1
    t_a = stats.t.ppf(0.975, dof)
    t_b = stats.t.ppf(0.80, dof)
    mde = (t_a + t_b) * sd_paired / np.sqrt(R)
    half_width = t_a * sd_paired / np.sqrt(R)
    # Nadeau and Bengio inflate the variance by (1 + rho/(1-rho)) for overlapping
    # resamples; reported as a sensitivity, not as the primary test.
    out = {"paired_sd_proxy": round(sd_paired, 6),
           "minimum_detectable_paired_difference_80pct": round(float(mde), 6),
           "expected_ci95_half_width": round(float(half_width), 6),
           "standardized_effect_detectable": round(float((t_a + t_b) / np.sqrt(R)), 4)}
    for rho in (0.1, 0.25, 0.5):
        infl = np.sqrt(1.0 + rho / (1.0 - rho))
        out[f"mde_corrected_rho_{rho}"] = round(float(mde * infl), 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    report = {"replications": R, "draws": args.draws,
              "method": ("half-split resampling of the 166 committed Stage A evaluations per "
                         "dataset; a conservative proxy for between-replication variability, "
                         "not a measurement of it"),
              "datasets": {}}
    print(f"What R = {R} resolves, from pre-campaign information only\n")
    print(f"{'dataset':16s}{'indicator':>10}{'paired sd':>11}{'min detectable':>15}"
          f"{'CI95 half-width':>17}{'x sd at rho=0.25':>18}")
    for ds in DATASETS:
        sds = paired_spread(ds, args.draws, rng)
        report["datasets"][ds] = {}
        for k, sd in sds.items():
            res = resolution(sd)
            report["datasets"][ds][k] = res
            print(f"{ds:16s}{k:>10}{sd:11.4f}"
                  f"{res['minimum_detectable_paired_difference_80pct']:15.4f}"
                  f"{res['expected_ci95_half_width']:17.4f}"
                  f"{res['mde_corrected_rho_0.25']:18.4f}")

    # Win-fraction resolution is analytic and needs no simulation.
    from statsmodels.stats.proportion import proportion_confint
    wins = []
    for k in range(R // 2, R + 1):
        lo, _ = proportion_confint(k, R, alpha=0.05, method="wilson")
        if lo > 0.5:
            wins.append(k)
    report["win_fraction"] = {
        "replications": R,
        "minimum_wins_for_wilson_interval_above_half": int(min(wins)) if wins else None,
        "as_fraction": round(min(wins) / R, 4) if wins else None,
        "note": ("the smallest number of wins out of 30 whose Wilson 95% interval excludes "
                 "0.5; below this the win fraction is not evidence of a direction"),
    }
    print(f"\nwin fraction: {report['win_fraction']['minimum_wins_for_wilson_interval_above_half']}"
          f"/{R} ({report['win_fraction']['as_fraction']:.1%}) is the smallest majority whose "
          "Wilson interval excludes one half")

    (OUT / "audits").mkdir(parents=True, exist_ok=True)
    (OUT / "audits" / "stage_b_sensitivity.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'audits'/'stage_b_sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
