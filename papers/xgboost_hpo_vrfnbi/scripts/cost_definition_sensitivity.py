#!/usr/bin/env python
"""Cost-definition sensitivity: winning arm and contrasts, leaf-count cost vs measured time (S4.7).

Purpose. EXPERIMENT_PROTOCOL.md section 11 item 4 (with section 6.1) declares that the
paper reports whether the primary cost objective (the leaf-count factor, f_cost) and
the secondary cost quantity (measured mean per-fold wall-clock time,
``Time_MeanFold``) disagree about the winning arm, as a frequency over replications.
The three identifying contrasts are tabulated under both definitions beside it.

Operationalization (post hoc; the protocol fixed none of it):
* In each replication the winning arm is the one of HISTORICAL-WS, WS-S, NBI-S and
  NBI-R with the largest CORE-relative hypervolume ratio, computed by the frozen
  ``scoring.indicators`` on the arm's stored real-revalidated candidates.
* Leaf-count definition: the frozen objectives (quality composite, f_cost) and the
  frozen CORE reference, exactly as the campaign scored them; the script stops unless
  it reproduces every stored CORE front and CORE hypervolume ratio of the four arms.
* Time definition: the cost axis of every candidate and of the CORE rows (88 design
  rows plus the 200 anchor-search rows) is the measured ``Time_MeanFold`` in seconds,
  untransformed; the CORE front is rebuilt in that space with the frozen
  ``reference_core``. The quality objective (variance-weighted composite) is unchanged.
* Arms whose ratio is within 1e-12 of the largest share the win, labelled with the
  tied arms in protocol order joined by "/" and suffixed " (tied)". The tolerance
  absorbs floating-point noise only: an arm can return a configuration whose stored
  responses equal those of a CORE row, and the batched factor transform then scores
  the two copies a few units in the last place apart. ``winner_differs`` counts
  replications whose winner, or set of tied winners, differs between the two
  definitions.
* Contrasts (``contrasts.leaf`` and ``contrasts.time``): the three identifying
  contrasts, from the same per-arm ratios as the winners: median paired difference
  (second arm minus first; positive favours the second) and wins/ties/losses, with a
  difference of absolute value at most 1e-12 counted as a tie. The frozen primary
  analysis counts only an exact zero as a tie; with that rule the leaf contrasts must
  reproduce ``analysis/primary_analysis.json`` (CORE reference), W/T/L exactly and
  medians to 1e-12 (that file was computed from ratios read back from
  ``indicators_long.csv``, whose float parsing can move a value by one unit in the
  last place). The script asserts both and prints every pair the 1e-12 rule
  classifies differently.
* The script checks that every non-zero winner gap and paired difference is either
  below 1e-13 or above 1e-6, so that any tolerance in that range gives the same
  classification, and stops otherwise. The ``definitions`` block of the output states
  these rules.
No arm was re-run; the timings are those stored with each evaluation, taken with
several worker processes in parallel, and carry load-dependent noise.

Provenance label: post hoc (operationalization of an analysis declared in the frozen
protocol, section 11 item 4).

Inputs (repo-relative), for every <dataset> and rep_XX under
experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/:
    candidate_revalidation.json, design.json, empirical_anchors.json,
    factor_model.json, reference_core.json, metrics_by_method.json
and papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json;
papers/xgboost_hpo_vrfnbi/analysis/primary_analysis.json for the reproduction check.
Output:
    papers/xgboost_hpo_vrfnbi/analysis/cost_definition_sensitivity.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage, from the repository root:
    python -B papers/xgboost_hpo_vrfnbi/scripts/cost_definition_sensitivity.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402
from doe_xgb.campaign.scoring import indicators, reference_core  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
ANALYSIS = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis"
OUT = ANALYSIS / "cost_definition_sensitivity.json"
SCRIPT = pathlib.Path(__file__).resolve().relative_to(REPO).as_posix()
LABEL = ("post hoc (operationalization of an analysis declared in the frozen protocol, "
         "section 11 item 4)")
# Arm label -> method key in the unit artifacts, in protocol order.
ARMS = {"HISTORICAL-WS": "historical_ws", "WS-S": "ws_s", "NBI-S": "nbi_s",
        "NBI-R": "nbi_r"}
# The primary family's three identifying contrasts, in primary_analysis.py's order.
CONTRASTS = (("HISTORICAL-WS", "WS-S"), ("WS-S", "NBI-S"), ("NBI-S", "NBI-R"))
TIME = "Time_MeanFold"
TIE_TOLERANCE = 1e-12     # on the hypervolume ratio; floating-point noise only
NOISE_CEILING, GAP_FLOOR = 1e-13, 1e-6   # the band no gap or difference may fall in
MEDIAN_TOL = 1e-12        # leaf medians vs the committed primary analysis
DEFINITIONS = {
    "contrast_order": [f"{a} -> {b}" for a, b in CONTRASTS],
    "difference": ("per replication, the CORE-relative hypervolume ratio of the second "
                   "arm minus that of the first; positive favours the second arm"),
    "tie_rule": ("a paired difference of absolute value at most 1e-12 is a tie, and arms "
                 "whose ratio is within 1e-12 of the largest share the win, labelled "
                 "'A/B (tied)'; no non-zero gap or difference lies between 1e-13 and "
                 "1e-6 (checked). The frozen primary analysis counts only an exact zero "
                 "as a tie; under that rule the leaf contrasts reproduce "
                 "analysis/primary_analysis.json (checked: W/T/L exactly, medians to "
                 "1e-12)"),
    "leaf": ("cost axis f_cost, the frozen leaf-count factor, against the stored CORE "
             "reference: the primary objective space"),
    "time": ("cost axis Time_MeanFold in seconds, untransformed, with the CORE front "
             "rebuilt in that space from the 88 design rows and the 200 anchor-search "
             "rows"),
    "quality": "the frozen variance-weighted quality composite under both definitions",
    "winner": ("the arm of HISTORICAL-WS, WS-S, NBI-S and NBI-R with the largest "
               "CORE-relative hypervolume ratio"),
}


def load(unit: pathlib.Path, stage: str) -> dict:
    return json.loads((unit / f"{stage}.json").read_text())


def same_point_set(a: np.ndarray, b: np.ndarray) -> bool:
    return ({tuple(map(float, r)) for r in np.asarray(a)}
            == {tuple(map(float, r)) for r in np.asarray(b)})


def winner(hv: dict[str, float], gaps: list[float]) -> str:
    """The arm with the largest ratio; arms within the tie tolerance share the win.

    Appends every non-zero gap to the largest ratio to ``gaps`` for the check that
    the tie rule does not depend on the tolerance chosen.
    """
    best = max(hv.values())
    gaps.extend(best - v for v in hv.values() if v != best)
    tied = [a for a in ARMS if hv[a] >= best - TIE_TOLERANCE]
    return tied[0] if len(tied) == 1 else "/".join(tied) + " (tied)"


def paired(hv: dict[str, list[float]], a: str, b: str, tol: float) -> tuple[np.ndarray, dict]:
    """Second minus first over replications, with ties at ``|d| <= tol``."""
    d = np.asarray(hv[b], dtype=float) - np.asarray(hv[a], dtype=float)
    return d, {"median_diff": float(np.median(d)), "wins": int((d > tol).sum()),
               "ties": int((np.abs(d) <= tol).sum()), "losses": int((d < -tol).sum())}


def committed_primary(ds: str) -> dict[str, dict]:
    """The committed CORE-reference contrasts of primary_analysis.py, by contrast name."""
    pa = json.loads((ANALYSIS / "primary_analysis.json").read_text())
    rows = pa["primary_family"]["core"].get(ds) or pa["boundary_control"]["core"][ds]
    return {r["contrast"]: r for r in rows}


def main() -> int:
    out = {"script": SCRIPT, "label": LABEL, "definitions": DEFINITIONS, "datasets": {}}
    max_hv_dev, n_checked, gaps, rule_differs, max_median_dev = 0.0, 0, [], [], 0.0
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        leaf = fm.objectives                                     # (quality, f_cost)

        def time_cost(df: pd.DataFrame) -> np.ndarray:          # (quality, seconds)
            return np.column_stack([fm.transform(df)["quality"],
                                    df[TIME].to_numpy(dtype=float)])

        winners_leaf, winners_time = [], []
        series = {"leaf": {a: [] for a in ARMS}, "time": {a: [] for a in ARMS}}
        for rep in range(N_REPLICATIONS):
            unit = ROOT / ds / f"rep_{rep:02d}"
            if not np.array_equal(np.asarray(load(unit, "factor_model")["rotated_loadings"]),
                                  fm.rotated_loadings):
                raise SystemExit(f"{ds} rep {rep}: unit did not apply the frozen model")
            frames = [pd.DataFrame(load(unit, "design")["rows"]),
                      pd.DataFrame(load(unit, "empirical_anchors")["rows"])]
            core_leaf = reference_core(frames, leaf)["front"]
            core_time = reference_core(frames, time_cost)["front"]
            if not same_point_set(core_leaf, load(unit, "reference_core")["front"]):
                raise SystemExit(f"{ds} rep {rep}: stored CORE front not reproduced")
            reval = load(unit, "candidate_revalidation")["arms"]
            stored = load(unit, "metrics_by_method")["methods"]
            hv_leaf, hv_time = {}, {}
            for arm, key in ARMS.items():
                df = pd.DataFrame(reval[key]["rows"])
                hv_leaf[arm] = indicators(leaf(df), core_leaf)["hv_ratio"]
                hv_time[arm] = indicators(time_cost(df), core_time)["hv_ratio"]
                dev = abs(hv_leaf[arm] - stored[key]["core"]["hv_ratio"])
                max_hv_dev, n_checked = max(max_hv_dev, dev), n_checked + 1
            if not all(np.isfinite(v) for v in (*hv_leaf.values(), *hv_time.values())):
                raise SystemExit(f"{ds} rep {rep}: undefined hypervolume ratio")
            winners_leaf.append(winner(hv_leaf, gaps))
            winners_time.append(winner(hv_time, gaps))
            for arm in ARMS:
                series["leaf"][arm].append(hv_leaf[arm])
                series["time"][arm].append(hv_time[arm])
        if max_hv_dev > 1e-12:
            raise SystemExit(f"{ds}: stored CORE hypervolume ratios not reproduced "
                             f"(max deviation {max_hv_dev:.3g})")

        contrasts = {"leaf": {}, "time": {}}
        committed = committed_primary(ds)
        for a, b in CONTRASTS:
            name = f"{a} -> {b}"
            for definition in ("leaf", "time"):
                d, stats = paired(series[definition], a, b, TIE_TOLERANCE)
                contrasts[definition][name] = stats
                gaps.extend(np.abs(d[d != 0]).tolist())
            # The leaf axis is the primary space: with the frozen exact-zero tie it must
            # reproduce the committed primary analysis. W/T/L must match exactly. The
            # medians agree to rounding only: primary_analysis.py read the per-unit
            # ratios back from indicators_long.csv, whose float parsing can move a
            # value by one unit in the last place.
            _, exact = paired(series["leaf"], a, b, 0.0)
            c = committed[name]
            median_dev = abs(exact["median_diff"] - c["median_diff"])
            max_median_dev = max(max_median_dev, median_dev)
            if (median_dev > MEDIAN_TOL
                    or (exact["wins"], exact["ties"], exact["losses"])
                    != (c["wins_for_second"], c["ties"], c["losses"])):
                raise SystemExit(f"{ds} {name}: committed primary contrast not reproduced")
            mine = contrasts["leaf"][name]
            if (mine["wins"], mine["ties"], mine["losses"]) != (
                    exact["wins"], exact["ties"], exact["losses"]):
                rule_differs.append(
                    f"{ds} {name}: primary_analysis.json W/T/L {exact['wins']}/"
                    f"{exact['ties']}/{exact['losses']} (exact-zero ties), here "
                    f"{mine['wins']}/{mine['ties']}/{mine['losses']} (1e-12 ties)")
        out["datasets"][ds] = {
            "n_replications": N_REPLICATIONS,
            "winner_differs": int(sum(a != b for a, b in zip(winners_leaf, winners_time))),
            "winner_counts_leaf": dict(Counter(winners_leaf)),
            "winner_counts_time": dict(Counter(winners_time)),
            "contrasts": contrasts}

    noise = [g for g in gaps if g <= TIE_TOLERANCE]
    real = [g for g in gaps if g > TIE_TOLERANCE]
    if max(noise, default=0.0) >= NOISE_CEILING or min(real, default=1.0) <= GAP_FLOOR:
        raise SystemExit(f"a winner gap or paired difference lies between "
                         f"{NOISE_CEILING:g} and {GAP_FLOOR:g}; the tie rule would depend "
                         f"on the tolerance chosen")

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"reproduced {n_checked} stored CORE hv_ratio values under the leaf-count "
          f"definition (max deviation {max_hv_dev:.3g}); with exact-zero ties, every "
          f"committed W/T/L of analysis/primary_analysis.json exactly and its medians "
          f"to {max_median_dev:.3g}")
    print(f"winner gaps and paired differences: {len(noise)} at floating-point noise "
          f"(largest {max(noise, default=0.0):.3g}, treated as ties); smallest other "
          f"{min(real, default=float('nan')):.3g}")
    print(f"pairs the 1e-12 tie rule classifies differently from the frozen exact-zero "
          f"rule: {rule_differs or 'none'}\n")
    tally = lambda c: ", ".join(f"{k} {n}" for k, n in  # noqa: E731
                                sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    for ds in DATASETS:
        v = out["datasets"][ds]
        print(f"{ds:15} winner differs {v['winner_differs']:2}/{v['n_replications']}")
        print(f"{'':15}   leaf: {tally(v['winner_counts_leaf'])}")
        print(f"{'':15}   time: {tally(v['winner_counts_time'])}")
        for name in DEFINITIONS["contrast_order"]:
            cell = {k: v["contrasts"][k][name] for k in ("leaf", "time")}
            print(f"{'':15}   {name:24} " + "  ".join(
                f"{k} {s['median_diff']:+.4f} ({s['wins']}/{s['ties']}/{s['losses']})"
                for k, s in cell.items()))
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
