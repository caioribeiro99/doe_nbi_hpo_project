#!/usr/bin/env python
"""Front-level weighting sensitivity: variance-weighted vs equal-weighted quality (S4.5).

Purpose. EXPERIMENT_PROTOCOL.md section 11 item 1 declares the aggregation
sensitivity: how often the variance-weighted quality composite (primary, section 6.3)
and the equal-weighted one (the unweighted mean of the z-scored quality factor
scores, ``quality_equal`` of the frozen factor model) disagree about the RETURNED
FRONT. Its decided reading: if the two weightings return different fronts in more
than 20% of replications on a dataset, that dataset's conclusions are reported as
conditional on the weighting, in the abstract as well. "Different fronts" means the
two returned non-dominated sets differ by at least one member after deduplication
at the evaluation tolerance.

Operationalization (post hoc; the protocol fixed none of the following):
* No arm is re-solved. Each arm's stored real-revalidated candidates
  (``candidate_revalidation.json``) are rescored with the frozen per-dataset model
  under the two composites, the cost objective unchanged; the non-dominated set of
  each weighting is taken with the frozen ``dominance_filter`` and the two sets of
  members are compared.
* Deduplication at the evaluation tolerance: one member per evaluation cache key
  (``scoring.canonical_key``: integers rounded, floats to 12 decimals), and one
  member per distinct stored evaluation, so configurations whose seven stored
  responses are identical (for example, ``colsample_bylevel`` varied where
  ``colsample_bytree`` leaves a single feature) count once, represented by the
  first in stored order. They are the same point in either objective space. Kept
  apart, the batched factor transform can score two copies a few units in the last
  place apart, so that one dominates the other under one weighting and not under
  the other; the script prints every verdict the collapse changes.
* The 20% rule is counted per arm (HISTORICAL-WS, WS-S, NBI-S, NBI-R) and read as
  met on a dataset if it is met for ANY of the four arms (``rule_fires``: some arm's
  share of replications with different fronts exceeds 0.20).
* Contrasts under equal weighting: the CORE reference (88 design rows plus the 200
  anchor-search rows, ``reference_core``) is rescored with the equal-weight composite
  and the three identifying contrasts are recomputed on the frozen CORE-relative
  hypervolume ratio (``scoring.indicators``, on each arm's full stored set exactly as
  the campaign scored it): median paired difference (second minus first, positive
  favours the second arm) and wins/ties/losses with an exact-zero tie, the
  convention of ``primary_analysis.py``. Descriptive; no test.

The script first reproduces, from the same stored rows, every stored CORE front and
every stored variance-weighted CORE hypervolume ratio of the four arms, and stops if
any differs.

Provenance label: retrospective (declared in the frozen protocol, section 11 item 1;
computed after the campaign from stored outputs); the rescoring operationalization is
post hoc.

Inputs (repo-relative), for every <dataset> and rep_XX under
experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/:
    candidate_revalidation.json, design.json, empirical_anchors.json,
    factor_model.json, reference_core.json, metrics_by_method.json
and papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json.
Output:
    papers/xgboost_hpo_vrfnbi/analysis/weighting_sensitivity_fronts.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage, from the repository root:
    python -B papers/xgboost_hpo_vrfnbi/scripts/weighting_sensitivity_fronts.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.evaluator import RESPONSES  # noqa: E402
from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402
from doe_xgb.campaign.scoring import canonical_key, indicators, reference_core  # noqa: E402
from doe_xgb.reporting import dominance_filter  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis" / "weighting_sensitivity_fronts.json"
SCRIPT = pathlib.Path(__file__).resolve().relative_to(REPO).as_posix()
LABEL = ("retrospective (declared in the frozen protocol, section 11 item 1; computed "
         "after the campaign from stored outputs); the rescoring operationalization is "
         "post hoc")
RULE_FRACTION = 0.20      # EXPERIMENT_PROTOCOL.md section 11 item 1, DECIDED
# Arm label -> method key in the unit artifacts, as aggregate_campaign.py maps them.
ARMS = {"HISTORICAL-WS": "historical_ws", "WS-S": "ws_s", "NBI-S": "nbi_s",
        "NBI-R": "nbi_r"}
# The primary family's three identifying contrasts, in primary_analysis.py's order.
CONTRASTS = (("HISTORICAL-WS", "WS-S"), ("WS-S", "NBI-S"), ("NBI-S", "NBI-R"))


def load(unit: pathlib.Path, stage: str) -> dict:
    return json.loads((unit / f"{stage}.json").read_text())


def members(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """Deduplicate at the evaluation tolerance, keeping the first in stored order.

    A row is a new member only if both its evaluation cache key and its seven stored
    responses are new. Returns the member rows and each member's canonical key.
    """
    seen_key, seen_eval, keep, ids = set(), set(), [], []
    for r in df.to_dict("records"):
        key = canonical_key(r)
        evaluation = tuple(float(r[c]) for c in RESPONSES)
        new = key not in seen_key and evaluation not in seen_eval
        seen_key.add(key)
        seen_eval.add(evaluation)
        keep.append(new)
        if new:
            ids.append(key)
    return df[np.asarray(keep, dtype=bool)].reset_index(drop=True), ids


def same_point_set(a: np.ndarray, b: np.ndarray) -> bool:
    return ({tuple(map(float, r)) for r in np.asarray(a)}
            == {tuple(map(float, r)) for r in np.asarray(b)})


def main() -> int:
    out = {"script": SCRIPT, "label": LABEL, "datasets": {}}
    max_hv_dev, n_checked, n_collapsed, changed = 0.0, 0, 0, []
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        variance = fm.objectives                                     # (quality, cost)

        def equal(df: pd.DataFrame) -> np.ndarray:                  # (quality_equal, cost)
            t = fm.transform(df)
            return np.column_stack([t["quality_equal"], t["cost"]])

        differs = {a: 0 for a in ARMS}
        hv_equal = {a: [] for a in ARMS}
        for rep in range(N_REPLICATIONS):
            unit = ROOT / ds / f"rep_{rep:02d}"
            if not np.array_equal(np.asarray(load(unit, "factor_model")["rotated_loadings"]),
                                  fm.rotated_loadings):
                raise SystemExit(f"{ds} rep {rep}: unit did not apply the frozen model")
            frames = [pd.DataFrame(load(unit, "design")["rows"]),
                      pd.DataFrame(load(unit, "empirical_anchors")["rows"])]
            core_v = reference_core(frames, variance)["front"]
            core_e = reference_core(frames, equal)["front"]
            if not same_point_set(core_v, load(unit, "reference_core")["front"]):
                raise SystemExit(f"{ds} rep {rep}: stored CORE front not reproduced")
            reval = load(unit, "candidate_revalidation")["arms"]
            stored = load(unit, "metrics_by_method")["methods"]
            for arm, key in ARMS.items():
                df = pd.DataFrame(reval[key]["rows"])
                mdf, ids = members(df)
                front_v = {ids[i] for i in dominance_filter(variance(mdf))}
                front_e = {ids[i] for i in dominance_filter(equal(mdf))}
                differs[arm] += int(front_v != front_e)
                # Hypervolume on the full stored set, exactly as the campaign scored it.
                Fv, Fe = variance(df), equal(df)
                dev = abs(indicators(Fv, core_v)["hv_ratio"] - stored[key]["core"]["hv_ratio"])
                max_hv_dev, n_checked = max(max_hv_dev, dev), n_checked + 1
                hv_equal[arm].append(indicators(Fe, core_e)["hv_ratio"])
                # Diagnostic only: the verdict with configurations kept apart.
                if len(mdf) < len(df):
                    n_collapsed += 1
                    keys = [canonical_key(r) for r in df.to_dict("records")]
                    raw = ({keys[i] for i in dominance_filter(Fv)}
                           != {keys[i] for i in dominance_filter(Fe)})
                    if raw != (front_v != front_e):
                        changed.append(f"{ds} rep_{rep:02d} {arm}: differs {raw} -> "
                                       f"{front_v != front_e}")
        if max_hv_dev > 1e-12:
            raise SystemExit(f"{ds}: stored CORE hypervolume ratios not reproduced "
                             f"(max deviation {max_hv_dev:.3g})")

        contrasts = {}
        for a, b in CONTRASTS:
            d = np.asarray(hv_equal[b], dtype=float) - np.asarray(hv_equal[a], dtype=float)
            contrasts[f"{a} -> {b}"] = {"median_diff": float(np.median(d)),
                                        "wins": int((d > 0).sum()),
                                        "ties": int((d == 0).sum()),
                                        "losses": int((d < 0).sum())}
        out["datasets"][ds] = {
            "n_replications": N_REPLICATIONS,
            "arms": {a: {"replications_front_differs": int(n)} for a, n in differs.items()},
            "rule_fires": bool(any(n / N_REPLICATIONS > RULE_FRACTION
                                   for n in differs.values())),
            "contrasts_equal_weight": contrasts}

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"reproduced {n_checked} stored CORE hv_ratio values (max deviation "
          f"{max_hv_dev:.3g}) and {N_REPLICATIONS * len(DATASETS)} stored CORE fronts")
    print(f"{n_collapsed} arm-units hold distinct configurations with identical stored "
          f"evaluations; verdicts the collapse changes: {changed or 'none'}\n")
    for ds in DATASETS:
        v = out["datasets"][ds]
        arms = "  ".join(f"{a} {v['arms'][a]['replications_front_differs']:2}/"
                         f"{N_REPLICATIONS}" for a in ARMS)
        print(f"{ds:15} fronts differ: {arms}  rule fires: {v['rule_fires']}")
        for c, r in v["contrasts_equal_weight"].items():
            print(f"{'':15}   {c:24} median {r['median_diff']:+.4f}  "
                  f"W/T/L {r['wins']}/{r['ties']}/{r['losses']}")
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
