#!/usr/bin/env python
"""Emit the authoritative registry of every entity the campaign executes.

The freeze report may not say "four arms" or "five arms". It must list every
executed scientific arm, control and baseline by exact identifier, with what each
one isolates and what it costs. Writing that list by hand is how the two
historical entities came to share the identifier ``HISTORICAL-WS`` in the first
place, so it is derived here from the runner's own stage tuple and budget registry
and will disagree with the code the moment the code changes.

Usage:
    python method_registry.py [--markdown]
"""
from __future__ import annotations

import argparse
import json

from doe_xgb.campaign.runner import (DATASETS, N_REPLICATIONS, NSGA2_GEN, NSGA2_POP,
                                     NSGA2_UNMATCHED_MULTIPLIER, SINGLE_OBJECTIVE,
                                     STAGES, campaign_budget, logical_budget_plan,
                                     method_stage_ledger)

# What each entity exists to isolate. The identifier and the cost come from the
# code; only this sentence is editorial, and it is the reason the entity is in the
# design at all.
ROLE = {
    "HISTORICAL-WS-asrun": ("arm", "historical_ws_asrun",
        "the dissertation's pipeline reproduced bit-faithfully by calling the frozen "
        "v0.1.0-dissertation solver: maximization orientation, uncoded natural-unit "
        "surfaces, observed-extrema normalization, asymmetric 20-point grid with no "
        "pure-quality vertex, and no gate. Reproduced, not repaired."),
    "HISTORICAL-WS": ("arm", "historical_ws",
        "the same weighted sum, solver, symmetric grid and surrogates as WS-S, "
        "differing ONLY in the normalization reference. WS-S to HISTORICAL-WS is "
        "therefore a single-factor contrast on normalization."),
    "WS-S": ("arm", "ws_s",
        "weighted-sum scalarization over the surrogate payoff reference. The "
        "geometry baseline for the primary contrast."),
    "NBI-S": ("arm", "nbi_s",
        "Normal Boundary Intersection on the same surrogates and the same surrogate "
        "reference as WS-S. WS-S to NBI-S isolates front-construction geometry."),
    "NBI-R": ("arm", "nbi_r",
        "NBI with anchors and payoff matrix from direct search on the REAL "
        "objectives. NBI-S to NBI-R isolates anchor and payoff provenance."),
    "ANCHOR-INJECTION-CONTROL": ("control", "anchor_injection_control",
        "NBI-S's own revalidated candidate set augmented with the same empirical "
        "anchors NBI-R receives, changing nothing else. Separates the part of any "
        "NBI-S to NBI-R gap that is set composition from the part that is relocated "
        "geometry. Costs no new real evaluations."),
    "GRID": ("baseline", "direct_baselines",
        "direct-search comparator at the frozen 386-evaluation comparator budget, "
        "which is matched to the most expensive arm (NBI-R) and not pairwise to each arm"),
    "RANDOM": ("baseline", "direct_baselines",
        "direct-search comparator at the frozen 386-evaluation comparator budget, "
        "matched to the most expensive arm and not pairwise to each arm"),
    "BAYES-QUALITY": ("baseline", "direct_baselines",
        "single-objective Bayesian optimization on quality, at the frozen comparator "
        "budget; excluded from front indicators and from the augmented reference"),
    "BAYES-COST": ("baseline", "direct_baselines",
        "single-objective Bayesian optimization on cost; same exclusion"),
    "TPE-QUALITY": ("baseline", "direct_baselines",
        "single-objective TPE on quality; same exclusion"),
    "TPE-COST": ("baseline", "direct_baselines",
        "single-objective TPE on cost; same exclusion"),
    "NSGA2-MATCHED": ("baseline", "direct_baselines",
        f"NSGA-II at {NSGA2_POP} x {NSGA2_GEN} = {NSGA2_POP * NSGA2_GEN} evaluations, "
        "run at the frozen comparator budget to within a disclosed two-evaluation "
        "shortfall (384 against 386); this is a budget-level match to the most "
        "expensive arm, not a pairwise match to NBI-S"),
    "NSGA2-UNMATCHED": ("context", "nsga2_unmatched",
        f"NSGA-II at {NSGA2_UNMATCHED_MULTIPLIER}x the matched budget, one replication "
        "per dataset. A CONTEXT baseline: no fairness claim attaches to it and it "
        "enters no budget-matched comparison."),
}

# Entities that produce no candidate set of their own but are executed stages.
SUPPORTING = {
    "DESIGN": ("shared stage", "design", "88 face-centred central composite runs"),
    "EXTERNAL-VALIDATION-AUDIT": ("shared stage, AUDIT-ONLY", "external_validation",
        "78 points: the design's complementary half fraction plus 14 axial runs at "
        "half radius. Reaches the reliability gate diagnostics and NOTHING else. A "
        "gate failure changes no execution."),
    "SURROGATE-ANCHORS": ("reference construction", "surrogate_anchors",
        "per-objective minimization of the fitted surrogates over the coded box"),
    "EMPIRICAL-ANCHORS": ("reference construction", "empirical_anchors",
        "direct search on the REAL objectives, one budget per objective. Best found "
        "within the declared budget; never described as certified optima."),
    "REFERENCE-CORE": ("reference construction", "reference_core",
        "the reference front used for indicators"),
    "AUGMENTED-REFERENCE": ("reference construction", "augmented_reference",
        "reports self_grading_share_of_front"),
    "HOLDOUT-CONFIRMATION": ("confirmation stage", "holdout_confirmation",
        "selected candidates re-measured on the held-out partition"),
}


def build() -> dict:
    ledger = method_stage_ledger(2)
    standalone = logical_budget_plan(2)["B_total_solution_per_arm"]
    rows = []
    for ident, (kind, stage, isolates) in {**ROLE, **SUPPORTING}.items():
        rows.append({
            "identifier": ident,
            "kind": kind,
            "runner_stage": stage,
            "stage_exists": stage in STAGES,
            "standalone_logical_budget": standalone.get(ident),
            "revalidated_on_real_learner":
                f"{stage}_revalidation" in ledger or kind == "control",
            "isolates": isolates,
        })
    cb = campaign_budget(2)
    return {
        "protocol_tag": "xgboost-hpo-protocol-v3",
        "objectives": 2,
        "datasets": list(DATASETS),
        "replications_per_dataset": N_REPLICATIONS,
        "confirmatory_units": len(DATASETS) * N_REPLICATIONS,
        "counts": {
            "arms": sum(1 for r in rows if r["kind"] == "arm"),
            "controls": sum(1 for r in rows if r["kind"] == "control"),
            "baselines": sum(1 for r in rows if r["kind"] == "baseline"),
            "context_baselines": sum(1 for r in rows if r["kind"] == "context"),
        },
        "single_objective_comparators_excluded_from_front_indicators":
            list(SINGLE_OBJECTIVE),
        "campaign_total_logical": cb["campaign_total_logical"],
        "registry": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    reg = build()
    bad = [r["identifier"] for r in reg["registry"] if not r["stage_exists"]]
    if bad:
        print(f"ERROR: registry names entities with no runner stage: {bad}")
        return 1
    if a.markdown:
        print("| identifier | kind | runner stage | standalone budget | revalidated | isolates |")
        print("|---|---|---|---|---|---|")
        for r in reg["registry"]:
            b = r["standalone_logical_budget"]
            print(f"| `{r['identifier']}` | {r['kind']} | `{r['runner_stage']}` | "
                  f"{b if b is not None else '—'} | "
                  f"{'yes' if r['revalidated_on_real_learner'] else 'n/a'} | {r['isolates']} |")
    else:
        print(json.dumps(reg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
