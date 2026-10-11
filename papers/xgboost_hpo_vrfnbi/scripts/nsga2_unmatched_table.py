#!/usr/bin/env python
"""The unmatched NSGA-II run, scored and tabulated (Supplement S13.2).

protocol/NSGA2_DECISION.md section 5 declares a secondary, unmatched-context NSGA-II
run at ten times the matched budget on one replication per dataset, reported in its
own table with the generation count stated. Its asked question is whether a weak
matched result is explained by budget starvation. The campaign stored the run's rows
(rep_00/nsga2_unmatched.json) but, by design, scored it nowhere: it enters no
budget-matched comparison and neither reference. This script scores those stored rows
against that replication's CORE reference with the frozen indicator code
(doe_xgb.campaign.scoring.indicators: dominance filter, normalization box spanned by
the CORE front, reference point 1.1 per normalized axis), which is how the campaign
scored matched NSGA-II, GRID and RANDOM from their whole stored archives. Matched
NSGA-II's value in the same replication is read from analysis/indicators_long.csv.

Provenance label:
retrospective (declared in protocol/NSGA2_DECISION.md, section 5; computed after the campaign from stored outputs)

The scoring path is validated before use, per dataset: the CORE front is rebuilt from
the stored design and anchor-search rows with the frozen reference_core() and must equal
the stored front, and matched NSGA-II, GRID and RANDOM rescored from their stored rows
must reproduce the stored metrics and the committed indicators_long.csv values. The
script stops on any disagreement.

n_front is the frozen code's count of non-dominated rows. It keeps repeated rows: a
configuration requested more than once, or distinct configurations sharing one
objective vector. Distinct counts are recorded under "diagnostics".

Inputs (read only), under experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_00/:
  nsga2_unmatched.json, reference_core.json, design.json, empirical_anchors.json,
  direct_baselines.json, metrics_by_method.json, metrics.json; and
  papers/xgboost_hpo_vrfnbi/analysis/indicators_long.csv,
  papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json.
Output: papers/xgboost_hpo_vrfnbi/analysis/nsga2_unmatched_table.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:  python papers/xgboost_hpo_vrfnbi/scripts/nsga2_unmatched_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402
from doe_xgb.campaign.runner import (DATASETS, NSGA2_GEN, NSGA2_POP,  # noqa: E402
                                     NSGA2_UNMATCHED_MULTIPLIER,
                                     NSGA2_UNMATCHED_REPLICATION)
from doe_xgb.campaign.scoring import (HV_REFERENCE_COORDINATE,  # noqa: E402
                                      canonical_key, indicators, reference_core)
from doe_xgb.reporting import dominance_filter  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
ANALYSIS = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis"
OUT = ANALYSIS / "nsga2_unmatched_table.json"
LABEL = ("retrospective (declared in protocol/NSGA2_DECISION.md, section 5; computed "
         "after the campaign from stored outputs)")
TOL = 1e-12        # the CSV round-trip of a float costs at most a few ulps
CHECKED = ("hv_ratio", "igd_plus", "gd", "joint_nondominated_fraction", "n_front")


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def main() -> int:
    rep = NSGA2_UNMATCHED_REPLICATION
    gens = NSGA2_GEN * NSGA2_UNMATCHED_MULTIPLIER
    ind = pd.read_csv(ANALYSIS / "indicators_long.csv")
    datasets, diagnostics, max_dev = {}, {}, 0.0
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        u = ROOT / ds / f"rep_{rep:02d}"
        load = lambda name: json.loads((u / name).read_text())  # noqa: E731

        # ---- validate the scoring path on what the campaign itself scored -----
        stored = load("reference_core.json")
        core_front = np.asarray(stored["front"], dtype=float)
        rebuilt = reference_core([pd.DataFrame(load("design.json")["rows"]),
                                  pd.DataFrame(load("empirical_anchors.json")["rows"])],
                                 fm.objectives)
        if rebuilt["front"].shape != core_front.shape or \
                not np.max(np.abs(rebuilt["front"] - core_front)) <= TOL:
            raise SystemExit(f"{ds}: CORE front rebuilt from stored rows differs")
        mbm = load("metrics_by_method.json")["methods"]
        db = load("direct_baselines.json")["methods"]
        for key, entity in (("nsga2", "NSGA2-MATCHED"), ("grid", "GRID"),
                            ("random", "RANDOM")):
            mine = indicators(fm.objectives(pd.DataFrame(db[key]["rows"])), core_front)
            row = ind[(ind.dataset == ds) & (ind.entity == entity)
                      & (ind.replication == rep)]
            if len(row) != 1:
                raise SystemExit(f"{ds}: {len(row)} indicators_long rows for {entity}")
            for k in CHECKED:
                for ref_val in (mbm[key]["core"][k], row[f"core__{k}"].iloc[0]):
                    dev = abs(float(mine[k]) - float(ref_val))
                    max_dev = max(max_dev, dev)
                    if not dev <= TOL:
                        raise SystemExit(f"{ds} {entity} {k}: rescored {mine[k]} vs "
                                         f"stored {ref_val}")
        matched = ind[(ind.dataset == ds) & (ind.entity == "NSGA2-MATCHED")
                      & (ind.replication == rep)].iloc[0]

        # ---- the unmatched run -------------------------------------------------
        un = load("nsga2_unmatched.json")
        ledger = {m["method"]: m for m in load("metrics.json")["accounting"]["per_method"]}
        led = ledger["nsga2_unmatched"]
        n_logical = int(un["evaluations"])
        if not (int(un["population"]) == NSGA2_POP and int(un["generations"]) == gens
                and n_logical == NSGA2_POP * gens == int(un["n_rows"]) == len(un["rows"])
                == int(led["logical_evaluations"])
                == int(led["by_stage"]["unmatched_context"]["logical"])):
            raise SystemExit(f"{ds}: unmatched run accounting does not reconcile")
        F = fm.objectives(pd.DataFrame(un["rows"]))
        if not np.isfinite(F).all():
            raise SystemExit(f"{ds}: non-finite objective in the unmatched rows")
        res = indicators(F, core_front)
        nd = dominance_filter(F)
        datasets[ds] = {"replication": int(rep),
                        "logical_evaluations": n_logical,
                        "population": int(un["population"]),
                        "generations": int(un["generations"]),
                        "core_hv_ratio": float(res["hv_ratio"]),
                        "matched_core_hv_ratio": float(matched["core__hv_ratio"]),
                        "n_front": int(res["n_front"])}
        Fm = fm.objectives(pd.DataFrame(db["nsga2"]["rows"]))
        ndm = dominance_filter(Fm)
        diagnostics[ds] = {
            "unique_configurations_requested": int(led["unique_logical_evaluations"]),
            "n_front_distinct_configurations": len({canonical_key(un["rows"][i]) for i in nd}),
            "n_front_distinct_objective_vectors": len({tuple(F[i]) for i in nd}),
            "matched_n_front": int(matched["core__n_front"]),
            "matched_n_front_distinct_objective_vectors": len({tuple(Fm[i]) for i in ndm}),
            "core_front_points": int(len(core_front))}

    unit = f"experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_{rep:02d}/"
    out = {
        "script": rel(Path(__file__).resolve()),
        "label": LABEL,
        "inputs": [unit + n for n in ("nsga2_unmatched.json", "reference_core.json",
                                      "design.json", "empirical_anchors.json",
                                      "direct_baselines.json", "metrics_by_method.json",
                                      "metrics.json")]
                  + ["papers/xgboost_hpo_vrfnbi/analysis/indicators_long.csv",
                     "papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json"],
        "definitions": {
            "core_hv_ratio": ("doe_xgb.campaign.scoring.indicators on every stored row of "
                              "the unmatched run against the replication's stored CORE "
                              "front: dominance filter, normalization box spanned by the "
                              "CORE front, reference point "
                              f"{HV_REFERENCE_COORDINATE} per normalized axis; as the "
                              "campaign scored matched NSGA-II, GRID and RANDOM"),
            "matched_core_hv_ratio": ("NSGA2-MATCHED core__hv_ratio, same dataset and "
                                      "replication, from analysis/indicators_long.csv"),
            "logical_evaluations": ("real evaluations requested by the run (population "
                                    "x generations), reconciled against the unit's "
                                    "cache ledger"),
            "n_front": ("frozen-code count of non-dominated rows; repeated rows are "
                        "kept, see diagnostics for distinct counts")},
        "checks": {
            "core_front_rebuilt_from_stored_rows": "equal to the stored front, all datasets",
            "rescored_entities": ["NSGA2-MATCHED", "GRID", "RANDOM"],
            "rescored_indicators": list(CHECKED),
            "max_abs_difference_rescored_vs_stored": max_dev},
        "diagnostics": diagnostics,
        "datasets": datasets,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"UNMATCHED NSGA-II, replication {rep}: CORE-relative HV ratio\n")
    for ds in DATASETS:
        v, d = datasets[ds], diagnostics[ds]
        print(f"  {ds:15s} {v['logical_evaluations']:,} evals ({v['population']} x "
              f"{v['generations']})  unmatched {v['core_hv_ratio']:.4f}  matched "
              f"{v['matched_core_hv_ratio']:.4f}  n_front {v['n_front']} "
              f"({d['n_front_distinct_objective_vectors']} distinct)")
    print(f"\nscoring path reproduces stored values to {max_dev:.1e}")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
