#!/usr/bin/env python
"""Aggregate the 120 confirmatory units into one analysis artifact.

Reads only what the campaign wrote. Computes nothing new about the science: every
indicator was computed inside its own unit, against that unit's own references, by
the frozen code. This collects them.

The frozen conventions this respects, from PROTOCOL_V3_FREEZE_REPORT.md:
  * the CORE reference carries the primary family; AUGMENTED is a mandatory sensitivity;
  * the primary indicator is hv_ratio;
  * the primary panel is MAGIC, Adult, Bank Marketing; Spambase is a boundary control;
  * dataset is the generalization unit -- no pooling across datasets.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.runner import (BOUNDARY_CONTROLS, DATASET_ROLES, DATASETS,  # noqa: E402
                                     N_REPLICATIONS, PRIMARY_GEOMETRY_PANEL,
                                     PRIMARY_INDICATOR, PRIMARY_REFERENCE,
                                     SINGLE_OBJECTIVE)

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis"

# Arm identifier -> the method key the metrics stage uses.
ARMS = {"HISTORICAL-WS-asrun": "historical_ws_asrun", "HISTORICAL-WS": "historical_ws",
        "WS-S": "ws_s", "NBI-S": "nbi_s", "NBI-R": "nbi_r"}
CONTROL = {"ANCHOR-INJECTION-CONTROL": "nbi_s_plus_anchors"}
BASELINES = {"GRID": "grid", "RANDOM": "random", "NSGA2-MATCHED": "nsga2"}
INDICATORS = ("hv_ratio", "igd_plus", "gd", "spacing", "spacing_cv",
              "joint_nondominated_fraction", "n_front")


def collect() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for rep in range(N_REPLICATIONS):
            u = ROOT / ds / f"rep_{rep:02d}"
            mbm = json.loads((u / "metrics_by_method.json").read_text())
            ext = json.loads((u / "external_validation.json").read_text())
            core = json.loads((u / "reference_core.json").read_text())
            aug = json.loads((u / "augmented_reference.json").read_text())
            gate = ext["gate_pass"]
            for label, key in {**ARMS, **CONTROL, **BASELINES}.items():
                if key not in mbm["methods"]:
                    continue
                blk = mbm["methods"][key]
                row = {"dataset": ds, "role": DATASET_ROLES[ds], "replication": rep,
                       "entity": label, "method_key": key,
                       "gate_quality": bool(gate[0]), "gate_cost": bool(gate[1]),
                       "gate_both": bool(gate[0] and gate[1]),
                       "core_n_points": core["n_points"],
                       "core_n_front": core["n_nondominated"],
                       "aug_n_points": aug["n_points"],
                       "self_grading_share":
                           aug.get("self_grading_share_of_front", {}).get(key)}
                for ref in ("core", "augmented"):
                    for ind in INDICATORS:
                        row[f"{ref}__{ind}"] = blk[ref].get(ind)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "indicators_long.csv", index=False)

    n_units = df.groupby("dataset")["replication"].nunique().to_dict()
    meta = {
        "protocol_tag": "xgboost-hpo-protocol-v3",
        "source_commit": json.loads(
            (ROOT / "campaign_manifest.json").read_text())["source_commit"],
        "units_per_dataset": n_units,
        "entities": sorted(df["entity"].unique()),
        "primary_indicator": PRIMARY_INDICATOR,
        "primary_reference": PRIMARY_REFERENCE,
        "primary_geometry_panel": list(PRIMARY_GEOMETRY_PANEL),
        "boundary_controls": list(BOUNDARY_CONTROLS),
        "single_objective_excluded_from_front_indicators": list(SINGLE_OBJECTIVE),
        "rows": int(len(df)),
        "note": ("every indicator was computed inside its own unit against that "
                 "unit's own references by the frozen code; this file collects them "
                 "and computes nothing new"),
    }
    (OUT / "aggregate_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    print(f"rows {len(df)}   entities {len(df['entity'].unique())}   "
          f"units/dataset {n_units}")
    print(f"\ngate pass rate (quality) by dataset, over {N_REPLICATIONS} replications:")
    g = (df[df["entity"] == "NBI-S"]
         .groupby("dataset")[["gate_quality", "gate_cost", "gate_both"]].mean())
    for ds in DATASETS:
        r = g.loc[ds]
        print(f"  {ds:16} quality {r['gate_quality']:.3f}  cost {r['gate_cost']:.3f}  "
              f"both {r['gate_both']:.3f}")
    print(f"\ncore reference composition (should be 288 everywhere): "
          f"{sorted(df['core_n_points'].unique())}")
    print(f"core front size range: {df['core_n_front'].min()}–{df['core_n_front'].max()}")
    print(f"\nwrote {OUT/'indicators_long.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
