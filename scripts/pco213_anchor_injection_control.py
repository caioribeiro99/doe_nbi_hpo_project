#!/usr/bin/env python
"""Control for the A-to-B anchor contrast: how much of the gap is set composition?

Raised by adversarial review. Because vertex-beta subproblems return the anchor directly,
NBI-B's candidate set CONTAINS the three real single-objective optima, which are also in the
empirical reference. Part of the NBI-B advantage over NBI-A may therefore be the injection of
those extreme points rather than the relocation of the CHIM.

This script isolates it, using only frozen artifacts and no new experiment: it rescores
NBI-A's candidate set augmented with the three real anchors, and reports what fraction of the
A-to-B hypervolume gap that augmentation alone closes.

Writes reports/pco213_postwork_benchmark/anchor_injection_control.csv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import mixens.pareto_tools as pt  # noqa: E402

BENCH = REPO / "experiments" / "pco213_postwork_benchmark"
OUT = REPO / "reports" / "pco213_postwork_benchmark"
NAMES = ["lr", "gnb", "knn", "rf", "xgb"]
WC = [f"w_{n}" for n in NAMES]
ANCHOR_KEYS = ("direct_auc_search", "slsqp_direct_logloss", "cheapest_vertex")
DATASETS = ["santander", "bnp", "porto", "uci_credit"]


def hv_ratio(df: pd.DataFrame, lo, hi, hvr) -> float:
    ok = df["success"].to_numpy(bool) if "success" in df else np.ones(len(df), bool)
    F = np.column_stack([-df.roc_auc, df.log_loss, df.cost_weighted])[ok]
    if len(F) == 0:
        return 0.0
    return pt.hypervolume(pt.normalize(F[pt.fast_pareto_mask(F)], lo, hi), np.full(3, 1.1)) / hvr


def main() -> int:
    rows = []
    for ds in DATASETS:
        for rep in range(30):
            d = BENCH / ds / f"rep_{rep:02d}"
            refs = json.loads((d / "references.json").read_text())
            anc = pd.DataFrame([
                {**{c: v["w"][j] for j, c in enumerate(WC)}, "roc_auc": v["oof_roc_auc"],
                 "log_loss": v["oof_log_loss"], "cost_weighted": v["cost_weighted"],
                 "cost_support": v["cost_support"], "success": True}
                for k, v in refs["references"].items() if k in ANCHOR_KEYS and "w" in v])
            A = pd.read_csv(d / "nbi_A_candidates.csv")
            B = pd.read_csv(d / "nbi_B_candidates.csv")
            cols = [*WC, "roc_auc", "log_loss", "cost_weighted", "cost_support", "success"]
            A_aug = pd.concat([A[cols], anc], ignore_index=True)
            er = pd.read_csv(d / "empirical_reference_front_weighted.csv")
            Fr = np.column_stack([-er.roc_auc, er.log_loss, er.cost_weighted])
            lo, hi = Fr.min(0), Fr.max(0)
            hvr = pt.hypervolume(pt.normalize(Fr, lo, hi), np.full(3, 1.1))
            hA, hB, hAa = (hv_ratio(x, lo, hi, hvr) for x in (A, B, A_aug))
            rows.append({"dataset": ds, "rep": rep, "hv_A": hA, "hv_B": hB, "hv_A_plus_anchors": hAa,
                         "gap_A_to_B": hB - hA, "gap_closed_by_injection": hAa - hA,
                         "fraction_closed": (hAa - hA) / (hB - hA) if hB - hA > 1e-9 else np.nan,
                         "residual_chim_effect": hB - hAa})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anchor_injection_control.csv", index=False)
    print("Fraction of the NBI-A -> NBI-B hypervolume gap closed by injecting the three real")
    print("anchors into NBI-A's candidate set, without relocating the CHIM:\n")
    g = df.dropna(subset=["fraction_closed"]).groupby("dataset")
    for ds in DATASETS:
        s = g.get_group(ds)
        print(f"  {ds:11s} n={len(s):2d}  median {s.fraction_closed.median():6.1%}  "
              f"IQR {s.fraction_closed.quantile(.25):.0%}-{s.fraction_closed.quantile(.75):.0%}  "
              f"| residual CHIM effect median {s.residual_chim_effect.median():+.4f} HV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
