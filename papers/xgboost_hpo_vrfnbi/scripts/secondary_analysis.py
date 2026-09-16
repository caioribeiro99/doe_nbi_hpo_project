#!/usr/bin/env python
"""Secondary indicators, gate regimes, controls, baselines and holdout confirmation.

Everything the frozen protocol declares SECONDARY AND DESCRIPTIVE: reported with
intervals and no tests, never described as significant. The primary family lives in
primary_analysis.py.

Covers freeze-report steps 6, 7, 10, 11, 12 and 13.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.runner import (BOUNDARY_CONTROLS, DATASETS, DATASET_ROLES,  # noqa: E402
                                     N_REPLICATIONS, NSGA2_GEN, NSGA2_POP,
                                     NSGA2_UNMATCHED_MULTIPLIER,
                                     NSGA2_UNMATCHED_REPLICATION,
                                     PRIMARY_GEOMETRY_PANEL, SINGLE_OBJECTIVE)

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
ANALYSIS = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis"
SECONDARY = ("igd_plus", "gd", "spacing", "spacing_cv", "joint_nondominated_fraction",
             "n_front")


def ci95(x: np.ndarray, rng, reps: int = 5000) -> list[float]:
    idx = rng.integers(0, len(x), (reps, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> int:
    df = pd.read_csv(ANALYSIS / "indicators_long.csv")
    rng = np.random.default_rng(20260915)
    out = {}

    # ---- 6. secondary indicators, per dataset per entity, core reference --------
    sec = {}
    for ds in DATASETS:
        sec[ds] = {}
        for ent in sorted(df.entity.unique()):
            sub = df[(df.dataset == ds) & (df.entity == ent)]
            if sub.empty:
                continue
            blk = {}
            for ind in ("hv_ratio",) + SECONDARY:
                v = sub[f"core__{ind}"].to_numpy(float)
                fin = v[np.isfinite(v)]
                # Schott spacing is UNDEFINED on a single-point front: there are no
                # gaps to take a standard deviation over. Those units are counted and
                # excluded rather than propagated as NaN through a median.
                blk[ind] = {
                    "median": float(np.median(fin)) if len(fin) else None,
                    "ci95": ci95(fin, rng) if len(fin) >= 2 else None,
                    "n_defined": int(len(fin)), "n_undefined": int(len(v) - len(fin)),
                    "median_augmented": (
                        lambda a: float(np.median(a[np.isfinite(a)]))
                        if np.isfinite(a).any() else None)(
                            sub[f"augmented__{ind}"].to_numpy(float))}
            sec[ds][ent] = blk
    out["secondary_indicators"] = sec

    # ---- 7. surrogate gate regimes ---------------------------------------------
    gate = {}
    for ds in DATASETS:
        sub = df[(df.dataset == ds) & (df.entity == "NBI-S")].sort_values("replication")
        q = sub.gate_quality.to_numpy(bool)
        gate[ds] = {
            "quality_pass_rate": float(q.mean()),
            "quality_passes": int(q.sum()), "of": int(len(q)),
            "cost_pass_rate": float(sub.gate_cost.mean()),
            "both_pass_rate": float(sub.gate_both.mean()),
            "role": DATASET_ROLES[ds]}
        # the primary contrast conditioned on gate status, as the protocol requires
        for a, b in (("WS-S", "NBI-S"), ("NBI-S", "NBI-R")):
            xa = df[(df.dataset == ds) & (df.entity == a)].sort_values("replication")["core__hv_ratio"].to_numpy(float)
            xb = df[(df.dataset == ds) & (df.entity == b)].sort_values("replication")["core__hv_ratio"].to_numpy(float)
            d = xb - xa
            blk = {}
            for name, mask in (("gate_pass", q), ("gate_fail", ~q)):
                if mask.sum() >= 2:
                    blk[name] = {"n": int(mask.sum()),
                                 "median_diff": float(np.median(d[mask])),
                                 "win_fraction": float((d[mask] > 0).mean())}
                else:
                    blk[name] = {"n": int(mask.sum()), "median_diff": None,
                                 "win_fraction": None}
            gate[ds][f"{a}->{b}_conditioned"] = blk
    out["gate_regimes"] = gate

    # ---- 10/11. historical reconstruction and anchor-injection controls ---------
    ctrl = {}
    for ds in DATASETS:
        g = lambda e, c="core__hv_ratio": df[(df.dataset == ds) & (df.entity == e)].sort_values("replication")[c].to_numpy(float)
        asrun, shared, wss = g("HISTORICAL-WS-asrun"), g("HISTORICAL-WS"), g("WS-S")
        nbi_s, inj, nbi_r = g("NBI-S"), g("ANCHOR-INJECTION-CONTROL"), g("NBI-R")
        gap = nbi_r - nbi_s
        inj_gap = inj - nbi_s
        share = (np.median(inj_gap) / np.median(gap)) if np.median(gap) != 0 else float("nan")
        ctrl[ds] = {
            "historical": {
                "asrun_median": float(np.median(asrun)),
                "shared_spec_median": float(np.median(shared)),
                "ws_s_median": float(np.median(wss)),
                "asrun_to_shared_median_diff": float(np.median(shared - asrun)),
                "asrun_to_shared_ci95": ci95(shared - asrun, rng),
                "note": ("asrun is the bit-faithful frozen dissertation solver; shared "
                         "uses this campaign's surrogates and symmetric grid. The gap "
                         "is the whole historical reconstruction effect.")},
            "anchor_injection": {
                "nbi_s_median": float(np.median(nbi_s)),
                "injected_median": float(np.median(inj)),
                "nbi_r_median": float(np.median(nbi_r)),
                "injection_effect_median": float(np.median(inj_gap)),
                "injection_effect_ci95": ci95(inj_gap, rng),
                "full_anchor_gap_median": float(np.median(gap)),
                "share_of_gap_explained_by_set_composition": float(share),
                "note": ("the control holds geometry at NBI-S and varies only set "
                         "composition, so this share is the part of the NBI-S to "
                         "NBI-R gap attributable to injected extreme points rather "
                         "than to the relocated CHIM")}}
    out["controls"] = ctrl

    # ---- 12. baselines, matched and unmatched ----------------------------------
    base = {}
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        matched = {e: {"median_hv_ratio": float(np.median(sub[sub.entity == e]["core__hv_ratio"])),
                       "ci95": ci95(sub[sub.entity == e]["core__hv_ratio"].to_numpy(float), rng)}
                   for e in ("GRID", "RANDOM", "NSGA2-MATCHED") if (sub.entity == e).any()}
        u = ROOT / ds / f"rep_{NSGA2_UNMATCHED_REPLICATION:02d}" / "nsga2_unmatched.json"
        unmatched = json.loads(u.read_text()) if u.exists() else None
        base[ds] = {
            "evaluation_matched": matched,
            "nsga2_matched_budget": NSGA2_POP * NSGA2_GEN,
            "nsga2_unmatched": {
                "scope": f"one replication per dataset (rep {NSGA2_UNMATCHED_REPLICATION})",
                "budget": (unmatched or {}).get("evaluations"),
                "multiplier": NSGA2_UNMATCHED_MULTIPLIER,
                "status": ("CONTEXT BASELINE — no fairness claim attaches to it; it "
                           "enters no budget-matched comparison and neither reference"),
                "n_rows": (unmatched or {}).get("n_rows")},
            "single_objective_endpoints_excluded_from_front_indicators": list(SINGLE_OBJECTIVE)}
    out["baselines"] = base

    # ---- 13. holdout confirmation ----------------------------------------------
    hold = {}
    for ds in DATASETS:
        per_arm = {}
        for rep in range(N_REPLICATIONS):
            h = json.loads((ROOT / ds / f"rep_{rep:02d}" / "holdout_confirmation.json").read_text())
            for arm, blk in h["arms"].items():
                per_arm.setdefault(arm, {"internal": [], "holdout": []})
                # the quality-role responses, internal vs holdout, for the selected point
                per_arm[arm]["internal"].append(blk["internal"].get("Accuracy_Mean"))
                per_arm[arm]["holdout"].append(blk["holdout"].get("Accuracy_Mean"))
        hold[ds] = {
            arm: {"n": len(v["internal"]),
                  "median_internal_accuracy": float(np.median(v["internal"])),
                  "median_holdout_accuracy": float(np.median(v["holdout"])),
                  "median_drop": float(np.median(np.asarray(v["internal"]) - np.asarray(v["holdout"]))),
                  "drop_ci95": ci95(np.asarray(v["internal"]) - np.asarray(v["holdout"]), rng)}
            for arm, v in per_arm.items()}
    out["holdout_confirmation"] = hold

    # ---- the finite-reference caveat, quantified -------------------------------
    # A core-relative hv_ratio above 1 is not an error: it means the method found
    # points dominating part of the FINITE method-independent reference. The
    # protocol forbids describing the ratio as a fraction of the true Pareto
    # hypervolume. At the rate observed here that caveat is load-bearing, so it is
    # measured rather than mentioned.
    over = {}
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        over[ds] = {"rows": int(len(sub)),
                    "rows_hv_ratio_above_one": int((sub["core__hv_ratio"] > 1).sum()),
                    "share": float((sub["core__hv_ratio"] > 1).mean()),
                    "max_hv_ratio": float(sub["core__hv_ratio"].max()),
                    "by_entity": {e: {"n_above_one": int((g["core__hv_ratio"] > 1).sum()),
                                      "of": int(len(g)),
                                      "max": float(g["core__hv_ratio"].max())}
                                  for e, g in sub.groupby("entity")}}
    out["finite_reference_caveat"] = {
        "per_dataset": over,
        "overall_share_above_one": float((df["core__hv_ratio"] > 1).mean()),
        "overall_max": float(df["core__hv_ratio"].max()),
        "interpretation": (
            "the core reference is a FINITE method-independent empirical set of 288 "
            "points, not the true Pareto front. A ratio above 1 means the scored "
            "method found points dominating part of it. The ratio is therefore a "
            "score relative to that finite reference and must never be described as "
            "a fraction of the true Pareto hypervolume."),
        "spacing_undefined_rule": (
            "Schott spacing is undefined on a single-point front and is counted as "
            "undefined rather than propagated as NaN")}

    (ANALYSIS / "secondary_analysis.json").write_text(json.dumps(out, indent=2))

    print("GATE REGIMES (confirmatory, R = 30 per dataset)\n")
    print(f"{'dataset':16} {'role':32} {'quality':>9} {'cost':>7} {'both':>7}")
    for ds in DATASETS:
        g = gate[ds]
        print(f"{ds:16} {g['role']:32} {g['quality_passes']:2}/{g['of']:<3}{g['quality_pass_rate']:>5.0%} "
              f"{g['cost_pass_rate']:>6.0%} {g['both_pass_rate']:>6.0%}")
    print("\nANCHOR-INJECTION CONTROL — share of the NBI-S->NBI-R gap that is set composition\n")
    for ds in DATASETS:
        c = ctrl[ds]["anchor_injection"]
        print(f"  {ds:16} injection effect {c['injection_effect_median']:+.4f}  "
              f"full gap {c['full_anchor_gap_median']:+.4f}  "
              f"share {c['share_of_gap_explained_by_set_composition']:+.3f}")
    print("\nHISTORICAL RECONSTRUCTION — asrun vs shared specification\n")
    for ds in DATASETS:
        h = ctrl[ds]["historical"]
        print(f"  {ds:16} asrun {h['asrun_median']:.4f}  shared {h['shared_spec_median']:.4f}  "
              f"WS-S {h['ws_s_median']:.4f}  Δ(shared−asrun) {h['asrun_to_shared_median_diff']:+.4f}")
    print(f"\nwrote {ANALYSIS/'secondary_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
