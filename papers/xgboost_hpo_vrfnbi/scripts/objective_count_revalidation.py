#!/usr/bin/env python
"""V0: recompute the objective-count evidence with the CORRECTED factor model.

The two-objective decision rested on two diagnostics: per-objective surrogate
reliability, and the semantic identity of the latent quality axes across datasets.
The final protocol review then established that the factor-score construction used
to produce both was wrong — rotated loadings and scores did not describe the same
coordinates, the "orthogonal" factors were correlated up to 0.698, role assignment
and sign orientation were read off the wrong matrix, and the aggregation weights
indexed unrotated eigenvalues by rotated component index.

The previous evidence is therefore not automatically valid, and this rebuilds it
from `doe_xgb.campaign.factor_model`, the corrected implementation the campaign
will use. It re-runs the same decision rule that existed before the confirmatory
campaign. It does not look at any arm.

Outcome is exactly one of:

    A. TWO-OBJECTIVE DECISION REVALIDATED
    B. PREVIOUS TWO-OBJECTIVE DECISION INVALIDATED BY THE FACTOR CORRECTION

Writes audits/objective_count_revalidation.json.

Usage:  python objective_count_revalidation.py [--permutations 400] [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.design import external_scores, gate_pass          # noqa: E402
from doe_xgb.campaign.evaluator import RESPONSES                        # noqa: E402
from doe_xgb.campaign.factor_model import (apply_transforms,            # noqa: E402
                                           fit_factor_model)
from doe_xgb.reporting import pareto_front                              # noqa: E402

PILOT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits"
DATASETS = ["magic", "spambase", "adult", "bank_marketing"]


def nondominated_count(F: np.ndarray) -> int:
    return int(pareto_front(np.asarray(F, dtype=float)).sum())


def permutation_null(F2: np.ndarray, third: np.ndarray, n: int, rng) -> dict:
    counts = np.array([nondominated_count(np.column_stack([F2, rng.permutation(third)]))
                       for _ in range(n)], dtype=float)
    obs = nondominated_count(np.column_stack([F2, third]))
    return {"observed": obs, "null_mean": round(float(counts.mean()), 2),
            "p_value": round(float((counts >= obs).mean()), 4),
            "exceeds_null": bool((counts >= obs).mean() < 0.05)}


def boot_spearman(a, b, n, rng) -> dict:
    idx = np.arange(len(a))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(a[s])) > 2 and len(np.unique(b[s])) > 2:
            vals.append(spearmanr(a[s], b[s]).statistic)
    v = np.asarray(vals)
    return {"estimate": round(float(spearmanr(a, b).statistic), 4),
            "ci95_lo": round(float(np.quantile(v, 0.025)), 4),
            "ci95_hi": round(float(np.quantile(v, 0.975)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=400)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    names = list(RESPONSES)
    q_rows = [i for i, c in enumerate(names) if RESPONSES[c]["role"] == "quality"]
    report: dict = {"purpose": "objective-count evidence recomputed with the corrected "
                              "factor model", "datasets": {}}

    for ds in DATASETS:
        d = pd.read_csv(PILOT / f"{ds}_design.csv")
        v = pd.read_csv(PILOT / f"{ds}_validation_complement.csv")
        fm = fit_factor_model(d)
        sd_ = fm.transform(d)["factor_scores"]
        sv_ = fm.transform(v)["factor_scores"]

        rot_ss = (fm.rotated_loadings ** 2).sum(axis=0)
        order = sorted(fm.quality_indices, key=lambda j: -rot_ss[j])
        label = {fm.cost_index: "cost", order[0]: "quality1", order[1]: "quality2"}

        M = apply_transforms(d)
        ref = np.column_stack([(M[:, i] - M[:, i].mean()) / M[:, i].std(ddof=1)
                               for i in q_rows]).mean(axis=1)

        rows = []
        for j in range(fm.rotated_loadings.shape[1]):
            sc = external_scores(d, sd_[:, j], v, sv_[:, j])
            rows.append({
                "objective": label[j], "factor_index": j,
                "external_r2": round(sc["external_r2"], 4),
                "external_rmse": round(sc["external_rmse"], 4),
                "external_spearman": round(sc["external_spearman"], 4),
                "surface_terms": sc["terms"],
                "gate_pass": gate_pass(sc),
                "rotated_ss_loading_share": round(float(rot_ss[j] / rot_ss.sum()), 4),
                "unrotated_eigenvalue_share":
                    round(float(fm.eigenvalues[j] / fm.eigenvalues.sum()), 4),
                "dominant_response": names[int(np.argmax(np.abs(fm.rotated_loadings[:, j])))],
                "spearman_with_mean_quality":
                    round(float(spearmanr(sd_[:, j], ref).statistic), 4),
                "sign_applied": round(float(fm.diagnostics["sign_flips_applied"][j]), 1),
            })

        fs_d, fs_v = fm.transform(d), fm.transform(v)
        comp = external_scores(d, fs_d["quality"], v, fs_v["quality"])
        F2 = np.column_stack([fs_d["quality"], fs_d["cost"]])
        F3 = np.column_stack([sd_[:, order[0]], sd_[:, order[1]], sd_[:, fm.cost_index]])

        report["datasets"][ds] = {
            "per_objective": rows,
            "aggregated_q2_composite": {
                "external_r2": round(comp["external_r2"], 4),
                "external_spearman": round(comp["external_spearman"], 4),
                "gate_pass": gate_pass(comp)},
            "quality_weights": [round(float(x), 4) for x in fm.quality_weights],
            "max_off_diagonal_factor_correlation":
                float(fm.diagnostics["max_off_diagonal_score_correlation"]),
            "inter_quality_factor_spearman": boot_spearman(
                sd_[:, order[0]], sd_[:, order[1]], args.bootstrap, rng),
            "nondominated": {
                "q2": nondominated_count(F2), "q3": nondominated_count(F3),
                "permutation_null_q3": permutation_null(
                    F3[:, [0, 2]], F3[:, 1], args.permutations, rng)},
            "kaiser_retained": fm.diagnostics["kaiser_retained"],
        }

    # --- apply the decision rule that existed before the campaign ---------------
    g = {ds: {r["objective"]: r["gate_pass"] for r in report["datasets"][ds]["per_objective"]}
         for ds in DATASETS}
    fails = [ds for ds in DATASETS if not g[ds].get("quality2", True)]
    # semantic identity: which axis carries overall quality on each dataset
    carrier = {}
    for ds in DATASETS:
        rows = report["datasets"][ds]["per_objective"]
        q = {r["objective"]: r["spearman_with_mean_quality"]
             for r in rows if r["objective"].startswith("quality")}
        carrier[ds] = max(q, key=lambda k: abs(q[k]))
    stable = len(set(carrier.values())) == 1

    report["decision_inputs"] = {
        "datasets_where_the_second_quality_factor_fails_the_gate": fails,
        "per_objective_gate_failures": sum(1 for ds in g for k in g[ds] if not g[ds][k]),
        "overall_quality_carrier_by_dataset": carrier,
        "semantic_identity_stable_across_panel": stable,
        "datasets_where_q3_exceeds_its_permutation_null":
            [ds for ds in DATASETS
             if report["datasets"][ds]["nondominated"]["permutation_null_q3"]["exceeds_null"]],
    }
    revalidated = bool(fails) or not stable
    report["outcome"] = ("A. TWO-OBJECTIVE DECISION REVALIDATED" if revalidated
                         else "B. PREVIOUS TWO-OBJECTIVE DECISION INVALIDATED BY THE "
                              "FACTOR CORRECTION")
    report["reasoning"] = (
        "The rule refuses promotion when the candidate objective fails the frozen "
        "surrogate gate on any panel dataset, or when what the second quality axis "
        "measures is not the same construct across datasets. Either alone is "
        "disqualifying, because three of the four arms optimize the surrogate and "
        "the study's synthesis is across datasets.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "objective_count_revalidation.json").write_text(json.dumps(report, indent=2))

    print("per-objective surrogate gate, CORRECTED factor model "
          "(R2 >= 0.5 AND Spearman >= 0.9)\n")
    print(f"{'dataset':16s}{'objective':>10}{'R2':>8}{'rho':>8}{'terms':>7}{'gate':>7}"
          f"{'rot share':>11}{'rho vs mean quality':>21}  dominant")
    for ds in DATASETS:
        for r in report["datasets"][ds]["per_objective"]:
            print(f"{ds:16s}{r['objective']:>10}{r['external_r2']:8.3f}"
                  f"{r['external_spearman']:8.3f}{r['surface_terms']:7d}"
                  f"{'PASS' if r['gate_pass'] else 'FAIL':>7}"
                  f"{r['rotated_ss_loading_share']:11.3f}"
                  f"{r['spearman_with_mean_quality']:21.3f}  {r['dominant_response']}")
    print(f"\n{'dataset':16s}{'q2 ND':>7}{'q3 ND':>7}{'null':>8}{'p':>7}"
          f"{'inter-factor rho':>18}{'max |corr|':>12}")
    for ds in DATASETS:
        n = report["datasets"][ds]["nondominated"]; p = n["permutation_null_q3"]
        s = report["datasets"][ds]["inter_quality_factor_spearman"]
        print(f"{ds:16s}{n['q2']:7d}{n['q3']:7d}{p['null_mean']:8.1f}{p['p_value']:7.3f}"
              f"{s['estimate']:+18.3f}"
              f"{report['datasets'][ds]['max_off_diagonal_factor_correlation']:12.1e}")
    print(f"\noverall-quality carrier by dataset: "
          f"{report['decision_inputs']['overall_quality_carrier_by_dataset']}")
    print(f"semantic identity stable across the panel: "
          f"{report['decision_inputs']['semantic_identity_stable_across_panel']}")
    print(f"second quality factor fails the gate on: "
          f"{report['decision_inputs']['datasets_where_the_second_quality_factor_fails_the_gate'] or 'no dataset'}")
    print(f"\nOUTCOME: {report['outcome']}")
    print(f"\nwrote {OUT/'objective_count_revalidation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
