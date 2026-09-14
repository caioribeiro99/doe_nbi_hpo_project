#!/usr/bin/env python
"""Regenerate every number in the objective-count decision, from committed artifacts.

The adversarial review of the two-to-three objective amendment found that none of
the eight figures carrying the proposal could be reproduced by any committed
script. Evidence a reviewer has to re-derive from scratch is not provenance, and
provenance is this paper's stated contribution.

Produces ``audits/objective_count_evidence.json`` with, per dataset:

  * the per-objective surrogate gate -- the protocol's own backward-eliminated
    surface fitted on the 88 design rows and scored on the 78-run complementary
    fraction, for EVERY objective the campaign would optimize, not only the
    aggregated composite;
  * inter-factor Spearman correlations, with bootstrap intervals, so that
    "the factors conflict" is stated at the precision the pilot supports;
  * non-dominated counts at two and three objectives, each against a permutation
    null, because adding any coordinate enlarges a non-dominated set;
  * what each factor actually is -- its correlation with a named reference and its
    dominant loading -- since the axes exchange roles across the panel;
  * eigenvalues and the Kaiser count, since the protocol fixes three components;
  * the aggregation-weighting agreement under the protocol's own factor stage,
    which is not the figure the dissertation audit reports.

Uses zero new real evaluations.

Usage:  python objective_count_evidence.py [--bootstrap 2000] [--permutations 400]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PILOT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits"
DATASETS = ["magic", "spambase", "adult", "bank_marketing"]
GATE_R2, GATE_RHO = 0.5, 0.9

_spec = importlib.util.spec_from_file_location("pilot", HERE / "pilot_stage_a_screening.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def factor_scores(fm, df: pd.DataFrame) -> np.ndarray:
    """All k standardized factor scores, not just the aggregated composite."""
    Z = (pilot.apply_transforms(df) - fm.mu_) / fm.sd_
    return (fm.pca_.transform(Z) @ fm.R_ - fm.score_mu_) / fm.score_sd_


def nondominated_count(F: np.ndarray) -> int:
    keep = 0
    for a in F:
        if not np.any(np.all(F <= a, axis=1) & np.any(F < a, axis=1)):
            keep += 1
    return keep


def permutation_null(F2: np.ndarray, third: np.ndarray, n: int, rng) -> dict:
    """Null for the non-dominated count at three objectives.

    Adding *any* third coordinate weakly enlarges a non-dominated set, so the raw
    growth factor is not evidence on its own. The null permutes the third
    coordinate against the first two, which preserves its marginal distribution
    and destroys only its association with them.
    """
    counts = []
    for _ in range(n):
        counts.append(nondominated_count(
            np.column_stack([F2, rng.permutation(third)])))
    counts = np.asarray(counts, dtype=float)
    obs = nondominated_count(np.column_stack([F2, third]))
    return {"observed": int(obs), "null_mean": round(float(counts.mean()), 2),
            "null_sd": round(float(counts.std(ddof=1)), 2),
            "null_q05": float(np.quantile(counts, 0.05)),
            "null_q95": float(np.quantile(counts, 0.95)),
            "p_value_observed_ge_null": round(float((counts >= obs).mean()), 4),
            "exceeds_null": bool((counts >= obs).mean() < 0.05)}


def boot_spearman(a: np.ndarray, b: np.ndarray, n: int, rng) -> dict:
    vals = []
    idx = np.arange(len(a))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(a[s])) < 3 or len(np.unique(b[s])) < 3:
            continue
        vals.append(spearmanr(a[s], b[s]).statistic)
    v = np.asarray(vals, dtype=float)
    return {"estimate": round(float(spearmanr(a, b).statistic), 4),
            "ci95_lo": round(float(np.quantile(v, 0.025)), 4),
            "ci95_hi": round(float(np.quantile(v, 0.975)), 4),
            "interval_covers_zero": bool(np.quantile(v, 0.025) <= 0 <= np.quantile(v, 0.975))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    report: dict = {
        "purpose": "regenerate the objective-count evidence from committed artifacts",
        "gate": {"external_r2_min": GATE_R2, "spearman_min": GATE_RHO},
        "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
        "datasets": {},
    }

    for ds in DATASETS:
        d = pd.read_csv(PILOT / f"{ds}_design.csv")
        v = pd.read_csv(PILOT / f"{ds}_validation_complement.csv")
        fm = pilot.FactorModel().fit(d)
        summ = fm.summary()
        share = np.asarray(summ["explained_variance_share"], dtype=float)
        sd_, sv_ = factor_scores(fm, d), factor_scores(fm, v)

        order = sorted(fm.q_idx_, key=lambda j: -share[j])
        names = {fm.cost_idx_: "cost"}
        for rank, j in enumerate(order):
            names[j] = f"quality{rank + 1}"

        # what each factor IS: correlation with a named reference
        M = pilot.apply_transforms(d)
        cols = list(pilot.RESPONSES)
        qi = [i for i, c in enumerate(cols) if pilot.RESPONSES[c]["role"] == "quality"]
        ref = np.column_stack([(M[:, i] - M[:, i].mean()) / M[:, i].std(ddof=1)
                               for i in qi]).mean(1)

        gate_rows = []
        for j in range(fm.k):
            r2, rho, nterms = pilot.external_scores(d, sd_[:, j], v, sv_[:, j])
            gate_rows.append({
                "objective": names[j], "factor_index": j + 1,
                "external_r2": round(float(r2), 4),
                "external_spearman": round(float(rho), 4),
                "surface_terms": int(nterms),
                "gate_pass": bool(r2 >= GATE_R2 and rho >= GATE_RHO),
                "explained_variance_share": round(float(share[j]), 4),
                "dominant_response": str(summ["loadings"].iloc[:, j].abs().idxmax()),
                "spearman_with_mean_quality": round(
                    float(spearmanr(sd_[:, j], ref).statistic), 4),
            })

        fs_d, fs_v = fm.transform(d), fm.transform(v)
        r2c, rhoc, _ = pilot.external_scores(d, fs_d["quality"], v, fs_v["quality"])

        F2 = np.column_stack([fs_d["quality"], fs_d["cost"]])
        F3 = np.column_stack([sd_[:, order[0]], sd_[:, order[1]], sd_[:, fm.cost_idx_]]) \
            if fm.cost_idx_ not in order else None
        F3 = np.column_stack([sd_[:, order[0]], sd_[:, order[1]], sd_[:, fm.cost_idx_]])

        from sklearn.decomposition import PCA
        Zfull = (M - fm.mu_) / fm.sd_
        lam = PCA(n_components=min(7, Zfull.shape[1]), random_state=0).fit(Zfull).explained_variance_

        report["datasets"][ds] = {
            "per_objective_gate": gate_rows,
            "aggregated_q2_composite_gate": {
                "external_r2": round(float(r2c), 4),
                "external_spearman": round(float(rhoc), 4),
                "gate_pass": bool(r2c >= GATE_R2 and rhoc >= GATE_RHO)},
            "inter_factor_spearman": boot_spearman(
                sd_[:, order[0]], sd_[:, order[1]], args.bootstrap, rng),
            "nondominated": {
                "q2": nondominated_count(F2),
                "q3": nondominated_count(F3),
                "raw_growth_factor": round(nondominated_count(F3) / max(nondominated_count(F2), 1), 2),
                "permutation_null_q3": permutation_null(
                    F3[:, [0, 2]], F3[:, 1], args.permutations, rng)},
            "eigenvalues": [round(float(x), 4) for x in lam],
            "kaiser_components_retained": int((lam > 1.0).sum()),
            "lambda3_over_lambda4": round(float(lam[2] / lam[3]), 3) if len(lam) > 3 else None,
            "aggregation_weighting_agreement_v2_pipeline": round(
                float(spearmanr(fs_d["quality"], fs_d["quality_equal"]).statistic), 4),
        }

    # panel-level summary of the facts the decision turns on
    g = {ds: {r["objective"]: r["gate_pass"]
              for r in report["datasets"][ds]["per_objective_gate"]}
         for ds in DATASETS}
    report["summary"] = {
        "per_objective_gate_cells": sum(len(v) for v in g.values()),
        "per_objective_gate_failures": sum(1 for ds in g for k, ok in g[ds].items() if not ok),
        "datasets_where_the_second_quality_factor_fails_the_gate":
            [ds for ds in DATASETS if not g[ds].get("quality2", True)],
        "datasets_where_any_quality_objective_fails_the_gate":
            [ds for ds in DATASETS
             if not (g[ds].get("quality1", True) and g[ds].get("quality2", True))],
        "datasets_where_q3_nondominated_count_exceeds_its_null":
            [ds for ds in DATASETS
             if report["datasets"][ds]["nondominated"]["permutation_null_q3"]["exceeds_null"]],
        "kaiser_retained_per_dataset":
            {ds: report["datasets"][ds]["kaiser_components_retained"] for ds in DATASETS},
        "aggregation_weighting_agreement_range": [
            min(report["datasets"][ds]["aggregation_weighting_agreement_v2_pipeline"]
                for ds in DATASETS),
            max(report["datasets"][ds]["aggregation_weighting_agreement_v2_pipeline"]
                for ds in DATASETS)],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "objective_count_evidence.json").write_text(json.dumps(report, indent=2))

    print("per-objective surrogate gate (R2 >= 0.5 AND Spearman >= 0.9)\n")
    print(f"{'dataset':16s}{'objective':>10}{'R2':>8}{'rho':>8}{'terms':>7}{'gate':>7}"
          f"{'share':>8}{'rho vs mean quality':>21}  dominant")
    for ds in DATASETS:
        for r in report["datasets"][ds]["per_objective_gate"]:
            print(f"{ds:16s}{r['objective']:>10}{r['external_r2']:8.3f}"
                  f"{r['external_spearman']:8.3f}{r['surface_terms']:7d}"
                  f"{'PASS' if r['gate_pass'] else 'FAIL':>7}"
                  f"{r['explained_variance_share']:8.3f}"
                  f"{r['spearman_with_mean_quality']:21.3f}  {r['dominant_response']}")
    print(f"\nnon-dominated counts against a permutation null")
    print(f"{'dataset':16s}{'q2':>5}{'q3':>5}{'raw x':>7}{'null mean':>11}{'p':>8}{'exceeds':>9}")
    for ds in DATASETS:
        n = report["datasets"][ds]["nondominated"]; p = n["permutation_null_q3"]
        print(f"{ds:16s}{n['q2']:5d}{n['q3']:5d}{n['raw_growth_factor']:7.2f}"
              f"{p['null_mean']:11.1f}{p['p_value_observed_ge_null']:8.3f}"
              f"{str(p['exceeds_null']):>9}")
    print(f"\ninter-factor Spearman with bootstrap intervals")
    for ds in DATASETS:
        s = report["datasets"][ds]["inter_factor_spearman"]
        print(f"  {ds:16s} {s['estimate']:+.3f}  [{s['ci95_lo']:+.3f}, {s['ci95_hi']:+.3f}]"
              f"{'  covers zero' if s['interval_covers_zero'] else ''}")
    print(f"\nKaiser components retained: {report['summary']['kaiser_retained_per_dataset']}")
    print(f"aggregation-weighting agreement under the protocol's own factor stage: "
          f"{report['summary']['aggregation_weighting_agreement_range']}")
    print(f"\nwrote {OUT/'objective_count_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
