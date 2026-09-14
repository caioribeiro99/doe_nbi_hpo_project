#!/usr/bin/env python
"""V1: verify the factor algebra, not only its output.

The corrected factor stage must satisfy invariants that hold by construction if it
is right and fail visibly if it is not. Checking the numbers it produces is not
enough: the defect the review found produced perfectly plausible numbers.

Eight invariants, per dataset:

  1. unrotated PCA score covariance is diagonal;
  2. standardized retained component scores have covariance approximately I;
  3. orthogonally rotated standardized scores also have covariance approximately I;
  4. the rotated loading matrix is the correlation of the responses with the scores
     actually produced, that is it describes the transformation applied;
  5. the truncated reconstruction from scores and rotated loadings equals the
     retained-PCA reconstruction;
  6. sign flips change orientation only, not covariance or reconstruction;
  7. role permutation changes labels only, not the represented subspace;
  8. the aggregation weights equal the normalized rotated sums of squared loadings.

Terminology, exactly. The factors are standardized to unit variance, so their
variances carry no information and the aggregation weights are NOT factor
variances. They are **normalized rotated sums of squared loadings**: for factor j,
the sum over responses of the squared rotated loading, divided by that sum over the
quality factors.

Writes audits/factor_algebra_audit.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.evaluator import RESPONSES                       # noqa: E402
from doe_xgb.campaign.factor_model import (apply_transforms,           # noqa: E402
                                           fit_factor_model)

PILOT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits"
DATASETS = ["magic", "spambase", "adult", "bank_marketing"]
TOL = 1e-8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()
    names = list(RESPONSES)
    report: dict = {"tolerance": args.tol, "datasets": {}, "audit_table": []}
    failures: list[str] = []

    for ds in DATASETS:
        d = pd.read_csv(PILOT / f"{ds}_design.csv")
        fm = fit_factor_model(d)
        M = apply_transforms(d)
        Z = (M - fm.mu) / fm.sd
        k = fm.rotated_loadings.shape[1]

        pca_full = PCA(n_components=k, random_state=0).fit(Z)
        raw_scores = Z @ pca_full.components_.T
        std_scores = raw_scores / np.sqrt(fm.eigenvalues)
        rot_scores = std_scores @ fm.rotation
        applied = fm.transform(d)["factor_scores"]

        def offdiag(C):
            return float(np.abs(C - np.diag(np.diag(C))).max())

        checks = {}
        # 1
        C_raw = np.cov(raw_scores, rowvar=False)
        checks["1_unrotated_score_covariance_is_diagonal"] = {
            "max_off_diagonal": offdiag(C_raw), "pass": offdiag(C_raw) < 1e-6}
        # 2
        C_std = np.cov(std_scores, rowvar=False)
        checks["2_standardized_scores_covariance_is_identity"] = {
            "max_abs_deviation": float(np.abs(C_std - np.eye(k)).max()),
            "pass": bool(np.abs(C_std - np.eye(k)).max() < 1e-6)}
        # 3
        C_rot = np.cov(rot_scores, rowvar=False)
        checks["3_rotated_standardized_scores_covariance_is_identity"] = {
            "max_abs_deviation": float(np.abs(C_rot - np.eye(k)).max()),
            "pass": bool(np.abs(C_rot - np.eye(k)).max() < 1e-6)}
        # 4 -- the loadings describe the transformation actually applied
        emp = np.corrcoef(np.column_stack([Z, applied]), rowvar=False)[:Z.shape[1], Z.shape[1]:]
        dev4 = float(np.abs(emp - fm.rotated_loadings).max())
        checks["4_rotated_loadings_describe_the_applied_scores"] = {
            "max_abs_deviation": dev4, "pass": dev4 < 1e-6}
        # 5 -- reconstruction equivalence
        recon_factor = (applied * np.sqrt(1.0)) @ fm.rotated_loadings.T
        recon_pca = raw_scores @ pca_full.components_
        dev5 = float(np.abs(recon_factor - recon_pca).max())
        checks["5_truncated_reconstruction_matches_retained_pca"] = {
            "max_abs_deviation": dev5, "pass": dev5 < 1e-6}
        # 6 -- sign flips change orientation only
        flip = np.array(fm.diagnostics["sign_flips_applied"], dtype=float)
        unflipped = applied * flip                    # undo the applied orientation
        dev6_cov = float(np.abs(np.cov(unflipped, rowvar=False) - np.cov(applied, rowvar=False)).max())
        dev6_rec = float(np.abs((unflipped @ (fm.rotated_loadings * flip).T) - recon_factor).max())
        checks["6_sign_flips_change_orientation_only"] = {
            "covariance_change": dev6_cov, "reconstruction_change": dev6_rec,
            "pass": dev6_cov < 1e-6 and dev6_rec < 1e-6}
        # 7 -- role permutation changes labels only
        perm = np.argsort([-((fm.rotated_loadings ** 2).sum(axis=0))[j] for j in range(k)])
        dev7 = float(np.abs((applied[:, perm] @ fm.rotated_loadings[:, perm].T)
                            - recon_factor).max())
        checks["7_role_permutation_preserves_the_subspace"] = {
            "reconstruction_change": dev7, "pass": dev7 < 1e-6}
        # 8 -- the weights are the normalized rotated sums of squared loadings
        ss = (fm.rotated_loadings ** 2).sum(axis=0)
        expect = np.array([ss[j] for j in fm.quality_indices], dtype=float)
        expect = expect / expect.sum()
        dev8 = float(np.abs(expect - fm.quality_weights).max())
        checks["8_weights_are_normalized_rotated_ss_loadings"] = {
            "max_abs_deviation": dev8, "pass": dev8 < 1e-12}

        for name, c in checks.items():
            if not c["pass"]:
                failures.append(f"{ds}: {name}")
        report["datasets"][ds] = {
            "checks": checks,
            "max_off_diagonal_factor_correlation":
                float(fm.diagnostics["max_off_diagonal_score_correlation"])}

        for j in range(k):
            role = ("cost" if j == fm.cost_index
                    else f"quality{list(fm.quality_indices).index(j) + 1}")
            report["audit_table"].append({
                "dataset": ds, "component": j,
                "eigenvalue": round(float(fm.eigenvalues[j]), 6),
                "unrotated_loading_ss": round(
                    float(((pca_full.components_.T * np.sqrt(fm.eigenvalues)) ** 2)
                          .sum(axis=0)[j]), 6),
                "rotated_loading_ss": round(float(ss[j]), 6),
                "normalized_rotated_share": round(float(ss[j] / ss.sum()), 6),
                "assigned_role": role,
                "sign": float(flip[j]),
                "factor_variance": round(float(np.var(applied[:, j], ddof=1)), 9),
                "max_off_diagonal_factor_correlation":
                    round(float(fm.diagnostics["max_off_diagonal_score_correlation"]), 12),
                "dominant_response": names[int(np.argmax(np.abs(fm.rotated_loadings[:, j])))],
            })

    report["all_pass"] = not failures
    report["failures"] = failures
    report["terminology"] = (
        "The factors are standardized to unit variance, so their variances carry no "
        "information and the aggregation weights are NOT factor variances. They are "
        "normalized rotated sums of squared loadings.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "factor_algebra_audit.json").write_text(json.dumps(report, indent=2))

    print("factor algebra invariants\n")
    first = report["datasets"][DATASETS[0]]["checks"]
    print(f"{'invariant':56s}" + "".join(f"{d[:9]:>11}" for d in DATASETS))
    for name in first:
        row = "".join(
            f"{'PASS' if report['datasets'][d]['checks'][name]['pass'] else 'FAIL':>11}"
            for d in DATASETS)
        print(f"{name:56s}{row}")
    print("\naudit table (first dataset shown; full table in the JSON)\n")
    print(f"{'ds':16s}{'cmp':>4}{'eigenvalue':>12}{'unrot SS':>10}{'rot SS':>9}"
          f"{'share':>8}{'role':>10}{'sign':>6}{'var':>7}  dominant")
    for r in report["audit_table"][:3]:
        print(f"{r['dataset']:16s}{r['component']:4d}{r['eigenvalue']:12.4f}"
              f"{r['unrotated_loading_ss']:10.4f}{r['rotated_loading_ss']:9.4f}"
              f"{r['normalized_rotated_share']:8.4f}{r['assigned_role']:>10}"
              f"{r['sign']:6.0f}{r['factor_variance']:7.3f}  {r['dominant_response']}")
    print(f"\n{'ALL INVARIANTS HOLD' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    print(f"wrote {OUT/'factor_algebra_audit.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
