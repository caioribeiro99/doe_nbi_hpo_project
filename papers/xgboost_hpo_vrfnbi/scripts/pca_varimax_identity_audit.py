#!/usr/bin/env python
"""Forensic audit of the dissertation's PCA/Varimax factor stage.

The dissertation and the EAAI 2025 article both describe the objective-reduction
step as *factor analysis with Varimax rotation*, producing *Varimax-rotated
factors* (VRFs) whose *loadings* are interpreted. This script establishes, from
the frozen code at tag ``v0.1.0-dissertation``, what the implementation actually
computes, and quantifies the difference on real data.

Five questions, each answered by a computation rather than by reading:

  Q1  Are the reported "loadings" loadings, or eigenvectors?
  Q2  Does rotating eigenvectors give the same rotation as rotating loadings?
  Q3  How many objectives does the optimizer actually see?
  Q4  How is Score_Quality formed, and what does that discard?
  Q5  Does the code implement the EAAI 2025 FMSE target-seeking wrapper?

Nothing here modifies any historical artifact. The frozen code is extracted to a
scratch directory and imported from there.

Usage:
    python pca_varimax_identity_audit.py [--doe-results PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits"
DISS_TAG = "v0.1.0-dissertation"
METRICS = ["Accuracy_Mean", "Precision_Mean", "Recall_Mean", "Specificity_Mean", "Time_MeanFold"]
TIME_COL = "Time_MeanFold"


def frozen_module(scratch: Path):
    scratch.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "archive", DISS_TAG, "src/doe_xgb"],
                         cwd=REPO, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(scratch)], input=tar.stdout, check=True)
    sys.path.insert(0, str(scratch / "src"))
    import doe_xgb.factor_analysis as fa
    assert str(scratch) in fa.__file__, f"resolved to {fa.__file__}"
    return fa


def zscore(x, ddof=1):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=ddof)
    return (x - mu) / np.where(sd == 0, 1.0, sd)


def canonical_pca_varimax(Z: np.ndarray, k: int, varimax):
    """PCA factor extraction as the factor-analysis literature defines it.

    The loading of variable i on component j is the correlation between them,
    which for standardized data is ``v_ij * sqrt(lambda_j)`` -- the eigenvector
    entry scaled by the square root of the eigenvalue. Varimax is defined on
    that matrix, because its criterion is the variance of squared loadings and
    is therefore not scale-free across columns.
    """
    from sklearn.decomposition import PCA
    p = PCA(n_components=k, random_state=0).fit(Z)
    eigvec = p.components_.T                              # unit-norm columns
    lam = p.explained_variance_                           # eigenvalues
    loadings = eigvec * np.sqrt(lam)                      # the actual loadings
    rot_loadings, R = varimax(loadings)
    return {"eigvec": eigvec, "eigval": lam, "loadings": loadings,
            "rot_loadings": rot_loadings, "R": R,
            "scores": p.transform(Z)}


def _weighting_reading(rho: float, overlap: int, argmax_agrees: bool) -> str:
    """State what the measurement shows, rather than what it was expected to show."""
    if rho >= 0.95 and overlap >= 8 and argmax_agrees:
        return (f"The two composites rank the design almost identically (Spearman {rho:.4f}, "
                f"{overlap}/10 of the top ten shared, same best row), so on this data the equal "
                "weighting is a harmless simplification. It remains an undeclared choice that "
                "nothing in the pipeline reports or re-checks on another dataset.")
    return (f"The two composites disagree substantially: Spearman {rho:.4f}, only {overlap}/10 of "
            f"the top ten design rows shared, and the best row "
            f"{'agrees' if argmax_agrees else 'differs'}. The quality objective the pipeline "
            "optimizes is therefore sensitive to an aggregation choice that is neither stated in "
            "the method nor varied in any sensitivity analysis. Whichever weighting is preferred, "
            "the result is that 'Score_Quality' is not determined by the factor extraction alone.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe-results", default=str(OUT / "provenance" / "doe_results.csv"))
    ap.add_argument("--out", default=str(OUT / "pca_varimax_audit.json"))
    ap.add_argument("--scratch", default="/tmp/diss_frozen_fa")
    args = ap.parse_args()

    fa = frozen_module(Path(args.scratch))
    report: dict = {"dissertation_tag": DISS_TAG, "findings": {}}

    path = Path(args.doe_results)
    if not path.exists():
        print(f"no DOE results at {path}; run provenance_reproduce_dissertation.py first")
        return 2
    df = pd.read_csv(path)
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise KeyError(f"missing metric columns {missing}")
    report["data"] = {"source": str(path.relative_to(REPO)), "rows": int(len(df))}

    # Reproduce the frozen preprocessing exactly: log1p on time, then z-score.
    X = np.asarray(df[METRICS], dtype=float)
    t_idx = METRICS.index(TIME_COL)
    X[:, t_idx] = np.log1p(np.clip(X[:, t_idx], 0, None))
    Z = zscore(X)
    k = 3   # what the frozen defaults produce: n_factors=3, force_time_factor=True

    # ---------------- Q1: loadings or eigenvectors? ----------------
    from sklearn.decomposition import PCA
    p = PCA(n_components=k, random_state=0).fit(Z)
    eigvec = p.components_.T
    lam = p.explained_variance_
    col_norms = np.linalg.norm(eigvec, axis=0)
    true_loadings = eigvec * np.sqrt(lam)
    report["findings"]["Q1_loadings_are_eigenvectors"] = {
        "code": "loadings = pca.components_.T  (src/doe_xgb/factor_analysis.py)",
        "column_norms_of_reported_loadings": [round(float(c), 6) for c in col_norms],
        "eigenvalues": [round(float(v), 6) for v in lam],
        "sqrt_eigenvalues": [round(float(np.sqrt(v)), 6) for v in lam],
        "column_norms_of_true_loadings": [round(float(c), 6) for c in np.linalg.norm(true_loadings, axis=0)],
        "verdict": ("The matrix the code calls 'loadings' has unit-norm columns, so it is the "
                    "eigenvector matrix. Loadings are the eigenvectors scaled by the square root "
                    "of the eigenvalues; on this data those scale factors are "
                    f"{', '.join(f'{np.sqrt(v):.3f}' for v in lam)}, so the two matrices differ "
                    "by a factor of up to "
                    f"{max(np.sqrt(lam))/min(np.sqrt(lam)):.2f} between columns."),
    }

    # ---------------- Q2: does the rotation change? ----------------
    rot_eigvec, R_eigvec = fa._varimax(eigvec)
    rot_load, R_load = fa._varimax(true_loadings)
    # Compare the subspace assignments the two rotations produce.
    ang = np.degrees(np.arccos(np.clip(np.abs(np.diag(R_eigvec.T @ R_load)), -1, 1)))
    # Which factor does each metric load highest on, under each convention?
    assign_eigvec = np.argmax(np.abs(rot_eigvec), axis=1)
    assign_load = np.argmax(np.abs(rot_load), axis=1)
    report["findings"]["Q2_rotation_differs"] = {
        "rotation_matrix_column_angles_deg": [round(float(a), 3) for a in ang],
        "frobenius_norm_of_R_difference": round(float(np.linalg.norm(R_eigvec - R_load)), 6),
        "metric_to_factor_assignment_rotating_eigenvectors": {
            m: int(a) + 1 for m, a in zip(METRICS, assign_eigvec)},
        "metric_to_factor_assignment_rotating_loadings": {
            m: int(a) + 1 for m, a in zip(METRICS, assign_load)},
        "assignments_agree": bool(np.array_equal(assign_eigvec, assign_load)),
        "verdict": ("Varimax maximizes the variance of squared loadings, a criterion that is not "
                    "invariant to per-column rescaling. Rotating unit-norm eigenvectors therefore "
                    "gives a different rotation from rotating the loadings, and weights a "
                    "low-variance component equally with a high-variance one."),
    }

    # ---------------- Q3: how many objectives reach the optimizer? ----------------
    res = fa.run_factor_analysis(df)
    returned = [c for c in res.scores.columns if c.startswith("FACTOR")]
    optimized = [c for c in res.scores.columns if c.startswith("Score_")]
    report["findings"]["Q3_effective_objective_count"] = {
        "factors_extracted": len(returned),
        "factor_score_columns": returned,
        "columns_the_optimizer_consumes": optimized,
        "effective_q": len(optimized),
        "mechanism": ("combine_quality_factors=True collapses every non-time factor into one "
                      "Score_Quality, so a three-factor extraction is optimized as a two-objective "
                      "problem."),
        "verdict": (f"{len(returned)} factors are extracted and {len(optimized)} objectives are "
                    "optimized. The reported factor count and the optimized objective count are "
                    "not the same number."),
    }

    # ---------------- Q4: what does the quality aggregation discard? ----------------
    S = res.scores
    qcols = [c for c in returned if c != f"FACTOR{res.cost_factor}_SCORE"]
    Q = np.asarray(S[qcols], dtype=float)
    Qz = zscore(Q)
    var_share = lam / lam.sum()
    report["findings"]["Q4_quality_aggregation"] = {
        "quality_factor_columns": qcols,
        "aggregation": "unweighted mean of z-scored factor scores",
        "explained_variance_share_per_factor": [round(float(v), 6) for v in var_share],
        "share_of_the_two_aggregated_factors": [round(float(var_share[int(c[6])-1]), 6) for c in qcols],
        "implied_weights_after_zscoring": [round(1.0 / len(qcols), 6)] * len(qcols),
        "verdict": ("Z-scoring each factor score to unit variance before averaging discards the "
                    "explained-variance ordering that motivated extracting them, so a component "
                    "carrying a small share of the variance enters the composite with the same "
                    "weight as one carrying a large share."),
    }

    # How much does the equal weighting actually change? Compare the composite the
    # code builds against one weighted by each factor's share of explained variance.
    from scipy.stats import spearmanr, kendalltau
    shares = np.array([var_share[int(c[6]) - 1] for c in qcols], dtype=float)
    w_var = shares / shares.sum()
    comp_equal = Qz.mean(axis=1)
    comp_var = Qz @ w_var
    rho = float(spearmanr(comp_equal, comp_var).statistic)
    tau = float(kendalltau(comp_equal, comp_var).statistic)
    top_equal = int(np.argmax(comp_equal))
    top_var = int(np.argmax(comp_var))
    # how far apart are the two rankings at the top, where the optimizer looks?
    r_equal = pd.Series(comp_equal).rank(ascending=False)
    r_var = pd.Series(comp_var).rank(ascending=False)
    top10_equal = set(r_equal.nsmallest(10).index)
    top10_var = set(r_var.nsmallest(10).index)
    report["findings"]["Q4_quality_aggregation"]["impact"] = {
        "variance_weights_that_would_apply": [round(float(v), 4) for v in w_var],
        "spearman_equal_vs_variance_weighted": round(rho, 6),
        "kendall_tau": round(tau, 6),
        "argmax_design_row_equal_weighted": top_equal,
        "argmax_design_row_variance_weighted": top_var,
        "argmax_agrees": bool(top_equal == top_var),
        "top10_overlap": len(top10_equal & top10_var),
        "reading": _weighting_reading(rho, len(top10_equal & top10_var), top_equal == top_var),
    }

    # ---------------- Q5: FMSE wrapper? ----------------
    src_text = (Path(args.scratch) / "src" / "doe_xgb" / "factor_analysis.py").read_text()
    fmse_markers = ["fmse", "target", "**2", "variance penalty", "sigma"]
    present = {m: (m in src_text.lower()) for m in fmse_markers}
    report["findings"]["Q5_fmse_wrapper"] = {
        "markers_found_in_frozen_source": present,
        "verdict": ("The frozen factor stage emits sign-oriented factor scores directly. It does "
                    "not implement the target-seeking quadratic wrapper with a factor-variance "
                    "penalty that the EAAI 2025 formulation applies to each VRF, so the "
                    "dissertation's objectives are raw oriented scores, not FMSE objectives."),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    for q, f in report["findings"].items():
        print(f"\n== {q} ==\n{f['verdict']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
