#!/usr/bin/env python
"""Centre-point sensitivity: the 88 design rows versus the 79 distinct configurations.

Question (coauthor review R2). The face-centred central composite design has 64
factorial rows, 14 axial rows and 10 centre rows, so 88 rows but 79 distinct
configurations. The surrogate regressions and the frozen factor models were fitted on
all 88 rows. If the ten centre rows carry identical responses, they add no
information about the shape of a surface, yet they weight the centre tenfold and
count toward the residual degrees of freedom of the backward-elimination t-tests.
This script measures how much that matters, using stored responses only.

Stage 1, the premise. For every confirmatory unit (dataset x replication) and for the
Stage A pilot rows that fixed the factor models: are the ten centre rows' recorded
responses identical, and are they one physical fit served ten times by the
evaluation cache or ten executions? The cache answers the second question directly:
its key is (dataset, split, fold, canonical configuration, seed, protocol version),
so a repeated configuration is served from the stored result of its first request.

Stage 2, the sensitivity. Holding the frozen per-dataset factor model and both
objective definitions fixed:
  A. the stored 88-row surrogate fits are reproduced exactly with the frozen
     ``design.fit_surface_backward`` (terms and coefficients), together with the
     stored external-audit scores, the gate decisions and the stored surrogate
     anchors (utopia point and payoff matrix) -- the script stops if any differs;
  B. the same procedure is refitted on the 79 distinct configurations, keeping the
     first centre row and dropping the other nine;
  C-E. the two fits are compared: selected terms, residual degrees of freedom,
     coefficients, basis rank and conditioning, predictions over a fixed evaluation
     set, external accuracy on the stored audit rows, gate decisions, and the
     surrogate anchors obtained by the frozen multistart minimizer
     (``arms.surrogate_reference``) on each surface.
  F. separately, the frozen factor model of each dataset is refitted on the 79
     distinct Stage A pilot configurations and compared with the committed one:
     loadings (Tucker congruence), role assignment and orientation, quality weights,
     explained variance, the linear coefficients of both objectives on the seven
     responses, and the agreement of the two objective definitions on stored rows.

What this is not. No learner is trained, no candidate is evaluated, and no NBI,
weighted-sum, NSGA-II, grid or Bayesian search is run: the anchors are minimized on
the polynomial surfaces only. Fronts are not recomputed, because judging a front
needs real revalidation of its candidates, i.e. new learner fits. The alternative
factor model replaces nothing and rescores nothing: it is compared with the frozen
one as a measurement definition. Neither alternative is an analysis the protocol
declared; both are post hoc diagnostics.

Descriptive additions. ``surrogate_chim_descriptive`` (the surrogate CHIM extent under
both fits and the extent ratio of Figure 3) and ``multistart_seed_control_88`` (the
88-row surfaces minimized again from a different set of multistart points) were added
after the first run, to size and locate the anchor changes that run found; neither
enters the classification.

Comparison criteria. Declared in ``CRITERIA`` below and committed before the script
was first run on the campaign records; ``classify`` applies them mechanically. The
reference scale is the objective unit: both objectives are factor scores
standardized on the Stage A rows, so one unit is one standard deviation there.

Provenance label: retrospective, post hoc; computed from stored outputs after the
campaign. Inputs (repo-relative):
    experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/
        design.json, surrogates.json, external_validation.json, nbi_s.json,
        evaluations.sqlite (read-only, immutable)
    papers/xgboost_hpo_vrfnbi/audits/pilot_stage_a/<dataset>_design.csv
    papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json
Output:
    papers/xgboost_hpo_vrfnbi/analysis/center_point_sensitivity.json

Usage, from the repository root:
    python -B papers/xgboost_hpo_vrfnbi/scripts/center_point_sensitivity.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.arms import surrogate_reference  # noqa: E402
from doe_xgb.campaign.design import (external_scores, fit_surface_backward,  # noqa: E402
                                     fit_surface_backward_uncoded, gate_pass,
                                     make_surrogate, surface_basis, surface_predict,
                                     surface_terms, to_coded)
from doe_xgb.campaign.evaluator import PARAMS, RESPONSES  # noqa: E402
from doe_xgb.campaign.factor_model import (FrozenFactorModel,  # noqa: E402
                                           apply_transforms, fit_factor_model,
                                           load_reference_factor_model,
                                           tucker_congruence)
from doe_xgb.campaign.runner import nbi_config, unit_seed  # noqa: E402

CAMPAIGN = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
PILOT = PAPER / "audits" / "pilot_stage_a"
OUT = PAPER / "analysis" / "center_point_sensitivity.json"
DATASETS = ("magic", "spambase", "adult", "bank_marketing")
N_REPS = 30
OBJECTIVES = ("quality", "cost")

# Declared before the first run on the campaign records. Not to be tuned to outcomes.
CRITERIA = {
    "evaluation_set": {
        "description": ("predictions of the two fits are compared on the 78 external "
                        "audit points, the 79 distinct design points and 4096 points "
                        "drawn uniformly from the coded box with a fixed seed"),
        "uniform_points": 4096,
        "seed": 20261013,
    },
    "term_stable": "symmetric difference of the selected term sets <= 2 terms",
    "prediction_stable": ("RMS prediction difference over the evaluation set <= 0.25 x "
                          "the external RMSE of the 88-row fit, AND Spearman between "
                          "the two fits' predictions over the evaluation set >= 0.99"),
    "anchor_changed": ("the coded location of the objective's surrogate minimizer moves "
                       "by more than 0.10 x the coded box diagonal (2 sqrt(7)), OR any "
                       "payoff-matrix entry changes by more than 0.10 objective units"),
    "factor_low": ("same cost-factor assignment and orientation, min |Tucker| >= 0.95 "
                   "over the three matched factors, |change in either quality weight| "
                   "<= 0.05, and Spearman >= 0.99 between the frozen and alternative "
                   "values of each objective on the 79 distinct pilot configurations "
                   "and on the pooled stored external audit rows"),
    "factor_high": ("cost-factor assignment or orientation changes, or min |Tucker| "
                    "< 0.85, or Spearman < 0.90 for either objective on either row set"),
    "dataset_high": ("factor_high; OR a gate decision flips in >= 10 of 30 replications "
                     "for either objective; OR the anchors change (anchor_changed) in "
                     "more than 15 of 30 replications for either objective"),
    "dataset_moderate": ("not high, and any of: factor model not factor_low; more than "
                         "3 of 30 fits not term_stable or not prediction_stable for "
                         "either objective; any gate flip; anchors change in more than "
                         "3 of 30 replications for either objective"),
    "dataset_low": "neither high nor moderate",
    "overall": "the most severe dataset class",
    "thresholds": {"term_symdiff_max": 2, "rms_over_external_rmse_max": 0.25,
                   "prediction_spearman_min": 0.99, "anchor_distance_frac_max": 0.10,
                   "payoff_change_max": 0.10, "tucker_low": 0.95, "tucker_high": 0.85,
                   "weight_change_max": 0.05, "objective_spearman_low": 0.99,
                   "objective_spearman_high": 0.90, "gate_flip_high": 10,
                   "anchor_change_high": 15, "moderate_count": 3},
}
T = CRITERIA["thresholds"]
DIAGONAL = 2.0 * np.sqrt(len(PARAMS))


def term_name(tm) -> str:
    if not tm:
        return "1"
    if len(tm) == 1:
        return PARAMS[tm[0]]
    if tm[0] == tm[1]:
        return f"{PARAMS[tm[0]]}^2"
    return f"{PARAMS[tm[0]]}*{PARAMS[tm[1]]}"


def centre_mask(df: pd.DataFrame) -> np.ndarray:
    """The ten centre rows: the only repeated configuration in the design.

    Not "coded zero": max_depth is an integer, so the centre's 10.5 is realized as an
    integer and its coded value is off zero. The design's only repeated configuration
    is located instead, and required to be the centre (every coded |value| <= 0.1).
    """
    key = df[list(PARAMS)].astype(float).round(9).astype(str).agg("|".join, axis=1)
    counts = key.map(key.value_counts())
    if sorted(set(counts[counts > 1])) != [10] or key[counts > 1].nunique() != 1:
        raise SystemExit("expected exactly one configuration repeated ten times")
    mask = (counts > 1).to_numpy()
    if np.abs(to_coded(df[mask])).max() > 0.1:
        raise SystemExit("the repeated configuration is not the design centre")
    return mask


def distinct_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first centre row and drop the other nine; every other row is distinct."""
    c = centre_mask(df)
    first = int(np.flatnonzero(c)[0])
    keep = ~c
    keep[first] = True
    out = df[keep].reset_index(drop=True)
    if len(out) != 79 or len(out.drop_duplicates(subset=list(PARAMS))) != 79:
        raise SystemExit(f"expected 79 distinct configurations, found {len(out)}")
    return out


def centre_spread(df: pd.DataFrame) -> dict:
    c = df[centre_mask(df)]
    cols = list(RESPONSES) + (["Time_MeanFold"] if "Time_MeanFold" in df else [])
    return {"n_centre_rows": int(len(c)),
            "max_abs_spread": {k: float(c[k].max() - c[k].min()) for k in cols}}


def design_cache_record(unit: Path) -> dict:
    db = sqlite3.connect(f"file:{unit / 'evaluations.sqlite'}?mode=ro&immutable=1", uri=True)
    try:
        rows = db.execute("SELECT key, served_from_cache, computed FROM requests "
                          "WHERE method = 'design'").fetchall()
    finally:
        db.close()
    return {"design_requests": len(rows),
            "distinct_keys": len({r[0] for r in rows}),
            "computed": int(sum(r[2] for r in rows)),
            "served_from_cache": int(sum(r[1] for r in rows))}


def fit_stats(design: pd.DataFrame, terms) -> dict:
    A = surface_basis(to_coded(design), terms)
    full = surface_basis(to_coded(design), surface_terms())
    return {"n_rows": int(len(design)), "n_terms": int(len(terms)),
            "residual_dof": int(len(design) - len(terms)),
            "rank": int(np.linalg.matrix_rank(A)),
            "condition_number": float(np.linalg.cond(A)),
            "full_quadratic_rank": int(np.linalg.matrix_rank(full)),
            "full_quadratic_condition_number": float(np.linalg.cond(full))}


def evaluation_set(external: pd.DataFrame, design79: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(CRITERIA["evaluation_set"]["seed"])
    U = rng.uniform(-1.0, 1.0, size=(CRITERIA["evaluation_set"]["uniform_points"],
                                     len(PARAMS)))
    return np.vstack([to_coded(external), to_coded(design79), U])


SEED_CONTROL_OFFSET = 7919   # any fixed offset; the control only needs a different start set


def anchors_of(fits, seed: int) -> dict:
    anchors, chim = surrogate_reference([make_surrogate(t, b) for t, b in fits],
                                        nbi_config(seed))
    return {"x_star": np.asarray(anchors.x_star, float),
            "F_star": np.asarray(anchors.F_star, float),
            "utopia": np.asarray(anchors.utopia, float),
            "n_hat": np.asarray(chim.n_hat, float)}


def compare_unit(ds: str, rep: int, fm: FrozenFactorModel) -> dict:
    unit = CAMPAIGN / ds / f"rep_{rep:02d}"
    design = pd.DataFrame(json.loads((unit / "design.json").read_text())["rows"])
    stored_fit = json.loads((unit / "surrogates.json").read_text())
    stored_ext = json.loads((unit / "external_validation.json").read_text())
    stored_nbi = json.loads((unit / "nbi_s.json").read_text())["diagnostics"]
    ext = pd.DataFrame(stored_ext["rows"])
    seed = unit_seed(ds, rep)

    Y = fm.objectives(design)
    Ye = fm.objectives(ext)
    d79 = distinct_rows(design)
    Y79 = fm.objectives(d79)

    # A. exact reproduction of the stored 88-row stage
    fits88 = [fit_surface_backward(design, Y[:, j]) for j in range(2)]
    for j, (t, b) in enumerate(fits88):
        st = [tuple(x) for x in stored_fit["terms"][j]]
        if st != list(t) or float(np.abs(np.asarray(stored_fit["beta"][j]) - b).max()) > 1e-9:
            raise SystemExit(f"{ds} rep {rep} {OBJECTIVES[j]}: stored surrogate not reproduced")
    sc88 = [external_scores(design, Y[:, j], ext, Ye[:, j]) for j in range(2)]
    for j in range(2):
        for k in ("external_r2", "external_spearman", "external_rmse"):
            if abs(sc88[j][k] - stored_ext["scores"][j][k]) > 1e-9:
                raise SystemExit(f"{ds} rep {rep}: stored external {k} not reproduced")
        if gate_pass(sc88[j]) != stored_ext["gate_pass"][j]:
            raise SystemExit(f"{ds} rep {rep}: stored gate decision not reproduced")
    an88 = anchors_of(fits88, seed)
    if (np.abs(an88["utopia"] - np.asarray(stored_nbi["utopia"])).max() > 1e-7
            or np.abs(an88["F_star"] - an88["utopia"][:, None]
                      - np.asarray(stored_nbi["payoff_matrix"])).max() > 1e-7):
        raise SystemExit(f"{ds} rep {rep}: stored surrogate anchors not reproduced")

    # B. the 79-row variant, same frozen objectives
    fits79 = [fit_surface_backward(d79, Y79[:, j]) for j in range(2)]
    sc79 = [external_scores(d79, Y79[:, j], ext, Ye[:, j]) for j in range(2)]
    an79 = anchors_of(fits79, seed)
    E = evaluation_set(ext, d79)

    per_obj = {}
    for j, name in enumerate(OBJECTIVES):
        (t88, b88), (t79, b79) = fits88[j], fits79[j]
        s88, s79 = set(t88), set(t79)
        union = sorted(s88 | s79, key=lambda tm: (len(tm), tm))
        c88 = {tm: float(v) for tm, v in zip(t88, b88)}
        c79 = {tm: float(v) for tm, v in zip(t79, b79)}
        coef_diff = [abs(c88.get(tm, 0.0) - c79.get(tm, 0.0)) for tm in union]
        p88 = surface_predict(t88, b88, E)
        p79 = surface_predict(t79, b79, E)
        rms = float(np.sqrt(np.mean((p88 - p79) ** 2)))
        rho = float(spearmanr(p88, p79).statistic)
        dist = float(np.linalg.norm(an88["x_star"][j] - an79["x_star"][j]))
        f88 = make_surrogate(t88, b88)
        f79 = make_surrogate(t79, b79)
        regret88 = float(f88(an79["x_star"][j]) - f88(an88["x_star"][j]))
        regret79 = float(f79(an88["x_star"][j]) - f79(an79["x_star"][j]))
        payoff_change = float(np.abs(an88["F_star"][:, j] - an79["F_star"][:, j]).max())
        per_obj[name] = {
            "terms_88": [term_name(tm) for tm in t88],
            "terms_79": [term_name(tm) for tm in t79],
            "terms_identical": s88 == s79,
            "terms_only_in_88": [term_name(tm) for tm in sorted(s88 - s79)],
            "terms_only_in_79": [term_name(tm) for tm in sorted(s79 - s88)],
            "term_symmetric_difference": int(len(s88 ^ s79)),
            "fit_88": fit_stats(design, t88),
            "fit_79": fit_stats(d79, t79),
            "max_abs_coefficient_change": float(max(coef_diff)),
            "intercept_change": float(abs(c88.get((), 0.0) - c79.get((), 0.0))),
            "prediction_rms_difference": rms,
            "prediction_max_abs_difference": float(np.abs(p88 - p79).max()),
            "prediction_spearman": rho,
            "rms_over_external_rmse": rms / sc88[j]["external_rmse"],
            "external_88": {k: sc88[j][k] for k in ("external_r2", "external_spearman",
                                                    "external_rmse")},
            "external_79": {k: sc79[j][k] for k in ("external_r2", "external_spearman",
                                                    "external_rmse")},
            "gate_88": gate_pass(sc88[j]),
            "gate_79": gate_pass(sc79[j]),
            "anchor_x_88": an88["x_star"][j].tolist(),
            "anchor_x_79": an79["x_star"][j].tolist(),
            "anchor_distance": dist,
            "anchor_distance_frac_diagonal": dist / DIAGONAL,
            "anchor_regret_on_88_surface": regret88,
            "anchor_regret_on_79_surface": regret79,
            "payoff_column_max_abs_change": payoff_change,
            "term_stable": len(s88 ^ s79) <= T["term_symdiff_max"],
            "prediction_stable": (rms <= T["rms_over_external_rmse_max"] * sc88[j]["external_rmse"]
                                  and rho >= T["prediction_spearman_min"]),
            "anchor_changed": (dist / DIAGONAL > T["anchor_distance_frac_max"]
                               or payoff_change > T["payoff_change_max"]),
        }
    # the dissertation's own uncoded fit (HISTORICAL-WS-asrun), terms only, descriptive
    hist = {}
    for j, name in enumerate(OBJECTIVES):
        n88, _ = fit_surface_backward_uncoded(design, Y[:, j])
        n79, _ = fit_surface_backward_uncoded(d79, Y79[:, j])
        hist[name] = {"n_terms_88": len(n88), "n_terms_79": len(n79),
                      "terms_identical": set(n88) == set(n79),
                      "term_symmetric_difference": len(set(n88) ^ set(n79))}
    # Descriptive, added after the first run; it does not enter ``classify``. The CHIM
    # extent ||Phi_1 - Phi_2|| of each surrogate reference, and the extent ratio of
    # Figure 3 (NBI-R's stored empirical extent over NBI-S's surrogate extent) under
    # both fits. NBI-R's payoff comes from real measurements and is unchanged.
    phi88 = an88["F_star"] - an88["utopia"][:, None]
    phi79 = an79["F_star"] - an79["utopia"][:, None]
    phi_r = np.asarray(json.loads((unit / "nbi_r.json").read_text())["diagnostics"]["payoff_matrix"])
    ext88 = float(np.linalg.norm(phi88[:, 0] - phi88[:, 1]))
    ext79 = float(np.linalg.norm(phi79[:, 0] - phi79[:, 1]))
    ext_r = float(np.linalg.norm(phi_r[:, 0] - phi_r[:, 1]))
    chim = {"payoff_88": phi88.tolist(), "payoff_79": phi79.tolist(),
            "chim_extent_88": ext88, "chim_extent_79": ext79,
            "chim_extent_ratio_79_over_88": ext79 / ext88,
            "nbi_r_over_nbi_s_extent_88": ext_r / ext88,
            "nbi_r_over_nbi_s_extent_79": ext_r / ext79}
    # Descriptive control, added after the first run; not in ``classify``. The SAME
    # 88-row surfaces, minimized from a different set of multistart points. If the
    # anchors move as often under this as under the 79-row refit, the movement is a
    # property of the surfaces' flat optima, not of the centre-row weighting.
    an88b = anchors_of(fits88, seed + SEED_CONTROL_OFFSET)
    seed_control = {}
    for j, name in enumerate(OBJECTIVES):
        dist_b = float(np.linalg.norm(an88["x_star"][j] - an88b["x_star"][j]))
        pay_b = float(np.abs(an88["F_star"][:, j] - an88b["F_star"][:, j]).max())
        seed_control[name] = {
            "anchor_distance_frac_diagonal": dist_b / DIAGONAL,
            "payoff_column_max_abs_change": pay_b,
            "anchor_changed": (dist_b / DIAGONAL > T["anchor_distance_frac_max"]
                               or pay_b > T["payoff_change_max"])}
    nhat_angle = float(np.degrees(np.arccos(np.clip(
        an88["n_hat"] @ an79["n_hat"]
        / (np.linalg.norm(an88["n_hat"]) * np.linalg.norm(an79["n_hat"])), -1.0, 1.0))))
    return {"dataset": ds, "replication": rep,
            "premise": {**centre_spread(design), "cache": design_cache_record(unit)},
            "objectives": per_obj,
            "utopia_88": an88["utopia"].tolist(), "utopia_79": an79["utopia"].tolist(),
            "quasi_normal_angle_degrees": nhat_angle,
            "surrogate_chim_descriptive": chim,
            "multistart_seed_control_88": seed_control,
            "historical_uncoded_terms": hist}


def objective_coefficients(fm: FrozenFactorModel) -> dict:
    """Each objective is affine in the transformed responses; per-SD coefficients."""
    A = (np.diag(1.0 / fm.sd) @ fm.components.T @ np.diag(1.0 / np.sqrt(fm.eigenvalues))
         @ fm.rotation @ np.diag(1.0 / fm.score_sd))
    q = A[:, list(fm.quality_indices)] @ fm.quality_weights
    c = A[:, fm.cost_index]
    return {"quality": q, "cost": c}


def compare_factor(ds: str) -> dict:
    pilot = pd.read_csv(PILOT / f"{ds}_design.csv")
    frozen = load_reference_factor_model(ds)
    refit88 = fit_factor_model(pilot)
    # the committed artifact is reproduced from the 88 pilot rows before comparing
    if (refit88.cost_index != frozen.cost_index
            or np.abs(refit88.rotated_loadings - frozen.rotated_loadings).max() > 1e-9
            or np.abs(refit88.quality_weights - frozen.quality_weights).max() > 1e-9):
        raise SystemExit(f"{ds}: the committed factor model is not reproduced from the pilot rows")
    p79 = distinct_rows(pilot)
    alt = fit_factor_model(p79)
    phi = tucker_congruence(frozen.rotated_loadings, alt.rotated_loadings)
    # matching and orientation, by the same greedy rule ``tucker_congruence`` uses
    na = np.linalg.norm(frozen.rotated_loadings, axis=0)
    nb = np.linalg.norm(alt.rotated_loadings, axis=0)
    C = (frozen.rotated_loadings.T @ alt.rotated_loadings) / np.outer(na, nb)
    used, match = set(), []
    for j in range(C.shape[0]):
        pick = next(i for i in np.argsort(-np.abs(C[j])) if i not in used)
        used.add(int(pick))
        match.append(int(pick))
    cost_same = match[frozen.cost_index] == alt.cost_index
    cost_oriented = bool(C[frozen.cost_index, alt.cost_index] > 0)
    quality_oriented = all(C[j, match[j]] > 0 for j in frozen.quality_indices)
    ext = pd.concat([pd.DataFrame(json.loads(
        (CAMPAIGN / ds / f"rep_{r:02d}" / "external_validation.json").read_text())["rows"])
        for r in range(N_REPS)], ignore_index=True)
    agree = {}
    for label, rows in (("pilot_79_distinct", p79), ("pooled_external_audit_rows", ext)):
        a, b = frozen.objectives(rows), alt.objectives(rows)
        agree[label] = {"n_rows": int(len(rows)),
                        "spearman": [float(spearmanr(a[:, j], b[:, j]).statistic) for j in range(2)],
                        "pearson": [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(2)]}
    cf, ca = objective_coefficients(frozen), objective_coefficients(alt)
    names = list(frozen.response_names)
    coeffs = {}
    for o in OBJECTIVES:
        # per standard deviation of each transformed response (frozen standardization)
        u, v = cf[o] * frozen.sd, ca[o] * frozen.sd
        coeffs[o] = {"frozen": dict(zip(names, u.round(6).tolist())),
                     "alternative": dict(zip(names, v.round(6).tolist())),
                     "cosine": float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))),
                     "max_abs_change": float(np.abs(u - v).max())}
    # each frozen quality weight against the weight of its matched alternative factor
    w_change = float("inf")
    if all(match[j] in alt.quality_indices for j in frozen.quality_indices):
        w_change = float(max(
            abs(frozen.quality_weights[a] - alt.quality_weights[alt.quality_indices.index(match[j])])
            for a, j in enumerate(frozen.quality_indices)))
    min_phi = float(np.abs(phi).min())
    sp_min = min(min(v["spearman"]) for v in agree.values())
    low = (cost_same and cost_oriented and quality_oriented and min_phi >= T["tucker_low"]
           and w_change <= T["weight_change_max"] and sp_min >= T["objective_spearman_low"])
    high = (not (cost_same and cost_oriented and quality_oriented)
            or min_phi < T["tucker_high"] or sp_min < T["objective_spearman_high"])
    return {"dataset": ds,
            "pilot_premise": centre_spread(pilot),
            "frozen_reproduced_from_88_pilot_rows": True,
            "tucker_congruence_per_frozen_factor": phi.tolist(),
            "min_abs_tucker": min_phi,
            "factor_matching": match,
            "cost_factor_same": bool(cost_same), "cost_orientation_same": cost_oriented,
            "quality_orientation_same": bool(quality_oriented),
            "dominant_response_frozen": frozen.diagnostics.get("dominant_response_per_factor"),
            "dominant_response_alternative": alt.diagnostics["dominant_response_per_factor"],
            "quality_weights_frozen": frozen.quality_weights.tolist(),
            "quality_weights_alternative": alt.quality_weights.tolist(),
            "max_quality_weight_change": w_change,
            "explained_share_unrotated_frozen": (frozen.eigenvalues / frozen.eigenvalues.sum()).tolist(),
            "eigenvalues_full_frozen": frozen.diagnostics.get("eigenvalues_full"),
            "eigenvalues_full_alternative": alt.diagnostics["eigenvalues_full"],
            "kaiser_frozen": frozen.diagnostics.get("kaiser_retained"),
            "kaiser_alternative": alt.diagnostics["kaiser_retained"],
            "rotated_ss_share_frozen": frozen.diagnostics.get("rotated_ss_loading_share"),
            "rotated_ss_share_alternative": alt.diagnostics["rotated_ss_loading_share"],
            "objective_coefficients_per_response_sd": coeffs,
            "objective_agreement": agree,
            "min_objective_spearman": sp_min,
            "factor_class": "low" if low else ("high" if high else "moderate")}


def classify(units: list[dict], factor: dict) -> dict:
    out = {"factor_class": factor["factor_class"]}
    high = factor["factor_class"] == "high"
    moderate = factor["factor_class"] != "low"
    for o in OBJECTIVES:
        rs = [u["objectives"][o] for u in units]
        n = {"terms_identical": sum(r["terms_identical"] for r in rs),
             "term_stable": sum(r["term_stable"] for r in rs),
             "prediction_stable": sum(r["prediction_stable"] for r in rs),
             "gate_flips": sum(r["gate_88"] != r["gate_79"] for r in rs),
             "gate_pass_88": sum(r["gate_88"] for r in rs),
             "gate_pass_79": sum(r["gate_79"] for r in rs),
             "anchor_changed": sum(r["anchor_changed"] for r in rs)}
        out[o] = n
        high |= n["gate_flips"] >= T["gate_flip_high"] or n["anchor_changed"] > T["anchor_change_high"]
        moderate |= (len(rs) - n["term_stable"] > T["moderate_count"]
                     or len(rs) - n["prediction_stable"] > T["moderate_count"]
                     or n["gate_flips"] > 0 or n["anchor_changed"] > T["moderate_count"])
    out["dataset_class"] = "high" if high else ("moderate" if moderate else "low")
    return out


def summarize(units: list[dict]) -> dict:
    s = {}
    for o in OBJECTIVES:
        rs = [u["objectives"][o] for u in units]
        arr = lambda k: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        d88 = np.array([r["fit_88"]["residual_dof"] for r in rs])
        d79 = np.array([r["fit_79"]["residual_dof"] for r in rs])
        s[o] = {
            "n_terms_88_median": float(np.median([r["fit_88"]["n_terms"] for r in rs])),
            "n_terms_79_median": float(np.median([r["fit_79"]["n_terms"] for r in rs])),
            "residual_dof_88_range": [int(d88.min()), int(d88.max())],
            "residual_dof_79_range": [int(d79.min()), int(d79.max())],
            "term_symmetric_difference_max": int(arr("term_symmetric_difference").max()),
            "max_abs_coefficient_change_median": float(np.median(arr("max_abs_coefficient_change"))),
            "max_abs_coefficient_change_max": float(arr("max_abs_coefficient_change").max()),
            "prediction_rms_difference_median": float(np.median(arr("prediction_rms_difference"))),
            "prediction_rms_difference_max": float(arr("prediction_rms_difference").max()),
            "prediction_max_abs_difference_max": float(arr("prediction_max_abs_difference").max()),
            "prediction_spearman_min": float(arr("prediction_spearman").min()),
            "rms_over_external_rmse_max": float(arr("rms_over_external_rmse").max()),
            "external_r2_change_max_abs": float(np.abs(
                [r["external_79"]["external_r2"] - r["external_88"]["external_r2"] for r in rs]).max()),
            "external_spearman_change_max_abs": float(np.abs(
                [r["external_79"]["external_spearman"] - r["external_88"]["external_spearman"]
                 for r in rs]).max()),
            "external_r2_88_median": float(np.median([r["external_88"]["external_r2"] for r in rs])),
            "external_r2_79_median": float(np.median([r["external_79"]["external_r2"] for r in rs])),
            "anchor_distance_frac_median": float(np.median(arr("anchor_distance_frac_diagonal"))),
            "anchor_distance_frac_max": float(arr("anchor_distance_frac_diagonal").max()),
            "payoff_change_max": float(arr("payoff_column_max_abs_change").max()),
            "anchor_regret_on_88_surface_max": float(arr("anchor_regret_on_88_surface").max()),
            "full_quadratic_condition_88_max": float(max(r["fit_88"]["full_quadratic_condition_number"] for r in rs)),
            "full_quadratic_condition_79_max": float(max(r["fit_79"]["full_quadratic_condition_number"] for r in rs)),
            "full_quadratic_rank_min_79": int(min(r["fit_79"]["full_quadratic_rank"] for r in rs)),
        }
    ang = np.array([u["quasi_normal_angle_degrees"] for u in units])
    s["quasi_normal_angle_degrees_median"] = float(np.median(ang))
    s["quasi_normal_angle_degrees_max"] = float(ang.max())
    s["quasi_normal_angle_over_5_degrees"] = int((ang > 5.0).sum())
    ch = [u["surrogate_chim_descriptive"] for u in units]
    er = np.array([c["chim_extent_ratio_79_over_88"] for c in ch])
    r88 = np.array([c["nbi_r_over_nbi_s_extent_88"] for c in ch])
    r79 = np.array([c["nbi_r_over_nbi_s_extent_79"] for c in ch])
    s["surrogate_chim_extent_ratio_79_over_88"] = {
        "median": float(np.median(er)), "min": float(er.min()), "max": float(er.max()),
        "outside_0.9_to_1.1": int(((er < 0.9) | (er > 1.1)).sum())}
    s["multistart_seed_control_anchor_changed"] = {
        o: int(sum(u["multistart_seed_control_88"][o]["anchor_changed"] for u in units))
        for o in OBJECTIVES}
    s["figure3_extent_ratio_median_88"] = float(np.median(r88))
    s["figure3_extent_ratio_median_79"] = float(np.median(r79))
    s["figure3_extent_ratio_range_79"] = [float(r79.min()), float(r79.max())]
    s["historical_uncoded_terms_identical"] = {
        o: int(sum(u["historical_uncoded_terms"][o]["terms_identical"] for u in units))
        for o in OBJECTIVES}
    return s


def _plain(o):
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o).__name__)


def main() -> None:
    result = {"provenance": ("retrospective, post hoc diagnostic from stored outputs (coauthor "
                             "review R2); no learner, candidate evaluation or optimizer campaign; "
                             "the alternative fits replace nothing"),
              "criteria": CRITERIA, "datasets": {}}
    worst = "low"
    rank = {"low": 0, "moderate": 1, "high": 2}
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        units = [compare_unit(ds, r, fm) for r in range(N_REPS)]
        prem = [u["premise"] for u in units]
        factor = compare_factor(ds)
        cls = classify(units, factor)
        worst = max(worst, cls["dataset_class"], key=rank.get)
        result["datasets"][ds] = {
            "premise": {
                "centre_rows_per_unit": sorted({p["n_centre_rows"] for p in prem}),
                "max_centre_response_spread": max(max(p["max_abs_spread"].values()) for p in prem),
                "cache_design_requests": sorted({p["cache"]["design_requests"] for p in prem}),
                "cache_distinct_keys": sorted({p["cache"]["distinct_keys"] for p in prem}),
                "cache_computed": sorted({p["cache"]["computed"] for p in prem}),
                "cache_served": sorted({p["cache"]["served_from_cache"] for p in prem}),
            },
            "reproduced_88_row_stage": True,
            "summary": summarize(units),
            "counts": cls,
            "factor_model": factor,
            "units": units,
        }
        print(ds, cls)
    result["overall_class"] = worst
    OUT.write_text(json.dumps(result, indent=1, default=_plain) + "\n")
    print("overall", worst, "->", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
