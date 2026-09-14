"""Sentinel: absurd validation responses must move nothing but the gate.

The strongest available proof that the 78 external points are audit-only is not an
inspection but an experiment: corrupt their responses beyond recognition and show
that every solution-producing artifact is bit-identical, while the external
diagnostic and the gate change.

The artifacts that must not move are the factor mapping, the response surfaces, the
anchors, the payoff matrix, the CHIM, the quasi-normal, every arm's candidates,
their realizations, the references and the normalization derived from them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.arms import (Realizer, empirical_reference, run_nbi_arm,
                                   run_ws_s, surrogate_reference, symmetric_weights)
from doe_xgb.campaign.design import (external_scores, external_validation_set,
                                     fit_surface_backward, gate_pass, load_design,
                                     make_surrogate)
from doe_xgb.campaign.evaluator import BOUNDS, INT_PARAMS, PARAMS, RESPONSES
from doe_xgb.campaign.factor_model import fit_factor_model
from doe_xgb.campaign.scoring import augmented_reference, indicators, reference_core
from doe_xgb.nbi_core import NBIConfig


def _synthetic_responses(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Deterministic pseudo-responses; no learner is fitted in this test."""
    rng = np.random.default_rng(seed)
    X = df[PARAMS].to_numpy(dtype=float)
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])
    C = 2 * (X - lo) / (hi - lo) - 1
    q = 0.8 + 0.05 * C[:, 3] - 0.02 * C[:, 4] ** 2
    p2 = 0.7 + 0.04 * C[:, 0] + 0.01 * C[:, 1]
    out = pd.DataFrame({
        "Accuracy_Mean": q, "Precision_Mean": p2, "Recall_Mean": q - 0.1,
        "Specificity_Mean": 1.8 - q - p2, "RocAuc_Mean": q + 0.05,
        "LogLoss_Mean": 0.6 - 0.3 * q,
        "Leaves_Mean": 200 + 900 * (C[:, 6] + 1) + 60 * (C[:, 4] + 1),
        "Time_MeanFold": 0.1 + 0.01 * rng.random(len(df)),
    })
    return pd.concat([df[PARAMS].reset_index(drop=True), out], axis=1)


def _solution_artifacts(design: pd.DataFrame, external: pd.DataFrame) -> dict:
    """Everything a method's solution depends on, plus the audit diagnostic."""
    fm = fit_factor_model(design)
    Y = fm.objectives(design)
    fits = [fit_surface_backward(design, Y[:, j]) for j in range(2)]
    surrogates = [make_surrogate(t, b) for t, b in fits]
    cfg = NBIConfig(objective_count=2, bounds=np.array([[-1.0, 1.0]] * len(PARAMS)),
                    n_starts=3, seed=5, maxiter=200)
    rz = Realizer(PARAMS, BOUNDS, list(INT_PARAMS))
    w = symmetric_weights(6)
    anchors, chim = surrogate_reference(surrogates, cfg)
    ws = run_ws_s(surrogates, cfg, anchors, rz, w)
    nbi = run_nbi_arm("NBI-S", surrogates, cfg, anchors, chim, rz, w)
    core = reference_core([design], fm.objectives)
    per = {"ws_s": fm.objectives(pd.DataFrame([c.config_realized for c in ws.candidates]
                                              ).assign(**{k: 0.0 for k in RESPONSES}))
           if False else np.zeros((0, 2))}
    aug = augmented_reference(core["front"], per)

    Yext = fm.objectives(external)
    audit = [external_scores(design, Y[:, j], external, Yext[:, j]) for j in range(2)]
    return {
        "solution": {
            "factor_mu": fm.mu.tolist(), "factor_sd": fm.sd.tolist(),
            "rotated_loadings": np.round(fm.rotated_loadings, 12).tolist(),
            "quality_weights": np.round(fm.quality_weights, 12).tolist(),
            "cost_index": fm.cost_index,
            "surface_terms": [[list(t) for t in f[0]] for f in fits],
            "surface_beta": [np.round(f[1], 12).tolist() for f in fits],
            "anchors": np.round(anchors.x_star, 10).tolist(),
            "payoff": np.round(anchors.F_star, 10).tolist(),
            "quasi_normal": np.round(chim.n_hat, 10).tolist(),
            "ws_candidates": [np.round(c.x_continuous, 9).tolist() for c in ws.candidates],
            "nbi_candidates": [np.round(c.x_continuous, 9).tolist() for c in nbi.candidates],
            "ws_realized": [c.config_realized for c in ws.candidates],
            "nbi_realized": [c.config_realized for c in nbi.candidates],
            "core_front": np.round(core["front"], 10).tolist(),
            "augmented_front": np.round(aug["front"], 10).tolist(),
        },
        "audit": {
            "external_r2": [round(a["external_r2"], 10) for a in audit],
            "external_spearman": [round(a["external_spearman"], 10) for a in audit],
            "gate_pass": [gate_pass(a) for a in audit],
        },
    }


@pytest.fixture(scope="module")
def frames():
    design = _synthetic_responses(load_design(), seed=1)
    external = _synthetic_responses(external_validation_set(), seed=2)
    return design, external


def test_absurd_validation_responses_change_nothing_a_method_uses(frames) -> None:
    design, external = frames
    baseline = _solution_artifacts(design, external)

    corrupted = external.copy()
    for c in RESPONSES:
        corrupted[c] = corrupted[c] * -1000.0 + 12345.0        # absurd on every scale
    after = _solution_artifacts(design, corrupted)

    assert after["solution"] == baseline["solution"], (
        "a solution-producing artifact moved when only the audit-only validation "
        "responses changed"
    )


def test_the_sentinel_can_actually_detect_a_change(frames) -> None:
    """If corrupting the DESIGN did not move the artifacts, the test proves nothing."""
    design, external = frames
    baseline = _solution_artifacts(design, external)
    corrupted_design = design.copy()
    corrupted_design["Accuracy_Mean"] = corrupted_design["Accuracy_Mean"] * -1.0
    after = _solution_artifacts(corrupted_design, external)
    assert after["solution"] != baseline["solution"]


def test_the_audit_diagnostic_does_respond_to_the_validation_responses(frames) -> None:
    """The external set must matter for the gate, or it is measuring nothing."""
    design, external = frames
    baseline = _solution_artifacts(design, external)
    corrupted = external.copy()
    for c in RESPONSES:
        corrupted[c] = corrupted[c] * -1000.0 + 12345.0
    after = _solution_artifacts(design, corrupted)
    assert after["audit"] != baseline["audit"]
