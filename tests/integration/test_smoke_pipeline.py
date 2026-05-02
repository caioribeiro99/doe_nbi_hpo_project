"""End-to-end smoke pipeline on synthetic data.

Exercises (in order): YAML config validation -> N-objective NBI core
-> selection -> conditional MBPA. Designed to finish in well under
30 seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doe_xgb.config_schema import load_config
from doe_xgb.diagnostics import PostOptTriggerThresholds
from doe_xgb.nbi_core import NBIConfig, run_nbi
from doe_xgb.post_optimization import MBPASpec, run_mbpa
from doe_xgb.selection import SelectionRule, select
from doe_xgb.simplex import generate_simplex_lattice

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.smoke
def test_smoke_q2_pipeline_validates_and_runs() -> None:
    cfg = load_config(CONFIGS / "dissertation_baseline_xgb_magic.yaml")
    assert cfg.nbi.weights.q == 2

    # Synthetic surrogates parametrized by the config's q.
    q = cfg.nbi.weights.q
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)

    def make_obj(t):
        return lambda x: float(np.dot(x - t, x - t))

    surrogates = [make_obj(targets[i]) for i in range(q)]
    nbi_cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=42)
    weights = generate_simplex_lattice(q, 5)
    run = run_nbi(surrogates, weights, nbi_cfg)
    assert len(run.candidates) > 0
    F_at_x = np.array([c.F_at_x for c in run.candidates])

    idx, info = select(F_at_x, SelectionRule.DISTANCE_TO_UTOPIA)
    assert 0 <= idx < len(F_at_x)
    assert info["rule"] == "distance_to_utopia"


@pytest.mark.smoke
def test_smoke_q3_pipeline_with_conditional_mbpa() -> None:
    cfg = load_config(CONFIGS / "article_3vrf_xgb_magic.yaml")
    assert cfg.nbi.weights.q == 3
    assert cfg.post_optimization.enabled in {"always", "conditional"}

    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)

    def make_obj(t):
        return lambda x: float(np.dot(x - t, x - t))

    surrogates = [make_obj(targets[i]) for i in range(q)]
    nbi_cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=0, maxiter=300)
    weights = generate_simplex_lattice(q, 4)
    primary = run_nbi(surrogates, weights, nbi_cfg)
    F_at_x = np.array([c.F_at_x for c in primary.candidates])
    assert F_at_x.shape == (15, 3)

    spec = MBPASpec(inner_simplex_q=q, inner_simplex_m=6, elliptical_radii=(0.5, 0.5, 0.5))
    res = run_mbpa(primary, spec, thresholds=PostOptTriggerThresholds(), enabled="conditional")
    assert isinstance(res.diagnostics.summary, dict)


@pytest.mark.smoke
def test_smoke_residuals_below_tolerance() -> None:
    """Smoke residual budget for q=3."""
    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)

    def make_obj(t):
        return lambda x: float(np.dot(x - t, x - t))

    surrogates = [make_obj(targets[i]) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=0, maxiter=300)
    weights = generate_simplex_lattice(q, 4)
    run = run_nbi(surrogates, weights, cfg)
    residuals = [c.residual_norm for c in run.candidates]
    assert max(residuals) < 1e-2
