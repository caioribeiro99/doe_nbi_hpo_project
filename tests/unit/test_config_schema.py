"""Unit tests for the Pydantic v2 config schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from doe_xgb.config_schema import ExperimentConfig, load_config


CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _read(path: str) -> dict:
    with (CONFIGS / path).open("r") as f:
        return yaml.safe_load(f)


def test_dissertation_baseline_config_validates() -> None:
    cfg = ExperimentConfig.model_validate(_read("dissertation_baseline_xgb_magic.yaml"))
    assert cfg.experiment.name == "dissertation_baseline_xgb_magic"
    assert cfg.nbi.weights.q == 2
    assert cfg.factor_model.n_factors == 2


def test_article_3vrf_config_validates_and_is_3objective() -> None:
    cfg = ExperimentConfig.model_validate(_read("article_3vrf_xgb_magic.yaml"))
    assert cfg.nbi.weights.q == 3
    assert cfg.factor_model.n_factors == 3
    primaries = [s for s in cfg.objectives.specs if s.role.value == "primary_nbi"]
    assert len(primaries) == 3
    assert all(s.transform.value == "fmse" for s in primaries)


def test_arbitrary_metrics_config_has_no_factor_analysis() -> None:
    cfg = ExperimentConfig.model_validate(_read("article_arbitrary_metrics_example.yaml"))
    assert cfg.factor_model.mode == "none"
    primaries = [s for s in cfg.objectives.specs if s.role.value == "primary_nbi"]
    assert {s.direction.value for s in primaries} == {"maximize", "minimize"}


def test_postopt_demo_forces_post_optimization() -> None:
    cfg = ExperimentConfig.model_validate(_read("article_postopt_demo.yaml"))
    assert cfg.post_optimization.enabled == "always"


def test_unknown_top_level_key_rejected() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml") | {"i_should_not_exist": True}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_at_least_two_primary_objectives_required() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml")
    bad["objectives"]["specs"] = [bad["objectives"]["specs"][0]]
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_target_direction_requires_target_value() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml")
    # Strip the target to trigger the validation downstream of ObjectiveSpec.
    bad["objectives"]["specs"][0]["target"] = None
    bad["objectives"]["specs"][0]["transform"] = "raw"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_factor_model_fixed_requires_n_factors() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml")
    bad["factor_model"] = {"mode": "fixed"}  # no n_factors
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_factor_bounds_must_be_strictly_increasing() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml")
    bad["design"]["factors"][0]["lo"] = 1.0
    bad["design"]["factors"][0]["hi"] = 1.0
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_simplex_lattice_design_requires_q_and_m() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml")
    bad["design"] = {"kind": "simplex_lattice", "factors": []}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)


def test_load_config_from_file(tmp_path: Path) -> None:
    src = CONFIGS / "article_3vrf_xgb_magic.yaml"
    dst = tmp_path / "tmp.yaml"
    dst.write_text(src.read_text())
    cfg = load_config(dst)
    assert cfg.nbi.weights.q == 3


def test_external_csv_design_requires_external_path() -> None:
    bad = _read("article_3vrf_xgb_magic.yaml")
    bad["design"]["external_path"] = None
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)
