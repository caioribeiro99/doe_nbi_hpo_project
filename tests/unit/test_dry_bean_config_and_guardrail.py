"""Unit tests for the Dry Bean multiclass config and the
``validate_task_metric_compatibility`` wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from doe_xgb.config_schema import ExperimentConfig, load_config
from doe_xgb.datasets import (
    MultiClassNotConfiguredError,
    validate_task_metric_compatibility,
)
from doe_xgb.datasets.metadata import DatasetMetadata

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
DRY_BEAN_YAML = CONFIGS / "article_3vrf_dry_bean.yaml"


# ---------------------------------------------------------------------------
# Config presence and content
# ---------------------------------------------------------------------------


def test_dry_bean_config_file_exists() -> None:
    assert DRY_BEAN_YAML.exists()


def test_dry_bean_config_validates() -> None:
    cfg = load_config(DRY_BEAN_YAML)
    assert cfg.experiment.name == "article_3vrf_dry_bean_appendix"
    assert cfg.factor_model.n_factors == 3
    assert cfg.nbi.weights.q == 3


def test_dry_bean_config_uses_multiclass_response_keys() -> None:
    cfg = load_config(DRY_BEAN_YAML)
    expected = {
        "F1Macro_Mean",
        "BalancedAccuracy_Mean",
        "MCC_Mean",
        "ROCAUC_OVR_Mean",
        "PRAUC_OVR_Mean",
        "BrierMC_Mean",
        "ECE_Mean",
        "Time_MeanFold",
    }
    assert expected == set(cfg.evaluation.raw_metrics)


def test_dry_bean_config_objective_directions_are_explicit() -> None:
    cfg = load_config(DRY_BEAN_YAML)
    by_name = {s.name: s for s in cfg.objectives.specs}
    # Quality / probability metrics: maximize.
    for name in ("F1Macro_Mean", "BalancedAccuracy_Mean", "MCC_Mean",
                 "ROCAUC_OVR_Mean", "PRAUC_OVR_Mean"):
        assert by_name[name].direction.value == "maximize"
    # Calibration / cost metrics: minimize.
    for name in ("BrierMC_Mean", "ECE_Mean", "Time_MeanFold"):
        assert by_name[name].direction.value == "minimize"
    # FA / NBI primaries are FMSE-target, not raw direction.
    for name in ("VRF1", "VRF2", "VRF3"):
        assert by_name[name].direction.value == "target"
        assert by_name[name].transform.value == "fmse"


def test_dry_bean_config_xgboost_kwargs_are_multiclass() -> None:
    raw = yaml.safe_load(DRY_BEAN_YAML.read_text())
    fixed = raw["model"]["fixed_kwargs"]
    assert fixed.get("objective") == "multi:softprob"
    assert fixed.get("num_class") == 7


# ---------------------------------------------------------------------------
# Guardrail wrapper
# ---------------------------------------------------------------------------


def test_guardrail_passes_for_dry_bean_config_against_registry() -> None:
    cfg = load_config(DRY_BEAN_YAML)
    # Resolves dataset_id from the experiment name; should be 'dry_bean'
    # and the multiclass metric set must satisfy the assertion.
    resolved = validate_task_metric_compatibility(cfg)
    assert resolved == "dry_bean"


def test_guardrail_fails_when_dry_bean_uses_binary_defaults() -> None:
    raw = yaml.safe_load(DRY_BEAN_YAML.read_text())
    raw["evaluation"]["raw_metrics"] = [
        "Accuracy_Mean", "Precision_Mean", "Recall_Mean",
        "Specificity_Mean", "Time_MeanFold",
    ]
    cfg = ExperimentConfig.model_validate(raw)
    with pytest.raises(MultiClassNotConfiguredError) as excinfo:
        validate_task_metric_compatibility(cfg)
    assert excinfo.value.dataset_id == "dry_bean"
    assert "F1Macro_Mean" in excinfo.value.required_metric_set


def test_guardrail_passes_for_binary_dataset() -> None:
    """The headline binary config must still pass the guardrail."""
    cfg = load_config(CONFIGS / "article_3vrf_xgb_magic.yaml")
    resolved = validate_task_metric_compatibility(cfg)
    assert resolved == "magic"


def test_guardrail_explicit_dataset_metadata_override() -> None:
    """An override DatasetMetadata wins over experiment.name inference."""
    cfg = load_config(DRY_BEAN_YAML)
    fake = DatasetMetadata(
        dataset_id="zzz_synthetic",
        display_name="Synthetic",
        source_type="local",
        task_type="multiclass",
        target_column="y",
    )
    resolved = validate_task_metric_compatibility(cfg, dataset_metadata=fake)
    assert resolved == "zzz_synthetic"


def test_guardrail_explicit_dataset_id_override_for_binary_metrics_raises() -> None:
    """If the caller forces ``dataset_id='dry_bean'`` on a binary-metrics
    config, the guardrail should still raise."""
    cfg = load_config(CONFIGS / "article_3vrf_xgb_magic.yaml")
    with pytest.raises(MultiClassNotConfiguredError):
        validate_task_metric_compatibility(cfg, dataset_id="dry_bean")
