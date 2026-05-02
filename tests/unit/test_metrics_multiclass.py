"""Unit tests for the multiclass metrics extension and the unified
classification dispatcher."""

from __future__ import annotations

import math

import numpy as np
import pytest

from doe_xgb.evaluation import (
    MultiClassNotConfiguredError,
    assert_metric_set_compatible_with_task,
)
from doe_xgb.metrics import (
    aggregate_metric_dicts,
    compute_binary_metrics,
    compute_classification_metrics,
    compute_multiclass_metrics,
)

# ---------------------------------------------------------------------------
# Binary backward-compat
# ---------------------------------------------------------------------------


def test_binary_path_unchanged() -> None:
    """The dissertation-era binary path keys / values are unchanged."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    fm = compute_binary_metrics(y_true, y_pred)
    assert fm.accuracy == pytest.approx(4 / 6)
    # precision = TP / (TP + FP) = 2 / (2 + 1) = 0.667
    assert fm.precision == pytest.approx(2 / 3)


def test_dispatcher_binary_emits_dissertation_keys() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    out = compute_classification_metrics(y_true, y_pred)
    assert out["task"] == "binary"
    assert {"accuracy", "precision", "recall", "specificity"}.issubset(out)
    assert "f1_macro" not in out  # multiclass-only key absent


def test_aggregate_metric_dicts_binary_yields_legacy_keys() -> None:
    folds = [
        {"task": "binary", "accuracy": 0.9, "precision": 0.8,
         "recall": 0.85, "specificity": 0.95, "warnings": []},
        {"task": "binary", "accuracy": 0.8, "precision": 0.7,
         "recall": 0.75, "specificity": 0.85, "warnings": []},
    ]
    agg = aggregate_metric_dicts(folds)
    assert agg == {
        "Accuracy_Mean": pytest.approx(0.85),
        "Precision_Mean": pytest.approx(0.75),
        "Recall_Mean": pytest.approx(0.80),
        "Specificity_Mean": pytest.approx(0.90),
    }


# ---------------------------------------------------------------------------
# Multiclass hard-label metrics
# ---------------------------------------------------------------------------


def test_multiclass_hard_labels_three_classes() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    out = compute_multiclass_metrics(y_true, y_pred)
    assert out["accuracy"] == pytest.approx(4 / 6)
    assert 0.0 <= out["f1_macro"] <= 1.0
    assert 0.0 <= out["balanced_accuracy"] <= 1.0
    assert -1.0 <= out["mcc"] <= 1.0
    # Probability-based metrics are NaN without y_prob.
    assert math.isnan(out["roc_auc_ovr_macro"])
    assert math.isnan(out["pr_auc_ovr_macro"])
    assert math.isnan(out["brier_multiclass"])
    assert math.isnan(out["ece_multiclass"])
    assert any("y_prob_not_supplied" in w for w in out["warnings"])


def test_multiclass_with_probabilities() -> None:
    rng = np.random.default_rng(0)
    n, k = 60, 3
    y_true = rng.integers(0, k, size=n)
    # Make the model "good but not perfect": correct class gets a
    # higher probability mass.
    y_prob = rng.uniform(0.1, 0.2, size=(n, k))
    y_prob[np.arange(n), y_true] += 0.6
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)
    out = compute_multiclass_metrics(y_true, y_pred, y_prob=y_prob)
    assert out["roc_auc_ovr_macro"] > 0.7
    assert 0.0 <= out["pr_auc_ovr_macro"] <= 1.0
    assert 0.0 <= out["brier_multiclass"] <= 2.0
    assert 0.0 <= out["ece_multiclass"] <= 1.0


def test_multiclass_with_missing_class_in_pred() -> None:
    """A class missing from y_pred should not break the metrics."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1])  # class 2 never predicted
    out = compute_multiclass_metrics(y_true, y_pred)
    assert math.isfinite(out["f1_macro"])
    assert math.isfinite(out["balanced_accuracy"])


def test_multiclass_with_non_zero_indexed_labels() -> None:
    y_true = np.array([10, 10, 20, 30, 30, 20])
    y_pred = np.array([10, 20, 20, 30, 10, 20])
    out = compute_multiclass_metrics(y_true, y_pred, labels=[10, 20, 30])
    assert math.isfinite(out["accuracy"])
    assert math.isfinite(out["f1_macro"])


def test_multiclass_proba_class_count_mismatch_returns_nan() -> None:
    """y_prob with the wrong number of columns is reported, not raised."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    bogus = np.tile([0.5, 0.5], (5, 1))   # 2 columns, but 3 classes
    out = compute_multiclass_metrics(y_true, y_pred, y_prob=bogus, labels=[0, 1, 2])
    assert math.isnan(out["roc_auc_ovr_macro"])
    assert any("y_prob_class_count_mismatch" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_auto_picks_multiclass_when_three_classes() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    out = compute_classification_metrics(y_true, y_pred)
    assert out["task"] == "multiclass"
    assert "f1_macro" in out


def test_dispatcher_explicit_task_overrides_auto() -> None:
    """Forcing task_type="multiclass" must not regress to binary."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    out = compute_classification_metrics(y_true, y_pred, task_type="multiclass")
    assert out["task"] == "multiclass"


def test_aggregate_metric_dicts_multiclass_keys_and_nan_handling() -> None:
    folds = [
        {"task": "multiclass", "accuracy": 0.9, "f1_macro": 0.85,
         "balanced_accuracy": 0.88, "mcc": 0.8,
         "roc_auc_ovr_macro": 0.92, "pr_auc_ovr_macro": 0.81,
         "brier_multiclass": 0.20, "ece_multiclass": 0.03, "warnings": []},
        {"task": "multiclass", "accuracy": 0.8, "f1_macro": 0.78,
         "balanced_accuracy": 0.82, "mcc": 0.7,
         # one fold without proba metrics:
         "roc_auc_ovr_macro": float("nan"), "pr_auc_ovr_macro": float("nan"),
         "brier_multiclass": float("nan"), "ece_multiclass": float("nan"),
         "warnings": ["y_prob_not_supplied"]},
    ]
    agg = aggregate_metric_dicts(folds)
    assert agg["Accuracy_Mean"] == pytest.approx(0.85)
    assert agg["F1Macro_Mean"] == pytest.approx(0.815)
    assert agg["MCC_Mean"] == pytest.approx(0.75)
    # NaN-skipping: only the first fold has ROC AUC.
    assert agg["ROCAUC_OVR_Mean"] == pytest.approx(0.92)
    assert agg["BrierMC_Mean"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Guardrail: multiclass + binary FA defaults must raise
# ---------------------------------------------------------------------------


def test_guardrail_raises_on_multiclass_with_binary_defaults() -> None:
    binary_defaults = (
        "Accuracy_Mean",
        "Precision_Mean",
        "Recall_Mean",
        "Specificity_Mean",
        "Time_MeanFold",
    )
    with pytest.raises(MultiClassNotConfiguredError) as excinfo:
        assert_metric_set_compatible_with_task(
            dataset_id="dry_bean",
            task="multiclass",
            fa_metrics=binary_defaults,
        )
    assert excinfo.value.dataset_id == "dry_bean"
    assert "F1Macro_Mean" in excinfo.value.required_metric_set


def test_guardrail_passes_when_multiclass_metric_set_used() -> None:
    multiclass_set = (
        "F1Macro_Mean",
        "BalancedAccuracy_Mean",
        "MCC_Mean",
        "ROCAUC_OVR_Mean",
        "PRAUC_OVR_Mean",
        "BrierMC_Mean",
        "ECE_Mean",
        "Time_MeanFold",
    )
    # Should not raise.
    assert_metric_set_compatible_with_task(
        dataset_id="dry_bean",
        task="multiclass",
        fa_metrics=multiclass_set,
    )


def test_guardrail_passes_for_binary_dataset_regardless_of_metric_set() -> None:
    """Binary tasks never trigger the multiclass guardrail."""
    assert_metric_set_compatible_with_task(
        dataset_id="magic",
        task="binary",
        fa_metrics=("Accuracy_Mean", "Precision_Mean", "Recall_Mean",
                    "Specificity_Mean", "Time_MeanFold"),
    )
