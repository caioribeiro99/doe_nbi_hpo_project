"""Unit tests for ObjectiveSpec and canonicalization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from doe_xgb.objectives import (
    ObjectiveCanonicalizer,
    ObjectiveDirection,
    ObjectiveRole,
    ObjectiveSpec,
    ObjectiveTransform,
    canonicalize_array,
    canonicalize_value,
    hint_direction,
)


def test_target_direction_requires_target() -> None:
    with pytest.raises(ValueError):
        ObjectiveSpec(name="x", source="raw:x", direction=ObjectiveDirection.TARGET)


def test_fmse_requires_target() -> None:
    with pytest.raises(ValueError):
        ObjectiveSpec(
            name="x",
            source="raw:x",
            direction=ObjectiveDirection.MINIMIZE,
            transform=ObjectiveTransform.FMSE,
        )


def test_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        ObjectiveSpec(
            name="x",
            source="raw:x",
            direction=ObjectiveDirection.MINIMIZE,
            bounds=(1.0, 0.5),
        )


def test_canonicalize_minimize_identity() -> None:
    spec = ObjectiveSpec(name="t", source="raw:t", direction=ObjectiveDirection.MINIMIZE)
    assert canonicalize_value(spec, 3.5) == pytest.approx(3.5)


def test_canonicalize_maximize_negates() -> None:
    spec = ObjectiveSpec(name="acc", source="raw:acc", direction=ObjectiveDirection.MAXIMIZE)
    assert canonicalize_value(spec, 0.95) == pytest.approx(-0.95)


def test_canonicalize_target_squared_deviation() -> None:
    spec = ObjectiveSpec(
        name="x",
        source="raw:x",
        direction=ObjectiveDirection.TARGET,
        target=2.0,
    )
    # TARGET without FMSE => (v - target)^2, no variance.
    assert canonicalize_value(spec, 5.0) == pytest.approx(9.0)
    assert canonicalize_value(spec, 2.0) == pytest.approx(0.0)


def test_canonicalize_fmse_includes_variance() -> None:
    spec = ObjectiveSpec(
        name="vrf",
        source="factor:1",
        direction=ObjectiveDirection.TARGET,
        transform=ObjectiveTransform.FMSE,
        target=0.0,
    )
    # (3.0 - 0.0)^2 + 1.5 = 10.5
    assert canonicalize_value(spec, 3.0, sigma2=1.5) == pytest.approx(10.5)


def test_canonicalize_array_minimize() -> None:
    spec = ObjectiveSpec(name="t", source="raw:t", direction=ObjectiveDirection.MINIMIZE)
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(canonicalize_array(spec, arr), arr)


def test_canonicalize_array_maximize() -> None:
    spec = ObjectiveSpec(name="acc", source="raw:acc", direction=ObjectiveDirection.MAXIMIZE)
    arr = np.array([0.9, 0.95, 0.93])
    np.testing.assert_allclose(canonicalize_array(spec, arr), -arr)


def test_canonicalize_array_fmse() -> None:
    spec = ObjectiveSpec(
        name="vrf",
        source="factor:1",
        direction=ObjectiveDirection.TARGET,
        transform=ObjectiveTransform.FMSE,
        target=1.0,
    )
    arr = np.array([1.0, 2.0, 0.0])
    np.testing.assert_allclose(canonicalize_array(spec, arr, sigma2=0.5), [0.5, 1.5, 1.5])


def test_canonicalizer_evaluates_minimization_only() -> None:
    specs = (
        ObjectiveSpec(name="acc", source="raw:acc", direction=ObjectiveDirection.MAXIMIZE),
        ObjectiveSpec(name="time", source="raw:time", direction=ObjectiveDirection.MINIMIZE),
    )
    callables = {
        "acc": lambda x: float(x[0]) ** 2,
        "time": lambda x: float(x[1]),
    }
    canon = ObjectiveCanonicalizer(specs=specs, callables=callables)
    out = canon.evaluate(np.array([0.7, 0.3]))
    # Should canonicalize to (-acc, time) = (-(0.49), 0.3)
    np.testing.assert_allclose(out, [-0.49, 0.3])


def test_role_filtering() -> None:
    s = (
        ObjectiveSpec(name="a", source="raw:a", direction=ObjectiveDirection.MINIMIZE, role=ObjectiveRole.PRIMARY_NBI),
        ObjectiveSpec(name="b", source="raw:b", direction=ObjectiveDirection.MINIMIZE, role=ObjectiveRole.REPORTING),
    )
    from doe_xgb.objectives import primary_specs

    assert tuple(p.name for p in primary_specs(s)) == ("a",)


def test_hint_direction_is_advisory() -> None:
    assert hint_direction("Accuracy_Mean") is ObjectiveDirection.MAXIMIZE
    assert hint_direction("Time_MeanFold") is ObjectiveDirection.MINIMIZE
    assert hint_direction("totally_unknown_metric") is None


def test_canonicalize_does_not_silently_pick_default_for_unknown_direction() -> None:
    """Sanity: every spec carries an explicit direction; no hidden default."""

    # Pydantic-level enforcement is in test_config_schema.py. Here we verify
    # that constructing ObjectiveSpec without `direction` raises a TypeError.
    with pytest.raises(TypeError):
        ObjectiveSpec(name="x", source="raw:x")  # type: ignore[call-arg]


def test_fmse_with_zero_variance_equivalent_to_target() -> None:
    spec_target = ObjectiveSpec(
        name="x",
        source="raw:x",
        direction=ObjectiveDirection.TARGET,
        target=2.0,
    )
    spec_fmse = ObjectiveSpec(
        name="x",
        source="raw:x",
        direction=ObjectiveDirection.TARGET,
        transform=ObjectiveTransform.FMSE,
        target=2.0,
    )
    assert canonicalize_value(spec_target, 5.0) == canonicalize_value(spec_fmse, 5.0, sigma2=0.0)
    assert math.isfinite(canonicalize_value(spec_fmse, 5.0, sigma2=0.0))
