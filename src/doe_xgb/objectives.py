"""Objective specification layer.

Every objective entering the NBI core must declare its direction explicitly.
This is the cure for the dissertation-era practice of inferring direction
from metric names or factor-loading signs (see
``docs/METHODOLOGY_DECISIONS.md`` D3).

Two transforms are supported for the article track:

- ``raw``: identity. Combined with ``direction`` to canonicalize to
  minimization (see :class:`ObjectiveCanonicalizer`).
- ``fmse`` (target / FMSE-style; Pereira et al., 2025, Eqs. 17-22):
  ``f_i(x) -> (f_i(x) - target_i)^2 + sigma2_i``. Always a minimization.

The core convention is that NBI minimizes every objective. Maximization
objectives are negated; FMSE objectives are squared-deviation-plus-variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    TARGET = "target"


class ObjectiveTransform(str, Enum):
    RAW = "raw"
    ZSCORE = "zscore"
    LOG1P = "log1p"
    FACTOR_SCORE = "factor_score"
    FMSE = "fmse"
    CUSTOM = "custom"


class ObjectiveRole(str, Enum):
    PRIMARY_NBI = "primary_nbi"
    POST_OPTIMIZATION = "post_optimization"
    REPORTING = "reporting"
    CONSTRAINT = "constraint"


class ObjectiveNormalization(str, Enum):
    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    PAYOFF = "payoff"


@dataclass(frozen=True)
class ObjectiveSpec:
    """Declarative description of one optimization objective.

    Attributes
    ----------
    name:
        Human-readable identifier; also the column name written to outputs.
    source:
        Origin of the value. Conventions:

        - ``"raw:<col>"``: a raw response column from CV results.
        - ``"factor:<k>"``: the k-th rotated factor score (1-indexed).
        - ``"callable:<id>"``: registered callable.
    direction:
        Required. ``minimize``, ``maximize``, or ``target``.
    transform:
        Optional transform applied before NBI. ``fmse`` requires a target.
    target:
        Required when ``direction == TARGET`` or ``transform == FMSE``.
        For factor-score objectives, the convention "individual_optimum"
        means the per-objective surrogate optimum (computed at runtime).
    weight:
        Used by post-optimization utility (MBPA / MCDM); not by core NBI.
    role:
        ``primary_nbi`` enters the payoff matrix Phi. ``reporting`` is
        tracked but not optimized. ``post_optimization`` participates in
        MBPA (e.g., GD or Shannon entropy).
    group:
        Construct label used by the manual factor-grouping mode.
    bounds:
        Optional (lo, hi). Used for normalization and feasibility checks.
    normalization:
        How the objective is rescaled before NBI. ``payoff`` uses the
        utopia/nadir from the payoff matrix at runtime.
    """

    name: str
    source: str
    direction: ObjectiveDirection
    transform: ObjectiveTransform = ObjectiveTransform.RAW
    target: Optional[float] = None
    weight: Optional[float] = None
    role: ObjectiveRole = ObjectiveRole.PRIMARY_NBI
    group: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None
    normalization: ObjectiveNormalization = ObjectiveNormalization.PAYOFF

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.direction is ObjectiveDirection.TARGET and self.target is None:
            raise ValueError(
                f"Objective {self.name!r}: direction=TARGET requires a target value."
            )
        if self.transform is ObjectiveTransform.FMSE and self.target is None:
            raise ValueError(
                f"Objective {self.name!r}: transform=FMSE requires a target value."
            )
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"Objective {self.name!r}: bounds must satisfy lo < hi, got {self.bounds}."
            )


@dataclass(frozen=True)
class ObjectivesConfig:
    specs: Tuple[ObjectiveSpec, ...]
    convention: str = "minimize"

    def primary(self) -> Tuple[ObjectiveSpec, ...]:
        return tuple(s for s in self.specs if s.role is ObjectiveRole.PRIMARY_NBI)

    def post_opt(self) -> Tuple[ObjectiveSpec, ...]:
        return tuple(s for s in self.specs if s.role is ObjectiveRole.POST_OPTIMIZATION)

    def reporting(self) -> Tuple[ObjectiveSpec, ...]:
        return tuple(s for s in self.specs if s.role is ObjectiveRole.REPORTING)


# ---------------------------------------------------------------------------
# Canonicalization to "lower is better" minimization.
# ---------------------------------------------------------------------------


def canonicalize_value(spec: ObjectiveSpec, value: float, sigma2: float = 0.0) -> float:
    """Return the value of `spec` mapped to a minimization convention.

    For raw / zscore / log1p / factor_score / custom transforms:

    - ``MINIMIZE``: identity.
    - ``MAXIMIZE``: ``-value``.
    - ``TARGET``: ``(value - target)**2`` (no variance term unless FMSE).

    For ``FMSE``: ``(value - target)**2 + sigma2`` always (minimization).

    Parameters
    ----------
    sigma2:
        Variance term used by FMSE (Pereira et al., 2025, Eq. 21).
        Ignored otherwise.
    """

    if spec.transform is ObjectiveTransform.FMSE:
        if spec.target is None:
            raise ValueError(f"Objective {spec.name}: FMSE without target.")
        return float((value - float(spec.target)) ** 2 + sigma2)

    if spec.direction is ObjectiveDirection.MINIMIZE:
        return float(value)
    if spec.direction is ObjectiveDirection.MAXIMIZE:
        return -float(value)
    if spec.direction is ObjectiveDirection.TARGET:
        if spec.target is None:
            raise ValueError(f"Objective {spec.name}: TARGET direction without target.")
        return float((value - float(spec.target)) ** 2)
    raise ValueError(f"Unknown direction for {spec.name}: {spec.direction!r}.")


def canonicalize_array(spec: ObjectiveSpec, values: np.ndarray, sigma2: float = 0.0) -> np.ndarray:
    """Vectorized version of :func:`canonicalize_value`."""
    if spec.transform is ObjectiveTransform.FMSE:
        if spec.target is None:
            raise ValueError(f"Objective {spec.name}: FMSE without target.")
        return (values - float(spec.target)) ** 2 + float(sigma2)
    if spec.direction is ObjectiveDirection.MINIMIZE:
        return np.asarray(values, dtype=float)
    if spec.direction is ObjectiveDirection.MAXIMIZE:
        return -np.asarray(values, dtype=float)
    if spec.direction is ObjectiveDirection.TARGET:
        if spec.target is None:
            raise ValueError(f"Objective {spec.name}: TARGET direction without target.")
        return (values - float(spec.target)) ** 2
    raise ValueError(f"Unknown direction for {spec.name}: {spec.direction!r}.")


@dataclass(frozen=True)
class ObjectiveCanonicalizer:
    """Wraps a list of :class:`ObjectiveSpec` and a runtime callable map.

    `callables` maps each spec name to an actual callable ``f_i(x) -> float``.
    `sigma2_provider` returns a variance term per spec name when FMSE is used.

    The canonicalizer turns a raw multi-objective callable
    ``F(x) -> (q,) ndarray`` into a minimization-only one.
    """

    specs: Tuple[ObjectiveSpec, ...]
    callables: Dict[str, Callable[[np.ndarray], float]]
    sigma2_provider: Optional[Callable[[str], float]] = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.specs), dtype=float)
        for i, spec in enumerate(self.specs):
            f = self.callables[spec.name]
            v = float(f(np.asarray(x, dtype=float)))
            sigma2 = (
                float(self.sigma2_provider(spec.name))
                if (
                    spec.transform is ObjectiveTransform.FMSE
                    and self.sigma2_provider is not None
                )
                else 0.0
            )
            out[i] = canonicalize_value(spec, v, sigma2=sigma2)
        return out

    def evaluate_one(self, name: str, x: np.ndarray) -> float:
        for spec in self.specs:
            if spec.name == name:
                f = self.callables[name]
                v = float(f(np.asarray(x, dtype=float)))
                sigma2 = (
                    float(self.sigma2_provider(name))
                    if (
                        spec.transform is ObjectiveTransform.FMSE
                        and self.sigma2_provider is not None
                    )
                    else 0.0
                )
                return canonicalize_value(spec, v, sigma2=sigma2)
        raise KeyError(f"Objective {name!r} not found.")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


_DEFAULT_DIRECTION_HINTS: Dict[str, ObjectiveDirection] = {
    "accuracy": ObjectiveDirection.MAXIMIZE,
    "precision": ObjectiveDirection.MAXIMIZE,
    "recall": ObjectiveDirection.MAXIMIZE,
    "specificity": ObjectiveDirection.MAXIMIZE,
    "f1": ObjectiveDirection.MAXIMIZE,
    "auc": ObjectiveDirection.MAXIMIZE,
    "balancedaccuracy": ObjectiveDirection.MAXIMIZE,
    "time": ObjectiveDirection.MINIMIZE,
    "time_meanfold": ObjectiveDirection.MINIMIZE,
    "time_mean_fold": ObjectiveDirection.MINIMIZE,
    "brier": ObjectiveDirection.MINIMIZE,
    "ece": ObjectiveDirection.MINIMIZE,
    "logloss": ObjectiveDirection.MINIMIZE,
    "bias": ObjectiveDirection.MINIMIZE,
    "instability": ObjectiveDirection.MINIMIZE,
    "fairnessscore": ObjectiveDirection.MAXIMIZE,
}


def hint_direction(name: str) -> Optional[ObjectiveDirection]:
    """Best-effort hint for human use only; never used by the core."""
    key = name.strip().lower().replace("-", "").replace("_", "")
    for token, direction in _DEFAULT_DIRECTION_HINTS.items():
        if token.replace("_", "") in key:
            return direction
    return None


def primary_specs(specs: Sequence[ObjectiveSpec]) -> Tuple[ObjectiveSpec, ...]:
    return tuple(s for s in specs if s.role is ObjectiveRole.PRIMARY_NBI)


__all__ = [
    "ObjectiveDirection",
    "ObjectiveTransform",
    "ObjectiveRole",
    "ObjectiveNormalization",
    "ObjectiveSpec",
    "ObjectivesConfig",
    "ObjectiveCanonicalizer",
    "canonicalize_value",
    "canonicalize_array",
    "hint_direction",
    "primary_specs",
]
