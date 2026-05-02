"""DesignProvider: build / load / validate experimental designs.

Built-in generators (pure NumPy):
- ``ccd_face_centered``: CCDFC / CCFCD with axial points at the box face.
- ``ccd_circumscribed``: classical CCD with axial alpha = (2^k)^{1/4}.
- ``ccd_inscribed``: scaled-down CCD that lives inside the box.
- ``box_behnken``: standard 3-level Box-Behnken for k in {3, ..., 7}.
- ``full_factorial``: 2-level (or arbitrary level) full factorial.
- ``fractional_factorial``: 2-level fractional factorial of given resolution.
- ``lhs``: Latin Hypercube via :class:`scipy.stats.qmc.LatinHypercube`.
- ``sobol``: Sobol sequence via :class:`scipy.stats.qmc.Sobol`.
- ``simplex_lattice``: Scheffé simplex-lattice (delegated to
  :mod:`doe_xgb.simplex`).
- ``external_csv``: load + validate a Minitab-generated CSV with an
  optional sidecar ``*.metadata.json``.

D-optimal generation is not implemented in this commit; we expose a
clear error so callers know to supply their own candidate matrix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .coding import decode, encode
from .diagnostics import design_coverage, matrix_diagnostics


class DesignKind(str, Enum):
    EXTERNAL_CSV = "external_csv"
    FULL_FACTORIAL = "full_factorial"
    FRACTIONAL_FACTORIAL = "fractional_factorial"
    CCD_CIRCUMSCRIBED = "ccd_circumscribed"
    CCD_INSCRIBED = "ccd_inscribed"
    CCD_FACE_CENTERED = "ccd_face_centered"
    BOX_BEHNKEN = "box_behnken"
    LATIN_HYPERCUBE = "lhs"
    SOBOL = "sobol"
    D_OPTIMAL = "d_optimal"
    SIMPLEX_LATTICE = "simplex_lattice"


@dataclass(frozen=True)
class FactorMeta:
    name: str
    lo: float
    hi: float
    type: Literal["float", "int"] = "float"


@dataclass(frozen=True)
class DesignSpec:
    kind: DesignKind
    factors: tuple[FactorMeta, ...] = ()
    n_center: int = 0
    fractional_resolution: int | None = None
    lhs_criterion: Literal["maximin", "correlation", "centermaximin"] = "maximin"
    d_optimal_model: str | None = None
    simplex_q: int | None = None
    simplex_m: int | None = None
    seed: int | None = None
    external_path: Path | None = None
    metadata_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignArtifact:
    matrix_coded: pd.DataFrame
    matrix_uncoded: pd.DataFrame
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    detail: dict[str, Any]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _lo_hi(factors: Sequence[FactorMeta]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([f.lo for f in factors], dtype=float)
    hi = np.array([f.hi for f in factors], dtype=float)
    return lo, hi


def _two_level_corners(k: int) -> np.ndarray:
    """All 2^k corners with values {-1, +1}."""
    return np.array(list(product([-1.0, 1.0], repeat=k)), dtype=float)


def _ccd_axial(k: int, alpha: float) -> np.ndarray:
    arr = np.zeros((2 * k, k), dtype=float)
    for i in range(k):
        arr[2 * i, i] = -alpha
        arr[2 * i + 1, i] = +alpha
    return arr


def _ccd_centers(k: int, n_center: int) -> np.ndarray:
    return np.zeros((n_center, k), dtype=float)


def _build_ccd(k: int, alpha: float, n_center: int) -> np.ndarray:
    return np.vstack(
        [
            _two_level_corners(k),
            _ccd_axial(k, alpha),
            _ccd_centers(k, n_center),
        ]
    )


def _box_behnken_design(k: int) -> np.ndarray:
    """Standard 3-level Box-Behnken design for k in {3, 4, 5, 6, 7}.

    Generates pairs of factors at +/-1 with the rest at 0, plus center
    points (n_center is set externally).
    """
    if k < 3 or k > 7:
        raise ValueError("Box-Behnken supports k in {3, 4, 5, 6, 7}")
    rows: list[np.ndarray] = []
    for i in range(k):
        for j in range(i + 1, k):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(k, dtype=float)
                    v[i] = s1
                    v[j] = s2
                    rows.append(v)
    return np.vstack(rows)


def _fractional_factorial(k: int, resolution: int, seed: int | None) -> np.ndarray:
    """Two-level fractional factorial with given resolution.

    Uses a defining-relation generator approach: take the full factorial
    of the first ``p = ceil(log2(2^k / 2^(k - resolution + 1)))`` columns
    and assign extra columns as products of subsets to achieve the target
    resolution. This is a lightweight implementation; use pyDOE3 for
    advanced cases.
    """
    if resolution < 3:
        raise ValueError("resolution must be >= 3")
    n_runs = 2 ** max(resolution - 1, 1)
    while n_runs < 2 ** k:
        # If we cannot keep the requested resolution, fall back to full factorial.
        return _two_level_corners(k)
    base_k = int(np.log2(n_runs))
    base = _two_level_corners(base_k)
    if base_k == k:
        return base
    extra_cols = []
    rng = np.random.default_rng(seed if seed is not None else 0)
    used: list[tuple[int, ...]] = []
    for _j in range(k - base_k):
        size = max(2, resolution - 1)
        candidate = tuple(sorted(rng.choice(base_k, size=size, replace=False).tolist()))
        while candidate in used:
            candidate = tuple(sorted(rng.choice(base_k, size=size, replace=False).tolist()))
        used.append(candidate)
        col = np.prod(base[:, list(candidate)], axis=1)
        extra_cols.append(col.reshape(-1, 1))
    return np.hstack([base] + extra_cols)


def _lhs(n: int, k: int, criterion: str, seed: int | None) -> np.ndarray:
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=k, seed=seed)
    sample = sampler.random(n=n)
    if criterion in ("maximin", "centermaximin"):
        try:
            sample = qmc.LatinHypercube(d=k, seed=seed, optimization="random-cd").random(n=n)
        except TypeError:
            pass
    # Map [0, 1] to coded [-1, 1].
    return 2.0 * sample - 1.0


def _sobol(n: int, k: int, seed: int | None) -> np.ndarray:
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=k, seed=seed)
    sample = sampler.random(n=n)
    return 2.0 * sample - 1.0


# ---------------------------------------------------------------------------
# External design loader
# ---------------------------------------------------------------------------


def _read_csv_flex(path: Path) -> pd.DataFrame:
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if head.count(";") > head.count(","):
        return pd.read_csv(path, sep=";", decimal=",")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";", decimal=",")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_external_csv(spec: DesignSpec) -> DesignArtifact:
    if spec.external_path is None:
        raise ValueError("external_csv design requires external_path")
    csv_path = Path(spec.external_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"design CSV not found: {csv_path}")

    df = _read_csv_flex(csv_path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    factor_names = [f.name for f in spec.factors]
    missing = [n for n in factor_names if n not in df.columns]
    if missing and not spec.factors:
        # No factor metadata: use all numeric columns as factors.
        factor_names = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    uncoded = df.loc[:, factor_names].astype(float).reset_index(drop=True)

    if spec.factors:
        lo, hi = _lo_hi(spec.factors)
        coded_arr = encode(uncoded.to_numpy(), lo, hi)
        coded = pd.DataFrame(coded_arr, columns=factor_names)
    else:
        coded = uncoded.copy()

    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")
    sidecar: dict[str, Any] = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    metadata: dict[str, Any] = {
        "kind": "external_csv",
        "external_path": str(csv_path),
        "sha256": _sha256(csv_path),
        "n_runs": int(len(df)),
        "factor_names": factor_names,
    }
    metadata.update(sidecar)
    metadata.update(spec.metadata_overrides or {})

    diagnostics: dict[str, Any] = {
        "matrix": matrix_diagnostics(uncoded.to_numpy()),
        "coverage": design_coverage(coded.to_numpy()) if spec.factors else {},
    }

    return DesignArtifact(
        matrix_coded=coded,
        matrix_uncoded=uncoded,
        metadata=metadata,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Top-level factories
# ---------------------------------------------------------------------------


def _wrap_coded(coded: np.ndarray, spec: DesignSpec, kind: DesignKind, extra_meta: dict[str, Any]) -> DesignArtifact:
    factor_names = [f.name for f in spec.factors]
    lo, hi = _lo_hi(spec.factors) if spec.factors else (None, None)
    if lo is not None and hi is not None:
        uncoded_arr = decode(coded, lo, hi)
        # Cast integer factors to int when applicable.
        for i, f in enumerate(spec.factors):
            if f.type == "int":
                uncoded_arr[:, i] = np.rint(uncoded_arr[:, i])
    else:
        uncoded_arr = coded
    coded_df = pd.DataFrame(coded, columns=factor_names)
    uncoded_df = pd.DataFrame(uncoded_arr, columns=factor_names)
    metadata = {
        "kind": kind.value,
        "n_runs": int(coded.shape[0]),
        "factor_names": factor_names,
        "factors": [
            {"name": f.name, "lo": f.lo, "hi": f.hi, "type": f.type} for f in spec.factors
        ],
    }
    metadata.update(extra_meta)
    diagnostics = {
        "matrix": matrix_diagnostics(coded),
        "coverage": design_coverage(coded),
    }
    return DesignArtifact(
        matrix_coded=coded_df,
        matrix_uncoded=uncoded_df,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def build(spec: DesignSpec) -> DesignArtifact:
    """Build a :class:`DesignArtifact` from a :class:`DesignSpec`."""

    kind = spec.kind
    if kind == DesignKind.EXTERNAL_CSV:
        return _load_external_csv(spec)

    if kind == DesignKind.SIMPLEX_LATTICE:
        from ..simplex import generate_simplex_lattice

        if spec.simplex_q is None or spec.simplex_m is None:
            raise ValueError("simplex_lattice design requires simplex_q and simplex_m")
        coded = generate_simplex_lattice(spec.simplex_q, spec.simplex_m)
        names = [f"w{i + 1}" for i in range(spec.simplex_q)]
        df = pd.DataFrame(coded, columns=names)
        return DesignArtifact(
            matrix_coded=df,
            matrix_uncoded=df.copy(),
            metadata={
                "kind": kind.value,
                "simplex_q": spec.simplex_q,
                "simplex_m": spec.simplex_m,
                "n_runs": int(coded.shape[0]),
                "factor_names": names,
            },
            diagnostics={"matrix": matrix_diagnostics(coded)},
        )

    factors = list(spec.factors)
    if not factors:
        raise ValueError(f"design kind {kind.value} requires factors")
    k = len(factors)

    if kind == DesignKind.FULL_FACTORIAL:
        coded = _two_level_corners(k)
        if spec.n_center > 0:
            coded = np.vstack([coded, np.zeros((spec.n_center, k))])
        return _wrap_coded(coded, spec, kind, {"n_center": spec.n_center})

    if kind == DesignKind.FRACTIONAL_FACTORIAL:
        if spec.fractional_resolution is None:
            raise ValueError("fractional_factorial requires fractional_resolution")
        coded = _fractional_factorial(k, spec.fractional_resolution, spec.seed)
        if spec.n_center > 0:
            coded = np.vstack([coded, np.zeros((spec.n_center, k))])
        return _wrap_coded(coded, spec, kind, {"n_center": spec.n_center, "resolution": spec.fractional_resolution})

    if kind == DesignKind.CCD_FACE_CENTERED:
        coded = _build_ccd(k, alpha=1.0, n_center=spec.n_center)
        return _wrap_coded(coded, spec, kind, {"n_center": spec.n_center, "alpha": 1.0})

    if kind == DesignKind.CCD_CIRCUMSCRIBED:
        alpha = float((2 ** k) ** 0.25)
        coded = _build_ccd(k, alpha=alpha, n_center=spec.n_center)
        return _wrap_coded(coded, spec, kind, {"n_center": spec.n_center, "alpha": alpha})

    if kind == DesignKind.CCD_INSCRIBED:
        alpha_outer = float((2 ** k) ** 0.25)
        coded = _build_ccd(k, alpha=alpha_outer, n_center=spec.n_center) / alpha_outer
        return _wrap_coded(coded, spec, kind, {"n_center": spec.n_center, "alpha": 1.0 / alpha_outer})

    if kind == DesignKind.BOX_BEHNKEN:
        coded = _box_behnken_design(k)
        if spec.n_center > 0:
            coded = np.vstack([coded, np.zeros((spec.n_center, k))])
        return _wrap_coded(coded, spec, kind, {"n_center": spec.n_center})

    if kind == DesignKind.LATIN_HYPERCUBE:
        n = max(2 * k + 1, 2 ** k) if spec.n_center == 0 else max(2 * k + 1, 16)
        coded = _lhs(n, k, spec.lhs_criterion, spec.seed)
        return _wrap_coded(coded, spec, kind, {"criterion": spec.lhs_criterion, "seed": spec.seed})

    if kind == DesignKind.SOBOL:
        n = max(8, 2 ** int(np.ceil(np.log2(max(2 * k + 1, 16)))))
        coded = _sobol(n, k, spec.seed)
        return _wrap_coded(coded, spec, kind, {"seed": spec.seed})

    if kind == DesignKind.D_OPTIMAL:
        raise NotImplementedError(
            "D-optimal generation is intentionally not implemented in this commit. "
            "Provide an external candidate set via kind=external_csv, or use pyDOE3 "
            "to pre-generate the candidate matrix."
        )

    raise ValueError(f"unsupported design kind: {kind}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_for_model(artifact: DesignArtifact, *, family: str, order: str) -> ValidationReport:
    """Sanity-check the design vs the requested surrogate model family.

    Compatibility table (warning, not blocking):
    - ``process_quadratic`` + CCD/BB/factorial: OK.
    - ``process_quadratic`` + LHS/Sobol: WARN (no inferential support).
    - ``mixture_scheffe`` + simplex_lattice/centroid: OK.
    - ``mixture_scheffe`` + non-mixture: ERROR.
    """
    warnings: list[str] = []
    errors: list[str] = []

    kind = str(artifact.metadata.get("kind", "")).lower()
    fam = family.lower()
    diag = artifact.diagnostics.get("matrix", {})

    if fam == "mixture_scheffe":
        if kind not in ("simplex_lattice", "simplex_centroid"):
            errors.append(
                f"mixture_scheffe model requires a simplex design; got kind={kind!r}."
            )
    elif fam in ("process_quadratic", "linear", "two_factor_interaction"):
        if kind in ("simplex_lattice", "simplex_centroid"):
            errors.append(
                f"process-variable model is not valid on a mixture design; got kind={kind!r}."
            )
        elif kind in ("lhs", "sobol"):
            warnings.append(
                "Space-filling design (LHS/Sobol) does not support classical RSM "
                "inference (no pure error, weak orthogonality). Use only for "
                "exploratory surrogate fitting."
            )

    if diag.get("rank_deficient", False):
        warnings.append(
            f"Design matrix appears rank-deficient (rank={diag.get('rank')} of "
            f"{diag.get('n_columns')} columns). Consider increasing runs or "
            "downgrading the model order."
        )

    cond = diag.get("condition_number", float("inf"))
    if cond is not None and cond > 1e6:
        warnings.append(f"Condition number is large ({cond:.2e}); results may be numerically unstable.")

    return ValidationReport(
        ok=not errors,
        warnings=tuple(warnings),
        errors=tuple(errors),
        detail={"kind": kind, "family": family, "order": order, "diag": diag},
    )


class DesignProvider:
    @staticmethod
    def build(spec: DesignSpec) -> DesignArtifact:
        return build(spec)

    @staticmethod
    def validate_for_model(artifact: DesignArtifact, *, family: str, order: str) -> ValidationReport:
        return validate_for_model(artifact, family=family, order=order)


__all__ = [
    "DesignKind",
    "FactorMeta",
    "DesignSpec",
    "DesignArtifact",
    "ValidationReport",
    "DesignProvider",
    "build",
    "validate_for_model",
]
