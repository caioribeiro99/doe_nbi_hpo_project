"""Backwards-compatibility shim.

Historically, ``doe_xgb.nbi`` exported a ``run_nbi_weighted_sum`` function
that the dissertation code (and its outputs) called "NBI". As discussed
in :doc:`METHODOLOGY_DECISIONS` D1, that implementation is in fact a
normalized weighted-sum scalarization, not Normal Boundary Intersection.

This shim now:

- re-exports the legacy implementation from :mod:`doe_xgb.scalarization`
  so existing scripts keep working;
- re-exports the true N-objective NBI from :mod:`doe_xgb.nbi_core` for
  callers that want to opt in;
- emits a single :class:`DeprecationWarning` when the legacy entry point
  is used to remind callers that "NBI" in the legacy code means
  "weighted-sum scalarization".

The dissertation baseline on ``main`` (tag ``v0.1.0-dissertation``) used
``run_nbi_weighted_sum``. The article-track pipeline calls
:func:`doe_xgb.nbi_core.run_nbi` directly.
"""

from __future__ import annotations

import warnings

from .scalarization import (  # noqa: F401
    NBICandidate,
    evaluate_term,
    load_coefficients_csv,
    nbi_candidates_to_df,
    predict_from_coeffs,
    save_nbi_candidates,
)
from .scalarization import run_nbi_weighted_sum as _legacy_run_nbi_weighted_sum
from .nbi_core import (  # noqa: F401
    AnchorSet,
    CHIM,
    NBIConfig,
    NBIRun,
    NBISubproblemResult,
    build_chim,
    compute_anchors,
    run_nbi,
    solve_nbi_subproblem,
)


def run_nbi_weighted_sum(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Legacy weighted-sum scalarization (NOT NBI). See docs/METHODOLOGY_DECISIONS.md D1."""
    warnings.warn(
        "doe_xgb.nbi.run_nbi_weighted_sum is the legacy weighted-sum "
        "scalarization, not Normal Boundary Intersection. Use "
        "doe_xgb.nbi_core.run_nbi for the true N-objective NBI. See "
        "docs/METHODOLOGY_DECISIONS.md D1.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _legacy_run_nbi_weighted_sum(*args, **kwargs)


__all__ = [
    # legacy re-exports
    "NBICandidate",
    "evaluate_term",
    "load_coefficients_csv",
    "nbi_candidates_to_df",
    "predict_from_coeffs",
    "run_nbi_weighted_sum",
    "save_nbi_candidates",
    # true NBI
    "AnchorSet",
    "CHIM",
    "NBIConfig",
    "NBIRun",
    "NBISubproblemResult",
    "build_chim",
    "compute_anchors",
    "run_nbi",
    "solve_nbi_subproblem",
]
