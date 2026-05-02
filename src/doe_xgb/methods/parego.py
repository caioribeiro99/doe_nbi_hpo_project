"""Adapter for ``parego``: Tchebycheff-scalarized BO on the ParEGO subset."""

from __future__ import annotations

from .base import AdapterBase


class ParegoAdapter(AdapterBase):
    method_id = "parego"
    # Two implementation routes (SMAC's multi-objective facade or pymoo's
    # ParEGO); both are listed so the audit reports whichever is missing.
    required_packages = ("smac", "pymoo")
    run_status = "stub_only"
    notes = (
        "Knowles 2006. Subset-only (48 of 72 CC18 tasks); the subset is "
        "frozen by benchmarks/doctoral/openml_cc18/parego_subset.csv."
    )
