"""Adapter for ``nsga2``: NSGA-II via pymoo."""

from __future__ import annotations

from .base import AdapterBase


class Nsga2Adapter(AdapterBase):
    method_id = "nsga2"
    required_packages = ("pymoo",)
    run_status = "stub_only"
    notes = "Deb et al. 2002. Reference evolutionary multi-objective baseline."
