"""Adapter for ``smac3``: SMAC3 BO Facade with RF surrogate."""

from __future__ import annotations

from .base import AdapterBase


class Smac3Adapter(AdapterBase):
    method_id = "smac3"
    required_packages = ("smac",)
    run_status = "stub_only"
    notes = (
        "Lindauer et al. 2022. RF surrogate with intensification. "
        "Native categorical handling via SMAC categorical hyperparameter type."
    )
