"""Adapter for ``dehb``: Differential Evolution + Hyperband."""

from __future__ import annotations

from .base import AdapterBase


class DehbAdapter(AdapterBase):
    method_id = "dehb"
    required_packages = ("dehb",)
    run_status = "stub_only"
    notes = "Awad, Mallik & Hutter 2021. DE inner loop on a Hyperband schedule."
