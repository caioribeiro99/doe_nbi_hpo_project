"""Adapter for ``motpe``: Multi-Objective TPE via Optuna."""

from __future__ import annotations

from .base import AdapterBase


class MotpeAdapter(AdapterBase):
    method_id = "motpe"
    required_packages = ("optuna",)
    run_status = "stub_only"
    notes = "Ozaki et al. 2020. Optuna MOTPESampler."
