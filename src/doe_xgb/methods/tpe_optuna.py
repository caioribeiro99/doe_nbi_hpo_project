"""Adapter for ``tpe_optuna``: Optuna's TPESampler with default prior."""

from __future__ import annotations

from .base import AdapterBase


class TpeOptunaAdapter(AdapterBase):
    method_id = "tpe_optuna"
    required_packages = ("optuna",)
    run_status = "stub_only"
    notes = "Akiba et al. 2019. Default-prior TPESampler."
