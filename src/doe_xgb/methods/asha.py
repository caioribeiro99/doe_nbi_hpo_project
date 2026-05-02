"""Adapter for ``asha``: Asynchronous Successive Halving via Optuna."""

from __future__ import annotations

from .base import AdapterBase


class AshaAdapter(AdapterBase):
    method_id = "asha"
    required_packages = ("optuna",)
    run_status = "stub_only"
    notes = (
        "Li et al. 2017/2020. Fidelity dimension = boosting iterations "
        "(n_estimators). Frozen choice over Hyperband in Commit 27."
    )
