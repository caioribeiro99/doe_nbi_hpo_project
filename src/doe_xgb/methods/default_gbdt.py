"""Adapter for the ``default_gbdt`` control: per-algorithm library defaults
with no HPO. One fit per CV split per replica."""

from __future__ import annotations

from .base import AdapterBase


class DefaultGbdtAdapter(AdapterBase):
    method_id = "default_gbdt"
    # The required packages are the GBDT libraries themselves; missing
    # any of them means the corresponding (algorithm) cell cannot run,
    # not that the method is broken in general.
    required_packages = ("xgboost", "lightgbm", "catboost")
    run_status = "stub_only"
    notes = (
        "No search; one fit per CV split with library defaults. "
        "supports_categorical_native=catboost_only in the method matrix."
    )
