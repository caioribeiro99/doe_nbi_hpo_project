"""Adapter for the ``random_search`` classical baseline."""

from __future__ import annotations

from .base import AdapterBase


class RandomSearchAdapter(AdapterBase):
    method_id = "random_search"
    required_packages = ("scipy", "joblib")
    run_status = "stub_only"
    notes = "Bergstra & Bengio 2012. Headline single-objective baseline."
