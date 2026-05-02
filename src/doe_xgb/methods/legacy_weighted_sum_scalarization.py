"""Adapter for the dissertation-era weighted-sum scalarization baseline."""

from __future__ import annotations

from .base import AdapterBase


class LegacyWeightedSumScalarizationAdapter(AdapterBase):
    method_id = "legacy_weighted_sum_scalarization"
    required_packages = ("doe_xgb",)
    run_status = "dispatch_only"
    notes = (
        "Ablation: dissertation-era weighted-sum solver kept verbatim "
        "(doe_xgb.scalarization.run_nbi_weighted_sum). Never referred to as "
        "NBI in the article text."
    )
