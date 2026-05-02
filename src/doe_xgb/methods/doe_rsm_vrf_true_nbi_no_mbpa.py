"""Adapter for the proposed-method ablation with the conditional MBPA stage
forced off."""

from __future__ import annotations

from .base import AdapterBase


class DoeRsmVrfTrueNbiNoMbpaAdapter(AdapterBase):
    method_id = "doe_rsm_vrf_true_nbi_no_mbpa"
    required_packages = ("doe_xgb",)
    run_status = "dispatch_only"
    notes = (
        "Ablation: same pipeline as doe_rsm_vrf_true_nbi but with MBPA "
        "stage disabled. Quantifies the contribution of the conditional "
        "post-optimization."
    )
