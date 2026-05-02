"""Adapter for the proposed method:
DOE + design-aware RSM + VRF/FMSE + true N-objective NBI + conditional MBPA."""

from __future__ import annotations

from .base import AdapterBase


class DoeRsmVrfTrueNbiAdapter(AdapterBase):
    method_id = "doe_rsm_vrf_true_nbi"
    # All in-tree; the doe_xgb package itself is the implementation.
    required_packages = ("doe_xgb",)
    # Dispatch is wired via the doe_xgb pipeline modules, which already
    # exist (factor_model, nbi_core, post_optimization, ...). The runner
    # can route a job here and call into them in the next commit.
    run_status = "dispatch_only"
    notes = (
        "Headline proposed method. All implementation lives in-tree under "
        "src/doe_xgb (factor_model, nbi_core, post_optimization). "
        "run() still raises NotImplementedError until Commit 30 wires the "
        "per-task config and CV harness."
    )
