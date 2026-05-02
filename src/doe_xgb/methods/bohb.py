"""Adapter for ``bohb``: Bayesian Optimization + Hyperband.

The original reference implementation lives in the ``hpbandster`` package,
which is not actively maintained for current Python versions. SMAC3 ships
a multi-fidelity facade that approximates BOHB without the legacy
dependency, so we list ``smac`` as the primary required package and flag
manual review in the audit notes.
"""

from __future__ import annotations

from .base import AdapterBase


class BohbAdapter(AdapterBase):
    method_id = "bohb"
    # Prefer SMAC3's multi-fidelity facade; do not list hpbandster as a
    # required package because installing it on Python 3.12 is fragile.
    required_packages = ("smac",)
    run_status = "stub_only"
    notes = (
        "Falkner et al. 2018. Implementation route: SMAC3 multi-fidelity "
        "facade. The hpbandster reference implementation is not actively "
        "maintained for Python 3.12 and is intentionally NOT a required "
        "package. Manual review may be needed before stage 0."
    )
