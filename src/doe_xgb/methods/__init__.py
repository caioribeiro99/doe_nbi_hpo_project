"""Method adapters for the OpenML-CC18 doctoral benchmark.

Each non-literature method_id from
``benchmarks/doctoral/openml_cc18/method_matrix.csv`` has a single
adapter module here that declares its required packages, exposes an
``import_check()`` for the capability audit, and a ``run()`` stub
(raising NotImplementedError) that later commits will fill in.

Importing this package never executes any HPO. It is safe to import
on a machine that has none of the optional baseline packages
installed; missing packages surface only at ``import_check()`` time.
"""

from __future__ import annotations

from .base import (
    AdapterBase,
    CapabilityStatus,
    PackageCheck,
    RunStatus,
    check_packages,
)
from .registry import ADAPTERS, ALL_METHOD_IDS, get_adapter

__all__ = [
    "ADAPTERS",
    "ALL_METHOD_IDS",
    "AdapterBase",
    "CapabilityStatus",
    "PackageCheck",
    "RunStatus",
    "check_packages",
    "get_adapter",
]
