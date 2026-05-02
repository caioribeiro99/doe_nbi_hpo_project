"""Method-id → adapter registry.

The registry is the single point a runner or capability-audit script
queries to translate a ``method_id`` from the SQLite job matrix into an
adapter instance. Every non-literature method_id from
``benchmarks/doctoral/openml_cc18/method_matrix.csv`` must appear here.
Literature-only methods are intentionally absent — they have no adapter
because they are not benchmarked.
"""

from __future__ import annotations

from .asha import AshaAdapter
from .base import AdapterBase
from .bohb import BohbAdapter
from .default_gbdt import DefaultGbdtAdapter
from .dehb import DehbAdapter
from .doe_rsm_vrf_true_nbi import DoeRsmVrfTrueNbiAdapter
from .doe_rsm_vrf_true_nbi_no_mbpa import DoeRsmVrfTrueNbiNoMbpaAdapter
from .legacy_weighted_sum_scalarization import LegacyWeightedSumScalarizationAdapter
from .motpe import MotpeAdapter
from .nsga2 import Nsga2Adapter
from .parego import ParegoAdapter
from .random_search import RandomSearchAdapter
from .smac3 import Smac3Adapter
from .tpe_optuna import TpeOptunaAdapter

ADAPTERS: dict[str, type[AdapterBase]] = {
    "default_gbdt": DefaultGbdtAdapter,
    "random_search": RandomSearchAdapter,
    "tpe_optuna": TpeOptunaAdapter,
    "smac3": Smac3Adapter,
    "asha": AshaAdapter,
    "bohb": BohbAdapter,
    "dehb": DehbAdapter,
    "nsga2": Nsga2Adapter,
    "motpe": MotpeAdapter,
    "parego": ParegoAdapter,
    "doe_rsm_vrf_true_nbi": DoeRsmVrfTrueNbiAdapter,
    "doe_rsm_vrf_true_nbi_no_mbpa": DoeRsmVrfTrueNbiNoMbpaAdapter,
    "legacy_weighted_sum_scalarization": LegacyWeightedSumScalarizationAdapter,
}

ALL_METHOD_IDS: tuple[str, ...] = tuple(ADAPTERS.keys())


class UnknownMethodError(KeyError):
    """Raised when a job_id row references a method_id with no adapter."""


def get_adapter(method_id: str) -> AdapterBase:
    cls = ADAPTERS.get(method_id)
    if cls is None:
        raise UnknownMethodError(
            f"no adapter for method_id={method_id!r}; "
            "literature-only methods have no adapter by design."
        )
    return cls()
