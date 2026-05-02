"""DOE + RSM + Varimax + NBI pipeline package.

The package is import-light: only pure-Python / pure-NumPy modules are
exposed eagerly. Heavy or platform-sensitive submodules
(:mod:`doe_xgb.evaluation`, :mod:`doe_xgb.benchmarks`,
:mod:`doe_xgb.doe_runner`) import :mod:`xgboost` at module load time and
are therefore loaded **lazily** through ``__getattr__``.

This keeps ``import doe_xgb`` (and importing any of the article-track
modules such as :mod:`doe_xgb.objectives`, :mod:`doe_xgb.config_schema`,
or :mod:`doe_xgb.design`) usable in environments that do not have a
working XGBoost runtime — for example, the macOS CI matrix while
``libomp.dylib`` is being installed, or a packaging-only environment.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0.dev0"

# Submodules that are safe to import eagerly (no xgboost / no native libs).
from . import config, io_utils, seeds, tracking  # noqa: F401

# Submodules that pull xgboost (or heavy optional deps) and must stay lazy.
_LAZY_MODULES: tuple[str, ...] = (
    "evaluation",
    "doe_runner",
    "benchmarks",
)

# Public attributes (eager + lazy). Names in ``_LAZY_MODULES`` only resolve
# on first attribute access (see ``__getattr__``).
__all__ = [
    "config",
    "io_utils",
    "seeds",
    "tracking",
    *list(_LAZY_MODULES),
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from . import benchmarks, doe_runner, evaluation  # noqa: F401
