"""Robust runtime package-version reporting for gate / audit artifacts.

Some packages have different distribution and import names (notably
``scikit-learn`` ↔ ``sklearn``); ``importlib.import_module`` only
works on the import name, so the dissertation-era reporter that hit
``importlib.import_module("scikit-learn")`` always returned ``None``.
Reports therefore showed ``scikit-learn = MISSING`` even though the
package was installed and used.

This module fixes that by trying distribution metadata first.
"""

from __future__ import annotations

import importlib
import importlib.metadata as _md
from collections.abc import Iterable

# Distribution-name → import-name overrides for packages whose two
# names differ. Names not in this map are tried under both forms.
_DISTRIBUTION_TO_IMPORT_NAME: dict[str, str] = {
    "scikit-learn": "sklearn",
    "scikit-optimize": "skopt",
    "python-dotenv": "dotenv",
}


def package_version(name: str) -> str | None:
    """Return the installed version of ``name`` or ``None``.

    Resolution order:

    1. ``importlib.metadata.version(name)`` — handles distribution
       names like ``scikit-learn`` even when the import name differs;
    2. ``importlib.import_module(import_name).__version__`` — handles
       single-file packages without a recorded distribution and
       editable installs where ``metadata.version`` may be stale.

    Never raises: callers can always emit a complete report.
    """
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        pass
    except Exception:
        pass
    import_name = _DISTRIBUTION_TO_IMPORT_NAME.get(name, name)
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return None
    ver = getattr(mod, "__version__", None)
    return str(ver) if ver else None


def collect_package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Map each name in ``names`` to its resolved version or ``None``."""
    return {n: package_version(n) for n in names}


__all__ = ["collect_package_versions", "package_version"]
