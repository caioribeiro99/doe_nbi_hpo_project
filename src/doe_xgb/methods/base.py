"""Common base class and capability-status types for CC18 method adapters."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
from dataclasses import dataclass
from typing import ClassVar, Literal

RunStatus = Literal["stub_only", "dispatch_only", "smoke_ready", "full_ready"]


@dataclass
class PackageCheck:
    """Result of inspecting one required package."""

    name: str
    importable: bool
    version: str | None = None
    error: str | None = None


@dataclass
class CapabilityStatus:
    """What an adapter advertises after a capability audit.

    ``adapter_import_ok`` is True if the adapter module imported cleanly;
    ``missing_packages`` lists any required package that failed to import.
    ``run_status`` is the adapter's self-declared readiness for the
    benchmark runner — ``stub_only`` means run() will raise
    NotImplementedError.
    """

    method_id: str
    adapter_import_ok: bool
    required_packages: tuple[str, ...]
    missing_packages: tuple[str, ...]
    package_versions: dict[str, str | None]
    supports_binary: bool
    supports_multiclass: bool
    supports_xgboost: bool
    supports_lightgbm: bool
    supports_catboost: bool
    run_status: RunStatus
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "method_id": self.method_id,
            "adapter_import_ok": self.adapter_import_ok,
            "required_packages": list(self.required_packages),
            "missing_packages": list(self.missing_packages),
            "package_versions": self.package_versions,
            "supports_binary": self.supports_binary,
            "supports_multiclass": self.supports_multiclass,
            "supports_xgboost": self.supports_xgboost,
            "supports_lightgbm": self.supports_lightgbm,
            "supports_catboost": self.supports_catboost,
            "run_status": self.run_status,
            "notes": self.notes,
        }


def check_packages(names: tuple[str, ...]) -> list[PackageCheck]:
    """Try to import each package and read its version. No exceptions
    propagate; missing/broken installs are recorded on PackageCheck."""
    out: list[PackageCheck] = []
    for name in names:
        try:
            importlib.import_module(name)
            try:
                ver = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                ver = None
            out.append(PackageCheck(name=name, importable=True, version=ver))
        except Exception as exc:  # noqa: BLE001
            out.append(PackageCheck(
                name=name, importable=False, version=None,
                error=f"{type(exc).__name__}: {exc}"[:200],
            ))
    return out


@dataclass
class AdapterBase:
    """Base class every CC18 method adapter inherits from.

    Subclasses must override the class attributes and may override
    ``supports`` / ``run``. The default ``run`` raises
    NotImplementedError; the default ``supports`` returns True for any
    (binary or multiclass) classification task on any of the three
    GBDTs unless the subclass narrows that.
    """

    method_id: ClassVar[str] = ""
    required_packages: ClassVar[tuple[str, ...]] = ()
    supports_binary: ClassVar[bool] = True
    supports_multiclass: ClassVar[bool] = True
    supports_xgboost: ClassVar[bool] = True
    supports_lightgbm: ClassVar[bool] = True
    supports_catboost: ClassVar[bool] = True
    run_status: ClassVar[RunStatus] = "stub_only"
    notes: ClassVar[str] = ""

    def import_check(self) -> CapabilityStatus:
        checks = check_packages(self.required_packages)
        missing = tuple(c.name for c in checks if not c.importable)
        versions = {c.name: c.version for c in checks}
        return CapabilityStatus(
            method_id=self.method_id,
            adapter_import_ok=True,
            required_packages=self.required_packages,
            missing_packages=missing,
            package_versions=versions,
            supports_binary=self.supports_binary,
            supports_multiclass=self.supports_multiclass,
            supports_xgboost=self.supports_xgboost,
            supports_lightgbm=self.supports_lightgbm,
            supports_catboost=self.supports_catboost,
            run_status=self.run_status,
            notes=self.notes,
        )

    def supports(self, task_metadata: dict, algorithm: str) -> bool:
        task_type = task_metadata.get("task_type", "binary")
        if task_type == "binary" and not self.supports_binary:
            return False
        if task_type == "multiclass" and not self.supports_multiclass:
            return False
        return {
            "xgboost": self.supports_xgboost,
            "lightgbm": self.supports_lightgbm,
            "catboost": self.supports_catboost,
        }.get(algorithm, False)

    def describe(self) -> dict:
        return {
            "method_id": self.method_id,
            "required_packages": list(self.required_packages),
            "supports_binary": self.supports_binary,
            "supports_multiclass": self.supports_multiclass,
            "supports_xgboost": self.supports_xgboost,
            "supports_lightgbm": self.supports_lightgbm,
            "supports_catboost": self.supports_catboost,
            "run_status": self.run_status,
            "notes": self.notes,
        }

    def run(self, *args, **kwargs):  # noqa: ARG002, D401
        """No-op stub. Real implementations land in later commits."""
        raise NotImplementedError(
            f"{self.method_id}.run() is a stub_only adapter in Commit 29"
        )
