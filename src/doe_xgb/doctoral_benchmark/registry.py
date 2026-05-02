"""Doctoral benchmark dataset registry.

Loads, validates, and merges the CSV registry under
``benchmarks/doctoral_82/datasets.csv``. The schema contract lives at
``benchmarks/doctoral_82/dataset_schema.json``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

VALID_SOURCES = {"uci", "openml", "sklearn", "kaggle", "local", "other"}
VALID_TASK = {"binary", "multiclass"}
VALID_LOADER_STATUS = {"registered", "pending", "broken"}
REQUIRED_COLUMNS = (
    "dataset_id",
    "display_name",
    "source",
    "task_type",
    "include",
    "loader_status",
)
ALL_COLUMNS = (
    "dataset_id",
    "display_name",
    "source",
    "openml_id",
    "uci_id",
    "task_type",
    "n_rows",
    "n_features",
    "n_classes",
    "has_categorical",
    "class_imbalance_ratio",
    "license",
    "include",
    "reason",
    "loader_status",
    "notes",
)


class RegistryError(ValueError):
    """Raised when the CSV registry violates the schema contract."""


@dataclass(frozen=True)
class DatasetRow:
    dataset_id: str
    display_name: str
    source: str
    task_type: str
    include: bool
    loader_status: str
    openml_id: int | None = None
    uci_id: int | None = None
    n_rows: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    has_categorical: bool | None = None
    class_imbalance_ratio: float | None = None
    license: str | None = None
    reason: str | None = None
    notes: str | None = None

    def to_csv_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for k in ALL_COLUMNS:
            v = getattr(self, k, None)
            if v is None:
                out[k] = ""
            elif isinstance(v, bool):
                out[k] = "True" if v else "False"
            else:
                out[k] = str(v)
        return out


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    s = value.strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    raise RegistryError(f"invalid bool value: {value!r}")


def _coerce_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _coerce_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _coerce_str(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip()
    return s if s != "" else None


def canonical_row(raw: dict[str, object]) -> DatasetRow:
    """Coerce a raw CSV row into a typed :class:`DatasetRow`. Validates
    required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw]
    if missing:
        raise RegistryError(f"missing required columns: {missing}")

    dataset_id = _coerce_str(str(raw["dataset_id"])) or ""
    if not dataset_id:
        raise RegistryError("dataset_id must not be empty")
    if " " in dataset_id:
        raise RegistryError(f"dataset_id {dataset_id!r} must not contain spaces")

    source = _coerce_str(str(raw["source"])) or ""
    if source not in VALID_SOURCES:
        raise RegistryError(f"invalid source {source!r}; expected one of {sorted(VALID_SOURCES)}")

    task_type = _coerce_str(str(raw["task_type"])) or ""
    if task_type not in VALID_TASK:
        raise RegistryError(f"invalid task_type {task_type!r}; expected one of {sorted(VALID_TASK)}")

    loader_status = _coerce_str(str(raw["loader_status"])) or ""
    if loader_status not in VALID_LOADER_STATUS:
        raise RegistryError(
            f"invalid loader_status {loader_status!r}; expected one of {sorted(VALID_LOADER_STATUS)}"
        )

    include = _coerce_bool(str(raw["include"]))
    if include is None:
        raise RegistryError("'include' must be a bool literal")

    if include is True and loader_status != "registered":
        raise RegistryError(
            f"dataset_id={dataset_id!r}: include=True requires loader_status=registered "
            f"(got {loader_status!r})."
        )

    n_classes = _coerce_int(str(raw.get("n_classes", "")))
    if task_type == "binary" and n_classes not in (None, 2):
        raise RegistryError(f"dataset_id={dataset_id!r}: task_type=binary requires n_classes IN {{NULL, 2}} (got {n_classes})")
    if task_type == "multiclass" and n_classes is not None and n_classes < 3:
        raise RegistryError(f"dataset_id={dataset_id!r}: task_type=multiclass requires n_classes >= 3 (got {n_classes})")

    return DatasetRow(
        dataset_id=dataset_id,
        display_name=_coerce_str(str(raw.get("display_name", ""))) or dataset_id,
        source=source,
        openml_id=_coerce_int(str(raw.get("openml_id", ""))),
        uci_id=_coerce_int(str(raw.get("uci_id", ""))),
        task_type=task_type,
        n_rows=_coerce_int(str(raw.get("n_rows", ""))),
        n_features=_coerce_int(str(raw.get("n_features", ""))),
        n_classes=n_classes,
        has_categorical=_coerce_bool(str(raw.get("has_categorical", ""))),
        class_imbalance_ratio=_coerce_float(str(raw.get("class_imbalance_ratio", ""))),
        license=_coerce_str(str(raw.get("license", ""))),
        include=include,
        reason=_coerce_str(str(raw.get("reason", ""))),
        loader_status=loader_status,
        notes=_coerce_str(str(raw.get("notes", ""))),
    )


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_registry_csv(path: Path | str) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RegistryError(f"empty CSV: {p}")
        for raw in reader:
            rows.append(canonical_row(raw))
    return rows


def validate_registry(rows: Iterable[DatasetRow]) -> None:
    rows_list = list(rows)
    seen_ids: set[str] = set()
    seen_openml: dict[int, str] = {}
    for r in rows_list:
        if r.dataset_id in seen_ids:
            raise RegistryError(f"duplicate dataset_id: {r.dataset_id!r}")
        seen_ids.add(r.dataset_id)
        if r.openml_id is not None:
            if r.openml_id in seen_openml:
                raise RegistryError(
                    f"duplicate openml_id={r.openml_id} between {seen_openml[r.openml_id]!r} "
                    f"and {r.dataset_id!r}"
                )
            seen_openml[r.openml_id] = r.dataset_id


# ---------------------------------------------------------------------------
# Merging two registries (idempotent on dataset_id)
# ---------------------------------------------------------------------------


def merge_registries(
    base: Iterable[DatasetRow],
    incoming: Iterable[DatasetRow],
    *,
    overwrite_existing: bool = False,
) -> list[DatasetRow]:
    """Merge ``incoming`` rows into ``base`` rows, keyed on ``dataset_id``.

    By default, existing rows are preserved (the importer is additive).
    Pass ``overwrite_existing=True`` to make incoming rows win.
    """
    by_id: dict[str, DatasetRow] = {}
    for r in base:
        by_id[r.dataset_id] = r
    for r in incoming:
        if r.dataset_id in by_id and not overwrite_existing:
            continue
        by_id[r.dataset_id] = r
    out = list(by_id.values())
    validate_registry(out)
    return out


def write_registry_csv(rows: Iterable[DatasetRow], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows_list = sorted(rows, key=lambda r: r.dataset_id)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for r in rows_list:
            writer.writerow(r.to_csv_dict())
    return p


__all__ = [
    "DatasetRow",
    "RegistryError",
    "canonical_row",
    "load_registry_csv",
    "merge_registries",
    "validate_registry",
    "write_registry_csv",
    "ALL_COLUMNS",
    "REQUIRED_COLUMNS",
]
