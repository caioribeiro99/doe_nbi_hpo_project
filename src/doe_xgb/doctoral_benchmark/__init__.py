"""Doctoral benchmark registry + job-matrix planning utilities.

Pure-Python and import-light. The registry CSV is the source of truth;
this module provides validation and (later) job-matrix generation.
"""

from __future__ import annotations

from .jobs import generate_job_rows, job_id, stage_topup_replicas
from .registry import (
    DatasetRow,
    RegistryError,
    canonical_row,
    load_registry_csv,
    merge_registries,
    validate_registry,
)

__all__ = [
    "DatasetRow",
    "RegistryError",
    "canonical_row",
    "generate_job_rows",
    "job_id",
    "load_registry_csv",
    "merge_registries",
    "stage_topup_replicas",
    "validate_registry",
]
