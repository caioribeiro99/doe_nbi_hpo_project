"""Article-track dataset registry, availability checks, and loaders.

Import-light: this package does not pull pandas / sklearn at import
time. Loaders import heavy deps lazily on first use.
"""

from __future__ import annotations

from .availability import (
    AvailabilityResult,
    check_all,
    check_dataset,
    write_availability_report,
)
from .loaders import (
    DatasetUnavailableError,
    load,
    load_adult,
    load_bank_marketing,
    load_breast_cancer,
    load_credit_card_default,
    load_dry_bean,
    load_german_credit,
    load_magic,
    load_mushroom,
    load_phishing,
    load_pima_diabetes,
    load_spambase,
    load_wine_quality,
)
from .metadata import DatasetMetadata, LoadedDataset
from .registry import REGISTRY, V1_INCLUDED, get_metadata, list_dataset_ids

__all__ = [
    "AvailabilityResult",
    "DatasetMetadata",
    "DatasetUnavailableError",
    "LoadedDataset",
    "REGISTRY",
    "V1_INCLUDED",
    "check_all",
    "check_dataset",
    "get_metadata",
    "list_dataset_ids",
    "load",
    "load_adult",
    "load_bank_marketing",
    "load_breast_cancer",
    "load_credit_card_default",
    "load_dry_bean",
    "load_german_credit",
    "load_magic",
    "load_mushroom",
    "load_phishing",
    "load_pima_diabetes",
    "load_spambase",
    "load_wine_quality",
    "write_availability_report",
]
