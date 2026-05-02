"""Unit tests for the v1 dataset registry, availability checks, and loaders.

Tests do not perform real network downloads. The availability probe
uses a monkeypatch so the suite is deterministic and CI-friendly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from doe_xgb.datasets import (
    REGISTRY,
    V1_INCLUDED,
    AvailabilityResult,
    DatasetUnavailableError,
    availability,
    check_all,
    check_dataset,
    get_metadata,
    list_dataset_ids,
    load,
    load_breast_cancer,
    load_magic,
    write_availability_report,
)
from doe_xgb.datasets.metadata import DatasetMetadata

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_has_twelve_v1_entries() -> None:
    assert len(V1_INCLUDED) == 12
    assert len(REGISTRY) == 12


def test_each_entry_is_consistent() -> None:
    expected_ids = {
        "magic", "breast_cancer", "pima_diabetes", "spambase",
        "adult", "bank_marketing", "credit_card_default", "german_credit",
        "wine_quality", "dry_bean", "mushroom", "phishing",
    }
    assert set(list_dataset_ids()) == expected_ids
    for did, meta in REGISTRY.items():
        assert isinstance(meta, DatasetMetadata)
        assert meta.dataset_id == did
        assert meta.task_type in ("binary", "multiclass")
        assert meta.source_type in ("uci", "openml", "sklearn", "local")
        assert meta.include_in_v1 is True


def test_get_metadata_round_trip() -> None:
    meta = get_metadata("breast_cancer")
    assert meta.source_type == "sklearn"
    with pytest.raises(KeyError):
        get_metadata("nope")


def test_dry_bean_is_multiclass_others_binary() -> None:
    multiclass = [d.dataset_id for d in REGISTRY.values() if d.task_type == "multiclass"]
    assert multiclass == ["dry_bean"]


# ---------------------------------------------------------------------------
# Availability tests (no real network)
# ---------------------------------------------------------------------------


def test_check_dataset_for_sklearn_skips_network() -> None:
    res = check_dataset(REGISTRY["breast_cancer"])
    assert isinstance(res, AvailabilityResult)
    assert res.status == "available"
    assert res.canonical_url is None
    assert "scikit-learn" in (res.reason or "") or "sklearn" in (res.reason or "")


def test_check_dataset_uci_uses_probe(monkeypatch) -> None:
    seen: list[str] = []

    def fake_probe(url: str, timeout: float = 5.0):
        seen.append(url)
        return True, 200, None

    monkeypatch.setattr(availability, "_probe_url", fake_probe)
    res = check_dataset(REGISTRY["magic"])
    assert res.status == "available"
    assert seen and "magic04.data" in seen[0]
    assert res.http_status == 200


def test_check_all_with_mocked_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        availability,
        "_probe_url",
        lambda url, timeout=5.0: (True, 200, None),
    )
    results = check_all(timeout=0.1)
    assert len(results) == 12
    assert all(r.status == "available" for r in results)


def test_unavailable_does_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(
        availability,
        "_probe_url",
        lambda url, timeout=5.0: (False, 404, "HTTP 404"),
    )
    res = check_dataset(REGISTRY["mushroom"])
    assert res.status == "unavailable"
    assert "404" in (res.reason or "")


def test_write_availability_report_creates_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        availability,
        "_probe_url",
        lambda url, timeout=5.0: (True, 200, None),
    )
    results = check_all(timeout=0.1)
    md = tmp_path / "AVAILABILITY_CHECK.md"
    js = tmp_path / "dataset_registry.json"
    write_availability_report(results, out_md=md, out_json=js)
    assert md.exists()
    assert js.exists()
    content = md.read_text()
    assert "# Dataset availability report" in content
    for did in list_dataset_ids():
        assert did in content


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_breast_cancer_runs_offline() -> None:
    ds = load_breast_cancer()
    assert ds.metadata.dataset_id == "breast_cancer"
    assert ds.X.shape[0] == ds.metadata.n_rows
    assert ds.X.shape[1] == ds.metadata.n_features
    assert ds.metadata.task_type == "binary"
    # Two classes, both present.
    assert ds.metadata.class_distribution is not None
    assert set(ds.metadata.class_distribution.keys()) == {"0", "1"}
    # Target removed from X.
    assert "target" not in ds.X.columns


def test_load_dispatch_unknown_id() -> None:
    with pytest.raises(KeyError):
        load("nope")


def test_load_magic_when_xlsx_cached() -> None:
    """Use the canonical telescope2.xlsx if present; otherwise skip."""
    from doe_xgb.datasets.loaders import _data_root

    xlsx = _data_root() / "telescope2.xlsx"
    if not xlsx.exists():
        pytest.skip("telescope2.xlsx not cached locally")
    ds = load_magic()
    assert ds.metadata.task_type == "binary"
    assert ds.X.shape[0] > 1000
    assert "y" not in ds.X.columns and "class" not in ds.X.columns


def test_load_magic_csv_fallback(tmp_path: Path, monkeypatch) -> None:
    """Construct a synthetic UCI-style .data file and verify loader parses it."""
    df = pd.DataFrame(
        {
            "fLength": [1.0, 2.0, 3.0, 4.0],
            "fWidth": [0.1, 0.2, 0.3, 0.4],
            "fSize": [1.1, 2.2, 3.3, 4.4],
            "fConc": [0.5] * 4,
            "fConc1": [0.5] * 4,
            "fAsym": [0.0] * 4,
            "fM3Long": [1.0] * 4,
            "fM3Trans": [0.0] * 4,
            "fAlpha": [10.0] * 4,
            "fDist": [100.0] * 4,
            "class": ["g", "g", "h", "h"],
        }
    )
    fake_root = tmp_path / "data" / "source"
    (fake_root / "magic").mkdir(parents=True)
    df.to_csv(fake_root / "magic" / "magic04.data", index=False, header=False)

    monkeypatch.setattr(
        "doe_xgb.datasets.loaders._data_root", lambda: fake_root
    )
    ds = load_magic()
    assert ds.metadata.n_rows == 4
    assert "class" not in ds.X.columns
    assert ds.metadata.class_distribution == {"0": 2, "1": 2}


def test_loaders_raise_when_cache_missing(tmp_path: Path, monkeypatch) -> None:
    """Each non-sklearn loader should raise DatasetUnavailableError pointing
    at the canonical source when the local cache is absent."""
    monkeypatch.setattr(
        "doe_xgb.datasets.loaders._data_root", lambda: tmp_path
    )
    for did in [
        "magic", "pima_diabetes", "spambase", "adult", "bank_marketing",
        "credit_card_default", "german_credit", "wine_quality", "dry_bean",
        "mushroom", "phishing",
    ]:
        with pytest.raises(DatasetUnavailableError) as excinfo:
            load(did)
        assert excinfo.value.metadata.dataset_id == did


def test_load_dry_bean_synthetic(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "Area": [1, 2, 3, 4, 5, 6],
            "Perimeter": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Class": ["A", "B", "C", "A", "B", "C"],
        }
    )
    fake_root = tmp_path
    (fake_root / "dry_bean").mkdir(parents=True)
    df.to_csv(fake_root / "dry_bean" / "dry_bean.csv", index=False)
    monkeypatch.setattr("doe_xgb.datasets.loaders._data_root", lambda: fake_root)
    ds = load("dry_bean")
    assert ds.metadata.task_type == "multiclass"
    assert set(ds.metadata.class_distribution.keys()) == {"0", "1", "2"}
    assert "Class" not in ds.X.columns


def test_load_mushroom_synthetic(tmp_path: Path, monkeypatch) -> None:
    """Mushroom loader maps {p,e} -> {1,0}."""
    columns = ["class"] + list(REGISTRY["mushroom"].categorical_columns)
    rows = [["e"] + ["x"] * (len(columns) - 1)] * 3 + [["p"] + ["y"] * (len(columns) - 1)] * 2
    df = pd.DataFrame(rows, columns=columns)
    fake_root = tmp_path
    (fake_root / "mushroom").mkdir(parents=True)
    df.to_csv(fake_root / "mushroom" / "mushroom.csv", index=False)
    monkeypatch.setattr("doe_xgb.datasets.loaders._data_root", lambda: fake_root)
    ds = load("mushroom")
    assert ds.metadata.class_distribution == {"0": 3, "1": 2}
    assert "class" not in ds.X.columns


def test_load_wine_quality_binarises(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "fixed acidity": [7.0, 7.5, 8.0, 6.0],
            "volatile acidity": [0.5, 0.4, 0.3, 0.6],
            "citric acid": [0.1] * 4,
            "residual sugar": [2.0] * 4,
            "chlorides": [0.05] * 4,
            "free sulfur dioxide": [10.0] * 4,
            "total sulfur dioxide": [50.0] * 4,
            "density": [0.99] * 4,
            "pH": [3.3] * 4,
            "sulphates": [0.6] * 4,
            "alcohol": [10.0, 10.5, 11.0, 9.0],
            "quality": [5, 6, 7, 4],
        }
    )
    fake_root = tmp_path
    (fake_root / "wine_quality").mkdir(parents=True)
    df.to_csv(fake_root / "wine_quality" / "winequality-red.csv", sep=";", index=False)
    monkeypatch.setattr("doe_xgb.datasets.loaders._data_root", lambda: fake_root)
    ds = load("wine_quality")
    assert ds.metadata.class_distribution == {"0": 2, "1": 2}
    assert "quality" not in ds.X.columns
    # Binarised at quality >= 6: rows with quality {6, 7} -> 1; rows {5, 4} -> 0.
    assert ds.y.sum() == 2
