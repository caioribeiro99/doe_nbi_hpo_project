"""Loader / cache for OpenML-CC18 task payloads.

Used by ``scripts/cc18_runner.py`` once it transitions from synthetic
canaries to real CC18 tasks (Commit 34, batch_01 onward). Restricted
by design to the ``task_id`` set the caller passes — the loader never
iterates the whole 72-task suite. That keeps the network footprint
small and makes the reduced-execution batches auditable.

Cache layout (gitignored):

    data/source/openml_cc18/
      _openml_cache/        # OpenML library cache (set via openml.config)
      <task_id>/
        manifest.json       # task / dataset metadata + payload SHA-256
        payload.pkl         # pickled {"X": np.ndarray, "y": np.ndarray}

The manifest is small and could in principle be versioned, but
batch_01 keeps it gitignored too: this commit is the first allowed
to touch real OpenML data, and we don't want to publish per-task
metadata until the panel composition is reviewed.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO / "data" / "source" / "openml_cc18"
DEFAULT_OPENML_CACHE_SUBDIR = "_openml_cache"


@dataclass(frozen=True)
class CC18TaskPayload:
    """In-memory representation of a CC18 task ready to feed an estimator.

    ``X`` is a dense ``np.float64`` array (categorical columns are
    one-hot encoded; missing entries are imputed with the column
    median). ``y`` is an ``int64`` label-encoded array; the original
    label strings are preserved in ``class_distribution``.
    """

    task_id: int
    dataset_id: int
    dataset_name: str
    target_name: str
    task_type: str  # "binary" | "multiclass"
    X: np.ndarray
    y: np.ndarray
    n_classes: int
    n_rows: int
    n_features: int
    feature_names: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    class_distribution: dict[str, int]
    cache_dir: Path
    payload_sha256: str = ""
    extra: dict = field(default_factory=dict)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def cache_dir_for_task(task_id: int, cache_root: Path | None = None) -> Path:
    cache_root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    return cache_root / str(int(task_id))


def _configure_openml_cache(cache_root: Path) -> None:
    """Point the OpenML library at our gitignored cache subdirectory.

    Done lazily so importing this module never imports openml.
    """
    import openml

    cache = (cache_root / DEFAULT_OPENML_CACHE_SUBDIR).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    openml.config.cache_directory = str(cache)


def _impute_with_column_median(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    col_med = np.nanmedian(X, axis=0)
    # Columns that are all-NaN fall back to 0.
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_med, idx[1])
    return X


def _build_payload_from_openml(task_id: int) -> dict:
    """Fetch one OpenML task + dataset and assemble a serializable payload.

    Returns a dict with the metadata fields plus ``X`` and ``y``
    arrays. Encoded form: one-hot for categorical features, label-
    encoded targets.
    """
    import openml
    import pandas as pd

    task = openml.tasks.get_task(
        int(task_id),
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset = task.get_dataset()
    target_name = task.target_name
    X_df, y_ser, categorical_mask, attribute_names = dataset.get_data(
        target=target_name, dataset_format="dataframe",
    )
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df, columns=list(attribute_names))
    if isinstance(y_ser, np.ndarray):
        y_ser = pd.Series(y_ser)

    categorical_columns = [
        n for n, is_cat in zip(attribute_names, categorical_mask, strict=False)
        if is_cat
    ]
    if categorical_columns:
        X_df = pd.get_dummies(
            X_df, columns=categorical_columns, dummy_na=False,
        )
    feature_names = tuple(map(str, X_df.columns))
    X = X_df.to_numpy(dtype=np.float64, copy=True)
    X = _impute_with_column_median(X)

    y_cat = pd.Series(y_ser).astype("category")
    class_levels = list(y_cat.cat.categories)
    y = y_cat.cat.codes.to_numpy(dtype=np.int64, copy=True)
    if int(y.min(initial=0)) < 0:
        raise ValueError(
            f"task {task_id}: target column has missing labels; refusing"
        )
    n_classes = len(class_levels)
    if n_classes < 2:
        raise ValueError(f"task {task_id}: <2 classes after encoding")
    task_type = "binary" if n_classes == 2 else "multiclass"
    n_rows, n_features = X.shape
    class_distribution = {
        str(class_levels[i]): int((y == i).sum()) for i in range(n_classes)
    }

    return {
        "task_id": int(task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(target_name),
        "task_type": task_type,
        "X": X,
        "y": y,
        "n_classes": int(n_classes),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "feature_names": feature_names,
        "categorical_columns": tuple(categorical_columns),
        "class_distribution": class_distribution,
    }


def load_cc18_task(
    task_id: int,
    *,
    cache_root: Path | None = None,
    allow_download: bool = True,
) -> CC18TaskPayload:
    """Load a single CC18 task, fetching from OpenML on cache miss.

    Parameters
    ----------
    task_id:
        OpenML task id (CC18 is suite 99).
    cache_root:
        Root cache directory. Defaults to
        ``data/source/openml_cc18/`` under the repo. The directory is
        created if missing and is expected to be gitignored.
    allow_download:
        When ``False``, raise ``RuntimeError`` on cache miss instead
        of contacting OpenML. Useful for offline tests.
    """
    cache_root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    cache_dir = cache_dir_for_task(task_id, cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_p = cache_dir / "payload.pkl"
    meta_p = cache_dir / "manifest.json"

    if payload_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        # Verify SHA-256 matches the manifest; rebuild on mismatch.
        try:
            recorded = str(meta.get("payload_sha256", ""))
            actual = _sha256_file(payload_p)
        except Exception:
            recorded = ""
            actual = ""
        if recorded and recorded == actual:
            with payload_p.open("rb") as f:
                obj = pickle.load(f)  # noqa: S301
            return CC18TaskPayload(
                task_id=int(meta["task_id"]),
                dataset_id=int(meta["dataset_id"]),
                dataset_name=str(meta["dataset_name"]),
                target_name=str(meta["target_name"]),
                task_type=str(meta["task_type"]),
                X=np.asarray(obj["X"]),
                y=np.asarray(obj["y"]),
                n_classes=int(meta["n_classes"]),
                n_rows=int(meta["n_rows"]),
                n_features=int(meta["n_features"]),
                feature_names=tuple(meta["feature_names"]),
                categorical_columns=tuple(meta["categorical_columns"]),
                class_distribution=dict(meta["class_distribution"]),
                cache_dir=cache_dir,
                payload_sha256=actual,
            )

    if not allow_download:
        raise RuntimeError(
            f"OpenML cache miss for task {task_id} (cache_dir={cache_dir}) "
            f"and allow_download=False"
        )

    _configure_openml_cache(cache_root)
    payload_dict = _build_payload_from_openml(int(task_id))

    tmp_payload = payload_p.with_suffix(payload_p.suffix + ".tmp")
    with tmp_payload.open("wb") as f:
        pickle.dump(
            {"X": payload_dict["X"], "y": payload_dict["y"]},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )
    payload_sha = _sha256_file(tmp_payload)
    os.replace(tmp_payload, payload_p)

    meta = {
        "task_id": int(payload_dict["task_id"]),
        "dataset_id": int(payload_dict["dataset_id"]),
        "dataset_name": payload_dict["dataset_name"],
        "target_name": payload_dict["target_name"],
        "task_type": payload_dict["task_type"],
        "n_classes": int(payload_dict["n_classes"]),
        "n_rows": int(payload_dict["n_rows"]),
        "n_features": int(payload_dict["n_features"]),
        "feature_names": list(payload_dict["feature_names"]),
        "categorical_columns": list(payload_dict["categorical_columns"]),
        "class_distribution": payload_dict["class_distribution"],
        "payload_filename": payload_p.name,
        "payload_sha256": payload_sha,
        "openml_url": f"https://www.openml.org/t/{int(task_id)}",
    }
    meta_p.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8",
    )

    return CC18TaskPayload(
        task_id=int(payload_dict["task_id"]),
        dataset_id=int(payload_dict["dataset_id"]),
        dataset_name=payload_dict["dataset_name"],
        target_name=payload_dict["target_name"],
        task_type=payload_dict["task_type"],
        X=payload_dict["X"],
        y=payload_dict["y"],
        n_classes=int(payload_dict["n_classes"]),
        n_rows=int(payload_dict["n_rows"]),
        n_features=int(payload_dict["n_features"]),
        feature_names=payload_dict["feature_names"],
        categorical_columns=payload_dict["categorical_columns"],
        class_distribution=payload_dict["class_distribution"],
        cache_dir=cache_dir,
        payload_sha256=payload_sha,
    )


def task_metadata_summary(payload: CC18TaskPayload) -> dict:
    """Small JSON-serializable summary suitable for gate artifacts."""
    return {
        "task_id": int(payload.task_id),
        "dataset_id": int(payload.dataset_id),
        "dataset_name": payload.dataset_name,
        "target_name": payload.target_name,
        "task_type": payload.task_type,
        "n_classes": int(payload.n_classes),
        "n_rows": int(payload.n_rows),
        "n_features": int(payload.n_features),
        "n_categorical_columns": len(payload.categorical_columns),
        "categorical_columns": list(payload.categorical_columns),
        "class_distribution": dict(payload.class_distribution),
        "payload_sha256": payload.payload_sha256,
        "cache_dir": str(payload.cache_dir),
    }


__all__ = [
    "CC18TaskPayload",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_OPENML_CACHE_SUBDIR",
    "cache_dir_for_task",
    "load_cc18_task",
    "task_metadata_summary",
]
