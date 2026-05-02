"""Unit tests for the shared downloader utility module."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from doe_xgb.datasets import download as dl


def test_sha256_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello world")
    h = dl.sha256_file(p)
    assert h == dl.sha256_bytes(b"hello world")
    assert len(h) == 64


def test_extract_zip(tmp_path: Path) -> None:
    src_zip = tmp_path / "data.zip"
    inner = tmp_path / "inner.csv"
    inner.write_text("a,b\n1,2\n")
    with ZipFile(src_zip, "w") as zf:
        zf.write(inner, arcname="inner.csv")
    out_dir = tmp_path / "out"
    files = dl.extract_zip(src_zip, out_dir)
    assert len(files) == 1
    assert files[0].read_text() == "a,b\n1,2\n"


def test_download_url_idempotent(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "downloaded.bin"
    target.write_bytes(b"stub-content")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=60.0):  # pragma: no cover - not invoked
        calls["n"] += 1
        raise AssertionError("download_url should not call network when file exists")

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    out = dl.download_url("http://example.invalid/file", target)
    assert out == target
    assert calls["n"] == 0


def test_download_url_force_redownload(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "downloaded.bin"
    target.write_bytes(b"stub-content")

    class FakeResp:
        def __init__(self) -> None:
            self._buf = b"new-content"

        def read(self, _size: int) -> bytes:
            data = self._buf
            self._buf = b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=60.0):
        return FakeResp()

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    dl.download_url("http://example.invalid/file", target, force=True)
    assert target.read_bytes() == b"new-content"


def test_manifest_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dl, "data_root", lambda: tmp_path)
    raw = tmp_path / "demo" / "raw"
    proc = tmp_path / "demo" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    (raw / "raw.csv").write_text("x\n1\n")
    (proc / "demo.csv").write_text("x\n1\n")
    dl.write_manifest(
        "demo",
        raw_files=[raw / "raw.csv"],
        processed_files=[proc / "demo.csv"],
        source_url="http://example.invalid/raw.csv",
        openml_id=None,
        target_column="x",
        target_transform=None,
        notes="test",
    )
    mp = tmp_path / "demo" / "manifest.json"
    assert mp.exists()
    blob = json.loads(mp.read_text())
    assert blob["dataset_id"] == "demo"
    assert blob["target_column"] == "x"
    assert len(blob["raw_files"]) == 1
    assert len(blob["processed_files"]) == 1
    assert dl.manifest_matches_disk("demo") is True


def test_manifest_mismatch_after_modification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dl, "data_root", lambda: tmp_path)
    raw = tmp_path / "demo" / "raw"
    proc = tmp_path / "demo" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    (raw / "raw.csv").write_text("x\n1\n")
    (proc / "demo.csv").write_text("x\n1\n")
    dl.write_manifest(
        "demo",
        raw_files=[raw / "raw.csv"],
        processed_files=[proc / "demo.csv"],
        source_url=None,
        openml_id=None,
        target_column="x",
        target_transform=None,
        notes=None,
    )
    assert dl.manifest_matches_disk("demo") is True
    # Mutate file -> mismatch.
    (proc / "demo.csv").write_text("x\n2\n")
    assert dl.manifest_matches_disk("demo") is False


def test_update_checksums_txt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dl, "data_root", lambda: tmp_path)
    raw = tmp_path / "demo" / "raw"
    proc = tmp_path / "demo" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    (raw / "raw.csv").write_text("x\n1\n")
    (proc / "demo.csv").write_text("x\n1\n")
    dl.write_manifest(
        "demo",
        raw_files=[raw / "raw.csv"],
        processed_files=[proc / "demo.csv"],
        source_url="http://example.invalid/raw.csv",
        openml_id=None,
        target_column="x",
        target_transform=None,
        notes=None,
    )
    cp = dl.update_checksums_txt("demo")
    body = cp.read_text()
    assert "# >>> demo" in body
    assert "# <<< demo" in body
    assert "demo/processed/demo.csv" in body


def test_verify_checksums_helper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dl, "data_root", lambda: tmp_path)
    raw = tmp_path / "demo" / "raw"
    proc = tmp_path / "demo" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    (raw / "raw.csv").write_text("a\n1\n")
    (proc / "demo.csv").write_text("a\n1\n")
    dl.write_manifest(
        "demo",
        raw_files=[raw / "raw.csv"],
        processed_files=[proc / "demo.csv"],
        source_url=None,
        openml_id=None,
        target_column="a",
        target_transform=None,
        notes=None,
    )
    out = dl.verify_checksums()
    assert out == {"demo": True}


def test_processed_csv_loads_as_dataframe(tmp_path: Path) -> None:
    """A processed CSV must be a normal pandas-readable file."""
    p = tmp_path / "demo.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(p, index=False)
    df = pd.read_csv(p)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3


def test_run_fetcher_with_synthetic_process(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: fake URL download + process -> manifest + checksums."""
    monkeypatch.setattr(dl, "data_root", lambda: tmp_path)

    # Patch downloader to write a deterministic raw file without network.
    def fake_download(url, dest, *, force=False, timeout=60.0, user_agent="x"):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("a,b\n1,0\n2,1\n")
        return Path(dest)

    monkeypatch.setattr(dl, "download_url", fake_download)

    # Inject our own dataset id into the registry temporarily.
    from doe_xgb.datasets import registry
    from doe_xgb.datasets.metadata import DatasetMetadata

    fake_meta = DatasetMetadata(
        dataset_id="zzz_demo",
        display_name="Demo",
        source_type="uci",
        source_url="http://example.invalid/raw.csv",
        target_column="b",
    )
    registry.REGISTRY["zzz_demo"] = fake_meta
    try:
        # Now invoke run_fetcher via a plain import of the helper module.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_ds_base",
            Path(__file__).resolve().parents[2] / "scripts" / "_dataset_fetch_base.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def process(raw_dir: Path, processed_dir: Path) -> list[Path]:
            src = raw_dir / "raw.csv"
            dst = processed_dir / "demo.csv"
            dst.write_text(src.read_text())
            return [dst]

        rc = mod.run_fetcher(
            dataset_id="zzz_demo",
            raw_url="http://example.invalid/raw.csv",
            raw_filename="raw.csv",
            process_fn=process,
            argv=[],
        )
        assert rc == 0
        assert (tmp_path / "zzz_demo" / "manifest.json").exists()
        assert (tmp_path / "zzz_demo" / "processed" / "demo.csv").exists()
        assert dl.manifest_matches_disk("zzz_demo") is True

        # Idempotency: calling again is a no-op.
        rc2 = mod.run_fetcher(
            dataset_id="zzz_demo",
            raw_url="http://example.invalid/raw.csv",
            raw_filename="raw.csv",
            process_fn=process,
            argv=[],
        )
        assert rc2 == 0
    finally:
        del registry.REGISTRY["zzz_demo"]
