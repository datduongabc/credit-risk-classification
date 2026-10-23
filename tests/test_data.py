import hashlib
import zipfile

import pytest

from creditrisk import config
from creditrisk.data import download_dataset

CSV_BYTES = b"person_age,loan_status\n25,0\n40,1\n"


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """Local file:// stand-ins for the mirror (bare CSV) and Kaggle (zip archive)."""
    monkeypatch.setattr(config, "DATASET_SHA256", hashlib.sha256(CSV_BYTES).hexdigest())
    src = tmp_path / "src"
    src.mkdir()
    bare = src / "mirror.csv"
    bare.write_bytes(CSV_BYTES)
    archive = src / "kaggle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(config.CSV_NAME, CSV_BYTES)
    missing = src / "missing.csv"
    return {"bare": bare.as_uri(), "zip": archive.as_uri(), "missing": missing.as_uri()}


def test_download_bare_csv(sources, tmp_path):
    path = download_dataset(tmp_path / "data", urls=(sources["bare"],))
    assert path.read_bytes() == CSV_BYTES


def test_download_unpacks_zip(sources, tmp_path):
    path = download_dataset(tmp_path / "data", urls=(sources["zip"],))
    assert path.read_bytes() == CSV_BYTES


def test_download_falls_back_when_first_source_fails(sources, tmp_path):
    path = download_dataset(tmp_path / "data", urls=(sources["missing"], sources["zip"]))
    assert path.read_bytes() == CSV_BYTES


def test_download_rejects_unexpected_file(sources, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASET_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="unexpected SHA-256"):
        download_dataset(tmp_path / "data", urls=(sources["bare"],))
    assert not (tmp_path / "data" / config.CSV_NAME).exists()
