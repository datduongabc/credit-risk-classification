"""Dataset download, loading, rule-based cleaning and the train/test split."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _fetch_csv(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    # Kaggle serves a zip archive, the mirror serves the bare CSV.
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            payload = zf.read(config.CSV_NAME)
    return payload


def download_dataset(data_dir: Path | None = None, urls: tuple[str, ...] = config.DATASET_URLS) -> Path:
    """Download the CSV once from the first working source; later calls reuse the local copy."""
    data_dir = Path(data_dir or config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / config.CSV_NAME
    if csv_path.exists():
        return csv_path
    errors = []
    for url in urls:
        try:
            payload = _fetch_csv(url)
        except (urllib.error.URLError, TimeoutError, KeyError, zipfile.BadZipFile) as exc:
            errors.append(f"{url}: {exc}")
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest != config.DATASET_SHA256:
            errors.append(f"{url}: unexpected SHA-256 {digest}")
            continue
        csv_path.write_bytes(payload)
        print(f"Downloaded {config.CSV_NAME} from {url}")
        return csv_path
    raise RuntimeError("Could not download the dataset from any source:\n" + "\n".join(errors))


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(csv_path or download_dataset())


@dataclass
class CleaningReport:
    n_raw: int
    n_duplicates: int
    n_invalid_age: int
    n_invalid_emp_length: int
    n_clean: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def impossible_masks(df: pd.DataFrame, max_age: int = 100) -> tuple[pd.Series, pd.Series]:
    """Rows whose age, or whose employment length, cannot be true."""
    bad_age = df["person_age"] > max_age
    # Employment that started before age 14 is impossible (e.g. a 22-year-old with 123 years).
    bad_emp = df["person_emp_length"] > (df["person_age"] - 14).clip(lower=0)
    return bad_age, bad_emp


def clean(df: pd.DataFrame, max_age: int = 100) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove exact duplicates and turn physically impossible values into missing values.

    The rules are deterministic and row-local (no statistics are learned), so applying them
    before the train/test split cannot leak information. Impossible values become NaN instead
    of being dropped so that the configurable imputer decides how to fill them.
    """
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dup = n_raw - len(df)

    bad_age, bad_emp = impossible_masks(df, max_age)
    df = df.copy()
    df.loc[bad_age, "person_age"] = np.nan
    df.loc[bad_emp, "person_emp_length"] = np.nan

    report = CleaningReport(n_raw, n_dup, int(bad_age.sum()), int(bad_emp.sum()), len(df))
    return df, report


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = config.SEED):
    """Stratified hold-out split; the test set is only touched for final evaluation."""
    X = df[config.FEATURES]
    y = df[config.TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def load_splits(test_size: float = 0.2, seed: int = config.SEED):
    df, report = clean(load_raw())
    X_train, X_test, y_train, y_test = split(df, test_size, seed)
    return X_train, X_test, y_train, y_test, report
