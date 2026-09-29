"""Read the raw SST-2 splits and check that they look as expected."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

COLUMNS = ("sentence", "label", "idx")
LABEL_NAMES = {0: "negative", 1: "positive"}


def raw_path(split: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return where the parquet file for ``split`` is stored locally."""
    return Path(raw_dir) / f"sst2_{split}.parquet"


def load_split(split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load one split, failing with instructions if it has not been downloaded."""
    path = raw_path(split, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make data` to download SST-2 "
            "(about 3 MB) from the Hugging Face Hub."
        )
    return pd.read_parquet(path)


def check_split(frame: pd.DataFrame, expected_rows: int) -> list[str]:
    """Return a list of problems with a labelled SST-2 split (empty if none).

    Checks the column set, the row count, that labels are 0 or 1, that no
    sentence is missing or blank, and that ``idx`` identifies rows uniquely.
    """
    problems = []
    if tuple(frame.columns) != COLUMNS:
        problems.append(f"columns are {list(frame.columns)}, expected {list(COLUMNS)}")
        return problems
    if len(frame) != expected_rows:
        problems.append(f"{len(frame)} rows, expected {expected_rows}")
    bad_labels = set(frame["label"].unique()) - set(LABEL_NAMES)
    if bad_labels:
        problems.append(f"unexpected labels {sorted(bad_labels)}")
    blank = frame["sentence"].isna() | (frame["sentence"].astype(str).str.strip() == "")
    if blank.any():
        problems.append(f"{int(blank.sum())} missing or blank sentences")
    if not frame["idx"].is_unique:
        problems.append("idx values are not unique")
    return problems
