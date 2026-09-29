"""Data validation: the raw SST-2 files are the ones the analysis expects."""

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype, is_string_dtype

from src.data import check_split, raw_path
from src.pipeline.download_data import sha256sum


def test_raw_files_match_pinned_checksums(config, splits):
    for split, spec in config.data.splits.items():
        assert sha256sum(raw_path(split)) == spec.sha256, split


@pytest.mark.parametrize("split", ["train", "validation"])
def test_split_passes_all_checks(config, splits, split):
    assert check_split(splits[split], config.data.splits[split].rows) == []


@pytest.mark.parametrize("split", ["train", "validation"])
def test_column_types(splits, split):
    frame = splits[split]
    assert is_string_dtype(frame["sentence"])
    assert is_integer_dtype(frame["label"])
    assert is_integer_dtype(frame["idx"])


def test_class_balance_matches_original_notebook(splits):
    # Printed by the original notebook: 29780/37569 (train), 428/444 (validation).
    assert splits["train"]["label"].value_counts().sort_index().tolist() == [29780, 37569]
    assert splits["validation"]["label"].value_counts().sort_index().tolist() == [428, 444]


def good_frame():
    return pd.DataFrame({"sentence": ["fine film", "dull"], "label": [1, 0], "idx": [0, 1]})


def test_check_split_accepts_a_clean_frame():
    assert check_split(good_frame(), expected_rows=2) == []


@pytest.mark.parametrize(
    "corrupt, expected",
    [
        (lambda f: f.rename(columns={"label": "y"}), "columns"),
        (lambda f: f.iloc[:1], "rows"),
        (lambda f: f.assign(label=[1, 2]), "unexpected labels"),
        (lambda f: f.assign(sentence=["fine film", None]), "blank"),
        (lambda f: f.assign(sentence=["fine film", "   "]), "blank"),
        (lambda f: f.assign(idx=[0, 0]), "not unique"),
    ],
)
def test_check_split_reports_each_problem(corrupt, expected):
    problems = check_split(corrupt(good_frame()), expected_rows=2)
    assert any(expected in problem for problem in problems), problems
