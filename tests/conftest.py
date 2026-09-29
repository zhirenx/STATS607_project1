"""Fixtures shared by the tests: configuration, raw data and committed artifacts."""

import pytest

from src.analysis.artifacts import load_runs
from src.config import load_config
from src.data import load_split, raw_path


@pytest.fixture(scope="session")
def config():
    """The experiment configuration from config.toml."""
    return load_config()


@pytest.fixture(scope="session")
def runs(config):
    """The committed artifacts of every model, keyed by model name."""
    return {run.key: run for run in load_runs(config)}


@pytest.fixture(scope="session")
def splits(config):
    """The raw SST-2 splits, keyed by split name.

    Fails, rather than skips, when the data have not been downloaded, so a
    data test can never pass by silently not running.
    """
    missing = [split for split in config.data.splits if not raw_path(split).exists()]
    if missing:
        pytest.fail(f"raw data missing for {missing}: run `make data` first "
                    "(`make test` downloads it automatically)")
    return {split: load_split(split) for split in config.data.splits}
