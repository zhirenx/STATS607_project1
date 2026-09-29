"""Load the experiment configuration and define the project's standard paths.

Every script takes its settings from ``config.toml`` at the repository root.
This module parses that file into frozen dataclasses, so a misspelled or
missing key fails loudly at start-up instead of silently changing an
experiment.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.toml"

RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Written by src/pipeline/train.py when a run finishes; marks a complete run.
RUN_INFO = "run_info.json"


@dataclass(frozen=True)
class SplitConfig:
    """Remote location, checksum and size of one SST-2 split."""

    name: str
    remote_path: str
    sha256: str
    rows: int


@dataclass(frozen=True)
class DataConfig:
    """Pinned source of the SST-2 data on the Hugging Face Hub."""

    hub_repo: str
    revision: str
    splits: dict[str, SplitConfig]

    def url(self, split: str) -> str:
        """Return the download URL of ``split`` at the pinned revision."""
        return (
            f"https://huggingface.co/datasets/{self.hub_repo}/resolve/"
            f"{self.revision}/{self.splits[split].remote_path}"
        )


@dataclass(frozen=True)
class ModelConfig:
    """A pre-trained checkpoint to fine-tune, pinned to a Hub revision."""

    key: str
    display_name: str
    hub_id: str
    revision: str


@dataclass(frozen=True)
class TrainingConfig:
    """Fine-tuning hyperparameters shared by all models."""

    max_length: int
    num_train_epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int


@dataclass(frozen=True)
class SmokeConfig:
    """Settings for the short run that exercises the training code."""

    train_rows: int
    validation_rows: int
    num_train_epochs: int


@dataclass(frozen=True)
class Config:
    """The complete experiment configuration."""

    seed: int
    data: DataConfig
    training: TrainingConfig
    smoke: SmokeConfig
    models: dict[str, ModelConfig]

    def model(self, key: str) -> ModelConfig:
        """Return the model called ``key``, or raise a helpful ``KeyError``."""
        try:
            return self.models[key]
        except KeyError:
            raise KeyError(
                f"unknown model {key!r}; expected one of {sorted(self.models)}"
            ) from None

    def restrict(self, keys: list[str]) -> Config:
        """Return a copy of the configuration that keeps only the models in ``keys``."""
        return dataclasses.replace(self, models={key: self.model(key) for key in keys})


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Parse the TOML file at ``path`` into a :class:`Config`.

    Models keep the order in which they appear in the file; tables and
    figures use that order.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    data = raw["data"]
    splits = {
        name: SplitConfig(name=name, **spec) for name, spec in data["splits"].items()
    }
    models = {
        key: ModelConfig(key=key, **spec) for key, spec in raw["models"].items()
    }
    return Config(
        seed=raw["seed"],
        data=DataConfig(
            hub_repo=data["hub_repo"], revision=data["revision"], splits=splits
        ),
        training=TrainingConfig(**raw["training"]),
        smoke=SmokeConfig(**raw["smoke"]),
        models=models,
    )
