"""Load the exported training logs and validation predictions.

These files are committed to the repository (see ``src/pipeline/export.py``),
so the analysis runs without model checkpoints, a GPU, or PyTorch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, ROOT, RUN_INFO, Config

PREDICTION_COLUMNS = ["idx", "label", "pred", "prob_positive", "logit_negative", "logit_positive"]


@dataclass(frozen=True)
class ModelRun:
    """Everything the analysis needs about one fine-tuned model.

    ``trainer_state`` is the Hugging Face Trainer's own log, ``run_info``
    records how and where the model was trained, ``export_info`` records how
    the predictions were produced, and ``predictions`` holds one row per
    validation sentence.
    """

    key: str
    display_name: str
    trainer_state: dict
    run_info: dict
    export_info: dict
    predictions: pd.DataFrame

    @property
    def log_history(self) -> list[dict]:
        """Return the Trainer's list of logged training and evaluation entries."""
        return self.trainer_state["log_history"]


def _read(path: Path) -> Path:
    """Return ``path``, or raise an error that says how to get the file back."""
    if path.exists():
        return path
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    raise FileNotFoundError(
        f"{shown} is missing. It is committed to the repository, so restore it with "
        f"`git checkout -- {shown}`, or re-export it with `make artifacts` on a machine "
        "that has the model checkpoints in artifacts/models/."
    )


def load_run(key: str, display_name: str, artifacts_dir: Path = ARTIFACTS_DIR) -> ModelRun:
    """Load the exported artifacts of model ``key`` from ``artifacts_dir``."""
    logs = Path(artifacts_dir) / "training_logs" / key
    predictions_csv = Path(artifacts_dir) / "predictions" / f"{key}_validation.csv"
    predictions = pd.read_csv(_read(predictions_csv))
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            f"{predictions_csv} has columns {list(predictions.columns)}, expected {PREDICTION_COLUMNS}"
        )
    return ModelRun(
        key=key,
        display_name=display_name,
        trainer_state=json.loads(_read(logs / "trainer_state.json").read_text()),
        run_info=json.loads(_read(logs / RUN_INFO).read_text()),
        export_info=json.loads(_read(predictions_csv.with_suffix(".json")).read_text()),
        predictions=predictions.sort_values("idx", ignore_index=True),
    )


def load_runs(config: Config, artifacts_dir: Path = ARTIFACTS_DIR) -> list[ModelRun]:
    """Load every model in ``config``, in configuration order."""
    return [load_run(spec.key, spec.display_name, artifacts_dir) for spec in config.models.values()]
