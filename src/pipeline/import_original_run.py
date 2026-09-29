"""Record provenance for the checkpoints trained by the original notebook.

The original notebook trained the three models before this pipeline
existed, so its checkpoint directories lack the ``run_info.json`` that
``src/pipeline/train.py`` writes at the end of a run. This one-off script
creates that file for each model from the notebook's saved outputs (the
wall-clock time shown by the Trainer's progress bar, the device and the
library versions it printed) and from the checkpoints' file times. Each
field names its source. The export step copies the file into the committed
training logs.

Usage (the notebook is kept in the commit tagged ``original``)::

    git show original:sst2_project.ipynb > /tmp/original.ipynb
    python -m src.pipeline.import_original_run --notebook /tmp/original.ipynb
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from src.config import DEFAULT_CONFIG, MODELS_DIR, RUN_INFO, load_config
from src.log import setup_logging

log = logging.getLogger(__name__)

PROGRESS = re.compile(r"\[(\d+)/(\d+) (\d+(?::\d\d)+), Epoch (\d+)/(\d+)\]")
MODEL_NAME = re.compile(r'model_name = "(\w+)"')
VERSION = re.compile(r"(PyTorch|Transformers) \S+: (\S+)")
DEVICE = re.compile(r"设备: (\w+)")  # the notebook printed "使用设备: mps"


def hms_to_seconds(text: str) -> int:
    """Convert ``H:MM:SS`` or ``MM:SS`` to seconds."""
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def cell_text(cell: dict) -> str:
    """Concatenate the stream and HTML/plain-text outputs of a notebook cell."""
    chunks = []
    for output in cell.get("outputs", []):
        chunks.append("".join(output.get("text", "")))
        data = output.get("data", {})
        chunks.append("".join(data.get("text/html", "")))
        chunks.append("".join(data.get("text/plain", "")))
    return "\n".join(chunks)


def parse_notebook(notebook: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``({display_name: runtime "H:MM:SS"}, {library or "device": value})``."""
    runtimes, versions = {}, {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = cell_text(cell)
        versions.update(dict(VERSION.findall(text)))
        device = DEVICE.search(text)
        if device:
            versions["device"] = device.group(1)
        name = MODEL_NAME.search("".join(cell["source"]))
        progress = PROGRESS.search(text)
        if name and progress:
            done, total = int(progress.group(1)), int(progress.group(2))
            if done != total:
                raise ValueError(f"{name.group(1)} training did not finish ({done}/{total} steps)")
            runtimes[name.group(1)] = progress.group(3)
    return runtimes, versions


def checkpoint_times(model_dir: Path) -> dict[str, str]:
    """Return when each checkpoint's ``trainer_state.json`` was last modified.

    These file times are the only record of how long each epoch took. They
    survive a ``mv`` within one disk but not a copy, so record them early.
    """
    times = {}
    for state in sorted(model_dir.glob("checkpoint-*/trainer_state.json"),
                        key=lambda p: int(p.parent.name.rsplit("-", 1)[1])):
        stamp = datetime.fromtimestamp(state.stat().st_mtime).astimezone()
        times[state.parent.name] = stamp.isoformat(timespec="seconds")
    return times


def main(argv: list[str] | None = None) -> int:
    """Write ``run_info.json`` for each model trained by the original notebook."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    setup_logging()

    config = load_config(args.config)
    notebook = json.loads(args.notebook.read_text())
    runtimes, versions = parse_notebook(notebook)
    python_version = notebook.get("metadata", {}).get("language_info", {}).get("version")

    for spec in config.models.values():
        if spec.display_name not in runtimes:
            raise SystemExit(f"no training progress bar for {spec.display_name} in {args.notebook}")
        model_dir = args.models_dir / spec.key
        if not model_dir.is_dir():
            raise SystemExit(f"{model_dir} does not exist; move the original checkpoints there first")
        run_info = {
            "model": spec.key,
            "trained_by": "the original notebook sst2_project.ipynb (git tag 'original'), "
                          "before this pipeline existed",
            "hub_id": spec.hub_id,
            "revision": spec.revision,
            "smoke": False,
            "train_rows": config.data.splits["train"].rows,
            "validation_rows": config.data.splits["validation"].rows,
            "seed": 42,
            "device": versions.get("device"),
            "train_runtime_seconds": hms_to_seconds(runtimes[spec.display_name]),
            "train_runtime_hms": runtimes[spec.display_name],
            "checkpoint_saved_at": checkpoint_times(model_dir),
            "versions": {
                "python": python_version,
                "torch": versions.get("PyTorch"),
                "transformers": versions.get("Transformers"),
            },
            "sources": {
                "revision": "the snapshot in the local Hugging Face cache that the notebook loaded "
                            "(the notebook itself did not pin one)",
                "seed": "Trainer default; applied after the classification head was initialised, "
                        "so the head's initial weights were not seeded",
                "device": "printed by the notebook's environment-check cell",
                "train_runtime_seconds": f"Trainer progress bar in the output of the "
                                         f"{spec.display_name} training cell (wall clock, "
                                         "including any idle time such as the computer sleeping)",
                "checkpoint_saved_at": "modification times of checkpoint-*/trainer_state.json "
                                       "on the author's machine",
                "versions": "notebook metadata and the environment-check cell",
            },
        }
        path = model_dir / RUN_INFO
        path.write_text(json.dumps(run_info, indent=2) + "\n")
        log.info("wrote %s (training took %s)", path, runtimes[spec.display_name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
