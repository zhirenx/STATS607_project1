"""Export the training log and validation predictions of a fine-tuned model.

The exported files are small (tens of kilobytes) and are committed to the
repository. They are the shareable checkpoint of the analysis: every table
and figure is built from them, so results can be regenerated without the
multi-gigabyte model weights or hours of GPU time.

For the run in ``<models-dir>/<model>/`` this writes, under ``<out>``:

* ``training_logs/<model>/trainer_state.json``: verbatim copy of the
  Trainer's log from the final checkpoint (every logged loss and every
  per-epoch validation score, plus the best checkpoint);
* ``training_logs/<model>/run_info.json``: verbatim copy of the run's
  provenance record (device, versions, recorded training time);
* ``predictions/<model>_validation.csv``: logits, positive-class
  probability and predicted label of the best checkpoint for every
  validation sentence, keyed by the SST-2 ``idx`` (no sentence text);
* ``predictions/<model>_validation.json``: how the predictions were made.

Usage::

    python -m src.pipeline.export --model roberta
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.metrics import softmax_positive
from src.config import ARTIFACTS_DIR, DEFAULT_CONFIG, MODELS_DIR, RUN_INFO, Config, load_config
from src.data import load_split
from src.log import setup_logging
from src.pipeline.download_data import sha256sum
from src.pipeline.runtime import pick_device, require_training_libraries

log = logging.getLogger(__name__)

PREDICTION_COLUMNS = ["idx", "label", "pred", "prob_positive", "logit_negative", "logit_positive"]


def checkpoint_step(path: Path) -> int:
    """Return the global step encoded in a ``checkpoint-<step>`` directory name."""
    return int(path.name.rsplit("-", 1)[1])


def final_checkpoint(model_dir: Path) -> Path:
    """Return the checkpoint with the highest step; its log covers the whole run."""
    checkpoints = sorted(model_dir.glob("checkpoint-*"), key=checkpoint_step)
    if not checkpoints:
        raise SystemExit(
            f"No checkpoints in {model_dir}. Fine-tune the model first (make rebuild), "
            "or build the results from the committed artifacts (make reproduce)."
        )
    return checkpoints[-1]


def best_checkpoint(model_dir: Path, trainer_state: dict) -> Path:
    """Return the checkpoint the Trainer recorded as best.

    The step is taken from ``best_global_step``; the stored
    ``best_model_checkpoint`` path is relative to wherever training ran.
    """
    step = trainer_state.get("best_global_step")
    if step is None:
        step = checkpoint_step(Path(trainer_state["best_model_checkpoint"]))
    path = model_dir / f"checkpoint-{step}"
    if not path.is_dir():
        raise SystemExit(f"best checkpoint {path} is missing")
    return path


def predict(checkpoint: Path, tokenizer_source: tuple[str, str | None], sentences: list[str],
            max_length: int, batch_size: int, device: str) -> tuple[np.ndarray, int]:
    """Score ``sentences`` with ``checkpoint``; return ``(logits, n_parameters)``.

    ``logits`` has shape ``(n, 2)``. Sentences are padded to ``max_length``
    as during training, and the model is loaded with ``from_pretrained``,
    which (unlike the Trainer's best-model reload in transformers 5.5.3)
    maps every saved weight.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name, revision = tokenizer_source
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, output_loading_info=True
    )
    # A partly loaded model is exactly what corrupted the original report's
    # confusion matrix, so refuse to score with one.
    problems = {key: sorted(value) for key, value in loading.items() if value}
    if problems:
        raise SystemExit(f"{checkpoint} did not load cleanly: {problems}")
    model = model.to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            encoded = tokenizer(
                sentences[start:start + batch_size],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            batches.append(model(**encoded).logits.float().cpu().numpy())
    return np.concatenate(batches), sum(p.numel() for p in model.parameters())


def export_model(config: Config, model_key: str, models_dir: Path = MODELS_DIR,
                 out_dir: Path = ARTIFACTS_DIR) -> Path:
    """Export the run of ``model_key`` into ``out_dir`` and return the predictions path."""
    require_training_libraries()
    import torch
    import transformers

    spec = config.model(model_key)
    model_dir = models_dir / model_key
    final = final_checkpoint(model_dir)
    state = json.loads((final / "trainer_state.json").read_text())
    best = best_checkpoint(model_dir, state)
    run_info_path = model_dir / RUN_INFO
    if not run_info_path.exists():
        raise SystemExit(f"{run_info_path} is missing, so the run is incomplete or unrecorded")
    run_info = json.loads(run_info_path.read_text())

    validation = load_split("validation").head(run_info["validation_rows"])
    # Checkpoints written by src/pipeline/train.py include the tokenizer; the
    # original notebook's checkpoints do not, so fall back to the pinned Hub copy.
    if (best / "tokenizer_config.json").exists():
        tokenizer_source, tokenizer_note = (str(best), None), "saved with the checkpoint"
    else:
        tokenizer_source = (spec.hub_id, spec.revision)
        tokenizer_note = f"{spec.hub_id}@{spec.revision}"

    device = pick_device()
    log.info("scoring %d validation sentences with %s/%s on %s",
             len(validation), model_key, best.name, device)
    logits, n_params = predict(
        best, tokenizer_source, validation["sentence"].tolist(),
        config.training.max_length, config.training.per_device_eval_batch_size, device,
    )
    predictions = pd.DataFrame({
        "idx": validation["idx"].to_numpy(),
        "label": validation["label"].to_numpy(),
        "pred": logits.argmax(axis=1),
        "prob_positive": softmax_positive(logits),
        "logit_negative": logits[:, 0],
        "logit_positive": logits[:, 1],
    })[PREDICTION_COLUMNS]
    accuracy = float((predictions["pred"] == predictions["label"]).mean())

    logs_dir = out_dir / "training_logs" / model_key
    predictions_dir = out_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final / "trainer_state.json", logs_dir / "trainer_state.json")
    shutil.copyfile(run_info_path, logs_dir / RUN_INFO)
    predictions_path = predictions_dir / f"{model_key}_validation.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.6f")
    export_info = {
        "model": model_key,
        "checkpoint": best.name,
        "checkpoint_sha256": sha256sum(best / "model.safetensors"),
        "num_parameters": int(n_params),
        "tokenizer": tokenizer_note,
        "max_length": config.training.max_length,
        "padding": "max_length",
        "batch_size": config.training.per_device_eval_batch_size,
        "device": device,
        "rows": len(predictions),
        "accuracy": accuracy,
        "trainer_logged_best_metric": state["best_metric"],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    predictions_path.with_suffix(".json").write_text(json.dumps(export_info, indent=2) + "\n")
    log.info("wrote %s (accuracy %.4f; the Trainer logged %.4f for %s)",
             predictions_path, accuracy, state["best_metric"], best.name)
    return predictions_path


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and export the requested model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model key from config.toml, e.g. bert")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="directory holding <model>/checkpoint-* (default: artifacts/models)")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR,
                        help="artifact root to write training_logs/ and predictions/ into")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    if args.model not in config.models:
        parser.error(f"--model must be one of {list(config.models)}")
    export_model(config, args.model, args.models_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
