"""Fine-tune one pre-trained model on SST-2 with the settings in config.toml.

This replaces the three copy-pasted training cells of the original notebook.
A run writes one checkpoint per epoch to ``<models-dir>/<model>/`` and
finishes by writing ``run_info.json`` there. A model whose directory already
holds a finished run is skipped, and leftovers of an unfinished run stop the
script, so existing checkpoints are never overwritten.

Usage::

    python -m src.pipeline.train --model distilbert --models-dir artifacts/rebuild/models
    python -m src.pipeline.train --model distilbert --smoke   # a few minutes
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.metrics import accuracy
from src.config import ARTIFACTS_DIR, DEFAULT_CONFIG, RUN_INFO, Config, load_config
from src.data import load_split
from src.log import setup_logging
from src.pipeline.runtime import pick_device, require_training_libraries

log = logging.getLogger(__name__)

REBUILD_MODELS_DIR = ARTIFACTS_DIR / "rebuild" / "models"
SMOKE_MODELS_DIR = ARTIFACTS_DIR / "smoke" / "models"


def encode(frame, tokenizer, max_length: int):
    """Tokenize a split as the original notebook did (pad every sentence to ``max_length``).

    Returns a map-style PyTorch dataset of dicts with ``input_ids``,
    ``attention_mask`` (and ``token_type_ids`` for BERT) and ``labels``.
    """
    import torch

    encodings = dict(tokenizer(
        frame["sentence"].tolist(), padding="max_length", truncation=True, max_length=max_length,
    ))
    labels = frame["label"].astype(int).tolist()

    class EncodedSentences(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, i: int) -> dict:
            item = {key: torch.tensor(values[i]) for key, values in encodings.items()}
            item["labels"] = torch.tensor(labels[i])
            return item

    return EncodedSentences()


def compute_metrics(eval_pred) -> dict[str, float]:
    """Return validation accuracy for the Trainer's evaluation loop."""
    logits, labels = eval_pred
    return {"accuracy": accuracy(labels, logits.argmax(axis=-1))}


def prepare_output_dir(out_dir: Path, smoke: bool) -> bool:
    """Decide whether to train into ``out_dir``; return False to skip the model.

    A finished run (``run_info.json`` present) is kept and training is
    skipped. Leftovers of an unfinished run stop the script rather than being
    deleted. Smoke-test output is disposable and is replaced.
    """
    if smoke:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        return True
    if (out_dir / RUN_INFO).exists():
        log.info("%s already holds a finished run; skipping it", out_dir)
        return False
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"{out_dir} contains files from an unfinished run. "
            "Move or delete that directory, then run training again."
        )
    return True


def train(config: Config, model_key: str, models_dir: Path, smoke: bool = False) -> Path | None:
    """Fine-tune ``model_key`` into ``models_dir``; return its run directory (None if skipped)."""
    require_training_libraries()
    import torch
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    spec = config.model(model_key)
    out_dir = models_dir / model_key
    if not prepare_output_dir(out_dir, smoke):
        return None

    train_frame = load_split("train")
    validation_frame = load_split("validation")
    epochs = config.training.num_train_epochs
    logging_steps = config.training.logging_steps
    if smoke:
        train_frame = train_frame.sample(n=config.smoke.train_rows, random_state=config.seed)
        validation_frame = validation_frame.head(config.smoke.validation_rows)
        epochs = config.smoke.num_train_epochs
        logging_steps = 4

    device = pick_device()
    log.info(
        "fine-tuning %s (%s@%s) on %d sentences for %d epoch(s) on %s",
        spec.display_name, spec.hub_id, spec.revision[:7], len(train_frame), epochs, device,
    )

    # Seed before the classification head is created so that its initial
    # weights are reproducible; the original notebook seeded only inside the
    # Trainer, after the head already existed.
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(spec.hub_id, revision=spec.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hub_id, revision=spec.revision, num_labels=2
    )
    log.info("%s has %s parameters", spec.display_name,
             f"{sum(p.numel() for p in model.parameters()):,}")

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        # The Trainer still records the best checkpoint in trainer_state.json;
        # the export step loads it from disk. Reloading it here is what left
        # the original notebook with a partly loaded model (docs/discrepancies.md).
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_only_model=True,  # skip optimizer state, about two thirds of each checkpoint
        logging_steps=logging_steps,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_pin_memory=device == "cuda",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encode(train_frame, tokenizer, config.training.max_length),
        eval_dataset=encode(validation_frame, tokenizer, config.training.max_length),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    result = trainer.train()
    runtime = result.metrics["train_runtime"]
    log.info("%s finished in %.1f min; best validation accuracy %.4f",
             spec.display_name, runtime / 60, trainer.state.best_metric)

    run_info = {
        "model": model_key,
        "trained_by": "src/pipeline/train.py" + (" --smoke" if smoke else ""),
        "hub_id": spec.hub_id,
        "revision": spec.revision,
        "smoke": smoke,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "seed": config.seed,
        "device": device,
        "train_runtime_seconds": round(runtime),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "sources": {"train_runtime_seconds": "train_runtime reported by Trainer.train()"},
    }
    (out_dir / RUN_INFO).write_text(json.dumps(run_info, indent=2) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and fine-tune the requested model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model key from config.toml, e.g. bert")
    parser.add_argument("--models-dir", type=Path,
                        help="where to write <model>/checkpoint-* "
                             "(default: artifacts/rebuild/models, or artifacts/smoke/models)")
    parser.add_argument("--smoke", action="store_true", help="short run on a data subset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    if args.model not in config.models:
        parser.error(f"--model must be one of {list(config.models)}")
    models_dir = args.models_dir or (SMOKE_MODELS_DIR if args.smoke else REBUILD_MODELS_DIR)
    train(config, args.model, models_dir, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
