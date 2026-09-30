"""Build the result tables from the exported artifacts and the raw data.

Each table is produced by one ``build_*`` function and written as CSV:

* ``dataset_summary``: size, class balance and sentence length of each split
* ``model_comparison``: best validation accuracy (with 95% CI), parameters,
  training compute, and the recorded (not reproducible) training time
* ``per_epoch_validation``: validation accuracy and loss after every epoch
* ``classification_metrics``: confusion counts, precision, recall and F1
* ``pairwise_comparison``: McNemar's exact test for each pair of models
* ``misclassified_examples``: validation sentences the best model gets wrong
* ``original_report_comparison``: numbers printed in the original report
  next to the values reproduced here

Usage::

    python -m src.analysis.tables --table model_comparison --out results/tables/model_comparison.csv
    python -m src.analysis.tables --all --out-dir results/tables
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import metrics
from src.analysis.artifacts import ModelRun, load_runs
from src.config import ARTIFACTS_DIR, DEFAULT_CONFIG, ROOT, Config, load_config
from src.data import LABEL_NAMES, load_split
from src.log import setup_logging

log = logging.getLogger(__name__)

ORIGINAL_REPORT_VALUES = ROOT / "data" / "reference" / "original_report_values.csv"


def build_dataset_summary(config: Config) -> pd.DataFrame:
    """Summarise each labelled split: rows, label counts and sentence length in words."""
    rows = []
    for split in config.data.splits:
        frame = load_split(split)
        words = frame["sentence"].str.split().str.len()
        counts = frame["label"].value_counts()
        rows.append({
            "split": split,
            "rows": len(frame),
            "negative": int(counts.get(0, 0)),
            "positive": int(counts.get(1, 0)),
            "positive_share": round(float(counts.get(1, 0)) / len(frame), 4),
            "mean_words": round(float(words.mean()), 2),
            "p95_words": round(float(np.percentile(words, 95)), 1),
            "max_words": int(words.max()),
        })
    return pd.DataFrame(rows)


def best_row(run: ModelRun) -> pd.Series:
    """Return the evaluation row of the checkpoint the Trainer recorded as best."""
    history = metrics.eval_history(run.log_history)
    step = run.trainer_state["best_global_step"]
    best = history.loc[history["step"] == step]
    if len(best) != 1:
        raise ValueError(f"{run.key}: best step {step} is not an evaluation step in the log")
    return best.iloc[0]


def build_model_comparison(runs: list[ModelRun]) -> pd.DataFrame:
    """One row per model: best epoch, accuracy with 95% CI, size, compute and recorded time."""
    rows = []
    for run in runs:
        best = best_row(run)
        preds = run.predictions
        n, correct = len(preds), int((preds["pred"] == preds["label"]).sum())
        low, high = metrics.wilson_interval(correct, n)
        runtime = run.run_info.get("train_runtime_seconds")
        saved_at = run.run_info.get("checkpoint_saved_at")
        per_epoch = (
            " / ".join(map(str, metrics.epoch_minutes(saved_at, runtime)))
            if saved_at and runtime is not None else None
        )
        rows.append({
            "model": run.display_name,
            "parameters_millions": round(run.export_info["num_parameters"] / 1e6, 1),
            "best_epoch": int(round(best["epoch"])),
            "best_checkpoint": run.export_info["checkpoint"],
            "correct": correct,
            "n_validation": n,
            "val_accuracy": round(correct / n, 4),
            "val_accuracy_ci95_low": round(low, 4),
            "val_accuracy_ci95_high": round(high, 4),
            "val_loss_at_best": round(float(best["eval_loss"]), 4),
            "train_pflops": round(run.trainer_state["total_flos"] / 1e15, 2),
            "train_time_hours_recorded": None if runtime is None else round(runtime / 3600, 2),
            "epoch_minutes_recorded": per_epoch,
        })
    return pd.DataFrame(rows)


def build_per_epoch_validation(runs: list[ModelRun]) -> pd.DataFrame:
    """Validation accuracy and loss after every epoch, flagging the best checkpoint."""
    frames = []
    for run in runs:
        history = metrics.eval_history(run.log_history)
        history.insert(0, "model", run.display_name)
        history["epoch"] = history["epoch"].round().astype(int)
        history["is_best"] = history["step"] == run.trainer_state["best_global_step"]
        frames.append(history)
    table = pd.concat(frames, ignore_index=True)
    return table.round({"eval_accuracy": 4, "eval_loss": 4}).rename(
        columns={"eval_accuracy": "val_accuracy", "eval_loss": "val_loss"}
    )


def build_classification_metrics(runs: list[ModelRun]) -> pd.DataFrame:
    """Confusion counts plus accuracy, precision, recall and F1 (positive class)."""
    rows = []
    for run in runs:
        preds = run.predictions
        counts = metrics.confusion_counts(preds["label"], preds["pred"])
        prf = metrics.precision_recall_f1(counts)
        rows.append({
            "model": run.display_name,
            "accuracy": round(metrics.accuracy(preds["label"], preds["pred"]), 4),
            **{key: round(value, 4) for key, value in prf.items()},
            **counts,
        })
    return pd.DataFrame(rows)


def build_pairwise_comparison(runs: list[ModelRun]) -> pd.DataFrame:
    """McNemar's exact test of equal accuracy for every pair of models.

    The models are scored on the same 872 sentences, so the comparison is
    paired: only sentences exactly one model gets right carry information.
    """
    rows = []
    for a, b in itertools.combinations(runs, 2):
        merged = a.predictions.merge(b.predictions, on=["idx", "label"], suffixes=("_a", "_b"))
        correct_a = merged["pred_a"] == merged["label"]
        correct_b = merged["pred_b"] == merged["label"]
        test = metrics.mcnemar_exact(correct_a, correct_b)
        rows.append({
            "model_a": a.display_name,
            "model_b": b.display_name,
            "accuracy_a": round(float(correct_a.mean()), 4),
            "accuracy_b": round(float(correct_b.mean()), 4),
            "difference_points": round(100 * float(correct_a.mean() - correct_b.mean()), 2),
            "only_a_correct": test["only_a"],
            "only_b_correct": test["only_b"],
            "mcnemar_exact_p": round(test["p_value"], 4),
        })
    return pd.DataFrame(rows)


def build_misclassified_examples(runs: list[ModelRun]) -> pd.DataFrame:
    """Validation sentences the most accurate model gets wrong, most confident first."""
    best = max(runs, key=lambda run: (run.predictions["pred"] == run.predictions["label"]).mean())
    validation = load_split("validation")[["idx", "sentence"]]
    wrong = best.predictions.loc[best.predictions["pred"] != best.predictions["label"]]
    table = wrong.merge(validation, on="idx", how="left")
    table["confidence"] = np.where(table["pred"] == 1, table["prob_positive"], 1 - table["prob_positive"])
    table.insert(0, "model", best.display_name)
    table["label"] = table["label"].map(LABEL_NAMES)
    table["pred"] = table["pred"].map(LABEL_NAMES)
    columns = ["model", "idx", "sentence", "label", "pred", "confidence"]
    return table.sort_values(["confidence", "idx"], ascending=[False, True])[columns].round(
        {"confidence": 4}
    ).reset_index(drop=True)


def reproduced_values(runs: list[ModelRun]) -> dict[tuple[str, str], float]:
    """Map ``(model, quantity)`` to the value reproduced from the artifacts."""
    values = {}
    comparison = build_model_comparison(runs).set_index("model")
    per_epoch = build_per_epoch_validation(runs)
    for run in runs:
        name = run.display_name
        row = comparison.loc[name]
        values[(name, "best_val_accuracy_pct")] = round(100 * row["val_accuracy"], 1)
        values[(name, "best_epoch")] = row["best_epoch"]
        values[(name, "parameters_millions")] = round(row["parameters_millions"])
        runtime = run.run_info.get("train_runtime_seconds")
        values[(name, "train_time_minutes")] = np.nan if runtime is None else round(runtime / 60)
        for _, epoch_row in per_epoch.loc[per_epoch["model"] == name].iterrows():
            epoch = epoch_row["epoch"]
            values[(name, f"epoch{epoch}_val_accuracy_pct")] = round(100 * epoch_row["val_accuracy"], 1)
            values[(name, f"epoch{epoch}_val_loss")] = round(epoch_row["val_loss"], 3)
        counts = metrics.confusion_counts(
            run.predictions["label"], run.predictions["pred"]
        )
        for key, value in counts.items():
            values[(name, f"confusion_{key}")] = value
    return values


def build_original_report_comparison(runs: list[ModelRun],
                                     reference: Path = ORIGINAL_REPORT_VALUES) -> pd.DataFrame:
    """Put each number printed in the original report next to its reproduced value."""
    report = pd.read_csv(reference)
    values = reproduced_values(runs)
    report["reproduced"] = [
        values.get((row.model, row.quantity), np.nan) for row in report.itertuples()
    ]
    report["difference"] = (report["reproduced"] - report["original_report"]).round(3)
    report["matches"] = np.isclose(report["original_report"], report["reproduced"])
    return report[["model", "quantity", "location", "original_report", "reproduced",
                   "difference", "matches"]]


def build_table(name: str, config: Config, runs: list[ModelRun]) -> pd.DataFrame:
    """Dispatch to the ``build_<name>`` function."""
    if name == "dataset_summary":
        return build_dataset_summary(config)
    builders = {
        "model_comparison": build_model_comparison,
        "per_epoch_validation": build_per_epoch_validation,
        "classification_metrics": build_classification_metrics,
        "pairwise_comparison": build_pairwise_comparison,
        "misclassified_examples": build_misclassified_examples,
        "original_report_comparison": build_original_report_comparison,
    }
    if name not in builders:
        raise KeyError(f"unknown table {name!r}; choose from {['dataset_summary', *builders]}")
    return builders[name](runs)


TABLES = [
    "dataset_summary",
    "model_comparison",
    "per_epoch_validation",
    "classification_metrics",
    "pairwise_comparison",
    "misclassified_examples",
    "original_report_comparison",
]


def main(argv: list[str] | None = None) -> int:
    """Build one table (``--table``) or all of them (``--all``) and write CSV files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--table", choices=TABLES)
    which.add_argument("--all", action="store_true", help="build every table into --out-dir")
    parser.add_argument("--out", type=Path, help="output file (with --table)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"),
                        help="output directory (with --all; default: results/tables)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR,
                        help="directory holding training_logs/ and predictions/")
    parser.add_argument("--models", nargs="+", help="restrict to these model keys")
    args = parser.parse_args(argv)
    setup_logging()

    if args.table and args.out is None:
        parser.error("--table needs --out")
    config = load_config(args.config)
    if args.models:
        config = config.restrict(args.models)
    names = TABLES if args.all else [args.table]
    runs = [] if names == ["dataset_summary"] else load_runs(config, args.artifacts)
    for name in names:
        out = args.out_dir / f"{name}.csv" if args.all else args.out
        table = build_table(name, config, runs)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
