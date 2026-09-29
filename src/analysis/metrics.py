"""Pure functions for scoring predictions and summarising training logs.

Nothing here reads or writes files, so every function can be tested on
small hand-made inputs.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Return the fraction of positions where ``y_pred`` equals ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    return float(np.mean(y_true == y_pred))


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, int]:
    """Return true/false negative/positive counts for binary labels 0 and 1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    """Return precision, recall and F1 of the positive class from confusion counts.

    A ratio whose denominator is zero is reported as ``nan``.
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion (95% by default).

    Unlike the normal-approximation interval, it stays inside [0, 1] and
    behaves well for proportions near 0 or 1, such as accuracies above 90%.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n > 0")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    """Return P(label = 1) from an ``(n, 2)`` array of logits, computed stably."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=1)


def eval_history(log_history: list[dict]) -> pd.DataFrame:
    """Extract the per-epoch validation metrics from a Trainer ``log_history``.

    Returns one row per evaluation with columns ``epoch``, ``step``,
    ``eval_accuracy`` and ``eval_loss``, sorted by step.
    """
    rows = [
        {
            "epoch": entry["epoch"],
            "step": entry["step"],
            "eval_accuracy": entry["eval_accuracy"],
            "eval_loss": entry["eval_loss"],
        }
        for entry in log_history
        if "eval_accuracy" in entry
    ]
    if not rows:
        raise ValueError("log_history contains no evaluation entries")
    return pd.DataFrame(rows).sort_values("step", ignore_index=True)


def train_loss_history(log_history: list[dict]) -> pd.DataFrame:
    """Extract the logged training loss (``step``, ``epoch``, ``loss``) from ``log_history``."""
    rows = [
        {"step": entry["step"], "epoch": entry["epoch"], "loss": entry["loss"]}
        for entry in log_history
        if "loss" in entry
    ]
    return pd.DataFrame(rows, columns=["step", "epoch", "loss"]).sort_values(
        "step", ignore_index=True
    )


def best_evaluation(history: pd.DataFrame) -> pd.Series:
    """Return the evaluation row the Trainer would record as best.

    The Hugging Face Trainer replaces its best checkpoint only when the
    metric is strictly greater, so on ties the earliest evaluation wins.
    """
    if history.empty:
        raise ValueError("history is empty")
    ordered = history.sort_values("step", ignore_index=True)
    return ordered.loc[ordered["eval_accuracy"].idxmax()]


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Return the two-sided exact McNemar p-value for two discordant counts.

    ``only_a`` and ``only_b`` count the examples that only classifier A, or
    only classifier B, gets right. Under the null hypothesis of equal
    accuracy, ``only_a`` is Binomial(only_a + only_b, 1/2) given the total;
    the p-value doubles the smaller tail and is capped at 1.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact(correct_a: Iterable[bool], correct_b: Iterable[bool]) -> dict[str, int | float]:
    """Compare two classifiers scored on the same examples with McNemar's exact test.

    Returns the discordant counts ``only_a`` and ``only_b`` and ``p_value``.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("correct_a and correct_b must have the same length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    return {"only_a": only_a, "only_b": only_b, "p_value": mcnemar_exact_p(only_a, only_b)}


def epoch_minutes(checkpoint_saved_at: dict[str, str], total_seconds: float) -> list[int]:
    """Return the wall-clock minutes of each epoch, from checkpoint save times.

    ``checkpoint_saved_at`` maps ``checkpoint-<step>`` to an ISO timestamp.
    Each epoch ends when its checkpoint is saved; the first starts
    ``total_seconds`` before the last checkpoint was saved.
    """
    stamps = [
        datetime.fromisoformat(checkpoint_saved_at[name])
        for name in sorted(checkpoint_saved_at, key=lambda n: int(n.rsplit("-", 1)[1]))
    ]
    edges = [stamps[-1] - timedelta(seconds=total_seconds), *stamps]
    return [round((end - start).total_seconds() / 60) for start, end in zip(edges, edges[1:])]
