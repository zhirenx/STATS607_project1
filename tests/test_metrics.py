"""Function correctness: the scoring helpers on small hand-checked inputs."""

import math

import numpy as np
import pytest

from src.analysis import metrics

Y_TRUE = [0, 0, 0, 1, 1, 1, 1, 0]
Y_PRED = [0, 1, 0, 1, 0, 1, 1, 0]  # one false positive, one false negative


def test_accuracy_and_confusion_counts():
    assert metrics.accuracy(Y_TRUE, Y_PRED) == 0.75
    assert metrics.confusion_counts(Y_TRUE, Y_PRED) == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}


def test_accuracy_rejects_mismatched_input():
    with pytest.raises(ValueError):
        metrics.accuracy([0, 1], [0])
    with pytest.raises(ValueError):
        metrics.accuracy([], [])


def test_precision_recall_f1():
    result = metrics.precision_recall_f1({"tn": 3, "fp": 1, "fn": 3, "tp": 3})
    assert result["precision"] == 0.75
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(0.6)


def test_precision_undefined_without_positive_predictions():
    result = metrics.precision_recall_f1({"tn": 5, "fp": 0, "fn": 2, "tp": 0})
    assert math.isnan(result["precision"]) and math.isnan(result["f1"])
    assert result["recall"] == 0.0


def test_softmax_positive_is_stable_and_correct():
    logits = np.array([[0.0, 0.0], [1000.0, 1001.0], [2.0, -1.0]])
    probs = metrics.softmax_positive(logits)
    assert probs[0] == 0.5
    assert probs[1] == pytest.approx(1 / (1 + math.exp(-1)))
    assert probs[2] == pytest.approx(math.exp(-1) / (math.exp(2) + math.exp(-1)))


LOG = [  # shaped like a Trainer log_history, deliberately out of order
    {"epoch": 2.0, "step": 20, "eval_accuracy": 0.90, "eval_loss": 0.30},
    {"epoch": 0.5, "step": 5, "loss": 0.69, "learning_rate": 1e-5},
    {"epoch": 1.0, "step": 10, "eval_accuracy": 0.90, "eval_loss": 0.35},
    {"epoch": 1.5, "step": 15, "loss": 0.40, "learning_rate": 5e-6},
    {"epoch": 3.0, "step": 30, "eval_accuracy": 0.88, "eval_loss": 0.40},
]


def test_log_history_is_split_into_training_and_evaluation():
    evals = metrics.eval_history(LOG)
    assert evals["step"].tolist() == [10, 20, 30]
    assert evals["eval_accuracy"].tolist() == [0.90, 0.90, 0.88]
    assert metrics.train_loss_history(LOG)["loss"].tolist() == [0.69, 0.40]


def test_best_evaluation_keeps_earliest_epoch_on_a_tie():
    # The Trainer only replaces its best checkpoint when accuracy is strictly
    # greater, which is why BERT's best checkpoint is epoch 1, not epoch 3.
    assert metrics.best_evaluation(metrics.eval_history(LOG))["step"] == 10


def test_wilson_interval_known_values():
    low, high = metrics.wilson_interval(50, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    low, high = metrics.wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)


def test_mcnemar_exact_known_values():
    assert metrics.mcnemar_exact_p(5, 0) == 0.0625  # 2 * (1/2)**5
    assert metrics.mcnemar_exact_p(1, 9) == pytest.approx(2 * 11 / 1024)
    assert metrics.mcnemar_exact_p(9, 1) == metrics.mcnemar_exact_p(1, 9)
    assert metrics.mcnemar_exact_p(0, 0) == 1.0
    assert metrics.mcnemar_exact_p(10, 10) == 1.0


def test_mcnemar_exact_counts_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    result = metrics.mcnemar_exact(a, b)
    assert (result["only_a"], result["only_b"]) == (2, 1)
    assert result["p_value"] == 1.0


def test_epoch_minutes_from_checkpoint_times():
    saved = {
        "checkpoint-20": "2026-04-20T10:30:00-04:00",
        "checkpoint-10": "2026-04-20T10:10:00-04:00",
        "checkpoint-30": "2026-04-20T11:30:00-04:00",
    }
    # The run took 100 minutes and ended at 11:30, so it started at 09:50.
    minutes = metrics.epoch_minutes(saved, total_seconds=100 * 60)
    assert minutes == [20, 20, 60]
    assert sum(minutes) == 100
