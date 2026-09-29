"""Integrity of the committed artifacts, the starting point of `make reproduce`.

The predictions and the training logs were produced separately (the logs
during training, the predictions by re-scoring the saved checkpoint), so
agreement between them shows that the exported model is the one that was
evaluated. The original report failed exactly this check: its confusion
matrix came from a partly reloaded model and sums to 819 correct, while the
logged best accuracy is 817/872.
"""

import numpy as np
import pytest

from src.analysis import metrics

MODELS = ["bert", "distilbert", "roberta"]


@pytest.mark.parametrize("key", MODELS)
def test_predictions_are_well_formed(runs, key):
    preds = runs[key].predictions
    assert len(preds) == 872
    assert preds["idx"].is_unique
    assert set(preds["pred"]) <= {0, 1} and set(preds["label"]) <= {0, 1}
    assert preds["prob_positive"].between(0, 1).all()
    logits = preds[["logit_negative", "logit_positive"]].to_numpy()
    assert (preds["pred"] == logits.argmax(axis=1)).all()
    assert np.allclose(preds["prob_positive"], metrics.softmax_positive(logits), atol=1e-5)


@pytest.mark.parametrize("key", MODELS)
def test_predictions_reproduce_the_logged_best_accuracy(runs, key):
    run = runs[key]
    preds = run.predictions
    correct = int((preds["pred"] == preds["label"]).sum())
    assert correct == round(run.trainer_state["best_metric"] * len(preds))


@pytest.mark.parametrize("key", MODELS)
def test_exported_checkpoint_is_the_trainers_best(runs, key):
    run = runs[key]
    history = metrics.eval_history(run.log_history)
    best = metrics.best_evaluation(history)
    assert best["step"] == run.trainer_state["best_global_step"]
    assert run.export_info["checkpoint"] == f"checkpoint-{int(best['step'])}"


@pytest.mark.parametrize("key", MODELS)
def test_artifacts_describe_the_configured_model(config, runs, key):
    spec, run = config.model(key), runs[key]
    assert run.run_info["hub_id"] == spec.hub_id
    assert run.run_info["revision"] == spec.revision
    assert run.export_info["model"] == key
    assert run.export_info["rows"] == len(run.predictions)


def test_known_values_of_the_original_run(runs):
    """Guard the numbers the README quotes against accidental changes."""
    correct = {k: int((r.predictions["pred"] == r.predictions["label"]).sum()) for k, r in runs.items()}
    assert correct == {"bert": 812, "distilbert": 791, "roberta": 817}
    best_epochs = {
        k: int(round(metrics.best_evaluation(metrics.eval_history(r.log_history))["epoch"]))
        for k, r in runs.items()
    }
    assert best_epochs == {"bert": 1, "distilbert": 3, "roberta": 2}
    roberta = runs["roberta"].predictions
    assert metrics.confusion_counts(roberta["label"], roberta["pred"]) == {
        "tn": 390, "fp": 38, "fn": 17, "tp": 427,
    }


@pytest.mark.parametrize("key", MODELS)
def test_prediction_labels_match_the_raw_data(runs, splits, key):
    merged = runs[key].predictions.merge(
        splits["validation"], on="idx", how="left", suffixes=("", "_raw")
    )
    assert merged["label_raw"].notna().all()
    assert (merged["label"] == merged["label_raw"]).all()
