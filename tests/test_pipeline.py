"""Pipeline integrity: the analysis turns the artifacts into the expected outputs."""

import shutil
import subprocess
import sys

import pandas as pd
import pytest

from src.analysis import figures, report, tables
from src.config import ROOT


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, splits):
    """Run every table, figure and report entry point into a temporary directory."""
    out = tmp_path_factory.mktemp("results")
    for name in tables.TABLES:
        assert tables.main(["--table", name, "--out", str(out / "tables" / f"{name}.csv")]) == 0
    for name in figures.FIGURES:
        assert figures.main(["--figure", name, "--out", str(out / "figures" / f"{name}.png")]) == 0
    assert report.main(["--tables-dir", str(out / "tables"), "--figures-dir", str(out / "figures"),
                        "--out", str(out / "report" / "summary.md")]) == 0
    return out


def test_every_output_is_written(outputs):
    for name in tables.TABLES:
        assert (outputs / "tables" / f"{name}.csv").stat().st_size > 0
    for name in figures.FIGURES:
        with open(outputs / "figures" / f"{name}.png", "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    assert "## Main findings" in (outputs / "report" / "summary.md").read_text()


def test_model_comparison_table(outputs, runs):
    table = pd.read_csv(outputs / "tables" / "model_comparison.csv")
    assert table["model"].tolist() == ["BERT", "DistilBERT", "RoBERTa"]
    assert table["best_epoch"].tolist() == [1, 3, 2]
    assert table["correct"].tolist() == [812, 791, 817]
    for key, row in zip(["bert", "distilbert", "roberta"], table.itertuples()):
        assert row.val_accuracy == round(runs[key].trainer_state["best_metric"], 4)
        assert row.val_accuracy_ci95_low < row.val_accuracy < row.val_accuracy_ci95_high


def test_per_epoch_table_marks_one_best_epoch_per_model(outputs):
    table = pd.read_csv(outputs / "tables" / "per_epoch_validation.csv")
    assert len(table) == 9
    assert table.groupby("model")["is_best"].sum().tolist() == [1, 1, 1]


def test_original_report_comparison_flags_the_known_discrepancies(outputs):
    table = pd.read_csv(outputs / "tables" / "original_report_comparison.csv")
    mismatched = set(zip(table.loc[~table["matches"], "model"], table.loc[~table["matches"], "quantity"]))
    assert ("RoBERTa", "confusion_tn") in mismatched
    assert ("BERT", "epoch1_val_accuracy_pct") in mismatched
    assert not any(model == "DistilBERT" for model, _ in mismatched)


def test_tables_are_deterministic(tmp_path, splits):
    for attempt in ("a", "b"):
        tables.main(["--table", "pairwise_comparison", "--out", str(tmp_path / f"{attempt}.csv")])
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_analysis_does_not_need_pytorch():
    code = ("import sys, src.analysis.tables, src.analysis.figures, src.analysis.report; "
            "sys.exit(any(m in sys.modules for m in ('torch', 'transformers')))")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_makefile_never_rebuilds_artifacts_or_deletes_checkpoints():
    def dry_run(target):
        return subprocess.run(["make", "-n", target], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout

    assert "src.pipeline" not in dry_run("reproduce").replace("src.pipeline.download_data", "")
    clean = dry_run("clean")
    for protected in ("artifacts/models", "artifacts/rebuild", "training_logs", "predictions"):
        assert protected not in clean
