# Reproduce "Comparative Analysis of BERT, DistilBERT, and RoBERTa on SST-2".
#
#   make reproduce  Rebuild every table, figure and the report from the
#                   committed artifacts (the default; also `make all`).
#                   Needs requirements.txt only; no GPU; about a minute.
#   make test       Run the test suite.
#   make rebuild    Full rebuild from the raw data: fine-tune all three models
#                   (roughly 7-10 hours on an Apple Silicon Mac), export them to
#                   artifacts/rebuild/, then build results/rebuild/.
#                   Needs requirements-train.txt.
#   make smoke      Fine-tune DistilBERT briefly on a small subset to check the
#                   training and export code (a few minutes).
#   make artifacts  Re-export the committed artifacts from the original
#                   checkpoints in artifacts/models/ (author's machine only).
#   make clean      Delete generated results. Never deletes checkpoints or
#                   committed artifacts.
#
# The virtual environment in .venv is used when it exists; otherwise python3.
# Override with `make PYTHON=/path/to/python`.

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export MPLBACKEND := Agg

MODELS := bert distilbert roberta
SMOKE_MODEL := distilbert

# Artifacts to build results from, and where to write the results.
# `make rebuild` points these at its own directories.
ART := artifacts
RES := results

RAW := data/raw/sst2_train.parquet data/raw/sst2_validation.parquet
ARTIFACTS := $(foreach m,$(MODELS),\
	$(ART)/training_logs/$(m)/trainer_state.json \
	$(ART)/training_logs/$(m)/run_info.json \
	$(ART)/predictions/$(m)_validation.csv \
	$(ART)/predictions/$(m)_validation.json)

# Outputs that need the raw sentences come last, so that without network
# access `make -k reproduce` still builds everything else.
TABLES := $(addprefix $(RES)/tables/,$(addsuffix .csv,\
	model_comparison per_epoch_validation classification_metrics \
	pairwise_comparison original_report_comparison \
	dataset_summary misclassified_examples))
FIGURES := $(addprefix $(RES)/figures/,$(addsuffix .png,\
	model_comparison training_curves confusion_matrices accuracy_vs_size \
	label_distribution sentence_lengths))
REPORT := $(RES)/report/summary.md

ANALYSIS_CODE := config.toml src/config.py src/data.py \
	src/analysis/artifacts.py src/analysis/metrics.py src/analysis/tables.py

.DEFAULT_GOAL := reproduce
.DELETE_ON_ERROR:
.PHONY: all reproduce data test check-env rebuild smoke artifacts clean help

all: reproduce

reproduce: check-env $(TABLES) $(FIGURES) $(REPORT)
	@echo "Done: tables in $(RES)/tables, figures in $(RES)/figures, report $(REPORT)"

data: check-env $(RAW)

# No prerequisites: the script verifies the SHA-256 of an existing file.
data/raw/sst2_%.parquet:
	$(PYTHON) -m src.pipeline.download_data --split $* --out $@

# The committed artifacts are inputs only. No rule rebuilds them from
# checkpoints or code, so a fresh clone never tries to retrain or re-export.
$(ARTIFACTS):
	@echo "error: $@ is missing. Restore the committed artifacts with" \
	      "'git checkout -- artifacts', or regenerate them with 'make artifacts'." >&2
	@exit 1

$(RES)/tables/%.csv: $(ARTIFACTS) $(ANALYSIS_CODE) data/reference/original_report_values.csv
	$(PYTHON) -m src.analysis.tables --table $* --artifacts $(ART) --out $@

$(RES)/figures/%.png: $(ARTIFACTS) $(ANALYSIS_CODE) src/analysis/figures.py
	$(PYTHON) -m src.analysis.figures --figure $* --artifacts $(ART) --out $@

# Only these outputs read the raw sentences.
$(RES)/tables/dataset_summary.csv $(RES)/tables/misclassified_examples.csv: $(RAW)
$(RES)/figures/label_distribution.png $(RES)/figures/sentence_lengths.png: $(RAW)

$(REPORT): $(TABLES) $(FIGURES) src/analysis/report.py
	$(PYTHON) -m src.analysis.report --tables-dir $(RES)/tables \
		--figures-dir $(RES)/figures --out $@

test: check-env $(RAW)
	$(PYTHON) -m pytest -q

# Fail early, with the fix, if the interpreter or packages are wrong.
check-env:
	@$(PYTHON) -c "import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 14))" \
		2>/dev/null || { \
		echo "error: '$(PYTHON)' is not Python 3.11-3.14. Create the environment first:" >&2; \
		echo "  python3.14 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2; \
		exit 1; }
	@$(PYTHON) -c "import certifi, matplotlib, numpy, pandas, pyarrow, pytest" \
		2>/dev/null || { \
		echo "error: packages from requirements.txt are missing. Install them with:" >&2; \
		echo "  $(PYTHON) -m pip install -r requirements.txt" >&2; \
		exit 1; }

# Full rebuild into separate directories, so the committed artifacts and the
# original checkpoints are never overwritten. A model whose directory already
# holds a finished run is skipped, so an interrupted rebuild can be resumed.
rebuild: check-env $(RAW)
	for m in $(MODELS); do \
		$(PYTHON) -m src.pipeline.train --model $$m --models-dir artifacts/rebuild/models \
		|| exit 1; done
	for m in $(MODELS); do \
		$(PYTHON) -m src.pipeline.export --model $$m --models-dir artifacts/rebuild/models \
			--out artifacts/rebuild || exit 1; done
	$(MAKE) reproduce ART=artifacts/rebuild RES=results/rebuild

smoke: check-env $(RAW)
	$(PYTHON) -m src.pipeline.train --model $(SMOKE_MODEL) --smoke \
		--models-dir artifacts/smoke/models
	$(PYTHON) -m src.pipeline.export --model $(SMOKE_MODEL) \
		--models-dir artifacts/smoke/models --out artifacts/smoke
	$(PYTHON) -m src.analysis.tables --table per_epoch_validation --models $(SMOKE_MODEL) \
		--artifacts artifacts/smoke --out results/smoke/per_epoch_validation.csv
	$(PYTHON) -m src.analysis.figures --figure training_curves --models $(SMOKE_MODEL) \
		--artifacts artifacts/smoke --out results/smoke/training_curves.png
	@echo "Smoke run finished: training, export and analysis code all ran."

artifacts: check-env $(RAW)
	for m in $(MODELS); do \
		$(PYTHON) -m src.pipeline.export --model $$m --models-dir artifacts/models \
			--out artifacts || exit 1; done

# Literal paths only: nothing under artifacts/ except the disposable smoke run.
clean:
	rm -f results/tables/*.csv results/figures/*.png results/report/*.md
	rm -rf results/rebuild results/smoke artifacts/smoke

help:
	@sed -n '1,20p' Makefile
