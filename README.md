# BERT, DistilBERT and RoBERTa on SST-2: a reproducible comparison

This project fine-tunes three pre-trained transformer models (BERT,
DistilBERT and RoBERTa) for binary sentiment classification of movie-review
sentences (SST-2) and compares their accuracy, size and training cost. It
began as a STATS 507 final project: one Jupyter notebook whose saved state no
longer produced the numbers in its own report. For the STATS 607 Unit 1
project it was rebuilt as a scripted, tested pipeline in which one command
regenerates every table and figure.

**Main result.** On the 872 validation sentences RoBERTa is the most accurate
model: 93.7% (95% CI 91.9-95.1%). BERT reaches 93.1% and DistilBERT 90.7%.
RoBERTa and BERT are statistically indistinguishable (McNemar exact test,
p = 0.58). DistilBERT has 39% fewer parameters than BERT and needs half the
training compute. It is 2.4-3.0 points less accurate than the other two
(p ≤ 0.004).

## Quick start

You need Python 3.11-3.14, GNU make, git and an internet connection
(see [Requirements](#requirements)).

```sh
git clone <repository-url> sst2-comparison
cd sst2-comparison
python3 --version        # must print 3.11-3.14; otherwise use python3.12, python3.14, ...
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
make reproduce           # every table and figure plus a report, in under a minute
make test                # 50 tests, a few seconds
```

`make reproduce` (also plain `make`) downloads the SST-2 data (3 MB, checksum
verified) and writes the tables to `results/tables/`, the figures to
`results/figures/` and a summary report to `results/report/summary.md`.
Running it a second time rebuilds nothing. No GPU or PyTorch is needed. The
Makefile uses `.venv/bin/python` automatically, so activating the environment
is optional.

## Two ways to reproduce

| | From the included checkpoint | Full rebuild from raw data |
|---|---|---|
| Command | `make reproduce` | `make rebuild` |
| Intended for | the instructor, anyone | someone with an Apple Silicon Mac or a GPU and a free evening |
| Environment | `requirements.txt` | `requirements-train.txt` |
| Starts from | training logs and predictions committed in `artifacts/` | the raw SST-2 files |
| Time | under a minute | about 7-10 hours on an Apple Silicon Mac |
| Writes | `results/` | `artifacts/rebuild/`, then `results/rebuild/` |

**Why the default starts from artifacts.** The raw data are public, but the
fine-tuned models cannot practically be shared: each weight file is 268-499
MB, above GitHub's 100 MB limit, the nine epoch checkpoints total about 10
GB, and retraining takes hours. Every result needs much less. It needs each
model's complete training log and its predictions (logits) for the 872
validation sentences. These files take about 200 KB and are committed:
`artifacts/training_logs/` and `artifacts/predictions/`. They were exported
from the original checkpoints with `make artifacts`. Re-exporting them in a
fresh environment reproduced every file byte for byte. The tests check that
each model's predictions reproduce the accuracy the Trainer logged during
training.

**Full rebuild.** `make rebuild` fine-tunes the three models with the
settings in `config.toml` into `artifacts/rebuild/models/`, exports them, and
builds the same outputs into `results/rebuild/`. The committed artifacts and
the original checkpoints are never overwritten. Expect small differences from
the committed results, of a few validation sentences. The original run did
not seed the classification head, and GPU and Apple MPS training are not
bit-for-bit repeatable. On a Mac, run `caffeinate -is make rebuild` so the
machine does not sleep. In the original run one RoBERTa epoch took 11 hours
instead of about one, most likely because the laptop slept. A model that has
finished is skipped, so an interrupted rebuild can be restarted. `make smoke` runs the same training and export code on 256
sentences in about a minute, after a 270 MB model download on first use, to
show that it works.

```sh
.venv/bin/python -m pip install --only-binary=:all: -r requirements-train.txt
make smoke
caffeinate -is make rebuild     # just `make rebuild` on Linux
```

## Requirements

- **Operating system**: developed and tested on macOS 15 (Apple Silicon). All
  packages also have wheels for macOS 12+ on Intel, Linux (glibc 2.28+) and
  Windows, checked with pip's resolver for Python 3.11-3.14. Linux and
  Windows (through WSL2, for make) were not run directly.
- **Python 3.11-3.14** with the `venv` module. The quick start above was run
  from a fresh clone with Python 3.14.0 and with Python 3.11.16. All 14
  outputs were byte-identical. macOS's `/usr/bin/python3` is 3.9 and too old;
  get Python from python.org or Homebrew. On Ubuntu, `sudo apt install
  python3-venv`. Python 3.15 is not supported yet, because several pinned
  packages have no 3.15 wheels.
- **GNU make and git**: `xcode-select --install` on macOS, `sudo apt install
  make git` on Ubuntu.
- **Network access** to pypi.org and huggingface.co; the data download
  redirects to a `*.hf.co` CDN. Behind a proxy that re-signs HTTPS traffic,
  set `SSL_CERT_FILE` to its certificate bundle. To use a Hugging Face
  mirror, set `HF_ENDPOINT`. `data/raw/README.md` explains how to download
  the files by hand.
- **Full rebuild only**: an Apple Silicon Mac, or Linux or Windows (PyTorch
  2.11 has no Intel-Mac wheels), and about 5 GB of free disk. On a Linux
  machine without a GPU, first run `pip install torch==2.11.0 --index-url
  https://download.pytorch.org/whl/cpu`.

## Data

The data are SST-2, the Stanford Sentiment Treebank (Socher et al., 2013) in
its GLUE version, hosted on the Hugging Face Hub as `nyu-mll/glue`
(configuration `sst2`). It has 67,349 training and 872 validation sentences,
each labelled negative (0) or positive (1). The test labels are hidden, so
all evaluation uses the validation split. `make data` downloads the two
parquet files from revision `bcdcba79d07b…`, the revision the original
analysis used, and accepts them only if their SHA-256 matches
`config.toml`. The data are public and need no account. They are not
redistributed here because the dataset card gives the license as "other".
The pre-trained models are pinned to Hub revisions in `config.toml` too.

## Expected outputs

The designated results are the regenerated versions of the original
report's tables and figures:

| In the original report | Regenerated as | Values you should see |
|---|---|---|
| Table I | `results/tables/model_comparison.csv` | validation accuracy 0.9312 / 0.9071 / 0.9369 (BERT / DistilBERT / RoBERTa), i.e. 812 / 791 / 817 of 872; best epoch 1 / 3 / 2 |
| Table II | `results/tables/per_epoch_validation.csv` | accuracy and loss for 3 epochs × 3 models |
| Fig. 1 | `results/figures/model_comparison.png` | accuracy with 95% CI; training compute 13.3 / 6.7 / 13.3 PFLOPs; parameters 110M / 67M / 125M |
| Fig. 2 | `results/figures/training_curves.png` | training loss by step; validation accuracy and loss by epoch |
| Fig. 3 | `results/figures/confusion_matrices.png` | RoBERTa: TN 390, FP 38, FN 17, TP 427 |
| Section III-D | `results/tables/classification_metrics.csv`, `results/tables/misclassified_examples.csv` | RoBERTa misclassifies 55 sentences |

Supplementary outputs:

- `results/tables/pairwise_comparison.csv`: McNemar tests for each pair of models.
- `results/tables/dataset_summary.csv`, `results/figures/label_distribution.png`, `results/figures/sentence_lengths.png`: the data.
- `results/figures/accuracy_vs_size.png`.
- `results/tables/original_report_comparison.csv`: every number in the original report next to its reproduced value.
- `results/report/summary.md`: all of the above in one page.

## Differences from the original report

19 of the 28 numbers printed in the original report are reproduced exactly.
`results/tables/original_report_comparison.csv` lists all 28. The other nine
come from three problems in the original work, which
[docs/discrepancies.md](docs/discrepancies.md) documents with evidence:

1. **The BERT results came from a run that no longer exists.** BERT was
   trained twice and the second run overwrote the first run's checkpoints.
   The report quotes the first run; the surviving run gives 93.1% at epoch 1
   (tied with epoch 3).
2. **The confusion matrix came from a partly loaded model.** The notebook
   reused the model in memory after the Trainer reloaded its best checkpoint.
   In transformers 5.5.3 that reload skipped the LayerNorm weights. The
   report's matrix sums to 819 correct; the saved best checkpoint gives 817,
   which matches the training log.
3. **The training times include hours of idle time.** One RoBERTa epoch took
   11 hours while the others took about an hour, so the report's speed ratios
   are not supported. The figure now shows training compute, which is
   independent of hardware: BERT and RoBERTa both need 13.3 PFLOPs and
   DistilBERT half that.

## Repository layout

```text
config.toml               every setting that affects results: data and model revisions,
                          hyperparameters, seed
Makefile                  make reproduce | test | rebuild | smoke | artifacts | clean
requirements.txt          environment for make reproduce and make test (every version pinned)
requirements-train.txt    adds PyTorch and transformers for training (every version pinned)
data/raw/                 SST-2 parquet files: downloaded by make data, not committed
data/reference/           numbers transcribed from the original report
src/pipeline/             download_data, train, export, import_original_run
                          (fills data/raw/ and artifacts/)
src/analysis/             metrics, artifacts, tables, figures, report (fills results/)
artifacts/training_logs/  committed: each model's Trainer log and a record of how it was trained
artifacts/predictions/    committed: validation predictions of each model's best checkpoint
artifacts/models/         fine-tuned checkpoints: about 10 GB, on the author's machine only
results/                  generated tables/, figures/ and report/ (not committed)
tests/                    pytest suite: data, functions, artifacts, statistics, pipeline
docs/                     original report and proposal, discrepancies, transformation log
```

## From notebook to pipeline

The original analysis is preserved as the first commit, `eeeb047`
(`eeeb04751632588a8bd174a89c028f39ad62e868`), tagged `original`. To see it,
run `git checkout original` or browse the tag on GitHub.

| In the original notebook | Now |
|---|---|
| `!pip install ...` inside a cell, no versions | `requirements.txt`, `requirements-train.txt` |
| `load_dataset("glue", "sst2")` at whatever version is current | `src/pipeline/download_data.py`: pinned revision, SHA-256 checked |
| three copy-pasted training cells | `src/pipeline/train.py --model <name>` |
| results kept in an in-memory dict | `src/pipeline/export.py` writes them to `artifacts/` |
| plotting cells writing PNGs into the project root | `src/analysis/` writes into `results/` |
| run the cells top to bottom by hand | `make reproduce` |

## Without make

Each Makefile step is a plain Python command, so the pipeline also runs where
make is missing (use `.venv\Scripts\python` on Windows):

```sh
.venv/bin/python -m src.pipeline.download_data --split train
.venv/bin/python -m src.pipeline.download_data --split validation
.venv/bin/python -m src.analysis.tables --table model_comparison --out results/tables/model_comparison.csv
.venv/bin/python -m src.analysis.figures --figure training_curves --out results/figures/training_curves.png
.venv/bin/python -m src.analysis.report --tables-dir results/tables --figures-dir results/figures --out results/report/summary.md
.venv/bin/python -m pytest -q
```

`python -m src.analysis.tables --help` lists every table, and
`python -m src.analysis.figures --help` lists every figure.
