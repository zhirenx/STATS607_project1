# How the project was transformed and verified

A factual record of the restructuring, for readers of the history. `git log`
has the complete sequence of commits.

## Starting point (tag `original`, commit `eeeb047`)

- A single notebook, `sst2_project.ipynb`, run cell by cell. Packages were
  installed inside a cell with `!pip install` and no versions, into the
  system Python 3.14.0. Tutorial-style comments were written in Chinese.
- Three copy-pasted training cells. Results were kept only in an in-memory
  dict, and plots were written to the project root.
- Its saved state was broken. Each training cell ended in a `RuntimeError`
  from the Trainer's notebook progress bar, so the summary CSV was empty, two
  PNGs were blank and two were never written.
- The report was assembled outside the project, and several of its numbers
  cannot be traced to any data ([discrepancies.md](discrepancies.md)).
- About 10 GB of checkpoints sat in `results/`, including optimizer state.
  Nothing was pinned: not the data revision, not the model revisions, and no
  seed before the classification head was created.

## Steps, in order

| Commit(s) | Step |
|---|---|
| `eeeb047` | Preserve the original state (tag `original`), without the weights |
| `bb5c73b` | `.gitignore` for environments, data, checkpoints and generated results |
| `7bd62f2` | Reorganize into `docs/`, `artifacts/` and `results/`. Each model's final training log is moved unchanged, so git records a pure rename |
| `dd9e8aa` | Pin the environment |
| `4c1c440` | Pinned, checksummed data download |
| `a038c5b` | Pure functions for metrics and statistics |
| `2a22d89` | Export from checkpoints to small artifacts, with a provenance record for each field |
| `8994cfc`, `b25e544` | Fixes found while checking steps 4-5: pin every dependency for Python 3.11-3.14; clearer download errors |
| `01f05d3` | Commit the exported artifacts, with the commands that produced them |
| `d7194d6`, `cc4270c`, `3bcd63c` | Tables, figures and a generated report, built from the artifacts |
| `b4f9f43` | One parameterized training script in place of the copy-pasted cells |
| `bdf98cf` | A Makefile: `make reproduce` by default, plus test, rebuild, smoke, artifacts and clean |
| `90ef0dd` | Tests of five kinds: data, functions, artifacts, statistics, pipeline |
| `416e45f`, `49ea8ab`, `d130ddb` | README and these documents, then removal of the notebook and its stale outputs |
| `380b170`, `fae2fce` and later | Fixes from reviewing the finished repository (stronger tests, CI, documentation) |

## Verification performed

- Re-exporting the artifacts from the original checkpoints, with the
  committed code in a fresh venv built from `requirements-train.txt`,
  reproduced every file byte for byte.
- Each model's re-scored predictions give exactly the best accuracy logged
  during training (812, 791 and 817 of 872). A separate re-scoring by one of
  the AI review passes, on CPU and on MPS, gave the same predictions.
- From a fresh clone, the README quick start (`make reproduce`, `make test`)
  was run on Python 3.14.0 and 3.11.16. All 14 outputs were byte-identical,
  all tests passed, a second `make reproduce` rebuilt nothing, and
  `git status` stayed clean.
- `make -n reproduce` lists no training or export commands, even after every
  source file is touched to look newer than the artifacts (GNU Make 4.x
  compares sub-second timestamps).
- `pip install --dry-run --only-binary=:all:` for Python 3.11-3.14 with the
  platform tags of Linux, macOS (Apple Silicon and Intel) and Windows. Every
  pinned package has a matching wheel, except PyTorch on Intel macOS. The
  check ran on macOS, so PyTorch's Linux-only CUDA dependencies were not
  resolved; the GitHub Actions workflow runs the analysis environment on
  Ubuntu for real.
- `make smoke` fine-tuned, saved, exported and re-scored DistilBERT
  consistently. The Makefile also ran under `/bin/dash`.
- Before implementation, four AI review passes (Claude Code sub-agents, not
  people) checked the design for rubric compliance, fresh-machine failures,
  technical risks and the instructor's perspective. Their findings shaped
  the Makefile graph, the dependency pins, the tests and the provenance
  records. The finished repository got a second round of AI reviews,
  including mutation testing of the test suite.

## Tools

The investigation, code, tests, documentation and commits were produced
with Claude Code, an AI coding assistant, working at the author's request.
The `Co-Authored-By` trailers on the commits reflect this.
