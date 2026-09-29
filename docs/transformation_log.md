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

## Steps (one commit each, in order)

1. Preserve the original state (tag `original`); leave out the weights.
2. `.gitignore` for environments, data, checkpoints and generated results.
3. Reorganize into `docs/`, `artifacts/` and `results/`. The final training
   log of each model is moved unchanged, so git records a pure rename.
4. Pin the environment. A second commit pins every dependency so that the
   files install on Python 3.11-3.14.
5. Pinned, checksummed data download, with clear errors.
6. Pure functions for metrics and statistics.
7. Export from checkpoints to small committed artifacts, with a provenance
   record for each field.
8. Commit the exported artifacts, with the commands that produced them.
9. Tables, figures and a generated report, built from the artifacts.
10. One parameterized training script in place of the copy-pasted cells.
11. A Makefile: `make reproduce` by default, plus test, rebuild, smoke,
    artifacts and clean.
12. Tests of five kinds: data, functions, artifacts, statistics, pipeline.
13. README and these documents; then removal of the notebook and its stale
    outputs from the working tree.

## Verification performed

- Re-exporting the artifacts from the original checkpoints, with the
  committed code in a fresh venv built from `requirements-train.txt`,
  reproduced every file byte for byte.
- Each model's re-scored predictions give exactly the best accuracy logged
  during training (812, 791 and 817 of 872). An independent re-scoring on CPU
  and on MPS gave the same predictions.
- From a fresh clone, the README quick start (`make reproduce`, `make test`)
  was run on Python 3.14.0 and 3.11.16. All 14 outputs were byte-identical,
  all tests passed, a second `make reproduce` rebuilt nothing, and
  `git status` stayed clean.
- `make -n reproduce` lists no training or export commands, even after every
  source file is touched to look newer than the artifacts (GNU Make 4.x
  compares sub-second timestamps).
- `pip install --dry-run` for Python 3.11-3.14 on Linux, macOS (Apple Silicon
  and Intel) and Windows. The analysis environment resolves everywhere. The
  training environment resolves everywhere except Intel macOS, where PyTorch
  2.11 has no wheels.
- `make smoke` fine-tuned, saved, exported and re-scored DistilBERT
  consistently. The Makefile also ran under `/bin/dash`.
- Before implementation, four independent reviews checked the design for
  rubric compliance, fresh-machine failures, technical risks and the
  instructor's perspective. Their findings shaped the Makefile graph, the
  dependency pins, the tests and the provenance records.

## Tools

The investigation, code, tests, documentation and commits were produced
with Claude Code, an AI coding assistant, working at the author's request.
The `Co-Authored-By` trailers on the commits reflect this.
