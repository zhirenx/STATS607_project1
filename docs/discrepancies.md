# Why some numbers differ from the original report

The original report ([original_report.pdf](original_report.pdf)) was put
together by hand outside the project. The notebook's saved state could not
regenerate it. Every training cell ended in
`RuntimeError: on_train_begin must be called before on_evaluate`, the summary
CSV was empty, two plots were blank axes, and two were never written. Rebuilt
from the data that survived, the pipeline reproduces 19 of the report's 28
numbers exactly; `results/tables/original_report_comparison.csv` lists all 28.
The other nine have three causes. The evidence below can be checked in the
commit tagged `original`, for example with
`git show original:sst2_project.ipynb`.

## 1. The BERT row describes a run that no longer exists (7 numbers)

The report gives BERT 92.2 / 91.5 / 93.2% validation accuracy by epoch,
169 minutes of training and a best epoch of 3. None of these numbers appears
anywhere in the project:

- The surviving training log (`artifacts/training_logs/bert/trainer_state.json`,
  byte-identical to `results/BERT/checkpoint-12630/trainer_state.json` at
  `original`) and the notebook's own progress bar in the BERT training cell
  agree: 93.12 / 91.74 / 93.12% and 4:53:46.
- On the author's machine, the BERT checkpoint folders were created on
  9 April, but the weights inside were written on 20 April. BERT was trained
  twice, and the second run overwrote the first.
- For DistilBERT and RoBERTa the report matches the surviving logs number
  for number. The BERT row most likely comes from the overwritten first run.

Epochs 1 and 3 tie at 812/872 correct. The Trainer replaces its best
checkpoint only when accuracy strictly improves, so it recorded epoch 1
(`checkpoint-4210`), not epoch 3.

## 2. The confusion matrix came from a partly loaded model (2 numbers)

The report's RoBERTa matrix is TN 392, FP 36, FN 17, TP 427, which is 819
correct (93.9%). No saved RoBERTa checkpoint scores that. The best checkpoint
(epoch 2) gives 390 / 38 / 17 / 427, which is 817 correct (93.69%) and equals
the accuracy the Trainer logged. Epoch 3 gives 815.

Here is why. transformers 5.5.3 saves LayerNorm weights under the legacy
names `gamma`/`beta`. With `load_best_model_at_end=True`, the Trainer
reloaded the best checkpoint by matching names directly. It found no
`LayerNorm.weight`/`bias`, and the RoBERTa training cell prints "There were
missing keys in the checkpoint model loaded: ['roberta.embeddings.LayerNorm.weight', …]".
It then kept the epoch-3 LayerNorm values. The notebook evaluated this mix of
epochs 2 and 3; its error-analysis cell prints "53 / 872" errors, which is
819 correct. The pipeline instead loads checkpoints with `from_pretrained`,
which maps the names. It refuses any checkpoint with missing or unexpected
weights, and training no longer reloads models inside the Trainer.

## 3. The training times include idle time (the efficiency claims)

The recorded wall-clock times are 4:53:46 for BERT, 1:13:22 for DistilBERT
and 13:25:37 for RoBERTa; the report's 169 minutes for BERT belongs to the
lost run. Checkpoint save times split them by epoch:

- BERT: 163 / 86 / 46 minutes
- DistilBERT: 22 / 28 / 23 minutes
- RoBERTa: 64 / 663 / 79 minutes

RoBERTa's second epoch took 11 hours while its others took about one,
almost certainly because the laptop slept. BERT's first epoch is inflated
too. So the report's "2.3x faster than BERT and 11x faster than RoBERTa" is
not supported. The Trainer's count of floating-point operations does not
depend on hardware or sleep. By that count BERT and RoBERTa cost the same
(13.3 PFLOPs) and DistilBERT half as much (6.7 PFLOPs). Figure 1(b) now shows
compute, and the wall-clock times are kept only as a record in
`model_comparison.csv`.

## Not a discrepancy

The report says "sentences are short (mean ~19 words)". That is true of the
validation split (19.5 words). The notebook measured the training split (9.4
words), which also contains short phrases. `dataset_summary.csv` reports
both.

## Effect on the conclusions

The accuracy differences are at most 0.9 points (8 of 872 sentences), which
is within sampling error: a 95% interval is about ±1.6 points. What they show
is that the report could not be traced back to its data. The ranking needs
restating as well. RoBERTa's lead over BERT is 5 sentences, which is not
significant (McNemar exact p = 0.58), while DistilBERT is significantly
worse than both (p ≤ 0.004).
