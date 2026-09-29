# Model checkpoints (not in git)

This directory holds fine-tuned model checkpoints. They are too large for
GitHub: each `model.safetensors` is 268-499 MB, and the nine epoch
checkpoints of the original run take about 10 GB including optimizer state.
Git ignores everything here except this file.

On the author's machine this directory holds the original run. It was moved
here unchanged from `results/<Model>/` of the original notebook (with `mv`
on the same disk, so file times were preserved):

    artifacts/models/<model>/checkpoint-4210/    epoch 1
    artifacts/models/<model>/checkpoint-8420/    epoch 2
    artifacts/models/<model>/checkpoint-12630/   epoch 3
    artifacts/models/<model>/run_info.json       where and how the run was made

The analysis never reads these files directly. The small files exported from
them to `artifacts/training_logs/` and `artifacts/predictions/` are
committed, and every table and figure is built from those.
