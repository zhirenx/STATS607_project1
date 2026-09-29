"""Draw the result figures from the exported artifacts and the raw data.

Figures (PNG, one per invocation):

* ``label_distribution``: class balance of the training and validation splits
* ``sentence_lengths``: words per sentence in each split
* ``model_comparison``: accuracy with 95% CI, training compute, size
* ``training_curves``: training loss by step; validation accuracy and loss by epoch
* ``confusion_matrices``: validation confusion matrix of each model
* ``accuracy_vs_size``: accuracy (95% CI) against number of parameters

Each model keeps the same colour in every figure.

Usage::

    python -m src.analysis.figures --figure training_curves --out results/figures/training_curves.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only; works on machines without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis import metrics  # noqa: E402
from src.analysis.artifacts import ModelRun, load_runs  # noqa: E402
from src.analysis.tables import build_model_comparison  # noqa: E402
from src.config import ARTIFACTS_DIR, DEFAULT_CONFIG, Config, load_config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.log import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1-3 of the reference palette; validated as colour-blind
# safe for all pairs, which the scatter plot needs.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
NEGATIVE, POSITIVE = "#e34948", "#1c5cab"  # the two poles of a diverging pair
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # ships with matplotlib, so output is machine-independent
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def style_axes(ax, grid_axis: str = "y") -> None:
    """Apply the shared look: no top/right spines, hairline grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)


def model_colors(runs: list[ModelRun], config: Config) -> dict[str, str]:
    """Map each model key to its fixed colour: its position in config.toml, never its rank."""
    order = config.all_model_keys or tuple(config.models)
    return {run.key: MODEL_COLORS[order.index(run.key) % len(MODEL_COLORS)] for run in runs}


def plot_label_distribution(config: Config, runs: list[ModelRun]):
    """Stacked bar per split showing the share of negative and positive sentences."""
    splits = list(config.data.splits)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for row, split in enumerate(splits):
        labels = load_split(split)["label"]
        n = len(labels)
        negative, positive = int((labels == 0).sum()), int((labels == 1).sum())
        gap = 0.002  # surface gap between the two segments
        ax.barh(row, negative / n - gap, color=NEGATIVE, height=0.5)
        ax.barh(row, positive / n - gap, left=negative / n + gap, color=POSITIVE, height=0.5)
        ax.text(0.01, row, f"negative {negative:,} ({negative / n:.0%})", va="center",
                color="white", fontsize=8.5)
        ax.text(0.99, row, f"positive {positive:,} ({positive / n:.0%})", va="center",
                ha="right", color="white", fontsize=8.5)
    ax.set_yticks(range(len(splits)), [f"{s} (n={config.data.splits[s].rows:,})" for s in splits])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("SST-2 class balance")
    style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig


def plot_sentence_lengths(config: Config, runs: list[ModelRun]):
    """Histogram of words per sentence for each split, with mean and 95th percentile."""
    splits = list(config.data.splits)
    fig, axes = plt.subplots(1, len(splits), figsize=(8, 3), sharex=True)
    for ax, split in zip(np.atleast_1d(axes), splits):
        words = load_split(split)["sentence"].str.split().str.len()
        bins = np.arange(0.5, words.max() + 1.5)
        ax.hist(words, bins=bins, color=MODEL_COLORS[0], edgecolor=SURFACE, linewidth=0.4)
        for value, name in [(words.mean(), "mean"), (np.percentile(words, 95), "95th pct")]:
            ax.axvline(value, color=INK_SECONDARY, linewidth=1)
            ax.text(value + 0.8, ax.get_ylim()[1] * 0.93, f"{name} {value:.1f}",
                    color=INK_SECONDARY, fontsize=8)
        ax.set_title(f"{split} (n={len(words):,})")
        ax.set_xlabel("words per sentence")
        style_axes(ax)
    np.atleast_1d(axes)[0].set_ylabel("sentences")
    fig.suptitle("Sentence length in SST-2", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_model_comparison(config: Config, runs: list[ModelRun]):
    """Accuracy with 95% CI, recorded training time and parameter count per model."""
    table = build_model_comparison(runs)
    colors = [model_colors(runs, config)[run.key] for run in runs]
    y = np.arange(len(runs))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)

    ax = axes[0]
    acc = 100 * table["val_accuracy"].to_numpy()
    low = 100 * table["val_accuracy_ci95_low"].to_numpy()
    high = 100 * table["val_accuracy_ci95_high"].to_numpy()
    for i in range(len(runs)):
        ax.plot([low[i], high[i]], [y[i], y[i]], color=colors[i], linewidth=2)
        ax.plot(acc[i], y[i], "o", color=colors[i], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(high[i] + 0.25, y[i], f"{acc[i]:.1f}%", va="center", color=INK, fontsize=8.5)
    ax.set_xlim(low.min() - 1, high.max() + 1.8)
    ax.set_title("(a) Validation accuracy, 95% CI")
    ax.set_xlabel("accuracy (%)")

    panels = [
        (axes[1], table["train_pflops"].to_numpy(dtype=float), "(b) Training compute*",
         "floating-point operations (peta)", "{:.1f}"),
        (axes[2], table["parameters_millions"].to_numpy(dtype=float), "(c) Model size",
         "parameters (millions)", "{:.0f}M"),
    ]
    for ax, values, title, xlabel, fmt in panels:
        ax.barh(y, values, color=colors, height=0.45)
        for i, value in enumerate(values):
            if not np.isnan(value):
                ax.text(value + values[~np.isnan(values)].max() * 0.02, y[i], fmt.format(value),
                        va="center", color=INK, fontsize=8.5)
        ax.set_xlim(0, np.nanmax(values) * 1.18)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    for ax in axes:
        style_axes(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, table["model"])
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, "* counted by the Hugging Face Trainer, independent of hardware. "
             "The recorded wall-clock times include idle time (see model_comparison.csv).",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_training_curves(config: Config, runs: list[ModelRun]):
    """Training loss by step, and validation accuracy and loss by epoch."""
    colors = model_colors(runs, config)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for run in runs:
        train = metrics.train_loss_history(run.log_history)
        evals = metrics.eval_history(run.log_history)
        color = colors[run.key]
        axes[0].plot(train["step"], train["loss"], color=color, linewidth=1.2, label=run.display_name)
        for ax, column in [(axes[1], "eval_accuracy"), (axes[2], "eval_loss")]:
            values = evals[column] * (100 if column == "eval_accuracy" else 1)
            ax.plot(evals["epoch"], values, "-o", color=color, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=run.display_name)
    axes[0].set_title("(a) Training loss")
    axes[0].set_xlabel("optimisation step")
    axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_title("(b) Validation accuracy")
    axes[1].set_ylabel("accuracy (%)")
    axes[2].set_title("(c) Validation loss")
    axes[2].set_ylabel("cross-entropy loss")
    for ax in axes[1:]:
        ax.set_xlabel("epoch")
        ax.set_xticks([1, 2, 3])
    for ax in axes:
        style_axes(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(runs), bbox_to_anchor=(1, 1))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_confusion_matrices(config: Config, runs: list[ModelRun]):
    """One 2x2 confusion matrix per model on a shared colour scale."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQUENTIAL)
    counts = {run.key: metrics.confusion_counts(run.predictions["label"], run.predictions["pred"])
              for run in runs}
    vmax = max(max(c.values()) for c in counts.values())
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 3.2))
    for ax, run in zip(np.atleast_1d(axes), runs):
        c = counts[run.key]
        grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax)
        for (i, j), value in np.ndenumerate(grid):
            ax.text(j, i, f"{value}", ha="center", va="center", fontsize=11,
                    color="white" if value > vmax * 0.55 else INK)
        acc = metrics.accuracy(run.predictions["label"], run.predictions["pred"])
        ax.set_title(f"{run.display_name} ({acc:.1%})")
        ax.set_xticks([0, 1], ["negative", "positive"])
        first = ax is np.atleast_1d(axes)[0]
        ax.set_yticks([0, 1], ["negative", "positive"] if first else [])
        ax.set_xlabel("predicted")
        if first:
            ax.set_ylabel("true")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, label="sentences")
    colorbar.outline.set_visible(False)
    return fig


def plot_accuracy_vs_size(config: Config, runs: list[ModelRun]):
    """Validation accuracy (95% CI) against parameter count."""
    table = build_model_comparison(runs)
    colors = model_colors(runs, config)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sizes = table["parameters_millions"].to_numpy()
    for run, row in zip(runs, table.itertuples()):
        acc = 100 * row.val_accuracy
        # Label to the right unless another model sits within 30M parameters
        # to the right, where the label would cross that model's interval.
        crowded = np.any((sizes > row.parameters_millions) & (sizes - row.parameters_millions < 30))
        offset, align = (-2.5, "right") if crowded else (2.5, "left")
        ax.plot([row.parameters_millions] * 2,
                [100 * row.val_accuracy_ci95_low, 100 * row.val_accuracy_ci95_high],
                color=colors[run.key], linewidth=2)
        ax.plot(row.parameters_millions, acc, "o", color=colors[run.key], markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(row.parameters_millions + offset, acc, f"{run.display_name} {acc:.1f}%",
                va="center", ha=align, color=INK, fontsize=8.5)
    ax.set_xlim(sizes.min() - 10, sizes.max() + 30)
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Accuracy vs. model size (bars: 95% CI)")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return fig


FIGURES = {
    "label_distribution": (plot_label_distribution, False),
    "sentence_lengths": (plot_sentence_lengths, False),
    "model_comparison": (plot_model_comparison, True),
    "training_curves": (plot_training_curves, True),
    "confusion_matrices": (plot_confusion_matrices, True),
    "accuracy_vs_size": (plot_accuracy_vs_size, True),
}


def main(argv: list[str] | None = None) -> int:
    """Draw one figure and save it as PNG."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", required=True, choices=list(FIGURES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR,
                        help="directory holding training_logs/ and predictions/")
    parser.add_argument("--models", nargs="+", help="restrict to these model keys")
    args = parser.parse_args(argv)
    setup_logging()

    config = load_config(args.config)
    if args.models:
        config = config.restrict(args.models)
    draw, needs_runs = FIGURES[args.figure]
    runs = load_runs(config, args.artifacts) if needs_runs else []
    fig = draw(config, runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Drop the matplotlib version from the PNG metadata so identical inputs
    # give byte-identical files.
    fig.savefig(args.out, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
