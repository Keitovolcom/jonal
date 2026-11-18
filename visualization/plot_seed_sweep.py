"""Plot CNN5Layer learning curves as a 3-by-N grid, one column per run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RUN_PATTERN = (
    "emnist-digits_cnn5layer_noise-0p2_lr-0p01_loss-crossentropy_bs-256_"
    "opt-sgd_width-1_ch-1_aug-none_seed-*"
)
DEFAULT_OUTPUT = RESULTS_DIR / "viz" / "cnn5layer_seed_learning_curves.pdf"

COLORS = {
    "test": "#d62728",   # red
    "train": "#1f77b4",  # blue
    "clean": "#2ca02c",  # green
    "noisy": "#9467bd",  # purple
}
ERROR_Y_LIMITS = (-0.05, 0.45)
NOISY_Y_LIMITS = (-0.01, 1.01)
LOSS_Y_LIMITS = (-0.01, 3.0)
NOISY_LOSS_Y_LIMITS = (-0.05, 7.0)
CLEAN_Y_MAX = 0.305
CLEAN_LOSS_Y_MAX = 0.5


def zero_aligned_limits(
    primary: tuple[float, float], upper: float
) -> tuple[float, float]:
    """Secondary-axis limits whose 0 sits at the same height as the primary 0.

    Solves (0 - lo) / (upper - lo) == (0 - p_lo) / (p_hi - p_lo) for lo.
    """
    zero_fraction = -primary[0] / (primary[1] - primary[0])
    return (-zero_fraction * upper / (1 - zero_fraction), upper)


ROW_Y_LIMITS = (
    ERROR_Y_LIMITS,
    NOISY_Y_LIMITS,
    LOSS_Y_LIMITS,
    NOISY_LOSS_Y_LIMITS,
)
# Metrics drawn on a secondary (right) y-axis: (limits, axis label).
SECONDARY_METRICS = {
    "train_error_clean": (
        zero_aligned_limits(NOISY_Y_LIMITS, CLEAN_Y_MAX),
        "Train clean error",
    ),
    "train_loss_clean": (
        zero_aligned_limits(NOISY_LOSS_Y_LIMITS, CLEAN_LOSS_Y_MAX),
        "Train clean loss",
    ),
}

ROW_SPECS = (
    (
        "Error",
        (
            ("test_error", "Test", COLORS["test"]),
            ("train_error", "Train", COLORS["train"]),
        ),
    ),
    (
        "Train noisy error",
        (
            ("train_error_noisy", "Train noisy", COLORS["noisy"]),
            ("train_error_clean", "Train clean", COLORS["clean"]),
        ),
    ),
    (
        "Loss",
        (
            ("test_loss", "Test", COLORS["test"]),
            ("train_loss", "Train", COLORS["train"]),
        ),
    ),
    (
        "Train noisy loss",
        (
            ("train_loss_noisy", "Train noisy", COLORS["noisy"]),
            ("train_loss_clean", "Train clean", COLORS["clean"]),
        ),
    ),
)


DEFAULT_TITLE = "CNN5Layer seed sweep (EMNIST Digits, 20% label noise, width = 1)"
# Config keys that can label the columns, with their short display names.
COLUMN_LABELS = {
    "seed": "seed",
    "label_noise": "noise",
    "model_width": "width",
}


@dataclass(frozen=True)
class Run:
    seed: int
    directory: Path
    metrics: pd.DataFrame
    config: dict


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot CNN5Layer learning curves with one column per run."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory searched recursively for run directories.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_RUN_PATTERN,
        help="Glob pattern for run directory names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path (format is inferred from the suffix).",
    )
    parser.add_argument(
        "--column-key",
        choices=sorted(COLUMN_LABELS),
        default="seed",
        help="Config key that orders and labels the columns.",
    )
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def load_run(directory: Path) -> Run:
    with (directory / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    metrics = pd.read_csv(directory / "metrics.csv")
    required_columns = {
        metric for _, series in ROW_SPECS for metric, _, _ in series
    } | {"epoch"}
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise ValueError(
            f"missing columns in {directory}: {', '.join(sorted(missing))}"
        )

    # A logarithmic x-axis cannot represent epoch 0.
    metrics = metrics.loc[metrics["epoch"] > 0].sort_values("epoch")
    return Run(
        seed=int(config["seed"]),
        directory=directory,
        metrics=metrics,
        config=config,
    )


def collect_runs(
    results_dir: Path, pattern: str, column_key: str = "seed"
) -> list[Run]:
    directories = [
        path.parent
        for path in results_dir.rglob(f"{pattern}/metrics.csv")
        if (path.parent / "config.json").is_file()
    ]
    runs = sorted(
        (load_run(path) for path in directories),
        key=lambda run: run.config[column_key],
    )
    if not runs:
        raise ValueError(f"no runs matching {pattern!r} under {results_dir}")
    values = [run.config[column_key] for run in runs]
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {column_key} values found: {values}")
    return runs


def plot_runs(
    runs: list[Run],
    output: Path,
    dpi: int,
    column_key: str = "seed",
    title: str = DEFAULT_TITLE,
) -> None:
    ncols = len(runs)
    figure, axes = plt.subplots(
        nrows=len(ROW_SPECS),
        ncols=ncols,
        figsize=(4 * ncols, 12),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for column, run in enumerate(runs):
        axes[0, column].set_title(
            f"{COLUMN_LABELS[column_key]} = {run.config[column_key]:g}",
            fontsize=13,
        )
        for row, (row_label, series) in enumerate(ROW_SPECS):
            axis = axes[row, column]
            secondary = None
            lines = []
            for metric, label, color in series:
                target = axis
                if metric in SECONDARY_METRICS:
                    if secondary is None:
                        secondary = axis.twinx()
                        limits, axis_label = SECONDARY_METRICS[metric]
                        secondary.set_ylim(*limits)
                        secondary.tick_params(axis="y", colors=color)
                        if column == ncols - 1:
                            secondary.set_ylabel(axis_label, color=color)
                        else:
                            secondary.tick_params(right=False, labelright=False)
                    target = secondary
                (line,) = target.plot(
                    run.metrics["epoch"],
                    run.metrics[metric],
                    color=color,
                    linewidth=1.35,
                    label=label,
                )
                lines.append(line)
            axis.set_xscale("log")
            axis.set_ylim(*ROW_Y_LIMITS[row])
            axis.grid(True, which="major", color="0.85", linewidth=0.7)
            axis.grid(True, which="minor", color="0.92", linewidth=0.5)
            if column == 0:
                axis.set_ylabel(row_label)
                axis.legend(
                    handles=lines, loc="upper right", frameon=True, fontsize=8
                )
            if row == len(ROW_SPECS) - 1:
                axis.set_xlabel("Epoch (log scale)")

    min_epoch = min(run.metrics["epoch"].min() for run in runs)
    max_epoch = max(run.metrics["epoch"].max() for run in runs)
    for axis in axes.flat:
        axis.set_xlim(min_epoch, max_epoch)

    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = create_parser().parse_args()
    runs = collect_runs(args.results_dir, args.pattern, args.column_key)
    for run in runs:
        print(f"{args.column_key} {run.config[args.column_key]:g}: {run.directory}")
    plot_runs(runs, args.output, args.dpi, args.column_key, args.title)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
