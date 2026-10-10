"""Plot all CNN5Layer widths together in a 5-by-1 metrics figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __package__:
    from .plot_width_sweep import (
    DEFAULT_SWEEP_DIR,
    DEFAULT_WIDTH_ONE_DIR,
    ERROR_Y_LIMITS,
    LOSS_Y_LIMITS,
    PROJECT_ROOT,
    Run,
    collect_runs,
    )
else:
    from plot_width_sweep import (
    DEFAULT_SWEEP_DIR,
    DEFAULT_WIDTH_ONE_DIR,
    ERROR_Y_LIMITS,
    LOSS_Y_LIMITS,
    PROJECT_ROOT,
    Run,
    collect_runs,
    )


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "cnn5layer-width-sweep-20260909T010104Z"
    / "clean_change_width_sweep_metrics_overlay.pdf"
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot four CNN5Layer widths together in a 5-row figure."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=DEFAULT_SWEEP_DIR,
        help="Directory containing the width-sweep run directories.",
    )
    parser.add_argument(
        "--width-one-dir",
        type=Path,
        default=DEFAULT_WIDTH_ONE_DIR,
        help="Run directory for width=1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path (format is inferred from the suffix).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def plot_runs(runs: list[Run], output: Path, dpi: int) -> None:
    widths = [run.width for run in runs]
    width_norm = LogNorm(vmin=min(widths), vmax=max(widths))
    width_cmap = plt.get_cmap("viridis")
    width_colors = [width_cmap(width_norm(width)) for width in widths]

    error_rows = (
        ("Test error", "test_error"),
        ("Train error", "train_error"),
        ("Train clean error", "train_error_clean"),
        ("Train noisy error", "train_error_noisy"),
    )
    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(5, 15),
        sharex=True,
        constrained_layout=True,
    )

    for axis, (ylabel, metric) in zip(axes[:4], error_rows):
        for run, color in zip(runs, width_colors):
            axis.plot(
                run.metrics["epoch"],
                run.metrics[metric],
                color=color,
                linewidth=1.4,
            )
        axis.set_ylabel(ylabel)
        y_limits_by_metric = {
            "train_error_clean": (-0.01, 0.41),
        }
        axis.set_ylim(*y_limits_by_metric.get(metric, ERROR_Y_LIMITS))

    loss_axis = axes[4]
    for run, color in zip(runs, width_colors):
        loss_axis.plot(
            run.metrics["epoch"],
            run.metrics["test_loss"],
            color=color,
            linestyle="-",
            linewidth=1.4,
        )
        loss_axis.plot(
            run.metrics["epoch"],
            run.metrics["train_loss"],
            color=color,
            linestyle="--",
            linewidth=1.4,
        )
    loss_axis.set_ylabel("Loss")
    loss_axis.set_ylim(*LOSS_Y_LIMITS)
    loss_axis.set_xlabel("Epoch (log scale)")

    min_epoch = min(run.metrics["epoch"].min() for run in runs)
    max_epoch = max(run.metrics["epoch"].max() for run in runs)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(min_epoch, max_epoch)
        axis.grid(True, which="major", color="0.85", linewidth=0.7)
        axis.grid(True, which="minor", color="0.92", linewidth=0.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = create_parser().parse_args()
    runs = collect_runs(args.sweep_dir, args.width_one_dir)
    plot_runs(runs, args.output, args.dpi)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
