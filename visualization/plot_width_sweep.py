"""Plot the CNN5Layer width sweep as a 5-by-4 metrics grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_DIR = (
    PROJECT_ROOT / "results" / "cnn5layer-width-sweep-20260909T010104Z"
)
DEFAULT_WIDTH_ONE_DIR = PROJECT_ROOT / "results" / (
    "emnist-digits_cnn5layer_noise-0p2_lr-0p01_loss-crossentropy_bs-256_"
    "opt-sgd_width-1_ch-1_aug-none_seed-42"
)
DEFAULT_OUTPUT = DEFAULT_SWEEP_DIR / "width_sweep_metrics.pdf"

COLORS = {
    "test": "#d62728",   # red
    "train": "#1f77b4",  # blue
    "clean": "#2ca02c",  # green
    "noisy": "#9467bd",  # purple
}
ERROR_Y_LIMITS = (-0.01, 1.01)
LOSS_Y_LIMITS = (-0.01, 5.01)


@dataclass(frozen=True)
class Run:
    width: float
    directory: Path
    metrics: pd.DataFrame


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot four CNN5Layer widths in a 5-row metrics grid."
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


def load_run(directory: Path) -> Run:
    config_path = directory / "config.json"
    metrics_path = directory / "metrics.csv"
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    try:
        width = float(config["model_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid model_width in {config_path}") from exc

    metrics = pd.read_csv(metrics_path)
    required_columns = {
        "epoch",
        "test_error",
        "train_error",
        "train_error_clean",
        "train_error_noisy",
        "test_loss",
        "train_loss",
    }
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise ValueError(
            f"missing columns in {metrics_path}: {', '.join(sorted(missing))}"
        )

    # A logarithmic x-axis cannot represent epoch 0.
    metrics = metrics.loc[metrics["epoch"] > 0].sort_values("epoch")
    if metrics.empty:
        raise ValueError(f"no positive epochs found in {metrics_path}")
    return Run(width=width, directory=directory, metrics=metrics)


def collect_runs(sweep_dir: Path, width_one_dir: Path) -> list[Run]:
    run_directories = sorted(path.parent for path in sweep_dir.glob("*/metrics.csv"))
    run_directories.append(width_one_dir)
    runs = sorted((load_run(path) for path in run_directories), key=lambda run: run.width)

    if len(runs) != 4:
        raise ValueError(f"expected exactly 4 runs, found {len(runs)}")
    widths = [run.width for run in runs]
    if len(set(widths)) != len(widths):
        raise ValueError(f"duplicate widths found: {widths}")
    return runs


def plot_runs(runs: list[Run], output: Path, dpi: int) -> None:
    row_specs = (
        ("Test error", (("test_error", "Test", COLORS["test"]),)),
        ("Train error", (("train_error", "Train", COLORS["train"]),)),
        (
            "Train clean error",
            (("train_error_clean", "Clean", COLORS["clean"]),),
        ),
        (
            "Train noisy error",
            (("train_error_noisy", "Noisy", COLORS["noisy"]),),
        ),
        (
            "Loss",
            (
                ("test_loss", "Test", COLORS["test"]),
                ("train_loss", "Train", COLORS["train"]),
            ),
        ),
    )

    figure, axes = plt.subplots(
        nrows=5,
        ncols=4,
        figsize=(16, 14),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for column, run in enumerate(runs):
        axes[0, column].set_title(f"width = {run.width:g}", fontsize=13)
        for row, (row_label, series) in enumerate(row_specs):
            axis = axes[row, column]
            for metric, label, color in series:
                axis.plot(
                    run.metrics["epoch"],
                    run.metrics[metric],
                    color=color,
                    linewidth=1.35,
                    label=label,
                )
            axis.set_xscale("log")
            axis.set_ylim(*(LOSS_Y_LIMITS if row == 4 else ERROR_Y_LIMITS))
            axis.grid(True, which="major", color="0.85", linewidth=0.7)
            axis.grid(True, which="minor", color="0.92", linewidth=0.5)
            if column == 0:
                axis.set_ylabel(row_label)
            if row == len(row_specs) - 1:
                axis.set_xlabel("Epoch (log scale)")
            if row == len(row_specs) - 1:
                axis.legend(loc="upper right", frameon=True, fontsize=8)

    min_epoch = min(run.metrics["epoch"].min() for run in runs)
    max_epoch = max(run.metrics["epoch"].max() for run in runs)
    for axis in axes.flat:
        axis.set_xlim(min_epoch, max_epoch)

    figure.suptitle("CNN5Layer width sweep (EMNIST Digits, 20% label noise)")
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
