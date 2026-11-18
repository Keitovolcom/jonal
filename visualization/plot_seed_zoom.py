"""Zoom into the second descent of CNN5Layer seeds and mark key epochs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from .plot_seed_sweep import (
        COLORS,
        DEFAULT_RUN_PATTERN,
        RESULTS_DIR,
        Run,
        collect_runs,
    )
else:
    from plot_seed_sweep import (
        COLORS,
        DEFAULT_RUN_PATTERN,
        RESULTS_DIR,
        Run,
        collect_runs,
    )


DEFAULT_OUTPUT = RESULTS_DIR / "viz" / "cnn5layer_seed_second_descent_zoom.pdf"
# Epoch window in which the peaks are searched for.
SEARCH_WINDOW = (20, 400)
SMOOTHING_WINDOW = 9
# Test error is "settled" once it stays within this margin of its final value.
SETTLE_MARGIN = 0.01

ROW_SPECS = (
    ("Test error", "test_error", COLORS["test"], (-0.005, 0.22)),
    ("Train noisy error", "train_error_noisy", COLORS["noisy"], (-0.01, 1.01)),
    ("Train clean error", "train_error_clean", COLORS["clean"], (-0.001, 0.032)),
)
# (key, label, color, linestyle)
EVENT_SPECS = (
    ("clean_peak", "Clean peak", COLORS["clean"], "--"),
    ("test_peak", f"Test peak ({SMOOTHING_WINDOW}-ep MA)", COLORS["test"], "--"),
    ("test_settle", "Test settles", COLORS["test"], "-"),
    ("clean_zero", "Clean = 0", COLORS["clean"], ":"),
    ("noisy_zero", "Noisy = 0", COLORS["noisy"], ":"),
)


@dataclass(frozen=True)
class Events:
    clean_peak: int
    test_peak: int
    test_settle: int
    clean_zero: int
    noisy_zero: int


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zoom into the second descent with one column per seed."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--pattern", default=DEFAULT_RUN_PATTERN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xmin", type=float, default=50)
    parser.add_argument("--xmax", type=float, default=200)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def smooth(series: pd.Series) -> pd.Series:
    return series.rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()


def detect_events(metrics: pd.DataFrame) -> Events:
    metrics = metrics.set_index("epoch")
    window = metrics.loc[SEARCH_WINDOW[0] : SEARCH_WINDOW[1]]
    clean = window["train_error_clean"]
    noisy = window["train_error_noisy"]
    test = window["test_error"]

    clean_peak = int(clean.idxmax())
    test_peak = int(smooth(test).idxmax())
    final_test = metrics["test_error"].iloc[-1]
    test_settle = int(test[test > final_test + SETTLE_MARGIN].index[-1]) + 1
    after_peak = clean.loc[clean_peak:]
    clean_zero = int(after_peak[after_peak == 0].index[0])
    noisy_zero = int(noisy[noisy == 0].index[0])
    return Events(clean_peak, test_peak, test_settle, clean_zero, noisy_zero)


def plot_runs(
    runs: list[Run], output: Path, xlim: tuple[float, float], dpi: int
) -> None:
    ncols = len(runs)
    figure, axes = plt.subplots(
        nrows=len(ROW_SPECS),
        ncols=ncols,
        figsize=(4 * ncols, 8),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    event_handles = []
    for column, run in enumerate(runs):
        events = detect_events(run.metrics)
        print(f"seed {run.seed}: {events}")
        axes[0, column].set_title(f"seed = {run.seed}", fontsize=13)
        epochs = run.metrics["epoch"]
        for row, (row_label, metric, color, ylim) in enumerate(ROW_SPECS):
            axis = axes[row, column]
            if metric == "test_error":
                axis.plot(epochs, run.metrics[metric], color=color,
                          linewidth=0.8, alpha=0.35, label="Raw")
                axis.plot(epochs, smooth(run.metrics[metric]), color=color,
                          linewidth=1.6, label=f"{SMOOTHING_WINDOW}-ep MA")
            else:
                axis.plot(epochs, run.metrics[metric], color=color,
                          linewidth=1.5)
            handles = []
            for key, label, event_color, linestyle in EVENT_SPECS:
                handles.append(axis.axvline(
                    getattr(events, key),
                    color=event_color,
                    linestyle=linestyle,
                    linewidth=1.1,
                    alpha=0.85,
                    label=f"{label}",
                ))
            if not event_handles:
                event_handles = handles
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.grid(True, color="0.88", linewidth=0.7)
            if column == 0:
                axis.set_ylabel(row_label)
            if row == len(ROW_SPECS) - 1:
                axis.set_xlabel("Epoch")
        axes[0, column].text(
            0.98, 0.97,
            "\n".join(
                f"{label}: {getattr(events, key)}"
                for key, label, _, _ in EVENT_SPECS
            ),
            transform=axes[0, column].transAxes,
            ha="right", va="top", fontsize=7.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85,
                  "edgecolor": "0.8"},
        )

    test_handles = axes[0, 0].get_lines()[:2]
    axes[0, 0].legend(
        handles=test_handles, loc="lower right", fontsize=8, frameon=True
    )
    figure.legend(
        handles=event_handles,
        loc="outside lower center",
        ncols=len(EVENT_SPECS),
        fontsize=9,
        frameon=False,
    )
    figure.suptitle(
        "CNN5Layer second descent: clean / noisy train error vs test error "
        "(EMNIST Digits, 20% label noise, width = 1)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = create_parser().parse_args()
    runs = collect_runs(args.results_dir, args.pattern)
    plot_runs(runs, args.output, (args.xmin, args.xmax), args.dpi)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
