"""Plot EMNIST learning curves (error / noisy-clean error / loss) from a run's metrics.csv.

    python visualization/plot_learning_curve.py [--csv results/<run>/metrics.csv] [--out OUTPUT_BASE]

OUTPUT_BASE has no extension; ``.png`` and ``.pdf`` are written next to each other.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / (
    "results/emnist-digits_cnn5layer_noise-0p2_lr-0p01_loss-crossentropy_bs-256_"
    "opt-sgd_width-1_ch-1_aug-none_seed-42"
)
FILE_PATH = RUN_DIR / "metrics.csv"
OUTPUT_BASE = RUN_DIR / "viz" / "learning_curve"
VERTICAL_EPOCHS = [47, 75]

# Times New Roman is not installed in the jonal-dev container; fall back to Times-like fonts.
SERIF_FALLBACKS = ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif")


# --------------------------------------------------------------------------- #
# Panel helpers (port of visualizations.training_curves)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeriesSpec:
    column: str
    label: str
    color: Optional[str] = None
    linestyle: str = "-"
    linewidth: float = 1.5
    marker: Optional[str] = None
    markersize: float = 6.0


@dataclass(frozen=True)
class PanelConfig:
    series: Sequence[SeriesSpec]
    ylabel: Optional[str] = None
    xlabel: Optional[str] = None
    ylim: Optional[tuple] = None
    yticks: Optional[Sequence[float]] = None
    ytick_format: Optional[str] = None
    xscale: str = "linear"
    xlim: Optional[tuple] = None
    xticks: Optional[Sequence[float]] = None
    grid: bool = False
    grid_axis: str = "both"
    hide_x_ticks: bool = False
    tick_labelsize: Optional[float] = None
    legend_fontsize: Optional[float] = None
    twin_series: Sequence[SeriesSpec] = ()
    twin_ylabel: Optional[str] = None
    twin_ylim: Optional[tuple] = None
    twin_yticks: Optional[Sequence[float]] = None
    twin_tick_labelsize: Optional[float] = None
    legend: bool = True


def load_training_metrics_from_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _resolve_font(family: str) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in (family, *SERIF_FALLBACKS):
        if name in available:
            return name
    return "serif"


def _draw_series(ax, df: pd.DataFrame, specs: Sequence[SeriesSpec], x: str = "epoch"):
    return [
        ax.plot(df[x], df[s.column], label=s.label, color=s.color,
                linestyle=s.linestyle, linewidth=s.linewidth,
                marker=s.marker, markersize=s.markersize)[0]
        for s in specs
    ]


def nice_ylim(lo: float, data_max: float):
    """Ticks from ``lo`` up to the first "nice" tick >= data_max, and an upper limit
    padded slightly above that tick (like the hand-tuned 0.301 / 1.01)."""
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, data_max)
    ticks = [t for t in ticks if t >= lo]
    top = min(t for t in ticks if t >= data_max - 1e-12)
    ticks = [t for t in ticks if t <= top + 1e-12]
    return ticks, top + (top - lo) * 0.005


def _apply_ylim(ax, df, specs, ylim, yticks, xlim, x: str = "epoch") -> None:
    """Set y limits/ticks. ``ylim=(lo, None)`` picks the upper limit from the data:
    the max of the plotted series within ``xlim`` is rounded up to the next "nice" tick
    and padded slightly (like the hand-tuned 0.301 / 1.01) so the top tick and line show."""
    if ylim is None:
        if yticks is not None:
            ax.set_yticks(yticks)
        return
    lo, hi = ylim
    if hi is None:
        visible = df
        if xlim is not None:
            visible = df[(df[x] >= xlim[0]) & (df[x] <= xlim[1])]
        data_max = max(visible[s.column].max() for s in specs)
        ticks, hi = nice_ylim(lo, data_max)
        if yticks is None:
            yticks = ticks
    ax.set_ylim(lo, hi)
    if yticks is not None:
        ax.set_yticks(yticks)


def _draw_panel(ax, df: pd.DataFrame, cfg: PanelConfig, vertical_lines: Sequence[float],
                x: str = "epoch") -> None:
    handles = _draw_series(ax, df, cfg.series, x)

    ax.set_xscale(cfg.xscale)
    if cfg.xlim is not None:
        ax.set_xlim(*cfg.xlim)
    if cfg.xticks is not None:
        ax.set_xticks(cfg.xticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    _apply_ylim(ax, df, cfg.series, cfg.ylim, cfg.yticks, cfg.xlim, x)
    if cfg.ytick_format is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(cfg.ytick_format))
    if cfg.ylabel:
        ax.set_ylabel(cfg.ylabel)
    if cfg.xlabel:
        ax.set_xlabel(cfg.xlabel)
    if cfg.grid:
        ax.grid(True, axis=cfg.grid_axis)
    if cfg.tick_labelsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=cfg.tick_labelsize)
    if cfg.hide_x_ticks:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    for epoch in vertical_lines:
        ax.axvline(epoch, color="black", linestyle=":", linewidth=2.5, zorder=0)

    legend_ax = ax
    if cfg.twin_series:
        twin = ax.twinx()
        handles += _draw_series(twin, df, cfg.twin_series, x)
        _apply_ylim(twin, df, cfg.twin_series, cfg.twin_ylim, cfg.twin_yticks, cfg.xlim, x)
        if cfg.twin_ylabel:
            twin.set_ylabel(cfg.twin_ylabel)
        if cfg.twin_tick_labelsize is not None:
            twin.tick_params(axis="y", labelsize=cfg.twin_tick_labelsize)
        legend_ax = twin  # draw the legend above both axes' lines

    if cfg.legend:
        legend_ax.legend(handles, [h.get_label() for h in handles], fontsize=cfg.legend_fontsize)


def plot_panels(
    df: pd.DataFrame,
    panels: Sequence[PanelConfig],
    output_base: Path,
    vertical_lines: Sequence[float] = (),
    figure_size: tuple = (13, 18),
    font_family: str = "Times New Roman",
    font_size: float = 12,
    formats: Sequence[str] = ("png", "pdf"),
    x: str = "epoch",
) -> list:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    rc = {"font.family": _resolve_font(font_family), "font.size": font_size, "mathtext.fontset": "stix"}
    saved = []
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(panels), 1, figsize=figure_size, sharex=True, squeeze=False)
        for ax, cfg in zip(axes[:, 0], panels):
            _draw_panel(ax, df, cfg, vertical_lines, x)
        fig.tight_layout()
        for ext in formats:
            out = output_base.with_name(f"{output_base.name}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            saved.append(str(out))
        plt.close(fig)
    return saved


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def learning_curve_panels() -> list:
    """The 3-panel layout (error / train noisy-clean error / loss) over epochs.

    metrics.csv stores errors as fractions in [0, 1]. xlim upper is left as None;
    callers set it (e.g. last epoch) with ``dataclasses.replace``."""
    panels = [
        PanelConfig(
            series=(
                SeriesSpec("test_error", "test", color="red", linewidth=3),
                SeriesSpec("train_error", "train", color="blue", linewidth=3),
            ),
            ylabel="error",
            ylim=(0.0, None),  # upper limit from data
            xscale="log",
            xlim=(1, None),  # upper = last epoch (or --xmax), set in main()
            grid=True,
            grid_axis="y",
            hide_x_ticks=True,
            tick_labelsize=35,
            legend_fontsize=35,
        ),
        PanelConfig(
            series=(
                SeriesSpec("train_error_noisy", "noisy", color="purple", linestyle="--", linewidth=3),
            ),
            xlim=(1, None),  # upper = last epoch (or --xmax), set in main()
            twin_series=(
                SeriesSpec("train_error_clean", "clean", color="green", linewidth=3),
            ),
            ylabel="train noisy error",
            yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ylim=(0.0, 1.01),
            twin_ylabel="train clean error",
            twin_ylim=(0.0, None),  # upper limit from data
            xscale="log",
            hide_x_ticks=True,
            grid=True,
            grid_axis="y",
            tick_labelsize=35,
            twin_tick_labelsize=35,
            legend_fontsize=35,
        ),
        PanelConfig(
            series=(
                SeriesSpec("test_loss", "test loss", color="red", linewidth=3),
                SeriesSpec("train_loss", "train loss", color="blue", linewidth=3),
            ),
            ylabel="loss",
            xlabel="epoch",
            xscale="log",
            grid=True,
            grid_axis="y",
            xlim=(1, None),  # upper = last epoch (or --xmax), set in main()
            ylim=(0.0, None),  # upper limit from data
            ytick_format="%.1f",
            tick_labelsize=35,
            legend_fontsize=35,
        ),
    ]
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=FILE_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_BASE, help="output path without extension")
    parser.add_argument("--vlines", type=int, nargs="*", default=VERTICAL_EPOCHS)
    parser.add_argument("--xmax", type=float, default=None,
                        help="upper x limit (epoch) for all panels; default: last epoch in the CSV")
    args = parser.parse_args()

    df = load_training_metrics_from_csv(args.csv)

    panels = learning_curve_panels()

    xmax = args.xmax if args.xmax is not None else float(df["epoch"].max())
    panels = [replace(p, xlim=(1, xmax)) for p in panels]

    saved = plot_panels(
        df,
        panels,
        args.out,
        vertical_lines=args.vlines,
        figure_size=(13, 18),
        font_family="Times New Roman",
        font_size=35,
    )
    print(f"[✓] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
