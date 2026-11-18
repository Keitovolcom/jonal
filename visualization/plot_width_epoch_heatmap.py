"""Width x epoch heatmaps of a width-sweep (Nakkiran et al., 2019, Fig. 2 style).

x = resnet18k width parameter k, y = epoch (log), color = metric.

Untrained widths (e.g. 13, 15, 17, ...) are not interpolated: each trained width
is drawn as a cell spanning halfway to its neighbours (midpoints for a linear
x axis, geometric midpoints for a log x axis). Trained widths are marked with
minor ticks on the x axis so the gaps stay visible.

Each panel's color range is [min, 95th percentile] over its cells: the first few
epochs (error ~0.9) would otherwise squeeze the double-descent region into one
color. Cells above the cap are drawn in the top color (arrow on the colorbar).

All metrics go into one figure (3 x 2 grid, same order as the learning-curve panels:
error / train noisy-clean error / loss), each panel with its own colorbar.

    python visualization/plot_width_epoch_heatmap.py [--sweep-dir results/<sweep>] [--metrics test_error ...]

Writes ``<out-dir>/heatmaps.{png,pdf}`` (linear width axis) and
``<out-dir>/heatmaps_logx.{png,pdf}`` (log width axis); default out-dir: ``<sweep-dir>/viz``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_learning_curve import _resolve_font  # noqa: E402
from plot_width_sweep_curves import SWEEP_DIR, load_runs  # noqa: E402

METRIC_LABELS = {
    "test_error": "test error",
    "train_error": "train error",
    "train_error_noisy": "train noisy error",
    "train_error_clean": "train clean error",
    "test_loss": "test loss",
    "train_loss": "train loss",
}


def cell_edges(centers: np.ndarray, log: bool) -> np.ndarray:
    """Edges halfway between neighbouring centers; the outer cells mirror their inner half."""
    c = np.log(centers) if log else centers.astype(float)
    mid = (c[1:] + c[:-1]) / 2
    first = c[0] - (mid[0] - c[0]) if len(c) > 1 else c[0] - 0.5
    last = c[-1] + (c[-1] - mid[-1]) if len(c) > 1 else c[-1] + 0.5
    edges = np.concatenate([[first], mid, [last]])
    return np.exp(edges) if log else edges


def build_grid(runs, metric: str):
    """(widths, epochs, values[epoch, width]) for epochs >= 1 (epoch 0 has no place on a log axis)."""
    table = pd.concat(
        [df[["epoch", metric]].assign(width=k) for k, df in runs]
    ).pivot(index="epoch", columns="width", values=metric)
    table = table[table.index >= 1].sort_index()
    missing = int(table.isna().sum().sum())
    if missing:
        print(f"[!] {metric}: {missing} (epoch, width) cells missing; drawn blank")
    return table.columns.to_numpy(float), table.index.to_numpy(float), table.to_numpy()


def draw_heatmap(fig, ax, runs, metric: str, xscale: str = "linear",
                 vmax_percentile: float = 95.0, cmap: str = "plasma",
                 show_xlabel: bool = True, show_ylabel: bool = True,
                 vmin: float | None = None, vmax: float | None = None,
                 cbar_label: str | None = None) -> None:
    """``vmin`` / ``vmax`` override the per-panel color range (e.g. to share it across panels)."""
    widths, epochs, values = build_grid(runs, metric)
    x_edges = cell_edges(widths, log=(xscale == "log"))
    y_edges = cell_edges(epochs, log=True)
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if vmin is None else vmin
    vmax = float(np.percentile(finite, vmax_percentile)) if vmax is None else vmax

    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values), cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    if xscale == "log":
        major = [k for k in (1, 2, 4, 8, 16, 32, 64) if x_edges[0] <= k <= x_edges[-1]]
    else:
        major = [k for k in (1, 10, 20, 30, 40, 50, 60) if x_edges[0] <= k <= x_edges[-1]]
    ax.set_xticks(major)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xticks(widths, minor=True)  # trained widths
    ax.tick_params(axis="x", which="minor", length=8, width=1.2, labelbottom=False)
    if show_xlabel:
        ax.set_xlabel("width $k$")
    if show_ylabel:
        ax.set_ylabel("epoch")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, extend="max" if finite.max() > vmax else "neither")
    cbar.set_label(cbar_label or METRIC_LABELS.get(metric, metric))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f" if "loss" in metric else "%.2f"))


def plot_heatmaps(runs, metrics, output_base: Path, xscale: str = "linear", ncols: int = 2,
                  vmax_percentile: float = 95.0, cmap: str = "plasma", panel_size=(14, 10),
                  font_size: float = 35, formats=("png", "pdf"), dpi: int = 200) -> list:
    nrows = math.ceil(len(metrics) / ncols)
    rc = {"font.family": _resolve_font("Times New Roman"), "font.size": font_size, "mathtext.fontset": "stix"}
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size[0], nrows * panel_size[1]),
                                 squeeze=False, layout="constrained")
        for i, ax in enumerate(axes.flat):
            if i >= len(metrics):
                ax.set_visible(False)
                continue
            row, col = divmod(i, ncols)
            draw_heatmap(fig, ax, runs, metrics[i], xscale=xscale, vmax_percentile=vmax_percentile,
                         cmap=cmap, show_xlabel=(row == nrows - 1 or i + ncols >= len(metrics)),
                         show_ylabel=(col == 0))
        for ext in formats:
            out = output_base.with_name(f"{output_base.name}.{ext}")
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            saved.append(str(out))
        plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <sweep-dir>/viz")
    parser.add_argument("--metrics", nargs="+", default=list(METRIC_LABELS), choices=list(METRIC_LABELS))
    parser.add_argument("--xscale", choices=["linear", "log", "both"], default="both")
    parser.add_argument("--vmax-percentile", type=float, default=95.0,
                        help="per-panel color cap as a percentile of that panel's values")
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--cmap", default="plasma")
    args = parser.parse_args()

    runs = load_runs(args.sweep_dir)
    out_dir = args.out_dir or args.sweep_dir / "viz"
    xscales = ["linear", "log"] if args.xscale == "both" else [args.xscale]
    for xscale in xscales:
        suffix = "" if xscale == "linear" else "_logx"
        saved = plot_heatmaps(runs, args.metrics, out_dir / f"heatmaps{suffix}", xscale=xscale,
                              ncols=args.ncols, vmax_percentile=args.vmax_percentile, cmap=args.cmap)
        print(f"[✓] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
