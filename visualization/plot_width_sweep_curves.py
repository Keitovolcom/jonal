"""Epoch-wise learning curves for every width of a width-sweep.

Writes (1) one figure per width in the style of ``plot_learning_curve.py`` and
(2) a single grid figure with all widths (one 3-panel cell per width, y limits
shared across widths so cells are comparable).

    python visualization/plot_width_sweep_curves.py [--sweep-dir results/<sweep>] [--out-dir DIR]

Outputs (``.png`` and ``.pdf``), under ``<sweep-dir>/viz`` by default:
    learning_curves/learning_curve_k-<k>.*   per width
    learning_curves_all.*                    all widths
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_learning_curve import (  # noqa: E402
    _draw_panel,
    _resolve_font,
    learning_curve_panels,
    nice_ylim,
    plot_panels,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results/cifar10-dd-resnet18k-width-sweep-20261113T140649Z"
BASE_WIDTH = 64  # resnet18k: --model-width = k / 64


def load_runs(sweep_dir: Path) -> list:
    """[(k, metrics DataFrame)] sorted by width k."""
    runs = []
    for run in sweep_dir.iterdir():
        if not (run / "metrics.csv").is_file():
            continue
        cfg = json.loads((run / "config.json").read_text())
        runs.append((round(cfg["model_width"] * BASE_WIDTH, 6), pd.read_csv(run / "metrics.csv")))
    if not runs:
        raise SystemExit(f"no runs found under {sweep_dir}")
    return sorted(runs, key=lambda r: r[0])


def fmt_k(k: float) -> str:
    return f"{k:g}"


def plot_per_width(runs, panels, out_dir: Path, vlines: Sequence[float]) -> list:
    saved = []
    for k, df in runs:
        saved += plot_panels(
            df,
            panels,
            out_dir / (f"learning_curve_k-{int(k):02d}" if float(k).is_integer() else f"learning_curve_k-{k:g}"),
            vertical_lines=vlines,
            figure_size=(13, 18),
            font_family="Times New Roman",
            font_size=35,
        )
    return saved


def _shared_ylims(runs, panels, xlim) -> list:
    """Fix every "upper limit from data" ylim to the max over all widths."""
    visible = pd.concat([df[(df["epoch"] >= xlim[0]) & (df["epoch"] <= xlim[1])] for _, df in runs])
    out = []
    for p in panels:
        if p.ylim is not None and p.ylim[1] is None:
            ticks, hi = nice_ylim(p.ylim[0], max(visible[s.column].max() for s in p.series))
            p = replace(p, ylim=(p.ylim[0], hi), yticks=ticks)
        if p.twin_series and p.twin_ylim is not None and p.twin_ylim[1] is None:
            ticks, hi = nice_ylim(p.twin_ylim[0], max(visible[s.column].max() for s in p.twin_series))
            p = replace(p, twin_ylim=(p.twin_ylim[0], hi), twin_yticks=ticks)
        out.append(p)
    return out


def _scale_series(specs, linewidth: float):
    return tuple(replace(s, linewidth=linewidth) for s in specs)


def plot_grid(runs, panels, output_base: Path, ncols: int, vlines: Sequence[float],
              font_size: float = 18, linewidth: float = 1.8,
              cell_size: tuple = (6.5, 9.0), formats=("png", "pdf"), dpi: int = 150) -> list:
    nrows = math.ceil(len(runs) / ncols)
    panels = [
        replace(p, tick_labelsize=None, twin_tick_labelsize=None, legend_fontsize=None,
                series=_scale_series(p.series, linewidth),
                twin_series=_scale_series(p.twin_series, linewidth))
        for p in panels
    ]
    rc = {"font.family": _resolve_font("Times New Roman"), "font.size": font_size, "mathtext.fontset": "stix"}
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(ncols * cell_size[0], nrows * cell_size[1]), layout="constrained")
        subfigs = fig.subfigures(nrows, ncols, squeeze=False)
        for i, sf in enumerate(subfigs.flat):
            if i >= len(runs):
                sf.set_visible(False)
                continue
            k, df = runs[i]
            row, col = divmod(i, ncols)
            axes = sf.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
            for ax, p in zip(axes, panels):
                p = replace(
                    p,
                    ylabel=p.ylabel if col == 0 else None,
                    # labels only on the outer column/row so every cell has the same size
                    twin_ylabel=p.twin_ylabel if col == ncols - 1 else None,
                    xlabel=p.xlabel if row == nrows - 1 else None,
                    legend=(i == 0),
                )
                _draw_panel(ax, df, p, vlines)
            sf.suptitle(f"$k = {fmt_k(k)}$", fontsize=font_size * 1.3)
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
    parser.add_argument("--vlines", type=float, nargs="*", default=[], help="epochs to mark with dotted lines")
    parser.add_argument("--xmax", type=float, default=None, help="upper x limit (epoch); default: last epoch")
    parser.add_argument("--ncols", type=int, default=6, help="columns of the all-widths grid")
    parser.add_argument("--skip-per-width", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()

    runs = load_runs(args.sweep_dir)
    out_dir = args.out_dir or args.sweep_dir / "viz"
    xmax = args.xmax if args.xmax is not None else float(max(df["epoch"].max() for _, df in runs))
    panels = [replace(p, xlim=(1, xmax)) for p in learning_curve_panels()]

    if not args.skip_per_width:
        saved = plot_per_width(runs, panels, out_dir / "learning_curves", args.vlines)
        print(f"[✓] Saved {len(saved)} files under {out_dir / 'learning_curves'}")
    if not args.skip_grid:
        saved = plot_grid(runs, _shared_ylims(runs, panels, (1, xmax)), out_dir / "learning_curves_all",
                          args.ncols, args.vlines)
        print(f"[✓] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
