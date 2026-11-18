"""Plot model-wise double descent: final metrics vs. model width for a width-sweep.

Each run directory under the sweep is expected to contain ``config.json`` and
``metrics.csv``. Per run, the row at ``--epoch`` (default: the last epoch) is taken;
x is the resnet18k width parameter k = model_width * 64. Panel layout and style
follow ``plot_learning_curve.py`` (error / noisy-clean error / loss).

    python visualization/plot_width_sweep.py [--sweep-dir results/<sweep>] [--out OUTPUT_BASE]

OUTPUT_BASE has no extension; ``.png`` and ``.pdf`` are written next to each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_learning_curve import PanelConfig, SeriesSpec, plot_panels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results/cifar10-dd-resnet18k-width-sweep-20261113T140649Z"
BASE_WIDTH = 64  # resnet18k: --model-width = k / 64


def load_sweep(sweep_dir: Path, epoch: Optional[int]) -> pd.DataFrame:
    rows = []
    for run in sorted(p for p in sweep_dir.iterdir() if (p / "metrics.csv").is_file()):
        cfg = json.loads((run / "config.json").read_text())
        metrics = pd.read_csv(run / "metrics.csv")
        target = metrics["epoch"].max() if epoch is None else epoch
        row = metrics[metrics["epoch"] == target]
        if row.empty:
            print(f"[!] skip {run.name}: epoch {target} not found (last={metrics['epoch'].max()})")
            continue
        row = row.iloc[-1].to_dict()
        row["width"] = round(cfg["model_width"] * BASE_WIDTH, 6)
        rows.append(row)
    if not rows:
        raise SystemExit(f"no runs found under {sweep_dir}")
    df = pd.DataFrame(rows).sort_values("width").reset_index(drop=True)
    if df["epoch"].nunique() > 1:
        print(f"[!] runs are at different epochs: {sorted(df['epoch'].unique())}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="output path without extension (default: <sweep-dir>/viz/width_sweep_epoch-<E>)")
    parser.add_argument("--epoch", type=int, default=None, help="epoch to read (default: last epoch of each run)")
    parser.add_argument("--xscale", choices=["log", "linear"], default="log")
    parser.add_argument("--vlines", type=float, nargs="*", default=[], help="widths to mark with dotted lines")
    parser.add_argument("--csv-out", action="store_true", help="also write the collected table as CSV")
    args = parser.parse_args()

    df = load_sweep(args.sweep_dir, args.epoch)
    epoch_label = int(df["epoch"].max())
    out = args.out or args.sweep_dir / "viz" / f"width_sweep_epoch-{epoch_label}"

    marker = dict(marker="o", markersize=10)
    panels = [
        PanelConfig(
            series=(
                SeriesSpec("test_error", "test", color="red", linewidth=3, **marker),
                SeriesSpec("train_error", "train", color="blue", linewidth=3, **marker),
            ),
            ylabel="error",
            ylim=(0.0, None),  # upper limit from data
            grid=True,
            grid_axis="y",
            hide_x_ticks=True,
            tick_labelsize=35,
            legend_fontsize=35,
        ),
        PanelConfig(
            series=(
                SeriesSpec("train_error_noisy", "noisy", color="purple", linestyle="--", linewidth=3, **marker),
            ),
            twin_series=(
                SeriesSpec("train_error_clean", "clean", color="green", linewidth=3, **marker),
            ),
            ylabel="train noisy error",
            yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ylim=(0.0, 1.01),
            twin_ylabel="train clean error",
            twin_ylim=(0.0, None),  # upper limit from data
            hide_x_ticks=True,
            grid=True,
            grid_axis="y",
            tick_labelsize=35,
            twin_tick_labelsize=35,
            legend_fontsize=35,
        ),
        PanelConfig(
            series=(
                SeriesSpec("test_loss", "test loss", color="red", linewidth=3, **marker),
                SeriesSpec("train_loss", "train loss", color="blue", linewidth=3, **marker),
            ),
            ylabel="loss",
            xlabel="width $k$",
            grid=True,
            grid_axis="y",
            ylim=(0.0, None),  # upper limit from data
            ytick_format="%.1f",
            tick_labelsize=35,
            legend_fontsize=35,
        ),
    ]
    xlim = (df["width"].min(), df["width"].max())
    if args.xscale == "linear":
        xlim = (0, xlim[1])
    xticks = [k for k in (1, 2, 4, 8, 16, 32, 64) if xlim[0] <= k <= xlim[1]] if args.xscale == "log" else None
    panels = [replace(p, xscale=args.xscale, xlim=xlim, xticks=xticks) for p in panels]

    saved = plot_panels(
        df,
        panels,
        out,
        vertical_lines=args.vlines,
        figure_size=(13, 18),
        font_family="Times New Roman",
        font_size=35,
        x="width",
    )
    if args.csv_out:
        csv_path = Path(out).with_name(f"{Path(out).name}.csv")
        df.to_csv(csv_path, index=False)
        saved.append(str(csv_path))
    print(f"[✓] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
