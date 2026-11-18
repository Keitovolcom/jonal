"""Overlay the error curves of every width of a width-sweep in one figure.

One figure, 2x2 panels (test error / train error / train clean error /
train noisy error), x = epoch, every width k drawn in each panel with a
viridis color gradient over k (log-normalized) and a shared colorbar.

    python visualization/plot_width_sweep_errors_overlay.py \
        [--sweep-dir results/<sweep>] [--xscale log|linear]

Outputs ``<sweep-dir>/viz/width_sweep_errors_overlay_<xscale>x.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results/cifar10-dd-resnet18k-width-sweep-20261113T140649Z"
BASE_WIDTH = 64  # resnet18k: --model-width = k / 64

PANELS = (
    ("test_error", "Test error"),
    ("train_error", "Train error"),
    ("train_error_clean", "Train error (clean)"),
    ("train_error_noisy", "Train error (noisy)"),
)


def load_runs(sweep_dir: Path) -> list[tuple[float, pd.DataFrame]]:
    runs = []
    for run in sweep_dir.iterdir():
        if not (run / "metrics.csv").is_file():
            continue
        cfg = json.loads((run / "config.json").read_text())
        k = round(cfg["model_width"] * BASE_WIDTH, 6)
        runs.append((k, pd.read_csv(run / "metrics.csv")))
    if not runs:
        raise SystemExit(f"no runs found under {sweep_dir}")
    return sorted(runs, key=lambda r: r[0])


def plot(runs, out_base: Path, xscale: str, cmap_name: str) -> list[str]:
    ks = [k for k, _ in runs]
    norm = LogNorm(vmin=min(ks), vmax=max(ks))
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, PANELS):
        for k, df in runs:  # small k first, large k drawn on top
            d = df[df["epoch"] >= 1] if xscale == "log" else df
            ax.plot(d["epoch"], d[col], color=cmap(norm(k)), linewidth=1.0, alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel("Error")
        ax.set_ylim(bottom=0)
        ax.set_xscale(xscale)
        ax.grid(True, which="major", color="0.88", linewidth=0.6)
        if xscale == "log":
            ax.grid(True, which="minor", color="0.94", linewidth=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    xmin = 1 if xscale == "log" else 0
    xmax = max(df["epoch"].max() for _, df in runs)
    for ax in axes[1]:
        ax.set_xlabel("Epoch" + (" (log scale)" if xscale == "log" else ""))
        ax.set_xlim(xmin, xmax)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.9, pad=0.015, aspect=35)
    ticks = [t for t in (1, 2, 4, 8, 16, 32, 64) if min(ks) <= t <= max(ks)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}" for t in ticks])
    cbar.minorticks_off()
    cbar.set_label("ResNet18 width k")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ("png", "pdf"):
        out = out_base.with_name(f"{out_base.name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        saved.append(str(out))
    plt.close(fig)
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <sweep-dir>/viz")
    p.add_argument("--xscale", choices=("log", "linear"), default="log")
    p.add_argument("--cmap", default="viridis")
    a = p.parse_args()

    runs = load_runs(a.sweep_dir)
    out_dir = a.out_dir or a.sweep_dir / "viz"
    out_base = out_dir / f"width_sweep_errors_overlay_{a.xscale}x"
    for path in plot(runs, out_base, a.xscale, a.cmap):
        print(path)


if __name__ == "__main__":
    main()
