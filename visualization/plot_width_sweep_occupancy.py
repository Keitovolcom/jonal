"""Predicted-label occupancy on the interpolation lines for every width of a width-sweep.

For every pair, epoch and width, the points of the line are split by their
predicted label (``interpolation_metrics.label_occupancy``):
    original  the original (clean) label of x0; x0 and x1 share it in clean-clean / test-test
    noisy     the current (noisy) training label of x1, clean-noisy pairs only
    others    any other label
and the fractions are averaged over the pairs.

Writes, under ``<sweep-dir>/viz/occupancy`` by default:
    cache/occupancy_k-<k>.npz                  per-epoch mean/std over pairs, per mode and alpha range
    occupancy.csv                              k, epoch, mean/std per mode, alpha range and label kind
    width_sweep_occupancy_epoch-<E>.{png,pdf,csv}
                                               occupancy at epoch E vs width k, one row per mode
    occupancy_all.{png,pdf}                    all widths, epoch-wise (like learning_curves_all)
    epoch_occupancy_k-<k>.{png,pdf}            one width (--epoch-curve-k, default 64), epoch-wise

    python visualization/plot_width_sweep_occupancy.py [--alpha-range all|in] [--epoch 1000]

``--alpha-range all`` uses the whole line (alpha in [-0.5, 1.5]), ``in`` only alpha in [0, 1].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpolation_metrics import label_occupancy  # noqa: E402
from plot_learning_curve import _resolve_font  # noqa: E402
from plot_width_sweep_curves import load_runs  # noqa: E402
from plot_width_sweep_spatial import ALPHA_RANGES, H5_ROOT, MODES, SWEEP_DIR, h5_dir  # noqa: E402

# (key, legend label, color): colors as in interpolation_metrics' occupancy figure.
KINDS = (
    ("original", "original label", "#1f3fbf"),
    ("noisy", "noisy label", "#d62728"),
    ("other", "others label", "#2ca02c"),
)
PANEL_ORDER = ("no_noise", "noise", "test_test_no_noise")  # clean-clean, clean-noisy, test-test


def pair_labels(log: h5py.File, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(original label of x0, noisy training label of x1 or -1) per pair, from pair_names."""
    names = [name.decode().split("/")[1].split("_") for name in log["pair_names"][:]]
    original = np.array([int(a) for a, _ in names])
    x1_label = np.array([int(b) for _, b in names])
    if mode == "noise":
        if (original == x1_label).any():
            raise ValueError("clean-noisy pair whose x1 label equals the x0 label")
        return original, x1_label
    if (original != x1_label).any():
        raise ValueError(f"{mode}: pair whose labels differ")
    return original, np.full_like(original, -1)


def occupancy_stats(path: Path, mode: str) -> dict[str, np.ndarray]:
    """Mean/std over pairs of each occupancy kind, [epochs], for both alpha ranges."""
    with h5py.File(path, "r") as log:
        if not log["present"][:].all():
            raise ValueError(f"{path} has missing (pair, epoch) entries")
        original, noisy = pair_labels(log, mode)
        preds = log["predicted_label"][:]
        alphas = log["alpha"][:]
        epochs = log["epochs"][:]
    i0, i1 = np.argmin(np.abs(alphas - 0.0)), np.argmin(np.abs(alphas - 1.0))
    out = {"epochs": epochs}
    for rng, sub in (("all", preds), ("in", preds[..., i0:i1 + 1])):
        for kind, rate in label_occupancy(sub, original, noisy).items():
            if mode != "noise" and kind == "noisy":
                continue
            out[f"{rng}_{mode}_{kind}"] = rate.mean(axis=0)
            out[f"{rng}_{mode}_{kind}_std"] = rate.std(axis=0)
    return out


def load_occupancy(sweep_dir: Path, h5_root: Path, cache_dir: Path, ks) -> pd.DataFrame:
    """Long table: k, epoch, occ_<range>_<mode>_<kind>{,_std}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in ks:
        cache = cache_dir / f"occupancy_k-{k:g}.npz"
        if cache.is_file():
            arrays = dict(np.load(cache))
        else:
            arrays = {}
            for mode, _, _ in MODES:
                arrays.update(occupancy_stats(h5_dir(h5_root, sweep_dir, k) / f"{mode}.h5", mode))
            np.savez_compressed(cache, **arrays)
            print(f"k={k:g}: cached {cache}", flush=True)
        frame = {"k": k, "epoch": arrays.pop("epochs")}
        frame.update({f"occ_{key}": value for key, value in arrays.items()})
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)


def mode_kinds(mode: str):
    return [kind for kind in KINDS if mode == "noise" or kind[0] != "noisy"]


def alpha_text(rng: str) -> str:
    return r"$\alpha\in[-0.5,1.5]$" if rng == "all" else r"$\alpha\in[0,1]$"


def draw_mode(ax, df: pd.DataFrame, x: str, rng: str, mode: str, xlim, xticks=None,
              linewidth: float = 3, marker: dict | None = None, vlines=()) -> None:
    """Occupancy of each label kind of one mode vs ``x`` (log axis), y in [0, 1]."""
    for kind, label, color in mode_kinds(mode):
        ax.plot(df[x], df[f"occ_{rng}_{mode}_{kind}"], label=label, color=color,
                linewidth=linewidth, **(marker or {}))
    for value in vlines:
        ax.axvline(value, color="black", linestyle=":", linewidth=2.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_ylim(0.0, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y")


def save(fig, output_base: Path, formats=("png", "pdf"), dpi: int = 200) -> list:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in formats:
        out = output_base.with_name(f"{output_base.name}.{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        saved.append(str(out))
    plt.close(fig)
    return saved


def plot_occupancy_rows(df: pd.DataFrame, x: str, rng: str, output_base: Path, vlines=(),
                        panel_size=(13, 7)) -> list:
    """One row per mode (PANEL_ORDER) against ``x``: "width" (with markers) or "epoch"."""
    if x == "width":
        values = df["width"].to_numpy()
        xlim = (values.min(), values.max())
        xticks = [k for k in (1, 2, 4, 8, 16, 32, 64) if xlim[0] <= k <= xlim[1]]
        xlabel, marker = "width $k$", dict(marker="o", markersize=10)
    else:
        xlim, xticks = (1, df["epoch"].max()), None
        xlabel, marker = "epoch", None
    rc = {"font.family": _resolve_font("Times New Roman"), "font.size": 35,
          "mathtext.fontset": "stix"}
    modes = {mode: label for mode, label, _ in MODES}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(PANEL_ORDER), 1,
                                 figsize=(panel_size[0], len(PANEL_ORDER) * panel_size[1]),
                                 sharex=True, squeeze=False, layout="constrained")
        for i, (ax, mode) in enumerate(zip(axes[:, 0], PANEL_ORDER)):
            draw_mode(ax, df, x, rng, mode, xlim, xticks, marker=marker, vlines=vlines)
            ax.set_title(modes[mode])
            if i == len(PANEL_ORDER) - 1:
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=30)
        fig.supylabel(f"predicted label occupancy ratio ({alpha_text(rng)})")
        return save(fig, output_base)


def plot_occupancy_grid(occ: pd.DataFrame, rng: str, output_base: Path, ncols: int = 6,
                        font_size: float = 18, cell_size=(6.5, 9.0)) -> list:
    """All widths, epoch-wise, like learning_curves_all: one 3-row cell (PANEL_ORDER) per k."""
    ks = sorted(occ["k"].unique())
    nrows = -(-len(ks) // ncols)
    xlim = (1, occ["epoch"].max())
    modes = {mode: label for mode, label, _ in MODES}
    rc = {"font.family": _resolve_font("Times New Roman"), "font.size": font_size,
          "mathtext.fontset": "stix"}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(ncols * cell_size[0], nrows * cell_size[1]), layout="constrained")
        subfigs = fig.subfigures(nrows, ncols, squeeze=False)
        for i, sf in enumerate(subfigs.flat):
            if i >= len(ks):
                sf.set_visible(False)
                continue
            row, col = divmod(i, ncols)
            df = occ[occ["k"] == ks[i]]
            axes = sf.subplots(len(PANEL_ORDER), 1, sharex=True, squeeze=False)[:, 0]
            for j, (ax, mode) in enumerate(zip(axes, PANEL_ORDER)):
                draw_mode(ax, df, "epoch", rng, mode, xlim, linewidth=1.8)
                if col == 0:
                    ax.set_ylabel(f"{modes[mode]}\noccupancy ratio")
                if j == len(PANEL_ORDER) - 1 and row == nrows - 1:
                    ax.set_xlabel("epoch")
                if i == 0:
                    ax.legend(fontsize=font_size * 0.8)
            sf.suptitle(f"$k = {ks[i]:g}$", fontsize=font_size * 1.3)
        fig.suptitle(f"predicted label occupancy ratio ({alpha_text(rng)})",
                     fontsize=font_size * 1.6)
        return save(fig, output_base, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--h5-root", type=Path, default=H5_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <sweep-dir>/viz/occupancy")
    parser.add_argument("--alpha-range", choices=ALPHA_RANGES, default="all")
    parser.add_argument("--epoch", type=int, default=1000, help="epoch of the width-sweep figure")
    parser.add_argument("--vlines", type=float, nargs="*", default=[14],
                        help="widths k marked with dotted lines in the width-sweep figure")
    parser.add_argument("--ncols", type=int, default=6, help="columns of the all-widths grid")
    parser.add_argument("--epoch-curve-k", type=float, nargs="*", default=[64],
                        help="widths k drawn epoch-wise in epoch_occupancy_k-<k>.*")
    args = parser.parse_args()

    out_dir = args.out_dir or args.sweep_dir / "viz" / "occupancy"
    runs = load_runs(args.sweep_dir)
    occ = load_occupancy(args.sweep_dir, args.h5_root, out_dir / "cache", [k for k, _ in runs])
    occ.to_csv(out_dir / "occupancy.csv", index=False)
    suffix = "" if args.alpha_range == "all" else "_alpha-in"

    rows = occ[occ["epoch"] == args.epoch].rename(columns={"k": "width"})
    if len(rows) != len(runs):
        raise SystemExit(f"epoch {args.epoch} missing for some widths")
    rows = rows.sort_values("width").reset_index(drop=True)
    out = out_dir / f"width_sweep_occupancy_epoch-{args.epoch}{suffix}"
    saved = plot_occupancy_rows(rows, "width", args.alpha_range, out, vlines=args.vlines)
    rows.to_csv(out.with_suffix(".csv"), index=False)
    print(f"[✓] Saved: {', '.join(saved + [str(out.with_suffix('.csv'))])}")

    # Epoch-wise: all widths in one grid, plus single widths (--epoch-curve-k).
    epochs_occ = occ[occ["epoch"] >= 1]  # epoch 0 has no place on a log axis
    saved = plot_occupancy_grid(epochs_occ, args.alpha_range,
                                out_dir / f"occupancy_all{suffix}", ncols=args.ncols)
    print(f"[✓] Saved: {', '.join(saved)}")
    for k in args.epoch_curve_k:
        df = epochs_occ[epochs_occ["k"] == k]
        if df.empty:
            raise SystemExit(f"width k={k:g} not in the sweep")
        saved = plot_occupancy_rows(df, "epoch", args.alpha_range,
                                    out_dir / f"epoch_occupancy_k-{k:g}{suffix}")
        print(f"[✓] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()
