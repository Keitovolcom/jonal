"""Spatial instability INST_S for every width of a width-sweep (alpha_h5 interpolation).

For every width k and pair mode, INST_S(pair, epoch) is the number of
predicted-label changes between neighbouring alphas on the interpolation line
(``interpolation_metrics.spatial_instability``), averaged over the pairs.

Writes, under ``<sweep-dir>/viz/spatial_instability`` by default:
    cache/inst_s_k-<k>.npz                  per-pair INST_S [pairs, epochs] per mode
    spatial_instability.csv                 k, epoch, mean/std INST_S per mode and alpha range
    spatial_instability_all.{png,pdf}       all widths, epoch-wise (like learning_curves_all)
    width_sweep_spatial_epoch-<E>.{png,pdf,csv}
                                            INST_S at epoch E vs width k (model-wise)

    python visualization/plot_width_sweep_spatial.py [--alpha-range all|in] [--epoch 1000]

``--alpha-range all`` counts changes over the whole line (alpha in [-0.5, 1.5]),
``in`` only between the two endpoints (alpha in [0, 1]).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpolation_metrics import spatial_instability  # noqa: E402
from plot_learning_curve import (  # noqa: E402
    PanelConfig,
    SeriesSpec,
    learning_curve_panels,
    nice_ylim,
    plot_panels,
)
from plot_learning_curve import _resolve_font  # noqa: E402
from plot_width_epoch_heatmap import draw_heatmap  # noqa: E402
from plot_width_sweep_curves import _shared_ylims, load_runs, plot_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results/cifar10-dd-resnet18k-width-sweep-20261113T140649Z"
H5_ROOT = Path("/mnt/ml-data/kobayashi/alpha_h5")
# (h5 file stem, legend label, color): colors follow learning_curve_panels
# (noisy = purple, clean = green, test = red).
MODES = (
    ("noise", "clean-noisy", "purple"),
    ("no_noise", "clean-clean", "green"),
    ("test_test_no_noise", "test-test", "red"),
)
ALPHA_RANGES = ("all", "in")


def h5_dir(h5_root: Path, sweep_dir: Path, k: float) -> Path:
    cfg = json.loads(next(sweep_dir.glob("*/config.json")).read_text())
    return h5_root / sweep_dir.name / cfg["dataset"] / f"{cfg['label_noise']:g}" / f"{k:g}"


def inst_s_per_pair(path: Path) -> dict[str, np.ndarray]:
    """INST_S [pairs, epochs] over the whole line ("all") and inside [0, 1] ("in")."""
    with h5py.File(path, "r") as log:
        if not log["present"][:].all():
            raise ValueError(f"{path} has missing (pair, epoch) entries")
        preds = log["predicted_label"][:]
        alphas = log["alpha"][:]
        epochs = log["epochs"][:]
    i0, i1 = np.argmin(np.abs(alphas - 0.0)), np.argmin(np.abs(alphas - 1.0))
    return {
        "epochs": epochs,
        "all": spatial_instability(preds).astype(np.int16),
        "in": spatial_instability(preds[..., i0:i1 + 1]).astype(np.int16),
    }


def load_inst_s(sweep_dir: Path, h5_root: Path, cache_dir: Path, ks) -> pd.DataFrame:
    """Long table: k, epoch, inst_s_<range>_<mode>{,_std}; per-pair arrays cached as npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in ks:
        cache = cache_dir / f"inst_s_k-{k:g}.npz"
        if cache.is_file():
            arrays = dict(np.load(cache))
        else:
            arrays = {}
            for mode, _, _ in MODES:
                values = inst_s_per_pair(h5_dir(h5_root, sweep_dir, k) / f"{mode}.h5")
                arrays["epochs"] = values["epochs"]
                for rng in ALPHA_RANGES:
                    arrays[f"{rng}_{mode}"] = values[rng]
            np.savez_compressed(cache, **arrays)
            print(f"k={k:g}: cached {cache}", flush=True)
        frame = {"k": k, "epoch": arrays["epochs"]}
        for rng in ALPHA_RANGES:
            for mode, _, _ in MODES:
                per_pair = arrays[f"{rng}_{mode}"]
                frame[f"inst_s_{rng}_{mode}"] = per_pair.mean(axis=0)
                frame[f"inst_s_{rng}_{mode}_std"] = per_pair.std(axis=0)
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)


def inst_s_panel(rng: str, marker: dict | None = None, **panel_kwargs) -> PanelConfig:
    series = tuple(SeriesSpec(f"inst_s_{rng}_{mode}", label, color=color, linewidth=3,
                              **(marker or {}))
                   for mode, label, color in MODES)
    alpha_text = r"$\alpha\in[-0.5,1.5]$" if rng == "all" else r"$\alpha\in[0,1]$"
    return PanelConfig(
        series=series,
        ylabel=f"spatial instability\n({alpha_text})",
        ylim=(0.0, None),  # upper limit from data
        grid=True,
        grid_axis="y",
        tick_labelsize=35,
        legend_fontsize=35,
        **panel_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--h5-root", type=Path, default=H5_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <sweep-dir>/viz/spatial_instability")
    parser.add_argument("--alpha-range", choices=ALPHA_RANGES, default="all")
    parser.add_argument("--epoch", type=int, default=1000, help="epoch of the width-sweep figure")
    parser.add_argument("--ncols", type=int, default=6, help="columns of the all-widths grid")
    parser.add_argument("--epoch-curve-k", type=float, nargs="*", default=[64],
                        help="widths k drawn epoch-wise in epoch_spatial_k-<k>.*")
    parser.add_argument("--vmax-percentile", type=float, default=95.0,
                        help="heatmap color cap, as a percentile over all modes' cells")
    args = parser.parse_args()

    out_dir = args.out_dir or args.sweep_dir / "viz" / "spatial_instability"
    runs = load_runs(args.sweep_dir)                       # [(k, metrics.csv)]
    inst = load_inst_s(args.sweep_dir, args.h5_root, out_dir / "cache", [k for k, _ in runs])
    inst.to_csv(out_dir / "spatial_instability.csv", index=False)
    suffix = "" if args.alpha_range == "all" else "_alpha-in"

    # (1) all widths, epoch-wise: error panel of learning_curves_all + INST_S panel.
    merged = [(k, df.merge(inst[inst["k"] == k].drop(columns="k"), on="epoch", how="inner"))
              for k, df in runs]
    xmax = float(inst["epoch"].max())
    error_panel = learning_curve_panels()[0]
    panels = [
        replace(error_panel, xlim=(1, xmax)),
        inst_s_panel(args.alpha_range, xlabel="epoch", xscale="log", xlim=(1, xmax)),
    ]
    saved = plot_grid(merged, _shared_ylims(merged, panels, (1, xmax)),
                      out_dir / f"spatial_instability_all{suffix}", args.ncols, vlines=(),
                      cell_size=(6.5, 6.0))
    print(f"[✓] Saved: {', '.join(saved)}")

    # (2) model-wise, at --epoch vs width k: test/train error, train noisy/clean error, INST_S.
    rows = pd.concat([df[df["epoch"] == args.epoch].assign(width=k) for k, df in merged])
    if len(rows) != len(runs):
        raise SystemExit(f"epoch {args.epoch} missing for some widths")
    rows = rows.sort_values("width").reset_index(drop=True)
    marker = dict(marker="o", markersize=10)
    xlim = (rows["width"].min(), rows["width"].max())
    xticks = [k for k in (1, 2, 4, 8, 16, 32, 64) if xlim[0] <= k <= xlim[1]]
    common = dict(xscale="log", xlim=xlim, xticks=xticks)

    def with_markers(panel: PanelConfig) -> PanelConfig:
        return replace(panel, series=tuple(replace(s, **marker) for s in panel.series),
                       twin_series=tuple(replace(s, **marker) for s in panel.twin_series),
                       **common)

    noisy_clean_panel = learning_curve_panels()[1]
    inst_panel = inst_s_panel(args.alpha_range, marker=marker, xlabel="width $k$", **common)
    # Headroom above the INST_S curves so the legend does not cover any point.
    inst_ticks, inst_top = nice_ylim(0.0, 1.5 * max(rows[s.column].max() for s in inst_panel.series))
    width_panels = [
        with_markers(error_panel),
        with_markers(noisy_clean_panel),
        replace(inst_panel, ylim=(0.0, inst_top), yticks=inst_ticks),
    ]
    out = out_dir / f"width_sweep_spatial_epoch-{args.epoch}{suffix}"
    saved = plot_panels(rows, width_panels, out, figure_size=(13, 18),
                        font_family="Times New Roman", font_size=35, x="width")
    rows.to_csv(out.with_suffix(".csv"), index=False)
    print(f"[✓] Saved: {', '.join(saved + [str(out.with_suffix('.csv'))])}")

    # (3) epoch-wise for single widths, same three panels as (2) with epoch on x.
    epoch_common = dict(xscale="log", xlim=(1, xmax), xticks=None)
    for k in args.epoch_curve_k:
        df = dict(merged).get(k)
        if df is None:
            raise SystemExit(f"width k={k:g} not in the sweep")
        inst_panel = inst_s_panel(args.alpha_range, xlabel="epoch", **epoch_common)
        visible = df[df["epoch"] >= 1]
        inst_ticks, inst_top = nice_ylim(
            0.0, 1.5 * max(visible[s.column].max() for s in inst_panel.series))
        epoch_panels = [
            replace(error_panel, xlim=(1, xmax)),
            replace(learning_curve_panels()[1], xlim=(1, xmax)),
            replace(inst_panel, ylim=(0.0, inst_top), yticks=inst_ticks),
        ]
        out = out_dir / f"epoch_spatial_k-{k:g}{suffix}"
        saved = plot_panels(df, epoch_panels, out, figure_size=(13, 18),
                            font_family="Times New Roman", font_size=35, x="epoch")
        print(f"[✓] Saved: {', '.join(saved)}")

    # (4) width x epoch heatmaps, one panel per mode, color range shared across modes.
    for xscale, xsuffix in (("log", "_logx"), ("linear", "")):
        out = out_dir / f"heatmap_spatial{xsuffix}{suffix}"
        saved = plot_inst_s_heatmaps(merged, args.alpha_range, out, xscale,
                                     args.vmax_percentile)
        print(f"[✓] Saved: {', '.join(saved)}")


def plot_inst_s_heatmaps(runs, rng: str, output_base: Path, xscale: str,
                         vmax_percentile: float, panel_size=(14, 10)) -> list:
    """x = width k, y = epoch (log), color = mean INST_S; one panel per pair mode."""
    columns = [f"inst_s_{rng}_{mode}" for mode, _, _ in MODES]
    values = np.concatenate([df.loc[df["epoch"] >= 1, c].to_numpy() for _, df in runs
                             for c in columns])
    vmin, vmax = float(values.min()), float(np.percentile(values, vmax_percentile))
    alpha_text = r"$\alpha\in[-0.5,1.5]$" if rng == "all" else r"$\alpha\in[0,1]$"
    rc = {"font.family": _resolve_font("Times New Roman"), "font.size": 35,
          "mathtext.fontset": "stix"}
    saved = []
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(MODES), figsize=(len(MODES) * panel_size[0], panel_size[1]),
                                 squeeze=False, layout="constrained")
        for i, (ax, column, (_, label, _)) in enumerate(zip(axes[0], columns, MODES)):
            draw_heatmap(fig, ax, runs, column, xscale=xscale, show_ylabel=(i == 0),
                         vmin=vmin, vmax=vmax,
                         cbar_label=f"spatial instability ({alpha_text})")
            ax.set_title(label)
        for ext in ("png", "pdf"):
            out = output_base.with_name(f"{output_base.name}.{ext}")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            saved.append(str(out))
        plt.close(fig)
    return saved


if __name__ == "__main__":
    main()
