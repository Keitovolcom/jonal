"""Spatial/temporal instability and label occupancy from interpolation logs.

Reads ``interpolation/<experiment_name>/<mode>/pair_XXXX/log.npz`` written by
``interpolation.py`` and writes metrics and figures to ``<mode>/metrics/``.

* INST_S(t): number of predicted-label changes between neighbouring alphas
  on the interpolation line (Fig. 3).
* INST_T(t, alpha): whether the prediction at alpha changed since the
  previously evaluated epoch, averaged over pairs (Fig. 4).
* INST_T(T_k, x): for each training phase T_k (split at ``--phase-epochs``)
  and each point x on the line, the fraction of epochs in T_k at which the
  prediction at x changed since the previously evaluated epoch.
* Occupancy: fraction of the line predicted as the original label, the noisy
  label (pairs containing a noisy sample) and any other label (Fig. 6).
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from string import ascii_uppercase

import numpy as np


def parse_epochs(value: str) -> list[float]:
    return [float(token) for token in value.split(",") if token.strip()]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute spatio-temporal instability from interpolation logs."
    )
    parser.add_argument(
        "--mode-dir", "--mode_dir", dest="mode_dirs", type=Path, nargs="+", required=True,
        help="One or more interpolation/<experiment_name>/<mode> directories",
    )
    parser.add_argument(
        "--phase-epochs", "--phase_epochs", dest="phase_epochs", type=parse_epochs,
        default=[], help="Comma-separated epochs drawn as phase boundaries, e.g. '30,60'",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def load_logs(mode_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted(mode_dir.glob("pair_*/log.npz"))
    if not paths:
        raise FileNotFoundError(f"no pair logs found under {mode_dir}")
    preds, original, noisy_label, pair_ids = [], [], [], []
    x0_is_noisy, x1_is_noisy = [], []
    epochs = alphas = None
    for path in paths:
        with np.load(path) as log:
            if epochs is None:
                epochs, alphas = log["epochs"], log["alphas"]
            elif (not np.array_equal(epochs, log["epochs"])
                  or not np.array_equal(alphas, log["alphas"])):
                raise ValueError(f"{path} has different epochs or alphas")
            preds.append(log["preds"])
            pair_ids.append(int(log["pair_id"]))
            original.append(int(log["x0_original_label"]))
            x0_is_noisy.append(bool(log["x0_is_noisy"]))
            x1_is_noisy.append(bool(log["x1_is_noisy"]))
            # The noisy sample's training label, or -1 when the pair has none.
            if bool(log["x0_is_noisy"]):
                noisy_label.append(int(log["x0_training_label"]))
            elif bool(log["x1_is_noisy"]):
                noisy_label.append(int(log["x1_training_label"]))
            else:
                noisy_label.append(-1)
    return {
        "preds": np.stack(preds),            # [pairs, epochs, alphas]
        "epochs": epochs,
        "alphas": alphas,
        "pair_ids": np.asarray(pair_ids),
        "original": np.asarray(original),
        "noisy_label": np.asarray(noisy_label),
        "x0_is_noisy": np.asarray(x0_is_noisy),
        "x1_is_noisy": np.asarray(x1_is_noisy),
    }


def spatial_instability(preds: np.ndarray) -> np.ndarray:
    """INST_S per pair and epoch: label changes along the line, [pairs, epochs]."""

    return (preds[..., 1:] != preds[..., :-1]).sum(axis=-1)


def temporal_instability(preds: np.ndarray) -> np.ndarray:
    """INST_T averaged over pairs, [epochs, alphas]; the first epoch is NaN."""

    flips = (preds[:, 1:] != preds[:, :-1]).mean(axis=0)
    first = np.full((1, preds.shape[-1]), np.nan)
    return np.concatenate([first, flips], axis=0)


def phase_segments(epochs: np.ndarray, phase_epochs: list[float]) -> list[tuple[float, float]]:
    """Training phases (start, end] split at the phase epochs inside the evaluated range."""

    first, last = float(epochs[0]), float(epochs[-1])
    edges = [first] + sorted({b for b in phase_epochs if first < b < last}) + [last]
    return list(zip(edges[:-1], edges[1:]))


def segmented_temporal_instability(
    preds: np.ndarray, epochs: np.ndarray, segments: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """INST_T(T_k, x) per pair, phase and alpha, [pairs, phases, alphas].

    T_k holds the evaluated epochs t with start < t <= end, and Fl_T(t, x)
    compares epoch t with the previously evaluated epoch, so the first epoch
    is never in any T_k. Also returns |T_k| per phase.
    """

    flips = preds[:, 1:] != preds[:, :-1]      # [pairs, epochs - 1, alphas]
    targets = np.asarray(epochs[1:], dtype=float)
    values, sizes = [], []
    for start, end in segments:
        mask = (targets > start) & (targets <= end)
        sizes.append(int(mask.sum()))
        if mask.any():
            values.append(flips[:, mask].mean(axis=1))
        else:
            values.append(np.full((preds.shape[0], preds.shape[-1]), np.nan))
    return np.stack(values, axis=1), np.asarray(sizes)


def label_occupancy(
    preds: np.ndarray, original: np.ndarray, noisy_label: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-pair occupancy rates on the line, each [pairs, epochs]."""

    original_rate = (preds == original[:, None, None]).mean(axis=-1)
    noisy_rate = (preds == noisy_label[:, None, None]).mean(axis=-1)
    noisy_rate[noisy_label < 0] = np.nan
    other_rate = 1.0 - original_rate - np.nan_to_num(noisy_rate)
    return {"original": original_rate, "noisy": noisy_rate, "other": other_rate}


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def compute_metrics(mode_dir: Path, phase_epochs: list[float] = ()) -> dict[str, np.ndarray]:
    logs = load_logs(mode_dir)
    preds, epochs, alphas = logs["preds"], logs["epochs"], logs["alphas"]
    output_dir = mode_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    inst_s = spatial_instability(preds)
    np.savez(
        output_dir / "spatial_instability_per_pair.npz",
        inst_s=inst_s, epochs=epochs, pair_ids=logs["pair_ids"],
    )
    write_csv(
        output_dir / "spatial_instability.csv",
        ["epoch", "inst_s_mean", "inst_s_std"],
        zip(epochs, inst_s.mean(axis=0), inst_s.std(axis=0)),
    )

    inst_t = temporal_instability(preds)
    np.savez(
        output_dir / "temporal_instability.npz",
        inst_t=inst_t, epochs=epochs, alphas=alphas,
    )
    write_csv(
        output_dir / "temporal_instability.csv",
        ["epoch", "inst_t_mean"],
        zip(epochs, inst_t.mean(axis=1)),
    )

    segments = phase_segments(epochs, list(phase_epochs))
    inst_t_phase, phase_sizes = segmented_temporal_instability(preds, epochs, segments)
    segment_array = np.asarray(segments, dtype=float)
    np.savez(
        output_dir / "temporal_instability_phase.npz",
        inst_t_phase=inst_t_phase, segments=segment_array, num_epochs=phase_sizes,
        alphas=alphas, pair_ids=logs["pair_ids"],
    )
    # Statistics over all points x (pairs x alphas) and over the interval
    # between the two samples (0 <= alpha <= 1).
    inner = (alphas >= 0) & (alphas <= 1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        phase_rows = [
            (start, end, size,
             np.nanmean(values), np.nanstd(values), np.nanmean(values[:, inner]))
            for (start, end), size, values in zip(
                segments, phase_sizes, np.moveaxis(inst_t_phase, 1, 0))
        ]
    write_csv(
        output_dir / "temporal_instability_phase.csv",
        ["epoch_start", "epoch_end", "num_epochs",
         "inst_t_mean", "inst_t_std", "inst_t_mean_alpha_0_1"],
        phase_rows,
    )

    occupancy = label_occupancy(preds, logs["original"], logs["noisy_label"])
    has_noisy = bool((logs["noisy_label"] >= 0).any())
    names = ["original", "noisy", "other"] if has_noisy else ["original", "other"]
    columns = []
    for name in names:
        columns += [np.nanmean(occupancy[name], axis=0), np.nanstd(occupancy[name], axis=0)]
    write_csv(
        output_dir / "occupancy.csv",
        ["epoch"] + [f"{name}_{stat}" for name in names for stat in ("mean", "std")],
        zip(epochs, *columns),
    )
    print(f"{mode_dir}: {len(preds)} pairs, {len(epochs)} epochs -> {output_dir}", flush=True)
    return {
        "epochs": epochs, "alphas": alphas, "inst_s": inst_s, "inst_t": inst_t,
        "inst_t_phase": inst_t_phase, "segments": segment_array,
        # Endpoint drawn as noisy when most pairs have a noisy sample there.
        "endpoint_noisy": np.array([logs["x0_is_noisy"].mean() > 0.5,
                                    logs["x1_is_noisy"].mean() > 0.5]),
        **{f"occupancy_{name}": occupancy[name] for name in names},
    }


def _epoch_axis(axis, epochs: np.ndarray, which: str = "x") -> None:
    scale = "log" if epochs.min() > 0 else "symlog"
    options = {"linthresh": 1} if scale == "symlog" else {}
    (axis.set_xscale if which == "x" else axis.set_yscale)(scale, **options)


def _phase_lines(axis, phase_epochs: list[float], which: str = "x") -> None:
    for epoch in phase_epochs:
        if which == "x":
            axis.axvline(epoch, color="black", linewidth=0.8)
        else:
            axis.axhline(epoch, color="white", linewidth=0.8, linestyle="--")


def phase_temporal_ylim(inst_t_phase: np.ndarray) -> tuple[float, float]:
    """Y range of ``plot_phase_temporal_instability`` for these values.

    The top is the highest mean + std band edge (clipped to 1) plus
    matplotlib's default 5% margin; the bottom leaves room for the endpoint
    markers at 0. Transition-type figures reuse it so both share one scale.
    """

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(inst_t_phase, axis=0)   # [phases, alphas]
        std = np.nanstd(inst_t_phase, axis=0)
    top = 1.05 * np.nanmax(np.clip(mean + std, 0, 1))
    return -0.02 * top, top


def plot_phase_temporal_instability(
    plt, figure_dir: Path, metrics: dict[str, np.ndarray],
    ylim: tuple[float, float] | None = None, file_name: str = "temporal_instability_phase",
) -> None:
    """One panel per training phase: mean INST_T(T_k, x) over pairs vs alpha, +-1 std.

    Phases are named by consecutive letters at their boundaries (T_AB, T_BC, ...)
    and labelled with the evaluated epoch range [start, end] they use. Endpoint
    markers are red for a noisy sample and blue for a clean one.
    """

    alphas, segments = metrics["alphas"], metrics["segments"]
    inst_t_phase = metrics["inst_t_phase"]
    endpoint_colors = ["red" if noisy else "blue" for noisy in metrics["endpoint_noisy"]]
    num_phases = len(segments)

    with plt.rc_context({"font.size": 23, "font.family": "DejaVu Sans"}):
        figure, axes = plt.subplots(
            1, num_phases, figsize=(6 * num_phases, 5.5), sharey=True, dpi=300, squeeze=False
        )
        axes = axes[0]
        for k, (axis, (start, end), values) in enumerate(
            zip(axes, segments, np.moveaxis(inst_t_phase, 1, 0))
        ):
            if np.isnan(values).all():
                axis.text(0.5, 0.5, "No Data", ha="center", va="center",
                          transform=axis.transAxes)
                continue
            mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
            axis.plot(alphas, mean, linewidth=2, zorder=3, color="blue",
                      label=r"$INST_{\mathrm{T}}(\mathcal{T},x)$")
            # INST_T lies in [0, 1], so the band is clipped to that range.
            axis.fill_between(alphas, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                              alpha=0.2, zorder=2, color="blue", linewidth=0)
            name = ascii_uppercase[k % 26] + ascii_uppercase[(k + 1) % 26]
            axis.plot([], [], linewidth=0,
                      label=rf"$\mathcal{{T}}_{{{name}}}$=[{start:g},{end:g}]")
            for alpha, color in zip((0.0, 1.0), endpoint_colors):
                axis.axvline(alpha, color="black", linestyle="-", linewidth=2, zorder=1.8)  # above the grid
                axis.plot(alpha, 0.0, "o", color=color, markersize=15, zorder=5,
                          clip_on=False)
            axis.set_xticks([0.0, 1.0])
            axis.set_xticklabels([r"$x_{0}$", r"$x_{1}$"], fontsize=30)
            if num_phases > 1:  # a single phase (no label noise) needs no legend
                axis.legend()
            axis.grid(True)
        axes[0].set_ylabel("Temporal Instability")
        # Shared with the transition-type figures; leaves room for the markers at 0.
        # ``ylim`` overrides it, e.g. to draw clean pairs on a clean-noisy scale.
        axes[0].set_ylim(*(phase_temporal_ylim(inst_t_phase) if ylim is None else ylim))
        # Fix every panel to the same size in inches, so a single-phase figure
        # (label noise 0.0) looks like one panel of the multi-phase figure
        # instead of a narrow panel squeezed by the y label and tick labels.
        # The single panel is ~10% wider.
        panel_width, panel_height = (5.3 if num_phases > 1 else 5.8), 4.2
        figure.tight_layout()
        for _ in range(3):
            width, height = figure.get_size_inches()
            box = axes[0].get_position()
            figure.set_size_inches(width + num_phases * (panel_width - box.width * width),
                                   height + (panel_height - box.height * height))
            figure.tight_layout()
        for suffix in ("pdf", "svg"):
            figure.savefig(figure_dir / f"{file_name}.{suffix}")
        plt.close(figure)


def plot_spatial_instability(
    plt, figure_dir: Path, metrics: dict[str, np.ndarray], phase_epochs=()
) -> None:
    """Mean INST_S(t) over pairs vs epoch with a +-1 std band.

    Same style as ``plot_phase_temporal_instability``. The phase boundaries
    (first epoch, ``phase_epochs``, last epoch) are lettered A, B, C, ... so
    they match the T_AB, T_BC, ... phases of the temporal figure.
    """

    from matplotlib.transforms import blended_transform_factory

    epochs = np.asarray(metrics["epochs"], dtype=float)
    inst_s = metrics["inst_s"]
    mean, std = inst_s.mean(axis=0), inst_s.std(axis=0)
    edges = [start for start, _ in phase_segments(epochs, list(phase_epochs))] + [epochs[-1]]

    with plt.rc_context({"font.size": 23, "font.family": "DejaVu Sans"}):
        figure, axis = plt.subplots(figsize=(12, 6), dpi=300)
        axis.plot(epochs, mean, linewidth=2, zorder=3, color="blue",
                  label=r"$INST_{\mathrm{S}}(t)$")
        # INST_S is a count of label changes, so the band stops at 0.
        axis.fill_between(epochs, np.clip(mean - std, 0, None), mean + std,
                          alpha=0.2, zorder=2, color="blue", linewidth=0)
        _epoch_axis(axis, epochs)
        axis.set_xlim(epochs[0], epochs[-1])
        label_transform = blended_transform_factory(axis.transData, axis.transAxes)
        for k, edge in enumerate(edges):
            if epochs[0] < edge < epochs[-1]:
                axis.axvline(edge, color="black", linestyle="--", linewidth=2, zorder=1.8)
            axis.text(edge, 1.02, ascii_uppercase[k % 26], transform=label_transform,
                      ha="center", va="bottom")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Spatial Instability")
        axis.set_ylim(bottom=0)
        axis.legend()
        axis.grid(True)
        figure.tight_layout()
        for suffix in ("pdf", "svg"):
            figure.savefig(figure_dir / f"spatial_instability.{suffix}")
        plt.close(figure)


def plot_metrics(mode_dir: Path, metrics: dict[str, np.ndarray], phase_epochs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = mode_dir / "metrics" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mode = mode_dir.name
    epochs, alphas = metrics["epochs"], metrics["alphas"]

    plot_spatial_instability(plt, figure_dir, metrics, phase_epochs)

    if len(epochs) > 1:
        figure, axis = plt.subplots(figsize=(4, 4))
        mesh = axis.pcolormesh(
            alphas, epochs[1:], metrics["inst_t"][1:], cmap="plasma",
            shading="nearest", vmin=0,
        )
        _epoch_axis(axis, epochs[1:], which="y")
        _phase_lines(axis, phase_epochs, which="y")
        axis.set_xticks([0.0, 1.0], [r"$x_0$", r"$x_1$"])
        axis.set(ylabel="epoch", title=mode)
        figure.colorbar(mesh, ax=axis, label="temporal instability")
        figure.tight_layout()
        figure.savefig(figure_dir / "temporal_instability_heatmap.pdf")
        plt.close(figure)

    if len(epochs) > 1:
        plot_phase_temporal_instability(plt, figure_dir, metrics)

    figure, axis = plt.subplots(figsize=(5, 3))
    colors = {"original": "#1f3fbf", "noisy": "#d62728", "other": "#2ca02c"}
    for name, color in colors.items():
        key = f"occupancy_{name}"
        if key in metrics:
            axis.plot(epochs, np.nanmean(metrics[key], axis=0), color=color,
                      linewidth=1, label=f"{name} label" + ("s" if name == "other" else ""))
    _epoch_axis(axis, epochs)
    _phase_lines(axis, phase_epochs)
    axis.set(xlabel="epoch", ylabel="predicted label occupancy ratio", title=mode)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "occupancy.pdf")
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    for mode_dir in args.mode_dirs:
        metrics = compute_metrics(mode_dir, args.phase_epochs)
        if not args.no_plot:
            plot_metrics(mode_dir, metrics, args.phase_epochs)


if __name__ == "__main__":
    main()
