"""Spatial/temporal instability and label occupancy from interpolation logs.

Reads ``interpolation/<experiment_name>/<mode>/pair_XXXX/log.npz`` written by
``interpolation.py`` and writes metrics and figures to ``<mode>/metrics/``.

* INST_S(t): number of predicted-label changes between neighbouring alphas
  on the interpolation line (Fig. 3).
* INST_T(t, alpha): whether the prediction at alpha changed since the
  previously evaluated epoch, averaged over pairs (Fig. 4).
* Occupancy: fraction of the line predicted as the original label, the noisy
  label (pairs containing a noisy sample) and any other label (Fig. 6).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
    }


def spatial_instability(preds: np.ndarray) -> np.ndarray:
    """INST_S per pair and epoch: label changes along the line, [pairs, epochs]."""

    return (preds[..., 1:] != preds[..., :-1]).sum(axis=-1)


def temporal_instability(preds: np.ndarray) -> np.ndarray:
    """INST_T averaged over pairs, [epochs, alphas]; the first epoch is NaN."""

    flips = (preds[:, 1:] != preds[:, :-1]).mean(axis=0)
    first = np.full((1, preds.shape[-1]), np.nan)
    return np.concatenate([first, flips], axis=0)


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


def compute_metrics(mode_dir: Path) -> dict[str, np.ndarray]:
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


def plot_metrics(mode_dir: Path, metrics: dict[str, np.ndarray], phase_epochs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = mode_dir / "metrics" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mode = mode_dir.name
    epochs, alphas = metrics["epochs"], metrics["alphas"]

    figure, axis = plt.subplots(figsize=(5, 3))
    mean, std = metrics["inst_s"].mean(axis=0), metrics["inst_s"].std(axis=0)
    axis.plot(epochs, mean, color="#1f3fbf", linewidth=1)
    axis.fill_between(epochs, mean - std, mean + std, color="#1f3fbf", alpha=0.25, linewidth=0)
    _epoch_axis(axis, epochs)
    _phase_lines(axis, phase_epochs)
    axis.set(xlabel="epoch", ylabel="spatial instability", title=mode)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_dir / "spatial_instability.pdf")
    plt.close(figure)

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
        metrics = compute_metrics(mode_dir)
        if not args.no_plot:
            plot_metrics(mode_dir, metrics, args.phase_epochs)


if __name__ == "__main__":
    main()
