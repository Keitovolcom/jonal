"""Plot per-sample loss distributions written by ``per_sample_metrics.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

if __package__:
    from .plot_seed_sweep import COLORS, RESULTS_DIR
else:
    from plot_seed_sweep import COLORS, RESULTS_DIR


DEFAULT_RUN_DIR = RESULTS_DIR / (
    "emnist-digits_cnn5layer_noise-0p2_lr-0p01_loss-crossentropy_bs-256_"
    "opt-sgd_width-1_ch-1_aug-none_seed-42"
)
DEFAULT_PREFIX = RESULTS_DIR / "viz" / "cnn5layer_seed42_per_sample"
# Cross-entropy above ln 2 means the label probability is below 0.5.
LN2 = float(np.log(2.0))
LOG_LOSS_RANGE = (-6.0, 1.5)
LOG_LOSS_BINS = 120
QUANTILES = (0.5, 0.9, 0.99)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Output path prefix; '_<figure>.<ext>' is appended.",
    )
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument(
        "--window",
        type=lambda value: tuple(int(part) for part in value.split("-", 1)),
        default=(60, 110),
        help="Epoch range used for the persistence histogram, e.g. 60-110.",
    )
    parser.add_argument("--xmin", type=float, default=30)
    parser.add_argument("--xmax", type=float, default=200)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def save(figure: plt.Figure, prefix: Path, name: str, formats: str, dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats.split(","):
        path = prefix.with_name(f"{prefix.name}_{name}.{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot to {path}")
    plt.close(figure)


def plot_distribution(data, epochs, loss, xlim, prefix, formats, dpi) -> None:
    """Epoch x log10(loss) density for clean and noisy samples."""
    edges = np.linspace(*LOG_LOSS_RANGE, LOG_LOSS_BINS + 1)
    log_loss = np.log10(np.clip(loss, 10 ** LOG_LOSS_RANGE[0], None))
    groups = (
        ("Clean samples", ~data["is_noisy"], COLORS["clean"]),
        ("Noisy samples", data["is_noisy"], COLORS["noisy"]),
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharey=True, constrained_layout=True
    )
    for axis, (title, mask, color) in zip(axes, groups):
        subset = log_loss[:, mask]
        density = np.stack([np.histogram(row, bins=edges)[0] for row in subset])
        density = density / mask.sum()
        image = axis.pcolormesh(
            epochs, (edges[:-1] + edges[1:]) / 2, density.T,
            shading="nearest", cmap="Greys",
            norm=LogNorm(vmin=1e-5, vmax=density.max()),
        )
        axis.plot(epochs, np.log10(loss[:, mask].mean(axis=1)), color=color,
                  linewidth=2.0, label="Mean loss")
        for quantile, style in zip(QUANTILES, ("-", "--", ":")):
            axis.plot(epochs, np.quantile(subset, quantile, axis=1),
                      color="#ff7f0e", linestyle=style, linewidth=1.2,
                      label=f"{quantile:.0%} quantile")
        axis.axhline(np.log10(LN2), color="black", linewidth=1.0,
                     linestyle="-.", label="ln 2 (misclassified above)")
        axis.set_xlim(*xlim)
        axis.set_ylim(*LOG_LOSS_RANGE)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend(loc="lower left", fontsize=8, frameon=True)
    axes[0].set_ylabel("log10(per-sample loss)")
    figure.colorbar(image, ax=axes, label="Fraction of samples per bin")
    figure.suptitle("Per-sample training loss (end-of-epoch, recomputed from checkpoints)")
    save(figure, prefix, "loss_distribution", formats, dpi)


def plot_decomposition(data, epochs, loss, correct, xlim, prefix, formats, dpi):
    """Clean mean loss = (1 - e) * mean(correct) + e * mean(misclassified)."""
    clean = ~data["is_noisy"]
    clean_loss = loss[:, clean]
    clean_correct = correct[:, clean]
    error = 1.0 - clean_correct.mean(axis=1)
    count = clean.sum()
    correct_part = np.where(clean_correct, clean_loss, 0.0).sum(axis=1) / count
    wrong_part = np.where(~clean_correct, clean_loss, 0.0).sum(axis=1) / count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_correct = np.where(clean_correct, clean_loss, 0.0).sum(axis=1) / clean_correct.sum(axis=1)
        mean_wrong = np.where(~clean_correct, clean_loss, 0.0).sum(axis=1) / (~clean_correct).sum(axis=1)

    figure, axes = plt.subplots(
        3, 1, figsize=(9, 10), sharex=True, constrained_layout=True
    )
    axes[0].plot(epochs, error, color=COLORS["clean"], linewidth=1.6)
    axes[0].set_ylabel("Clean error")
    axes[1].plot(epochs, clean_loss.mean(axis=1), color="black", linewidth=1.8,
                 label="Clean mean loss (total)")
    axes[1].plot(epochs, correct_part, color=COLORS["train"], linewidth=1.5,
                 label="Contribution of correctly classified")
    axes[1].plot(epochs, wrong_part, color=COLORS["test"], linewidth=1.5,
                 label="Contribution of misclassified")
    axes[1].set_ylabel("Loss contribution\n(sum / #clean)")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(epochs, mean_correct, color=COLORS["train"], linewidth=1.5,
                     label="Mean loss of correctly classified")
    axes[2].semilogy(epochs, mean_wrong, color=COLORS["test"], linewidth=1.5,
                     label="Mean loss of misclassified")
    axes[2].set_ylabel("Mean loss within group")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlim(*xlim)
        axis.grid(True, color="0.88", linewidth=0.7)
    figure.suptitle("Clean samples: mean loss split by correct / misclassified")
    save(figure, prefix, "clean_decomposition", formats, dpi)
    return error


def plot_turnover(data, epochs, correct, error, window, xlim, prefix,
                  formats, dpi) -> None:
    """How persistent is the misclassification of clean samples?"""
    clean = ~data["is_noisy"]
    wrong = ~correct[:, clean]
    persistence = np.full(len(epochs), np.nan)
    for index in range(1, len(epochs)):
        if wrong[index].any():
            persistence[index] = wrong[index - 1][wrong[index]].mean()
    in_window = (epochs >= window[0]) & (epochs <= window[1])
    counts = wrong[in_window].sum(axis=0)
    n_epochs = int(in_window.sum())
    print(
        f"epochs {window[0]}-{window[1]}: clean ever wrong {(counts > 0).mean():.3f}, "
        f"wrong in >=25% of epochs {(counts >= 0.25 * n_epochs).mean():.4f}, "
        f"median persistence {np.nanmedian(persistence[in_window]):.3f}"
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    axis = axes[0]
    axis.plot(epochs, error, color=COLORS["clean"], linewidth=1.5,
              label="Clean error (end of epoch)")
    axis.set_ylabel("Clean error", color=COLORS["clean"])
    twin = axis.twinx()
    twin.plot(epochs, persistence, color=COLORS["test"], linewidth=1.3,
              label="Wrong at e that were also wrong at e-1")
    twin.set_ylim(0, 1)
    twin.set_ylabel("Fraction also wrong one epoch before", color=COLORS["test"])
    axis.axvspan(*window, color="0.9", zorder=0)
    axis.set_xlim(*xlim)
    axis.set_xlabel("Epoch")
    axis.grid(True, color="0.88", linewidth=0.7)
    axis.set_title("The misclassified clean set turns over every epoch")
    axis.legend(handles=axis.get_lines() + twin.get_lines(), fontsize=8,
                loc="upper right")

    axis = axes[1]
    axis.hist(counts, bins=np.arange(n_epochs + 2) - 0.5, color=COLORS["clean"])
    axis.set_yscale("log")
    axis.set_xlabel(f"Epochs misclassified within epoch {window[0]}-{window[1]} "
                    f"(of {n_epochs})")
    axis.set_ylabel("Number of clean samples")
    axis.grid(True, color="0.88", linewidth=0.7)
    axis.set_title(f"{(counts > 0).mean():.0%} of clean samples are misclassified "
                   "at least once")
    save(figure, prefix, "clean_turnover", formats, dpi)


def main() -> None:
    args = create_parser().parse_args()
    data = np.load(args.run_dir / "per_sample" / "metrics.npz")
    epochs = data["epochs"]
    loss = data["loss"]
    correct = data["prediction"] == data["training_label"][None, :]
    xlim = (args.xmin, args.xmax)
    plot_distribution(data, epochs, loss, xlim, args.prefix, args.formats, args.dpi)
    error = plot_decomposition(
        data, epochs, loss, correct, xlim, args.prefix, args.formats, args.dpi
    )
    plot_turnover(data, epochs, correct, error, args.window, xlim,
                  args.prefix, args.formats, args.dpi)


if __name__ == "__main__":
    main()
