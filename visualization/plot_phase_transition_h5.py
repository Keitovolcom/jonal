"""Phase-wise temporal instability split by transition type, stacked over alpha.

For the clean-noisy pairs in ``alpha_h5/alpha_test/<dataset>/<gamma>/64/noise.h5``
every prediction is classed as c (the pair's clean label), n (its noisy label)
or o (any other label). A flip between consecutive evaluated epochs then has
one of seven types: c->n, n->c, o->c, c->o, n->o, o->n, o->o'. Their rates,
averaged over pairs and over the epochs of each training phase, add up to
INST_T(T_k, x) of ``plot_phase_temporal_h5.py``, so the top of each stack is
that curve.

Three figures per dataset: all seven types, four groups
(c<->n, o->c, c->o, rest = n->o + o->n + o->o'), and two groups
(c<->n, other = every flip involving an o label).

The clean-clean pairs of ``no_noise.h5`` share one label c at both ends and
have no noisy label, so only o->c, c->o and o->o' can occur; they get one
figure with those three types (colours as in the four-group figure, where
o->o' is what remains of "rest").

    python visualization/plot_phase_transition_h5.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from string import ascii_uppercase

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interpolation_metrics import (  # noqa: E402
    phase_segments, phase_temporal_ylim, segmented_temporal_instability, write_csv,
)
from visualization.plot_phase_temporal_h5 import PHASE_EPOCHS  # noqa: E402

C, N, O = 0, 1, 2
# (name, label, from, to); o->o' is a change between two different other labels.
TRANSITIONS = [
    ("c_to_n", r"$c\to n$", C, N),
    ("n_to_c", r"$n\to c$", N, C),
    ("o_to_c", r"$o\to c$", O, C),
    ("c_to_o", r"$c\to o$", C, O),
    ("n_to_o", r"$n\to o$", N, O),
    ("o_to_n", r"$o\to n$", O, N),
    ("o_to_o", r"$o\to o'$", O, O),
]
GROUPS = [
    ("c_n", r"$c\leftrightarrow n$", ["c_to_n", "n_to_c"]),
    ("o_to_c", r"$o\to c$", ["o_to_c"]),
    ("c_to_o", r"$c\to o$", ["c_to_o"]),
    ("rest", r"$n\to o,\ o\to n,\ o\to o'$", ["n_to_o", "o_to_n", "o_to_o"]),
]
# Clean-noisy tug of war vs every flip that involves another label.
GROUPS_2 = [
    ("c_n", r"$c\leftrightarrow n$", ["c_to_n", "n_to_c"]),
    ("other", "other", ["o_to_c", "c_to_o", "n_to_o", "o_to_n", "o_to_o"]),
]
# (dataset, gamma) whose clean-clean figure takes the y range of the
# clean-noisy temporal figure instead of its own, to compare the two directly.
NO_NOISE_YLIM_FROM_NOISE = {("emnist_digits", "0.2")}
# Clean-clean pairs: no noisy label, so only these types occur.
NO_NOISE_TYPES = ["o_to_c", "c_to_o", "o_to_o"]
# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1 on light).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def pair_labels(pair_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clean and noisy label per pair from names like ``pair1/0_6``."""

    parts = [name.decode().split("/")[-1].split("_") for name in pair_names]
    clean = np.array([int(p[0]) for p in parts])
    noisy = np.array([int(p[1]) for p in parts])
    if (clean == noisy).any():
        raise ValueError("pair with identical clean and noisy labels")
    return clean, noisy


def shared_labels(pair_names: np.ndarray) -> np.ndarray:
    """The common label per clean-clean pair from names like ``pair1/5_5``."""

    parts = [name.decode().split("/")[-1].split("_") for name in pair_names]
    first = np.array([int(p[0]) for p in parts])
    if (first != np.array([int(p[1]) for p in parts])).any():
        raise ValueError("clean-clean pair with different labels")
    return first


def segmented_transition_rates(
    preds: np.ndarray, clean: np.ndarray, noisy: np.ndarray,
    epochs: np.ndarray, segments: list[tuple[float, float]],
) -> np.ndarray:
    """Rate of each transition type per phase and alpha, [phases, types, alphas].

    Same averaging as ``segmented_temporal_instability``: over the pairs and
    over the epochs t in (start, end], each compared with the previous epoch.
    """

    kind = np.full(preds.shape, O, dtype=np.int8)
    kind[preds == clean[:, None, None]] = C
    if noisy is not None:  # clean-clean pairs have no noisy label
        kind[preds == noisy[:, None, None]] = N
    code = kind[:, :-1] * 3 + kind[:, 1:]      # [pairs, epochs - 1, alphas]
    changed = preds[:, 1:] != preds[:, :-1]
    targets = np.asarray(epochs[1:], dtype=float)

    rates = np.full((len(segments), len(TRANSITIONS), preds.shape[-1]), np.nan)
    for k, (start, end) in enumerate(segments):
        mask = (targets > start) & (targets <= end)
        if not mask.any():
            continue
        phase_code, phase_changed = code[:, mask], changed[:, mask]
        for j, (_, _, source, target) in enumerate(TRANSITIONS):
            rates[k, j] = ((phase_code == source * 3 + target) & phase_changed).mean(axis=(0, 1))
    return rates


def plot_stacked(figure_path: Path, alphas, segments, layers, labels, ylim,
                 colors=None, endpoint_colors=("blue", "red")) -> None:
    """One panel per phase: rates stacked over alpha, bottom layer first.

    Endpoint markers are blue for a clean sample and red for a noisy one. A
    single phase (no label noise) gets no title and a fixed, wider panel, as
    in ``plot_phase_temporal_instability``. ``ylim`` is the y range of the
    matching temporal-instability figure, so the two share one scale.
    """

    num_phases = len(segments)
    colors = COLORS[:len(labels)] if colors is None else colors
    with plt.rc_context({"font.size": 23, "font.family": "DejaVu Sans"}):
        figure, axes = plt.subplots(
            1, num_phases, figsize=(6 * num_phases, 6.5), sharey=True, dpi=300, squeeze=False
        )
        axes = axes[0]
        for k, (axis, (start, end)) in enumerate(zip(axes, segments)):
            name = ascii_uppercase[k % 26] + ascii_uppercase[(k + 1) % 26]
            if num_phases > 1:
                axis.set_title(rf"$\mathcal{{T}}_{{{name}}}$=[{start:g},{end:g}]")
            if np.isnan(layers[k]).all():
                axis.text(0.5, 0.5, "No Data", ha="center", va="center", transform=axis.transAxes)
                continue
            # A thin white edge separates neighbouring fills.
            axis.stackplot(alphas, layers[k], colors=colors, labels=labels,
                           edgecolor="white", linewidth=0.6, zorder=3)
            for alpha, color in zip((0.0, 1.0), endpoint_colors):
                axis.axvline(alpha, color="black", linestyle="-", linewidth=2, zorder=4)
                axis.plot(alpha, 0.0, "o", color=color, markersize=15, zorder=5, clip_on=False)
            axis.set_xticks([0.0, 1.0])
            axis.set_xticklabels([r"$x_{0}$", r"$x_{1}$"], fontsize=30)
            axis.set_xlim(alphas[0], alphas[-1])
            axis.grid(True, zorder=0)
        axes[0].set_ylabel("Temporal Instability")
        axes[0].set_ylim(*ylim)
        # Stack order (bottom layer first); columns fill top-down, so the
        # seven-type legend pairs c->n/n->c, o->c/c->o, n->o/o->n.
        handles, names = axes[0].get_legend_handles_labels()
        figure.legend(handles, names, loc="lower center",
                      ncol=min(len(labels), 4), frameon=False)
        rows = -(-len(labels) // 4)
        legend_height = 0.52 * rows  # inches kept free for the legend
        figure.tight_layout(rect=(0, legend_height / figure.get_size_inches()[1], 1, 1))
        if num_phases == 1:
            # Same panel as the single-phase temporal figure (5.8 x 4.4 in).
            for _ in range(3):
                width, height = figure.get_size_inches()
                box = axes[0].get_position()
                figure.set_size_inches(width + 5.8 - box.width * width,
                                       height + 4.4 - box.height * height)
                figure.tight_layout(rect=(0, legend_height / figure.get_size_inches()[1], 1, 1))
        for suffix in ("pdf", "svg"):
            figure.savefig(figure_path.with_suffix(f".{suffix}"))
        plt.close(figure)


def load_log(path: Path):
    with h5py.File(path, "r") as log:
        if not log["present"][:].all():
            raise ValueError(f"{path} has missing (pair, epoch) entries")
        return (log["predicted_label"][:], log["epochs"][:],
                np.round(log["alpha"][:].astype(float), 2), log["pair_names"][:])


def write_rate_csvs(figure_dir: Path, segments, rates, alphas, names) -> None:
    inner = (alphas >= 0) & (alphas <= 1)
    write_csv(
        figure_dir / "transition_types_phase.csv",
        ["epoch_start", "epoch_end", "inst_t_mean_alpha_0_1"]
        + [f"{n}_mean_alpha_0_1" for n in names],
        [(s, e, r[:, inner].sum(axis=0).mean(), *r[:, inner].mean(axis=1))
         for (s, e), r in zip(segments, rates)],
    )
    write_csv(
        figure_dir / "transition_types_phase_alpha.csv",
        ["epoch_start", "epoch_end", "alpha", *names],
        [(s, e, a, *rates[k, :, i])
         for k, (s, e) in enumerate(segments) for i, a in enumerate(alphas)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5-root", type=Path, default=ROOT / "alpha_h5/alpha_test")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "analysis_alpha/figs/phase_transition_types")
    args = parser.parse_args()
    names = [t[0] for t in TRANSITIONS]

    for (dataset, gamma), phase_epochs in PHASE_EPOCHS.items():
        noise_ylim = None
        # Clean-noisy pairs (label-noise runs only).
        path = args.h5_root / dataset / gamma / "64" / "noise.h5"
        if path.exists():
            preds, epochs, alphas, pair_names = load_log(path)
            clean, noisy = pair_labels(pair_names)
            segments = phase_segments(epochs, phase_epochs)
            rates = segmented_transition_rates(preds, clean, noisy, epochs, segments)
            ylim = noise_ylim = phase_temporal_ylim(
                segmented_temporal_instability(preds, epochs, segments)[0])

            def group(groups):
                return np.stack(
                    [rates[:, [names.index(m) for m in members]].sum(axis=1)
                     for _, _, members in groups],
                    axis=1,
                )

            figure_dir = args.out_dir / f"{dataset}_{gamma}_noise"
            figure_dir.mkdir(parents=True, exist_ok=True)
            write_rate_csvs(figure_dir, segments, rates, alphas, names)
            plot_stacked(figure_dir / "transition_types_7", alphas, segments, rates,
                         [t[1] for t in TRANSITIONS], ylim)
            plot_stacked(figure_dir / "transition_types_4", alphas, segments, group(GROUPS),
                         [g[1] for g in GROUPS], ylim)
            plot_stacked(figure_dir / "transition_types_2", alphas, segments, group(GROUPS_2),
                         [g[1] for g in GROUPS_2], ylim)
            print(f"{path.relative_to(ROOT)}: segments={segments} -> {figure_dir}", flush=True)

        # Clean-clean pairs: only o->c, c->o, o->o' can occur.
        path = args.h5_root / dataset / gamma / "64" / "no_noise.h5"
        if path.exists():
            preds, epochs, alphas, pair_names = load_log(path)
            segments = phase_segments(epochs, phase_epochs)
            rates = segmented_transition_rates(
                preds, shared_labels(pair_names), None, epochs, segments)
            if (dataset, gamma) in NO_NOISE_YLIM_FROM_NOISE:
                ylim = noise_ylim
            else:
                ylim = phase_temporal_ylim(
                    segmented_temporal_instability(preds, epochs, segments)[0])
            other = [names.index(n) for n in names if n not in NO_NOISE_TYPES]
            if np.nansum(rates[:, other]) > 0:
                raise ValueError(f"{path}: transition involving a noisy label")
            keep = [names.index(n) for n in NO_NOISE_TYPES]
            figure_dir = args.out_dir / f"{dataset}_{gamma}_no_noise"
            figure_dir.mkdir(parents=True, exist_ok=True)
            write_rate_csvs(figure_dir, segments, rates[:, keep], alphas, NO_NOISE_TYPES)
            # Colours of o->c, c->o and rest in the four-group figure.
            plot_stacked(figure_dir / "transition_types_3", alphas, segments, rates[:, keep],
                         [TRANSITIONS[i][1] for i in keep], ylim, colors=COLORS[1:4],
                         endpoint_colors=("blue", "blue"))
            print(f"{path.relative_to(ROOT)}: segments={segments} -> {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
