"""Phase-wise temporal instability INST_T(T_k, x) from the alpha_h5 interpolation logs.

Reads ``alpha_h5/alpha_test/<dataset>/<gamma>/64/<kind>.h5`` (predicted_label
[pairs, epochs, alphas]) and draws one panel per training phase with
``interpolation_metrics.plot_phase_temporal_instability``. In the ``noise``
(clean-noisy) pairs x0 is the clean sample and x1 the noisy one.

    python visualization/plot_phase_temporal_h5.py [--out-dir DIR]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interpolation_metrics import (  # noqa: E402
    phase_segments, phase_temporal_ylim, plot_phase_temporal_instability,
    segmented_temporal_instability, write_csv,
)

# Phase boundaries per (dataset, label-noise rate); segments end at the last evaluated epoch.
PHASE_EPOCHS = {
    ("emnist_digits", "0.2"): [47, 75, 177],
    ("cifar10", "0.2"): [23, 59],
    ("emnist_digits", "0.0"): [],
    ("cifar10", "0.0"): [],
}
KINDS = {"0.2": ["noise", "no_noise", "test_test_no_noise"], "0.0": ["no_noise", "test_test_no_noise"]}
# (dataset, gamma, kind) also drawn on the y scale of that run's clean-noisy
# figure, as an extra temporal_instability_phase_noise_scale.{pdf,svg}.
NOISE_SCALE_EXTRA = {("emnist_digits", "0.2", "no_noise")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5-root", type=Path, default=ROOT / "alpha_h5/alpha_test")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "analysis_alpha/figs/phase_temporal_instability")
    args = parser.parse_args()

    for (dataset, gamma), phase_epochs in PHASE_EPOCHS.items():
        noise_ylim = None
        for kind in KINDS[gamma]:  # "noise" comes first
            path = args.h5_root / dataset / gamma / "64" / f"{kind}.h5"
            with h5py.File(path, "r") as log:
                if not log["present"][:].all():
                    raise ValueError(f"{path} has missing (pair, epoch) entries")
                preds = log["predicted_label"][:]
                epochs = log["epochs"][:]
                alphas = np.round(log["alpha"][:].astype(float), 2)
            segments = phase_segments(epochs, phase_epochs)
            inst_t_phase, sizes = segmented_temporal_instability(preds, epochs, segments)

            figure_dir = args.out_dir / f"{dataset}_{gamma}_{kind}"
            figure_dir.mkdir(parents=True, exist_ok=True)
            inner = (alphas >= 0) & (alphas <= 1)
            write_csv(
                figure_dir / "temporal_instability_phase.csv",
                ["epoch_start", "epoch_end", "num_epochs",
                 "inst_t_mean", "inst_t_std", "inst_t_mean_alpha_0_1"],
                [(s, e, n, v.mean(), v.std(), v[:, inner].mean())
                 for (s, e), n, v in zip(segments, sizes, np.moveaxis(inst_t_phase, 1, 0))],
            )
            metrics = {
                "alphas": alphas,
                "segments": np.asarray(segments, dtype=float),
                "inst_t_phase": inst_t_phase,
                "endpoint_noisy": np.array([False, kind == "noise"]),
            }
            plot_phase_temporal_instability(plt, figure_dir, metrics)
            if kind == "noise":
                noise_ylim = phase_temporal_ylim(inst_t_phase)
            if (dataset, gamma, kind) in NOISE_SCALE_EXTRA:
                plot_phase_temporal_instability(
                    plt, figure_dir, metrics, ylim=noise_ylim,
                    file_name="temporal_instability_phase_noise_scale")
            print(f"{path.relative_to(ROOT)}: segments={segments} -> {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
