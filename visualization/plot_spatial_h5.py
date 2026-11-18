"""Spatial instability INST_S(t) per epoch from the alpha_h5 interpolation logs.

Reads ``alpha_h5/alpha_test/<dataset>/<gamma>/64/<kind>.h5`` (predicted_label
[pairs, epochs, alphas]). For every pair and epoch INST_S is the number of
predicted-label changes between neighbouring alphas on the line; the mean and
std over pairs are written to CSV and drawn with
``interpolation_metrics.plot_spatial_instability``, with the phase boundaries of
``plot_phase_temporal_h5.py`` lettered A, B, C, ...

    python visualization/plot_spatial_h5.py [--out-dir DIR]
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
from interpolation_metrics import plot_spatial_instability, spatial_instability, write_csv  # noqa: E402
from visualization.plot_phase_temporal_h5 import KINDS, PHASE_EPOCHS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5-root", type=Path, default=ROOT / "alpha_h5/alpha_test")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "analysis_alpha/figs/spatial_instability")
    args = parser.parse_args()

    for (dataset, gamma), phase_epochs in PHASE_EPOCHS.items():
        for kind in KINDS[gamma]:
            path = args.h5_root / dataset / gamma / "64" / f"{kind}.h5"
            with h5py.File(path, "r") as log:
                if not log["present"][:].all():
                    raise ValueError(f"{path} has missing (pair, epoch) entries")
                preds = log["predicted_label"][:]
                epochs = log["epochs"][:]
            inst_s = spatial_instability(preds)          # [pairs, epochs]

            figure_dir = args.out_dir / f"{dataset}_{gamma}_{kind}"
            figure_dir.mkdir(parents=True, exist_ok=True)
            write_csv(
                figure_dir / "spatial_instability.csv",
                ["epoch", "mean_spatial_instability", "std_spatial_instability"],
                zip(epochs, inst_s.mean(axis=0), inst_s.std(axis=0)),
            )
            plot_spatial_instability(
                plt, figure_dir, {"epochs": epochs, "inst_s": inst_s}, phase_epochs
            )
            print(f"{path.relative_to(ROOT)}: phases={phase_epochs} -> {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
