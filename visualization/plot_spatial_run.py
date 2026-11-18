"""Spatial instability INST_S(t) per epoch for a single run's interpolation_h5.py logs.

Reads ``<h5-dir>/{noise,no_noise,test_test_no_noise}.h5`` and, per mode, counts the
predicted-label changes between neighbouring alphas on every line (INST_S), then
writes the mean/std over pairs and draws them with
``interpolation_metrics.plot_spatial_instability``.

Writes, under ``<run-dir>/viz/alpha_interpolation/spatial_instability/<kind>`` by default:
    spatial_instability.csv    epoch, mean, std
    spatial_instability.{pdf,svg}

    python visualization/plot_spatial_run.py --run-dir results/<name> --h5-dir <alpha_h5>/<name>/<ds>/<gamma>/64
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

KINDS = ("noise", "no_noise", "test_test_no_noise")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--h5-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/viz/alpha_interpolation/spatial_instability")
    parser.add_argument("--phase-epochs", type=float, nargs="*", default=[])
    parser.add_argument("--pair-chunk", type=int, default=100,
                        help="pairs read at once (keeps memory bounded)")
    args = parser.parse_args()
    out_root = args.out_dir or args.run_dir / "viz" / "alpha_interpolation" / "spatial_instability"

    for kind in KINDS:
        path = args.h5_dir / f"{kind}.h5"
        with h5py.File(path, "r") as log:
            if not log["present"][:].all():
                raise ValueError(f"{path} has missing (pair, epoch) entries")
            epochs = log["epochs"][:]
            dataset = log["predicted_label"]
            inst_s = np.concatenate([
                spatial_instability(dataset[start:start + args.pair_chunk])
                for start in range(0, dataset.shape[0], args.pair_chunk)
            ])                                                    # [pairs, epochs]
        figure_dir = out_root / kind
        figure_dir.mkdir(parents=True, exist_ok=True)
        write_csv(figure_dir / "spatial_instability.csv",
                  ["epoch", "mean_spatial_instability", "std_spatial_instability"],
                  zip(epochs, inst_s.mean(axis=0), inst_s.std(axis=0)))
        plot_spatial_instability(plt, figure_dir, {"epochs": epochs, "inst_s": inst_s},
                                 args.phase_epochs)
        print(f"{path} -> {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
