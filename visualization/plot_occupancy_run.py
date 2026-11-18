"""Predicted-label occupancy on the interpolation lines of a single run, epoch-wise.

Reads ``<h5-dir>/{noise,no_noise,test_test_no_noise}.h5`` written by
``interpolation_h5.py`` and, per mode, splits the points of every line by their
predicted label (``interpolation_metrics.label_occupancy``: original / noisy /
others), averaged over pairs. Same figure style as
``plot_width_sweep_occupancy.py``'s ``epoch_occupancy_k-<k>``.

Writes, under ``<run-dir>/viz/alpha_interpolation/occupancy`` by default:
    occupancy.csv                      epoch, mean/std per alpha range, mode and label kind
    occupancy{,_alpha-in}.{png,pdf}    one row per mode, x = epoch (log)

    python visualization/plot_occupancy_run.py --run-dir results/<name> --h5-dir <alpha_h5>/<name>/<ds>/<gamma>/64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_width_sweep_occupancy import occupancy_stats, plot_occupancy_rows  # noqa: E402
from plot_width_sweep_spatial import ALPHA_RANGES, MODES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--h5-dir", type=Path, required=True,
                        help="directory holding noise.h5, no_noise.h5, test_test_no_noise.h5")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/viz/alpha_interpolation/occupancy")
    parser.add_argument("--vlines", type=float, nargs="*", default=[],
                        help="epochs marked with dotted lines")
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_dir / "viz" / "alpha_interpolation" / "occupancy"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for mode, _, _ in MODES:
        stats = occupancy_stats(args.h5_dir / f"{mode}.h5", mode)
        arrays["epochs"] = stats.pop("epochs")
        arrays.update(stats)
        print(f"{mode}: done", flush=True)
    df = pd.DataFrame({"epoch": arrays.pop("epochs"),
                       **{f"occ_{key}": value for key, value in arrays.items()}})
    df.to_csv(out_dir / "occupancy.csv", index=False)

    df = df[df["epoch"] >= 1]  # epoch 0 has no place on a log axis
    for rng in ALPHA_RANGES:
        suffix = "" if rng == "all" else "_alpha-in"
        saved = plot_occupancy_rows(df, "epoch", rng, out_dir / f"occupancy{suffix}",
                                    vlines=args.vlines)
        print(f"[✓] Saved: {', '.join(saved)}")
    print(f"[✓] Saved: {out_dir / 'occupancy.csv'}")


if __name__ == "__main__":
    main()
