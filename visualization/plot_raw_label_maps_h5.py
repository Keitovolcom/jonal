"""Raw per-pair predictions on the (alpha, epoch) plane from the alpha_h5 logs.

Each panel is one pair: x is alpha on the interpolation line, y is the epoch
(log scale), and the colour is the predicted label, classed as c (the pair's
original/clean label, blue), n (the noisy label of a clean-noisy pair, red) or
o (any other label, grey). Changes between two different o labels are not
visible. Dashed lines mark the phase boundaries of ``plot_phase_temporal_h5``.

    python visualization/plot_raw_label_maps_h5.py [--num-pairs 16] [--pages 1] [--seed 0]
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visualization.plot_phase_temporal_h5 import KINDS, PHASE_EPOCHS  # noqa: E402

C, N, O = 0, 1, 2
CMAP = ListedColormap(["#2a78d6", "#e34948", "#d4d4d0"])
KIND_TITLES = {"noise": "clean-noisy", "no_noise": "clean-clean", "test_test_no_noise": "test-test"}


def load_pairs(path: Path, num: int, rng: np.random.Generator):
    """Predicted labels classed as c/n/o for ``num`` random pairs, [pairs, epochs, alphas]."""

    with h5py.File(path, "r") as log:
        names = log["pair_names"][:]
        index = np.sort(rng.choice(len(names), size=min(num, len(names)), replace=False))
        preds = log["predicted_label"][index]
        epochs = log["epochs"][:]
        alphas = np.round(log["alpha"][:].astype(float), 2)
    names = [names[i].decode() for i in index]
    labels = np.array([[int(x) for x in n.split("/")[-1].split("_")] for n in names])
    clean, noisy = labels[:, 0], labels[:, 1]
    kind = np.full(preds.shape, O, dtype=np.int8)
    kind[preds == clean[:, None, None]] = C
    has_noisy = clean != noisy
    kind[(preds == noisy[:, None, None]) & has_noisy[:, None, None]] = N
    return kind, names, has_noisy, epochs, alphas


def plot_page(path: Path, title: str, kind, names, has_noisy, epochs, alphas, phase_epochs):
    num = len(names)
    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))
    keep = epochs >= 1  # log-scaled epoch axis
    ep = epochs[keep].astype(float)
    y_edges = np.r_[ep - 0.5, ep[-1] + 0.5]
    step = alphas[1] - alphas[0]
    x_edges = np.r_[alphas - step / 2, alphas[-1] + step / 2]

    figure, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows + 0.8),
                                sharex=True, sharey=True, squeeze=False)
    for axis in axes.flat[num:]:
        axis.set_visible(False)
    for axis, k, name, noisy in zip(axes.flat, kind, names, has_noisy):
        axis.pcolormesh(x_edges, y_edges, k[keep], cmap=CMAP, vmin=0, vmax=2,
                        shading="flat", rasterized=True)
        axis.set_yscale("log")
        for alpha in (0.0, 1.0):
            axis.axvline(alpha, color="black", linewidth=0.8)
        for boundary in phase_epochs:
            axis.axhline(boundary, color="black", linestyle="--", linewidth=0.9)
        pair_id, labels = name.split("/")
        c_label, n_label = labels.split("_")
        axis.set_title(f"{pair_id}  c={c_label}" + (f", n={n_label}" if noisy else ""),
                       fontsize=10)
        axis.set_xticks([0.0, 1.0])
        axis.set_xticklabels([r"$x_0$", r"$x_1$"])
    for axis in axes[:, 0]:
        axis.set_ylabel("epoch")
    handles = [Patch(color=CMAP(0), label="c (original label)"),
               Patch(color=CMAP(1), label="n (noisy label)"),
               Patch(color=CMAP(2), label="o (other label)")]
    if not has_noisy.any():
        handles.pop(1)
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.03, 1, 0.98))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5-root", type=Path, default=ROOT / "alpha_h5/alpha_test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis_alpha/figs/raw_label_maps")
    parser.add_argument("--num-pairs", type=int, default=16, help="pairs per page")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for (dataset, gamma), phase_epochs in PHASE_EPOCHS.items():
        for kind_name in KINDS[gamma]:
            path = args.h5_root / dataset / gamma / "64" / f"{kind_name}.h5"
            # One draw per file, split into pages, so pages never repeat a pair.
            rng = np.random.default_rng(args.seed)
            kind, names, has_noisy, epochs, alphas = load_pairs(
                path, args.num_pairs * args.pages, rng)
            for page in range(args.pages):
                part = slice(page * args.num_pairs, (page + 1) * args.num_pairs)
                if not names[part]:
                    break
                out = args.out_dir / f"{dataset}_{gamma}_{kind_name}_page{page + 1}.png"
                title = f"{dataset}  gamma={gamma}  {KIND_TITLES[kind_name]}  (page {page + 1})"
                plot_page(out, title, kind[part], names[part], has_noisy[part],
                          epochs, alphas, phase_epochs)
                print(out.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
