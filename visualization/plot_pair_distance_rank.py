"""Where does an interpolation partner sit in its query's distance distribution?

Rebuilds the pairs of ``interpolation.py`` (same rule and pair seed), measures
the distance from each query x0 to every training image in the model's input
space, and reports the rank of the partner x1 together with the composition of
the points that are closer than x1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .plot_seed_sweep import PROJECT_ROOT, RESULTS_DIR
else:
    from plot_seed_sweep import PROJECT_ROOT, RESULTS_DIR

sys.path.insert(0, str(PROJECT_ROOT))
from dataset import load_dataset  # noqa: E402
from interpolation import build_pairs, load_images  # noqa: E402


DEFAULT_PREFIX = RESULTS_DIR / "viz" / "emnist_digits_pair_distance_rank"
# Kinds of points closer to x0 than its partner, relative to x0's original label.
KINDS = {
    "clean, same label": "#1f77b4",
    "noisy, same original label": "#2ca02c",
    "clean, other label": "#ff7f0e",
    "noisy, other original label": "#d62728",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("clean-noisy", "clean-clean"),
                        default="clean-noisy")
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--label-noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--formats", default="pdf,png")
    return parser


def save(fig: plt.Figure, prefix: Path, name: str, formats: list[str]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = prefix.with_name(f"{prefix.name}_{name}.{extension}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = create_parser().parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, _ = load_dataset(
        "emnist-digits", data_augmentation=False,
        label_noise_rate=args.label_noise, seed=args.seed, download=False,
    )
    images = load_images(train_dataset, num_workers=4)
    original = train_dataset.original_targets.numpy()
    training = train_dataset.training_targets.numpy()
    is_noisy = original != training
    pairs = build_pairs(args.mode, args.num_pairs, args.pair_seed, images,
                        original, training, None, None, device)
    x0 = np.array([p["x0"]["index"] for p in pairs])
    x1 = np.array([p["x1"]["index"] for p in pairs])

    data = images.flatten(1).to(device)
    ranks = np.empty(len(pairs), dtype=np.int64)
    closer = np.zeros((len(pairs), len(KINDS)), dtype=np.int64)
    rows = []
    for start in range(0, len(pairs), 64):
        stop = min(start + 64, len(pairs))
        dist = torch.cdist(data[x0[start:stop]], data)
        dist[torch.arange(stop - start), torch.as_tensor(x0[start:stop])] = float("inf")
        dist = dist.cpu().numpy()
        for row, i in enumerate(range(start, stop)):
            d = dist[row]
            nearer = d < d[x1[i]]
            same = original == original[x0[i]]
            ranks[i] = nearer.sum() + 1
            closer[i] = [
                np.sum(nearer & same & ~is_noisy), np.sum(nearer & same & is_noisy),
                np.sum(nearer & ~same & ~is_noisy), np.sum(nearer & ~same & is_noisy),
            ]
            if i < args.num_examples:
                rows.append((i, d[np.isfinite(d)], d[x1[i]]))
    percentile = 100.0 * ranks / (len(original) - 1)
    condition = (
        f"EMNIST-digits train, noise={args.label_noise} (seed={args.seed}), "
        f"mode={args.mode}, {len(pairs)} pairs (pair_seed={args.pair_seed}), "
        f"model input space (32x32, normalized)"
    )

    print(condition)
    print("rank of x1 among all training points (1 = nearest):")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"  q{q:<4}: rank {np.quantile(ranks, q):8.0f}  "
              f"({np.quantile(percentile, q):.3f} %)")
    print(f"  x1 is the 1-NN of x0: {np.mean(ranks == 1):.3f}")
    print(f"  x1 within 10-NN: {np.mean(ranks <= 10):.3f}, "
          f"within 100-NN: {np.mean(ranks <= 100):.3f}")
    print("points closer than x1 (mean / median per pair):")
    for (kind, _), counts in zip(KINDS.items(), closer.T):
        print(f"  {kind:<28}: {counts.mean():8.1f} / {np.median(counts):6.0f}")
    others = closer[:, 2] + closer[:, 3]
    print(f"  pairs with >=1 other-label point closer: {np.mean(others > 0):.3f}")

    # 1. Example pairs: x0's distance distribution with the partner marked.
    ncols = 5
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows),
                             squeeze=False)
    for ax, (i, d, d1) in zip(axes.flat, rows):
        ax.hist(d, bins=120, color="#777777")
        ax.axvline(d1, color="#d62728", lw=1.5)
        ax.set_title(
            f"pair {i}: x0={x0[i]} (y={original[x0[i]]}->{training[x0[i]]})\n"
            f"x1 rank {ranks[i]:,} ({percentile[i]:.3f} %)", fontsize=8)
    for ax in axes.flat[len(rows):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("Distance from x0")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    fig.legend([plt.Line2D([], [], color="#d62728", lw=1.5)],
               ["distance to partner x1"], loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(condition, fontsize=9, y=1.05)
    fig.tight_layout()
    save(fig, args.prefix, f"{args.mode}_examples", formats)

    # 2. Rank of x1 over all pairs, and what lies closer than x1.
    fig, (ax_rank, ax_kind) = plt.subplots(1, 2, figsize=(12, 4))
    edges = np.logspace(0, np.log10(ranks.max() + 1), 40)
    ax_rank.hist(ranks, bins=edges, color="#555555")
    ax_rank.set(xscale="log", xlabel="Rank of x1 among all training points (log)",
                ylabel="Pairs", title="How near is the interpolation partner?")
    order = np.argsort(ranks)
    bottom = np.zeros(len(pairs))
    for (kind, color), counts in zip(KINDS.items(), closer.T):
        ax_kind.bar(np.arange(len(pairs)), counts[order], bottom=bottom[order],
                    width=1.0, color=color, label=kind)
        bottom += counts
    ax_kind.set(yscale="symlog", xlabel="Pairs sorted by rank of x1",
                ylabel="Points closer than x1 (symlog)",
                title="Composition of points closer than x1")
    ax_kind.legend(frameon=False, fontsize=8, title="relative to x0's original label")
    fig.suptitle(condition, fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, args.prefix, f"{args.mode}_rank", formats)

    np.savez_compressed(
        args.prefix.with_name(f"{args.prefix.name}_{args.mode}.npz"),
        x0=x0, x1=x1, ranks=ranks, closer=closer, kinds=np.array(list(KINDS)),
    )


if __name__ == "__main__":
    main()
