"""Euclidean-distance distributions from random EMNIST-digits training points.

Samples ``--num-queries`` training images, measures the Euclidean distance from
each one to every other training image (raw pixels scaled to [0, 1], 784-d),
and plots the distributions: uncolored, colored by the other point's class,
and per query split into same-class / other-class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

if __package__:
    from .plot_seed_sweep import RESULTS_DIR
else:
    from plot_seed_sweep import RESULTS_DIR


DATA_ROOT = RESULTS_DIR.parent / "data" / "emnist_digits"
DEFAULT_PREFIX = RESULTS_DIR / "viz" / "emnist_digits_pixel_distances"
CLASS_COLORS = plt.get_cmap("tab10").colors
SAME_COLOR = "#1f77b4"   # blue
OTHER_COLOR = "#ff7f0e"  # orange
BINS = 120


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--formats", default="pdf,png")
    return parser


def load_train() -> tuple[torch.Tensor, np.ndarray]:
    dataset = datasets.EMNIST(
        root=DATA_ROOT, split="digits", train=True, download=False
    )
    images = dataset.data.reshape(len(dataset.data), -1).float() / 255.0
    return images, dataset.targets.numpy()


def distances_to_all(images: torch.Tensor, queries: np.ndarray) -> np.ndarray:
    """Return a (num_queries, N) float64 distance matrix; self-distance is NaN."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = images.to(device, torch.float64)
    query = data[torch.as_tensor(queries, device=device)]
    result = torch.cdist(query, data).cpu().numpy()
    result[np.arange(len(queries)), queries] = np.nan
    return result


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
    images, labels = load_train()
    rng = np.random.default_rng(args.seed)
    queries = np.sort(rng.choice(len(labels), size=args.num_queries, replace=False))
    dist = distances_to_all(images, queries)
    others = ~np.isnan(dist)
    edges = np.linspace(0.0, np.nanmax(dist) * 1.01, BINS + 1)
    condition = (
        f"EMNIST-digits train (N={len(labels):,}), raw pixels in [0,1] (784-d), "
        f"{args.num_queries} random queries (seed={args.seed})"
    )

    # Summary table.
    print(f"queries: {queries.tolist()}")
    print(f"{'idx':>7} {'lbl':>3} {'mean':>6} {'min':>6} {'same':>6} {'other':>6} {'1NN lbl':>7}")
    for row, index in enumerate(queries):
        d, same = dist[row], labels == labels[index]
        nn = int(np.nanargmin(d))
        print(
            f"{index:>7} {labels[index]:>3} {np.nanmean(d):6.2f} {np.nanmin(d):6.2f} "
            f"{np.nanmean(d[same]):6.2f} {np.nanmean(d[~same]):6.2f} {labels[nn]:>7}"
        )
    pooled = dist[others]
    print(f"pooled: n={pooled.size:,} mean={pooled.mean():.3f} std={pooled.std():.3f}")

    # 1. Pooled histogram, uncolored.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pooled, bins=edges, color="#555555")
    ax.axvline(pooled.mean(), color="black", ls="--", lw=1,
               label=f"mean = {pooled.mean():.2f}")
    ax.set(xlabel="Euclidean distance", ylabel="Count",
           title=f"All query-to-other distances (n={pooled.size:,})")
    ax.legend(frameon=False)
    fig.suptitle(condition, fontsize=8, y=1.0)
    save(fig, args.prefix, "all", formats)

    # 2. Pooled, colored by the other point's class.
    other_labels = np.broadcast_to(labels, dist.shape)[others]
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in range(10):
        ax.hist(pooled[other_labels == c], bins=edges, histtype="step", lw=1.5,
                color=CLASS_COLORS[c], label=str(c))
    ax.set(xlabel="Euclidean distance", ylabel="Count",
           title="Query-to-other distances by class of the other point")
    ax.legend(title="class", ncols=2, frameon=False, fontsize=8)
    fig.suptitle(condition, fontsize=8, y=1.0)
    save(fig, args.prefix, "by_other_class", formats)

    # 3. Per query: one panel each, colored by the other point's class.
    #    The query's own class is drawn thicker.
    rows = int(np.ceil(len(queries) / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(16, 3.2 * rows), sharex=True,
                             squeeze=False)
    for row, (ax, index) in enumerate(zip(axes.flat, queries)):
        own = labels[index]
        for c in range(10):
            mask = others[row] & (labels == c)
            ax.hist(dist[row, mask], bins=edges, histtype="step",
                    lw=2.5 if c == own else 1.0, color=CLASS_COLORS[c],
                    zorder=3 if c == own else 2)
        ax.set_title(f"idx {index}, label {own}", fontsize=9)
        thumb = ax.inset_axes([0.02, 0.6, 0.28, 0.38])
        thumb.imshow(images[index].reshape(28, 28).T, cmap="gray_r")
        thumb.set_axis_off()
    for ax in axes.flat[len(queries):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("Euclidean distance")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    handles = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=1.5) for c in range(10)]
    fig.legend(handles, [str(c) for c in range(10)], title="class of other point",
               loc="upper center", ncols=10, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(condition + " — thick line = query's own class", fontsize=9,
                 y=1.06)
    fig.tight_layout()
    save(fig, args.prefix, "per_query_by_class", formats)

    # 4. Per query: same class vs other classes (density, so shapes compare).
    fig, axes = plt.subplots(rows, 5, figsize=(16, 3.2 * rows), sharex=True,
                             squeeze=False)
    for row, (ax, index) in enumerate(zip(axes.flat, queries)):
        same = labels == labels[index]
        ax.hist(dist[row, others[row] & same], bins=edges, density=True,
                histtype="step", lw=1.5, color=SAME_COLOR)
        ax.hist(dist[row, others[row] & ~same], bins=edges, density=True,
                histtype="step", lw=1.5, color=OTHER_COLOR)
        ax.set_title(f"idx {index}, label {labels[index]}", fontsize=9)
        thumb = ax.inset_axes([0.02, 0.6, 0.28, 0.38])
        thumb.imshow(images[index].reshape(28, 28).T, cmap="gray_r")
        thumb.set_axis_off()
    for ax in axes.flat[len(queries):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("Euclidean distance")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.legend([plt.Line2D([], [], color=SAME_COLOR, lw=1.5),
                plt.Line2D([], [], color=OTHER_COLOR, lw=1.5)],
               ["same class as query", "other classes"],
               loc="upper center", ncols=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(condition, fontsize=9, y=1.06)
    fig.tight_layout()
    save(fig, args.prefix, "per_query_same_vs_other", formats)

    # 5. Distances among the sampled points themselves.
    sub = dist[:, queries]
    np.fill_diagonal(sub, 0.0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(sub, cmap="viridis")
    ticks = [f"{i}\n(y={labels[i]})" for i in queries]
    ax.set_xticks(range(len(queries)), ticks, fontsize=6)
    ax.set_yticks(range(len(queries)), ticks, fontsize=6)
    for i in range(len(queries)):
        for j in range(len(queries)):
            ax.text(j, i, f"{sub[i, j]:.1f}", ha="center", va="center",
                    fontsize=6, color="white" if sub[i, j] < sub.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Euclidean distance")
    ax.set_title("Pairwise distances among the sampled points", fontsize=9)
    fig.suptitle(condition, fontsize=7, y=1.0)
    save(fig, args.prefix, "query_pairwise", formats)

    np.savez_compressed(
        args.prefix.with_name(f"{args.prefix.name}.npz"),
        queries=queries, query_labels=labels[queries], distances=dist,
    )


if __name__ == "__main__":
    main()
