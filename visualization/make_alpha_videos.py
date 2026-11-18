"""Render per-epoch interpolation videos from the ``alpha_h5`` files.

For each target (dataset/gamma) and pair type a fixed random subset of pairs is
drawn on a grid (one panel per pair, class probabilities along alpha), and one
mp4 is written per phase and page. Every target also gets a video over
``--start-epoch``..``--end-epoch`` ("all"). Static artists are rendered
once; each frame only redraws the probability lines and the epoch label
(blitting on the Agg buffer).
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
H5_ROOT = PROJECT_ROOT / "alpha_h5" / "alpha_test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "viz" / "alpha_videos"

TYPES = ("no_noise", "noise", "test_test_no_noise")
DEFAULT_TARGETS = "emnist_digits/0.2,emnist_digits/0.0,cifar10/0.2,cifar10/0.0"
# Inclusive epoch ranges; boundary epochs appear in both neighbouring phases.
PHASES = {("emnist_digits", "0.2"): (("phase1", 0, 47), ("phase2", 47, 75), ("phase3", 75, 200))}
CLEAN_COLOR, NOISY_COLOR = "blue", "red"
OTHER_COLORS = ("green", "orange", "purple", "brown", "gray",
                "pink", "olive", "cyan", "lime", "navy")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=DEFAULT_TARGETS, help="Comma-separated <dataset>/<gamma>.")
    parser.add_argument("--width", default="64", help="Model width directory under alpha_h5.")
    parser.add_argument("--types", default=",".join(TYPES), help="Types missing for a target are skipped.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Range of the unsplit 'all' video.")
    parser.add_argument("--end-epoch", type=int, default=200)
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--pairs-per-page", type=int, default=15)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--video-size", default="1920x1080")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def pair_labels(name: str) -> tuple[int, int]:
    # Same parsing as analysis_alpha/scripts/extract.py: the 2nd path component is "<a>_<b>".
    a, b = name.split("/")[1].split("_")
    return int(a), int(b)


def select_pairs(h5_path: Path, num_pairs: int, seed: int) -> list[tuple[int, str]]:
    with h5py.File(h5_path, "r") as f:
        names = [n.decode() for n in f["pair_names"][:]]
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(names), size=num_pairs, replace=False))
    return [(int(i), names[i]) for i in indices]


def load_probs(h5_path: Path, pair_indices: list[int], epochs: list[int]):
    """Return alphas and probabilities shaped (pairs, epochs, classes, alphas)."""
    with h5py.File(h5_path, "r") as f:
        alphas = f["alpha"][:]
        stored = f["epochs"][:]
        positions = np.searchsorted(stored, epochs)
        if (positions >= len(stored)).any() or (stored[positions] != epochs).any():
            raise ValueError(f"{h5_path} lacks some of epochs {epochs[0]}..{epochs[-1]}")
        lo, hi = positions.min(), positions.max() + 1
        probs = np.stack([f["probs"][p, lo:hi][positions - lo] for p in pair_indices])
    return alphas, np.ascontiguousarray(probs.transpose(0, 1, 3, 2), dtype=np.float32)


def render_page(job: dict) -> list[Path]:
    pair_type, pairs, page = job["pair_type"], job["pairs"], job["page"]
    width, height = job["video_size"]
    phases = job["phases"]
    epochs = list(range(min(lo for _, lo, _ in phases), max(hi for _, _, hi in phases) + 1))
    epoch_pos = {e: i for i, e in enumerate(epochs)}
    alphas, probs = load_probs(job["h5_path"], [i for i, _ in pairs], epochs)
    num_classes = probs.shape[2]

    plt.rcParams["font.size"] = 10
    dpi = 100
    cols = job["cols"]
    rows = -(-job["pairs_per_page"] // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(width / dpi, height / dpi), dpi=dpi,
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.06, top=0.89, wspace=0.08, hspace=0.32)

    noisy_type = pair_type == "noise"
    # Per panel, other classes first so the highlighted lines are drawn on top.
    panel_lines: list[list[tuple[Line2D, int]]] = []
    for ax, (_, name) in zip(axes.flat, pairs):
        a, b = pair_labels(name)
        highlight = {a: CLEAN_COLOR, b: NOISY_COLOR if noisy_type else CLEAN_COLOR}
        others, highlighted = [], []
        for t in range(num_classes):
            if t in highlight:
                line, = ax.plot(alphas, probs[0, 0, t], color=highlight[t], lw=1.8, animated=True)
                highlighted.append((line, t))
            else:
                line, = ax.plot(alphas, probs[0, 0, t], color=OTHER_COLORS[t % len(OTHER_COLORS)],
                                ls="--", lw=0.8, animated=True)
                others.append((line, t))
        panel_lines.append(others + highlighted)

        ax.axvline(0.0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(1.0, color="0.85", lw=0.6, zorder=0)
        ax.plot(0.0, 0.0, "o", color=CLEAN_COLOR, ms=5, zorder=5, clip_on=False)
        ax.plot(1.0, 0.0, "o", color=highlight[b], ms=5, zorder=5, clip_on=False)
        ax.set_title(name, fontsize=8)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)

    ax = axes[0, 0]
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r"$x_0$", r"$x_1$"], fontsize=12)
    ax.set_yticks([0.0, 0.5, 1.0])

    handles = [Line2D([0], [0], color=CLEAN_COLOR, lw=1.8, label="label of $x_0$ / $x_1$")]
    if noisy_type:
        handles = [Line2D([0], [0], color=CLEAN_COLOR, lw=1.8, label="clean label"),
                   Line2D([0], [0], color=NOISY_COLOR, lw=1.8, label="noisy label")]
    handles.append(Line2D([0], [0], color="black", ls="--", lw=0.8, label="other labels"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False)
    fig.text(0.04, 0.965, f"{job['dataset']}  γ={job['gamma']}  {pair_type}  page {page + 1}",
             fontsize=13, va="center")
    epoch_text = fig.text(0.5, 0.965, "", fontsize=16, ha="center", va="center",
                          weight="bold", animated=True)

    canvas = fig.canvas
    canvas.draw()  # animated artists are skipped, so this is the static background
    if canvas.get_width_height() != (width, height):
        raise RuntimeError(f"canvas is {canvas.get_width_height()}, expected {(width, height)}")
    background = canvas.copy_from_bbox(fig.bbox)

    out_dir = job["out_dir"]
    written = []
    for stem, lo, hi in phases:
        path = out_dir / f"{stem}_ep{lo:03d}-{hi:03d}_page{page + 1}.mp4"
        label = "" if stem == "all" else f"{stem.replace('phase', 'phase ')}   "
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), job["fps"], (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {path}")
        for epoch in range(lo, hi + 1):
            e = epoch_pos[epoch]
            canvas.restore_region(background)
            for i, lines in enumerate(panel_lines):
                for line, t in lines:
                    line.set_ydata(probs[i, e, t])
                    fig.draw_artist(line)
            epoch_text.set_text(f"{label}epoch {epoch}")
            fig.draw_artist(epoch_text)
            writer.write(cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        writer.release()
        written.append(path)
    plt.close(fig)
    return written


def make_jobs(args, dataset, gamma, pair_type, h5_path, phases, video_size) -> list[dict]:
    out_dir = args.output_dir / f"{dataset}_{gamma}" / pair_type
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_pairs(h5_path, args.num_pairs, args.seed)
    with open(out_dir / "selected_pairs.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["page", "h5_index", "pair_name"])
        for j, (i, name) in enumerate(pairs):
            w.writerow([j // args.pairs_per_page + 1, i, name])
    return [{
        "pair_type": pair_type, "h5_path": h5_path, "out_dir": out_dir,
        "pairs": pairs[start:start + args.pairs_per_page], "page": page,
        "pairs_per_page": args.pairs_per_page, "cols": args.cols, "phases": phases,
        "video_size": video_size, "fps": args.fps, "dataset": dataset, "gamma": gamma,
    } for page, start in enumerate(range(0, len(pairs), args.pairs_per_page))]


def main() -> None:
    args = create_parser().parse_args()
    width, height = map(int, args.video_size.split("x"))

    jobs = []
    for target in args.targets.split(","):
        dataset, gamma = target.split("/")
        phases = PHASES.get((dataset, gamma), ()) + (("all", args.start_epoch, args.end_epoch),)
        for pair_type in args.types.split(","):
            h5_path = H5_ROOT / dataset / gamma / args.width / f"{pair_type}.h5"
            if h5_path.exists():
                jobs += make_jobs(args, dataset, gamma, pair_type, h5_path, phases, (width, height))
            else:
                print(f"skip {h5_path} (not found)", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for paths in pool.map(render_page, jobs):
            for path in paths:
                print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
