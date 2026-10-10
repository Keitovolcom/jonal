"""Linear-interpolation inference between sample pairs over saved checkpoints.

For each pair ``(x0, x1)`` the inputs ``x(alpha) = (1 - alpha) * x0 + alpha * x1``
are evaluated with every checkpoint of a ``train.py`` run. Each pair's softmax
probabilities and predicted labels for all epochs are written to one file:

    interpolation/<experiment_name>/<mode>/pair_XXXX/log.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import create_model
from train import DATASET_SPECS, nonnegative_int, positive_int, resolve_device


MODES = ("clean-noisy", "clean-clean", "test-test")
CHECKPOINT_PATTERN = re.compile(r"epoch_(\d+)\.pt$")


def parse_epochs(value: str) -> list[int]:
    """Parse a comma-separated list such as ``0,1,2,10-20``."""

    epochs: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid epoch range: {token}")
            epochs.update(range(start, end + 1))
        else:
            epochs.add(int(token))
    if not epochs:
        raise argparse.ArgumentTypeError("epochs must not be empty")
    return sorted(epochs)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate linear interpolations between sample pairs for each checkpoint."
    )
    parser.add_argument(
        "--run-dir", "--run_dir", dest="run_dir", type=Path, required=True,
        help="train.py result directory containing config.json and checkpoint/",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--num-pairs", "--num_pairs", dest="num_pairs", type=positive_int, default=1000
    )
    parser.add_argument(
        "--pair-seed", "--pair_seed", dest="pair_seed", type=nonnegative_int, default=0,
        help="Seed for choosing query samples",
    )
    parser.add_argument("--alpha-start", dest="alpha_start", type=float, default=-0.5)
    parser.add_argument("--alpha-end", dest="alpha_end", type=float, default=1.5)
    parser.add_argument("--alpha-step", dest="alpha_step", type=float, default=0.01)
    parser.add_argument(
        "--epochs", type=parse_epochs, default=None,
        help="Checkpoint epochs to evaluate, e.g. '0,1,2,10-20' (default: all)",
    )
    parser.add_argument(
        "--experiment-name", "--experiment_name", dest="experiment_name", default=None,
        help="Defaults to the name of --run-dir",
    )
    parser.add_argument(
        "--output-root", "--output_root", dest="output_root", type=Path,
        default=Path("interpolation"),
    )
    parser.add_argument("--gpu", default="auto")
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=positive_int, default=2048,
        help="Number of interpolated inputs per forward pass",
    )
    parser.add_argument(
        "--pairs-per-chunk", "--pairs_per_chunk", dest="pairs_per_chunk",
        type=positive_int, default=100,
        help="Pairs kept in memory while sweeping all checkpoints",
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers",
        type=nonnegative_int, default=4,
    )
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser


def make_alphas(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0 or end < start:
        raise ValueError("alpha range must satisfy start <= end and step > 0")
    count = int(round((end - start) / step)) + 1
    return np.round(start + step * np.arange(count), 10)


def find_checkpoints(run_dir: Path, epochs: list[int] | None) -> dict[int, Path]:
    available = {}
    for path in (run_dir / "checkpoint").glob("epoch_*.pt"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match:
            available[int(match.group(1))] = path
    if not available:
        raise FileNotFoundError(f"no checkpoints found under {run_dir / 'checkpoint'}")
    if epochs is None:
        return dict(sorted(available.items()))
    missing = [epoch for epoch in epochs if epoch not in available]
    if missing:
        raise FileNotFoundError(f"checkpoints not found for epochs: {missing}")
    return {epoch: available[epoch] for epoch in epochs}


def load_images(dataset, num_workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=num_workers)
    return torch.cat([batch[0] for batch in loader])


def nearest_neighbors(
    queries: torch.Tensor,
    query_indices: np.ndarray,
    candidates: torch.Tensor,
    candidate_indices: np.ndarray,
    device: torch.device,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest candidate index and L2 distance for each query."""

    candidate_matrix = candidates.flatten(1).to(device)
    candidate_ids = torch.as_tensor(candidate_indices, device=device)
    nearest = np.empty(len(queries), dtype=np.int64)
    distances = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk_size):
        stop = start + chunk_size
        query_matrix = queries[start:stop].flatten(1).to(device)
        ids = torch.as_tensor(query_indices[start:stop], device=device)
        squared = torch.cdist(query_matrix, candidate_matrix).square()
        # A sample is never its own neighbour.
        squared[ids[:, None].eq(candidate_ids[None, :])] = float("inf")
        values, positions = squared.min(dim=1)
        nearest[start:stop] = candidate_indices[positions.cpu().numpy()]
        distances[start:stop] = values.sqrt().cpu().numpy()
    return nearest, distances


def build_pairs(
    mode: str,
    num_pairs: int,
    seed: int,
    train_images: torch.Tensor | None,
    original: np.ndarray | None,
    training: np.ndarray | None,
    test_images: torch.Tensor | None,
    test_labels: np.ndarray | None,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Pick query samples x0 and pair each with its nearest same-label x1."""

    if mode == "test-test":
        split, images, labels = "test", test_images, test_labels
        query_pool = np.arange(len(labels))
        candidate_pool = query_pool
    else:
        split, images, labels = "train", train_images, original
        is_noisy = original != training
        clean = np.flatnonzero(~is_noisy)
        # clean-noisy: a noisy query is paired with the nearest clean sample
        # sharing the query's original (correct) label.
        query_pool = np.flatnonzero(is_noisy) if mode == "clean-noisy" else clean
        candidate_pool = clean
    if len(query_pool) < num_pairs:
        raise ValueError(
            f"{mode}: requested {num_pairs} pairs but only {len(query_pool)} queries exist"
        )

    rng = np.random.default_rng(seed)
    queries = rng.choice(query_pool, size=num_pairs, replace=False)
    partners = np.empty(num_pairs, dtype=np.int64)
    distances = np.empty(num_pairs, dtype=np.float64)
    for label in np.unique(labels[queries]):
        positions = np.flatnonzero(labels[queries] == label)
        candidates = candidate_pool[labels[candidate_pool] == label]
        nearest, distance = nearest_neighbors(
            images[queries[positions]], queries[positions],
            images[candidates], candidates, device,
        )
        partners[positions] = nearest
        distances[positions] = distance

    def describe(index: int) -> dict[str, Any]:
        if split == "test":
            label = int(test_labels[index])
            return {"split": "test", "index": int(index), "original_label": label,
                    "training_label": label, "is_noisy": False}
        return {"split": "train", "index": int(index),
                "original_label": int(original[index]),
                "training_label": int(training[index]),
                "is_noisy": bool(original[index] != training[index])}

    return [
        {"pair_id": pair_id, "x0": describe(x0), "x1": describe(x1),
         "distance": float(distance)}
        for pair_id, (x0, x1, distance) in enumerate(zip(queries, partners, distances))
    ]


def pair_directory(mode_dir: Path, pair_id: int) -> Path:
    return mode_dir / f"pair_{pair_id:04d}"


def save_pair_log(
    path: Path,
    pair: dict[str, Any],
    probs: np.ndarray,
    epochs: np.ndarray,
    alphas: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for end in ("x0", "x1"):
        for key, value in pair[end].items():
            metadata[f"{end}_{key}"] = np.asarray(value)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        temporary,
        probs=probs.astype(np.float32, copy=False),
        preds=probs.argmax(axis=-1).astype(np.int16),
        epochs=epochs.astype(np.int64),
        alphas=alphas.astype(np.float64),
        pair_id=np.asarray(pair["pair_id"]),
        distance=np.asarray(pair["distance"]),
        **metadata,
    )
    temporary.replace(path)


def load_or_create_pairs(
    mode_dir: Path, settings: dict[str, Any], create
) -> list[dict[str, Any]]:
    path = mode_dir / "pairs.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["settings"] != settings:
            raise ValueError(
                f"{path} was created with different settings; "
                "use another --output-root or remove the directory"
            )
        return stored["pairs"]
    pairs = create()
    mode_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"settings": settings, "pairs": pairs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return pairs


@torch.inference_mode()
def run_interpolation(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    experiment_name = args.experiment_name or run_dir.resolve().name
    mode_dir = args.output_root / experiment_name / args.mode
    device = resolve_device(args.gpu)
    alphas = make_alphas(args.alpha_start, args.alpha_end, args.alpha_step)
    checkpoints = find_checkpoints(run_dir, args.epochs)
    epochs = np.asarray(list(checkpoints), dtype=np.int64)

    # Augmentation is disabled so train images use the evaluation transform.
    train_dataset, test_dataset = load_dataset(
        config["dataset"],
        data_augmentation=False,
        label_noise_rate=config["label_noise"],
        seed=config["seed"],
        input_channels=config["input_channels"],
        data_dir=args.data_dir,
        download=not args.no_download,
    )
    if args.mode == "test-test":
        train_images = original = training = None
        test_images = load_images(test_dataset, args.num_workers)
        test_labels = np.asarray(test_dataset.targets, dtype=np.int64)
    else:
        train_images = load_images(train_dataset, args.num_workers)
        original = train_dataset.original_targets.numpy()
        training = train_dataset.training_targets.numpy()
        test_images = test_labels = None

    settings = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "num_pairs": args.num_pairs,
        "pair_seed": args.pair_seed,
        "dataset": config["dataset"],
        "label_noise": config["label_noise"],
        "seed": config["seed"],
    }
    pairs = load_or_create_pairs(
        mode_dir, settings,
        lambda: build_pairs(
            args.mode, args.num_pairs, args.pair_seed, train_images, original,
            training, test_images, test_labels, device,
        ),
    )
    (mode_dir / "config.json").write_text(
        json.dumps(
            {**settings, "experiment_name": experiment_name,
             "alphas": alphas.tolist(), "epochs": epochs.tolist(),
             "train_config": config},
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    pending = [
        pair for pair in pairs
        if not (pair_directory(mode_dir, pair["pair_id"]) / "log.npz").exists()
    ]
    print(f"{mode_dir}: {len(pairs) - len(pending)} done, {len(pending)} pending, "
          f"{len(epochs)} epochs, {len(alphas)} alphas", flush=True)
    if not pending:
        return mode_dir

    specification = DATASET_SPECS[config["dataset"]]
    model_options = {
        "in_channels": config["input_channels"],
        "num_classes": specification["num_classes"],
        "width_multiplier": config["model_width"],
    }
    if config["model"] == "cnn5layer":
        model_options["input_size"] = specification["input_size"]
    model = create_model(config["model"], **model_options).to(device)
    model.eval()

    def image(end: dict[str, Any]) -> torch.Tensor:
        source = test_images if end["split"] == "test" else train_images
        return source[end["index"]]

    # Shape [1, alphas, 1, 1, 1] broadcasts over [pairs, alphas, C, H, W].
    alpha_tensor = torch.as_tensor(
        alphas, dtype=torch.float32, device=device
    ).view(1, -1, 1, 1, 1)
    num_classes = specification["num_classes"]

    for start in range(0, len(pending), args.pairs_per_chunk):
        chunk = pending[start:start + args.pairs_per_chunk]
        x0 = torch.stack([image(pair["x0"]) for pair in chunk]).to(device)
        x1 = torch.stack([image(pair["x1"]) for pair in chunk]).to(device)
        # [pairs, alphas, C, H, W] -> [pairs * alphas, C, H, W]
        inputs = (x0.unsqueeze(1) + alpha_tensor * (x1 - x0).unsqueeze(1)).flatten(0, 1)
        probs = np.empty((len(chunk), len(epochs), len(alphas), num_classes), dtype=np.float32)

        for epoch_position, (epoch, path) in enumerate(checkpoints.items()):
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            outputs = [
                model(inputs[i:i + args.batch_size]).float().softmax(dim=1)
                for i in range(0, len(inputs), args.batch_size)
            ]
            probs[:, epoch_position] = (
                torch.cat(outputs).view(len(chunk), len(alphas), num_classes).cpu().numpy()
            )
            print(f"pairs {start}-{start + len(chunk) - 1}: epoch {epoch}", flush=True)

        for pair, pair_probs in zip(chunk, probs):
            save_pair_log(
                pair_directory(mode_dir, pair["pair_id"]) / "log.npz",
                pair, pair_probs, epochs, alphas,
            )
    return mode_dir


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    mode_dir = run_interpolation(args)
    print(f"results={mode_dir}", flush=True)


if __name__ == "__main__":
    main()
