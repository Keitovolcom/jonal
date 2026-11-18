"""Recompute per-sample training metrics from saved epoch checkpoints.

For every requested epoch, loads ``checkpoint/epoch_N.pt`` of a run, evaluates
the whole training set (with the noisy labels used for training) and stores
per-sample loss, prediction and logit margin in ``per_sample/metrics.npz``.

CNN5Layer has no batch normalization or dropout, so these are the exact
end-of-epoch values, unlike the running averages logged in ``metrics.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import create_model
from train import DATASET_SPECS, create_loss, resolve_device


def parse_epochs(value: str) -> list[int]:
    """Parse ``"40-200"`` or ``"40,50,60"`` (ranges are inclusive)."""
    epochs: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            epochs.extend(range(start, end + 1))
        else:
            epochs.append(int(token))
    return sorted(set(epochs))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute per-sample training metrics from checkpoints."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory with config.json")
    parser.add_argument("--epochs", type=parse_epochs, default=parse_epochs("30-200"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <run_dir>/per_sample/metrics.npz).",
    )
    return parser


@torch.inference_mode()
def evaluate_per_sample(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses, predictions, margins = [], [], []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        target = labels[start : start + batch_size]
        logits = model(batch)
        losses.append(criterion(logits, target))
        predictions.append(logits.argmax(dim=1))
        # Margin = label logit minus the largest other logit (< 0: misclassified).
        label_logit = logits.gather(1, target[:, None]).squeeze(1)
        others = logits.scatter(1, target[:, None], float("-inf"))
        margins.append(label_logit - others.max(dim=1).values)
    return (
        torch.cat(losses).float().cpu().numpy(),
        torch.cat(predictions).to(torch.int8).cpu().numpy(),
        torch.cat(margins).float().cpu().numpy(),
    )


def load_training_tensors(
    config: dict, device: torch.device, batch_size: int, num_workers: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    train_dataset, _ = load_dataset(
        config["dataset"],
        data_augmentation=config["augmentation"],
        label_noise_rate=config["label_noise"],
        seed=config["seed"],
        input_channels=config["input_channels"],
        data_dir=config["data_dir"],
        download=False,
    )
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    images, original, training, noisy = [], [], [], []
    for batch_images, batch_original, batch_training, batch_noisy in loader:
        images.append(batch_images)
        original.append(batch_original)
        training.append(batch_training)
        noisy.append(batch_noisy)
    # The whole training set fits on the GPU (240k x 1 x 32 x 32 floats ~ 1 GB).
    return (
        torch.cat(images).to(device),
        torch.cat(training).to(device),
        torch.cat(original).numpy().astype(np.int8),
        torch.cat(noisy).numpy().astype(bool),
    )


def main() -> None:
    args = create_parser().parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("augmentation"):
        raise ValueError("per-sample metrics assume no data augmentation")
    device = resolve_device(args.gpu)
    output = args.output or args.run_dir / "per_sample" / "metrics.npz"

    images, labels, original_labels, is_noisy = load_training_tensors(
        config, device, args.batch_size, args.num_workers
    )
    specification = DATASET_SPECS[config["dataset"]]
    model_options = {
        "in_channels": config["input_channels"],
        "num_classes": specification["num_classes"],
        "width_multiplier": config["model_width"],
    }
    if config["model"] == "cnn5layer":
        model_options["input_size"] = specification["input_size"]
    model = create_model(config["model"], **model_options).to(device)
    criterion = create_loss(config["loss"])

    losses, predictions, margins = [], [], []
    for epoch in args.epochs:
        checkpoint_path = args.run_dir / "checkpoint" / f"epoch_{epoch}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loss, prediction, margin = evaluate_per_sample(
            model, images, labels, criterion, args.batch_size
        )
        losses.append(loss)
        predictions.append(prediction)
        margins.append(margin)
        clean = ~is_noisy
        print(
            f"epoch {epoch}: clean loss {loss[clean].mean():.4f} "
            f"error {(prediction[clean] != labels.cpu().numpy()[clean]).mean():.4f} | "
            f"noisy loss {loss[is_noisy].mean():.4f}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        epochs=np.asarray(args.epochs, dtype=np.int32),
        loss=np.stack(losses),
        prediction=np.stack(predictions),
        margin=np.stack(margins),
        training_label=labels.cpu().numpy().astype(np.int8),
        original_label=original_labels,
        is_noisy=is_noisy,
    )
    print(f"Saved per-sample metrics to {output}")


if __name__ == "__main__":
    main()
