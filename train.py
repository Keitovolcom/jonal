"""Command-line training entry point for reproducible image experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import MODEL_REGISTRY, RESNET_INITS, create_model


DATASET_SPECS = {
    "cifar10": {"num_classes": 10, "input_size": 32, "input_channels": 3},
    "cifar100": {"num_classes": 100, "input_size": 32, "input_channels": 3},
    "emnist-digits": {"num_classes": 10, "input_size": 32, "input_channels": 1},
    "mnist": {"num_classes": 10, "input_size": 28, "input_channels": 1},
}
METRIC_NAMES = (
    "test_error",
    "test_loss",
    "train_error",
    "train_loss",
    "train_error_noisy",
    "train_loss_noisy",
    "train_error_clean",
    "train_loss_clean",
)


def probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number



def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def parse_augmentation(value: str) -> Any:
    """Parse none/default, a comma-separated list, or a JSON specification."""

    text = value.strip()
    lowered = text.lower()
    if lowered in {"none", "false", "off", "no"}:
        return False
    if lowered in {"default", "true", "on", "yes"}:
        return True
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"invalid augmentation JSON: {exc}") from exc
        if not isinstance(parsed, (dict, list)):
            raise argparse.ArgumentTypeError(
                "augmentation JSON must be an object or an array"
            )
        return parsed
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("augmentation must not be empty")
    return names[0] if len(names) == 1 else names


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an image classifier and record clean/noisy metrics."
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="cnn5layer")
    parser.add_argument("--lr", type=positive_float, default=0.1)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=positive_int, default=128
    )
    parser.add_argument(
        "--gpu",
        default="auto",
        help="'auto', 'cpu', a GPU index such as 0, or a device such as cuda:1",
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers",
        type=nonnegative_int, default=4,
    )
    parser.add_argument(
        "--augmentation",
        type=parse_augmentation,
        default=False,
        help="none, default, comma-separated names, or a JSON object/array",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="cifar10")
    parser.add_argument(
        "--label-noise", "--label_noise", dest="label_noise", type=probability,
        default=0.0,
    )
    parser.add_argument(
        "--optimizer", choices=("sgd", "adam", "adamw", "rmsprop"), default="sgd"
    )
    parser.add_argument(
        "--loss", choices=("crossentropy", "mse", "mae"), default="crossentropy"
    )
    parser.add_argument("--momentum", type=probability, default=0.0)
    parser.add_argument(
        "--model-width", "--model_width", dest="model_width",
        type=positive_float, default=1.0,
    )
    parser.add_argument(
        "--init", choices=RESNET_INITS, default="kaiming",
        help="ResNet weight init: 'kaiming' (fan_out normal, default) or "
             "'default' (PyTorch layer defaults); only for resnet18/preactresnet18 "
             "(cnn5layer and resnet18k always use PyTorch defaults)",
    )
    parser.add_argument(
        "--input-channels", "--input_channels", dest="input_channels",
        type=positive_int, default=None,
        help="Defaults to the dataset's native channel count; supported values: 1 or 3",
    )
    parser.add_argument(
        "--experiment-name", "--experiment_name", dest="experiment_name", default=None,
        help="Optional override; by default it is generated from major conditions",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument(
        "--checkpoint-interval", "--checkpoint_interval",
        dest="checkpoint_interval", type=nonnegative_int, default=1,
        help="Save every N epochs (default: 1); use 0 to disable checkpoints",
    )
    parser.add_argument(
        "--checkpoint-model-only", "--checkpoint_model_only",
        dest="checkpoint_model_only", action="store_true",
        help="Save only model weights in checkpoints (omit optimizer state)",
    )
    parser.add_argument("--wandb-project", "--wandb_project", default="jonal")
    parser.add_argument("--wandb-entity", "--wandb_entity", default=None)
    parser.add_argument("--wandb-group", "--wandb_group", default=None)
    parser.add_argument(
        "--wandb-mode", "--wandb_mode",
        choices=("online", "offline", "disabled"), default="offline",
    )
    parser.add_argument(
        "--weight-decay", "--weight_decay", dest="weight_decay",
        type=nonnegative_float, default=0.0,
    )
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Fail instead of downloading when dataset files are absent",
    )
    return parser


def _number_token(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def _augmentation_token(augmentation: Any) -> str:
    if augmentation is False:
        return "none"
    if augmentation is True:
        return "default"
    encoded = json.dumps(augmentation, sort_keys=True, separators=(",", ":"))
    names = re.sub(r"[^a-zA-Z0-9]+", "-", encoded).strip("-").lower()
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:8]
    return f"{names[:24]}-{digest}"


def generate_experiment_name(args: argparse.Namespace) -> str:
    """Generate a filesystem-safe name from the main experimental conditions."""

    parts = (
        args.dataset,
        args.model,
        f"noise-{_number_token(args.label_noise)}",
        f"lr-{_number_token(args.lr)}",
        f"loss-{args.loss}",
        f"bs-{args.batch_size}",
        f"opt-{args.optimizer}",
        f"width-{_number_token(args.model_width)}",
        *(() if args.model not in ("resnet18", "preactresnet18")
          or args.init == "kaiming"
          else (f"init-{args.init}",)),
        f"ch-{args.input_channels}",
        f"aug-{_augmentation_token(args.augmentation)}",
        f"seed-{args.seed}",
    )
    return "_".join(parts)


def resolve_device(gpu: str) -> torch.device:
    specification = str(gpu).strip().lower()
    if specification == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if specification == "cpu":
        return torch.device("cpu")
    if specification.isdigit():
        specification = f"cuda:{specification}"
    if not re.fullmatch(r"cuda:\d+", specification):
        raise ValueError("gpu must be 'auto', 'cpu', an integer, or 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{specification} was requested, but CUDA is unavailable")
    index = int(specification.split(":")[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{specification} was requested, but only {torch.cuda.device_count()} GPU(s) exist"
        )
    torch.cuda.set_device(index)
    return torch.device(specification)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ProbabilityTargetLoss(nn.Module):
    """Per-sample MSE or MAE between softmax probabilities and one-hot labels."""

    def __init__(self, loss_type: str):
        super().__init__()
        if loss_type not in {"mse", "mae"}:
            raise ValueError(f"unsupported probability loss: {loss_type}")
        self.loss_type = loss_type

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(
            dtype=logits.dtype
        )
        differences = probabilities - one_hot
        if self.loss_type == "mse":
            return differences.square().mean(dim=1)
        return differences.abs().mean(dim=1)


def create_loss(name: str) -> nn.Module:
    if name == "crossentropy":
        return nn.CrossEntropyLoss(reduction="none")
    if name in {"mse", "mae"}:
        return ProbabilityTargetLoss(name)
    raise ValueError(f"unknown loss: {name}")


class TrainingMetrics:
    """Accumulate per-sample losses/errors for all, noisy, and clean samples.

    Sums stay on the loss tensor's device and are read back once in
    ``compute``, so ``update`` never blocks on the GPU. Each batch's subset
    loss is the same float32 reduction as ``losses[mask].sum()`` and is added
    to a float64 accumulator, which reproduces a per-batch Python-float
    accumulation bit for bit.
    """

    SUBSETS = ("all", "noisy", "clean")

    def __init__(self) -> None:
        self.counts = dict.fromkeys(self.SUBSETS, 0)
        self.loss_sums: dict[str, torch.Tensor] = {}
        self.error_sums: dict[str, torch.Tensor] = {}

    def update(
        self,
        logits: torch.Tensor,
        losses: torch.Tensor,
        training_labels: torch.Tensor,
        is_noisy: torch.Tensor,
    ) -> None:
        """``is_noisy`` should be on the CPU: subset sizes and indices are
        derived there, which keeps subset selection from synchronizing."""
        device = losses.device
        errors = logits.argmax(dim=1).ne(training_labels)
        noisy = is_noisy.cpu().bool()
        subset_indices = {
            "all": None,
            "noisy": noisy.nonzero().squeeze(1),
            "clean": (~noisy).nonzero().squeeze(1),
        }
        for subset, indices in subset_indices.items():
            if indices is None:
                count = losses.numel()
                subset_losses, subset_errors = losses, errors
            else:
                count = indices.numel()
                if not count:
                    continue
                if device.type == "cuda":
                    indices = indices.pin_memory()
                indices = indices.to(device, non_blocking=True)
                subset_losses = losses.index_select(0, indices)
                subset_errors = errors.index_select(0, indices)
            self.counts[subset] += count
            loss_sum = subset_losses.sum().double()
            error_sum = subset_errors.sum()
            if subset in self.loss_sums:
                self.loss_sums[subset] += loss_sum
                self.error_sums[subset] += error_sum
            else:
                self.loss_sums[subset] = loss_sum
                self.error_sums[subset] = error_sum

    def _average(self, sums: dict[str, torch.Tensor], subset: str) -> float:
        count = self.counts[subset]
        if not count:
            return float("nan")
        return float(sums[subset].item() / count)

    def compute(self) -> dict[str, float]:
        loss = {s: self._average(self.loss_sums, s) for s in self.SUBSETS}
        error = {s: self._average(self.error_sums, s) for s in self.SUBSETS}
        return {
            "train_error": error["all"],
            "train_loss": loss["all"],
            "train_error_noisy": error["noisy"],
            "train_loss_noisy": loss["noisy"],
            "train_error_clean": error["clean"],
            "train_loss_clean": loss["clean"],
        }


@torch.inference_mode()
def evaluate_training_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = TrainingMetrics()
    for images, _original_labels, training_labels, is_noisy in loader:
        images = images.to(device, non_blocking=True)
        training_labels = training_labels.to(device, non_blocking=True)

        logits = model(images)
        losses = criterion(logits, training_labels)
        metrics.update(logits.detach(), losses.detach(), training_labels, is_noisy)
    return metrics.compute()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    metrics = TrainingMetrics()
    for images, _original_labels, training_labels, is_noisy in loader:
        images = images.to(device, non_blocking=True)
        training_labels = training_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        losses = criterion(logits, training_labels)
        losses.mean().backward()
        optimizer.step()

        metrics.update(logits.detach(), losses.detach(), training_labels, is_noisy)
    return metrics.compute()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    # Sums stay on the device (float32 batch sum -> float64 accumulator) and
    # are read back once, matching a per-batch Python-float accumulation.
    count = 0
    loss_sum: torch.Tensor | None = None
    error_sum: torch.Tensor | None = None
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        losses = criterion(logits, labels)
        count += labels.numel()
        batch_loss = losses.sum().double()
        batch_error = logits.argmax(dim=1).ne(labels).sum()
        if loss_sum is None:
            loss_sum, error_sum = batch_loss, batch_error
        else:
            loss_sum += batch_loss
            error_sum += batch_error
    if not count:
        return {"test_error": float("nan"), "test_loss": float("nan")}
    return {
        "test_error": int(error_sum.item()) / count,
        "test_loss": float(loss_sum.item()) / count,
    }


def create_optimizer(
    name: str,
    parameters,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"unknown optimizer: {name}")


def _serializable_config(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"]) if config["data_dir"] else None
    config["output_dir"] = str(config["output_dir"])
    config["device"] = str(device)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    run_directory: Path,
    epoch: int,
) -> Path:
    checkpoint_directory = run_directory / "checkpoint"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    path = checkpoint_directory / f"epoch_{epoch}.pt"
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    if not config.get("checkpoint_model_only", False):
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)
    return path


def initialize_wandb(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_directory: Path,
):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required; install dependencies with 'pip install -r requirements.txt'"
        ) from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.experiment_name,
        config=config,
        mode=args.wandb_mode,
        dir=str(run_directory.resolve()),
        job_type="train",
        save_code=False,
    )


def run_experiment(args: argparse.Namespace) -> Path:
    specification = DATASET_SPECS[args.dataset]
    if args.input_channels is None:
        args.input_channels = specification["input_channels"]
    if args.input_channels not in (1, 3):
        raise ValueError("input_channels must be 1 or 3 for the supported datasets")
    if args.experiment_name is None:
        args.experiment_name = generate_experiment_name(args)

    if (args.experiment_name in {"", ".", ".."}
            or "/" in args.experiment_name or "\\" in args.experiment_name):
        raise ValueError("experiment_name must be a single non-empty path component")

    seed_everything(args.seed)
    device = resolve_device(args.gpu)
    run_directory = args.output_dir / args.experiment_name
    run_directory.mkdir(parents=True, exist_ok=False)

    train_dataset, test_dataset = load_dataset(
        args.dataset,
        data_augmentation=args.augmentation,
        label_noise_rate=args.label_noise,
        seed=args.seed,
        input_channels=args.input_channels,
        data_dir=args.data_dir,
        download=not args.no_download,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model_options = {
        "in_channels": args.input_channels,
        "num_classes": specification["num_classes"],
        "width_multiplier": args.model_width,
    }
    if args.model == "cnn5layer":
        model_options["input_size"] = specification["input_size"]
    elif args.model in ("resnet18", "preactresnet18"):
        model_options["init"] = args.init
    model = create_model(args.model, **model_options).to(device)
    optimizer = create_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = create_loss(args.loss)

    config = _serializable_config(args, device)
    config_path = run_directory / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics_path = run_directory / "metrics.csv"
    with (
        initialize_wandb(args, config, run_directory) as wandb_run,
        metrics_path.open("w", newline="", encoding="utf-8") as output,
    ):
        writer = csv.DictWriter(output, fieldnames=("epoch",) + METRIC_NAMES)
        writer.writeheader()

        initial_train_metrics = evaluate_training_metrics(
            model, train_loader, criterion, device
        )
        initial_test_metrics = evaluate(model, test_loader, criterion, device)
        initial_row = {"epoch": 0, **initial_test_metrics, **initial_train_metrics}
        writer.writerow(initial_row)
        output.flush()
        wandb_run.log({"epoch": 0, **initial_row}, step=0)
        initial_shown = " ".join(
            f"{name}={initial_row[name]:.6g}" for name in METRIC_NAMES
        )
        print(f"epoch=0/init {initial_shown}", flush=True)
        if args.checkpoint_interval > 0:
            # The initial model is kept so analyses can start from epoch 0.
            save_checkpoint(model, optimizer, config, run_directory, 0)

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_metrics = evaluate(model, test_loader, criterion, device)
            row = {"epoch": epoch, **test_metrics, **train_metrics}
            writer.writerow(row)
            output.flush()
            wandb_run.log(row, step=epoch)

            if (args.checkpoint_interval > 0
                    and (epoch % args.checkpoint_interval == 0
                         or epoch == args.epochs)):
                save_checkpoint(model, optimizer, config, run_directory, epoch)

            shown = " ".join(f"{name}={row[name]:.6g}" for name in METRIC_NAMES)
            print(f"epoch={epoch}/{args.epochs} {shown}", flush=True)

    return run_directory


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    run_directory = run_experiment(args)
    print(f"results={run_directory}", flush=True)


if __name__ == "__main__":
    main()
