"""Dataset loading and reproducible label-noise utilities."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parent / "data"

_ALIASES = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "emnist-digits": "emnist-digits",
    "emnist_digits": "emnist-digits",
    "emnistdigits": "emnist-digits",
}

_CONFIG = {
    "cifar10": {
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "emnist-digits": {
        "num_classes": 10,
        "mean": (0.173469787916794,),
        "std": (0.316553955125998,),
    },
}


class LabelNoiseDataset(Dataset):
    """Training dataset retaining both original and learning labels.

    Each item is ``(image, original_label, training_label, is_noisy)``.
    ``targets`` remains an alias for the learning labels so utilities expecting
    a torchvision-style dataset continue to work.
    """

    def __init__(self, dataset: Dataset, training_targets: Any):
        if not hasattr(dataset, "targets"):
            raise TypeError("The wrapped dataset must expose a 'targets' attribute")
        original = _targets_as_numpy(dataset.targets)
        training = _targets_as_numpy(training_targets)
        if original.shape != training.shape:
            raise ValueError("original and training labels must have the same shape")

        self.dataset = dataset
        self.original_targets = torch.as_tensor(original, dtype=torch.long)
        self.training_targets = torch.as_tensor(training, dtype=torch.long)
        self.is_noisy = self.original_targets.ne(self.training_targets)
        self.targets = self.training_targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        image = sample[0]
        return (
            image,
            self.original_targets[index],
            self.training_targets[index],
            self.is_noisy[index],
        )

    def __getattr__(self, name: str):
        # Keep useful torchvision attributes such as classes and transform.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)



def _canonical_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(_CONFIG)
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Supported datasets: {supported}"
        ) from exc


SUPPORTED_AUGMENTATIONS = (
    "random_crop",
    "horizontal_flip",
    "vertical_flip",
    "random_rotation",
    "random_affine",
    "color_jitter",
    "random_perspective",
    "random_resized_crop",
    "random_grayscale",
    "gaussian_blur",
    "random_erasing",
)

_AUGMENTATION_ALIASES = {
    "crop": "random_crop",
    "random_crop": "random_crop",
    "horizontal_flip": "horizontal_flip",
    "random_horizontal_flip": "horizontal_flip",
    "hflip": "horizontal_flip",
    "vertical_flip": "vertical_flip",
    "random_vertical_flip": "vertical_flip",
    "vflip": "vertical_flip",
    "rotation": "random_rotation",
    "random_rotation": "random_rotation",
    "affine": "random_affine",
    "random_affine": "random_affine",
    "color_jitter": "color_jitter",
    "jitter": "color_jitter",
    "perspective": "random_perspective",
    "random_perspective": "random_perspective",
    "resized_crop": "random_resized_crop",
    "random_resized_crop": "random_resized_crop",
    "grayscale": "random_grayscale",
    "random_grayscale": "random_grayscale",
    "gaussian_blur": "gaussian_blur",
    "blur": "gaussian_blur",
    "random_erasing": "random_erasing",
    "erasing": "random_erasing",
}

_DEFAULT_AUGMENTATIONS = {
    "cifar10": ("random_crop", "horizontal_flip"),
    "cifar100": ("random_crop", "horizontal_flip"),
    "emnist-digits": ("random_affine",),
}

_SHORT_ARGUMENTS = {
    "random_crop": "padding",
    "horizontal_flip": "p",
    "vertical_flip": "p",
    "random_rotation": "degrees",
    "random_affine": "degrees",
    "color_jitter": "brightness",
    "random_perspective": "p",
    "random_resized_crop": "size",
    "random_grayscale": "p",
    "gaussian_blur": "kernel_size",
    "random_erasing": "p",
}


def _canonical_augmentation_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _AUGMENTATION_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_AUGMENTATIONS)
        raise ValueError(
            f"Unknown data augmentation {name!r}. Supported: {supported}"
        ) from exc


def _augmentation_parameters(name: str, value: Any) -> dict[str, Any] | None:
    if value is False:
        return None
    if value is None or value is True:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return {_SHORT_ARGUMENTS[name]: value}
    raise TypeError(
        f"Parameters for {name!r} must be a mapping, number, true, or null"
    )


def _resolve_augmentations(
    dataset_name: str,
    augment: bool | str | Mapping[str, Any] | Sequence[Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    resolved: list[tuple[str, dict[str, Any]]] = []

    def add(raw_name: Any, raw_parameters: Any = None) -> None:
        if not isinstance(raw_name, str):
            raise TypeError("Each data augmentation name must be a string")
        key = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
        if key == "default":
            if raw_parameters not in (None, True):
                raise ValueError("The 'default' preset does not accept parameters")
            for default_name in _DEFAULT_AUGMENTATIONS[dataset_name]:
                resolved.append((default_name, {}))
            return
        name = _canonical_augmentation_name(raw_name)
        parameters = _augmentation_parameters(name, raw_parameters)
        if parameters is not None:
            resolved.append((name, parameters))

    def add_mapping(specification: Mapping[str, Any]) -> None:
        if "name" in specification or "type" in specification:
            item = dict(specification)
            raw_name = item.pop("name", item.pop("type", None))
            add(raw_name, item)
        else:
            for raw_name, raw_parameters in specification.items():
                add(raw_name, raw_parameters)

    if augment is False or augment is None:
        return resolved
    if augment is True:
        add("default")
    elif isinstance(augment, str):
        add(augment)
    elif isinstance(augment, Mapping):
        add_mapping(augment)
    elif isinstance(augment, Sequence):
        for item in augment:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, Mapping):
                add_mapping(item)
            else:
                raise TypeError(
                    "Each data augmentation must be a name or parameter mapping"
                )
    else:
        raise TypeError(
            "data_augmentation must be a bool, name, mapping, sequence, or None"
        )
    return resolved


def _create_augmentation(
    name: str,
    parameters: dict[str, Any],
    dataset_name: str,
):
    image_size = 32
    kwargs = dict(parameters)
    try:
        if name == "random_crop":
            kwargs.setdefault("size", image_size)
            if not parameters:
                kwargs["padding"] = 4
            return transforms.RandomCrop(**kwargs)
        if name == "horizontal_flip":
            return transforms.RandomHorizontalFlip(**kwargs)
        if name == "vertical_flip":
            return transforms.RandomVerticalFlip(**kwargs)
        if name == "random_rotation":
            kwargs.setdefault("degrees", 15)
            return transforms.RandomRotation(**kwargs)
        if name == "random_affine":
            kwargs.setdefault("degrees", 10)
            if not parameters:
                kwargs["translate"] = (0.1, 0.1)
            return transforms.RandomAffine(**kwargs)
        if name == "color_jitter":
            if not parameters:
                kwargs = {"brightness": 0.2, "contrast": 0.2}
                if dataset_name != "emnist-digits":
                    kwargs["saturation"] = 0.2
            return transforms.ColorJitter(**kwargs)
        if name == "random_perspective":
            return transforms.RandomPerspective(**kwargs)
        if name == "random_resized_crop":
            kwargs.setdefault("size", image_size)
            return transforms.RandomResizedCrop(**kwargs)
        if name == "random_grayscale":
            return transforms.RandomGrayscale(**kwargs)
        if name == "gaussian_blur":
            kwargs.setdefault("kernel_size", 3)
            return transforms.GaussianBlur(**kwargs)
        if name == "random_erasing":
            return transforms.RandomErasing(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for {name!r}: {exc}") from exc
    raise AssertionError(f"Unhandled data augmentation: {name}")


def build_transforms(
    dataset_name: str,
    augment: bool | str | Mapping[str, Any] | Sequence[Any] | None = False,
    input_channels: int | None = None,
):
    """Build transforms with mandatory resize/normalization for both splits."""

    name = _canonical_name(dataset_name)
    config = _CONFIG[name]
    native_channels = len(config["mean"])
    channels = native_channels if input_channels is None else input_channels
    if channels not in (1, 3):
        raise ValueError("input_channels must be 1 or 3 for the supported datasets")

    resize = [transforms.Resize((32, 32))] if name == "emnist-digits" else []
    before_tensor: list[Any] = list(resize)
    after_normalization: list[Any] = []
    for augmentation_name, parameters in _resolve_augmentations(name, augment):
        augmentation = _create_augmentation(
            augmentation_name, parameters, dataset_name=name
        )
        if augmentation_name == "random_erasing":
            after_normalization.append(augmentation)
        else:
            before_tensor.append(augmentation)

    if channels != native_channels:
        before_tensor.append(transforms.Grayscale(num_output_channels=channels))
    if channels == native_channels:
        mean, std = config["mean"], config["std"]
    elif channels == 1:
        mean, std = (float(np.mean(config["mean"])),), (float(np.mean(config["std"])),)
    else:
        mean, std = tuple(config["mean"] * 3), tuple(config["std"] * 3)
    normalization = transforms.Normalize(mean, std)
    train_transform = transforms.Compose(
        before_tensor
        + [transforms.ToTensor(), normalization]
        + after_normalization
    )
    evaluation_before_tensor = list(resize)
    if channels != native_channels:
        evaluation_before_tensor.append(
            transforms.Grayscale(num_output_channels=channels)
        )
    evaluation_transform = transforms.Compose(
        evaluation_before_tensor + [transforms.ToTensor(), normalization]
    )
    return train_transform, evaluation_transform


def _targets_as_numpy(targets: Any) -> np.ndarray:
    if torch.is_tensor(targets):
        return targets.detach().cpu().numpy().astype(np.int64, copy=True)
    return np.asarray(targets, dtype=np.int64).copy()


def generate_noisy_labels(
    labels: Any,
    noise_rate: float,
    num_classes: int,
    seed: int = 0,
) -> np.ndarray:
    """Return a copy with exactly ``int(len(labels) * noise_rate)`` changes."""

    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be between 0.0 and 1.0")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    clean = _targets_as_numpy(labels)
    if clean.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if clean.size and (clean.min() < 0 or clean.max() >= num_classes):
        raise ValueError("labels contain a value outside the class range")

    noisy = clean.copy()
    count = int(clean.size * noise_rate)
    if count == 0:
        return noisy

    rng = np.random.default_rng(seed)
    indices = rng.choice(clean.size, size=count, replace=False)
    # Offsets 1..C-1 guarantee a new class and are uniform over alternatives.
    offsets = rng.integers(1, num_classes, size=count)
    noisy[indices] = (clean[indices] + offsets) % num_classes
    return noisy


def _set_targets(dataset: Any, labels: np.ndarray) -> None:
    current = dataset.targets
    if torch.is_tensor(current):
        dataset.targets = torch.as_tensor(
            labels, dtype=current.dtype, device=current.device
        )
    elif isinstance(current, np.ndarray):
        dataset.targets = labels.astype(current.dtype, copy=True)
    else:
        dataset.targets = labels.tolist()


def _rate_token(noise_rate: float) -> str:
    return format(noise_rate, ".12g").replace(".", "p")


def label_cache_path(
    dataset_name: str,
    noise_rate: float,
    seed: int,
    data_dir: str | Path | None = None,
) -> Path:
    """Return a path identifying dataset, noise rate, and seed."""

    name = _canonical_name(dataset_name)
    root = DATA_DIR if data_dir is None else Path(data_dir)
    dataset_root = root / name.replace("-", "_")
    return dataset_root / (
        f"{name}_labels_noise-{_rate_token(noise_rate)}_seed-{seed}.npz"
    )


def _load_cache(
    path: Path,
    clean: np.ndarray,
    dataset_name: str,
    noise_rate: float,
    seed: int,
    num_classes: int,
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "labels", "clean_labels", "dataset_name",
                "noise_rate", "seed", "num_classes",
            }
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(f"missing fields: {sorted(missing)}")
            labels = np.asarray(cache["labels"], dtype=np.int64)
            cached_clean = np.asarray(cache["clean_labels"], dtype=np.int64)
            metadata_matches = (
                str(cache["dataset_name"].item()) == dataset_name
                and float(cache["noise_rate"].item()) == noise_rate
                and int(cache["seed"].item()) == seed
                and int(cache["num_classes"].item()) == num_classes
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid label-noise cache {path}: {exc}") from exc

    if not metadata_matches:
        raise ValueError(f"Label-noise cache metadata does not match: {path}")
    if labels.shape != clean.shape or not np.array_equal(cached_clean, clean):
        raise ValueError(f"Label-noise cache does not match the dataset: {path}")
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"Label-noise cache contains an invalid class: {path}")
    expected = int(clean.size * noise_rate)
    actual = int(np.count_nonzero(labels != clean))
    if actual != expected:
        raise ValueError(
            f"Label-noise cache has {actual} changes; expected {expected}: {path}"
        )
    return labels


def _save_cache(
    path: Path,
    labels: np.ndarray,
    clean: np.ndarray,
    dataset_name: str,
    noise_rate: float,
    seed: int,
    num_classes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                labels=labels,
                clean_labels=clean,
                dataset_name=np.asarray(dataset_name),
                noise_rate=np.asarray(noise_rate, dtype=np.float64),
                seed=np.asarray(seed, dtype=np.int64),
                num_classes=np.asarray(num_classes, dtype=np.int64),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def add_label_noise(
    train_dataset: Any,
    noise_rate: float,
    *,
    dataset_name: str,
    seed: int = 0,
    num_classes: int | None = None,
    data_dir: str | Path | None = None,
) -> LabelNoiseDataset:
    """Return a dataset retaining original, learning, and noise labels."""

    name = _canonical_name(dataset_name)
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be between 0.0 and 1.0")
    if num_classes is None:
        num_classes = int(_CONFIG[name]["num_classes"])

    clean = _targets_as_numpy(train_dataset.targets)
    path = label_cache_path(name, noise_rate, seed, data_dir)
    if path.exists():
        noisy = _load_cache(
            path, clean, name, noise_rate, seed, num_classes
        )
    else:
        noisy = generate_noisy_labels(clean, noise_rate, num_classes, seed)
        _save_cache(
            path, noisy, clean, name, noise_rate, seed, num_classes
        )

    return LabelNoiseDataset(train_dataset, noisy)


def load_dataset(
    dataset_name: str,
    *,
    data_augmentation: bool | str | Mapping[str, Any] | Sequence[Any] | None = False,
    label_noise_rate: float = 0.0,
    seed: int = 0,
    input_channels: int | None = None,
    data_dir: str | Path | None = None,
    download: bool = True,
):
    """Load and return ``(train_dataset, evaluation_dataset)``.

    Supported names are ``cifar10``, ``cifar100`` and ``emnist-digits``.
    Noise is applied only to the training split. A zero rate creates no cache.
    """

    name = _canonical_name(dataset_name)
    if not 0.0 <= label_noise_rate <= 1.0:
        raise ValueError("label_noise_rate must be between 0.0 and 1.0")

    root = DATA_DIR if data_dir is None else Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_transform, evaluation_transform = build_transforms(
        name, augment=data_augmentation, input_channels=input_channels
    )
    dataset_root = root / name.replace("-", "_")

    if name == "emnist-digits":
        train_dataset = datasets.EMNIST(
            root=dataset_root, split="digits", train=True,
            download=download, transform=train_transform,
        )
        evaluation_dataset = datasets.EMNIST(
            root=dataset_root, split="digits", train=False,
            download=download, transform=evaluation_transform,
        )
    else:
        dataset_class = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train_dataset = dataset_class(
            root=dataset_root, train=True, download=download,
            transform=train_transform,
        )
        evaluation_dataset = dataset_class(
            root=dataset_root, train=False, download=download,
            transform=evaluation_transform,
        )
    clean_targets = _targets_as_numpy(train_dataset.targets)

    if label_noise_rate > 0.0:
        train_dataset = add_label_noise(
            train_dataset,
            label_noise_rate,
            dataset_name=name,
            seed=seed,
            num_classes=int(_CONFIG[name]["num_classes"]),
            data_dir=root,
        )
    else:
        # Always use the same item format, including the zero-noise case.
        train_dataset = LabelNoiseDataset(train_dataset, clean_targets)
    return train_dataset, evaluation_dataset


def main() -> None:
    """Download every supported dataset with normalized transforms."""

    for name in _CONFIG:
        train_dataset, evaluation_dataset = load_dataset(name)
        print(
            f"{name}: {len(train_dataset):,} training / "
            f"{len(evaluation_dataset):,} evaluation"
        )
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()


__all__ = [
    "DATA_DIR",
    "LabelNoiseDataset",
    "SUPPORTED_AUGMENTATIONS",
    "add_label_noise",
    "build_transforms",
    "generate_noisy_labels",
    "label_cache_path",
    "load_dataset",
]
