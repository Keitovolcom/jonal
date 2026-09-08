from pathlib import Path

from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parent / "data"
TRANSFORM = transforms.ToTensor()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    emnist_train = datasets.EMNIST(
        root=DATA_DIR / "emnist_digits",
        split="digits",
        train=True,
        download=True,
        transform=TRANSFORM,
    )

    emnist_test = datasets.EMNIST(
        root=DATA_DIR / "emnist_digits",
        split="digits",
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    cifar10_train = datasets.CIFAR10(
        root=DATA_DIR / "cifar10",
        train=True,
        download=True,
        transform=TRANSFORM,
    )

    cifar10_test = datasets.CIFAR10(
        root=DATA_DIR / "cifar10",
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    cifar100_train = datasets.CIFAR100(
        root=DATA_DIR / "cifar100",
        train=True,
        download=True,
        transform=TRANSFORM,
    )

    cifar100_test = datasets.CIFAR100(
        root=DATA_DIR / "cifar100",
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    print()
    print("Download complete")
    print(f"EMNIST-digits: {len(emnist_train):,} / {len(emnist_test):,}")
    print(f"CIFAR-10:      {len(cifar10_train):,} / {len(cifar10_test):,}")
    print(f"CIFAR-100:     {len(cifar100_train):,} / {len(cifar100_test):,}")
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()