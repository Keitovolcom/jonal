"""PyTorch image classifiers. Outputs are logits (no softmax).

Examples:
    model = create_model("cnn5layer", in_channels=1, num_classes=10,
                         input_size=32, width_multiplier=0.5)
    model = create_model("resnet18", in_channels=3, num_classes=10)

Dependency: torch. All models use random initialization.
"""

import math
from numbers import Integral, Real

import torch
from torch import nn


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _scaled_channels(width_multiplier, bases=(64, 128, 256, 512)):
    if (isinstance(width_multiplier, bool)
            or not isinstance(width_multiplier, Real)
            or not math.isfinite(width_multiplier) or width_multiplier <= 0):
        raise ValueError("width_multiplier must be finite and positive")
    channels = []
    for base in bases:
        scaled = base * width_multiplier
        if scaled < 1 or not float(scaled).is_integer():
            raise ValueError("width_multiplier must produce positive integer channels")
        channels.append(int(scaled))
    return channels


class CNN5Layer(nn.Module):
    """Five convolution/ReLU layers with pooling after layers 1, 2 and 4.

    input_size is an integer or (height, width). Forward inputs must match it.
    Each base channel count times width_multiplier must be a positive integer.
    No batch normalization or dropout is used.
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=32,
                 width_multiplier=1.0):
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        num_classes = _positive_int("num_classes", num_classes)
        if isinstance(input_size, Integral):
            input_size = (input_size, input_size)
        if not isinstance(input_size, (tuple, list)) or len(input_size) != 2:
            raise ValueError("input_size must be an integer or (height, width)")
        height, width = (_positive_int("input_size", size) for size in input_size)
        if min(height, width) < 8:
            raise ValueError("Both input dimensions must be at least 8")
        channels = _scaled_channels(width_multiplier, (64, 128, 256, 512, 1024))
        layers = []
        previous = in_channels
        for index, current in enumerate(channels, start=1):
            layers.extend([nn.Conv2d(previous, current, 3, stride=1, padding=1),
                           nn.ReLU(inplace=True)])
            if index in (1, 2, 4):
                layers.append(nn.MaxPool2d(2, stride=2))
            previous = current
        self.input_shape = (in_channels, height, width)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1] * (height // 8) * (width // 8),
                                    num_classes)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape}], got {tuple(x.shape)}")
        return self.classifier(torch.flatten(self.features(x), 1))


class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """CIFAR-style ResNet18: 3x3/stride-1 stem, no initial max pooling.

    Uses standard [2, 2, 2, 2] basic blocks and [64, 128, 256, 512] channels.
    Intended for 32x32 images; adaptive pooling also supports other sizes.
    """

    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.0):
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        num_classes = _positive_int("num_classes", num_classes)
        c1, c2, c3, c4 = _scaled_channels(width_multiplier)
        self.conv1 = nn.Conv2d(self.in_channels, c1, 3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._stage(c1, c1, stride=1)
        self.layer2 = self._stage(c1, c2, stride=2)
        self.layer3 = self._stage(c2, c3, stride=2)
        self.layer4 = self._stage(c3, c4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(in_channels, out_channels, stride):
        return nn.Sequential(_BasicBlock(in_channels, out_channels, stride),
                             _BasicBlock(out_channels, out_channels))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [batch, {self.in_channels}, height, width], "
                             f"got {tuple(x.shape)}")
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class _PreActBasicBlock(nn.Module):
    """Full pre-activation block; identity shortcuts bypass BN and ReLU."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        activated = self.relu(self.bn1(x))
        identity = x if self.shortcut is None else self.shortcut(activated)
        out = self.conv1(activated)
        out = self.conv2(self.relu(self.bn2(out)))
        return out + identity


class PreActResNet18(nn.Module):
    """CIFAR-style full pre-activation ResNet18, randomly initialized.

    A 3x3/stride-1 stem without max pooling feeds [2, 2, 2, 2] blocks
    with [64, 128, 256, 512] channels. Blocks use BN-ReLU-Conv ordering,
    with no activation after addition. Projection shortcuts use pre-activated
    inputs. Final BN/ReLU precedes adaptive pooling and the classifier.
    Defaults to RGB 32x32 images and 10 classes; other spatial sizes work too.
    """

    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.0):
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        num_classes = _positive_int("num_classes", num_classes)
        c1, c2, c3, c4 = _scaled_channels(width_multiplier)
        self.conv1 = nn.Conv2d(self.in_channels, c1, 3, padding=1, bias=False)
        self.layer1 = self._stage(c1, c1, stride=1)
        self.layer2 = self._stage(c1, c2, stride=2)
        self.layer3 = self._stage(c2, c3, stride=2)
        self.layer4 = self._stage(c3, c4, stride=2)
        self.bn = nn.BatchNorm2d(c4)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(in_channels, out_channels, stride):
        return nn.Sequential(_PreActBasicBlock(in_channels, out_channels, stride),
                             _PreActBasicBlock(out_channels, out_channels))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [batch, {self.in_channels}, height, width], "
                             f"got {tuple(x.shape)}")
        x = self.conv1(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.relu(self.bn(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))


MODEL_REGISTRY = {"cnn5layer": CNN5Layer, "resnet18": ResNet18,
                  "preactresnet18": PreActResNet18}


def create_model(name, **kwargs):
    """Create a model by name, forwarding keyword arguments to its constructor."""
    if not isinstance(name, str) or name.lower() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name.lower()](**kwargs)


__all__ = ["CNN5Layer", "ResNet18", "PreActResNet18", "create_model", "MODEL_REGISTRY"]
