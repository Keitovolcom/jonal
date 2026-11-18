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
import torch.nn.functional as F


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


RESNET_INITS = ("kaiming", "default")


def _initialize_resnet(model, init):
    """Apply ResNet weight initialization.

    ``"kaiming"``: Conv2d weights use kaiming_normal_(fan_out, relu) as in
    torchvision; BatchNorm uses weight=1, bias=0. ``"default"`` keeps the
    PyTorch layer defaults (Conv2d/Linear: kaiming_uniform_(a=sqrt(5))).
    The Linear classifier always keeps the PyTorch default.
    """
    if init == "default":
        return
    if init != "kaiming":
        raise ValueError(f"init must be one of {RESNET_INITS}, got {init!r}")
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


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

    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.0,
                 init="kaiming"):
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
        _initialize_resnet(self, init)

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

    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.0,
                 init="kaiming"):
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
        _initialize_resnet(self, init)

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


class _ResNet18kBlock(nn.Module):
    """PreActBlock of kuangliu/pytorch-cifar (as used by Nakkiran et al.)."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet18k(nn.Module):
    """ResNet18 with width parameter k from Nakkiran et al., "Deep Double Descent".

    A faithful port of their ``make_resnet18k`` (based on kuangliu/pytorch-cifar
    ``preact_resnet.py``): channels [k, 2k, 4k, 8k], no BN/ReLU after the stem
    or after layer4, ``avg_pool2d(4)`` and PyTorch default initialization.
    Module names and construction order match the original, so state dicts are
    interchangeable and the same seed gives the same initial weights.

    Differences from the original: ``in_channels`` is configurable (the
    original is fixed to 3), and k is given as ``width_multiplier = k / 64``
    for consistency with the other models (k=64 is the standard ResNet18).
    Expects 32x32 inputs (layer4 output 4x4).
    """

    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.0):
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        num_classes = _positive_int("num_classes", num_classes)
        c = _scaled_channels(width_multiplier)[0]
        self.in_planes = c
        block = _ResNet18kBlock
        self.conv1 = nn.Conv2d(self.in_channels, c, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, 2, stride=1)
        self.layer2 = self._make_layer(block, 2 * c, 2, stride=2)
        self.layer3 = self._make_layer(block, 4 * c, 2, stride=2)
        self.layer4 = self._make_layer(block, 8 * c, 2, stride=2)
        self.linear = nn.Linear(8 * c * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [batch, {self.in_channels}, height, width], "
                             f"got {tuple(x.shape)}")
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


MODEL_REGISTRY = {"cnn5layer": CNN5Layer, "resnet18": ResNet18,
                  "preactresnet18": PreActResNet18, "resnet18k": ResNet18k}


def create_model(name, **kwargs):
    """Create a model by name, forwarding keyword arguments to its constructor."""
    if not isinstance(name, str) or name.lower() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name.lower()](**kwargs)


__all__ = ["CNN5Layer", "ResNet18", "PreActResNet18", "ResNet18k", "create_model",
           "MODEL_REGISTRY",
           "RESNET_INITS"]
