"""CV architectures — byte-identical to v1 (paper_codes/cv/models.py).

SimpleCNN (~0.69-1.19M params, depending on input channels and class count) and torchvision ResNet18 (~11M params) with the same
adaptations used in v1: first conv swapped for 1-channel inputs, final fc
resized to num_classes. No other changes, so v2 reruns remain architecture-
comparable with the v1 results.
"""

import torch
import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        if input_channels == 1:   # Fashion-MNIST 28x28 -> 3x3 after 3 pools
            self.fc1 = nn.Linear(128 * 3 * 3, 512)
        else:                     # CIFAR 32x32 -> 4x4 after 3 pools
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                          stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def build_model(name: str, num_classes: int, input_channels: int):
    if name == "SimpleCNN":
        return SimpleCNN(num_classes, input_channels)
    if name == "ResNet18":
        return ResNet18(num_classes, input_channels)
    raise ValueError(f"Unknown CV model: {name}")
