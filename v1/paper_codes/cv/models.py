"""
Neural Network Models for Computer Vision Benchmark
====================================================

This module implements:
1. SimpleCNN: A simple 3-layer convolutional neural network
2. ResNet18: ResNet-18 architecture adapted for CIFAR and Fashion-MNIST

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import torch
import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network.
    
    Architecture:
        - Conv2d(in_channels, 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
        - Conv2d(32, 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
        - Conv2d(64, 128, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
        - Flatten
        - Linear(128 * spatial_dim, 512) + ReLU + Dropout(0.5)
        - Linear(512, num_classes)
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB, 1 for grayscale)
    """
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened dimension based on input size
        if input_channels == 1:  # Fashion-MNIST (28x28)
            # After 3 pooling: 28 -> 14 -> 7 -> 3
            self.fc1 = nn.Linear(128 * 3 * 3, 512)
        else:  # CIFAR (32x32)
            # After 3 pooling: 32 -> 16 -> 8 -> 4
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
        
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))  # 32 channels
        x = self.pool(torch.relu(self.conv2(x)))  # 64 channels
        x = self.pool(torch.relu(self.conv3(x)))  # 128 channels
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class ResNet18(nn.Module):
    """
    ResNet-18 architecture adapted for CIFAR and Fashion-MNIST datasets.
    
    Uses torchvision's pretrained architecture with modifications:
    - Adapted first convolutional layer for non-RGB inputs if needed
    - Modified final fully connected layer for custom number of classes
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB, 1 for grayscale)
    """
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(ResNet18, self).__init__()
        
        # Load ResNet18 without pretrained weights
        self.resnet = torchvision.models.resnet18(weights=None)
        
        # Adapt first layer if input channels != 3
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
        
        # Modify final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
