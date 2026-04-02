"""
model.py

Purpose:
- Define the two CNN models used in Assignment 3.
- Return one logit for binary classification.

Complete the TODOs in this file only.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        # FINISHED: define Conv -> BatchNorm -> ReLU -> MaxPool(2,2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModelM1(nn.Module):
    def __init__(self):
        super().__init__()
        # FINISHED: channels 3 -> 32 -> 64 -> 128 using three ConvBlock modules
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(128, 1)               


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FINISHED: apply features, GAP, flatten, and FC
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ModelM2(nn.Module):
    def __init__(self):
        super().__init__()
        # FINISHED: channels 3 -> 64 -> 128 -> 256
        # Block 1 uses kernel_size=5, blocks 2 and 3 use kernel_size=3
        self.features = nn.Sequential(
            ConvBlock(3, 64, 5),
            ConvBlock(64, 128, 3),
            ConvBlock(128, 256, 3)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(256, 1)       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FINISHED: apply features, GAP, flatten, and FC
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(model_name: str) -> nn.Module:
    # FINISHED: return the correct model object for M1 or M2
    if model_name == "M1":
        return ModelM1()
    elif model_name == "M2":
        return ModelM2()
    else:
        raise Exception("Error: Invalid model_name")
