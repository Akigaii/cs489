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

        # TODO: define Conv -> BatchNorm -> ReLU -> MaxPool(2,2)
        self.block = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return the block output
        raise NotImplementedError


class ModelM1(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: channels 3 -> 32 -> 64 -> 128 using three ConvBlock modules
        self.features = None
        self.pool = None
        self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply features, GAP, flatten, and FC
        raise NotImplementedError


class ModelM2(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: channels 3 -> 64 -> 128 -> 256
        # Block 1 uses kernel_size=5, blocks 2 and 3 use kernel_size=3
        self.features = None
        self.pool = None
        self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply features, GAP, flatten, and FC
        raise NotImplementedError


def build_model(model_name: str) -> nn.Module:
    # TODO: return the correct model object for M1 or M2
    raise NotImplementedError
