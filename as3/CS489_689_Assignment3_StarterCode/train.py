"""
train.py

Purpose:
- Define BCE, WBCE, and Focal losses.
- Run one epoch of training or evaluation.

Complete the TODOs in this file only.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import compute_metrics


class FocalLoss(nn.Module):
    def __init__(self, minority_weight: float = 2.0, majority_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.minority_weight = minority_weight
        self.majority_weight = majority_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO:
        # 1) compute elementwise BCE with logits (no reduction)
        # 2) compute probabilities using sigmoid
        # 3) compute pt
        # 4) assign alpha based on class weights
        # 5) compute focal loss and return the mean
        raise NotImplementedError


def build_criterion(loss_name: str, minority_weight: float, majority_weight: float, gamma: float):
    # TODO:
    # - bce  -> BCEWithLogitsLoss()
    # - wbce -> BCEWithLogitsLoss(pos_weight=...)
    # - focal -> FocalLoss(...)
    raise NotImplementedError


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool = True) -> Tuple[float, Dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_probs = []
    all_targets = []

    with torch.set_grad_enabled(train):
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            # TODO: forward pass and loss computation
            logits = None
            loss = None

            if train:
                # TODO: zero gradients, backpropagate, and update weights
                pass

            # TODO: convert logits to probabilities and store probabilities/targets

    # TODO: compute mean loss and metric dictionary
    mean_loss = 0.0
    metrics = {}
    return mean_loss, metrics
