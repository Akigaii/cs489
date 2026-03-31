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
        
        # FINISHED:
        
        # 2) compute probabilities using sigmoid
        probabilities = torch.sigmoid(logits)
        
        # 1) compute elementwise BCE with logits (no reduction)
        elementwise_BCE = -(targets * torch.log(probabilities) + (1 - targets) * torch.log(1 - probabilities))
        
        # 3) compute pt
        pt = torch.where(targets == 1, probabilities, 1 - probabilities)
        
        # 4) assign alpha based on class weights
        alpha = torch.where(targets == 1, self.minority_weight, self.majority_weight)

        # 5) sompute focal loss and return the mean
        focal_loss = alpha * (1 - pt) ** self.gamma * elementwise_BCE
        return focal_loss.mean()


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
