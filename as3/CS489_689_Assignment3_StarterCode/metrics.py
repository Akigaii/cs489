"""
metrics.py

Purpose:
- Compute the binary classification metrics required by the assignment.

Complete the TODOs in this file only.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # TODO: convert probabilities to predicted labels using threshold
    y_pred = None

    # TODO: compute TN, FP, FN, TP using confusion_matrix
    # TODO: compute specificity

    # TODO: return a dictionary with these keys:
    # accuracy, precision, recall, f1, auroc, auprc, specificity
    metrics = {}
    return metrics
