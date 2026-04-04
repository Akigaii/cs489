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

    # FINSIHED: convert probabilities to predicted labels using threshold
    y_pred = np.where(y_prob >= threshold, 1, 0)

    # FINISHED: compute TN, FP, FN, TP using confusion_matrix
    matrix = confusion_matrix(y_true, y_pred)
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]
    
    # FINISHED: compute specificity
    specificity = TN / (TN + FP)

    # FINISHED: return a dictionary with these keys:
    # accuracy, precision, recall, f1, auroc, auprc, specificity
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "specificity": specificity,
        # Additional metrics I need for figures.py
        "confusion_matrix": matrix.tolist(),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist()
    }
    
    return metrics
