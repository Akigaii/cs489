import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


# Open best model's test_metrics.json
BEST_MODEL = "M1_focal_w3.0"
BEST_MODEL_DIR = os.path.join(os.getcwd(), BEST_MODEL)
BEST_MODEL_TEST_METRICS = os.path.join(BEST_MODEL_DIR, "test_metrics.json")
FIGURES_DIR = os.path.join(os.getcwd(), "figures")

with open(BEST_MODEL_TEST_METRICS) as file:
    metrics = json.load(file)
    
y_true = np.array(metrics["y_true"])
y_pred = np.array(metrics["y_pred"])
y_prob = np.array(metrics["y_prob"]) 
auprc  = np.array(metrics["auprc"])
auroc  = np.array(metrics["auroc"])

os.makedirs(FIGURES_DIR, exist_ok = True)


# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.plot()
plt.plot(fpr, tpr, color = "blue", lw = 2)
plt.plot([0, 1], [0, 1], color="gray", lw = 1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Breast Cancer Binary CNN - ROC Curve")
plt.savefig(f"{FIGURES_DIR}/{BEST_MODEL}_roc_curve.png")
plt.show()


# PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
baseline = y_true.mean() 

plt.plot(recall, precision, color = "red", lw = 2)
plt.axhline(y = baseline, color = "gray", lw = 1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Breast Cancer Binary CNN - Precision-Recall Curve")
plt.savefig(f"{FIGURES_DIR}/{BEST_MODEL}_pr_curve.png")
plt.show()


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot using Matplotlib
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap = plt.cm.Blues)
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Breast Cancer Binary CNN - Confusion Matrix")
plt.savefig(f"{FIGURES_DIR}/{BEST_MODEL}_confusion_matrix.png")
plt.show()