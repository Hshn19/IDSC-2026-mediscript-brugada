## Set up a unified evaluate.py script that computes: AUROC, F1-score, sensitivity, specificity, precision, confusion matrix 

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate classification model performance.

    Parameters:
    y_true : actual labels (0 or 1)
    y_pred : predicted labels (0 or 1)
    y_prob : predicted probabilities (for class 1)

    Returns:
    dict of evaluation metrics
    """

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0   # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)

    results = {
        "AUROC": auroc,
        "F1-score": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Confusion Matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }
    }

    return results
