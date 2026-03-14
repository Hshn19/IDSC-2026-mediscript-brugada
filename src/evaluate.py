import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


def find_best_threshold(y_true, y_prob):
    """
    Find the probability threshold that maximizes F1 on the validation set.

    CRITICAL: Always tune threshold on VAL set, never on test set.
    Default 0.5 is almost always wrong for imbalanced datasets.

    Args:
        y_true : list/array of true binary labels
        y_prob : list/array of predicted probabilities

    Returns:
        best_threshold : float
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (np.array(y_prob) >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"Best threshold: {best_t:.2f}  (F1={best_f1:.4f} on val set)")
    return best_t

def find_sensitivity_threshold(y_true, y_prob, min_sensitivity=0.80):
    """
    Find the highest threshold that still achieves min_sensitivity.
    
    For clinical Brugada detection, we accept more false positives
    (unnecessary follow-ups) over false negatives (missed sudden death risk).
    
    Args:
        y_true           : true labels
        y_prob           : predicted probabilities  
        min_sensitivity  : minimum acceptable sensitivity (default 0.80)
    
    Returns:
        best_threshold : float
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_spec = 0.5, 0.0

    for t in thresholds:
        preds = (np.array(y_prob) >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sensitivity >= min_sensitivity and specificity > best_spec:
            best_spec = specificity
            best_t    = t

    preds = (np.array(y_prob) >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn)
    print(f"Clinical threshold : {best_t:.2f}  "
          f"(Sensitivity={sens:.4f}, Specificity={best_spec:.4f})")
    return best_t

def full_report(y_true, y_prob, threshold=None, split_name="Test"):
    """
    Complete evaluation report for binary ECG classification.

    Metrics reported:
    - AUROC      : primary metric, threshold-independent
    - F1         : harmonic mean of precision and recall
    - Sensitivity: true positive rate — most critical for Brugada
                   (missing a Brugada case = risk of sudden cardiac death)
    - Specificity: true negative rate
    - Precision  : positive predictive value

    Args:
        y_true     : array of true labels
        y_prob     : array of predicted probabilities
        threshold  : decision threshold (tune on val, apply to test)
        split_name : label for printing

    Returns:
        dict of all metrics
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auroc       = roc_auc_score(y_true, y_prob)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  {split_name} Results  (threshold = {threshold:.2f})")
    print(f"{'='*45}")
    print(f"  AUROC        : {auroc:.4f}  ← primary metric")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  ← minimize false negatives")
    print(f"  Specificity  : {specificity:.4f}")
    print(f"  Precision    : {precision:.4f}")
    print(f"{'─'*45}")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"  False Negatives (missed Brugada): {fn}  ← target = 0")
    print(f"{'='*45}")

    return {
        'auroc': auroc, 'f1': f1, 'sensitivity': sensitivity,
        'specificity': specificity, 'precision': precision,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'threshold': threshold
    }


def plot_training_curves(history_path, save_path=None):
    """Plot loss and AUROC curves from saved training history."""
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='crimson')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUROC
    axes[1].plot(epochs, history['train_auroc'], label='Train', color='steelblue')
    axes[1].plot(epochs, history['val_auroc'],   label='Val',   color='crimson')
    axes[1].axhline(y=max(history['val_auroc']), color='green',
                    linestyle='--', alpha=0.7,
                    label=f"Best Val: {max(history['val_auroc']):.4f}")
    axes[1].set_title('AUROC Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('BrugadaCNN Training History', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved: {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve with AUROC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.fill_between(fpr, tpr, alpha=0.1, color='crimson')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve — BrugadaCNN')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 ROC curve saved: {save_path}")
    plt.show()