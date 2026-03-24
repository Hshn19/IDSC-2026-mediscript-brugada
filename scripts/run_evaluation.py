import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import DATA_DIR, SPLITS_PATH, CHECKPOINT, HISTORY_PATH, THRESHOLD

import torch
from torch.utils.data import DataLoader

from src.model import BrugadaCNN
from src.dataset import BrugadaDataset
from src.preprocessing import load_splits, preprocess_signal
from src.evaluate import (find_best_threshold, full_report,
                           plot_training_curves, plot_roc_curve)
from src.train import evaluate

DATA_DIR      = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"   # ← same as before
SPLITS_PATH   = r"outputs/splits.json"
CHECKPOINT    = r"outputs/checkpoints/best_model.pt"
HISTORY_PATH  = r"outputs/training_history.json"

if __name__ == "__main__":
    device = torch.device('cpu')

    # ── Load splits ───────────────────────────────────────────────────────
    train_split, val_split, test_split = load_splits(SPLITS_PATH)

    # ── Load model ────────────────────────────────────────────────────────
    model = BrugadaCNN(dropout=0.3)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print("✅ Checkpoint loaded")

    # ── Build data loaders ────────────────────────────────────────────────
    import torch.nn as nn
    criterion = nn.BCEWithLogitsLoss()

    val_dataset  = BrugadaDataset(val_split['ids'],  val_split['labels'],
                                   DATA_DIR, transform=preprocess_signal)
    test_dataset = BrugadaDataset(test_split['ids'], test_split['labels'],
                                   DATA_DIR, transform=preprocess_signal)

    val_loader  = DataLoader(val_dataset,  batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ── Step 1: find best threshold on VAL set ────────────────────────────
    _, _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    from src.evaluate import find_sensitivity_threshold
    best_threshold = find_sensitivity_threshold(val_labels, val_probs,
                                             min_sensitivity=0.80)

    # ── Step 2: evaluate on VAL ───────────────────────────────────────────
    val_metrics = full_report(val_labels, val_probs,
                              threshold=best_threshold, split_name="Validation")

    # ── Step 3: evaluate on TEST — run this only once ─────────────────────
    _, _, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = full_report(test_labels, test_probs,
                               threshold=best_threshold, split_name="Test")

    # ── Step 4: plots ─────────────────────────────────────────────────────
    plot_training_curves(HISTORY_PATH,
                         save_path='outputs/figures/training_curves.png')
    plot_roc_curve(test_labels, test_probs,
                   save_path='outputs/figures/roc_curve.png')

    print("\n✅ Evaluation complete. Figures saved to outputs/figures/")