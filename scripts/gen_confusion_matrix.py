import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from config import DATA_DIR, SPLITS_PATH, ENSEMBLE_DIR, SEEDS, THRESHOLD
from src.model import BrugadaCNN
from src.dataset import BrugadaDataset
from src.preprocessing import load_splits, preprocess_signal
from src.train import evaluate
from src.evaluate import plot_confusion_matrix, full_report

if __name__ == "__main__":
    _, _, test_split = load_splits(SPLITS_PATH)

    device    = torch.device('cpu')
    criterion = nn.BCEWithLogitsLoss()

    test_dataset = BrugadaDataset(
        test_split['ids'], test_split['labels'],
        DATA_DIR, transform=preprocess_signal
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load all ensemble models and average probabilities
    all_probs = []
    for seed in SEEDS:
        ckpt = os.path.join(ENSEMBLE_DIR, f"best_model_seed{seed}.pt")
        model = BrugadaCNN(dropout=0.3)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        _, _, probs, labels = evaluate(model, test_loader, criterion, device)
        all_probs.append(probs)
        print(f"  Seed {seed} loaded ✓")

    # Ensemble average
    ensemble_probs = np.mean(all_probs, axis=0)
    preds = (ensemble_probs >= THRESHOLD).astype(int)

    # Print full metrics report
    full_report(labels, ensemble_probs, threshold=THRESHOLD, split_name="Test")

    # Save confusion matrix figure
    os.makedirs('outputs/figures', exist_ok=True)
    plot_confusion_matrix(
        labels, preds,
        save_path='outputs/figures/confusion_matrix.png'
    )