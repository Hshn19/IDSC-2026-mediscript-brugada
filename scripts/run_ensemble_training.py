import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import DATA_DIR, SPLITS_PATH, ENSEMBLE_DIR, SEEDS, THRESHOLD

import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.model import BrugadaCNN
from src.dataset import BrugadaDataset
from src.preprocessing import load_splits, preprocess_signal
from src.train import run_training, evaluate
from src.evaluate import find_sensitivity_threshold, full_report, plot_roc_curve

DATA_DIR    = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"   # ← same as before
SPLITS_PATH = r"outputs/splits.json"

SEEDS    = [42, 123, 7]
ENSEMBLE_DIR = "outputs/checkpoints"

config = {
    'batch_size' : 16,
    'lr'         : 1e-3,
    'epochs'     : 80,
    'patience'   : 15,
}

if __name__ == "__main__":
    train_split, val_split, test_split = load_splits(SPLITS_PATH)

    val_probs_all  = []
    test_probs_all = []

    # ── Train one model per seed ──────────────────────────────────────────
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Training model with seed {seed}")
        print(f"{'='*60}")

        config['seed'] = seed
        model = BrugadaCNN(dropout=0.3)

        # Temporarily save to seed-specific checkpoint
        ckpt_path = f"{ENSEMBLE_DIR}/best_model_seed{seed}.pt"

        history = run_training(model, train_split, val_split,
                               DATA_DIR, config)

        # Rename the saved checkpoint to seed-specific name
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        os.rename(f"{ENSEMBLE_DIR}/best_model.pt", ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # ── Ensemble inference ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Ensemble Inference")
    print(f"{'='*60}")

    import torch.nn as nn
    criterion = nn.BCEWithLogitsLoss()
    device    = torch.device('cpu')

    val_dataset  = BrugadaDataset(val_split['ids'],  val_split['labels'],
                                   DATA_DIR, transform=preprocess_signal)
    test_dataset = BrugadaDataset(test_split['ids'], test_split['labels'],
                                   DATA_DIR, transform=preprocess_signal)
    val_loader   = DataLoader(val_dataset,  batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    val_probs_all  = []
    test_probs_all = []

    for seed in SEEDS:
        ckpt_path = f"{ENSEMBLE_DIR}/best_model_seed{seed}.pt"
        model     = BrugadaCNN(dropout=0.3)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        _, _, val_probs,  val_labels  = evaluate(model, val_loader,
                                                  criterion, device)
        _, _, test_probs, test_labels = evaluate(model, test_loader,
                                                  criterion, device)

        val_auroc  = roc_auc_score(val_labels,  val_probs)
        test_auroc = roc_auc_score(test_labels, test_probs)
        print(f"  Seed {seed:3d} | Val AUROC: {val_auroc:.4f} "
              f"| Test AUROC: {test_auroc:.4f}")

        val_probs_all.append(val_probs)
        test_probs_all.append(test_probs)

    # Average probabilities across models
    val_probs_ensemble  = np.mean(val_probs_all,  axis=0)
    test_probs_ensemble = np.mean(test_probs_all, axis=0)

    val_auroc_ens  = roc_auc_score(val_labels,  val_probs_ensemble)
    test_auroc_ens = roc_auc_score(test_labels, test_probs_ensemble)

    print(f"\n  Ensemble | Val AUROC: {val_auroc_ens:.4f} "
          f"| Test AUROC: {test_auroc_ens:.4f}")

    # ── Clinical threshold on val ensemble probs ──────────────────────────
    best_threshold = find_sensitivity_threshold(
        val_labels, val_probs_ensemble, min_sensitivity=0.80
    )

    # ── Final reports ─────────────────────────────────────────────────────
    full_report(val_labels,  val_probs_ensemble,
                threshold=best_threshold, split_name="Ensemble Validation")
    full_report(test_labels, test_probs_ensemble,
                threshold=best_threshold, split_name="Ensemble Test")

    plot_roc_curve(test_labels, test_probs_ensemble,
                   save_path='outputs/figures/roc_curve_ensemble.png')

    # ── Save ensemble probs for Grad-CAM ──────────────────────────────────
    np.save('outputs/val_probs_ensemble.npy',  val_probs_ensemble)
    np.save('outputs/test_probs_ensemble.npy', test_probs_ensemble)
    np.save('outputs/test_labels.npy',         test_labels)

    print("\n✅ Ensemble training and evaluation complete.")