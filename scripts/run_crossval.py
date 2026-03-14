# run_crossval.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.nn as nn

from src.model import BrugadaCNN
from src.dataset import BrugadaDataset
from src.preprocessing import preprocess_signal, merge_labels
from src.train import run_training, evaluate

DATA_DIR      = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"   # ← same as before
METADATA_PATH = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\metadata.csv"

config = {
    'batch_size': 16,
    'lr'        : 1e-3,
    'epochs'    : 80,
    'patience'  : 15,
    'seed'      : 42,
}

if __name__ == "__main__":
    df     = pd.read_csv(METADATA_PATH)
    ids    = df['patient_id'].astype(str).tolist()
    labels = merge_labels(df['brugada'].tolist())

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aurocs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(ids, labels)):
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1}/5")
        print(f"{'='*50}")

        train_split = {
            'ids'   : [ids[i] for i in train_idx],
            'labels': [labels[i] for i in train_idx]
        }
        val_split = {
            'ids'   : [ids[i] for i in val_idx],
            'labels': [labels[i] for i in val_idx]
        }

        model   = BrugadaCNN(dropout=0.3)
        history = run_training(model, train_split, val_split,
                               DATA_DIR, config)

        # Load best checkpoint and evaluate
        model.load_state_dict(torch.load(
            'outputs/checkpoints/best_model.pt', map_location='cpu'))

        val_dataset = BrugadaDataset(
            val_split['ids'], val_split['labels'],
            DATA_DIR, transform=preprocess_signal
        )
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        criterion  = nn.BCEWithLogitsLoss()

        _, auroc, _, _ = evaluate(model, val_loader, criterion,
                                   torch.device('cpu'))
        fold_aurocs.append(auroc)
        print(f"  Fold {fold+1} AUROC: {auroc:.4f}")

    print(f"\n{'='*50}")
    print(f"  5-Fold CV Results")
    print(f"{'='*50}")
    for i, a in enumerate(fold_aurocs):
        print(f"  Fold {i+1}: {a:.4f}")
    print(f"\n  Mean AUROC : {np.mean(fold_aurocs):.4f}")
    print(f"  Std  AUROC : {np.std(fold_aurocs):.4f}")
    print(f"  95% CI     : [{np.mean(fold_aurocs)-1.96*np.std(fold_aurocs):.4f}, "
          f"{np.mean(fold_aurocs)+1.96*np.std(fold_aurocs):.4f}]")