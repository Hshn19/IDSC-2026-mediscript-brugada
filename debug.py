import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import BrugadaDataset
from src.preprocessing import load_splits, preprocess_signal

DATA_DIR    = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"   # ← same as run_training.py
SPLITS_PATH = r"outputs/splits.json"

if __name__ == "__main__":
    train_split, val_split, _ = load_splits(SPLITS_PATH)

    # ── Check 1: label distribution ───────────────────────────────────────
    print("=== Label Check ===")
    print(f"Unique labels in train : {set(train_split['labels'])}")
    print(f"Unique labels in val   : {set(val_split['labels'])}")
    print(f"Train label counts     : 0={train_split['labels'].count(0)}, "
                                  f"1={train_split['labels'].count(1)}")

    # ── Check 2: single sample signal stats ───────────────────────────────
    print("\n=== Signal Check (first 3 samples) ===")
    dataset = BrugadaDataset(
        train_split['ids'], train_split['labels'],
        DATA_DIR, transform=preprocess_signal
    )

    for i in range(3):
        signal, label = dataset[i]
        sig = signal.numpy()
        print(f"  Sample {i} | label={int(label)} | "
              f"shape={sig.shape} | "
              f"min={sig.min():.3f} max={sig.max():.3f} "
              f"mean={sig.mean():.3f} std={sig.std():.3f} "
              f"nan={np.isnan(sig).any()} inf={np.isinf(sig).any()}")

    # ── Check 3: full batch label distribution ────────────────────────────
    print("\n=== Batch Label Check ===")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch_idx, (signals, labels) in enumerate(loader):
        unique = labels.unique().tolist()
        print(f"  Batch {batch_idx} | labels present: {unique} "
              f"| size: {len(labels)}")
        if batch_idx == 4:
            break