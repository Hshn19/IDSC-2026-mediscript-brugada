import os
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


# ── Signal Processing ─────────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """
    Butterworth bandpass filter applied to each lead independently.

    Clinical reasoning:
    - Below 0.5 Hz  : baseline wander (patient movement, respiration)
    - Above 40.0 Hz : high-frequency noise, muscle artifacts
    - Keeps         : QRS complex, ST segment, T-wave — exactly what we need

    Args:
        signal : np.array of shape (12, 1200)
        lowcut : lower frequency bound in Hz
        highcut: upper frequency bound in Hz
        fs     : sampling frequency (100 Hz for this dataset)
        order  : filter order — 4 is standard for ECG

    Returns:
        filtered signal of shape (12, 1200)
    """
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # filtfilt = zero-phase filtering (no time shift) — important for ECG morphology
    return filtfilt(b, a, signal, axis=1)


def normalize_signal(signal):
    """
    Z-score normalize each lead independently.

    Why per-lead: each lead has different amplitude ranges.
    Normalizing globally would let dominant leads (II, V5) overwhelm others.

    Args:
        signal : np.array of shape (12, 1200)

    Returns:
        normalized signal of shape (12, 1200)
    """
    mean = signal.mean(axis=1, keepdims=True)
    std  = signal.std(axis=1, keepdims=True)
    std[std == 0] = 1.0   # flat leads (e.g. aVR in some patients) — avoid div/0
    return (signal - mean) / std


def preprocess_signal(signal):
    """
    Full preprocessing pipeline: filter → normalize.
    Applied to every signal before it enters the model.

    Args:
        signal : raw np.array of shape (12, 1200)

    Returns:
        preprocessed np.array of shape (12, 1200)
    """
    signal = bandpass_filter(signal)
    signal = normalize_signal(signal)
    return signal.astype(np.float32)

def merge_labels(labels):
    """
    Merge label 2 (probable/borderline Brugada) into label 1 (Brugada).

    Clinical justification:
    - Label 2 patients exhibit Brugada-type ECG patterns (basal_pattern=1
      in 4/7 cases) and one experienced sudden cardiac death.
    - For a safety-critical classifier, borderline cases should be treated
      as positive to minimize dangerous false negatives.
    - Final label mapping: 0 = Normal, 1 = Brugada (including borderline)

    Args:
        labels : list of ints (0, 1, or 2)

    Returns:
        list of ints (0 or 1 only)
    """
    merged = [1 if l >= 1 else 0 for l in labels]
    print(f"After merging label 2 → 1:")
    print(f"  Normal  (0): {merged.count(0)}")
    print(f"  Brugada (1): {merged.count(1)}")
    return merged

# ── Data Splitting ────────────────────────────────────────────────────────────

def get_splits(metadata_path, save_path=None, seed=42):
    """
    Generate stratified train/val/test splits and optionally save to JSON.

    CRITICAL RULES:
    1. Splits are generated ONCE and saved — every team member loads the same file
    2. Stratified by label — preserves 83:280 ratio in every split
    3. SMOTE (if used) is applied to training set ONLY, after splitting

    Split sizes (approximate):
    - Train : 70% → ~254 samples (58 Brugada, 196 Normal)
    - Val   : 15% → ~54  samples (12 Brugada,  42 Normal)
    - Test  : 15% → ~55  samples (13 Brugada,  42 Normal)

    Args:
        metadata_path : path to metadata.csv
        save_path     : if provided, saves splits as JSON to this path
        seed          : random seed for reproducibility

    Returns:
        train, val, test — each a dict with keys 'ids' and 'labels'
    """
    df     = pd.read_csv(metadata_path)
    ids    = df['patient_id'].astype(str).tolist()
    labels = merge_labels(df['brugada'].tolist())

    # Step 1: carve out 15% test set
    ids_tv, ids_test, y_tv, y_test = train_test_split(
        ids, labels,
        test_size=0.15,
        stratify=labels,
        random_state=seed
    )

    # Step 2: split remaining 85% into train (82.4%) and val (17.6%)
    # 0.176 * 0.85 ≈ 0.15 of total
    ids_train, ids_val, y_train, y_val = train_test_split(
        ids_tv, y_tv,
        test_size=0.176,
        stratify=y_tv,
        random_state=seed
    )

    splits = {
        'train': {'ids': ids_train, 'labels': y_train},
        'val'  : {'ids': ids_val,   'labels': y_val},
        'test' : {'ids': ids_test,  'labels': y_test}
    }

    # Print summary
    for name, split in splits.items():
        n_pos = sum(split['labels'])
        n_neg = len(split['labels']) - n_pos
        print(f"{name:5s} → {len(split['labels']):3d} samples "
              f"| Brugada: {n_pos:2d} | Normal: {n_neg:3d}")

    # Save splits so the whole team uses identical data
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"\n💾 Splits saved to: {save_path}")

    return splits['train'], splits['val'], splits['test']


def load_splits(save_path):
    """
    Load previously saved splits from JSON.
    Use this instead of get_splits() after the first run.

    Args:
        save_path : path to the saved splits JSON file

    Returns:
        train, val, test — each a dict with keys 'ids' and 'labels'
    """
    with open(save_path, 'r') as f:
        splits = json.load(f)

    print(f"✅ Splits loaded from: {save_path}")
    for name, split in splits.items():
        print(f"  {name:5s} → {len(split['labels'])} samples")

    return splits['train'], splits['val'], splits['test']