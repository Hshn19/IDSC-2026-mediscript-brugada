import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                              confusion_matrix, classification_report)
import wfdb
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import load_splits, merge_labels

DATA_DIR      = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"       # ← update
METADATA_PATH = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\metadata.csv" # ← update
SPLITS_PATH   = r"outputs/splits.json"

# ── Feature Extraction ────────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal, axis=1)

def extract_features(patient_id, data_dir):
    """
    Extract hand-crafted ECG features clinically relevant to Brugada.
    Focus on V1 (idx=6), V2 (idx=7), V3 (idx=8).

    Features per lead:
    - Mean, std, max, min amplitude
    - Peak-to-peak range
    - Energy (sum of squares)
    - Zero crossing rate
    - ST segment mean (samples 50-80 after each R-peak proxy)

    Returns: 1D feature vector
    """
    path   = os.path.join(data_dir, str(patient_id), str(patient_id))
    record = wfdb.rdrecord(path)
    signal = record.p_signal.T.astype(np.float32)  # (12, 1200)
    signal = bandpass_filter(signal)

    features = []
    # Extract from all 12 leads but weight V1-V3 by including extra features
    for lead_idx in range(12):
        lead = signal[lead_idx]
        features.extend([
            np.mean(lead),
            np.std(lead),
            np.max(lead),
            np.min(lead),
            np.max(lead) - np.min(lead),          # peak-to-peak
            np.sum(lead**2) / len(lead),           # energy
            np.mean(np.abs(np.diff(lead))),        # mean absolute diff
            np.sum(np.diff(np.sign(lead)) != 0) / len(lead),  # zero crossing rate
            np.percentile(lead, 25),
            np.percentile(lead, 75),
            np.percentile(lead, 75) - np.percentile(lead, 25),  # IQR
        ])

    return np.array(features, dtype=np.float32)

def load_features(split, data_dir):
    X, y = [], []
    for pid, label in zip(split['ids'], split['labels']):
        try:
            feats = extract_features(pid, data_dir)
            X.append(feats)
            y.append(label)
        except Exception as e:
            print(f"  Warning: skipped {pid} — {e}")
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_test, name):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    auroc       = roc_auc_score(y_test, y_prob)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

    return {
        'auroc': auroc, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_split, val_split, test_split = load_splits(SPLITS_PATH)

    print("Extracting features — this takes ~5 minutes...")
    X_train, y_train = load_features(train_split, DATA_DIR)
    X_val,   y_val   = load_features(val_split,   DATA_DIR)
    X_test,  y_test  = load_features(test_split,  DATA_DIR)

    # Combine train+val for final classical ML training
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    print(f"\nFeature shape: {X_train.shape}")
    print(f"Train+Val: {X_tv.shape} | Test: {X_test.shape}")

    # Scale features
    scaler  = StandardScaler()
    X_tv_sc = scaler.fit_transform(X_tv)
    X_ts_sc = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ───────────────────────────────────────────────
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
    lr.fit(X_tv_sc, y_tv)
    results['Logistic Regression'] = evaluate_model(
        lr, X_ts_sc, y_test, "Logistic Regression")

    # ── SVM ───────────────────────────────────────────────────────────────
    svm = SVC(kernel='rbf', class_weight='balanced',
              probability=True, C=1.0, gamma='scale')
    svm.fit(X_tv_sc, y_tv)
    results['SVM (RBF)'] = evaluate_model(
        svm, X_ts_sc, y_test, "SVM (RBF kernel)")

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                 max_depth=8, random_state=42)
    rf.fit(X_tv_sc, y_tv)
    results['Random Forest'] = evaluate_model(
        rf, X_ts_sc, y_test, "Random Forest")

    # ── Summary Table ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Classical ML Summary (Test Set)")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'AUROC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}")
    print(f"  {'-'*52}")
    for name, r in results.items():
        print(f"  {name:<22} {r['auroc']:>7.4f} {r['f1']:>7.4f} "
              f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")
    print(f"\n  BrugadaCNN Ensemble  "
          f"{'0.9748':>7} {'0.8000':>7} {'0.6667':>7} {'1.0000':>7}  ← DL model")

    # Save results
    results['BrugadaCNN Ensemble'] = {
        'auroc': 0.9748, 'f1': 0.8000,
        'sensitivity': 0.6667, 'specificity': 1.0000
    }
    with open('outputs/classical_ml_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n💾 Saved to outputs/classical_ml_results.json")