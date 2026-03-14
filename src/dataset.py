import os
import wfdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BrugadaDataset(Dataset):
    """
    PyTorch Dataset for the Brugada-HUCA 12-lead ECG dataset.

    Args:
        record_ids  : list of patient_id strings e.g. ['188981', '200012']
        labels      : list of ints (0 = Normal, 1 = Brugada)
        data_dir    : path to the folder containing per-patient subfolders
        transform   : optional callable applied to the signal numpy array
    """

    def __init__(self, record_ids, labels, data_dir, transform=None):
        self.record_ids = record_ids
        self.labels     = labels
        self.data_dir   = data_dir
        self.transform  = transform

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        pid    = str(self.record_ids[idx])
        path   = os.path.join(self.data_dir, pid, pid)
        record = wfdb.rdrecord(path)

        # p_signal shape: (1200 samples, 12 leads) → transpose to (12, 1200)
        signal = record.p_signal.T.astype(np.float32)

        # Replace NaNs — some leads can have missing values
        signal = np.nan_to_num(signal, nan=0.0)

        if self.transform:
            signal = self.transform(signal)

        return torch.tensor(signal), torch.tensor(self.labels[idx],
                                                   dtype=torch.float32)


def load_metadata(metadata_path):
    """
    Load metadata CSV and return record IDs and labels.
    
    Returns:
        ids    : list of patient_id strings
        labels : list of ints (0 or 1)
    """
    df     = pd.read_csv(metadata_path)
    ids    = df['patient_id'].astype(str).tolist()
    labels = df['brugada'].tolist()

    print(f"Total records  : {len(df)}")
    print(f"Brugada (1)    : {sum(labels)}")
    print(f"Normal  (0)    : {len(labels) - sum(labels)}")
    print(f"Imbalance ratio: {(len(labels) - sum(labels)) / sum(labels):.2f}:1")

    return ids, labels


def verify_all_records(record_ids, data_dir):
    """
    Sanity check — attempts to load every record and reports:
    - How many loaded successfully
    - Any records that are missing or corrupt
    - Signal shape consistency
    Run this once before training. Never again.
    """
    print(f"\nVerifying {len(record_ids)} records...")
    failed  = []
    shapes  = set()

    for pid in record_ids:
        path = os.path.join(data_dir, str(pid), str(pid))
        try:
            record = wfdb.rdrecord(path)
            signal = record.p_signal
            shapes.add(signal.shape)

            if np.isnan(signal).any():
                print(f"  ⚠️  NaN detected in patient {pid}")

        except Exception as e:
            failed.append((pid, str(e)))

    print(f"\n✅ Successfully loaded : {len(record_ids) - len(failed)}/{len(record_ids)}")
    print(f"❌ Failed              : {len(failed)}")
    print(f"📐 Unique signal shapes: {shapes}  ← should be only (1200, 12)")

    if failed:
        print("\nFailed records:")
        for pid, err in failed:
            print(f"  Patient {pid}: {err}")

    return len(failed) == 0