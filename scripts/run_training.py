import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import BrugadaCNN, count_parameters
from src.preprocessing import load_splits
from src.train import run_training

# ── Edit these paths ──────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"         # ← same as verify.py
SPLITS_PATH = r"outputs/splits.json"
# ─────────────────────────────────────────────────────────────────────────────

config = {
    'batch_size' : 16,
    'lr'         : 1e-3,    # ← back to original
    'epochs'     : 80,      # ← back to original
    'patience'   : 15,      # ← back to original
    'seed'       : 42,
}

if __name__ == "__main__":
    train_split, val_split, test_split = load_splits(SPLITS_PATH)

    model = BrugadaCNN(dropout=0.3)
    count_parameters(model)

    history = run_training(model, train_split, val_split, DATA_DIR, config)