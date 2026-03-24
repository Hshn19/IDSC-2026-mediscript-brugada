import sys
import os

# Add repo root to path so config.py can be found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import DATA_DIR, METADATA_PATH
from src.dataset import load_metadata, verify_all_records

# ── Edit these two paths to match where your data lives ──────────────────────
METADATA_PATH = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\metadata.csv"
DATA_DIR      = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ids, labels = load_metadata(METADATA_PATH)
    all_ok      = verify_all_records(ids, DATA_DIR)

    if all_ok:
        print("\n✅ All records verified. Safe to proceed to preprocessing.")
    else:
        print("\n⚠️  Fix the failed records before proceeding.")