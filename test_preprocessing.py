import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import get_splits

METADATA_PATH = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\metadata.csv"   # ← same path as verify.py
SPLITS_PATH   = r"outputs/splits.json"

if __name__ == "__main__":
    train, val, test = get_splits(METADATA_PATH, save_path=SPLITS_PATH)