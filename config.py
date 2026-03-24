# ══════════════════════════════════════════════════════════════════
#  CONFIG — edit these two paths before running any script
# ══════════════════════════════════════════════════════════════════

# Path to the 'files' folder inside your downloaded Brugada-HUCA dataset
# Example (Windows): r"C:\Users\you\Downloads\brugada-huca\files"
# Example (Mac/Linux): "/home/you/data/brugada-huca/files"
DATA_DIR = r"C:\your\path\to\brugada-huca\files"

# Path to metadata.csv inside your downloaded Brugada-HUCA dataset
# Example (Windows): r"C:\Users\you\Downloads\brugada-huca\metadata.csv"
METADATA_PATH = r"C:\your\path\to\brugada-huca\metadata.csv"

# ══════════════════════════════════════════════════════════════════
#  Do not edit below this line
# ══════════════════════════════════════════════════════════════════
SPLITS_PATH   = "outputs/splits.json"
CHECKPOINT    = "outputs/checkpoints/best_model.pt"
ENSEMBLE_DIR  = "outputs/checkpoints"
HISTORY_PATH  = "outputs/training_history.json"
SEEDS         = [42, 123, 7]
THRESHOLD     = 0.55