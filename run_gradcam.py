import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.model import BrugadaCNN
from src.dataset import BrugadaDataset
from src.preprocessing import load_splits, preprocess_signal
from src.gradcam import run_gradcam_analysis
from src.train import evaluate
import torch.nn as nn

DATA_DIR    = r"C:\Users\Harshini\Downloads\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\files"   # ← same as before
SPLITS_PATH = r"outputs/splits.json"
SEEDS       = [42, 123, 7]
THRESHOLD   = 0.55

if __name__ == "__main__":
    _, val_split, test_split = load_splits(SPLITS_PATH)

    test_dataset = BrugadaDataset(
        test_split['ids'], test_split['labels'],
        DATA_DIR, transform=preprocess_signal
    )

    # Load ensemble — use seed 123 for GradCAM (best individual AUROC 0.9649)
    model = BrugadaCNN(dropout=0.3)
    model.load_state_dict(torch.load(
        'outputs/checkpoints/best_model_seed123.pt',
        map_location='cpu'
    ))
    model.eval()
    print("✅ Model loaded (seed 123 — best individual val AUROC)")

    # Find true positives and false negatives in test set for analysis
    criterion   = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    _, _, test_probs, test_labels = evaluate(
        model, test_loader, criterion, torch.device('cpu')
    )

    preds = (test_probs >= THRESHOLD).astype(int)

    tp_indices = [i for i,(p,l) in enumerate(zip(preds, test_labels))
                  if p==1 and l==1]
    fn_indices = [i for i,(p,l) in enumerate(zip(preds, test_labels))
                  if p==0 and l==1]
    tn_indices = [i for i,(p,l) in enumerate(zip(preds, test_labels))
                  if p==0 and l==0][:2]   # just 2 normals for comparison

    print(f"\nTrue Positives  : {len(tp_indices)} → {tp_indices}")
    print(f"False Negatives : {len(fn_indices)} → {fn_indices}")
    print(f"True Negatives  : {tn_indices} (2 samples for comparison)")

    # Analyze: all TPs + all FNs + 2 TNs
    indices_to_analyze = tp_indices + fn_indices + tn_indices

    print(f"\nGenerating Grad-CAM for {len(indices_to_analyze)} samples...")
    results = run_gradcam_analysis(
        model, test_dataset,
        indices  = indices_to_analyze,
        save_dir = 'outputs/figures/gradcam',
        leads_to_plot = [6, 7, 8],   # V1, V2, V3
        threshold     = THRESHOLD
    )

    print(f"\n✅ Grad-CAM complete. Check outputs/figures/gradcam/")
    print(f"\nSummary:")
    for r in results:
        print(f"  Sample {r['idx']:3d} | {r['type']:16s} | prob={r['prob']:.3f}")