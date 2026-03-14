# Brugada Syndrome Detection — IDSC 2026

Automated classification of Brugada syndrome from 12-lead ECG recordings
using an ensemble of 1D CNNs with Grad-CAM interpretability.

Built for the [International Data Science Challenge 2026](https://idsc2026.github.io/)
hosted by UPM Malaysia.

---

## Results Summary

| Metric | Validation | Test |
|--------|-----------|------|
| AUROC | 0.9483 | **0.9748** |
| F1 Score | 0.8571 | 0.8000 |
| Sensitivity | 0.8182 | 0.6667 |
| Specificity | 0.9773 | **1.0000** |
| Precision | 0.9000 | **1.0000** |
| False Positives | 1 | **0** |

Model: Ensemble of 3 BrugadaCNN (seeds 42, 123, 7) | Threshold: 0.55

---

## Dataset

[Brugada-HUCA](https://physionet.org/content/brugada-huca/1.0.0/) — 363 subjects
(76 Brugada, 287 Normal), 12-lead ECG, 12s @ 100 Hz.

See `data/README.md` for download instructions. Raw data is not stored in this repo.

---

## Project Structure
```
src/                    # Core Python modules
  dataset.py            # ECGDataset class
  preprocessing.py      # Bandpass filter, normalization, splits
  model.py              # BrugadaCNN architecture (477K params)
  train.py              # Training loop with early stopping
  evaluate.py           # Metrics, ROC curve, confusion matrix
  gradcam.py            # Grad-CAM 1D implementation

scripts/                # Runnable scripts
  verify.py             # Verify all 363 records load correctly
  run_training.py       # Train a single model
  run_ensemble_training.py  # Train 3-seed ensemble + evaluate
  run_evaluation.py     # Evaluate saved checkpoint
  run_gradcam.py        # Generate Grad-CAM figures
  run_crossval.py       # 5-fold cross-validation

notebooks/
  01_eda.ipynb          # Exploratory data analysis
  02_training.ipynb     # Training walkthrough
  03_xai.ipynb          # Interpretability analysis

outputs/
  figures/              # ROC curves, training curves, Grad-CAM plots
  checkpoints/          # Saved model weights (not tracked in git)
  final_results.json    # All metrics for report
```

---

## Setup & Reproduction

**1. Clone and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download dataset** following `data/README.md`

**3. Set your data paths** in any script you run:
```python
DATA_DIR      = r"path/to/brugada-huca/files"
METADATA_PATH = r"path/to/brugada-huca/metadata.csv"
```

**4. Verify data loads correctly:**
```bash
python scripts/verify.py
```

**5. Run ensemble training + evaluation:**
```bash
python scripts/run_ensemble_training.py
```

**6. Generate Grad-CAM figures:**
```bash
python scripts/run_gradcam.py
```

---

## Key Design Decisions

- **Label merge:** Label 2 (borderline Brugada, n=7) merged into label 1 based on
  clinical evidence — 4/7 show spontaneous Brugada ECG pattern, 1/7 experienced
  sudden cardiac death
- **Class weighting:** pos_weight=3.77 in BCEWithLogitsLoss to address 3.37:1 imbalance
- **Ensemble:** 3 models with different seeds — stabilizes AUROC variance on small dataset
- **Clinical threshold:** 0.55 (tuned on val set) prioritizes sensitivity over specificity
- **Grad-CAM target:** Last Conv1d in encoder — captures highest-level learned features

---

## Citations

Costa Cortez, N., & Garcia Iglesias, D. (2026). Brugada-HUCA: 12-Lead ECG Recordings
for the Study of Brugada Syndrome (version 1.0.0). PhysioNet.
https://doi.org/10.13026/0m2w-dy83

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation, 101(23), e215–e220.