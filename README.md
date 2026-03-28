# BrugadaCNN — Automated Brugada Syndrome Detection from 12-Lead ECG

> **IDSC 2026 Submission | Team MediScript | Dataset: Brugada-HUCA (PhysioNet)**

Automated detection of Brugada syndrome — a rare but potentially fatal cardiac 
arrhythmia — from 12-lead ECG recordings using an ensemble of 1D Convolutional 
Neural Networks with Grad-CAM interpretability.

---

## 🏆 Results at a Glance

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.9612** |
| **Sensitivity** | **0.9167** ✅ exceeds 80–90% clinical benchmark |
| **Specificity** | 0.9767 |
| **F1 Score** | 0.9130 |
| **False Positives** | 1 |
| **False Negatives** | 1 |
| **5-Fold CV AUROC** | 0.9404 ± 0.0217 |

Model: Ensemble of 3 × BrugadaCNN (seeds 42, 123, 7) | Threshold: τ = 0.55

---

## 📁 Repository Structure
```
src/                        # Core modules
  dataset.py                # ECGDataset — loads WFDB records
  preprocessing.py          # Bandpass filter, normalization, splits
  model.py                  # BrugadaCNN architecture (477K params)
  train.py                  # Training loop with early stopping
  evaluate.py               # Metrics, ROC curve, confusion matrix
  gradcam.py                # Grad-CAM 1D implementation

scripts/                    # Runnable pipelines
  verify.py                 # Verify all 363 records load correctly
  run_training.py           # Train a single model
  run_ensemble_training.py  # Train 3-seed ensemble + evaluate
  run_evaluation.py         # Evaluate saved checkpoint
  run_gradcam.py            # Generate Grad-CAM saliency figures
  run_crossval.py           # 5-fold cross-validation
  run_classical_ml.py       # Classical ML baseline
  gen_confusion_matrix.py   # Generate confusion matrix figure

notebooks/
  01_eda.ipynb              # Exploratory data analysis
  02_training.ipynb         # Training walkthrough & results
  03_xai.ipynb              # Grad-CAM interpretability analysis

outputs/
  figures/                  # ROC curves, training curves, Grad-CAM plots
  checkpoints/              # Model weights (not tracked in git)
  final_results.json        # All metrics
  classical_ml_results.json # Classical ML comparison
```

---

## ⚙️ Setup & Reproduction

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Follow the instructions in `data/README.md` to download Brugada-HUCA from PhysioNet.

### 3. Set your data paths
Open `config.py` and update the two path variables:
```python
DATA_DIR      = r"path/to/brugada-huca/files"
METADATA_PATH = r"path/to/brugada-huca/metadata.csv"
```

### 4. Verify data integrity
```bash
python scripts/verify.py
# Expected: 363/363 records loaded, shape (1200, 12)
```

### 5. Run the full ensemble pipeline
```bash
python scripts/run_ensemble_training.py
```

### 6. Generate interpretability figures
```bash
python scripts/run_gradcam.py
```

---

## 🧠 Model Architecture

BrugadaCNN processes each ECG as a **(12 leads × 1200 samples)** multichannel 
1D signal through four convolutional blocks (channels: 12→32→64→128→256, 
kernel sizes: 7→5→5→3), Global Average Pooling, and a two-layer classifier. 
Total: **477,761 parameters**.

Class imbalance (3.37:1) is handled via weighted BCE loss (w⁺ = 3.77). 
Three independently trained models (seeds 42, 123, 7) are ensembled by 
probability averaging to reduce initialization variance.

---

## 🔬 Dataset

**Brugada-HUCA** — 363 12-lead ECG recordings, 100 Hz, 12 seconds.  
76 Brugada syndrome cases | 287 Normal controls  
Source: Hospital Universitario Central de Asturias, Spain.

### Mandatory Citations
Costa Cortez, N., & Garcia Iglesias, D. (2026). Brugada-HUCA: 12-Lead ECG 
Recordings for the Study of Brugada Syndrome (version 1.0.0). PhysioNet.  
https://doi.org/10.13026/0m2w-dy83

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.  
Circulation, 101(23), e215–e220.