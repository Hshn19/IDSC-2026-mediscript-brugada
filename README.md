cat > README.md << 'EOF'
# Brugada Syndrome Detection — IDSC 2026

Automated classification of Brugada syndrome from 12-lead ECG recordings using deep learning.
Built for the [International Data Science Challenge 2026](https://idsc2026.github.io/).

## Dataset
[Brugada-HUCA](https://physionet.org/content/brugada-huca/1.0.0/) — 363 subjects
(76 Brugada, 287 Normal), 12-lead ECG, 12s @ 100 Hz. See `data/README.md` for setup.

## Project Structure
```
src/               # Core Python modules
notebooks/         # EDA, training, XAI notebooks
outputs/           # Figures and model checkpoints
data/              # Download instructions (no raw data in repo)
```

## Reproducing Results

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset following `data/README.md`

3. Run notebooks in order:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_training.ipynb`
   - `notebooks/03_xai.ipynb`

## Team
IDSC 2026 — UPM Malaysia

## Citation
See `data/README.md` for mandatory dataset citations.
EOF