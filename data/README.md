cat > data/README.md << 'EOF'
# Data

The dataset used in this project is **Brugada-HUCA** from PhysioNet.

## Download Instructions

1. Go to: https://physionet.org/content/brugada-huca/1.0.0/
2. Download the ZIP file (14.4 MB) or use wget:
```bash
wget -r -N -c -np https://physionet.org/files/brugada-huca/1.0.0/
```

3. Place the contents so the structure looks like:
```
data/
├── metadata.csv
├── metadata_dictionary.csv
├── RECORDS
└── files/
    ├── 100001/
    │   ├── 100001.dat
    │   └── 100001.hea
    ├── 100002/
    ...
```

## Citation

Costa Cortez, N., & Garcia Iglesias, D. (2026). Brugada-HUCA: 12-Lead ECG Recordings
for the Study of Brugada Syndrome (version 1.0.0). PhysioNet.
https://doi.org/10.13026/0m2w-dy83

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation, 101(23), e215–e220.
EOF