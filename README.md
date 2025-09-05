```markdown
# Cross-Regional Radiomics for Parkinson's Disease Motor Subtyping

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A machine learning framework for distinguishing Parkinson's disease motor subtypes (tremor-dominant vs. postural instability gait difficulty) using novel cross-regional brain features.

## Overview

This repository implements a comprehensive analysis comparing **Motor-Circuit Features** (capturing relationships between brain regions) versus traditional **Single-ROI Features** (analyzing regions independently) for Parkinson's disease subtype classification.

**Key Finding**: Motor-circuit features significantly outperform single-region approaches when enhanced through proper feature engineering (AUC: 0.821±0.117 vs 0.650±0.220, p=0.0012).

## Installation

```bash
git clone https://github.com/yourusername/parkinson-motor-circuit-classification.git
cd parkinson-motor-circuit-classification
pip install -r requirements.txt
```

## Data Structure

Your data should be organized as follows:

```
Your-Data-Folder/
├── TD/                              # Tremor-Dominant patients
│   ├── 001/                         # Patient ID folder
│   │   └── T1/                      # T1-weighted images
│   │       ├── 001.nii.gz           # Main T1 image
│   │       └── w_thrp/              # ROI masks folder
│   │           └── Subject/         
│   │               ├── L_Pu.nii.gz  # Left Putamen
│   │               ├── R_Pu.nii.gz  # Right Putamen
│   │               ├── L_CN.nii.gz  # Left Caudate
│   │               ├── R_CN.nii.gz  # Right Caudate
│   │               ├── L_Pa.nii.gz  # Left Pallidum
│   │               ├── R_Pa.nii.gz  # Right Pallidum
│   │               ├── L_SN.nii.gz  # Left Substantia Nigra
│   │               └── R_SN.nii.gz  # Right Substantia Nigra
├── PIGD/                            # PIGD patients (same structure)
├── TD_metadata.xlsx                 # TD patient metadata
└── PIGD_metadata.xlsx              # PIGD patient metadata
```

**Metadata files must contain:**

| SUBJECT | SCANNER_MODEL | CENTER | AGE | SEX |
|---------|---------------|--------|-----|-----|
| 001 | Siemens_Skyra | Center_A | 65 | M |
| 002 | Siemens_Prisma | Center_B | 72 | F |

## Configuration

Edit `config.py` to set your data paths:

```python
METADATA_PATHS = {
    'td': r'C:\Your\Path\To\Data\TD_metadata.xlsx',
    'pigd': r'C:\Your\Path\To\Data\PIGD_metadata.xlsx'
}

BASE_PATHS = {
    'td': r'C:\Your\Path\To\Data\TD',
    'pigd': r'C:\Your\Path\To\Data\PIGD'
}
```

## Usage

Run the complete analysis:

```bash
python main.py
```

This will:
1. Load data and extract features from both feature types
2. Test 6 feature engineering scenarios
3. Perform center-based cross-validation with 6 classifiers
4. Generate comparison statistics and visualizations

## Output

The analysis generates:
- `summary_metrics_table.csv`: Performance comparison across scenarios
- `detailed_classifier_metrics.xlsx`: Detailed per-classifier results  
- `comprehensive_motor_vs_single_chord_comparison.html`: Interactive feature consistency visualization

## Methodology

**Single-ROI Features (48 features)**
- 6 statistical measures from 8 bilateral motor regions
- Traditional approach analyzing regions independently

**Motor-Circuit Features (60 features)**
- Cross-regional ratios (putamen-substantia nigra, caudate-putamen)
- Asymmetry indices across hemispheres
- Circuit volume relationships
- Shape and distribution patterns

**Feature Engineering Scenarios**
1. Raw Features (baseline)
2. Raw + Preprocessing (robust scaling)
3. Enhanced + Preprocessing (polynomial features, interactions)
4. Enhanced + Robust (+ feature selection, k=15)
5. Enhanced + Standard (alternative scaling)
6. Enhanced + MinMax (alternative scaling)

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- nibabel
- plotly
- openpyxl
- scipy

## Citation

```bibtex
@article{hosseini2024crossregional,
  title={Cross-Regional Radiomics: A Novel Framework for Relationship-Based Feature Extraction with Validation in Parkinson's Disease Motor Subtyping},
  author={Hosseini, Mahboube Sadat and Aghamiri, Seyyed Mahmoud Reza and Panahi, Mehdi},
  journal={Submitted},
  year={2024}
}
```

## License

MIT License
```
