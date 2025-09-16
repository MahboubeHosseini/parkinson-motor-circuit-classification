# Motor-Circuit vs Single-ROI Features for Parkinson's Disease Subtype Classification

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-preprint-orange.svg)](#)

## 🧠 Overview

This repository contains the implementation of our comprehensive analysis comparing **Motor-Circuit-Features** versus traditional **Single-ROI-Features** for Parkinson's disease (PD) subtype classification. Our study demonstrates that novel motor circuit-based features, when properly enhanced through sophisticated feature engineering, provide superior discriminative performance for distinguishing between tremor-dominant (TD) and postural instability gait difficulty (PIGD) subtypes.

## 🔬 Research Highlights

- **Novel Motor-Circuit-Features**: Comprehensive feature set capturing cross-regional relationships within the basal ganglia-thalamocortical circuit
- **Multi-Scenario Analysis**: Systematic evaluation across 6 distinct feature engineering scenarios
- **Asymmetry Features**: 24 specialized features capturing the inherent asymmetric nature of PD pathology
- **Rigorous Validation**: Center-based cross-validation ensuring real-world clinical generalizability
- **Interactive Visualizations**: Combined chord diagrams showing feature consistency analysis

## 📊 Key Findings

- **Motor-Circuit-Features** demonstrate superior performance over traditional Single-ROI-Features
- **Feature engineering is crucial**: Raw features show limited discriminative power due to noise contamination
- **Optimal configuration**: Enhanced features with robust scaling and feature selection (k=15) achieve best performance
- **Cross-center validation**: Results generalize across different imaging centers
- **Comprehensive analysis**: Tested across 6 classifiers with consistent superiority patterns

## 🎯 What This Code Does

This repository implements a comprehensive machine learning pipeline for **Parkinson's Disease (PD) subtype classification**, specifically distinguishing between:

- **Tremor-Dominant (TD)**: Patients with predominant tremor symptoms
- **Postural Instability Gait Difficulty (PIGD)**: Patients with balance and gait problems

### 🔬 Core Innovation

The main contribution is demonstrating that **Motor-Circuit-Features** (capturing relationships between brain regions) outperform traditional **Single-ROI-Features** (analyzing regions independently) when properly enhanced through sophisticated feature engineering.

### 🧠 Scientific Rationale

**Why Motor-Circuit-Features?**
- PD affects the **basal ganglia-thalamocortical circuit** as a network
- Different subtypes show **distinct circuit dysfunction patterns**
- Traditional approaches ignore **cross-regional relationships**
- Circuit-based features capture **pathophysiologically relevant** information

### 🚀 Complete Usage Guide

#### Step 1: Installation & Setup

**1.1 Clone the repository:**
```bash
git clone https://github.com/yourusername/Motor-Circuit-PD-Classification.git
cd Motor-Circuit-PD-Classification
```

**1.2 Install dependencies:**
```bash
pip install -r requirements.txt
```

#### Step 2: Prepare Your Data

**2.1 Create folder structure:**
```
📦 Your-Data-Folder/
├── 📁 TD/                              # Tremor-Dominant patients
│   ├── 📁 001/                         # Patient ID folder
│   │   └── 📁 T1/                      # T1-weighted images
│   │       ├── 📄 001.nii.gz           # Main T1 image
│   │       └── 📁 w_thrp/              # Processed masks
│   │           └── 📁 Subject/         # Subject-specific masks
│   │               ├── 📄 L_Pu.nii.gz  # Left Putamen
│   │               ├── 📄 R_Pu.nii.gz  # Right Putamen
│   │               ├── 📄 L_CN.nii.gz  # Left Caudate
│   │               ├── 📄 R_CN.nii.gz  # Right Caudate
│   │               ├── 📄 L_Pa.nii.gz  # Left Pallidum
│   │               ├── 📄 R_Pa.nii.gz  # Right Pallidum
│   │               ├── 📄 L_SN.nii.gz  # Left Substantia Nigra
│   │               └── 📄 R_SN.nii.gz  # Right Substantia Nigra
│   └── 📁 002/, 📁 003/, ...           # More patients
├── 📁 PIGD/                            # PIGD patients (same structure)
├── 📄 TD_metadata.xlsx                 # TD metadata
└── 📄 PIGD_metadata.xlsx              # PIGD metadata
```

**2.2 Create Excel metadata files:**

**TD_metadata.xlsx & PIGD_metadata.xlsx must contain:**
| SUBJECT | SCANNER_MODEL | CENTER | AGE | SEX |
|---------|---------------|--------|-----|-----|
| 1 | Siemens_Skyra | Center_A | 45 | M |
| 2 | GE_Discovery | Center_B | 67 | F |
| 3 | Philips_Achieva | Center_A | 52 | M |

**Required columns:**
- `SUBJECT`: Patient ID (must match folder name)
- `SCANNER_MODEL`: MRI scanner model
- `CENTER`: Imaging center name
- `AGE`: Patient age
- `SEX`: Patient gender (M/F)

#### Step 3: Configure Paths

**3.1 Edit `config.py`:**
```python
# Update lines 15-25 with your actual paths
METADATA_PATHS = {
    'td': r'C:\Your\Path\To\Data\TD_metadata.xlsx',
    'pigd': r'C:\Your\Path\To\Data\PIGD_metadata.xlsx'
}

BASE_PATHS = {
    'td': r'C:\Your\Path\To\Data\TD',
    'pigd': r'C:\Your\Path\To\Data\PIGD'
}
```

#### Step 4: Run Analysis

**4.1 Execute complete analysis:**
```bash
python main.py
```

**4.2 What happens during execution:**
1. **Data Loading**: Reads metadata and validates file paths
2. **Feature Extraction**: Extracts Single-ROI and Motor-Circuit features
3. **Multi-Scenario Testing**: Tests 6 different feature engineering approaches
4. **Cross-Validation**: Performs center-based 5-fold cross-validation
5. **Statistical Analysis**: Compares feature types with significance testing
6. **Visualization Generation**: Creates interactive chord diagrams and tables

#### Step 5: Interpret Results

**5.1 Generated files:**
```
📁 outputs/
├── 📄 summary_metrics_table.csv                               # Main comparison table
├── 📄 detailed_classifier_metrics.xlsx                        # Detailed per-classifier results
└── 📄 comprehensive_motor_vs_single_chord_comparison.html     # Interactive visualization
```

**5.2 Key outputs to examine:**
- **CSV Table**: Shows Motor-Circuit vs Single-ROI performance across scenarios
- **Excel File**: Detailed breakdown by classifier and feature type
- **HTML Chord Diagram**: Interactive visualization of feature consistency

**5.3 Understanding results:**
- **Win Rate**: Percentage of cases where Motor-Circuit features outperform Single-ROI
- **Statistical Significance**: p-values from paired comparisons
- **Effect Sizes**: Cohen's d values indicating practical significance
- **Best Scenario**: Optimal feature engineering configuration

#### Step 6: Advanced Usage

**6.1 Customize analysis parameters:**
```python
# In config.py, modify:
SAMPLE_SIZE = 65          # Change sample size
N_SPLITS = 5              # Modify cross-validation folds
SCENARIOS = {...}         # Add/modify feature engineering scenarios
```

**6.2 Add new classifiers:**
```python
# In config.py, add to CLASSIFIERS dict:
'Your_Classifier': YourClassifier(param1=value1, random_state=42)
```


## 📚 Citation

If you use this code or methodology in your research, please cite:

Hosseini MS, Aghamiri SMR, Panahi M. Cross-regional radiomics: a novel framework for relationship-based feature extraction with validation in Parkinson's disease motor subtyping. BioData Mining. 2025. DOI: 10.1186/s13040-025-00483-4
```bibtex
@article{hosseini2025crossregional,
  title={Cross-regional radiomics: a novel framework for relationship-based feature extraction with validation in Parkinson's disease motor subtyping},
  author={Hosseini, Mahboube Sadat and Aghamiri, Seyyed Mahmoud Reza and Panahi, Mehdi},
  journal={BioData Mining},
  year={2025},
  publisher={Springer},
  doi={10.1186/s13040-025-00483-4}
}


## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.


