"""
Configuration file for Parkinson's Disease Subtype Classification
Contains all configuration parameters and constants used throughout the project.
"""

import os

# Data paths configuration
METADATA_PATHS = {
    'td': r'C:\Your\Path\To\Data\TD_metadata.xlsx',
    'pigd': r'C:\Your\Path\To\Data\PIGD_metadata.xlsx'
}

BASE_PATHS = {
    'td': r'C:\Your\Path\To\Data\TD',
    'pigd': r'C:\Your\Path\To\Data\PIGD'
}

# ROI list for motor circuit analysis
MOTOR_ROI = [
    'L_Pu.nii.gz', 'L_CN.nii.gz', 'L_Pa.nii.gz', 'L_SN.nii.gz',
    'R_Pu.nii.gz', 'R_CN.nii.gz', 'R_Pa.nii.gz', 'R_SN.nii.gz'
]

# Analysis parameters
SAMPLE_SIZE = 140
RANDOM_SEED = 42
N_SPLITS = 5

# Scenario configurations
SCENARIOS = {
    'Raw_Features': {'method': None, 'k_best': None, 'outlier_removal': False, 'enhanced': False},
    'Raw_Preprocessing': {'method': 'robust', 'k_best': None, 'outlier_removal': True, 'enhanced': False},
    'Enhanced_Preprocessing': {'method': 'robust', 'k_best': None, 'outlier_removal': True, 'enhanced': True},
    'Enhanced_Robust': {'method': 'robust', 'k_best': 15, 'outlier_removal': True, 'enhanced': True},
    'Enhanced_Standard': {'method': 'standard', 'k_best': 15, 'outlier_removal': True, 'enhanced': True},
    'Enhanced_MinMax': {'method': 'minmax', 'k_best': 15, 'outlier_removal': True, 'enhanced': True}
}

# Classifier configurations
CLASSIFIERS_CONFIG = {
    'SVM_Poly': {'kernel': 'poly', 'degree': 3, 'probability': True, 'class_weight': 'balanced', 'random_state': RANDOM_SEED},
    'Random_Forest': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': RANDOM_SEED},
    'Extra_Trees': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': RANDOM_SEED},
    'Logistic_Regression': {'class_weight': 'balanced', 'random_state': RANDOM_SEED, 'max_iter': 1000},
    'Naive_Bayes': {},
    'MLP': {'hidden_layer_sizes': (100,), 'random_state': RANDOM_SEED, 'max_iter': 500}
}

# Feature enhancement parameters
FEATURE_ENHANCEMENT_PARAMS = {
    'n_poly': 10,
    'n_interact': 6,
    'n_log': 10,
    'iqr_factor': 2.5,
    'variance_threshold': 1e-10
}

# Output configuration
OUTPUT_CONFIG = {
    'dpi': 600,
    'figure_format': 'png',
    'excel_engine': 'openpyxl'
}
