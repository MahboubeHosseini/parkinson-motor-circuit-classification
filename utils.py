"""
Utility functions for Parkinson's Disease Subtype Classification
Contains helper functions used across multiple modules.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_data(td_metadata_path: str, pigd_metadata_path: str, 
                td_base_path: str, pigd_base_path: str) -> Tuple[List[Dict], LabelEncoder, LabelEncoder]:
    """Prepare data paths and metadata"""
    
    # Load metadata
    if td_metadata_path.endswith('.csv'):
        td_df = pd.read_csv(td_metadata_path, delim_whitespace=True)
        pigd_df = pd.read_csv(pigd_metadata_path, delim_whitespace=True)
    else:
        td_df = pd.read_excel(td_metadata_path)
        pigd_df = pd.read_excel(pigd_metadata_path)
    
    # Clean scanner names
    if td_df['SCANNER_MODEL'].dtype == 'object':
        td_df['SCANNER_MODEL'] = td_df['SCANNER_MODEL'].astype(str).str.strip().str.replace('\r\n', '')
    else:
        td_df['SCANNER_MODEL'] = td_df['SCANNER_MODEL'].astype(str)
        
    if pigd_df['SCANNER_MODEL'].dtype == 'object':
        pigd_df['SCANNER_MODEL'] = pigd_df['SCANNER_MODEL'].astype(str).str.strip().str.replace('\r\n', '')
    else:
        pigd_df['SCANNER_MODEL'] = pigd_df['SCANNER_MODEL'].astype(str)
    
    # Clean CENTERs
    td_df['CENTER'] = td_df['CENTER'].astype(str).str.strip()
    pigd_df['CENTER'] = pigd_df['CENTER'].astype(str).str.strip()
    
    # Encode scanners
    all_scanners = list(td_df['SCANNER_MODEL'].unique()) + list(pigd_df['SCANNER_MODEL'].unique())
    scanner_encoder = LabelEncoder()
    scanner_encoder.fit(all_scanners)
    
    # Encode centers
    all_centers = list(td_df['CENTER'].unique()) + list(pigd_df['CENTER'].unique())
    center_encoder = LabelEncoder()
    center_encoder.fit(all_centers)
    
    data_paths = []
    
    # Add TD patients
    for _, row in td_df.iterrows():
        data_paths.append({
            'patient_id': str(int(row['SUBJECT'])),
            'base_path': td_base_path,
            'label': 0,  # TD = 0
            'scanner': scanner_encoder.transform([row['SCANNER_MODEL']])[0],
            'scanner_name': row['SCANNER_MODEL'],
            'center': center_encoder.transform([row['CENTER']])[0],
            'center_name': row['CENTER'],
            'age': row['AGE'],
            'sex': row['SEX']
        })
    
    # Add PIGD patients
    for _, row in pigd_df.iterrows():
        data_paths.append({
            'patient_id': str(int(row['SUBJECT'])),
            'base_path': pigd_base_path,
            'label': 1,  # PIGD = 1
            'scanner': scanner_encoder.transform([row['SCANNER_MODEL']])[0],
            'scanner_name': row['SCANNER_MODEL'],
            'center': center_encoder.transform([row['CENTER']])[0],
            'center_name': row['CENTER'],
            'age': row['AGE'],
            'sex': row['SEX']
        })
    
    return data_paths, scanner_encoder, center_encoder


def calculate_dca_net_benefit(y_true, y_prob, thresholds):
    """Calculate Decision Curve Analysis Net Benefit"""
    net_benefits = []
   
    for threshold in thresholds:
        if threshold == 0:
            net_benefit = 0
        elif threshold == 1:
            net_benefit = np.mean(y_true) - 1
        else:
            # Convert probabilities to decisions based on threshold
            y_pred = (y_prob >= threshold).astype(int)
           
            # Calculate confusion matrix components
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
           
            n_total = len(y_true)
           
            # Net benefit calculation
            net_benefit = (tp / n_total) - (fp / n_total) * (threshold / (1 - threshold))
       
        net_benefits.append(net_benefit)
   
    return np.array(net_benefits)


def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def categorize_brain_region(feature_name):
    """Categorize feature by primary brain region"""
    feature_upper = feature_name.upper()
   
    if 'PU' in feature_upper:
        return 'Putamen'
    elif 'SN' in feature_upper:
        return 'Substantia Nigra'
    elif 'CA' in feature_upper:
        return 'Caudate'
    elif 'PA' in feature_upper:
        return 'Pallidum'
    else:
        return 'Mixed/Other'


def categorize_feature_type(feature_name):
    """Categorize feature by type"""
    feature_upper = feature_name.upper()
   
    if '_CR' in feature_upper or 'RATIO' in feature_upper:
        return 'Cross-Regional'
    elif '_ASYM' in feature_upper or 'ASYMMETRY' in feature_upper:
        return 'Asymmetry'
    elif 'VOLUME' in feature_upper or '_VOL' in feature_upper:
        return 'Volume'
    elif 'ENTROPY' in feature_upper:
        return 'Entropy'
    elif 'MEDIAN' in feature_upper:
        return 'Statistics'
    elif 'CV' in feature_upper:
        return 'Variability'
    else:
        return 'Other'


def categorize_motor_circuit_feature(feature_name):
    """Categorize motor circuit features into families"""
    feature = feature_name.upper()
   
    # Cross-Regional Features (CR)
    if '_CR' in feature:
        return 'Cross-Regional (CR)'
   
    # Asymmetry Features (Asym)
    elif '_ASYM' in feature:
        return 'Asymmetry (Asym)'
   
    # Motor Circuit Volume Features (MC)
    elif '_MC' in feature:
        return 'Motor Circuit Volume (MC)'
   
    # Shape & Distribution Features
    elif any(x in feature for x in ['_SHAPE', '_DIST', '_VAR', 'ENTROPY', 'SKEW']):
        return 'Shape & Distribution'
   
    # Combined Motor Circuit Features
    elif 'COMB' in feature:
        return 'Combined Circuit'
   
    # Putamen-specific features
    elif 'PU' in feature and not any(x in feature for x in ['_CR', '_ASYM', '_MC']):
        return 'Putamen-Specific'
   
    # Substantia Nigra-specific features
    elif 'SN' in feature and not any(x in feature for x in ['_CR', '_ASYM', '_MC']):
        return 'Substantia Nigra-Specific'
   
    # Caudate-specific features
    elif 'CA' in feature and not any(x in feature for x in ['_CR', '_ASYM', '_MC']):
        return 'Caudate-Specific'
   
    # Pallidum-specific features
    elif 'PA' in feature and not any(x in feature for x in ['_CR', '_ASYM', '_MC']):
        return 'Pallidum-Specific'
   
    # Other/Unclassified
    else:
        return 'Other'