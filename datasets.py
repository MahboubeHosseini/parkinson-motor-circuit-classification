"""
Dataset classes for feature extraction from neuroimaging data
Contains SingleROIDataset and MotorCircuitDataset for different feature types.
"""

import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from scipy import stats
from typing import Dict, List


class SingleROIDataset(Dataset):
    """Dataset class for single ROI neuroimaging data"""
    
    def __init__(self, data_paths: List[Dict], roi_names: List[str], 
                transform=None, extract_features=True, debug=False):
        self.data_paths = data_paths
        self.roi_names = roi_names
        self.transform = transform
        self.extract_features = extract_features
        self.debug = debug
        
        # Create descriptive feature names for Single-ROI-Features with abbreviations
        self.feature_names = []
        roi_mapping = {
            'L_Pu.nii.gz': 'L_Pu',
            'R_Pu.nii.gz': 'R_Pu',
            'L_CN.nii.gz': 'L_Ca',
            'R_CN.nii.gz': 'R_Ca',
            'L_Pa.nii.gz': 'L_Pa',
            'R_Pa.nii.gz': 'R_Pa',
            'L_SN.nii.gz': 'L_SN',
            'R_SN.nii.gz': 'R_SN'
        }
        
        for roi in roi_names:
            roi_clean = roi_mapping.get(roi, roi.replace('.nii.gz', ''))
            for metric in ['Median', 'CV', 'Volume', 'Entropy', 'IQR', 'Skewness']:
                self.feature_names.append(f"{roi_clean}_{metric}")
        
        if extract_features:
            self.features = self._extract_all_features()
    
    def get_feature_names(self):
        """Return feature names"""
        return self.feature_names
        
    def _robust_outlier_removal(self, voxels):
        """Remove outliers using robust method"""
        if len(voxels) <= 5:
            return voxels
            
        median = np.median(voxels)
        mad = np.median(np.abs(voxels - median))
        
        if mad == 0:
            return voxels
            
        modified_z_scores = 0.6745 * (voxels - median) / mad
        mask = np.abs(modified_z_scores) < 3.5
        
        return voxels[mask] if mask.sum() > len(voxels) * 0.1 else voxels
    
    def _extract_roi_features(self, image_path: str, mask_path: str) -> np.ndarray:
        """Extract features from ROI"""
        try:
            if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                return np.array([0, 0, 0, 0, 0, 0])
            
            img = nib.load(image_path)
            mask = nib.load(mask_path)
            
            img_data = img.get_fdata()
            mask_data = mask.get_fdata()
            
            roi_voxels = img_data[mask_data > 0]
            
            if len(roi_voxels) == 0:
                return np.array([0, 0, 0, 0, 0, 0])
            
            filtered_voxels = self._robust_outlier_removal(roi_voxels)
            
            mean_val = np.mean(filtered_voxels)
            
            features = np.array([
                np.median(filtered_voxels),                                      # median
                np.std(filtered_voxels) / (mean_val + 1e-7),                   # cv
                len(roi_voxels),                                                # volume
                stats.entropy(np.histogram(filtered_voxels, bins=20)[0] + 1e-7), # entropy
                np.percentile(filtered_voxels, 75) - np.percentile(filtered_voxels, 25), # iqr
                stats.skew(filtered_voxels)                                     # skewness
            ])
            
            return features
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting ROI stats from {mask_path}: {e}")
            return np.array([0, 0, 0, 0, 0, 0])
    
    def _extract_patient_features(self, patient_data: Dict) -> np.ndarray:
        """Extract features for one patient"""
        patient_id = patient_data['patient_id']
        base_path = patient_data['base_path']
        
        # Paths for images
        t1_path = os.path.join(base_path, patient_id, 'T1', f'{patient_id}.nii.gz')
        
        all_features = []
        
        # Extract features from each ROI
        for roi in self.roi_names:
            # T1 ROI mask path 
            t1_mask_path = os.path.join(base_path, patient_id, 'T1', 'w_thrp', 'Subject', roi)
            
            # Extract T1 features    
            if os.path.exists(t1_path) and os.path.exists(t1_mask_path):
                t1_features = self._extract_roi_features(t1_path, t1_mask_path)
            else:
                t1_features = np.zeros(6)
                if self.debug:
                    print(f"Missing T1 files for {patient_id}, ROI {roi}")
            
            all_features.extend(t1_features)
        
        return np.array(all_features)
    
    def _extract_all_features(self) -> np.ndarray:
        """Extract features for all patients"""
        all_features = []
        
        for i, patient_data in enumerate(self.data_paths):
            if self.debug or i % 10 == 0:
                print(f"Processing patient {i+1}/{len(self.data_paths)}: {patient_data['patient_id']}")
            features = self._extract_patient_features(patient_data)
            all_features.append(features)
        
        feature_array = np.array(all_features)
        
        return feature_array


class MotorCircuitDataset(Dataset):
    """Motor Circuit Cross-Regional Features Dataset"""
    
    def __init__(self, data_paths, roi_names, transform=None, extract_features=True, debug=False):
        self.data_paths = data_paths
        self.roi_names = roi_names
        self.transform = transform
        self.extract_features = extract_features
        self.debug = debug
        
        # Motor circuit ROIs mapping
        self.motor_rois = {
            'putamen_l': 'L_Pu.nii.gz',
            'putamen_r': 'R_Pu.nii.gz', 
            'caudate_l': 'L_CN.nii.gz',
            'caudate_r': 'R_CN.nii.gz',
            'pallidum_l': 'L_Pa.nii.gz',
            'pallidum_r': 'R_Pa.nii.gz',
            'sn_l': 'L_SN.nii.gz',
            'sn_r': 'R_SN.nii.gz'
        }
        
        # Create detailed descriptive feature names for Motor-Circuit-Features with abbreviations
        self.feature_names = self._create_descriptive_feature_names()
        self.total_features = len(self.feature_names)
        
        if extract_features:
            self.features = self._extract_all_features()
    
    def get_feature_names(self):
        """Return feature names"""
        return self.feature_names
    
    def _create_descriptive_feature_names(self):
        """Create detailed descriptive feature names with abbreviations"""
        feature_names = []
        
        # Basic metrics for each region
        metrics = ['Median', 'CV', 'Volume', 'Entropy', 'IQR', 'Skewness']
        
        # 1. Putamen-SubstantiaNigra Cross-Regional Ratios (12 features)
        for side in ['L', 'R']:
            for metric in metrics:
                feature_names.append(f"Pu{side}_SN{side}_{metric}_CR")
        
        # 2. Caudate-Putamen Cross-Regional Ratios (12 features)
        for side in ['L', 'R']:
            for metric in metrics:
                feature_names.append(f"Ca{side}_Pu{side}_{metric}_CR")
        
        # 3. Motor Circuit Volume Relationships (5 features)
        feature_names.extend([
            "SNL_PuL_Vol_MC",
            "SNR_PuR_Vol_MC", 
            "CaL_SNL_Vol_MC",
            "CaR_SNR_Vol_MC",
            "PaL_PaR_Vol_Asym"
        ])
        
        # 4. Left-Right Asymmetry Indices (24 features)
        regions = ['Pu', 'SN', 'Ca', 'Pa']
        for region in regions:
            for metric in metrics:
                feature_names.append(f"{region}L_{region}R_{metric}_Asym")
        
        # 5. Combined Motor Circuit Ratios (2 features)
        feature_names.extend([
            "Pu_SN_CombCV_MC",
            "Ca_Pa_CombCV_MC"
        ])
        
        # 6. Motor Circuit Shape & Distribution Features (5 features)
        feature_names.extend([
            "PuL_SNL_EntRatio_Shape",
            "PuR_SNR_EntRatio_Shape",
            "PuL_SNL_SkewDiff_Dist",
            "PuR_SNR_SkewDiff_Dist",
            "Pu_SN_IQR_CombRatio_Var"
        ])
        
        return feature_names
    
    def _robust_outlier_removal(self, voxels):
        """Remove outliers using robust method"""
        if len(voxels) <= 5:
            return voxels
            
        median = np.median(voxels)
        mad = np.median(np.abs(voxels - median))
        
        if mad == 0:
            return voxels
            
        modified_z_scores = 0.6745 * (voxels - median) / mad
        mask = np.abs(modified_z_scores) < 3.5
        
        return voxels[mask] if mask.sum() > len(voxels) * 0.1 else voxels
    
    def _extract_roi_statistics(self, image_path, mask_path):
        """Extract statistics from ROI"""
        try:
            if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                return {'median': 0, 'cv': 0, 'volume': 0, 'entropy': 0, 'iqr': 0, 'skewness': 0}
            
            img = nib.load(image_path)
            mask = nib.load(mask_path)
            
            img_data = img.get_fdata()
            mask_data = mask.get_fdata()
            
            roi_voxels = img_data[mask_data > 0]
            
            if len(roi_voxels) == 0:
                return {'median': 0, 'cv': 0, 'volume': 0, 'entropy': 0, 'iqr': 0, 'skewness': 0}
            
            filtered_voxels = self._robust_outlier_removal(roi_voxels)
            
            mean_val = np.mean(filtered_voxels)
            
            return {
                'median': np.median(filtered_voxels),
                'cv': np.std(filtered_voxels) / (mean_val + 1e-7),
                'volume': len(roi_voxels),
                'entropy': stats.entropy(np.histogram(filtered_voxels, bins=20)[0] + 1e-7),
                'iqr': np.percentile(filtered_voxels, 75) - np.percentile(filtered_voxels, 25),
                'skewness': stats.skew(filtered_voxels)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting ROI stats from {mask_path}: {e}")
            return {'median': 0, 'cv': 0, 'volume': 0, 'entropy': 0, 'iqr': 0, 'skewness': 0}
    
    def _extract_motor_circuit_features(self, patient_data):
        """Extract Motor-Circuit-Features with descriptive names"""
        patient_id = patient_data['patient_id']
        base_path = patient_data['base_path']
        
        t1_path = os.path.join(base_path, patient_id, 'T1', f'{patient_id}.nii.gz')
        
        # Extract statistics from each ROI
        motor_stats = {}
        
        for region_name, roi_file in self.motor_rois.items():
            mask_path = os.path.join(base_path, patient_id, 'T1', 'w_thrp', 'Subject', roi_file)
            motor_stats[region_name] = self._extract_roi_statistics(t1_path, mask_path)
        
        # Calculate features in same order as feature_names
        features = []
        
        # 1. Putamen-SubstantiaNigra Cross-Regional Ratios (12 features)
        features.extend([
            motor_stats['putamen_l']['median'] / (motor_stats['sn_l']['median'] + 1e-7),
            motor_stats['putamen_l']['cv'] / (motor_stats['sn_l']['cv'] + 1e-7),
            motor_stats['putamen_l']['volume'] / (motor_stats['sn_l']['volume'] + 1e-7),
            motor_stats['putamen_l']['entropy'] / (motor_stats['sn_l']['entropy'] + 1e-7),
            motor_stats['putamen_l']['iqr'] / (motor_stats['sn_l']['iqr'] + 1e-7),
            motor_stats['putamen_l']['skewness'] / (motor_stats['sn_l']['skewness'] + 1e-7),
            
            motor_stats['putamen_r']['median'] / (motor_stats['sn_r']['median'] + 1e-7),
            motor_stats['putamen_r']['cv'] / (motor_stats['sn_r']['cv'] + 1e-7),
            motor_stats['putamen_r']['volume'] / (motor_stats['sn_r']['volume'] + 1e-7),
            motor_stats['putamen_r']['entropy'] / (motor_stats['sn_r']['entropy'] + 1e-7),
            motor_stats['putamen_r']['iqr'] / (motor_stats['sn_r']['iqr'] + 1e-7),
            motor_stats['putamen_r']['skewness'] / (motor_stats['sn_r']['skewness'] + 1e-7),
        ])
        
        # 2. Caudate-Putamen Cross-Regional Ratios (12 features)
        features.extend([
            motor_stats['caudate_l']['median'] / (motor_stats['putamen_l']['median'] + 1e-7),
            motor_stats['caudate_l']['cv'] / (motor_stats['putamen_l']['cv'] + 1e-7),
            motor_stats['caudate_l']['volume'] / (motor_stats['putamen_l']['volume'] + 1e-7),
            motor_stats['caudate_l']['entropy'] / (motor_stats['putamen_l']['entropy'] + 1e-7),
            motor_stats['caudate_l']['iqr'] / (motor_stats['putamen_l']['iqr'] + 1e-7),
            motor_stats['caudate_l']['skewness'] / (motor_stats['putamen_l']['skewness'] + 1e-7),
            
            motor_stats['caudate_r']['median'] / (motor_stats['putamen_r']['median'] + 1e-7),
            motor_stats['caudate_r']['cv'] / (motor_stats['putamen_r']['cv'] + 1e-7),
            motor_stats['caudate_r']['volume'] / (motor_stats['putamen_r']['volume'] + 1e-7),
            motor_stats['caudate_r']['entropy'] / (motor_stats['putamen_r']['entropy'] + 1e-7),
            motor_stats['caudate_r']['iqr'] / (motor_stats['putamen_r']['iqr'] + 1e-7),
            motor_stats['caudate_r']['skewness'] / (motor_stats['putamen_r']['skewness'] + 1e-7),
        ])
        
        # 3. Motor Circuit Volume Relationships (5 features)
        features.extend([
            min(5.0, (motor_stats['sn_l']['volume'] + 10) / (motor_stats['putamen_l']['volume'] + 10)),
            min(5.0, (motor_stats['sn_r']['volume'] + 10) / (motor_stats['putamen_r']['volume'] + 10)),
            min(5.0, (motor_stats['caudate_l']['volume'] + 10) / (motor_stats['sn_l']['volume'] + 10)),
            min(5.0, (motor_stats['caudate_r']['volume'] + 10) / (motor_stats['sn_r']['volume'] + 10)),
            min(5.0, (motor_stats['pallidum_l']['volume'] + 10) / (motor_stats['pallidum_r']['volume'] + 10))
        ])

        # 4. Left-Right Asymmetry Indices (24 features)
        # Putamen L/R asymmetry
        features.extend([
            min(5.0, motor_stats['putamen_l']['median'] / (motor_stats['putamen_r']['median'] + 1e-7)),
            min(5.0, motor_stats['putamen_l']['cv'] / (motor_stats['putamen_r']['cv'] + 1e-7)),
            min(5.0, motor_stats['putamen_l']['volume'] / (motor_stats['putamen_r']['volume'] + 1e-7)),
            min(5.0, motor_stats['putamen_l']['entropy'] / (motor_stats['putamen_r']['entropy'] + 1e-7)),
            min(5.0, motor_stats['putamen_l']['iqr'] / (motor_stats['putamen_r']['iqr'] + 1e-7)),
            min(5.0, motor_stats['putamen_l']['skewness'] / (motor_stats['putamen_r']['skewness'] + 1e-7)),
        ])

        # SubstantiaNigra L/R asymmetry
        features.extend([
            min(5.0, motor_stats['sn_l']['median'] / (motor_stats['sn_r']['median'] + 1e-7)),
            min(5.0, motor_stats['sn_l']['cv'] / (motor_stats['sn_r']['cv'] + 1e-7)),
            min(5.0, motor_stats['sn_l']['volume'] / (motor_stats['sn_r']['volume'] + 1e-7)),
            min(5.0, motor_stats['sn_l']['entropy'] / (motor_stats['sn_r']['entropy'] + 1e-7)),
            min(5.0, motor_stats['sn_l']['iqr'] / (motor_stats['sn_r']['iqr'] + 1e-7)),
            min(5.0, motor_stats['sn_l']['skewness'] / (motor_stats['sn_r']['skewness'] + 1e-7)),
        ])

        # Caudate L/R asymmetry
        features.extend([
            min(5.0, motor_stats['caudate_l']['median'] / (motor_stats['caudate_r']['median'] + 1e-7)),
            min(5.0, motor_stats['caudate_l']['cv'] / (motor_stats['caudate_r']['cv'] + 1e-7)),
            min(5.0, motor_stats['caudate_l']['volume'] / (motor_stats['caudate_r']['volume'] + 1e-7)),
            min(5.0, motor_stats['caudate_l']['entropy'] / (motor_stats['caudate_r']['entropy'] + 1e-7)),
            min(5.0, motor_stats['caudate_l']['iqr'] / (motor_stats['caudate_r']['iqr'] + 1e-7)),
            min(5.0, motor_stats['caudate_l']['skewness'] / (motor_stats['caudate_r']['skewness'] + 1e-7)),
        ])

        # Pallidum L/R asymmetry
        features.extend([
            min(5.0, motor_stats['pallidum_l']['median'] / (motor_stats['pallidum_r']['median'] + 1e-7)),
            min(5.0, motor_stats['pallidum_l']['cv'] / (motor_stats['pallidum_r']['cv'] + 1e-7)),
            min(5.0, motor_stats['pallidum_l']['volume'] / (motor_stats['pallidum_r']['volume'] + 1e-7)),
            min(5.0, motor_stats['pallidum_l']['entropy'] / (motor_stats['pallidum_r']['entropy'] + 1e-7)),
            min(5.0, motor_stats['pallidum_l']['iqr'] / (motor_stats['pallidum_r']['iqr'] + 1e-7)),
            min(5.0, motor_stats['pallidum_l']['skewness'] / (motor_stats['pallidum_r']['skewness'] + 1e-7)),
        ]) 

        # 5. Combined Motor Circuit ratios (2 features)
        features.extend([
            min(3.0, (motor_stats['putamen_l']['cv'] + motor_stats['putamen_r']['cv']) / 
            (motor_stats['sn_l']['cv'] + motor_stats['sn_r']['cv'] + 2.0)),
            min(3.0, (motor_stats['caudate_l']['cv'] + motor_stats['caudate_r']['cv']) / 
            (motor_stats['pallidum_l']['cv'] + motor_stats['pallidum_r']['cv'] + 2.0))
        ])

        # 6. Motor Circuit Shape & Distribution Features (5 features)
        features.extend([
            motor_stats['putamen_l']['entropy'] / (motor_stats['sn_l']['entropy'] + 1e-7),
            motor_stats['putamen_r']['entropy'] / (motor_stats['sn_r']['entropy'] + 1e-7),
            abs(motor_stats['putamen_l']['skewness'] - motor_stats['sn_l']['skewness']),
            abs(motor_stats['putamen_r']['skewness'] - motor_stats['sn_r']['skewness']),
            (motor_stats['putamen_l']['iqr'] + motor_stats['putamen_r']['iqr']) / 
            (motor_stats['sn_l']['iqr'] + motor_stats['sn_r']['iqr'] + 1e-7)
        ])

        # Handle NaN/inf values
        features = np.array([np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in features])

        return features

    def _extract_patient_features(self, patient_data):
        """Extract features for one patient"""
        return self._extract_motor_circuit_features(patient_data)

    def _extract_all_features(self):
        """Extract features for all patients"""
        all_features = []

        for i, patient_data in enumerate(self.data_paths):
            if self.debug or i % 10 == 0:
                print(f"Processing patient {i+1}/{len(self.data_paths)}: {patient_data['patient_id']}")
   
            features = self._extract_patient_features(patient_data)
            all_features.append(features)

        feature_array = np.array(all_features)
        return feature_array