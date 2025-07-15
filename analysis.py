"""
Main analysis class for Parkinson's Disease Subtype Classification
Contains the PaperAnalysis class that orchestrates the entire analysis pipeline.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, 
                            f1_score, roc_curve, precision_recall_curve, auc)
from sklearn.pipeline import Pipeline

from config import *
from utils import set_all_seeds, prepare_data
from cross_validation import CenterBasedSplitter
from datasets import SingleROIDataset, MotorCircuitDataset
from preprocessing import create_leakage_free_pipeline
from feature_importance import FeatureImportanceAnalyzer


class PaperAnalysis:
    """Paper Analysis with enhanced feature importance methods"""
   
    def __init__(self, metadata_paths, base_paths, roi_list):
        self.td_metadata_path = metadata_paths['td']
        self.pigd_metadata_path = metadata_paths['pigd']
        self.td_base_path = base_paths['td']
        self.pigd_base_path = base_paths['pigd']
        self.roi_list = roi_list
        self.sample_size = SAMPLE_SIZE
       
        # Define core classifiers for paper (6 classifiers)
        self.classifiers = {
            'SVM_Poly': SVC(**CLASSIFIERS_CONFIG['SVM_Poly']),
            'Random_Forest': RandomForestClassifier(**CLASSIFIERS_CONFIG['Random_Forest']),
            'Extra_Trees': ExtraTreesClassifier(**CLASSIFIERS_CONFIG['Extra_Trees']),
            'Logistic_Regression': LogisticRegression(**CLASSIFIERS_CONFIG['Logistic_Regression']),
            'Naive_Bayes': GaussianNB(**CLASSIFIERS_CONFIG['Naive_Bayes']),
            'MLP': MLPClassifier(**CLASSIFIERS_CONFIG['MLP'])
        }
       
        # ALL classifiers will have feature importance analysis
        self.feature_importance_classifiers = list(self.classifiers.keys())
       
        # Define 6 scenarios
        self.scenarios = SCENARIOS
        
        # Initialize feature importance analyzer
        self.importance_analyzer = FeatureImportanceAnalyzer()
       
    def load_and_prepare_data(self):
        """Load and prepare dataset with sample size 65"""
        print("Loading and preparing data...")
       
        set_all_seeds(RANDOM_SEED)
        
        self.data_paths, self.scanner_encoder, self.center_encoder = prepare_data(
            self.td_metadata_path, self.pigd_metadata_path, 
            self.td_base_path, self.pigd_base_path
        )
       
        # Sample to exactly 65 patients
        if len(self.data_paths) > self.sample_size:
            # Use stratified sampling by center to maintain distribution
            center_counts = defaultdict(int)
            for patient in self.data_paths:
                center_counts[patient['center_name']] += 1
           
            sampling_ratio = self.sample_size / len(self.data_paths)
           
            sampled_indices = []
            for center, count in center_counts.items():
                center_indices = [i for i, p in enumerate(self.data_paths) if p['center_name'] == center]
                n_sample = max(1, int(count * sampling_ratio))
                np.random.seed(RANDOM_SEED)
                if n_sample < len(center_indices):
                    selected = np.random.choice(center_indices, n_sample, replace=False)
                    sampled_indices.extend(selected)
                else:
                    sampled_indices.extend(center_indices)
           
            # Ensure we don't exceed sample_size
            if len(sampled_indices) > self.sample_size:
                np.random.seed(RANDOM_SEED)
                sampled_indices = np.random.choice(sampled_indices, self.sample_size, replace=False)
           
            self.data_paths = [self.data_paths[i] for i in sorted(sampled_indices)]
       
        self.labels = np.array([d['label'] for d in self.data_paths])
       
    def extract_features(self):
        """Extract both Single-ROI-Features and Motor-Circuit-Features"""

        print("Extracting Single-ROI features...")
        
        # Extract Single-ROI-Features
        self.single_roi_dataset = SingleROIDataset(self.data_paths, self.roi_list, extract_features=True, debug=False)
        self.single_roi_features = self.single_roi_dataset.features
        self.single_roi_feature_names = self.single_roi_dataset.get_feature_names()

        print("Extracting Motor-Circuit features...")
        # Extract Motor-Circuit-Features
        self.motor_circuit_dataset = MotorCircuitDataset(self.data_paths, self.roi_list, extract_features=True, debug=False)
        self.motor_circuit_features = self.motor_circuit_dataset.features
        self.motor_circuit_feature_names = self.motor_circuit_dataset.get_feature_names()
       
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate all metrics for paper"""
        metrics = {}
       
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['sensitivity'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['auc'] = roc_auc_score(y_true, y_prob)
       
        return metrics
   
    def get_processed_feature_names(self, pipeline, original_feature_names, scenario_config):
        """Get feature names after all preprocessing steps"""
        current_names = original_feature_names.copy()
       
        # Apply transformations step by step to track feature names
        if 'enhancer' in pipeline.named_steps and scenario_config.get('enhanced', False):
            current_names = pipeline.named_steps['enhancer'].get_feature_names(current_names)
       
        if 'variance_filter' in pipeline.named_steps:
            # Get valid features after variance filtering
            valid_features = pipeline.named_steps['variance_filter'].valid_features_
            current_names = [current_names[i] for i in range(len(current_names)) if i < len(valid_features) and valid_features[i]]
       
        if 'feature_selector' in pipeline.named_steps:
            # Get selected features
            selected_indices = pipeline.named_steps['feature_selector'].get_support(indices=True)
            current_names = [current_names[i] for i in selected_indices if i < len(current_names)]
       
        return current_names
   
    def cross_validate_scenario(self, scenario_name, scenario_config):
        """Cross-validation for one scenario with enhanced feature importance"""
        print(f"Testing {scenario_name}...")
       
        # Results storage
        results = {
            'single_roi': defaultdict(list),
            'motor_circuit': defaultdict(list),
            'fold_results': [],
            'overfitting_analysis': {'single_roi': {}, 'motor_circuit': {}},
            'feature_importances': {'single_roi': {}, 'motor_circuit': {}},
            'feature_importance_methods': {'single_roi': {}, 'motor_circuit': {}},
            'selected_features': {'single_roi': {}, 'motor_circuit': {}},
            'processed_feature_names': {'single_roi': {}, 'motor_circuit': {}}
        }
       
        # Center-based cross-validation
        center_splitter = CenterBasedSplitter(n_splits=N_SPLITS, random_state=RANDOM_SEED)
       
        fold_idx = 0
        for train_idx, test_idx in center_splitter.split(self.single_roi_features, self.labels, self.data_paths):
            fold_idx += 1
           
            fold_result = {'fold': fold_idx, 'classifiers': {}}
           
            for clf_name, clf in self.classifiers.items():
                try:
                    # Test Single-ROI-Features
                    single_roi_pipeline = Pipeline([
                        ('preprocessing', create_leakage_free_pipeline(scenario_config)),
                        ('classifier', clf.__class__(**clf.get_params()))
                    ])
                   
                    single_roi_pipeline.fit(self.single_roi_features[train_idx], self.labels[train_idx])
                   
                    # Get processed feature names for this fold and classifier
                    if clf_name not in results['processed_feature_names']['single_roi']:
                        results['processed_feature_names']['single_roi'][clf_name] = []
                   
                    processed_names = self.get_processed_feature_names(
                        single_roi_pipeline.named_steps['preprocessing'], 
                        self.single_roi_feature_names, 
                        scenario_config
                    )
                    results['processed_feature_names']['single_roi'][clf_name].append(processed_names)
                   
                    # Store selected features if feature selection was used
                    if 'feature_selector' in single_roi_pipeline.named_steps['preprocessing'].named_steps:
                        if clf_name not in results['selected_features']['single_roi']:
                            results['selected_features']['single_roi'][clf_name] = []
                        selected_indices = single_roi_pipeline.named_steps['preprocessing'].named_steps['feature_selector'].get_support(indices=True)
                        results['selected_features']['single_roi'][clf_name].append(selected_indices)
                   
                    # Training performance (for overfitting analysis)
                    single_roi_train_pred = single_roi_pipeline.predict(self.single_roi_features[train_idx])
                    single_roi_train_prob = single_roi_pipeline.predict_proba(self.single_roi_features[train_idx])[:, 1]
                    single_roi_train_metrics = self.calculate_metrics(
                        self.labels[train_idx], single_roi_train_pred, single_roi_train_prob
                    )
                   
                    # Test performance
                    single_roi_pred = single_roi_pipeline.predict(self.single_roi_features[test_idx])
                    single_roi_prob = single_roi_pipeline.predict_proba(self.single_roi_features[test_idx])[:, 1]
                    single_roi_metrics = self.calculate_metrics(
                        self.labels[test_idx], single_roi_pred, single_roi_prob
                    )
                   
                    # Overfitting analysis
                    if clf_name not in results['overfitting_analysis']['single_roi']:
                        results['overfitting_analysis']['single_roi'][clf_name] = {'train': [], 'test': []}
                    results['overfitting_analysis']['single_roi'][clf_name]['train'].append(single_roi_train_metrics['auc'])
                    results['overfitting_analysis']['single_roi'][clf_name]['test'].append(single_roi_metrics['auc'])
                   
                    # Enhanced Feature importance for ALL classifiers
                    if clf_name in self.feature_importance_classifiers:
                        # Get the transformed features for importance calculation
                        X_train_transformed = single_roi_pipeline.named_steps['preprocessing'].transform(self.single_roi_features[train_idx])
                       
                        importance, method_used = self.importance_analyzer.get_feature_importance_advanced(
                            single_roi_pipeline.named_steps['classifier'], 
                            X_train_transformed, 
                            self.labels[train_idx],
                            processed_names,
                            clf_name
                        )
                       
                        if clf_name not in results['feature_importances']['single_roi']:
                            results['feature_importances']['single_roi'][clf_name] = []
                            results['feature_importance_methods']['single_roi'][clf_name] = []
                       
                        results['feature_importances']['single_roi'][clf_name].append(importance)
                        results['feature_importance_methods']['single_roi'][clf_name].append(method_used)
                   
                    # Test Motor-Circuit-Features
                    motor_circuit_pipeline = Pipeline([
                        ('preprocessing', create_leakage_free_pipeline(scenario_config)),
                        ('classifier', clf.__class__(**clf.get_params()))
                    ])
                   
                    motor_circuit_pipeline.fit(self.motor_circuit_features[train_idx], self.labels[train_idx])
                   
                    # Get processed feature names for this fold and classifier
                    if clf_name not in results['processed_feature_names']['motor_circuit']:
                        results['processed_feature_names']['motor_circuit'][clf_name] = []
                   
                    processed_names_mc = self.get_processed_feature_names(
                        motor_circuit_pipeline.named_steps['preprocessing'], 
                        self.motor_circuit_feature_names, 
                        scenario_config
                    )
                    results['processed_feature_names']['motor_circuit'][clf_name].append(processed_names_mc)
                   
                    # Store selected features if feature selection was used
                    if 'feature_selector' in motor_circuit_pipeline.named_steps['preprocessing'].named_steps:
                        if clf_name not in results['selected_features']['motor_circuit']:
                            results['selected_features']['motor_circuit'][clf_name] = []
                        selected_indices = motor_circuit_pipeline.named_steps['preprocessing'].named_steps['feature_selector'].get_support(indices=True)
                        results['selected_features']['motor_circuit'][clf_name].append(selected_indices)
                   
                    # Training performance (for overfitting analysis)
                    motor_circuit_train_pred = motor_circuit_pipeline.predict(self.motor_circuit_features[train_idx])
                    motor_circuit_train_prob = motor_circuit_pipeline.predict_proba(self.motor_circuit_features[train_idx])[:, 1]
                    motor_circuit_train_metrics = self.calculate_metrics(
                        self.labels[train_idx], motor_circuit_train_pred, motor_circuit_train_prob
                    )
                   
                    # Test performance
                    motor_circuit_pred = motor_circuit_pipeline.predict(self.motor_circuit_features[test_idx])
                    motor_circuit_prob = motor_circuit_pipeline.predict_proba(self.motor_circuit_features[test_idx])[:, 1]
                    motor_circuit_metrics = self.calculate_metrics(
                        self.labels[test_idx], motor_circuit_pred, motor_circuit_prob
                    )
                   
                    # Overfitting analysis
                    if clf_name not in results['overfitting_analysis']['motor_circuit']:
                        results['overfitting_analysis']['motor_circuit'][clf_name] = {'train': [], 'test': []}
                    results['overfitting_analysis']['motor_circuit'][clf_name]['train'].append(motor_circuit_train_metrics['auc'])
                    results['overfitting_analysis']['motor_circuit'][clf_name]['test'].append(motor_circuit_metrics['auc'])
                   
                    # Enhanced Feature importance for ALL classifiers
                    if clf_name in self.feature_importance_classifiers:
                        # Get the transformed features for importance calculation
                        X_train_transformed_mc = motor_circuit_pipeline.named_steps['preprocessing'].transform(self.motor_circuit_features[train_idx])
                       
                        importance_mc, method_used_mc = self.importance_analyzer.get_feature_importance_advanced(
                            motor_circuit_pipeline.named_steps['classifier'], 
                            X_train_transformed_mc, 
                            self.labels[train_idx],
                            processed_names_mc,
                            clf_name
                        )
                       
                        if clf_name not in results['feature_importances']['motor_circuit']:
                            results['feature_importances']['motor_circuit'][clf_name] = []
                            results['feature_importance_methods']['motor_circuit'][clf_name] = []
                       
                        results['feature_importances']['motor_circuit'][clf_name].append(importance_mc)
                        results['feature_importance_methods']['motor_circuit'][clf_name].append(method_used_mc)
                   
                    # Store all metrics
                    for metric, value in single_roi_metrics.items():
                        results['single_roi'][f'{clf_name}_{metric}'].append(value)
                    
                    for metric, value in motor_circuit_metrics.items():
                        results['motor_circuit'][f'{clf_name}_{metric}'].append(value)
                   
                    # Store fold-specific results
                    fold_result['classifiers'][clf_name] = {
                        'single_roi': single_roi_metrics,
                        'motor_circuit': motor_circuit_metrics,
                        'single_roi_pred': single_roi_pred,
                        'single_roi_prob': single_roi_prob,
                        'motor_circuit_pred': motor_circuit_pred,
                        'motor_circuit_prob': motor_circuit_prob,
                        'y_true': self.labels[test_idx],
                        'single_roi_train_metrics': single_roi_train_metrics,
                        'motor_circuit_train_metrics': motor_circuit_train_metrics
                    }
                   
                except Exception as e:
                    continue
           
            results['fold_results'].append(fold_result)
       
        return results

    def run_complete_analysis(self):
        """Run complete enhanced analysis for paper"""
        try:
            print("Starting Enhanced Analysis...")
           
            # Load and prepare data
            self.load_and_prepare_data()
           
            # Extract features
            self.extract_features()
           
            # Run analysis for all scenarios
            all_results = {}
           
            for scenario_name, scenario_config in self.scenarios.items():
                print(f"Running scenario: {scenario_name}")
                results = self.cross_validate_scenario(scenario_name, scenario_config)
                all_results[scenario_name] = results
           
            return all_results
           
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
            return None