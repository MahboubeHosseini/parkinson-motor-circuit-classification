"""
Visualization functions for the Parkinson's Disease analysis
Contains all plotting and chart generation functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from collections import defaultdict, Counter

from utils import (calculate_dca_net_benefit, calculate_jaccard_similarity, 
                  categorize_brain_region, categorize_feature_type, 
                  categorize_motor_circuit_feature)
from statistics import StatisticalComparison


def create_summary_metrics_table(all_results, classifiers):
    """Create summary metrics table with Motor-Circuit vs Single-ROI comparison"""
    # Metrics to include
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
   
    # Create table data
    table_data = []
   
    for scenario_name, results in all_results.items():
        # Calculate means and stds across all classifiers for Motor-Circuit
        motor_circuit_stats = {}
        for metric in metrics:
            all_values = []
            for clf_name in classifiers.keys():
                values = results['motor_circuit'].get(f'{clf_name}_{metric}', [])
                if values:
                    all_values.extend(values)
           
            if all_values:
                motor_circuit_stats[metric] = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values)
                }
            else:
                motor_circuit_stats[metric] = {'mean': 0, 'std': 0}
       
        # Calculate means and stds across all classifiers for Single-ROI
        single_roi_stats = {}
        for metric in metrics:
            all_values = []
            for clf_name in classifiers.keys():
                values = results['single_roi'].get(f'{clf_name}_{metric}', [])
                if values:
                    all_values.extend(values)
           
            if all_values:
                single_roi_stats[metric] = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values)
                }
            else:
                single_roi_stats[metric] = {'mean': 0, 'std': 0}
       
        # Statistical comparison
        statistical_comparison = {}
        for metric in metrics:
            motor_values = []
            single_values = []
            for clf_name in classifiers.keys():
                motor_vals = results['motor_circuit'].get(f'{clf_name}_{metric}', [])
                single_vals = results['single_roi'].get(f'{clf_name}_{metric}', [])
                if motor_vals and single_vals:
                    motor_values.extend(motor_vals)
                    single_values.extend(single_vals)
           
            if len(motor_values) >= 3 and len(single_values) >= 3:
                comparison = StatisticalComparison.paired_comparison(
                    single_values, motor_values, metric.upper()
                )
                statistical_comparison[metric] = comparison
            else:
                statistical_comparison[metric] = {'min_p_value': 1.0, 'significance': 'ns', 'effect_size': 0}
       
        # Add row to table
        row = {
            'Scenario': scenario_name.replace('_', ' '),
            'Motor_Circuit_Accuracy': f"{motor_circuit_stats['accuracy']['mean']:.3f}±{motor_circuit_stats['accuracy']['std']:.3f}",
            'Single_ROI_Accuracy': f"{single_roi_stats['accuracy']['mean']:.3f}±{single_roi_stats['accuracy']['std']:.3f}",
            'Motor_Circuit_AUC': f"{motor_circuit_stats['auc']['mean']:.3f}±{motor_circuit_stats['auc']['std']:.3f}",
            'Single_ROI_AUC': f"{single_roi_stats['auc']['mean']:.3f}±{single_roi_stats['auc']['std']:.3f}",
            'Motor_Circuit_Sensitivity': f"{motor_circuit_stats['sensitivity']['mean']:.3f}±{motor_circuit_stats['sensitivity']['std']:.3f}",
            'Single_ROI_Sensitivity': f"{single_roi_stats['sensitivity']['mean']:.3f}±{single_roi_stats['sensitivity']['std']:.3f}",
            'Motor_Circuit_Specificity': f"{motor_circuit_stats['specificity']['mean']:.3f}±{motor_circuit_stats['specificity']['std']:.3f}",
            'Single_ROI_Specificity': f"{single_roi_stats['specificity']['mean']:.3f}±{single_roi_stats['specificity']['std']:.3f}",
            'Motor_Circuit_F1': f"{motor_circuit_stats['f1']['mean']:.3f}±{motor_circuit_stats['f1']['std']:.3f}",
            'Single_ROI_F1': f"{single_roi_stats['f1']['mean']:.3f}±{single_roi_stats['f1']['std']:.3f}",
            'Statistical_Comparison': f"p={statistical_comparison['auc']['min_p_value']:.4f} {statistical_comparison['auc']['significance']}, d={statistical_comparison['auc']['effect_size']:.3f}"
        }
        table_data.append(row)
   
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    filename = f'summary_metrics_table.csv'
    df.to_csv(filename, index=False)
   
    print(f"Summary metrics table saved: {filename}")
    return df, filename

def create_detailed_classifier_tables(all_results, classifiers):
    """Create detailed tables for each classifier separately with mean±std format"""
   
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
   
    # Create Excel writer
    filename = f'detailed_classifier_metrics.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
       
        for clf_name in classifiers.keys():
            # Data for this classifier
            clf_data = []
           
            for scenario_name, results in all_results.items():
                # Motor-Circuit data
                motor_row = {'Scenario': scenario_name.replace('_', ' '), 'Feature_Type': 'Motor-Circuit'}
                for metric in metrics:
                    values = results['motor_circuit'].get(f'{clf_name}_{metric}', [])
                    if values:
                        motor_row[f'{metric.title()}'] = f"{np.mean(values):.3f}±{np.std(values):.3f}"
                    else:
                        motor_row[f'{metric.title()}'] = "N/A"
               
                # Add feature importance method
                if clf_name in results.get('feature_importance_methods', {}).get('motor_circuit', {}):
                    methods = results['feature_importance_methods']['motor_circuit'][clf_name]
                    motor_row['Importance_Method'] = methods[0] if methods else "N/A"
                else:
                    motor_row['Importance_Method'] = "N/A"
               
                # Overfitting analysis
                if clf_name in results['overfitting_analysis']['motor_circuit']:
                    train_aucs = results['overfitting_analysis']['motor_circuit'][clf_name]['train']
                    test_aucs = results['overfitting_analysis']['motor_circuit'][clf_name]['test']
                    if train_aucs and test_aucs:
                        overfitting = np.mean(train_aucs) - np.mean(test_aucs)
                        motor_row['Overfitting_Score'] = f"{overfitting:.3f}"
                    else:
                        motor_row['Overfitting_Score'] = "N/A"
                else:
                    motor_row['Overfitting_Score'] = "N/A"
               
                clf_data.append(motor_row)
               
                # Single-ROI data
                single_row = {'Scenario': scenario_name.replace('_', ' '), 'Feature_Type': 'Single-ROI'}
                for metric in metrics:
                    values = results['single_roi'].get(f'{clf_name}_{metric}', [])
                    if values:
                        single_row[f'{metric.title()}'] = f"{np.mean(values):.3f}±{np.std(values):.3f}"
                    else:
                        single_row[f'{metric.title()}'] = "N/A"
               
                # Add feature importance method
                if clf_name in results.get('feature_importance_methods', {}).get('single_roi', {}):
                    methods = results['feature_importance_methods']['single_roi'][clf_name]
                    single_row['Importance_Method'] = methods[0] if methods else "N/A"
                else:
                    single_row['Importance_Method'] = "N/A"
               
                # Overfitting analysis
                if clf_name in results['overfitting_analysis']['single_roi']:
                    train_aucs = results['overfitting_analysis']['single_roi'][clf_name]['train']
                    test_aucs = results['overfitting_analysis']['single_roi'][clf_name]['test']
                    if train_aucs and test_aucs:
                        overfitting = np.mean(train_aucs) - np.mean(test_aucs)
                        single_row['Overfitting_Score'] = f"{overfitting:.3f}"
                    else:
                        single_row['Overfitting_Score'] = "N/A"
                else:
                    single_row['Overfitting_Score'] = "N/A"
               
                clf_data.append(single_row)
           
            # Create DataFrame for this classifier
            clf_df = pd.DataFrame(clf_data)
           
            # Write to Excel sheet
            sheet_name = clf_name.replace(' ', '_')[:31]  # Excel sheet name limit
            clf_df.to_excel(writer, sheet_name=sheet_name, index=False)
   
    print(f"Detailed classifier tables saved: {filename}")
    return filename

def create_roc_curves(all_results, scenarios, labels):
    """Create ROC curves for Motor-Circuit and Single-ROI across all scenarios"""
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
   
    # Colors for scenarios
    scenario_colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
   
    # Motor-Circuit ROC curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['motor_circuit_prob'])
       
        if all_y_true and all_y_prob:
            fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
            auc_score = auc(fpr, tpr)
           
            ax1.plot(fpr, tpr, color=scenario_colors[idx], linewidth=2.5,
                    label=f'{scenario_name.replace("_", " ")} (AUC={auc_score:.3f})', alpha=0.8)
   
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=16)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=16)
    ax1.set_title('ROC Curves - Motor-Circuit-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    legend= ax1.legend(loc="lower right", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax1.grid(True, alpha=0.3)
   
    # Single-ROI ROC curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['single_roi_prob'])
       
        if all_y_true and all_y_prob:
            fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
            auc_score = auc(fpr, tpr)
           
            ax2.plot(fpr, tpr, color=scenario_colors[idx], linewidth=2.5,
                    label=f'{scenario_name.replace("_", " ")} (AUC={auc_score:.3f})', alpha=0.8)
   
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=16)
    ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=16)
    ax2.set_title('ROC Curves - Single-ROI-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    legend = ax2.legend(loc="lower right", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
    filename = f'roc_curves.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
   
    print(f"ROC curves saved: {filename}")
    return filename


def create_pr_curves(all_results, labels):
    """Create Precision-Recall curves for Motor-Circuit and Single-ROI across all scenarios"""
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
   
    # Colors for scenarios
    scenario_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
   
    # Motor-Circuit PR curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['motor_circuit_prob'])
       
        if all_y_true and all_y_prob:
            precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
            pr_auc = auc(recall, precision)
           
            ax1.plot(recall, precision, color=scenario_colors[idx], linewidth=2.5,
                    label=f'{scenario_name.replace("_", " ")} (PR-AUC={pr_auc:.3f})', alpha=0.8)
   
    baseline = np.sum(labels) / len(labels)
    ax1.axhline(y=baseline, color='k', linestyle='--', alpha=0.6, linewidth=2, label=f'Baseline ({baseline:.3f})')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Recall', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Precision', fontweight='bold', fontsize=16)
    ax1.set_title('Precision-Recall Curves - Motor-Circuit-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    legend = ax1.legend(loc="lower left", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax1.grid(True, alpha=0.3)
   
    # Single-ROI PR curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['single_roi_prob'])
       
        if all_y_true and all_y_prob:
            precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
            pr_auc = auc(recall, precision)
           
            ax2.plot(recall, precision, color=scenario_colors[idx], linewidth=2.5,
                    label=f'{scenario_name.replace("_", " ")} (PR-AUC={pr_auc:.3f})', alpha=0.8)
   
    ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.6, linewidth=2, label=f'Baseline ({baseline:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Precision', fontweight='bold', fontsize=16)
    ax2.set_title('Precision-Recall Curves - Single-ROI-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    legend = ax2.legend(loc="lower left", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
    filename = f'pr_curves.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
   
    print(f"PR curves saved: {filename}")
    return filename


def create_calibration_curves(all_results):
    """Create calibration curves for Motor-Circuit and Single-ROI across all scenarios"""
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
   
    # Colors for scenarios
    scenario_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
   
    # Motor-Circuit calibration curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['motor_circuit_prob'])
       
        if all_y_true and all_y_prob and len(set(all_y_true)) > 1:
            try:
                fraction_pos, mean_pred_value = calibration_curve(all_y_true, all_y_prob, n_bins=10)
               
                ax1.plot(mean_pred_value, fraction_pos, color=scenario_colors[idx], linewidth=2.5,
                        marker='o', markersize=6, label=f'{scenario_name.replace("_", " ")}', alpha=0.8)
            except:
                continue
   
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=16)
    ax1.set_title('Calibration Curves - Motor-Circuit-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    legend = ax1.legend(loc="lower right", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax1.grid(True, alpha=0.3)
   
    # Single-ROI calibration curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['single_roi_prob'])
       
        if all_y_true and all_y_prob and len(set(all_y_true)) > 1:
            try:
                fraction_pos, mean_pred_value = calibration_curve(all_y_true, all_y_prob, n_bins=10)
                
                ax2.plot(mean_pred_value, fraction_pos, color=scenario_colors[idx], linewidth=2.5,
                        marker='o', markersize=6, label=f'{scenario_name.replace("_", " ")}', alpha=0.8)
            except:
                continue
   
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=16)
    ax2.set_title('Calibration Curves - Single-ROI-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    legend = ax2.legend(loc="lower right", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
    filename = f'calibration_curves.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
   
    print(f"Calibration curves saved: {filename}")
    return filename


def create_dca_curves(all_results, labels):
    """Create Decision Curve Analysis for Motor-Circuit and Single-ROI across all scenarios"""
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
   
    # Colors for scenarios
    scenario_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
   
    # Threshold range for DCA
    thresholds = np.linspace(0, 1, 101)
   
    # Motor-Circuit DCA curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['motor_circuit_prob'])
       
        if all_y_true and all_y_prob:
            net_benefits = calculate_dca_net_benefit(np.array(all_y_true), np.array(all_y_prob), thresholds)
           
            ax1.plot(thresholds, net_benefits, color=scenario_colors[idx], linewidth=2.5,
                    label=f'{scenario_name.replace("_", " ")}', alpha=0.8)
   
    # Add reference lines
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.6, linewidth=2, label='Treat None')
   
    # Treat all line
    prevalence = np.mean(labels)
    treat_all_benefits = [prevalence - (1-prevalence) * t/(1-t) if t < 1 else prevalence-1 for t in thresholds]
    ax1.plot(thresholds, treat_all_benefits, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Treat All')
   
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([-0.1, max(0.5, prevalence)])
    ax1.set_xlabel('Threshold Probability', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Net Benefit', fontweight='bold', fontsize=16)
    ax1.set_title('Decision Curve Analysis - Motor-Circuit-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    legend = ax1.legend(loc="upper right", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax1.grid(True, alpha=0.3)
   
    # Single-ROI DCA curves
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        # Collect all predictions across folds and classifiers
        all_y_true = []
        all_y_prob = []
       
        for fold_result in results['fold_results']:
            for clf_name, clf_data in fold_result['classifiers'].items():
                all_y_true.extend(clf_data['y_true'])
                all_y_prob.extend(clf_data['single_roi_prob'])
       
        if all_y_true and all_y_prob:
            net_benefits = calculate_dca_net_benefit(np.array(all_y_true), np.array(all_y_prob), thresholds)
           
            ax2.plot(thresholds, net_benefits, color=scenario_colors[idx], linewidth=2.5,
                    label=f'{scenario_name.replace("_", " ")}', alpha=0.8)
   
    # Add reference lines
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.6, linewidth=2, label='Treat None')
    ax2.plot(thresholds, treat_all_benefits, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Treat All')
   
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([-0.1, max(0.5, prevalence)])
    ax2.set_xlabel('Threshold Probability', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Net Benefit', fontweight='bold', fontsize=16)
    ax2.set_title('Decision Curve Analysis - Single-ROI-Features\nAcross All Scenarios', fontweight='bold', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    legend = ax2.legend(loc="upper right", fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
    filename = f'dca_curves.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
   
    print(f"DCA curves saved: {filename}")
    return filename


def create_overfitting_heatmap(all_results, classifiers, scenarios):
    """Create overfitting heatmap: classifiers vs scenarios (AUC only)"""
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
   
    scenario_list = list(scenarios.keys())
    classifier_list = list(classifiers.keys())
   
    # Motor-Circuit overfitting matrix (classifiers × scenarios)
    motor_matrix = np.zeros((len(classifier_list), len(scenario_list)))
   
    for i, clf_name in enumerate(classifier_list):
        for j, scenario in enumerate(scenario_list):
            if scenario in all_results:
                if clf_name in all_results[scenario]['overfitting_analysis']['motor_circuit']:
                    train_aucs = all_results[scenario]['overfitting_analysis']['motor_circuit'][clf_name]['train']
                    test_aucs = all_results[scenario]['overfitting_analysis']['motor_circuit'][clf_name]['test']
                    if train_aucs and test_aucs:
                        overfitting = np.mean(train_aucs) - np.mean(test_aucs)
                        motor_matrix[i, j] = overfitting
   
    # Single-ROI overfitting matrix (classifiers × scenarios)
    single_matrix = np.zeros((len(classifier_list), len(scenario_list)))
   
    for i, clf_name in enumerate(classifier_list):
        for j, scenario in enumerate(scenario_list):
            if scenario in all_results:
                if clf_name in all_results[scenario]['overfitting_analysis']['single_roi']:
                    train_aucs = all_results[scenario]['overfitting_analysis']['single_roi'][clf_name]['train']
                    test_aucs = all_results[scenario]['overfitting_analysis']['single_roi'][clf_name]['test']
                    if train_aucs and test_aucs:
                        overfitting = np.mean(train_aucs) - np.mean(test_aucs)
                        single_matrix[i, j] = overfitting
   
    # Plot Motor-Circuit heatmap
    im1 = ax1.imshow(motor_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.2)
    ax1.set_title('Motor-Circuit-Features Overfitting Analysis\n(Train - Test AUC)', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(scenario_list)))
    ax1.set_xticklabels([s.replace('_', ' ') for s in scenario_list], rotation=45, ha='right', fontweight='bold', fontsize=14)
    ax1.set_yticks(range(len(classifier_list)))
    ax1.set_yticklabels([c.replace('_', ' ') for c in classifier_list], fontweight='bold', fontsize=14)
   
    # Add text annotations
    for i in range(len(classifier_list)):
        for j in range(len(scenario_list)):
            value = motor_matrix[i, j]
            color = 'white' if value > 0.1 else 'black'
            ax1.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color=color, fontsize=14, fontweight='bold')
   
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Overfitting Score', fontsize=14, fontweight='bold')
   
    # Plot Single-ROI heatmap
    im2 = ax2.imshow(single_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.2)
    ax2.set_title('Single-ROI-Features Overfitting Analysis\n(Train - Test AUC)', fontweight='bold', fontsize=14)
    ax2.set_xticks(range(len(scenario_list)))
    ax2.set_xticklabels([s.replace('_', ' ') for s in scenario_list], rotation=45, ha='right', fontweight='bold', fontsize=14)
    ax2.set_yticks(range(len(classifier_list)))
    ax2.set_yticklabels([c.replace('_', ' ') for c in classifier_list], fontweight='bold', fontsize=14)
   
    # Add text annotations
    for i in range(len(classifier_list)):
        for j in range(len(scenario_list)):
            value = single_matrix[i, j]
            color = 'white' if value > 0.1 else 'black'
            ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color=color, fontsize=14, fontweight='bold')
   
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.ax.tick_params(labelsize=14)  
    cbar2.set_label('Overfitting Score', fontsize=14, fontweight='bold')
   
    plt.tight_layout()
    filename = f'overfitting_heatmap_auc_classifiers_scenarios.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
   
    print(f"Overfitting heatmap saved: {filename}")
    return filename


def create_radar_charts(all_results, classifiers):
    """Create radar charts for Motor-Circuit and Single-ROI performance comparison across scenarios"""

    # Metrics to include
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    metric_labels = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'F1-Score']

    # Prepare data for radar charts
    scenario_names = list(all_results.keys())
    scenario_labels = [name.replace('_', ' ') for name in scenario_names]

    # Motor-Circuit data
    motor_circuit_data = []
    for scenario_name in scenario_names:
        results = all_results[scenario_name]
        scenario_values = []
    
        for metric in metrics:
            all_values = []
            for clf_name in classifiers.keys():
                values = results['motor_circuit'].get(f'{clf_name}_{metric}', [])
                if values:
                    all_values.extend(values)
        
            if all_values:
                scenario_values.append(np.mean(all_values))
            else:
                scenario_values.append(0)
    
        motor_circuit_data.append(scenario_values)

    # Single-ROI data
    single_roi_data = []
    for scenario_name in scenario_names:
        results = all_results[scenario_name]
        scenario_values = []
    
        for metric in metrics:
            all_values = []
            for clf_name in classifiers.keys():
                values = results['single_roi'].get(f'{clf_name}_{metric}', [])
                if values:
                    all_values.extend(values)
        
            if all_values:
                scenario_values.append(np.mean(all_values))
            else:
                scenario_values.append(0)
     
        single_roi_data.append(scenario_values)

    # Create radar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(metrics)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Colors for scenarios
    colors = ["#10f610", "#7a1bd3", '#8c564b', "#1291ec", '#ff7f0e', '#d62728']

    # Motor-Circuit radar chart
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)

    # Add labels
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metric_labels, fontweight='bold', fontsize=14)

    # Set y-axis limits
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot data for each scenario
    for i, (scenario_label, values) in enumerate(zip(scenario_labels, motor_circuit_data)):
        values += values[:1]  # Complete the circle
        ax1.plot(angles, values, 'o-', linewidth=2.5, label=scenario_label, 
                color=colors[i % len(colors)], markersize=6)
        ax1.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])

    ax1.set_title('Motor-Circuit Features Performance\nAcross All Scenarios', 
                 fontweight='bold', fontsize=16, pad=30)
    legend = ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Single-ROI radar chart
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)

    # Add labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels, fontweight='bold', fontsize=14)

    # Set y-axis limits
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot data for each scenario
    for i, (scenario_label, values) in enumerate(zip(scenario_labels, single_roi_data)):
        values += values[:1]  # Complete the circle
        ax2.plot(angles, values, 'o-', linewidth=2.5, label=scenario_label, 
                color=colors[i % len(colors)], markersize=6)
        ax2.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])

    ax2.set_title('Single-ROI Features Performance\nAcross All Scenarios', 
                 fontweight='bold', fontsize=16, pad=30)
    legend = ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()

    # Save the plot
    filename = 'performance_radar_charts.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


    print(f"Performance radar charts saved: {filename}")
    return filename


def create_combined_radar_chart(all_results, classifiers):
    """Create a combined radar chart showing Motor-Circuit vs Single-ROI for best scenario"""

    # Metrics to include
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    metric_labels = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'F1-Score']

    # Find best performing scenario for each feature type
    best_motor_scenario = None
    best_single_scenario = None
    best_motor_score = 0
    best_single_score = 0

    for scenario_name, results in all_results.items():
        # Motor-Circuit average
        motor_values = []
        for metric in metrics:
            all_values = []
            for clf_name in classifiers.keys():
                values = results['motor_circuit'].get(f'{clf_name}_{metric}', [])
                if values:
                    all_values.extend(values)
            if all_values:
                motor_values.append(np.mean(all_values))
            else:
                motor_values.append(0)
    
        motor_avg = np.mean(motor_values)
        if motor_avg > best_motor_score:
            best_motor_score = motor_avg
            best_motor_scenario = scenario_name
    
        # Single-ROI average
        single_values = []
        for metric in metrics:
            all_values = []
            for clf_name in classifiers.keys():
                values = results['single_roi'].get(f'{clf_name}_{metric}', [])
                if values:
                    all_values.extend(values)
            if all_values:
                single_values.append(np.mean(all_values))
            else:
                single_values.append(0)
     
        single_avg = np.mean(single_values)
        if single_avg > best_single_score:
            best_single_score = single_avg
            best_single_scenario = scenario_name

    # Extract data for best scenarios
    motor_best_data = []
    single_best_data = []

    # Motor-Circuit best scenario data
    results = all_results[best_motor_scenario]
    for metric in metrics:
        all_values = []
        for clf_name in classifiers.keys():
            values = results['motor_circuit'].get(f'{clf_name}_{metric}', [])
            if values:
                all_values.extend(values)
        if all_values:
            motor_best_data.append(np.mean(all_values))
        else:
            motor_best_data.append(0)

    # Single-ROI best scenario data
    results = all_results[best_single_scenario]
    for metric in metrics:
        all_values = []
        for clf_name in classifiers.keys():
            values = results['single_roi'].get(f'{clf_name}_{metric}', [])
            if values:
                all_values.extend(values)
        if all_values:
            single_best_data.append(np.mean(all_values))
        else:
            single_best_data.append(0)

    # Create combined radar chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(metrics)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Complete the circle for data
    motor_best_data += motor_best_data[:1]
    single_best_data += single_best_data[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontweight='bold', fontsize=18)

    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontweight='bold', fontsize=18)
    ax.grid(True, alpha=0.3)

    # Plot Motor-Circuit data
    ax.plot(angles, motor_best_data, 'o-', linewidth=3, 
           label=f'Motor-Circuit ({best_motor_scenario.replace("_", " ")})', 
           color='#E74C3C', markersize=8)
    ax.fill(angles, motor_best_data, alpha=0.15, color='#E74C3C')

    # Plot Single-ROI data
    ax.plot(angles, single_best_data, 'o-', linewidth=3, 
           label=f'Single-ROI ({best_single_scenario.replace("_", " ")})', 
           color='#3498DB', markersize=8)
    ax.fill(angles, single_best_data, alpha=0.15, color='#3498DB')

    # Add value labels on the plot
    for angle, motor_val, single_val, metric in zip(angles[:-1], motor_best_data[:-1], single_best_data[:-1], metric_labels):
        # Motor-Circuit values
        ax.text(angle, motor_val + 0.05, f'{motor_val:.3f}', 
               ha='center', va='center', fontweight='bold', fontsize=18, 
               color='#E74C3C', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
        # Single-ROI values
        ax.text(angle, single_val - 0.05, f'{single_val:.3f}', 
               ha='center', va='center', fontweight='bold', fontsize=18, 
               color='#3498DB', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_title('Best Performance Comparison:\nMotor-Circuit vs Single-ROI Features', 
                fontweight='bold', fontsize=18, pad=30)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=18)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()

    # Save the plot
    filename = 'combined_radar_chart_best_scenarios.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


    print(f"Combined radar chart saved: {filename}")
    return filename


def create_motor_circuit_top_features_analysis(all_results, classifiers, scenarios, save_path=None):
    """Create analysis for Top Motor-Circuit Features with FULL feature names displayed"""
   
    # Collect all feature importance data
    all_feature_occurrences = []
    all_feature_importance_values = defaultdict(list)
    total_combinations = 0
    total_all_features = 0
   
    for scenario_name, results in all_results.items():
        if 'feature_importances' not in results or 'motor_circuit' not in results['feature_importances']:
            continue
           
        for clf_name in classifiers.keys():
            if clf_name not in results['feature_importances']['motor_circuit']:
                continue
               
            importances_list = results['feature_importances']['motor_circuit'][clf_name]
            feature_names_list = results['processed_feature_names']['motor_circuit'].get(clf_name, [])
           
            if not importances_list or not feature_names_list:
                continue
           
            avg_importance = np.mean(importances_list, axis=0)
            feature_names = feature_names_list[0] if feature_names_list else []
           
            if len(avg_importance) == 0 or len(feature_names) == 0:
                continue
           
            n_top = min(15, len(avg_importance))
            top_indices = np.argsort(avg_importance)[-n_top:]
           
            total_combinations += 1
            total_all_features += n_top
           
            # Record each top feature with its exact name
            for idx in top_indices:
                if int(idx) < len(feature_names):
                    feature_name = feature_names[int(idx)]
                    importance_value = avg_importance[int(idx)]
                   
                    # Use exact feature name (not cleaned)
                    all_feature_occurrences.append(feature_name)
                    all_feature_importance_values[feature_name].append(importance_value)
   
    # Calculate occurrence frequency for each feature
    feature_counts = Counter(all_feature_occurrences)
   
    # Calculate occurrence percentage
    feature_occurrence_data = []
    for feature_name, count in feature_counts.items():
        occurrence_percentage = (count / total_all_features) * 100  
        avg_importance = np.mean(all_feature_importance_values[feature_name])
       
        feature_occurrence_data.append({
            'feature': feature_name,
            'count': count,
            'occurrence_percentage': occurrence_percentage,
            'avg_importance': avg_importance,
            'region': categorize_brain_region(feature_name),
            'feature_type': categorize_feature_type(feature_name)
        })
   
    # Sort by occurrence percentage
    feature_occurrence_data.sort(key=lambda x: x['occurrence_percentage'], reverse=True)
   
    # Get top features (show more - up to 30)
    top_features = feature_occurrence_data[:30]
   
    # Create larger figure for full names
    fig, ax = plt.subplots(figsize=(22, 18))  
   
    # Prepare data
    feature_names = [f['feature'] for f in top_features]
    occurrences = [f['occurrence_percentage'] for f in top_features]
    counts = [f['count'] for f in top_features]
   
    # Enhanced colors based on feature type
    colors = []
    for f in top_features:
        feature_name = f['feature']
        if '_CR' in feature_name.upper():
            if 'PU' in feature_name.upper() and 'SN' in feature_name.upper():
                colors.append('#E74C3C')  # Red for Pu-SN cross
            elif 'CA' in feature_name.upper() and 'PU' in feature_name.upper():
                colors.append('#3498DB')  # Blue for Ca-Pu cross
            elif 'CA' in feature_name.upper() and 'SN' in feature_name.upper():
                colors.append('#2ECC71')  # Green for Ca-SN cross
            else:
                colors.append('#9B59B6')  # Purple for other cross
        elif '_ASYM' in feature_name.upper():
            colors.append('#1ABC9C')  # Teal for asymmetry
        elif '_MC' in feature_name.upper():
            colors.append('#95A5A6')  # Gray for motor circuit volume
        elif any(x in feature_name.upper() for x in ['SHAPE', 'DIST', 'VAR', 'ENTROPY', 'SKEW']):
            colors.append('#E67E22')  # Orange for shape/distribution
        elif 'COMB' in feature_name.upper():
            colors.append('#8E44AD')  # Purple for combined features
        else:
            colors.append('#F39C12')  # Yellow for others
   
    # Create horizontal bar plot
    y_pos = np.arange(len(feature_names))
    bars = ax.barh(y_pos, occurrences, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6)
   
    # FULL FEATURE NAMES (no shortening!) with bold font
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=20, fontweight='bold')  
    ax.set_xlabel('Occurrence Frequency (% of All Selected Features)', fontweight='bold', fontsize=20)
    ax.set_title(f'Top 30 Most Frequent Motor-Circuit Features - COMPLETE NAMES\n'
                 f'(From {total_all_features} Total Selections Across All Scenarios & Classifiers)', 
                fontweight='bold', fontsize=18)
   
    # Add count and percentage labels on bars with bold font
    for i, (bar, occ, count) in enumerate(zip(bars, occurrences, counts)):
        width = bar.get_width()
        ax.text(width + max(occurrences) * 0.008, bar.get_y() + bar.get_height()/2,
               f'{count}\n({occ:.1f}%)', ha='left', va='center', fontsize=14, fontweight='bold')
   
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(occurrences) * 1.15)
   
    # Invert y-axis to show highest first
    ax.invert_yaxis()
   
    # Enhanced legend for colors with bold font
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#E74C3C', alpha=0.85, label='Pu-SN Cross-Regional'),
        plt.Rectangle((0,0),1,1, facecolor='#3498DB', alpha=0.85, label='Ca-Pu Cross-Regional'),
        plt.Rectangle((0,0),1,1, facecolor='#2ECC71', alpha=0.85, label='Ca-SN Cross-Regional'),
        plt.Rectangle((0,0),1,1, facecolor='#9B59B6', alpha=0.85, label='Other Cross-Regional'),
        plt.Rectangle((0,0),1,1, facecolor='#1ABC9C', alpha=0.85, label='Asymmetry Features'),
        plt.Rectangle((0,0),1,1, facecolor='#95A5A6', alpha=0.85, label='Motor Circuit Volume'),
        plt.Rectangle((0,0),1,1, facecolor='#E67E22', alpha=0.85, label='Shape & Distribution'),
        plt.Rectangle((0,0),1,1, facecolor='#8E44AD', alpha=0.85, label='Combined Features'),
        plt.Rectangle((0,0),1,1, facecolor='#F39C12', alpha=0.85, label='Other')
    ]
   
    legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=20, 
             bbox_to_anchor=(0.99, 0.01), framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontweight('bold')
   
    plt.tight_layout()
   
    # Save with high quality
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
   
    return fig, ax


def create_motor_circuit_feature_family_analysis(all_results, classifiers, scenarios):
    """Create stacked bar chart showing feature family representation in top 15 features across scenarios"""
    
    def get_all_motor_circuit_families():
        """Get all possible feature families and their total counts"""
        # Based on the MotorCircuitDataset feature creation
        families = {
            'Cross-Regional (CR)': 24,  # 12 Pu-SN + 12 Ca-Pu cross-regional ratios
            'Asymmetry (Asym)': 24,     # 6 features × 4 regions (Pu, SN, Ca, Pa)
            'Motor Circuit Volume (MC)': 5,  # 5 volume relationship features
            'Shape & Distribution': 5,   # 5 shape & distribution features
            'Combined Circuit': 2        # 2 combined motor circuit ratios
        }
        return families
    
    def extract_top_features_with_families(results, scenario, top_n=15):
        """Extract top features and categorize them by families for all classifiers"""
        scenario_results = results.get(scenario, {})
        
        # Collect ALL top features from ALL classifiers (including duplicates)
        all_selected_features = []
        classifier_count = 0
        
        for clf_name in classifiers.keys():
            # Get feature importances for this classifier
            if clf_name not in scenario_results.get('feature_importances', {}).get('motor_circuit', {}):
                continue
                
            importances_list = scenario_results['feature_importances']['motor_circuit'][clf_name]
            feature_names_list = scenario_results['processed_feature_names']['motor_circuit'].get(clf_name, [])
            
            if not importances_list or not feature_names_list:
                continue
            
            # Average importance across folds
            avg_importance = np.mean(importances_list, axis=0)
            feature_names = feature_names_list[0] if feature_names_list else []
            
            if len(avg_importance) == 0 or len(feature_names) == 0:
                continue
            
            # Get top N features for this classifier
            n_features = min(top_n, len(avg_importance))
            if n_features > 0:
                top_indices = np.argsort(avg_importance)[-n_features:]
                
                # Add each top feature to the list (including duplicates across classifiers)
                for idx in top_indices:
                    if int(idx) < len(feature_names):
                        feature_name = feature_names[int(idx)]
                        all_selected_features.append(feature_name)
                
                classifier_count += 1
        
        # Now categorize all features (including duplicates)
        family_counts = {}
        for family in get_all_motor_circuit_families().keys():
            family_counts[family] = 0
        family_counts['Other'] = 0  # For unclassified features
        
        for feature_name in all_selected_features:
            family = categorize_motor_circuit_feature(feature_name)
            if family in family_counts:
                family_counts[family] += 1
            else:
                family_counts['Other'] += 1
        
        # Calculate percentages from total selected features
        total_selected = len(all_selected_features)  # This should be classifier_count * top_n
        
        family_percentages = {}
        for family, count in family_counts.items():
            if total_selected > 0:
                family_percentages[family] = (count / total_selected) * 100
            else:
                family_percentages[family] = 0
        
        return family_percentages, classifier_count, total_selected
    
    # Collect data for all scenarios
    scenarios_data = {}
    family_names = list(get_all_motor_circuit_families().keys())
    
    for scenario_name in scenarios.keys():
        family_percentages, n_classifiers, total_features = extract_top_features_with_families(
            all_results, scenario_name, top_n=15
        )
        
        if n_classifiers > 0:
            scenarios_data[scenario_name] = {
                'percentages': family_percentages,
                'n_classifiers': n_classifiers,
                'total_features': total_features
            }
    
    if not scenarios_data:
        return None
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Prepare data for stacking
    scenario_names = list(scenarios_data.keys())
    scenario_labels = [name.replace('_', ' ') for name in scenario_names]
    
    # Colors for each family (using a colorblind-friendly palette)
    colors = {
        'Cross-Regional (CR)': '#1f77b4',        # Blue
        'Asymmetry (Asym)': '#ff7f0e',           # Orange  
        'Motor Circuit Volume (MC)': '#2ca02c',   # Green
        'Shape & Distribution': '#d62728',        # Red
        'Combined Circuit': '#9467bd',            # Purple
        'Other': '#8c564b'                        # Brown for unclassified
    }
    
    # Get all possible families (including 'Other')
    all_families = list(get_all_motor_circuit_families().keys()) + ['Other']
    
    # Create stacked bars
    bottom_values = np.zeros(len(scenario_names))
    
    for family in all_families:
        if family in colors:  # Only plot families we have colors for
            percentages = []
            for scenario in scenario_names:
                pct = scenarios_data[scenario]['percentages'].get(family, 0)
                percentages.append(pct)
            
            # Only plot if there are non-zero values
            if any(p > 0 for p in percentages):
                bars = ax.bar(scenario_labels, percentages, bottom=bottom_values,
                            label=family, color=colors[family], alpha=0.8,
                            edgecolor='white', linewidth=1)
                
                # Add percentage labels on bars (only if percentage > 3%) with bold font
                for i, (bar, pct) in enumerate(zip(bars, percentages)):
                    if pct > 3:  # Show label if percentage is significant
                        height = bar.get_height()
                        # Calculate actual count for label
                        total_features = scenarios_data[scenario_names[i]]['total_features']
                        count = int((pct/100) * total_features)
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bottom_values[i] + height/2,
                               f'{count}\n({pct:.1f}%)', ha='center', va='center',
                               fontweight='bold', fontsize=12, color='black')
                
                bottom_values += percentages
    
    # Customize the plot with bold fonts
    ax.set_xlabel('Scenarios', fontweight='bold', fontsize=16)
    ax.set_ylabel('Family Representation Percentage (%)', fontweight='bold', fontsize=16)
    ax.set_title('Motor-Circuit Feature Family Representation in Top 15 Features\n' +
                 'Average Across All Classifiers per Scenario', 
                 fontweight='bold', fontsize=16, pad=30)
    
    # Rotate x-axis labels for better readability with bold font
    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)
    
    # Add legend with bold font (FIXED)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis limits (should be exactly 100% now)
    ax.set_ylim(0, 105)
    
    # Add horizontal line at 100% for reference
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add total percentage labels on top of each bar with bold font
    for i, scenario in enumerate(scenario_names):
        total_pct = sum(scenarios_data[scenario]['percentages'].values())
        total_features = scenarios_data[scenario]['total_features']
        ax.text(i, total_pct + 2, f'{total_features} features\n(100%)',
               ha='center', va='bottom', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'motor_circuit_feature_family_stacked_bar.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    
    # Create detailed summary table
    summary_data = []
    for scenario_name in scenario_names:
        scenario_data = scenarios_data[scenario_name]
        row = {
            'Scenario': scenario_name.replace('_', ' '),
            'Classifiers_Analyzed': scenario_data['n_classifiers']
        }
        
        # Add each family percentage
        for family in all_families:
            pct = scenario_data['percentages'].get(family, 0)
            count = int((pct/100) * scenario_data['total_features']) if scenario_data['total_features'] > 0 else 0
            row[f'{family}_Count'] = count
            row[f'{family}_Percentage'] = f"{pct:.2f}%"
        
        # Add total (should always be the same as total_features)
        total_pct = sum(scenario_data['percentages'].values())
        row['Total_Features'] = scenario_data['total_features']
        row['Total_Percentage_Check'] = f"{total_pct:.2f}%"
        
        summary_data.append(row)
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_filename = 'motor_circuit_feature_family_summary.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    return {
        'plot_file': filename,
        'summary_file': summary_filename,
        'analysis_data': scenarios_data
    }


def create_individual_classifier_feature_importance_plots(all_results, classifiers, scenarios):
    """Create feature importance plots for each classifier separately with enhanced methods"""
   
    generated_files = []
   
    for clf_name in classifiers.keys():
        # Motor-Circuit Features for this classifier
        fig1, axes1 = plt.subplots(2, 3, figsize=(24, 16))
        axes1 = axes1.ravel()
       
        # Enhanced color mapping for Motor-Circuit features
        def get_motor_circuit_color_and_category(name):
            """Enhanced categorization for Motor-Circuit features"""
            if '_CR' in name:
                return '#9B59B6', 'CR'
            elif '_Asym' in name:
                return '#1ABC9C', 'Asym'
            elif '_MC' in name:
                return '#95A5A6', 'MC'
            elif '_Shape' in name or '_Dist' in name or '_Var' in name:
                return '#E67E22', 'Shp'
            elif 'Pu' in name:
                return '#E74C3C', 'Pu'
            elif 'SN' in name:
                return '#3498DB', 'SN'
            elif 'Ca' in name:
                return '#2ECC71', 'Ca'
            elif 'Pa' in name:
                return '#F39C12', 'Pa'
            else:
                return '#34495E', 'Gen'
       
        # Get the method used for this classifier
        method_used = "Unknown"
        for scenario_name, results in all_results.items():
            if clf_name in results.get('feature_importance_methods', {}).get('motor_circuit', {}):
                methods = results['feature_importance_methods']['motor_circuit'][clf_name]
                if methods:
                    method_used = methods[0]
                    break
       
        for idx, (scenario_name, scenario_config) in enumerate(scenarios.items()):
            ax = axes1[idx]
           
            # Get importance for this classifier
            if scenario_name in all_results:
                if clf_name in all_results[scenario_name]['feature_importances']['motor_circuit']:
                    importances = all_results[scenario_name]['feature_importances']['motor_circuit'][clf_name]
                    feature_names_list = all_results[scenario_name]['processed_feature_names']['motor_circuit'].get(clf_name, [])
                   
                    if importances and feature_names_list:
                        try:
                            # Average across folds for this classifier
                            avg_importance = np.mean(importances, axis=0)
                            std_importance = np.std(importances, axis=0)
                           
                            # Use first fold's feature names (should be consistent across folds)
                            current_feature_names = feature_names_list[0] if feature_names_list else []
                           
                            # Ensure we have valid data
                            if len(avg_importance) > 0 and len(current_feature_names) > 0:
                                # Get top 15 features
                                n_features = min(15, len(avg_importance))
                                if n_features > 0:
                                    top_indices = np.argsort(avg_importance)[-n_features:]
                                    top_importance = avg_importance[top_indices]
                                    top_std = std_importance[top_indices]
                                   
                                    # Get feature names and assign colors
                                    top_names = []
                                    colors = []
                                   
                                    for idx_val in top_indices:
                                        # Convert numpy scalar to int to avoid array comparison
                                        idx_int = int(idx_val)
                                       
                                        if idx_int < len(current_feature_names):
                                            name = current_feature_names[idx_int]
                                        else:
                                            name = f'Feature_{idx_int}'
                                       
                                        color, category = get_motor_circuit_color_and_category(name)
                                       
                                        # Create shortened feature name
                                        short_name = name
                                        if len(short_name) > 25:
                                            short_name = short_name[:22] + '...'
                                       
                                        top_names.append(f'[{category}] {short_name}')
                                        colors.append(color)
                                   
                                    # Create horizontal bar chart
                                    y_pos = np.arange(len(top_names))
                                    bars = ax.barh(y_pos, top_importance, xerr=top_std, 
                                                 color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                                   
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels(top_names, fontsize=8, fontweight='bold')
                                    ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                                    ax.set_xlabel('Importance', fontweight='bold', fontsize=10)
                                    ax.grid(True, alpha=0.3, axis='x')
                                   
                                    # Add importance values on bars with bold font
                                    if len(top_importance) > 0 and np.max(top_importance) > 0:
                                        for bar, imp in zip(bars, top_importance):
                                            width = bar.get_width()
                                            if width > 0:
                                                ax.text(width + max(top_importance) * 0.02, bar.get_y() + bar.get_height()/2,
                                                       f'{imp:.3f}', ha='left', va='center', fontsize=7, fontweight='bold')
                           
                            else:
                                ax.text(0.5, 0.5, 'No Valid Features\nAfter Processing', 
                                       ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                                ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                       
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Processing Error\n{str(e)[:20]}...', 
                                   ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                            ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                   
                    else:
                        ax.text(0.5, 0.5, 'No Feature\nImportance Available', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                        ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No Feature\nImportance Available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                    ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No Data\nAvailable', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
       
        # Title with bold font
        plt.suptitle(f'Motor-Circuit-Features Importance Analysis\n{clf_name.replace("_", " ")} Classifier ({method_used})', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
       
        motor_filename = f'motor_circuit_feature_importance_{clf_name}.png'
        plt.savefig(motor_filename, dpi=600, bbox_inches='tight')
        plt.close()
        generated_files.append(motor_filename)
       
        # Single-ROI Features for this classifier
        fig2, axes2 = plt.subplots(2, 3, figsize=(24, 16))
        axes2 = axes2.ravel()
       
        single_color_map = {
            'L_Pu': '#E74C3C', 'R_Pu': '#C0392B',
            'L_SN': '#3498DB', 'R_SN': '#2980B9',
            'L_Ca': '#2ECC71', 'R_Ca': '#27AE60',
            'L_Pa': '#F39C12', 'R_Pa': '#D68910'
        }

        # Get the method used for this classifier (Single-ROI)
        method_used_single = "Unknown"
        for scenario_name, results in all_results.items():
            if clf_name in results.get('feature_importance_methods', {}).get('single_roi', {}):
                methods = results['feature_importance_methods']['single_roi'][clf_name]
                if methods:
                    method_used_single = methods[0]
                    break
       
        for idx, (scenario_name, scenario_config) in enumerate(scenarios.items()):
            ax = axes2[idx]
           
            # Get importance for this classifier
            if scenario_name in all_results:
                if clf_name in all_results[scenario_name]['feature_importances']['single_roi']:
                    importances = all_results[scenario_name]['feature_importances']['single_roi'][clf_name]
                    feature_names_list = all_results[scenario_name]['processed_feature_names']['single_roi'].get(clf_name, [])
                   
                    if importances and feature_names_list:
                        try:
                            # Average across folds for this classifier
                            avg_importance = np.mean(importances, axis=0)
                            std_importance = np.std(importances, axis=0)
                           
                            # Use first fold's feature names (should be consistent across folds)
                            current_feature_names = feature_names_list[0] if feature_names_list else []
                           
                            # Ensure we have valid data
                            if len(avg_importance) > 0 and len(current_feature_names) > 0:
                                # Get top 15 features
                                n_features = min(15, len(avg_importance))
                                if n_features > 0:
                                    top_indices = np.argsort(avg_importance)[-n_features:]
                                    top_importance = avg_importance[top_indices]
                                    top_std = std_importance[top_indices]
                                   
                                    # Get feature names and assign colors
                                    top_names = []
                                    colors = []
                                   
                                    for idx_val in top_indices:
                                        # Convert numpy scalar to int to avoid array comparison
                                        idx_int = int(idx_val)
                                       
                                        if idx_int < len(current_feature_names):
                                            name = current_feature_names[idx_int]
                                        else:
                                            name = f'Feature_{idx_int}'
                                       
                                        # Categorize feature by ROI and assign color
                                        roi_found = False
                                        for roi_key, color in single_color_map.items():
                                            if roi_key in name:
                                                colors.append(color)
                                                # Extract metric
                                                parts = name.split('_')
                                                if len(parts) >= 2:
                                                    metric = parts[-1]
                                                    roi_short = roi_key
                                                    top_names.append(f'[{roi_short}] {metric}')
                                                else:
                                                    top_names.append(f'[{roi_key}] {name}')
                                                roi_found = True
                                                break
                                       
                                        if not roi_found:
                                            colors.append('#95A5A6')  # Default gray
                                            # Try to extract meaningful name
                                            clean_name = name.replace('_', ' ')
                                            if len(clean_name) > 20:
                                                clean_name = clean_name[:17] + '...'
                                            top_names.append(clean_name)
                                   
                                    # Create horizontal bar chart
                                    y_pos = np.arange(len(top_names))
                                    bars = ax.barh(y_pos, top_importance, xerr=top_std, 
                                                 color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                                   
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels(top_names, fontsize=8, fontweight='bold')
                                    ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                                    ax.set_xlabel('Importance', fontweight='bold', fontsize=10)
                                    ax.grid(True, alpha=0.3, axis='x')
                                   
                                    # Add importance values on bars with bold font
                                    if len(top_importance) > 0 and np.max(top_importance) > 0:
                                        for bar, imp in zip(bars, top_importance):
                                            width = bar.get_width()
                                            if width > 0:
                                                ax.text(width + max(top_importance) * 0.02, bar.get_y() + bar.get_height()/2,
                                                       f'{imp:.3f}', ha='left', va='center', fontsize=7, fontweight='bold')
                           
                            else:
                                ax.text(0.5, 0.5, 'No Valid Features\nAfter Processing', 
                                       ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                                ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                       
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Processing Error\n{str(e)[:20]}...', 
                                   ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                            ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                   
                    else:
                        ax.text(0.5, 0.5, 'No Feature\nImportance Available', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                        ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No Feature\nImportance Available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                    ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No Data\nAvailable', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
                ax.set_title(f'{scenario_name.replace("_", " ")}', fontweight='bold', fontsize=12)
       
        # Title with bold font
        plt.suptitle(f'Single-ROI-Features Importance Analysis\n{clf_name.replace("_", " ")} Classifier ({method_used_single})', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
       
        single_filename = f'single_roi_feature_importance_{clf_name}.png'
        plt.savefig(single_filename, dpi=600, bbox_inches='tight')
        plt.close()
        generated_files.append(single_filename)
   
    return generated_files


def create_comprehensive_consistency_analysis(all_results):
    """Create comprehensive consistency analysis with combined chord diagrams"""
   
    generated_files = []
   
    # Combined comprehensive chord diagram
    try:
        combined_fig, combined_file = create_combined_comprehensive_chord_diagrams(all_results)
        generated_files.append(combined_file)
        print(f"Combined chord diagram saved: {combined_file}")
    except Exception as e:
        print(f"Error creating combined comprehensive chord: {e}")
   
    return generated_files


def create_combined_comprehensive_chord_diagrams(all_results):
    """Create combined side-by-side chord diagrams for Motor-Circuit and Single-ROI"""
   
    # Extract feature sets
    motor_feature_sets = extract_top_features_by_combination(all_results, 'motor_circuit', top_n=15)
    single_feature_sets = extract_top_features_by_combination(all_results, 'single_roi', top_n=15)
   
    classifiers = ['SVM_Poly', 'Random_Forest', 'Extra_Trees', 'Logistic_Regression', 'Naive_Bayes', 'MLP']
    scenarios = ['Raw_Features', 'Raw_Preprocessing', 'Enhanced_Preprocessing', 'Enhanced_Robust', 'Enhanced_Standard', 'Enhanced_MinMax']
   
    # Create node combinations
    node_combinations = []
    for scenario in scenarios:
        for clf in classifiers:
            node_combinations.append((scenario, clf))
   
    # Calculate similarity matrices
    motor_similarity_matrix = calculate_similarity_matrix(motor_feature_sets, node_combinations)
    single_similarity_matrix = calculate_similarity_matrix(single_feature_sets, node_combinations)
   
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Motor-Circuit Features', 'Single-ROI Features'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        horizontal_spacing=0.05
    )
   
    # Calculate positions (same for both)
    x_pos, y_pos = calculate_node_positions(scenarios, classifiers)
   
    # Color scheme
    scenario_colors = [
        '#1f77b4',  # Raw Features - Blue
        '#ff7f0e',  # Raw Preprocessing - Orange
        '#2ca02c',  # Enhanced Preprocessing - Green
        '#d62728',  # Enhanced Robust - Red
        '#9467bd',  # Enhanced Standard - Purple
        '#8c564b'   # Enhanced MinMax - Brown
    ]
   
    node_colors = calculate_node_colors(scenarios, classifiers, scenario_colors)
   
    # Add legend traces
    add_legend_traces(fig, scenario_colors, scenarios)
   
    # Add connections and nodes for Motor-Circuit (left panel)
    add_chord_connections(fig, motor_similarity_matrix, x_pos, y_pos, row=1, col=1, connection_threshold=0.15)
    add_chord_nodes(fig, x_pos, y_pos, node_colors, node_combinations, motor_similarity_matrix, 
                    name_suffix="_motor", show_legend=False, row=1, col=1)
   
    # Add connections and nodes for Single-ROI (right panel)
    add_chord_connections(fig, single_similarity_matrix, x_pos, y_pos, row=1, col=2, connection_threshold=0.15)
    add_chord_nodes(fig, x_pos, y_pos, node_colors, node_combinations, single_similarity_matrix,
                    name_suffix="_single", show_legend=False, row=1, col=2)
   
    # Add scenario separators ONLY (no labels)
    add_scenario_separators_only(fig, scenarios, classifiers, row_col_pairs=[(1,1), (1,2)])
   
    # Enhanced layout
    fig.update_layout(
        title=dict(
            text="<b>Comprehensive Feature Consistency Analysis: Motor-Circuit vs Single-ROI</b><br>" +
                 "<span style='font-size:16px; font-weight:bold; color:black'>Jaccard Similarity of Top 15 Features (6 Scenarios × 6 Classifiers = 36 Nodes Each)</span>",
            x=0.5,
            y=0.97,
            font=dict(size=20, color='black', family='Arial', weight='bold')
        ),
        width=1400,
        height=800,
        showlegend=True,
        template="plotly_white",
        font=dict(size=16, family="Arial", color='black', weight='bold'),
       
        legend=dict(
            x=0.42,
            y=0.95,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12, family="Arial", color='black', weight='bold'),
            orientation="v",
            title=dict(
                text="<b>Legend</b>",
                font=dict(size=14, color="black", family="Arial", weight='bold')
            ),
            tracegroupgap=8,
            itemsizing="constant"
        ),
       
        modebar=dict(
            bgcolor="rgba(255,255,255,0.9)",
            color="black",
            activecolor="blue"
        ),
       
        margin=dict(l=20, r=20, t=80, b=20)
    )
   
    # Update subplot axis properties
    for i in range(1, 3):
        fig.update_xaxes(visible=False, range=[-1.4, 1.4], row=1, col=i)
        fig.update_yaxes(visible=False, range=[-1.4, 1.4], row=1, col=i)
   
    # Update subplot titles
    fig.update_annotations(
        font=dict(size=18, color="black", family="Arial", weight='bold'),
        y=0.88
    )
   
    # High-quality export configuration
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'motor_vs_single_chord_comparison_HD',
            'height': 800,
            'width': 1400,
            'scale': 4
        },
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'responsive': True
    }
   
    # Save main HTML file
    filename = 'comprehensive_motor_vs_single_chord_comparison.html'
    fig.write_html(filename, config=config, include_plotlyjs=True)
   
    return fig, filename


# Helper functions for chord diagrams
def extract_top_features_by_combination(all_results, feature_type='motor_circuit', top_n=15):
    """Extract top features for each classifier-scenario combination"""
   
    feature_sets = {}
   
    for scenario_name, results in all_results.items():
        feature_sets[scenario_name] = {}
       
        for clf_name in ['SVM_Poly', 'Random_Forest', 'Extra_Trees', 'Logistic_Regression', 'Naive_Bayes', 'MLP']:
           
            importances_list = results.get('feature_importances', {}).get(feature_type, {}).get(clf_name, [])
            feature_names_list = results.get('processed_feature_names', {}).get(feature_type, {}).get(clf_name, [])
           
            if not importances_list or not feature_names_list:
                continue
               
            # Average importance across folds
            avg_importance = np.mean(importances_list, axis=0)
            feature_names = feature_names_list[0] if feature_names_list else []
           
            if len(avg_importance) == 0 or len(feature_names) == 0:
                continue
               
            # Get top N features
            n_features = min(top_n, len(avg_importance))
            if n_features > 0:
                top_indices = np.argsort(avg_importance)[-n_features:]
                top_features = set()
               
                for idx in top_indices:
                    if int(idx) < len(feature_names):
                        feature_name = feature_names[int(idx)]
                        top_features.add(feature_name)
               
                feature_sets[scenario_name][clf_name] = top_features
   
    return feature_sets


def calculate_similarity_matrix(feature_sets, node_combinations):
    """Calculate similarity matrix for given feature sets"""
    n_nodes = len(node_combinations)
    similarity_matrix = np.zeros((n_nodes, n_nodes))
   
    for i, (scenario1, clf1) in enumerate(node_combinations):
        for j, (scenario2, clf2) in enumerate(node_combinations):
            if i != j:
                set1 = set()
                set2 = set()
               
                if scenario1 in feature_sets and clf1 in feature_sets[scenario1]:
                    set1 = feature_sets[scenario1][clf1]
               
                if scenario2 in feature_sets and clf2 in feature_sets[scenario2]:
                    set2 = feature_sets[scenario2][clf2]
               
                if set1 and set2:
                    similarity_matrix[i, j] = calculate_jaccard_similarity(set1, set2)
                else:
                    similarity_matrix[i, j] = 0.0
            else:
                similarity_matrix[i, j] = 1.0
   
    return similarity_matrix


def calculate_node_positions(scenarios, classifiers):
    """Calculate node positions in circular layout"""
    angles = []
    scenario_angle_step = 2 * np.pi / len(scenarios)
    classifier_angle_step = scenario_angle_step / len(classifiers)
   
    for s_idx, scenario in enumerate(scenarios):
        base_angle = s_idx * scenario_angle_step
        for c_idx, classifier in enumerate(classifiers):
            angle = base_angle + c_idx * classifier_angle_step
            angles.append(angle)
   
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
   
    return x_pos, y_pos


def calculate_node_colors(scenarios, classifiers, scenario_colors):
    """Calculate node colors based on scenario"""
    node_colors = []
    for s_idx, scenario in enumerate(scenarios):
        base_color = scenario_colors[s_idx]
        for c_idx in range(len(classifiers)):
            alpha = 0.6 + (c_idx * 0.1)
            alpha = min(1.0, alpha)
           
            rgb = mcolors.hex2color(base_color)
            rgba_color = f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{alpha})'
            node_colors.append(rgba_color)
   
    return node_colors


def add_legend_traces(fig, scenario_colors, scenarios):
    """Add legend traces for connections and scenarios"""
   
    # Connection legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#27AE60', width=5),
        name='🟢 High Similarity (>60%)',
        showlegend=True
    ))
   
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#F39C12', width=4),
        name='🟠 Medium Similarity (30-60%)',
        showlegend=True
    ))
   
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#E74C3C', width=3),
        name='🔴 Low Similarity (15-30%)',
        showlegend=True
    ))
   
    # Scenario legend
    scenario_names = [
        'Raw Features', 'Raw Preprocessing', 'Enhanced Preprocessing',
        'Enhanced Robust', 'Enhanced Standard', 'Enhanced MinMax'
    ]
   
    for s_idx, (scenario, display_name) in enumerate(zip(scenarios, scenario_names)):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=scenario_colors[s_idx], size=12, 
                       line=dict(width=1, color='black')),
            name=f'{display_name}',
            showlegend=True
        ))


def add_chord_connections(fig, similarity_matrix, x_pos, y_pos, row, col, connection_threshold=0.15):
    """Add connection lines to subplot"""
    n_nodes = len(x_pos)
   
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            similarity = similarity_matrix[i, j]
            if similarity > connection_threshold:
               
                x0, y0 = x_pos[i], y_pos[i]
                x1, y1 = x_pos[j], y_pos[j]
               
                # Control point for curve
                control_factor = 0.3
                control_x = (x0 + x1) / 2 * control_factor
                control_y = (y0 + y1) / 2 * control_factor
               
                # Line properties
                line_width = max(0.5, similarity * 6)
                opacity = min(0.8, similarity + 0.2)
               
                if similarity > 0.6:
                    line_color = '#27AE60'
                elif similarity > 0.3:
                    line_color = '#F39C12'
                else:
                    line_color = '#E74C3C'
               
                # Create path
                path = f"M {x0},{y0} Q {control_x},{control_y} {x1},{y1}"
               
                fig.add_shape(
                    type="path",
                    path=path,
                    line=dict(color=line_color, width=line_width),
                    opacity=opacity,
                    row=row, col=col
                )


def add_chord_nodes(fig, x_pos, y_pos, node_colors, node_combinations, similarity_matrix, 
                    name_suffix="", show_legend=False, row=1, col=1):
    """Add nodes to subplot"""
   
    # Create hover texts
    hover_texts = []
    for i, (scenario, clf) in enumerate(node_combinations):
        similarities = [similarity_matrix[i, j] for j in range(len(node_combinations)) if j != i]
        avg_similarity = np.mean(similarities) if similarities else 0
        strong_connections = sum(1 for s in similarities if s > 0.6)
        total_connections = sum(1 for s in similarities if s > 0.15)
       
        hover_text = f"<b>{scenario.replace('_', ' ')}</b><br>" + \
                    f"<b>{clf.replace('_', ' ')}</b><br><br>" + \
                    f"📊 Avg Similarity: {avg_similarity:.3f}<br>" + \
                    f"📈 Max Similarity: {max(similarities) if similarities else 0:.3f}<br>" + \
                    f"🟢 Strong Connections: {strong_connections}<br>" + \
                    f"🔗 Total Connections: {total_connections}"
        hover_texts.append(hover_text)
   
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers',
        marker=dict(
            size=18,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        name=f'Nodes{name_suffix}',
        showlegend=show_legend
    ), row=row, col=col)


def add_scenario_separators_only(fig, scenarios, classifiers, row_col_pairs):
    """Add scenario separators WITHOUT labels to subplots"""
    scenario_angle_step = 2 * np.pi / len(scenarios)
   
    for row, col in row_col_pairs:
        # Add separators ONLY (no labels)
        for s_idx in range(len(scenarios)):
            angle = s_idx * scenario_angle_step
            x_sep = np.cos(angle) * 1.15
            y_sep = np.sin(angle) * 1.15
           
            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=x_sep, y1=y_sep,
                line=dict(color="gray", width=1, dash="dash"),
                opacity=0.3,
                row=row, col=col
            )