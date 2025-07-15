"""
Statistical analysis functions for comparing feature types
Contains statistical comparison and analysis utilities.
"""

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


class StatisticalComparison:
    """Enhanced statistical comparison class"""
   
    @staticmethod
    def paired_comparison(single_roi_scores, motor_circuit_scores, metric_name="AUC"):
        """Perform comprehensive paired statistical comparison"""
        results = {}
       
        # Ensure same length
        min_len = min(len(single_roi_scores), len(motor_circuit_scores))
        single_roi_scores = single_roi_scores[:min_len]
        motor_circuit_scores = motor_circuit_scores[:min_len]
       
        if min_len < 3:
            return {
                'metric': metric_name,
                'n_pairs': min_len,
                'single_roi_mean': np.mean(single_roi_scores) if single_roi_scores else 0,
                'motor_circuit_mean': np.mean(motor_circuit_scores) if motor_circuit_scores else 0,
                'difference': 0,
                'effect_size': 0,
                'paired_ttest_p': 1.0,
                'wilcoxon_p': 1.0,
                'significance': 'ns',
                'interpretation': 'Insufficient data'
            }
       
        # Basic statistics
        single_roi_mean = np.mean(single_roi_scores)
        motor_circuit_mean = np.mean(motor_circuit_scores)
        difference = motor_circuit_mean - single_roi_mean
       
        # Effect size (Cohen's d for paired samples)
        diff_scores = np.array(motor_circuit_scores) - np.array(single_roi_scores)
        effect_size = np.mean(diff_scores) / (np.std(diff_scores) + 1e-8)
       
        # Statistical tests
        try:
            # Paired t-test
            t_stat, paired_t_p = ttest_rel(motor_circuit_scores, single_roi_scores)
           
            # Wilcoxon signed-rank test (non-parametric)
            w_stat, wilcoxon_p = wilcoxon(motor_circuit_scores, single_roi_scores, alternative='two-sided')
           
        except Exception as e:
            paired_t_p = 1.0
            wilcoxon_p = 1.0
       
        # Determine significance level
        min_p = min(paired_t_p, wilcoxon_p)
        if min_p < 0.001:
            significance = '***'
        elif min_p < 0.01:
            significance = '**'
        elif min_p < 0.05:
            significance = '*'
        else:
            significance = 'ns'
       
        # Interpretation
        if significance != 'ns':
            if difference > 0:
                interpretation = f"Motor-Circuit-Features significantly better (p={min_p:.4f})"
            else:
                interpretation = f"Single-ROI-Features significantly better (p={min_p:.4f})"
        else:
            interpretation = f"No significant difference (p={min_p:.4f})"
       
        return {
            'metric': metric_name,
            'n_pairs': min_len,
            'single_roi_mean': single_roi_mean,
            'single_roi_std': np.std(single_roi_scores),
            'motor_circuit_mean': motor_circuit_mean,
            'motor_circuit_std': np.std(motor_circuit_scores),
            'difference': difference,
            'effect_size': effect_size,
            'paired_ttest_p': paired_t_p,
            'wilcoxon_p': wilcoxon_p,
            'min_p_value': min_p,
            'significance': significance,
            'interpretation': interpretation
        }