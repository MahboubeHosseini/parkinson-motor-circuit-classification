"""
Cross-validation strategies for Parkinson's Disease Subtype Classification
Implements center-based cross-validation to prevent data leakage.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


class CenterBasedSplitter:
    """Center-based cross-validation splitter ensuring no center overlap"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        
    def analyze_center_distribution(self, data_paths):
        """Analyze center distribution across classes"""
        center_stats = defaultdict(lambda: {'td': 0, 'pigd': 0, 'total': 0})
        
        for patient in data_paths:
            center = patient['center_name']
            if patient['label'] == 0:  # TD
                center_stats[center]['td'] += 1
            else:  # PIGD
                center_stats[center]['pigd'] += 1
            center_stats[center]['total'] += 1
        
        return center_stats
    
    def create_balanced_center_splits(self, data_paths):
        """Create balanced center-based splits"""
        center_stats = self.analyze_center_distribution(data_paths)
        
        # Group centers by their class balance
        balanced_centers = []  # Centers with both TD and PIGD
        td_only_centers = []   # Centers with only TD
        pigd_only_centers = [] # Centers with only PIGD
        
        for center, stats in center_stats.items():
            if stats['td'] > 0 and stats['pigd'] > 0:
                balanced_centers.append((center, stats))
            elif stats['td'] > 0:
                td_only_centers.append((center, stats))
            else:
                pigd_only_centers.append((center, stats))
        
        if len(balanced_centers) < self.n_splits:
            self.n_splits = max(2, len(balanced_centers))
        
        # Create splits ensuring balance
        np.random.seed(self.random_state)
        splits = []
        
        balanced_centers.sort(key=lambda x: x[1]['total'], reverse=True)
        
        fold_size = len(balanced_centers) // self.n_splits
        remaining = len(balanced_centers) % self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            if i < remaining:
                end_idx += 1
            
            test_centers = [center for center, _ in balanced_centers[start_idx:end_idx]]
            
            # Add single-class centers to maintain balance
            if i < len(td_only_centers):
                test_centers.append(td_only_centers[i][0])
            if i < len(pigd_only_centers):
                test_centers.append(pigd_only_centers[i][0])
            
            train_centers = [center for center, _ in balanced_centers if center not in test_centers]
            
            # Add remaining single-class centers to training
            remaining_td = [center for center, _ in td_only_centers if center not in test_centers]
            remaining_pigd = [center for center, _ in pigd_only_centers if center not in test_centers]
            train_centers.extend(remaining_td)
            train_centers.extend(remaining_pigd)
            
            splits.append((train_centers, test_centers))
        
        return splits
    
    def split(self, X, y, data_paths):
        """Generate center-based train/test splits"""
        center_splits = self.create_balanced_center_splits(data_paths)
        
        for fold_idx, (train_centers, test_centers) in enumerate(center_splits):
            train_indices = []
            test_indices = []
            
            for idx, patient in enumerate(data_paths):
                if patient['center_name'] in train_centers:
                    train_indices.append(idx)
                elif patient['center_name'] in test_centers:
                    test_indices.append(idx)
            
            # Check split quality
            train_labels = [data_paths[i]['label'] for i in train_indices]
            test_labels = [data_paths[i]['label'] for i in test_indices]
            
            train_td = sum(1 for l in train_labels if l == 0)
            train_pigd = sum(1 for l in train_labels if l == 1)
            test_td = sum(1 for l in test_labels if l == 0)
            test_pigd = sum(1 for l in test_labels if l == 1)
            
            # Ensure both classes exist in both sets
            if train_td > 0 and train_pigd > 0 and test_td > 0 and test_pigd > 0:
                yield np.array(train_indices), np.array(test_indices)