"""
Feature preprocessing and enhancement classes
Contains feature enhancement, outlier removal, and pipeline creation functions.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


class FeatureEnhancer(BaseEstimator, TransformerMixin):
    """Feature enhancement transformer - LEAKAGE FREE"""
   
    def __init__(self, n_poly=10, n_interact=6, n_log=10):
        self.n_poly = n_poly
        self.n_interact = n_interact 
        self.n_log = n_log
        self.log_indices_ = None
        self.original_feature_names_ = None
       
    def fit(self, X, y=None):
        """Fit the enhancer - find which features can be log transformed"""
        # Find features that are always positive (for log transform)
        self.log_indices_ = []
        for i in range(min(self.n_log, X.shape[1])):
            if np.all(X[:, i] > 0):
                self.log_indices_.append(i)
       
        return self
   
    def get_feature_names(self, input_feature_names):
        """Get feature names after enhancement"""
        enhanced_names = input_feature_names.copy()
       
        # Add polynomial feature names
        n_poly = min(self.n_poly, len(input_feature_names))
        for i in range(n_poly):
            enhanced_names.append(f"{input_feature_names[i]}_Poly2")
       
        # Add interaction feature names
        n_interact = min(self.n_interact, len(input_feature_names))
        for i in range(n_interact):
            for j in range(i+1, min(i+4, n_interact)):
                if j < len(input_feature_names):
                    enhanced_names.append(f"{input_feature_names[i]}×{input_feature_names[j]}")
       
        # Add log feature names
        for i in self.log_indices_:
            if i < len(input_feature_names):
                enhanced_names.append(f"Log_{input_feature_names[i]}")
       
        # Add statistical feature names
        enhanced_names.extend(['Features_Std', 'Features_Mean'])
       
        return enhanced_names
   
    def transform(self, X):
        """Transform features - NO DATA LEAKAGE"""
        enhanced_features = [X]  # Start with original
       
        # 1. Polynomial features (squared) - same number for both
        n_poly = min(self.n_poly, X.shape[1])
        if n_poly > 0:
            poly_features = X[:, :n_poly] ** 2
            enhanced_features.append(poly_features)
       
        # 2. Interaction features - same number for both
        n_interact = min(self.n_interact, X.shape[1])
        interactions = []
        for i in range(n_interact):
            for j in range(i+1, min(i+4, n_interact)):
                if j < X.shape[1]:
                    interaction = X[:, i] * X[:, j]
                    interactions.append(interaction)
       
        if interactions:
            enhanced_features.append(np.column_stack(interactions))
       
        # 3. Log transformations - only for features found in fit
        if self.log_indices_:
            log_features = []
            for i in self.log_indices_:
                if i < X.shape[1]:
                    log_features.append(np.log(X[:, i] + 1e-8))
           
            if log_features:
                enhanced_features.append(np.column_stack(log_features))
       
        # 4. Statistical features
        stat_features = []
        stat_features.append(np.std(X, axis=1, keepdims=True))
        stat_features.append(np.mean(X, axis=1, keepdims=True))
        enhanced_features.append(np.column_stack(stat_features))
       
        # Combine all
        final_features = np.column_stack(enhanced_features)
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
       
        return final_features


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Outlier removal transformer - LEAKAGE FREE"""
   
    def __init__(self, iqr_factor=2.5):
        self.iqr_factor = iqr_factor
        self.bounds_ = None
       
    def fit(self, X, y=None):
        """Fit outlier bounds on training data only"""
        self.bounds_ = []
       
        for i in range(X.shape[1]):
            feature_col = X[:, i]
            if np.var(feature_col) > 1e-10:
                Q1 = np.percentile(feature_col, 25)
                Q3 = np.percentile(feature_col, 75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower = Q1 - self.iqr_factor * IQR
                    upper = Q3 + self.iqr_factor * IQR
                    self.bounds_.append((lower, upper))
                else:
                    self.bounds_.append((None, None))
            else:
                self.bounds_.append((None, None))
       
        return self
    
    def transform(self, X):
        """Apply outlier removal using training bounds"""
        X_clipped = X.copy()
       
        for i, (lower, upper) in enumerate(self.bounds_):
            if lower is not None and upper is not None:
                X_clipped[:, i] = np.clip(X[:, i], lower, upper)
       
        return X_clipped


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Zero variance filter - LEAKAGE FREE"""
   
    def __init__(self, threshold=1e-10):
        self.threshold = threshold
        self.valid_features_ = None
       
    def fit(self, X, y=None):
        """Find valid features on training data"""
        feature_vars = np.var(X, axis=0)
        self.valid_features_ = feature_vars > self.threshold
       
        if np.sum(self.valid_features_) == 0:
            # Keep at least one feature
            self.valid_features_[0] = True
           
        return self
   
    def transform(self, X):
        """Apply feature filtering"""
        return X[:, self.valid_features_]


def create_leakage_free_pipeline(scenario_config):
    """Create data leakage free pipeline"""
    steps = []
   
    # Step 1: Feature Enhancement (if requested)
    if scenario_config.get('enhanced', False):
        steps.append(('enhancer', FeatureEnhancer()))
   
    # Step 2: Outlier removal (if requested)
    if scenario_config.get('outlier_removal', False):
        steps.append(('outlier_remover', OutlierRemover()))
   
    # Step 3: Variance filtering (always)
    steps.append(('variance_filter', VarianceFilter()))
   
    # Step 4: Scaling (if method specified)
    method = scenario_config.get('method')
    if method == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif method == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif method == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
   
    # Step 5: Feature selection (if requested)
    k_best = scenario_config.get('k_best')
    if k_best:
        steps.append(('feature_selector', SelectKBest(score_func=f_classif, k=k_best)))
   
    return Pipeline(steps)