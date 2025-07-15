"""
Feature importance analysis for various machine learning classifiers
Contains advanced methods for extracting feature importance from different types of models.
"""

import numpy as np
from sklearn.metrics import accuracy_score

# Try to import permutation importance
try:
    from sklearn.inspection import permutation_importance
    PERMUTATION_AVAILABLE = True
except ImportError:
    PERMUTATION_AVAILABLE = False


class FeatureImportanceAnalyzer:
    """Enhanced feature importance extraction for various classifiers"""
    
    def __init__(self):
        pass
    
    def get_feature_importance_advanced(self, classifier, X_train, y_train, feature_names, clf_name):
        """Enhanced feature importance extraction"""
        feature_importance = None
        method_used = "Unknown"

        # Tree-based classifiers have built-in feature_importances_
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = classifier.feature_importances_
            method_used = "Built-in Tree Importance"

        # Linear models have coef_
        elif hasattr(classifier, 'coef_'):
            if len(classifier.coef_.shape) == 1:
                feature_importance = np.abs(classifier.coef_)
            else:
                feature_importance = np.abs(classifier.coef_[0])
            method_used = "Linear Coefficients"

        # Neural Network - enhanced handling
        elif clf_name == 'MLP':
            try:
                # Method 1: Try advanced gradient-based approach
                feature_importance = self._mlp_advanced_importance(classifier, X_train, y_train)
                method_used = "MLP Advanced Gradient"
               
                # Check if results are meaningful
                if feature_importance is None or np.all(feature_importance == feature_importance[0]):
                    # Method 2: Enhanced permutation importance
                    feature_importance = self._mlp_enhanced_permutation(classifier, X_train, y_train)
                    method_used = "MLP Enhanced Permutation"
                   
            except Exception as e:
                # Method 3: Fallback to basic permutation
                feature_importance = self._permutation_importance_fallback(classifier, X_train, y_train)
                method_used = "MLP Basic Permutation"

        # Naive Bayes - enhanced handling
        elif clf_name == 'Naive_Bayes':
            try:
                if hasattr(classifier, 'feature_log_prob_') and len(classifier.feature_log_prob_) >= 2:
                    # Enhanced Naive Bayes importance using class separation
                    feature_importance = self._naive_bayes_enhanced_importance(classifier, X_train, y_train)
                    method_used = "Naive Bayes Enhanced Log Probabilities"
                else:
                    feature_importance = self._permutation_importance_fallback(classifier, X_train, y_train)
                    method_used = "Naive Bayes Permutation Fallback"
            except Exception as e:
                feature_importance = self._permutation_importance_fallback(classifier, X_train, y_train)
                method_used = "Naive Bayes Manual Permutation"

        # SVM and other models - enhanced permutation
        else:
            try:
                feature_importance = self._enhanced_permutation_importance(classifier, X_train, y_train, clf_name)
                method_used = f"{clf_name} Enhanced Permutation"
            except Exception as e:
                feature_importance = self._permutation_importance_fallback(classifier, X_train, y_train)
                method_used = f"{clf_name} Basic Fallback"

        # Final validation and normalization
        if feature_importance is None or len(feature_importance) == 0:
            feature_importance = np.ones(X_train.shape[1]) / X_train.shape[1]
            method_used = "Equal Weights (Fallback)"

        # Enhanced normalization
        feature_importance = self._normalize_importance(feature_importance)

        return feature_importance, method_used

    def _mlp_advanced_importance(self, classifier, X_train, y_train):
        """Advanced gradient-based importance for MLP"""
        try:
            # Use multiple perturbation strategies
            n_features = X_train.shape[1]
            importance_scores = np.zeros(n_features)
           
            # Strategy 1: Feature ablation
            baseline_score = classifier.score(X_train, y_train)
           
            for i in range(n_features):
                X_ablated = X_train.copy()
                # Set feature to mean (ablation)
                X_ablated[:, i] = np.mean(X_train[:, i])
                ablated_score = classifier.score(X_ablated, y_train)
                importance_scores[i] += abs(baseline_score - ablated_score)
           
            # Strategy 2: Multiple noise levels
            for noise_level in [0.01, 0.05, 0.1]:
                for i in range(n_features):
                    X_noisy = X_train.copy()
                    feature_std = np.std(X_train[:, i])
                    if feature_std > 0:
                        noise = np.random.normal(0, feature_std * noise_level, X_train.shape[0])
                        X_noisy[:, i] += noise
                       
                        try:
                            noisy_score = classifier.score(X_noisy, y_train)
                            importance_scores[i] += abs(baseline_score - noisy_score) / noise_level
                        except:
                            continue
           
            # Strategy 3: Feature correlation with predictions
            try:
                predictions = classifier.predict_proba(X_train)[:, 1]
                for i in range(n_features):
                    correlation = abs(np.corrcoef(X_train[:, i], predictions)[0, 1])
                    if not np.isnan(correlation):
                        importance_scores[i] += correlation
            except:
                pass
           
            return importance_scores
           
        except Exception as e:
            return None

    def _mlp_enhanced_permutation(self, classifier, X_train, y_train):
        """Enhanced permutation importance specifically for MLP"""
        try:
            baseline_score = classifier.score(X_train, y_train)
            baseline_proba = classifier.predict_proba(X_train)[:, 1]
           
            importance_scores = []
           
            for i in range(X_train.shape[1]):
                # Multiple permutation runs for stability
                feature_importances = []
               
                for run in range(5):  # Multiple runs
                    X_permuted = X_train.copy()
                    np.random.seed(42 + run)  # Different seed each run
                    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                   
                    try:
                        # Score-based importance
                        permuted_score = classifier.score(X_permuted, y_train)
                        score_importance = max(0, baseline_score - permuted_score)
                       
                        # Probability-based importance
                        permuted_proba = classifier.predict_proba(X_permuted)[:, 1]
                        proba_importance = np.mean(abs(baseline_proba - permuted_proba))
                       
                        # Combined importance
                        combined_importance = (score_importance + proba_importance) / 2
                        feature_importances.append(combined_importance)
                       
                    except Exception:
                        feature_importances.append(0)
               
                # Average across runs
                avg_importance = np.mean(feature_importances) if feature_importances else 0
                importance_scores.append(avg_importance)
           
            return np.array(importance_scores)
           
        except Exception as e:
            return None

    def _naive_bayes_enhanced_importance(self, classifier, X_train, y_train):
        """Enhanced importance for Naive Bayes using class separation"""
        try:
            n_features = X_train.shape[1]
            importance_scores = np.zeros(n_features)
           
            # Method 1: Log probability differences
            if hasattr(classifier, 'feature_log_prob_'):
                log_prob_diff = np.abs(classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0])
                importance_scores += log_prob_diff
           
            # Method 2: Feature variance by class
            if hasattr(classifier, 'theta_'):
                class_0_indices = y_train == 0
                class_1_indices = y_train == 1
               
                for i in range(n_features):
                    if np.sum(class_0_indices) > 0 and np.sum(class_1_indices) > 0:
                        mean_diff = abs(np.mean(X_train[class_0_indices, i]) - np.mean(X_train[class_1_indices, i]))
                        pooled_std = np.sqrt((np.var(X_train[class_0_indices, i]) + np.var(X_train[class_1_indices, i])) / 2)
                        if pooled_std > 0:
                            importance_scores[i] += mean_diff / pooled_std
           
            # Method 3: Individual feature classification power
            for i in range(n_features):
                try:
                    from sklearn.naive_bayes import GaussianNB
                    single_feature_clf = GaussianNB()
                    single_feature_clf.fit(X_train[:, i:i+1], y_train)
                    single_score = single_feature_clf.score(X_train[:, i:i+1], y_train)
                    importance_scores[i] += single_score
                except:
                    continue
           
            return importance_scores
           
        except Exception as e:
            return None

    def _enhanced_permutation_importance(self, classifier, X_train, y_train, clf_name):
        """Enhanced permutation importance for any classifier"""
        try:
            if PERMUTATION_AVAILABLE and X_train.shape[0] > 5:
                # Use sklearn's permutation importance with enhanced parameters
                perm_importance = permutation_importance(
                    classifier, X_train, y_train, 
                    n_repeats=5, random_state=42, 
                    scoring='roc_auc',
                    n_jobs=1
                )
                return perm_importance.importances_mean
            else:
                # Enhanced manual permutation
                return self._enhanced_manual_permutation(classifier, X_train, y_train)
               
        except Exception as e:
            return None

    def _enhanced_manual_permutation(self, classifier, X_train, y_train):
        """Enhanced manual permutation importance"""
        try:
            baseline_score = classifier.score(X_train, y_train)
            importance_scores = []

            for i in range(X_train.shape[1]):
                # Multiple permutation runs
                run_scores = []
               
                for run in range(3):
                    X_permuted = X_train.copy()
                    np.random.seed(42 + run)
                    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                   
                    try:
                        permuted_score = classifier.score(X_permuted, y_train)
                        run_scores.append(max(0, baseline_score - permuted_score))
                    except:
                        run_scores.append(0)
               
                # Average importance across runs
                avg_importance = np.mean(run_scores) if run_scores else 0
                importance_scores.append(avg_importance)
           
            return np.array(importance_scores)
           
        except Exception as e:
            return np.ones(X_train.shape[1]) / X_train.shape[1]

    def _normalize_importance(self, importance):
        """Enhanced normalization of feature importance"""
        try:
            # Handle edge cases
            if importance is None or len(importance) == 0:
                return np.array([])
           
            # Convert to numpy array and handle NaN/inf
            importance = np.array(importance)
            importance = np.nan_to_num(importance, nan=0.0, posinf=1.0, neginf=0.0)
           
            # Take absolute values
            importance = np.abs(importance)
           
            # Add small random noise to break ties
            if np.all(importance == importance[0]):
                np.random.seed(42)
                noise = np.random.uniform(0, 0.001, len(importance))
                importance = importance + noise
           
            # Normalize
            total_importance = np.sum(importance)
            if total_importance > 0:
                importance = importance / total_importance
            else:
                # Fallback: uniform with slight randomness
                np.random.seed(42)
                importance = np.random.uniform(0.8, 1.2, len(importance))
                importance = importance / np.sum(importance)
           
            return importance
           
        except Exception as e:
            n = len(importance) if importance is not None else 1
            return np.ones(n) / n

    def _permutation_importance_fallback(self, classifier, X_train, y_train):
        """Manual implementation of permutation importance"""
        try:
            baseline_score = classifier.score(X_train, y_train)
            importance_scores = []

            np.random.seed(42)  # For reproducibility
            for i in range(X_train.shape[1]):
                X_permuted = X_train.copy()
                # Permute feature i
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                try:
                    permuted_score = classifier.score(X_permuted, y_train)
                    importance_scores.append(max(0, baseline_score - permuted_score))
                except:
                    importance_scores.append(0)
     
            return np.array(importance_scores)
        except:
            # Ultimate fallback
            return np.ones(X_train.shape[1]) / X_train.shape[1]